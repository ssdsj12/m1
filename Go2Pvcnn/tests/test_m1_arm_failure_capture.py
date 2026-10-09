"""Raw arm rejection survives synchronization, wrapper holds, and strict JSONL."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.dual_arm_mpc import (
    DualArmMpcCoordinator, DualArmMpcInput,
)
from go2_pvcnn.control.m1_bimanual_coordination.whole_body_qp import BimanualWholeBodyQp
from go2_pvcnn.control.m1_panda_coordination import arm_mpc
from go2_pvcnn.control.m1_panda_coordination.qp_backend import DenseQpResult
from tests.test_m1_bimanual_dual_arm_mpc import _arm_input, _object_solution
from tests.test_m1_bimanual_whole_body_qp import _request
from tests.test_m1_wire_contact_diagnostics import _acceptance_wrapper
from tests.test_m1_wire_contact_probe_static import _recorder
from scripts.m1_dual_panda_o6_bimanual_play import _write_wire_contact_report_atomic


def _rejected(monkeypatch, failures):
    coordinator = DualArmMpcCoordinator()
    sample = DualArmMpcInput(_arm_input(), _arm_input(), _object_solution())
    safe = coordinator.plan(sample)
    real_solve = arm_mpc.solve_reference_qp
    calls = []

    def solve(problem, **kwargs):
        side = ("left", "right")[len(calls)]
        result = real_solve(problem, **kwargs)
        if side in failures:
            # Distinct upper/lower acceleration rows, with known +1 violations.
            solution = result.solution.clone()
            solution[1] = 3.0 if side == "left" else -3.0
            row = 841 if side == "left" else 981
            result = replace(result, success=False, solution=solution,
                             iterations=17 if side == "left" else 23,
                             max_inequality_violation=1.0, active_set=(row,))
        calls.append((problem, result))
        return result

    monkeypatch.setattr(arm_mpc, "solve_reference_qp", solve)
    rejected = coordinator.plan(sample)
    return coordinator, sample, safe, rejected, calls


@pytest.mark.parametrize("failures", [("left",), ("right",), ("left", "right")])
def test_raw_sided_failure_survives_hold_wbc_wrapper_and_jsonl(monkeypatch, tmp_path, failures):
    coordinator, sample, safe, rejected, calls = _rejected(monkeypatch, failures)
    capture = getattr(rejected, "failure_capture", None)
    assert capture is not None, "synchronized hold discarded the raw arm rejection"
    assert capture["failed_sides"] == list(failures)
    assert not rejected.both_feasible and rejected.synchronized_fallback
    for index, side in enumerate(("left", "right")):
        assert not getattr(rejected, side).diagnostics.feasible
        assert torch.equal(getattr(rejected, side).q_ref, getattr(safe, side).q_ref)
        raw = capture[side]
        assert raw["diagnostics"]["feasible"] is (side not in failures)
        assert raw["diagnostics"]["fallback_reason"] == ("qp_infeasible" if side in failures else None)
        assert raw["qp_result"]["success"] is (side not in failures)
        assert raw["input"]["q"] == getattr(sample, side).q.tolist()
        assert raw["input"]["qdd_max"] == [2.0] * 7
        assert raw["input"]["target_pose_b"][0][0] == pytest.approx(.02 if side == "left" else .04)
        assert raw["qp_problem"]["inequality_upper"] == calls[index][0].inequality_upper.tolist()
        if side in failures:
            row_index = 841 if side == "left" else 981
            row = next(row for row in raw["constraint_rows"] if row["row"] == row_index)
            assert row["family"] == "acceleration"
            assert row["bound"] == ("upper" if side == "left" else "lower")
            assert (row["node"], row["joint"]) == (0, 1)
            assert row["upper"] == 2.0
            assert row["lhs"] == 3.0 and row["residual"] == 1.0
            assert row["active"]
            assert raw["qp_result"]["iterations"] == (17 if side == "left" else 23)
    # WBC still rejects the aggregate source and retains its accepted effort.
    wbc = BimanualWholeBodyQp()
    previous = wbc.solve(_request())
    result = wbc.solve(_request(arm_solution=rejected))
    assert not result.feasible
    assert result.diagnostics.fallback_reason == "dual_arm_mpc_infeasible"
    assert torch.equal(result.effort, previous.effort)

    _, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    lane.runtime.latest_solutions["arm"] = rejected
    lane.runtime.latest_solutions["wbc"] = result
    command = wrapper._apply_wire_acceptance(lane, snapshot, None, lane.baseline_command)
    assert not command.feasible and command.effort.shape == (43,)
    diagnostics = wrapper.wire_acceptance_diagnostics
    persisted = diagnostics.get("first_arm_failure")
    assert persisted is not None, "wrapper must publish the first raw arm rejection"
    assert persisted["arm"]["failed_sides"] == list(failures)
    assert persisted["timestamp_ns"] == snapshot.timestamp_ns
    recorder = _recorder()
    recorder.record(step=1457, diagnostics=diagnostics, acceptance_state=wrapper.wire_acceptance_state)
    # Later attempts and source tensor mutations cannot overwrite the first capture.
    sample.left.q.fill_(0.5)
    lane.runtime.latest_solutions["arm"] = safe
    wrapper._apply_wire_acceptance(lane, snapshot, None, lane.baseline_command)
    assert wrapper.wire_acceptance_diagnostics["first_arm_failure"] == persisted
    recorder.finish("safety_rejected")
    path = tmp_path / "failure.jsonl"
    _write_wire_contact_report_atomic(path, recorder.records)
    rows = [json.loads(line, parse_constant=lambda value: pytest.fail(value))
            for line in path.read_text().splitlines()]
    assert rows[-1]["first_arm_failure"]["arm"]["failed_sides"] == list(failures)
    assert rows[-1]["first_arm_failure"]["arm"]["left"]["input"]["q"] == [0.] * 7


def test_real_infeasible_position_row_has_actual_bound_and_residual():
    sample = _arm_input()
    sample.q[1], sample.qd[1] = -.999, -.16
    coordinator = DualArmMpcCoordinator()
    rejected = coordinator.plan(DualArmMpcInput(sample, _arm_input(), _object_solution()))
    assert not rejected.both_feasible
    capture = getattr(rejected, "failure_capture", None)
    assert capture is not None, "raw failed QP rows must remain inspectable"
    assert capture["failed_sides"] == ["left"]
    raw = capture["left"]
    row = next(row for row in raw["constraint_rows"] if row["row"] == 141)
    assert (row["family"], row["bound"], row["node"], row["joint"]) == ("position", "lower", 0, 1)
    assert row["upper"] == pytest.approx(-.0022)
    matrix = torch.tensor(raw["qp_problem"]["inequality_matrix"], dtype=torch.float64)
    vector = torch.tensor(raw["qp_result"]["solution"], dtype=torch.float64)
    assert row["residual"] == pytest.approx(float(matrix[141] @ vector) - row["upper"])
    # Row 141 requires qdd[0,1] >= 11, beyond the actual configured +2 bound.
    assert row["upper"] / matrix[141, 1] > sample.qdd_max[1]


def test_nonfinite_raw_solver_evidence_is_strict_json_and_not_relabelled(monkeypatch):
    def solve(problem, **kwargs):
        return DenseQpResult(torch.full_like(problem.gradient, torch.nan), False,
                             7, float("nan"), float("inf"), (141,))
    monkeypatch.setattr(arm_mpc, "solve_reference_qp", solve)
    rejected = DualArmMpcCoordinator().plan(DualArmMpcInput(_arm_input(), _arm_input(), _object_solution()))
    capture = getattr(rejected, "failure_capture", None)
    assert capture is not None, "nonfinite raw solver results need lossless JSON evidence"
    encoded = json.dumps(capture, allow_nan=False)
    payload = json.loads(encoded)
    assert payload["left"]["qp_result"]["max_equality_residual"] == "NaN"
    assert payload["left"]["qp_result"]["max_inequality_violation"] == "Infinity"
    assert payload["left"]["qp_result"]["solution"][0] == "NaN"
    assert payload["left"]["diagnostics"]["fallback_reason"] == "qp_infeasible"


def test_wrapper_keeps_four_prior_commands_and_both_safety_projections(monkeypatch):
    from tests.test_m1_bimanual_safety_projection import _input
    from go2_pvcnn.control.m1_bimanual_coordination.safety_projection import SafetyResult
    from go2_pvcnn.control.m1_bimanual_coordination.contracts import BimanualPhase

    _, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    wrapper.mode, wrapper._step = "teacher", 1
    lane.runtime.mission.phase = BimanualPhase.APPROACH
    lane.teacher = SimpleNamespace(fixed_base=True)
    lane.last_teacher_solution = SimpleNamespace(diagnostics=SimpleNamespace(feasible=True))
    lane.safety = SimpleNamespace(project=lambda sample: SafetyResult(
        sample.candidate_effort, True, (), None, 0.0))
    assert hasattr(wrapper, "_record_wire_command"), "missing bounded pre-failure command history"
    for index in range(6):
        snapshot = replace(snapshot, timestamp_ns=index + 1)
        command = wrapper._compute_lane_command(lane, snapshot, _input().dynamics, None)
        command = wrapper._apply_wire_acceptance(lane, snapshot, _input().dynamics, command)
        wrapper._record_wire_command(lane, snapshot, command, snapshot.left_hand.q, snapshot.right_hand.q)
    assert len(lane.wire_command_history) == 4
    assert [row["timestamp_ns"] for row in lane.wire_command_history] == [3, 4, 5, 6]
    _, _, _, rejected, _ = _rejected(monkeypatch, ("left",))
    lane.runtime.latest_solutions["arm"] = rejected
    snapshot = replace(snapshot, timestamp_ns=7)
    projections = []

    def project(sample):
        projections.append(sample)
        return SafetyResult(sample.candidate_effort, len(projections) > 1,
                            (), "candidate_rejected" if len(projections) == 1 else None, 0.)

    lane.safety = SimpleNamespace(project=project)
    command = wrapper._compute_lane_command(lane, snapshot, _input().dynamics, None)
    command = wrapper._apply_wire_acceptance(lane, snapshot, _input().dynamics, command)
    wrapper._record_wire_command(lane, snapshot, command, snapshot.left_hand.q, snapshot.right_hand.q)
    failure = wrapper.wire_acceptance_diagnostics["first_arm_failure"]
    assert [row["timestamp_ns"] for row in failure["preceding_commands"]] == [3, 4, 5, 6]
    cycle = failure["rejected_cycle"]
    assert cycle["candidate_projection"]["result"]["feasible"] is False
    assert cycle["terminal_hold_projection"]["result"]["feasible"] is True
    assert cycle["candidate_projection"]["input"]["active_q"] == projections[0].active_q.tolist()
    assert cycle["terminal_hold_projection"]["input"]["safe_effort"] == projections[1].safe_effort.tolist()
    assert cycle["command"]["effort"] == command.effort.tolist()
    assert cycle["left_hand_position_target"] == snapshot.left_hand.q.tolist()
    assert cycle["measured_applied_effort"] is None  # unavailable in this adapter fixture
    assert cycle["hand_actuation"] == "external_position_servo"
    assert not command.feasible
    json.dumps(wrapper.wire_acceptance_diagnostics, allow_nan=False)
    wrapper._reset_wire_acceptance(lane, snapshot)
    assert lane.wire_first_arm_failure is None and not lane.wire_command_history


def test_step_captures_submitted_hold_hand_targets_and_measured_effort(monkeypatch):
    _, _, _, rejected, _ = _rejected(monkeypatch, ("right",))
    module, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    lane.runtime.latest_solutions["arm"] = rejected
    lane.wire_last_safe_hand_targets = (
        torch.full((6,), .11, dtype=torch.float64),
        torch.full((6,), .22, dtype=torch.float64),
    )
    applied = torch.arange(53, dtype=torch.float32)[None]
    submitted, targets = [], []
    robot = SimpleNamespace(
        data=SimpleNamespace(applied_torque=applied),
        set_joint_position_target=lambda target, **kwargs: targets.append(target.clone()),
    )
    lane.adapter = SimpleNamespace(
        snapshot=lambda: snapshot, dynamics=lambda: None,
        teacher_task_jacobian=lambda: None, robot=robot,
        active_joint_ids=tuple(range(10, 53)),
        left_hand_ids=tuple(range(6)), right_hand_ids=tuple(range(6, 12)),
    )
    lane.runtime.compute = lambda snap: lane.baseline_command
    wrapper._compute_lane_command = lambda *args: lane.baseline_command
    monkeypatch.setattr(module, "build_teacher_input", lambda **kwargs: None)
    wrapper.env.unwrapped.device = "cpu"
    wrapper.env.step = lambda action: submitted.append(action.clone())
    wrapper.startup_complete, wrapper._step = True, 0
    wrapper._effort_limits = torch.ones(43, dtype=torch.float64)
    wrapper._step_impl()
    cycle = wrapper.wire_acceptance_diagnostics["first_arm_failure"]["rejected_cycle"]
    assert cycle["left_hand_position_target"] == [.11] * 6
    assert cycle["right_hand_position_target"] == [.22] * 6
    assert cycle["measured_applied_effort"] == list(range(10, 53))
    assert cycle["measured_effort_timing"] == "snapshot_before_new_command"
    assert torch.equal(submitted[0][0], torch.tensor(cycle["command"]["effort"]))
    assert torch.equal(targets[0][0], torch.tensor(cycle["left_hand_position_target"]))
    assert torch.equal(targets[1][0], torch.tensor(cycle["right_hand_position_target"]))
    # Later simulator buffer reuse does not rewrite the saved measurement.
    applied.fill_(99.)
    assert cycle["measured_applied_effort"] == list(range(10, 53))


def test_object_rejection_does_not_misattribute_stale_arm_solver_evidence():
    coordinator = DualArmMpcCoordinator()
    coordinator.plan(DualArmMpcInput(_arm_input(), _arm_input(), _object_solution()))
    result = coordinator.plan(DualArmMpcInput(_arm_input(), _arm_input(), _object_solution(False)))
    assert result.failure_capture is None
    assert result.left.diagnostics.fallback_reason == "object_mpc_infeasible"
