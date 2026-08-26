import pytest

from rsl_rl.modules import ActorCritic
from rsl_rl.runners.on_policy_runner import resolve_policy_class


def test_legacy_bare_policy_name_remains_supported():
    assert resolve_policy_class("ActorCritic") is ActorCritic


def test_dotted_policy_name_resolves_without_application_import_in_rsl():
    resolved = resolve_policy_class(
        "go2_pvcnn.control.m1_panda_coordination.residual_actor_critic.ResidualActorCritic"
    )
    assert resolved.__name__ == "ResidualActorCritic"


@pytest.mark.parametrize("name", ["", "missing", "bad.module.Class"])
def test_invalid_policy_name_fails_explicitly(name):
    with pytest.raises((ValueError, ImportError, AttributeError), match="policy class"):
        resolve_policy_class(name)
