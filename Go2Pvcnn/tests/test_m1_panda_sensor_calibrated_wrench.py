import torch

from go2_pvcnn.control.m1_panda_coordination.sensor_calibrated_wrench import (
    sensor_calibrated_wrench,
)


def test_sensor_calibration_fuses_model_and_measurement():
    model = torch.zeros((2, 6), dtype=torch.float64)
    measured = torch.ones((2, 6), dtype=torch.float64)
    result = sensor_calibrated_wrench(model, measured, observation_gain=0.75)
    torch.testing.assert_close(result, torch.full_like(result, 0.75))


def test_sensor_calibration_rejects_invalid_gain():
    wrench = torch.zeros(6, dtype=torch.float64)
    try:
        sensor_calibrated_wrench(wrench, wrench, observation_gain=1.1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected gain validation")
