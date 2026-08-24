"""Agent configuration for Go2 training."""

from .m1_train_cfg import get_m1_train_cfg
from .m1_panda_coordinated_train_cfg import get_m1_panda_coordinated_train_cfg
from .m1_panda_teacher_train_cfg import get_m1_panda_teacher_train_cfg
from .train_cfg import get_train_cfg

__all__ = [
    "get_train_cfg",
    "get_m1_train_cfg",
    "get_m1_panda_coordinated_train_cfg",
    "get_m1_panda_teacher_train_cfg",
]
