"""Own two SHA-bound frozen workers before a simulation can be launched."""

from __future__ import annotations

from pathlib import Path
import weakref

from .runtime import FrozenO6FingertipPrior


def _close_priors(priors: tuple[FrozenO6FingertipPrior, ...]) -> None:
    for prior in priors:
        try:
            prior.close()
        except BaseException:
            # Each frozen adapter retains bounded asynchronous-reaper ownership.
            pass


class PreparedO6FingertipPriors:
    """Pre-launch owner; callers cannot inject a model or an unvalidated pair."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use PreparedO6FingertipPriors.from_artifact")

    @classmethod
    def from_artifact(
        cls, path: str | Path, *, expected_metadata_sha256: str
    ) -> PreparedO6FingertipPriors:
        created: list[FrozenO6FingertipPrior] = []
        try:
            for _side in ("left", "right"):
                # Both strict byte snapshots and worker startup must finish
                # before AppLauncher/scene startup. Each snapshot is bound to
                # the same external metadata identity (including weight/report
                # SHAs); replacing the path between loads rejects pre-launch.
                created.append(FrozenO6FingertipPrior.from_artifact(
                    path, expected_metadata_sha256=expected_metadata_sha256
                ))
        except BaseException:
            _close_priors(tuple(created))
            raise
        self = object.__new__(cls)
        self._metadata_pin = expected_metadata_sha256
        self._priors = tuple(created)
        self._closed = False
        self._finalizer = weakref.finalize(self, _close_priors, self._priors)
        return self

    def for_metadata_pin(
        self, expected_metadata_sha256: str
    ) -> tuple[FrozenO6FingertipPrior, FrozenO6FingertipPrior]:
        if self._closed:
            raise ValueError("prepared fingertip priors are closed")
        if expected_metadata_sha256 != self._metadata_pin:
            raise ValueError("prepared fingertip priors metadata SHA-256 pin mismatch")
        return self._priors

    def close(self) -> None:
        self._closed = True
        self._finalizer()

    def __enter__(self) -> PreparedO6FingertipPriors:
        return self

    def __exit__(self, *unused) -> bool:
        self.close()
        return False
