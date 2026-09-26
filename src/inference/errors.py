"""Exceptions of the inference package and their remediation hints.

Every :class:`ModelNotReady` carries a ``hint`` chosen by its type (and by the raise site when a
more specific fix applies), so user interfaces show the right remedy instead of guessing one:
a checkpoint that does not fit the current graph or data needs rebuilt datasets *and* a retrain,
an unfinished run needs ``--resume``, a missing model needs training.
"""

from __future__ import annotations

__all__ = [
    "CheckpointMismatch", "FINALIZE_HINT", "GRAPH_HINT", "GraphNotReady", "ModelNotFinalized", "ModelNotReady",
    "PredictionError", "RETRAIN_HINT", "TRAIN_HINT", "remediation",
]

TRAIN_HINT = "train one with: python src/training/train.py (or use the physics baseline: --physics)"
RETRAIN_HINT = ("rebuild the datasets and retrain: python src/data_pipeline/04_dataset_builder.py --force && "
                "python src/training/train.py")
FINALIZE_HINT = ("finish or resume training (python src/training/train.py --resume) so it publishes a finalized "
                 "best.pt, or restore the previous one")
GRAPH_HINT = ("build it with: python src/data_pipeline/01_extract_network.py && "
              "python src/data_pipeline/02_elevation_engine.py")


class ModelNotReady(RuntimeError):
    """No usable trained model: the checkpoint is missing, unreadable or incomplete.

    ``hint`` is the remediation for this failure (UIs show it instead of guessing one).
    """

    default_hint = TRAIN_HINT

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint or self.default_hint


class CheckpointMismatch(ModelNotReady):
    """The checkpoint does not fit the current graph / features (rebuild the datasets and retrain, or
    restore the graph the model was trained on)."""

    default_hint = RETRAIN_HINT


class ModelNotFinalized(ModelNotReady):
    """The checkpoint is a per-epoch file of an in-progress / interrupted run (not calibrated)."""

    default_hint = FINALIZE_HINT


def remediation(error: BaseException) -> tuple[str, str | None]:
    """``(what went wrong, how to fix it)`` of a :class:`ModelNotReady` (the hint is chosen by type).

    The message's trailing ``"; <hint>"`` is removed from the first part, so a UI can word the
    two separately without truncating the fix (a mismatch needs rebuilt datasets, not only a retrain).
    """
    message = str(error)
    hint = getattr(error, "hint", None)
    what = message
    if hint and message.endswith(hint):
        what = message[: -len(hint)].rstrip().rstrip(";").rstrip()
    elif "; " in message:
        what = message.split("; ")[0]
    return what.rstrip("."), hint


class GraphNotReady(RuntimeError):
    """The road graph file is missing, unreadable or lacks the attributes a predictor needs."""


class PredictionError(ValueError):
    """Invalid prediction input (rain shape / values, target range)."""
