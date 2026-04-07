from abc import ABC, abstractmethod


class ArrivalModel(ABC):
    """Interface for session arrival timing models.

    Two implementations: MMPPArrivalModel and TODArrivalModel.
    """

    @abstractmethod
    def get_next_arrival_delay(self, current_dt) -> float:
        """Return seconds until the next session arrival."""

    @abstractmethod
    def fit(self) -> None:
        """Fit the model from events_clean.parquet."""

    @abstractmethod
    def save(self, path) -> None:
        """Serialize model to disk."""

    @abstractmethod
    def load(self, path) -> None:
        """Load model from disk."""
