from abc import ABC, abstractmethod

class KSpaceProvider(ABC):

    @abstractmethod
    def count(self) -> int:
        """Return the total number of k-points."""
        ...

    @abstractmethod
    def iter_chunks(self, chunk_size: int):
        """Yield chunks of k-points."""
        ...

    @abstractmethod
    def metadata(self) -> dict:
        """Return geometry and output-coordinate metadata."""
        ...