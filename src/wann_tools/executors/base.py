from abc import ABC, abstractmethod
from collections.abc import Callable

class Executor(ABC):

    @abstractmethod
    def map_chunks(
        self,
        worker: Callable[[int], object],
        n_chunks: int,
    ) -> list:
        '''Run worker(chunk_id) for each chunk. 
        
        Returns results in chunk order. In distributed execustion,
        only the root rank returns the complete result list.
        '''
        raise NotImplementedError


    @abstractmethod
    def close(self) -> None:
        """Release executor resources."""
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
