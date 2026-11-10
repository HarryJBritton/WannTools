from abc import ABC, abstractmethod
from typing import Any


# TODO This doesn't have to be abstract currently as no methods are required. 
class KSpaceProvider(ABC):

    @property
    @abstractmethod
    def count(self) -> int:
        ''' Counts the number of k points'''
        ...

    @property
    @abstractmethod
    def space_group(self) -> dict:
        ...

    @property
    @abstractmethod
    def bvec(self) -> Any:
        ...

    @property
    @abstractmethod
    def k(self) -> Any:
        ...