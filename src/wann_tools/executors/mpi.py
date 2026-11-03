
from wann_tools.executors.base import Executor

class MPIExecutor(Executor):

    def __init__(self, comm=None) -> None:
        from mpi4py import MPI

        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()



    def map_chunks(self, worker, n_chunks: int) -> list:
        if n_chunks < 0:
            raise ValueError('n_chunks cannot be negative')

        local_results = []
        local_error = None

        try:
            for chunk_id in range(self.rank, n_chunks, self.size):
                local_results.append((chunk_id, worker(chunk_id)))
        except Exception as exc:
            local_error = (
                type(exc).__name__,
                str(exc),
                self.rank,
            )

        errors = self.comm.allgather(local_error)

        if any(error is not None for error in errors):
            details = [
                error for error in errors if error is not None
            ]
            raise RuntimeError(f'MPI chunk execution failed: {details}')
        
        gathered = self.comm.gather(local_results, root=0)

        if self.rank != 0:
            return []

        results = [None] * n_chunks
        for rank_results in gathered: # type: ignore
            for chunk_id, result in rank_results:
                results[chunk_id] = result

        return results

    def close(self) -> None:
        pass