from wann_tools.executors.base import Executor

class SerialExecutor(Executor):
    def map_chunks(self, worker, n_chunks: int) -> list:
        if n_chunks < 0:
            raise ValueError('n_chunks cannot be negative')

        return [worker(chunk_id) for chunk_id in range(n_chunks)]


    def close(self) -> None:
        pass