from wann_tools.executors.serial import SerialExecutor

def build_executor(config):
    if config.backend == 'serial':
        return SerialExecutor()

    if config.backend == 'mpi':
        from wann_tools.executors.mpi import MPIExecutor
        return MPIExecutor()

    raise ValueError(f'Unsupported executor backend: {config.backend!r}')