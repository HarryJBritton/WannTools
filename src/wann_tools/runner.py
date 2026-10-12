from wann_tools.calculations.base import Calculation
from wann_tools.config import AppConfig
from wann_tools.wannham import build_hamiltonian
from wann_tools.kspace import build_kspace
from wann_tools.calculations.registry import get_calculation
from wann_tools.context import build_context
from wann_tools.executors import build_executor

import numpy as np


def run(config: AppConfig):
    ham = build_hamiltonian(config)
    kspace = build_kspace(config)

    calculation = get_calculation(config.calculation.type)

    context = build_context(config, ham, kspace)

    calculation.validate(
        config=config,
        context=context,
    )

    with build_executor(config.execution) as executor:
        return calculation.run(
            config=config,
            context=context,
            executor=executor,
        )
