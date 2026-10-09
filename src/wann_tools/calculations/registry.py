from wann_tools.calculations.base import Calculation


CALCULATIONS = {}

def register_calculation(cls):
    name = cls.name

    if name in CALCULATIONS:
        raise ValueError(
            f'Calculation already registerd: {name}'
        )

    CALCULATIONS[name] = cls
    return cls

def get_calculation(name) -> Calculation:
    try:
        return CALCULATIONS[name]()
    except KeyError:
        available = ", ".join(sorted(CALCULATIONS))
        raise ValueError(
            f"Unknown calculation {name!r}. "
            f"Available: {available}"
        )