# wannrun

A config-driven runner for `wann_tools` k-space calculations, with resource estimation and an extensible calculation registry.

## Quickstart

### 1. Install

Activate your Python environment and install the project:

```bash
python -m pip install -e .
```

### 2. Configure

Create `config.yaml`:

```yaml
hamiltonian:
  hr_file: data/in/wannier90_hr.dat
  nnkp_file: data/in/PdBi2.nnkp
  dtype: 64
  finite:
    enabled: true
    layers: 10

kspace:
  type: surface

  surface:
    n1: 60
    n2: 60
    level: 0
    fixed_axis: 2



calculation:
  type: spectral

  spectral:
    energies: [8.1159948861]
    eta: 0.01
    spin_projection: x
    layer_projection: 0


execution:
  backend: serial
  chunk_size: 8
  memory_limit_gb: 16

output:
  directory: data/out
  save_data: true
  plot: true
  format: pdf
```

### 3. Run

```bash
# Validate the configuration
wannrun validate config.yaml

# Estimate memory and runtime
wannrun estimate config.yaml

# Execute the calculation
wannrun run config.yaml
```

Alternatively, use `python -m wannrun.cli` instead of `wannrun` if the console command is not installed.

## Supported features

- **K-space:** surface, with path and volume support planned or under development.
- **Calculations:** spectral function with spin and layer projections.
- **Resource estimation:** estimate computational requirements before execution.
- **Extensible architecture:** add calculation implementations through the calculation registry.

## Project structure

```text
src/wannrun/
├── cli.py
├── config.py
├── runner.py
├── estimator.py
├── context.py
└── calculations/
    ├── base.py
    ├── registry.py
    └── spectral.py
```

## Requirements

- Python 3.11+
- NumPy
- Matplotlib
- PyYAML
- `wann_tools`

Ensure `wann_tools` is installed and the Hamiltonian file specified in `config.yaml` exists.

## Development

Install in editable mode:

```bash
python -m pip install -e .
```

New calculations should implement the `Calculation` base class and register themselves. Configuration validation, execution, and resource estimation should use the same `AppConfig`.