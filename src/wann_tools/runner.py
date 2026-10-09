from wann_tools.config import load_config


config = load_config("config.yaml")

print(config.hamiltonian.hr_file)
print(config.kspace.type)
print(config.calculation.type)

if config.kspace.type == "path":
    print(config.kspace.path.path)

if config.calculation.type == "spectral":
    print(config.calculation.spectral.energies)