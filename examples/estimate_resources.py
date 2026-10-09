
import argparse


def estimate(
    nkx,
    nky,
    orbitals,
    layers,
    chunk,
    ranks,
    energies=1,
    dtype="complex128",
    baseline_gib=1.0,
    workspace_factor=1.5,
    seconds_per_k=None,
    efficiency=0.85,
):
    # nb = orbitals * layers * 2 spins
    nb = orbitals * layers * 2
    nk = nkx * nky

    # Bytes per complex number in hk and V
    complex_bytes = {
        "complex64": 8,
        "complex128": 16,
    }[dtype]

    # Per k-point, in bytes
    hk = complex_bytes * nb**2
    eigenvectors = complex_bytes * nb**2
    eigenvalues = 8 * nb
    green_function = 16 * nb * energies
    output = 8 * energies

    # Explicit arrays during Spectral(), per k-point
    explicit_per_k = (
        hk + eigenvectors + eigenvalues
        + green_function + output
    )

    # Conservative allowance for eigensolver temporary workspace.
    # This is a heuristic, not a guaranteed upper bound.
    batch_bytes = chunk * (
        hk + eigenvectors + eigenvalues
    )
    peak_gib = (
        baseline_gib
        + workspace_factor * batch_bytes / 2**30
        + chunk * (green_function + output) / 2**30
    )

    # Estimate memory across ranks on a single node.
    node_gib = ranks * peak_gib

    result = {
        "nk": nk,
        "nb": nb,
        "matrix_mib": hk / 2**20,
        "hk_mib": chunk * hk / 2**20,
        "eigenvectors_mib": chunk * eigenvectors / 2**20,
        "eigenvalues_mib": chunk * eigenvalues / 2**20,
        "green_function_mib": green_function / 2**20,
        "estimated_rank_peak_gib": peak_gib,
        "estimated_node_gib": node_gib,
    }

    if seconds_per_k is not None:
        result["estimated_runtime_seconds"] = (
            nk * seconds_per_k / (ranks * efficiency)
        )

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nkx", type=int, default=100)
    p.add_argument("--nky", type=int, default=100)
    p.add_argument("--orbitals", type=int, default=12)
    p.add_argument("--layers", type=int, default=10)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--ranks", type=int, default=16)
    p.add_argument("--energies", type=int, default=1)
    p.add_argument(
        "--dtype",
        choices=["complex64", "complex128"],
        default="complex128",
    )
    p.add_argument("--baseline-gib", type=float, default=1.0)
    p.add_argument("--workspace-factor", type=float, default=1.5)
    p.add_argument("--seconds-per-k", type=float)
    p.add_argument("--efficiency", type=float, default=0.85)

    args = p.parse_args()
    result = estimate(**vars(args))

    print("\nResource estimate")
    print("-" * 40)
    for key, value in result.items():
        print(f"{key:30s}: {value:,.3f}")