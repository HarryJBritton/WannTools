import numpy as np

def batched_eigh(A, chunk_size=256):
    n = A.shape[0]

    eigenvalues = np.empty((n, A.shape[1]), dtype=A.real.dtype)
    eigenvectors = np.empty_like(A)

    for i in range(0, n, chunk_size):
        print(f'Starting batch {i // chunk_size}')
        j = min(i + chunk_size, n)

        w, v = np.linalg.eigh(A[i:j])

        eigenvalues[i:j] = w
        eigenvectors[i:j] = v

    return eigenvalues, eigenvectors

