from pathlib import Path

import numpy as np
import torch


class MatrixNorm:
    def __init__(self):
        self.data_dir = Path("/home/miket/StereoDataset")
        self.types = ["Clean", "Noisy"]

    def fitMN(self, X: np.ndarray, max_iter=300, tol=1e-6, reg=1e-8, verbose=True):
        n, p, q = X.shape
        U = np.eye(p)
        V = np.eye(q)

        for it in range(max_iter):
            # Update U given V
            V_inv = np.linalg.inv(V + reg * np.eye(q))
            U_new = np.zeros((p, p))
            for i in range(n):
                U_new += X[i] @ V_inv @ X[i].T
            U_new /= n * q

            # Update V given new U
            U_inv = np.linalg.inv(U_new + reg * np.eye(p))
            V_new = np.zeros((q, q))
            for i in range(n):
                V_new += X[i].T @ U_inv @ X[i]
            V_new /= n * p

            # Put trace constraint: normalize U's trace, push scale into V
            scale = np.trace(U_new) / p
            U_new /= scale
            V_new *= scale

            delta_U = np.linalg.norm(U_new - U) / np.linalg.norm(U)
            delta_V = np.linalg.norm(V_new - V) / np.linalg.norm(V)

            if verbose:
                print(f"Iter {it}: dU={delta_U:.2e}, dV={delta_V:.2e}", flush=True)

            U, V = U_new, V_new

            if delta_U < tol and delta_V < tol:
                if verbose:
                    print(f"Converged at iter {it}", flush=True)
                break
        else:
            if verbose:
                print("Hit max_iter without converging", flush=True)

        return U, V

    def saveErrorCov(self):
        errors = []
        for type_ in self.types:
            for pt in (self.data_dir / f"{type_}Normalized").glob("*.pt"):
                error = torch.load(pt)[3].numpy()
                errors.append(error)
        errors = np.stack(errors)

        # Center the data if it's not already zero-mean
        mean = errors.mean(axis=0)
        errors = errors - mean

        row_cov, col_cov = self.fitMN(errors)

        torch.save(
            torch.tensor(np.linalg.cholesky(row_cov)), self.data_dir / "cholrow.pt"
        )
        torch.save(
            torch.tensor(np.linalg.cholesky(col_cov)), self.data_dir / "cholcol.pt"
        )


matrix_data = MatrixNorm()
matrix_data.saveErrorCov()
print("Matrix Normal Fit Complete.")
