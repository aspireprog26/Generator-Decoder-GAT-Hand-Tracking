import torch
import numpy as np
from pathlib import Path
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri


class MatrixNorm:
    def __init__(self):
        self.data_dir = Path("/home/mrtcloud-1/Documents/StereoDataset")
        self.types = ["Clean", "Noisy"]

    def saveErrorCov(self):
        errors = []

        for type in self.types:
            for pt in (self.data_dir / f"{type}Normalized").glob("*.pt"):
                error = (torch.load(pt)[3]).numpy()
                errors.append(error)
        errors = np.stack(errors)

        numpy2ri.activate()
        ro.r("library(mixMatrix)")

        r_data = ro.ListVector(
            {
                str(i + 1): ro.r.matrix(self.data[i, :, :], nrow=21, ncol=7)
                for i in range(self.data.shape[0])
            }
        )

        fit = ro.r("""
            function (data) {
                fit <- matrixNormal(data)
                return fit
            }

        """)(r_data)

        row_cov = np.array(fit.rx2("U"))
        col_cov = np.array(fit.rx2("V"))

        chol_row = torch.tensor(np.linalg.cholesky(row_cov))
        chol_col = torch.tensor(np.linalg.cholesky(col_cov))

        torch.save(chol_row, self.data_dir / "cholrow.pt")
        torch.save(chol_col, self.data_dir / "cholcol.pt")


matrix_data = MatrixNorm()
matrix_data.saveErrorCov()
