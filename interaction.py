import numpy as np


class Interaction:
    """Hard-core, finite-range nearest-neighbor pair interaction."""

    names = {"hard-rod", "square-well", "square-shoulder", "triangle-well", "ramp"}

    def __init__(self, kind: str, sigma: float, r0: float, epsilon: float):
        if kind not in self.names:
            raise ValueError(f"Unknown interaction type: {kind!r}")
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma must be finite and strictly positive")
        if not np.isfinite(epsilon) or epsilon < 0.0:
            raise ValueError("epsilon must be finite and non-negative")
        if not np.isfinite(r0) or r0 <= sigma:
            raise ValueError("r0 must be finite and greater than sigma")
        if r0 > 2.0 * sigma * (1.0 + 1e-14):
            raise ValueError("Nearest-neighbor reduction requires r0 <= 2*sigma")

        self.kind = kind #So far only one kind of potential is implemented
        self.sigma = float(sigma)
        self.r0 = float(r0)
        self.eps = float(epsilon)

    # This is very optimized for the discontinuities in the square-well. If later on one wants to include different potentials
    # or different numbers of discontinuities this should be changed
    def integrated_boltzmann_over_cell(self, separation: np.ndarray, dx: float, beta: float) -> np.ndarray:
        """
        - Integrate exp[-beta u(r)] across one source cell. This avoids assigning a full quadrature weight to either side of the
        discontinuities at sigma and r0.
        - The returned matrix already contains the length factor; do not multiply it by dx again later on.
        """
        if self.kind != "square-well":
            raise NotImplementedError("Cell integration is implemented for square-well only")

        lo = separation - 0.5 * dx
        hi = separation + 0.5 * dx

        # This took the a while to derive but I am pretty sure it is correct
        well_length = np.maximum(0.0, np.minimum(hi, self.r0) - np.maximum(lo, self.sigma))
        free_length = np.maximum(0.0, hi - np.maximum(lo, self.r0))
        return np.exp(beta * self.eps) * well_length + free_length


class ExternalPotential:
    def __init__(self, filename: str, mu: float, temp: float):
        if not np.isfinite(temp) or temp <= 0.0:
            raise ValueError("temperature must be finite and positive")
        self.file = filename
        self.mu = float(mu)
        self.temp = float(temp)
        self.beta = 1.0 / self.temp
        self.load_file(filename)

    def load_file(self, filename: str) -> None:
        dat = np.loadtxt(filename)
        if dat.ndim != 2 or dat.shape[1] < 2 or dat.shape[0] < 2:
            raise ValueError("potential file must contain at least two rows and two columns")

        self.xg = np.asarray(dat[:, 0], dtype=float)
        if not np.all(np.diff(self.xg) > 0.0):
            raise ValueError("x grid must be strictly increasing")
        self.dx = float(self.xg[1] - self.xg[0])
        if not np.allclose(np.diff(self.xg), self.dx, rtol=1e-12, atol=1e-14):
            raise ValueError("x grid is not equally spaced")

        #Add -self.temp and set mu=0 when comparing with Florians files
        self.Vext = -self.temp*np.asarray(dat[:, 1], dtype=float)
        # Read in my own external potential
        #self.Vext = np.asarray(dat[:, 1], dtype=float)
        self.nsize = self.xg.size
