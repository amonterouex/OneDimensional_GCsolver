import numpy as np
import random

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
        
    def phi(self, r):
        scalar_input = np.isscalar(r)
        r = np.asarray(r)

        values = np.zeros_like(r, dtype=float)

        values[r < self.sigma] = np.inf
        values[(r >= self.sigma) & (r <= self.r0)] = -self.eps
        values[r > self.r0] = 0.0

        if scalar_input:
            return float(values)

        return values
        

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

class ExternalPotentialFlorian:
    def __init__(self, length, dx):
        self.dx = dx
        self.length = length
        self.xg = np.arange(dx/2.0, length, dx)
        self.nsize = len(self.xg)

class ExternalPotential:
    def __init__(self, length=None, dx=None, filename=None):
        if filename is not None:
            self.fname = filename

        elif length is not None and dx is not None:
            self.dx = dx
            self.length = length
            self.xg = np.arange(dx / 2.0, length, dx)
            self.nsize = len(self.xg)

        else:
            raise ValueError("Provide either filename or both length and dx")
        
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

        #Add - and set mu=0 when comparing with Florians files
        self.V = -np.asarray(dat[:, 1], dtype=float)
        # Read in my own external potential
        #self.Vext = np.asarray(dat[:, 1], dtype=float)
        self.length = self.xg.size
       
    def Vext_sin(self, x, *, n: int, A: float, phi: float, L: float):
        x = np.asarray(x)
        return A * np.sin(2 * np.pi * x * n / L + phi)

    def Vext_lin(self, x, *, x1: float, x2: float, E1: float, E2: float):
        x = np.asarray(x)
        # Avoid division by zero if x1 == x2
        if x1 == x2:
            return np.zeros_like(x, dtype=float)
        mask = (x > x1) & (x < x2)
        values = E1 + (x - x1) * (E2 - E1) / (x2 - x1)
        return np.where(mask, values, 0.0)

    def Vext_wall(self, x, *, xw: float, L: float):
        x = np.asarray(x)
        if xw < 1.0:
            mask = (x < xw) | (x > L - (2.0-xw)) #Change this with respect to Florian's code to avoid PBC issues in the simulation
        else:
            mask = (x < xw) | (x > L - xw)
            
        return np.where(mask, np.inf, 0.0)

    def generate_Vext(self, *, num_sin: int = 4, num_lin: int | None = None, wall: bool = True,):
        
        L = self.length
        x = self.xg
        
        if num_lin is None:
            num_lin = np.random.randint(1, 6)

        Avar = 1.0
        sin_parameters = []
        for n in range(1, num_sin + 1):
            sin_parameters.append({"n": n, "A": np.random.randn() * Avar, "phi": np.random.rand() * 2 * np.pi, "L": L, })

        Evar = 1.0
        lin_parameters = []
        for _ in range(num_lin):  lin_parameters.append({"x1": round(np.random.rand() * L, 2),"x2": round(np.random.rand() * L, 2), "E1": np.random.randn() * Evar,"E2": np.random.randn() * Evar,})

        xwmax = 2.0
        wall_params = {"xw": round(np.random.rand() * xwmax, 2), "L": L,}

        result = np.zeros_like(x, dtype=float)

        for sin_params in sin_parameters:
            result += self.Vext_sin(x, **sin_params)
            
        for lin_params in lin_parameters:
            result += self.Vext_lin(x, **lin_params)

        if wall:
            result += self.Vext_wall(x, **wall_params)

        self.V = result
        self.xg = x
        
