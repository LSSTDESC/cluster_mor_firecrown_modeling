"""The mass richness kernel module.

This module holds the classes that define the mass richness relations
that can be included in the cluster abundance integrand.  These are
implementations of Kernels.
"""

from abc import abstractmethod

import numpy as np
import numpy.typing as npt
from scipy import special

from firecrown import parameters
from firecrown.updatable import Updatable


class MassRichnessGaussian(Updatable):
    """The representation of mass richness relations that are of a gaussian form."""

    @staticmethod
    def lognormal_value(
        p: tuple[float, float, float],
        lambda_value: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        pivot_lambda: float,
        pivot_redshift: float,
    ) -> npt.NDArray[np.float64]:
        """Return predicted mass quantity for a given richness and mass."""
        delta_log_lambda = lambda_value - np.log10(pivot_lambda)
        delta_z = np.log10(1.0+z) - np.log10(1.0+pivot_redshift)

        result = p[0] + p[1] * delta_log_lambda + p[2] * delta_z
        assert isinstance(result, np.ndarray)
        return result

    @abstractmethod
    def get_mass_mean(
        self,
        lambda_value: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return observed quantity corrected by redshift and mass."""

MCCLINTOCK_DEFAULT_M0 = 14.489
MCCLINTOCK_DEFAULT_FLAMBDA = 1.356
MCCLINTOCK_DEFAULT_GZ = -0.3

class mcclintock2019(MassRichnessGaussian):
    """The mass richness relation defined in Murata 19 for a unbinned data vector."""

    def __init__(
        self,
        pivot_lambda: float,
        pivot_redshift: float,
    ):
        super().__init__()
        self.pivot_redshift = pivot_redshift
        self.pivot_lambda = pivot_lambda 

        # Updatable parameters
        self.mcclintock_m0 = parameters.register_new_updatable_parameter(
            default_value=MCCLINTOCK_DEFAULT_M0
        )
        self.mcclintock_flambda = parameters.register_new_updatable_parameter(
            default_value=MCCLINTOCK_DEFAULT_FLAMBDA
        )
        self.mcclintock_gz = parameters.register_new_updatable_parameter(
            default_value=MCCLINTOCK_DEFAULT_GZ
        )

    def get_mass_mean(
        self,
        lambda_value: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return observed quantity corrected by redshift and mass."""
        return MassRichnessGaussian.lognormal_value(
            (self.mcclintock_m0, self.mcclintock_flambda, self.mcclintock_gz),
            lambda_value,
            z,
            self.pivot_lambda,
            self.pivot_redshift,
        )

