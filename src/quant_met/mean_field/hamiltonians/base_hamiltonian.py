# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Provides the base class for Hamiltonians."""

import pathlib
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd

from quant_met.geometry import BaseLattice
from quant_met.mean_field._utils import _check_valid_array
from quant_met.parameters.hamiltonians import GenericParameters, HamiltonianParameters

GenericHamiltonian = TypeVar("GenericHamiltonian", bound="BaseHamiltonian[HamiltonianParameters]")


class BaseHamiltonian(Generic[GenericParameters], ABC):
    """Base class for Hamiltonians.

    This abstract class provides the essential framework for defining various
    Hamiltonians used in solid-state physics. It includes methods for constructing
    the Hamiltonian based on a set of parameters, calculating properties such as
    energy bands, conducting derivatives, and diagonalizing the Hamiltonian to
    obtain eigenstates and eigenvalues. Subclasses should implement methods to
    provide specific Hamiltonian forms.

    Parameters
    ----------
    parameters : :class:`quant_met.parameters.hamiltonians.GenericParameters`
        An object containing the necessary parameters to define the Hamiltonian,
        including lattice parameters, critical constants, and Hubbard interaction
        strengths.

    Attributes
    ----------
    name : str
        Name or identifier of the Hamiltonian.
    beta : float
        Inverse temperature (related to thermal excitations).
    q : :class:`numpy.ndarray`
        A two-dimensional array defining a momentum offset, typically in
        reciprocal space.
    lattice : :class:`quant_met.geometry.BaseLattice`
        The lattice structure in which the Hamiltonian is defined.
    hubbard_int_orbital_basis : :class:`numpy.ndarray`
        Interaction terms for Hubbard-type models represented in orbital basis.
    number_of_bands : int
        The total number of bands calculated based on the orbital basis provided.
    delta_orbital_basis : :class:`numpy.ndarray`
        An array initialized for the order parameter or pairing potentials.
    """

    def __init__(self, parameters: GenericParameters) -> None:
        self.name = parameters.name
        self.beta = parameters.beta
        self.q = parameters.q if parameters.q is not None else np.zeros(2)

        self.lattice = self.setup_lattice(parameters)
        self.hubbard_int_orbital_basis = parameters.hubbard_int_orbital_basis
        self.number_of_bands = len(self.hubbard_int_orbital_basis)
        self.delta_orbital_basis = np.zeros(self.number_of_bands, dtype=np.complex64)

    @abstractmethod
    def setup_lattice(self, parameters: GenericParameters) -> BaseLattice:  # pragma: no cover
        """Set up the lattice based on the provided parameters.

        Parameters
        ----------
        parameters : GenericParameters
            Input parameters containing necessary information for lattice construction.

        Returns
        -------
        BaseLattice
            An instance of a lattice object configured according to the input parameters.
        """

    @classmethod
    @abstractmethod
    def get_parameters_model(cls) -> type[GenericParameters]:  # pragma: no cover
        """Return the specific parameters model for the subclass.

        This method should provide the structure of parameters required by
        subclasses to initialize the Hamiltonian.

        Returns
        -------
        type
            The parameters model class type specific to the Hamiltonian subclass.
        """

    @abstractmethod
    def hamiltonian(
        self, k: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.complex64]:  # pragma: no cover
        """Return the normal state Hamiltonian.

        Parameters
        ----------
        k : numpy.ndarray
            List of k points in reciprocal space.

        Returns
        -------
        class `numpy.ndarray`
            The Hamiltonian matrix evaluated at the provided k points.
        """

    @abstractmethod
    def hamiltonian_derivative(
        self, k: npt.NDArray[np.float64], direction: str
    ) -> npt.NDArray[np.complex64]:  # pragma: no cover
        """Calculate the spatial derivative of the Hamiltonian.

        Parameters
        ----------
        k : numpy.ndarray
            List of k points in reciprocal space.
        direction : str
            Direction for the derivative, either 'x' or 'y'.

        Returns
        -------
        :class: `numpy.ndarray`
            The derivative of the Hamiltonian matrix in the specified direction.
        """

    def save(self, filename: pathlib.Path) -> None:
        """Save the Hamiltonian configuration as an HDF5 file.

        This method stores Hamiltonian parameters and the delta orbital basis in
        a specified HDF5 file format for later retrieval.

        Parameters
        ----------
        filename : class:`pathlib.Path`
            The file path where the Hamiltonian will be saved. Must end with .hdf5.
        """
        with h5py.File(f"{filename.absolute()}", "w") as f:
            f.create_dataset("delta", data=self.delta_orbital_basis)
            for key, value in vars(self).items():
                if key != "lattice":
                    f.attrs[key.strip("_")] = value
            f.attrs["lattice_constant"] = self.lattice.lattice_constant

    @classmethod
    def from_file(cls: type[GenericHamiltonian], filename: pathlib.Path) -> GenericHamiltonian:
        """Initialize a Hamiltonian from a previously saved HDF5 file.

        This class method allows users to reconstruct a Hamiltonian object
        from saved attributes and matrix configurations stored in an HDF5 file.

        Parameters
        ----------
        filename : :class:`pathlib.Path`
            The file path to the HDF5 file containing Hamiltonian data.

        Returns
        -------
        class:`BaseHamiltonian[GenericParameters]`
            An instance of the Hamiltonian initialized with data from the file.
        """
        with h5py.File(str(filename), "r") as f:
            config_dict = dict(f.attrs.items())
            config_dict["delta"] = f["delta"][()]

        parameters_model = cls.get_parameters_model()
        parameters = parameters_model.model_validate(config_dict)
        return cls(parameters=parameters)

    def bdg_hamiltonian(self, k: npt.NDArray[np.float64]) -> npt.NDArray[np.complex64]:
        """Generate the Bogoliubov-de Gennes (BdG) Hamiltonian.

        The BdG Hamiltonian incorporates pairing interactions and is used to
        study superfluid and superconducting phases. This method constructs a
        2x2 block Hamiltonian based on the normal state Hamiltonian and the
        pairing terms.

        Parameters
        ----------
        k : :class:`numpy.ndarray`
            List of k points in reciprocal space.

        Returns
        -------
        :class:`numpy.ndarray`
            The BdG Hamiltonian matrix evaluated at the specified k points.
        """
        assert _check_valid_array(k)
        if k.ndim == 1:
            k = np.expand_dims(k, axis=0)

        h = np.zeros(
            (k.shape[0], 2 * self.number_of_bands, 2 * self.number_of_bands), dtype=np.complex64
        )

        h[:, 0 : self.number_of_bands, 0 : self.number_of_bands] = self.hamiltonian(k)
        h[
            :,
            self.number_of_bands : 2 * self.number_of_bands,
            self.number_of_bands : 2 * self.number_of_bands,
        ] = -self.hamiltonian(self.q - k).conjugate()

        for i in range(self.number_of_bands):
            h[:, self.number_of_bands + i, i] = self.delta_orbital_basis[i]

        h[:, 0 : self.number_of_bands, self.number_of_bands : self.number_of_bands * 2] = (
            h[:, self.number_of_bands : self.number_of_bands * 2, 0 : self.number_of_bands]
            .copy()
            .conjugate()
        )

        return h.squeeze()

    def bdg_hamiltonian_derivative(
        self, k: npt.NDArray[np.float64], direction: str
    ) -> npt.NDArray[np.complex64]:
        """Calculate the derivative of the BdG Hamiltonian.

        This method computes the spatial derivative of the Bogoliubov-de Gennes
        Hamiltonian with respect to the specified direction.

        Parameters
        ----------
        k : :class:`numpy.ndarray`
            List of k points in reciprocal space.
        direction : str
            Direction for the derivative, either 'x' or 'y'.

        Returns
        -------
        :class:`numpy.ndarray`
            The derivative of the BdG Hamiltonian matrix in the specified direction.
        """
        assert _check_valid_array(k)
        if k.ndim == 1:
            k = np.expand_dims(k, axis=0)

        h = np.zeros(
            (k.shape[0], 2 * self.number_of_bands, 2 * self.number_of_bands), dtype=np.complex64
        )

        h[:, 0 : self.number_of_bands, 0 : self.number_of_bands] = self.hamiltonian_derivative(
            k, direction
        )
        h[
            :,
            self.number_of_bands : 2 * self.number_of_bands,
            self.number_of_bands : 2 * self.number_of_bands,
        ] = -self.hamiltonian_derivative(-k, direction).conjugate()

        return h.squeeze()

    def diagonalize_nonint(
        self, k: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Diagonalizes the normal state Hamiltonian.

        This method computes the eigenvalues and eigenvectors of the normal state
        Hamiltonian for the given k points. It is essential for analyzing the
        electronic properties and band structure of materials.

        Parameters
        ----------
        k : :class:`numpy.ndarray`
            List of k points in reciprocal space.

        Returns
        -------
        tuple
            - :class:`numpy.ndarray`: Eigenvalues of the normal state Hamiltonian.
            - :class:`numpy.ndarray`: Eigenvectors (Bloch wavefunctions) corresponding to
            the eigenvalues.
        """
        k_point_matrix = self.hamiltonian(k)
        if k_point_matrix.ndim == 2:
            k_point_matrix = np.expand_dims(k_point_matrix, axis=0)
            k = np.expand_dims(k, axis=0)

        bloch_wavefunctions = np.zeros(
            (len(k), self.number_of_bands, self.number_of_bands),
            dtype=complex,
        )
        band_energies = np.zeros((len(k), self.number_of_bands))

        for i in range(len(k)):
            band_energies[i], bloch_wavefunctions[i] = np.linalg.eigh(k_point_matrix[i])

        return band_energies.squeeze(), bloch_wavefunctions.squeeze()

    def diagonalize_bdg(
        self,
        k: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex64]]:
        """Diagonalizes the BdG Hamiltonian.

        This method computes the eigenvalues and eigenvectors of the Bogoliubov-de
        Gennes Hamiltonian, providing insight into the quasiparticle excitations in
        superconducting states.

        Parameters
        ----------
        k : :class:`numpy.ndarray`
            List of k points in reciprocal space.

        Returns
        -------
        tuple
            - :class:`numpy.ndarray`: Eigenvalues of the BdG Hamiltonian.
            - :class:`numpy.ndarray`: Eigenvectors corresponding to the eigenvalues of the
              BdG Hamiltonian.
        """
        bdg_matrix = self.bdg_hamiltonian(k=k)
        if bdg_matrix.ndim == 2:
            bdg_matrix = np.expand_dims(bdg_matrix, axis=0)
            k = np.expand_dims(k, axis=0)

        bdg_wavefunctions = np.zeros(
            (len(k), 2 * self.number_of_bands, 2 * self.number_of_bands),
            dtype=np.complex64,
        )
        bdg_energies = np.zeros((len(k), 2 * self.number_of_bands))

        for i in range(len(k)):
            bdg_energies[i], bdg_wavefunctions[i] = np.linalg.eigh(bdg_matrix[i])

        return bdg_energies.squeeze(), bdg_wavefunctions.squeeze()

    def gap_equation(
        self,
        k: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.complex64]:
        """Calculate the gap equation.

        The gap equation determines the order parameter for superconductivity by
        relating the pairings to the spectral properties of the BdG Hamiltonian.

        Parameters
        ----------
        k : :class:`numpy.ndarray`
            List of k points in reciprocal space.

        Returns
        -------
        :class:`numpy.ndarray`
            New pairing gap in orbital basis, adjusted to remove global phase.
        """
        bdg_energies, bdg_wavefunctions = self.diagonalize_bdg(k=k)
        delta = np.zeros(self.number_of_bands, dtype=np.complex64)

        for i in range(self.number_of_bands):
            sum_tmp = 0
            for j in range(2 * self.number_of_bands):
                for k_index in range(len(k)):
                    sum_tmp += (
                        np.conjugate(bdg_wavefunctions[k_index, i, j])
                        * bdg_wavefunctions[k_index, i + self.number_of_bands, j]
                        * _fermi_dirac(bdg_energies[k_index, j].item(), self.beta)
                    )
            delta[i] = (-self.hubbard_int_orbital_basis[i] * sum_tmp / len(k)).conjugate()

        delta_without_phase: npt.NDArray[np.complex64] = delta * np.exp(
            -1j * np.angle(delta[np.argmax(np.abs(delta))])
        )
        return delta_without_phase

    def calculate_bandstructure(
        self,
        k: npt.NDArray[np.float64],
        overlaps: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
    ) -> pd.DataFrame:
        """Calculate the band structure.

        This method computes the energy bands of the system by diagonalizing
        the normal state Hamiltonian over the provided k points. It can also
        calculate overlaps with provided wavefunctions if available.

        Parameters
        ----------
        k : :class:`numpy.ndarray`
            List of k points in reciprocal space.
        overlaps : tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`), optional
            A tuple containing two sets of wavefunctions for overlap calculations.

        Returns
        -------
        `pandas.DataFrame`
            A DataFrame containing the calculated band energies with optional
            overlap information.
        """
        results = pd.DataFrame(
            index=range(len(k)),
            dtype=float,
        )
        energies, wavefunctions = self.diagonalize_nonint(k)

        for i, (energy_k, wavefunction_k) in enumerate(zip(energies, wavefunctions, strict=False)):
            if self.number_of_bands == 1:
                results.loc[i, "band"] = energy_k
            else:
                for band_index in range(self.number_of_bands):
                    results.loc[i, f"band_{band_index}"] = energy_k[band_index]

                    if overlaps is not None:
                        results.loc[i, f"wx_{band_index}"] = (
                            np.abs(np.dot(wavefunction_k[:, band_index], overlaps[0])) ** 2
                            - np.abs(np.dot(wavefunction_k[:, band_index], overlaps[1])) ** 2
                        )

        return results

    def calculate_density_of_states(
        self,
        k: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate the density of states (DOS).

        This method computes the density of states by evaluating the eigenvalues
        of the BdG Hamiltonian over a specified energy range. The DOS provides
        insights into the allowed energy levels of the system.

        Parameters
        ----------
        k : :class:`numpy.ndarray`
            List of k points in reciprocal space.

        Returns
        -------
        tuple
            - numpy.ndarray: Energy levels over which the density of states is calculated.
            - numpy.ndarray: The density of states corresponding to each energy level.
        """
        bands, _ = self.diagonalize_bdg(k=k)
        energies = np.linspace(start=np.min(bands), stop=np.max(bands), num=5000)
        density_of_states = np.zeros(shape=energies.shape, dtype=np.float64)

        for i, energy in enumerate(energies):
            density_of_states[i] = np.sum(
                _gaussian(x=(energy - bands.flatten()), sigma=1e-2)
            ) / len(k)
        return energies, density_of_states

    def calculate_spectral_gap(self, k: npt.NDArray[np.float64]) -> float:
        """Calculate the spectral gap.

        This method evaluates the spectral gap of the system by examining the
        density of states. It identifies the range of energy where there are no
        states and thus determines the energy difference between the highest
        occupied and lowest unoccupied states.

        Parameters
        ----------
        k : :class:`numpy.ndarray`
            List of k points in reciprocal space.

        Returns
        -------
        float
            The calculated spectral gap.
        """
        energies, density_of_states = self.calculate_density_of_states(k=k)

        coherence_peaks = np.where(np.isclose(density_of_states, np.max(density_of_states)))[0]

        gap_region = density_of_states[coherence_peaks[0] : coherence_peaks[1] + 1] / np.max(
            density_of_states
        )
        energies_gap_region = energies[coherence_peaks[0] : coherence_peaks[1] + 1]
        zero_indeces = np.where(gap_region <= 1e-10)[0]
        if len(zero_indeces) == 0:
            gap = 0
        else:
            gap = (
                energies_gap_region[zero_indeces[-1]] - energies_gap_region[zero_indeces[0]]
            ).item()

        return gap


def _gaussian(x: npt.NDArray[np.float64], sigma: float) -> npt.NDArray[np.float64]:
    gaussian: npt.NDArray[np.float64] = np.exp(-(x**2) / (2 * sigma**2)) / np.sqrt(
        2 * np.pi * sigma**2
    )
    return gaussian


def _fermi_dirac(energy: float, beta: float | None) -> float:
    fermi_dirac: float = (
        (1 if energy < 0 else 0) if beta is None else 1 / (1 + np.exp(beta * energy))
    )
    return fermi_dirac
