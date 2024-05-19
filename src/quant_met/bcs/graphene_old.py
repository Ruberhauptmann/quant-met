import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import interpolate, optimize, special

from quant_met.configuration import Configuration


class DeltaVector:
    def __init__(
        self,
        hdf_file=None,
        k_points: npt.NDArray | None = None,
        initial: float | None = None,
    ):
        if hdf_file is not None:
            self.data = pd.DataFrame(pd.read_hdf(hdf_file, key="table"))
            self.k_points = np.column_stack(
                (np.array(self.data.loc[:, "kx"]), np.array(self.data.loc[:, "ky"]))
            )
        else:
            self.k_points = k_points
            self.data = pd.DataFrame(
                columns=["kx", "ky", "delta_1", "delta_2"],
                index=range(len(k_points)),
                dtype=np.float64,
            )
            self.data.loc[:, "kx"] = self.k_points[:, 0]
            self.data.loc[:, "ky"] = self.k_points[:, 1]
            if initial is not None:
                self.data.loc[:, "delta_1"] = initial
                self.data.loc[:, "delta_2"] = initial

    def __repr__(self):
        return self.data.to_string(index=False)

    def update_from_1d_vector(self, delta: npt.NDArray):
        for n in range(2):
            offset = int(n * len(delta) / 2)
            self.data.loc[:, f"delta_{n + 1}"] = delta[
                offset : offset + len(self.k_points)
            ]

    def save(self, path):
        self.data.to_hdf(path, key="table", format="table", data_columns=True)

    @property
    def as_1d_vector(self) -> npt.NDArray:
        return np.concatenate(
            [
                np.array(self.data.loc[:, "delta_1"].values),
                np.array(self.data.loc[:, "delta_2"].values),
            ]
        )


def generate_non_interacting_hamiltonian(k: npt.NDArray, config: Configuration):
    h = np.zeros((2, 2), dtype=complex)

    t_gr = config.t_gr
    t_prime = config.t_x
    a = config.a

    h[0, 1] = t_gr * (
        np.exp(1j * k[1] * a / np.sqrt(3))
        + 2 * np.exp(-0.5j * a / np.sqrt(3) * k[1]) * (np.cos(0.5 * a * k[0]))
    )

    h[1, 0] = h[0, 1].conjugate()

    """
    h[0, 0] = (
            -2
            * t_prime
            * (
                    np.cos(a * k[0])
                    + 2 * np.cos(0.5 * a * k[0]) * np.cos(0.5 * np.sqrt(3) * a * k[1])
            )
    )

    h[1, 1] = h[0, 0]
    """

    # h = h - config.mu * np.eye(2)

    return h


def generate_bloch(k_points: npt.NDArray, config: Configuration):
    bloch = np.zeros((len(k_points), 2, 2), dtype=complex)
    energies = np.zeros((len(k_points), 2))

    for i, k in enumerate(k_points):
        h = generate_non_interacting_hamiltonian(k, config)

        energies[i], bloch[i] = np.linalg.eigh(h)
        energies[i] = energies[i] - config.mu

    bloch_absolute = np.power(np.absolute(bloch), 2)
    print(bloch_absolute)

    return energies, bloch_absolute


def gap_equation_real(
    delta_k: npt.NDArray,
    U: npt.NDArray,
    beta: float,
    bloch_absolute: npt.NDArray,
    energies: npt.NDArray,
    # filling_in: float
    mu: float,
):
    # number_k_points = len(energies)
    # print(number_k_points)
    return_vector = np.zeros(len(delta_k))

    number_k_points = int(len(return_vector) / 2)

    """
    for n in [0, 1]:
        offset = int(len(delta_k) / 3 * n)
        for k_index in range(0, number_k_points):
            sum_tmp = 0
            for alpha in [0, 1]:
                prefactor = U[alpha] * bloch_absolute[k_index][alpha][n]
                integral = 0
                for m in [0, 1]:
                    for k_prime_index in range(0, number_k_points):
                        integral += 1 / np.sqrt(
                            (energies[k_prime_index][m] - mu) ** 2
                            + np.abs(delta_k[k_prime_index+offset]) ** 2
                        ) * bloch_absolute[k_prime_index][alpha][m] * delta_k[k_prime_index] * 1
                            # * np.tanh(
                            #    0.5
                            #    * beta
                            #    * np.sqrt(
                            #        energies[k_prime_index][m] ** 2
                            #        + np.abs(delta_k[k_prime_index]) ** 2
                            #    )
                            # )
                sum_tmp += prefactor * integral / (number_k_points * 2.5980762113533156)

            return_vector[k_index + offset] = sum_tmp
    """

    for n in [0, 1]:
        offset_n = int(len(delta_k) / 2 * n)
        for k_prime_index in range(0, number_k_points):
            sum_tmp = 0
            for alpha in [0, 1]:
                for m in [0, 1]:
                    offset = int(len(delta_k) / 2 * m)
                    for k_index in range(0, number_k_points):
                        sum_tmp += (
                            U[alpha]
                            * bloch_absolute[k_prime_index][alpha][n]
                            * bloch_absolute[k_index][alpha][m]
                            * delta_k[k_index + offset]
                            / (
                                2
                                * np.sqrt(
                                    (energies[k_index][m]) ** 2
                                    + np.abs(delta_k[k_index + offset]) ** 2
                                )
                            )
                        )
                        # sum_tmp += U[alpha] * bloch_absolute[k_prime_index][alpha][n] * bloch_absolute[k_index][alpha][m] * delta_k[k_index+offset] / (2 * np.sqrt((energies[k_index][m] - mu) ** 2 + np.abs(delta_k[k_index+offset]) ** 2))
                        # sum_tmp += delta_k[k_index+offset] / (2 * np.sqrt((energies[k_index][m] - mu) ** 2 + np.abs(delta_k[k_index+offset]) ** 2))

                    # U[alpha] * bloch_absolute[k_prime_index][alpha][n]
            return_vector[k_prime_index + offset_n] = sum_tmp / (
                2.5980762113533156 * number_k_points
            )
            # return_vector[k_prime_index + offset_n] = U[0] * sum_tmp / (2.5980762113533156 * number_k_points)
    # return delta - delta_out
    # print(delta)
    # return return_vector - delta_k
    return return_vector

    # return 1 - 0.5 * U[0] * sum_tmp / 2.5980762113533156
    # return return_vector - delta_k


"""
def gap_equation_real(
        delta_k: npt.NDArray,
        U: npt.NDArray,
        beta: float,
        bloch_absolute: npt.NDArray,
        energies: npt.NDArray,
        #filling_in: float
        mu: float
):
    return_vector = np.zeros(len(delta_k))
    #mu = delta_k[-1]

    #number_k_points = int(len(return_vector) / 2) - 1
    number_k_points = int(len(return_vector) / 2)

    for n in [0, 1]:
        offset = int(len(delta_k) / 2 * n)
        for k_index in range(0, number_k_points):
            sum_tmp = 0
            for alpha in [0, 1]:
                prefactor = U[alpha] * bloch_absolute[k_index][alpha][n]
                integral = 0
                for m in [0, 1]:
                    for k_prime_index in range(0, number_k_points):
                        integral += 1 / np.sqrt(
                            (energies[k_prime_index][m] - mu) ** 2
                            + np.abs(delta_k[k_prime_index+offset]) ** 2
                        ) * bloch_absolute[k_prime_index][alpha][m] * delta_k[k_prime_index+offset] * np.tanh( 0.5 * beta * np.sqrt((energies[k_prime_index][m] - mu) ** 2 + np.abs(delta_k[k_prime_index+offset]) ** 2))
                sum_tmp += prefactor * integral / (number_k_points * 2.5980762113533156)

            return_vector[k_index + offset] = sum_tmp

    return_vector -= delta_k

    #filling = 4 * 2 * integral

    #return_vector[-1] = filling - filling_in

    return return_vector
"""


def generate_k_space_grid(nx, nrows, corner_1, corner_2):
    k_points = np.concatenate(
        [
            np.linspace(
                i / (nrows - 1) * corner_2,
                corner_1 + i / (nrows - 1) * corner_2,
                num=nx,
            )
            for i in range(nrows)
        ]
    )

    return k_points


def solve_gap_equation(config: Configuration, k_points: npt.NDArray) -> DeltaVector:
    energies, bloch_absolute = generate_bloch(k_points, config)

    # print(energies)

    delta_vector = DeltaVector(k_points=k_points, initial=0.1)
    # mu_initial = 1

    try:
        solution = optimize.fixed_point(
            gap_equation_real,
            delta_vector.as_1d_vector,
            args=(config.U, config.beta, bloch_absolute, energies, config.mu),
            # xtol=1e-10
        )
    except RuntimeError:
        print("Failed")
        solution = DeltaVector(k_points=k_points, initial=0).as_1d_vector
    # solution = optimize.fixed_point(
    #    gap_equation_real,
    #    1,
    #    args=(config.U, config.beta, bloch_absolute, energies, config.mu)
    # )

    # solution = optimize.root_scalar(
    #    f=gap_equation_real,
    #    x0=0.01,
    #    args=(config.U, config.beta, bloch_absolute, energies, config.mu),
    #    method="newton",
    #    # rtol=1e-9,
    # )

    """
    solution = optimize.root(
        gap_equation_real,
        delta_vector.as_1d_vector,
        args=(config.U, config.beta, bloch_absolute, energies, config.mu),
        tol=1e-9
        #method="krylov",
    )
    """
    # delta_vector.update_from_1d_vector(solution.x[:-1])
    delta_vector.update_from_1d_vector(solution)
    # if solution.success:
    #    print("Success")
    #    delta_vector.update_from_1d_vector(solution.x)
    # else:
    #    delta_vector = DeltaVector(k_points=k_points, initial=0)

    """
    bog_energies = []

    #deltas = delta_vector.data[['delta_1', 'delta_2']].to_numpy().T
    #for index, (band, delta) in enumerate(zip(energies.T, deltas)):
    for index, (band, delta) in enumerate(zip(energies.T, [solution, solution])):
        bog_energies.append(np.sqrt(band**2 + np.abs(delta) ** 2))
        bog_energies.append(-np.sqrt(band ** 2 + np.abs(delta) ** 2))

    free_energy_with_gap = -1/config.beta * special.logsumexp(-config.beta * np.array(bog_energies).flatten())

    bog_energies = []
    #for index, (band, delta) in enumerate(zip(energies, deltas)):
    for index, (band, delta) in enumerate(zip(energies.T, [solution, solution])):
        bog_energies.append(np.sqrt(band**2 + np.abs(0) ** 2))
        bog_energies.append(-np.sqrt(band ** 2 + np.abs(0) ** 2))

    free_energy = -1/config.beta * special.logsumexp(-config.beta * np.array(bog_energies).flatten())
    """

    """
    free_energy_with_gap = 0
    for index, (band, delta) in enumerate(zip(energies.T, [solution, solution])):
        for k in range(len(band)):
            free_energy_with_gap += band[k] - 2 * np.abs(delta)**2 / config.U_Gr

    bog_energies = []
    for index, (band, delta) in enumerate(zip(energies.T, [solution, solution])):
        bog_energies.append(np.sqrt(band**2 + np.abs(delta) ** 2))
        bog_energies.append(-np.sqrt(band ** 2 + np.abs(delta) ** 2))
    free_energy_with_gap -= 1/config.beta * special.logsumexp(-config.beta * np.array(bog_energies).flatten())

    if free_energy_with_gap < free_energy:
        print("SC order is stable")
    """
    return delta_vector

    # return delta_vector, solution.x[-1]
    # return delta_vector
    # if solution.converged:
    #    return solution.root
    # else:
    #    return 0


def interpolate_gap(
    delta_vector_on_grid: DeltaVector, bandpath: npt.NDArray
) -> DeltaVector:
    delta_vector_interpolated = DeltaVector(k_points=bandpath)

    for band in [1, 2]:
        delta_vector_interpolated.data.loc[:, f"delta_{band}"] = interpolate.griddata(
            delta_vector_on_grid.k_points,
            delta_vector_on_grid.data.loc[:, f"delta_{band}"],
            bandpath,
            method="cubic",
        )

    return delta_vector_interpolated


def calculate_bandstructure(config: Configuration, k_point_list):
    results = pd.DataFrame(
        columns=["band_1", "wx_1", "band_2", "wx_2"],
        index=range(len(k_point_list)),
        dtype=float,
    )

    for i, k in enumerate(k_point_list):
        h = generate_non_interacting_hamiltonian(k=k, config=config)

        energies, eigenvectors = np.linalg.eigh(h)
        energies = energies - config.mu

        results.at[i, "band_1"] = energies[0]
        results.at[i, "wx_1"] = (
            np.abs(np.dot(eigenvectors[:, 0], np.array([0, 0]))) ** 2
            - np.abs(np.dot(eigenvectors[:, 0], np.array([1, 0]))) ** 2
        )
        results.at[i, "band_2"] = energies[1]
        results.at[i, "wx_2"] = (
            np.abs(np.dot(eigenvectors[:, 1], np.array([0, 0]))) ** 2
            - np.abs(np.dot(eigenvectors[:, 1], np.array([1, 0]))) ** 2
        )

    return results
