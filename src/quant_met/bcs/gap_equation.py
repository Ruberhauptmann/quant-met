import numpy as np
import numpy.typing as npt

"""
def gap_equation_complex(delta_k, U, beta):
    return_vector = np.zeros(len(delta_k))

    number_k_points = int(len(return_vector) / (2 * 3))

    for n in [0, 1, 2]:
        offset_re = int(len(delta_k) / 6 * n)
        offset_im = int(len(delta_k) / 6 * n + 0.5 * len(delta_k))
        for k_index in range(0, number_k_points):
            # for k_index, k, w in enumerate(MK_grid.iter(ret_weight=True)):
            sum_tmp = 0
            delta_k_complex = delta_k[k_index + offset_re] + 1j * delta_k[k_index + offset_im]
            for alpha in [0, 1, 2]:
                # prefactor = U[alpha] * np.conj(bloch[k_index][alpha][n])
                prefactor = U[alpha] * bloch_absolute[k_index][alpha][n]
                integral = 0
                for m in [0, 1, 2]:
                    for k_prime_index in range(0, number_k_points):
                        # integral += np.conj(bloch[k_prime_index][alpha][m]) * bloch[k_prime_index][alpha][m] * delta_k_complex * np.tanh(0.5 * beta * np.sqrt(energies[k_prime_index][m]**2 + np.abs(delta_k_complex)**2)) * k_area
                        # integral += bloch_absolute[k_prime_index][alpha][m] * delta_k_complex * np.tanh(0.5 * beta * np.sqrt(energies[k_prime_index][m]**2 + np.abs(delta_k_complex)**2)) * weights[k_prime_index]
                        integral += bloch_absolute[k_prime_index][alpha][m] * delta_k_complex * np.tanh(
                            0.5 * beta * np.sqrt(
                                energies[k_prime_index][m] ** 2 + np.abs(delta_k_complex) ** 2)) * k_area
                sum_tmp += prefactor * integral

            return_vector[k_index + offset_re] = np.real(sum_tmp)
            return_vector[k_index + offset_im] = np.imag(sum_tmp)

    # print(return_vector)
    # print(delta_k - return_vector)

    return return_vector
"""


def gap_equation_real(
    delta_k: npt.NDArray,
    U: npt.NDArray,
    beta: float,
    bloch_absolute: npt.NDArray,
    energies: npt.NDArray,
    mu: float,
):
    return_vector = np.zeros(len(delta_k))

    number_k_points = int(len(return_vector) / 3)

    for n in [0, 1, 2]:
        offset = int(len(delta_k) / 3 * n)
        for k_index in range(0, number_k_points):
            sum_tmp = 0
            for alpha in [0, 1, 2]:
                prefactor = U[alpha] * bloch_absolute[k_index][alpha][n]
                integral = 0
                for m in [0, 1, 2]:
                    for k_prime_index in range(0, number_k_points):
                        integral += (
                            0.5
                            * 1
                            / np.sqrt(
                                (energies[k_prime_index][m] - mu) ** 2
                                + np.abs(delta_k[k_prime_index]) ** 2
                            )
                            * bloch_absolute[k_prime_index][alpha][m]
                            * delta_k[k_prime_index]
                            * 1
                            # * np.tanh(
                            #    0.5
                            #    * beta
                            #    * np.sqrt(
                            #        energies[k_prime_index][m] ** 2
                            #        + np.abs(delta_k[k_prime_index]) ** 2
                            #    )
                            # )
                        )
                sum_tmp += prefactor * integral / number_k_points

            return_vector[k_index + offset] = sum_tmp

    return return_vector
