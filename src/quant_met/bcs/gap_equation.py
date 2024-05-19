import numpy as np
import numpy.typing as npt


def gap_equation_real(
    delta_k: npt.NDArray,
    U: npt.NDArray,
    beta: float,
    bloch_absolute: npt.NDArray,
    energies: npt.NDArray,
    number_k_points: int,
):
    return_vector = np.zeros(len(delta_k))

    number_bands = int(len(return_vector) / number_k_points)

    for n in range(number_bands):
        offset_n = int(len(delta_k) / 2 * n)
        for k_prime_index in range(0, number_k_points):
            sum_tmp = 0
            for alpha in range(number_bands):
                for m in range(number_bands):
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

            return_vector[k_prime_index + offset_n] = sum_tmp / (
                2.5980762113533156 * number_k_points
            )

    return return_vector
