import matplotlib.pyplot as plt


def plot_into_BZ(all_K_points, k_points):
    fig, ax = plt.subplots()

    ax.scatter(*zip(*k_points))
    ax.scatter(*zip(*all_K_points), alpha=0.6)

    ax.set_aspect("equal", adjustable="box")

    fig.show()


def scatter_into_BZ(all_K_points, k_points, data):
    fig, ax = plt.subplots()

    scatter = ax.scatter(*zip(*k_points), c=data, cmap="viridis")
    ax.scatter(*zip(*all_K_points), alpha=0.8)
    fig.colorbar(scatter, ax=ax)

    ax.set_aspect("equal", adjustable="box")

    fig.show()
