import click

from quant_met.bcs import find_fixpoint


@click.command()
def cli():
    find_fixpoint.solve_gap_equation()


def hello_world():
    return "Hello World!"


if __name__ == "__main__":
    cli()
