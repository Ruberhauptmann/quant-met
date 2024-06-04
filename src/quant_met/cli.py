import click

from quant_met import minimize_free_energy


@click.command()
def cli():
    minimize_free_energy.minimize_loop()


if __name__ == "__main__":
    cli()
