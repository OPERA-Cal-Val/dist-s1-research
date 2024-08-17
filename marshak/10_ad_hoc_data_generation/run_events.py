from pathlib import Path

import click
import papermill as pm
from tqdm import tqdm


@click.option(
    "--event_name",
    required=True,
    type=str,
    multiple=True,
    help="Provide names of events",
)
@click.option(
    "--start_step",
    required=False,
    type=int,
    default=1,
    help="step to start event on - do not use with more than one event - see notebooks 1 - 4",
)
@click.option(
    "--stop_step",
    required=False,
    type=int,
    default=4,
    help="step to start event on - do not use with more than one event - see notebooks 1 to 4",
)
@click.command()
def main(event_name: list | tuple, start_step: int, stop_step: int):
    event_names = list(event_name)

    in_nbs = [
        "1__DIST-HLS.ipynb",
        "2__RTC-S1.ipynb",
        "3__Validation_Data.ipynb",
        "4__Water_Mask.ipynb",
    ]

    for step in [start_step, stop_step]:
        assert step in list(range(1, 5)), 'start and stop must be 1, 2, 3, 4'

    in_nbs = in_nbs[start_step - 1: stop_step]

    ipynb_out_dir = Path("out_notebooks")
    ipynb_out_dir.mkdir(exist_ok=True, parents=True)

    for event_name in tqdm(event_names, desc="events"):
        print(event_name)
        out_site_nb_dir = ipynb_out_dir / event_name
        out_site_nb_dir.mkdir(exist_ok=True, parents=True)
        for in_nb in in_nbs:
            print(in_nb)
            pm.execute_notebook(
                in_nb,
                output_path=out_site_nb_dir / in_nb,
                parameters=dict(EVENT_NAME=event_name),
            )


if __name__ == "__main__":
    main()
