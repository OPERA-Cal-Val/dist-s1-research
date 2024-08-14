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
@click.command()
def main(event_name: list):
    event_names = list(event_name)

    in_nbs = [
        "1__DIST-HLS.ipynb",
        "2__RTC-S1.ipynb",
        "3__Validation_Data.ipynb",
        "4__Water_Mask.ipynb",
    ]

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
