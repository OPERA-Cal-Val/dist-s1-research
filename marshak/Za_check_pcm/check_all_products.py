import papermill as pm
from pathlib import Path
from tqdm import tqdm

with open("dist-s1_tile-ids_for_charlie_2026-01-21.txt") as f:
    prod_ids = f.read().splitlines()
prod_ids = list(map(lambda x: x.strip(), prod_ids))
prod_ids = list(filter(lambda x: x and ("missing" not in x), prod_ids))

mgrs_tile_ids = list(set([p.split("_")[3][1:] for p in prod_ids]))


out_nb = Path("out/out_nb")
out_nb.mkdir(exist_ok=True, parents=True)

for mgrs_tile_id in tqdm(mgrs_tile_ids):
    print(f"Checking {mgrs_tile_id}")
    pm.execute_notebook(
        "1_Validating_inputs_of_many-final.ipynb",
        out_nb / f"{mgrs_tile_id}.ipynb",
        parameters={"MGRS_TILE_ID": mgrs_tile_id},
    )
