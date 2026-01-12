from pathlib import Path

import papermill as pm
import pandas as pd
from tqdm import tqdm

# Hard Sites (no change in validation but in DIST-S1)
# site_ids = pd.read_csv('reference_tables/nochange_ALLsub_conf.csv')[['ID'].tolist()

# Manual
site_ids = ["40284_2", "913366_4", "97785_11", "372152_4"]

out_nb_dir = Path("out_nbs")
out_nb_dir.mkdir(parents=True, exist_ok=True)

for site_id in tqdm(site_ids):
    out_nb_path = out_nb_dir / f"{site_id}.ipynb"
    pm.execute_notebook(
        "1_viz_dist_only.ipynb",
        out_nb_path,
        parameters=dict(SITE_ID=site_id),
    )
