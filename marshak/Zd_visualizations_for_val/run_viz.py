import os
from pathlib import Path

import papermill as pm
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
dist_s1_data_dir = Path(os.getenv("DIST_S1_DATA_DIR"))


### Sites #########################################################
# Hard Sites (no change in validation but in DIST-S1)
# site_ids = pd.read_csv('reference_tables/nochange_ALLsub_conf.csv')[['ID'].tolist()

# Manual
# site_ids = ["40284_2", "913366_4", "97785_11", "372152_4"]

# All Sites
site_ids = pd.read_csv("reference_tables/selectedpointsLL.csv")["ID"].tolist()

out_nb_dir = Path("out_nbs")
out_nb_dir.mkdir(parents=True, exist_ok=True)

out_dir = Path("out_all")
out_dir.mkdir(parents=True, exist_ok=True)

for site_id in tqdm(site_ids):
    out_nb_path = out_nb_dir / f"{site_id}.ipynb"
    pm.execute_notebook(
        "0_viz_it_all.ipynb",
        out_nb_path,
        parameters=dict(SITE_ID=site_id, DIST_S1_DATA_DIR=str(dist_s1_data_dir), out_dir=str(out_dir)),
    )
