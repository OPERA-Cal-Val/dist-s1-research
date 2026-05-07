import papermill as pm
from pathlib import Path
from tqdm import tqdm


out_nb_dir = Path("out_nbs")
out_nb_dir.mkdir(parents=True, exist_ok=True)


event_ids = [
    "attica_fire_2024",
    "chile_fire_2024",
    "los_angeles_fires_2025",
    "papau_new_guinea_landslide_2024",
    "chilcotin_river_landslide_and_flood_2024",
    "hokkaido_landslides_2018",
    "tuscany_flood_2023",
    "afghanistan_flood_2024",
    "tlacotalpan_flood_2024",
]

for event_id in tqdm(event_ids):
    out_nb_path = out_nb_dir / f"accuracy_{event_id}.ipynb"
    pm.execute_notebook(
        "metric_accuracy.ipynb",
        out_nb_path,
        parameters=dict(EVENT_ID=event_id),
    )
