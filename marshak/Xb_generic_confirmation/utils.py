import zipfile
from pathlib import Path
from dist_s1 import run_sequential_confirmation_of_dist_products_workflow


def unzip_file(zip_path, extract_to):
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    
    subdirectory_name = zip_path.stem
    
    full_extract_path = extract_to / subdirectory_name
    full_extract_path.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(full_extract_path)
    
    return full_extract_path


def unzip_dist_s1_prod(zip_path: Path, unconfirmed_products_dir=None, use_zip_parent=True):
    mgrs_tile_id = zip_path.name.split('_')[3][1:]    

    if unconfirmed_products_dir is None:
        unconfirmed_products_dir = Path(f'unconfirmed_products')
    unconfirmed_products_dir.mkdir(parents=True, exist_ok=True)
    if use_zip_parent:
        dst_dir = Path(f'{unconfirmed_products_dir}/{zip_path.parent.name}/{mgrs_tile_id}')
    else:
        dst_dir = Path(f'{unconfirmed_products_dir}/{mgrs_tile_id}')
    dst_dir.mkdir(exist_ok=True, parents=True)
    unzip_file(zip_path, dst_dir)
    return zip_path.name

def wrap_run_sequential_confirmation_of_dist_products_workflow(unconfirmed_product_ts_dir, confirmed_products_dir=None, unconfirmed_products_dir=None, confirm_kwargs: dict | None = None):
    if unconfirmed_products_dir is None:
        unconfirmed_products_dir = Path('unconfirmed_products')
    unconfirmed_products_dir.mkdir(parents=True, exist_ok=True)
    
    if confirmed_products_dir is None:
        confirmed_products_dir = Path('confirmed_products')
    confirmed_products_dir.mkdir(parents=True, exist_ok=True)

    confirm_kwargs = confirm_kwargs or {}
    target_dir = confirmed_products_dir / unconfirmed_product_ts_dir.relative_to(unconfirmed_products_dir)
    run_sequential_confirmation_of_dist_products_workflow(
        unconfirmed_product_ts_dir, 
        target_dir,
        **confirm_kwargs
    )