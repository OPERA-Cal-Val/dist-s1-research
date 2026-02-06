from dist_s1.workflows import run_sequential_confirmation_of_dist_products_workflow
from multiprocessing import Pool
from tqdm import tqdm


def sequential_confirmation_of_dist_products_workflow_wrapper(input_data: tuple[str, list[str]]):
    return run_sequential_confirmation_of_dist_products_workflow(
        dist_s1_data=input_data[1], dst_dist_product_parent=input_data[0], tqdm_enabled=False
    )


def run_confirmation_wrapper_parallel(input_data_list: list[tuple[str, list[str]]], n_workers: int = 5):
    with Pool(n_workers) as pool:
        _ = list(
            tqdm(
                pool.imap(sequential_confirmation_of_dist_products_workflow_wrapper, input_data_list),
                total=len(input_data_list),
            )
        )
