from dist_s1.workflows import run_sequential_confirmation_of_dist_products_workflow


with open("s3_urls.txt", "r") as f:
    s3_paths = f.readlines()
    s3_paths = [p.strip() for p in s3_paths]


run_sequential_confirmation_of_dist_products_workflow(s3_paths[:3], "test_s3_confirmation")
