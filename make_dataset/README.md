# Dataset Creation

This `make_dataset` is a package that contains all code required to transform raw data exported from `data_query` into a labeled dataset that we can use for machine learning and evaluations.

Scripts in the `scripts` directory should be executed using the poetry environment specific to the `make_dataset` module.
`make_dataset/scripts/create_dataset.py` is the script that transforms all raw data into the final datasets and data splits
and is dependent on the data cleaning and transformation logic in `make_dataset/make_dataset`.