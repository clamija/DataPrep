from copy import deepcopy

GLOBAL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "numeric_impute_strategy": "median",
    "categorical_impute_strategy": "mode_or_unknown",
    "outlier_method": "iqr_clip",
    "iqr_multiplier": 1.5,
}

COMMON_VALUE_MAPPINGS = {
    "yes": "yes",
    "y": "yes",
    "1": "yes",
    "true": "yes",
    "no": "no",
    "n": "no",
    "0": "no",
    "false": "no",
    "male": "male",
    "m": "male",
    "female": "female",
    "f": "female",
}

DATASET_CONFIGS = {
    "titanic": {
        "dataset_path": "data/titanic.csv",
        "target_column": "Survived",
        "drop_columns": ["PassengerId", "Name", "Ticket"],
        "value_mappings": COMMON_VALUE_MAPPINGS,
        "target_mappings": {"1": 1, "0": 0},
    },
    "telco": {
        "dataset_path": "data/telco.csv",
        "target_column": "Churn",
        "drop_columns": ["customerID"],
        "value_mappings": COMMON_VALUE_MAPPINGS,
        "target_mappings": {"yes": 1, "no": 0},
    },
    "custom": {
        "dataset_path": None,
        "target_column": None,
        "drop_columns": [],
        "value_mappings": COMMON_VALUE_MAPPINGS,
        "target_mappings": {"yes": 1, "no": 0, "1": 1, "0": 0},
    },
}


def get_config(dataset_name: str) -> dict:
    base = deepcopy(GLOBAL_CONFIG)
    dataset = deepcopy(DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["custom"]))
    base.update(dataset)
    return base
