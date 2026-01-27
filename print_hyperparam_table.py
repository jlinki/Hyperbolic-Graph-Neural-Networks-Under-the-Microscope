import os
import yaml
import pandas as pd
import argparse

# -----------------------------
# User configuration
# -----------------------------
BASE_DIR = "configs"       # root directory
# -----------------------------


def load_yaml(path):
    """Load a YAML file safely."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_param(task, dataset, model, param_key):
    """Extract a specific parameter value from a config YAML."""
    yaml_path = os.path.join(BASE_DIR, task, dataset, f"{model}.yaml")
    data = load_yaml(yaml_path)['parameters']
    if data is None:
        return "FILE MISSING"
    return data.get(param_key, "KEY MISSING")


def build_table(key, task, datasets, models):
    """Create a table (DataFrame) of parameter values."""
    rows = []
    for model in models:
        row = {"model": model}
        for dataset in datasets:
            value = extract_param(task, dataset, model, key)
            row[dataset] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GCN training or sweep.")
    parser.add_argument('--param', type=str, default='model.hidden_dim', help='Parameter key to extract from YAML files')
    parser.add_argument('--task', type=str, default='lp', help='Task type (e.g., nc (node classification), lp (link prediction))')
    args = parser.parse_args()

    if args.task == "lp":
        datasets = ["disease_lp", "airport", "cora", "citeseer", "pubmed"]
    else:
        datasets = ["disease_nc", "airport", "cora", "citeseer", "pubmed"]
    models = ["mlp", "gcn", "gat", "hybonet-lorentz", "hybonet-euclidean", "hgcn"]
    df = build_table(args.param, args.task, datasets, models)
    print(df.to_string(index=False))
