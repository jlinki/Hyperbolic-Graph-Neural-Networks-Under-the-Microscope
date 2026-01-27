import yaml
import os
from data_utils import GraphDatasetLoader
from torch.optim import Adam, SGD
from copy import deepcopy
import numpy as np
import argparse
import wandb
from tqdm import tqdm
import torch
import copy

from geoopt.optim.radam import RiemannianAdam
from models.base_models import NCModel, NRModel, LPModel, PairsModel
from types import SimpleNamespace


GLOBAL_CUDA_IDX = -1


def apply_random_features(
    data,
    random_features,
    device,
    is_lp: bool,
    exp: int = 0,
):
    """
    Apply random / noisy features to data.

    Args:
        data:
            - non-LP: a single data object with .x
            - LP: list of split dicts with ['train','val','test'].x
        random_features: str or float
        device: torch device
        is_lp: whether this is the LP setting
        exp: experiment index (noise only added at exp == 0)
    """

    def generate_features(x, is_lp=is_lp):
        n, d = x.shape

        if random_features == "random":
            return torch.randn_like(x)
        elif random_features == "random10":
            return torch.randn(n, 10, device=x.device) + 10 * torch.rand(n, 10, device=x.device)
        elif random_features == "random100":
            return torch.randn(n, 100, device=x.device)
        elif random_features == "random1000":
            return torch.randn(n, 1000, device=x.device)
        elif random_features == "const":
            return torch.ones_like(x) / d
        elif random_features == "onehot_perm":
            perm = torch.randperm(n, device=x.device)
            return torch.eye(n, device=x.device)[perm]
        elif random_features == "onehot":
            return torch.eye(n, device=x.device)
        elif random_features in ["None", None]:
            return "None"
        else:
            # interpreted as sigma
            sigma = float(random_features)
            if is_lp:
                return sigma
            else:
                return x + torch.randn_like(x) * sigma

    # LP case
    if not is_lp:
        rf = generate_features(data.x)
        if rf != "None":
            data.x = rf.to(device)
        if hasattr(data, "to"):
            data = data.to(device)
        return data

    # non-LP case
    if exp != 0:
        return data

    for split_idx in range(len(data)):
        base_x = data[split_idx]["train"].x
        rf = generate_features(base_x)
        if rf in ["None", None]:
            noise_add = None
        elif isinstance(random_features, str):
            noise_add = None
        else:
            sigma = rf
            noise_add = torch.randn_like(base_x) * sigma
        for split in ["train", "val", "test"]:
            if rf not in ["None", None]:
                if isinstance(random_features, str):
                    data[split_idx][split].x = rf
                else:
                    data[split_idx][split].x = copy.copy(
                        data[split_idx][split].x + noise_add
                    )
            if hasattr(data[split_idx][split], "to"):
                data[split_idx][split] = data[split_idx][split].to(device)

    return data


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train_and_evaluate(exp, model, data, optimizer, epoch, config, use_wandb=False):
    all_metrics = {}
    for split in ['train', 'val', 'test']:
        if config["dataset.task"] in ['lp', 'pairs']:
            data_split = data[exp][split]
        else:
            data_split = data
        # compute loss
        if split == 'train':
            model.train()
            optimizer.zero_grad()
            embeddings, embeddings_layer = model.encode(data_split.x, data_split.edge_index, data_split.edge_attr)
            loss = model.compute_loss(exp, embeddings, data_split, split, mode='train')
            loss.backward()
            if config.get('training.grad_clip', None) is not None:
                if config['training.grad_clip'] != "None":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training.grad_clip'])
            optimizer.step()
        else:
            model.eval()
            embeddings, embeddings_layer = model.encode(data_split.x, data_split.edge_index, data_split.edge_attr)
            loss = model.compute_loss(exp, embeddings, data_split, split)


        #compute metrics
        model.eval()
        metrics = model.compute_metrics(exp, embeddings, data_split, split)
        metrics['loss'] = loss.item()
        all_metrics[split] = metrics
        if use_wandb:
            model.wandb_log(exp, epoch, split, all_metrics[split])

    return all_metrics


def best_model_eval(exp, model, data, epoch, config, use_wandb=False):
    model.eval()
    if config["dataset.task"] in ['lp', 'pairs']:
        best_metrics = {}

        for split in ['train', 'val', 'test']:
            data_split = data[exp][split]
            embeddings, embeddings_layers = model.encode(data_split.x, data_split.edge_index, data_split.edge_attr, return_all_layers=True)
            best_metrics[split] = model.compute_metrics(exp, embeddings, data_split, split)

            # Save embeddings only on test set
            if config["save_embeddings"] and split == "test":
                h_np_list = []
                for layer_idx, embedding in enumerate(embeddings_layers):
                    h_np = embedding.detach().cpu().numpy()
                    h_np_list.append(h_np)
                h_np = np.stack(h_np_list, axis=0)
                save_dir = "./save_embeddings"
                os.makedirs(save_dir, exist_ok=True)
                num = np.random.randint(0, 10000)
                embedding_folder_name = config.get("name", f"default_run{str(num).zfill(3)}")
                if config["dataset.task"] in ['pairs']:
                    save_path = os.path.join(
                        save_dir, config['dataset.name'], "embeddings", config["model.name"], f"dim_{config['model.hidden_dim']}",
                        f"node_embeddings_{config['dataset.name']}_{config['model.manifold']}_model_{config['model.name']}_seed_{config['training.seeds'][exp]}.npy")
                else:
                    if config.get("dataset.random_features", None) not in [None, "None"]:
                        embedding_folder_name = config["model.name"] + f"_{config['dataset.random_features']}"
                    else:
                        embedding_folder_name = config["model.name"]
                    save_path = os.path.join(
                        save_dir, config['dataset.name'], "embeddings", embedding_folder_name,
                        f"node_embeddings_{config['dataset.name']}_{config['model.manifold']}_model_{config['model.name']}_seed_{config['training.seeds'][exp]}.npy")
                if not os.path.exists(save_path):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, h_np)

            if use_wandb:
                model.wandb_log(exp, epoch, split, best_metrics[split], best_eval=True)
    else:
        data_split = data
        embeddings, embeddings_layers = model.encode(data_split.x, data_split.edge_index, data_split.edge_attr, return_all_layers=True)
        if config["save_embeddings"]:
            h_np_list = []
            for layer_idx, embedding in enumerate(embeddings_layers):
                h_np = embedding.detach().cpu().numpy()
                h_np_list.append(h_np)
            h_np = np.stack(h_np_list, axis=0)
            save_dir = "./save_embeddings"
            os.makedirs(save_dir, exist_ok=True)
            num = np.random.randint(0, 10000)
            embedding_folder_name = config.get("name", f"default_run{str(num).zfill(3)}")
            save_path = os.path.join(
                save_dir, config['dataset.name'], "embeddings", embedding_folder_name,
                f"node_embeddings_{config['dataset.name']}_{config['model.manifold']}_model_{config['model.name']}_seed_{config['training.seeds'][exp]}.npy")
            if not os.path.exists(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, h_np)

        best_metrics = {}

        for split in ['train', 'val', 'test']:
            best_metrics[split] = model.compute_metrics(exp, embeddings, data_split, split)

            if use_wandb:
                model.wandb_log(exp, epoch, split, best_metrics[split], best_eval=True)

    return best_metrics


def run_single_experiment(data, exp, config, seed, device='cpu', use_wandb=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Load dataset
    random_features = config.get('dataset.random_features', "None")


    if config['dataset.task'] in 'nc':
        data = apply_random_features(data, random_features, device, is_lp=False, exp=exp)
        Model = NCModel
        args_model = SimpleNamespace(
            in_dim=data.num_features,
            hidden_dim=config['model.hidden_dim'],
            num_layers = config.get('model.num_layers', 2) - 1,
            act=config.get('model.act', None),
            n_classes=int(torch.max(data.y) + 1),
            dropout=config['model.dropout'],
            model=config['model.name'],
            cuda=config.get('cuda', -1),
            device=device,
            manifold=config['model.manifold'],
            bias=config.get('model.bias', True),
            num_heads=config.get('model.num_heads', 1),
            concat=bool(config.get('model.concat', 1)),
            use_att = config.get('model.use_att', False),
            c=config.get('model.c', 1.0),
            task=config['dataset.task'],
            exp_map_clip=config.get('training.exp_map_clip', None),
            margin=config.get('training.margin', 2.0),
            seed=seed,
            loss = config.get('training.loss', 'ce'),
            loss_weights = config.get('training.loss_weights', False),
            local_agg = config.get('model.local_agg', False),
            c_learnable=config.get('model.c_learnable', True),
        )
    elif config['dataset.task'] in 'nr':
        data = apply_random_features(data, random_features, device, is_lp=False, exp=exp)
        Model = NRModel
        args_model = SimpleNamespace(
            in_dim=data.num_features,
            hidden_dim=config['model.hidden_dim'],
            num_layers = config.get('model.num_layers', 2) - 1,
            act=config.get('model.act', None),
            n_classes=int(torch.max(data.y) + 1),
            dropout=config['model.dropout'],
            model=config['model.name'],
            cuda=config.get('cuda', -1),
            device=device,
            manifold=config['model.manifold'],
            bias=config.get('model.bias', True),
            num_heads=config.get('model.num_heads', 1),
            concat=bool(config.get('model.concat', 1)),
            use_att = config.get('model.use_att', False),
            c=config.get('model.c', 1.0),
            task=config['dataset.task'],
            exp_map_clip=config.get('training.exp_map_clip', None),
            margin=config.get('training.margin', 2.0),
            seed=seed,
            loss = config.get('training.loss', 'ce'),
            loss_weights = config.get('training.loss_weights', False),
            local_agg = config.get('model.local_agg', False),
            c_learnable = config.get('model.c_learnable', True),
        )
    elif config['dataset.task'] == 'lp':
        Model = LPModel
        data = apply_random_features(data, random_features, device, is_lp=True, exp=exp)
        args_model = SimpleNamespace(
            in_dim=data[0]["train"].num_features,
            hidden_dim=config['model.hidden_dim'],
            num_layers = config.get('model.num_layers', 2),
            act = config.get('model.act', None),
            n_classes=None,
            dropout=config['model.dropout'],
            model=config['model.name'],
            cuda=config.get('cuda', -1),
            device=device,
            manifold=config['model.manifold'],
            bias=config['model.bias'],
            num_heads=config.get('model.num_heads', 1),
            concat=bool(config.get('model.concat', 1)),
            use_att = config.get('model.use_att', False),
            c=config['model.c'],
            margin=config.get('training.margin', 2.0),
            task=config['dataset.task'],
            loss = config.get('training.loss', 'margin'),
            dataset_name = config['dataset.name'],
            exp_map_clip=config.get('training.exp_map_clip', None),
            local_agg=config.get('model.local_agg', False),
            c_learnable=config.get('model.c_learnable', True),
        )
    elif config['dataset.task'] == 'pairs':
        Model = PairsModel
        # move Data objects to device
        for split_idx in range(len(data)):
            for split in ["train", "val", "test"]:
                data_split = data[split_idx][split]
                data[split_idx][split] = data_split.to(device)

        args_model = SimpleNamespace(
            in_dim = data[0]["train"].x.shape[1],
            hidden_dim = config['model.hidden_dim'],
            num_layers = config.get('model.num_layers', 2),
            act = config.get('model.act', None),
            dropout = config['model.dropout'],
            model = config['model.name'],
            cuda = config.get('cuda', -1),
            device = device,
            manifold = config['model.manifold'],
            bias = config['model.bias'],
            num_heads=config.get('model.num_heads', 1),
            concat=bool(config.get('model.concat', 1)),
            use_att = config.get('model.use_att', False),
            c = config['model.c'],
            task = config['dataset.task'],
            exp_map_clip=config.get('training.exp_map_clip', None),
            local_agg=config.get('model.local_agg', False),
            c_learnable = config.get('model.c_learnable', True),
        )
    else:
        raise ValueError(f"Unknown task: {config['dataset.task']}")
    model = Model(args_model).to(device)
    if exp==0:
        print(model)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {num_params}")
    # store the metric types for the current task
    which_metrics = model.which_metrics()

    if config['training.optimizer'] == 'adam':
        optimizer = Adam(model.parameters(), lr=config['training.lr'], weight_decay=float(config['training.weight_decay']))
    elif config['training.optimizer'] == 'sgd':
        optimizer = SGD(model.parameters(), lr=config['training.lr'], weight_decay=float(config['training.weight_decay']))
    elif config['training.optimizer'] == 'radam':
        optimizer = RiemannianAdam(model.parameters(), lr=config['training.lr'], weight_decay=float(config['training.weight_decay']), stabilize=10)
    else:
        raise ValueError(f"Unknown optimizer: {config['training.optimizer']}")

    best_metric = model.init_best_metric()
    patience_counter = 0
    best_epoch = 0
    try:
        best_model = deepcopy(model)
    except TypeError:
        best_model_state = deepcopy(model.state_dict())
        best_model = model.__class__(args_model).to(device)
        best_model.load_state_dict(best_model_state)

    for epoch in tqdm(range(config['training.epochs'])):
        metrics = train_and_evaluate(exp, model, data, optimizer, epoch, config, use_wandb)
        improvement, best_metric = model.has_improved(best_metric, metrics['val'])
        if improvement:
            try:
                best_model = deepcopy(model)
            except TypeError:
                best_model_state = deepcopy(model.state_dict())
                best_model = model.__class__(args_model).to(device)
                best_model.load_state_dict(best_model_state)
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch + 1 >= config['training.min_epochs'] and patience_counter >= config['training.patience']:
            break

    # Final test accuracy from best model
    best_model.eval()
    best_metrics = best_model_eval(exp, best_model, data, best_epoch, config, use_wandb)

    return best_metrics, which_metrics


def run_experiments(config=None, use_wandb=True):
    with wandb.init(config=config):
        from __main__ import GLOBAL_CUDA_IDX
        from __main__ import GLOBAL_SAVE_EMBEDDINGS
        config = wandb.config

        # Choosing device
        cuda_idx = config.get("cuda", GLOBAL_CUDA_IDX)
        config["save_embeddings"] = config.get("save_embeddings", GLOBAL_SAVE_EMBEDDINGS)
        device = f"cuda:{cuda_idx}" if cuda_idx >= 0 and torch.cuda.is_available() else "cpu"
        print(f"[W&B sweep] Using device: {device}")

        loader = GraphDatasetLoader(config['dataset.name'],
                                    task=config['dataset.task'],
                                    split=config['dataset.split'],
                                    num_val=config.get('dataset.num_val', None),
                                    num_test=config.get('dataset.num_test', None),
                                    normalize_feats=config.get('dataset.normalize_feats', False),
                                    normalize_adj=config.get('dataset.normalize_adj', False),
                                    add_self_loops=config.get('dataset.add_self_loops', False))
        data = loader.load()
        seeds = config.get('training.seeds', list(range(10)))

        results = None

        for i, seed in enumerate(seeds):
            best_metrics, which_metrics = run_single_experiment(data, i, config, seed, device=device, use_wandb=use_wandb)
            if results is None:
                results = {
                    'val': {metric: [] for metric in which_metrics},
                    'test': {metric: [] for metric in which_metrics}
                }
            print_str = f"Seed {seed}: "
            for metric in which_metrics:
                print_str += f"Val {metric}: {best_metrics['val'][metric]:.4f}; "
                results['val'][metric].append(best_metrics['val'][metric])
                results['test'][metric].append(best_metrics['test'][metric])
            print(print_str)


        avg_metrics = {
            'val': {metric: 0 for metric in which_metrics},
            'test': {metric: 0 for metric in which_metrics}
        }
        std_metrics = {
            'val': {metric: 0 for metric in which_metrics},
            'test': {metric: 0 for metric in which_metrics}
        }
        for metric in which_metrics:
            avg_metrics['val'][metric] = np.mean([value for value in results['val'][metric]])
            std_metrics['val'][metric] = np.std([value for value in results['val'][metric]])
            avg_metrics['test'][metric] = np.mean([value for value in results['test'][metric]])
            std_metrics['test'][metric] = np.std([value for value in results['test'][metric]])
            print(f"\nAvg Val {metric}: {avg_metrics['val'][metric]:.4f} ± {std_metrics['val'][metric]:.4f}")
            print(f"Avg Test {metric}: {avg_metrics['test'][metric]:.4f} ± {std_metrics['test'][metric]:.4f}")


        if use_wandb:
            for metric in which_metrics:
                wandb.run.summary.update({
                    f"avg_val_{metric}": avg_metrics['val'][metric],
                    f"std_val_{metric}": std_metrics['val'][metric],
                    f"avg_test_{metric}": avg_metrics['test'][metric],
                    f"std_test_{metric}": std_metrics['test'][metric],
                })
            wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Run GCN training or sweep.")
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset name')
    parser.add_argument('--task', type=str, default='nc', help='Task type (e.g., nc (node classification), lp (link prediction)), pairs (node pairs regression)), nr (node regression))')
    parser.add_argument('--model', type=str, default='gcn', help='Model type (e.g., gcn, hgcn)')
    parser.add_argument('--sweep', action='store_true', default=False, help='True for running a sweep config, False for normal config')
    parser.add_argument('--cuda', type=int, default=-1, help='CUDA device index (-1 for CPU)')
    parser.add_argument('--save_embeddings', action='store_true', default=False, help='Save node embeddings after training')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='Enable W&B logging')
    parser.add_argument('--wandb_entity', type=str, default='default_entity', help='W&B entity name (if not set, uses default entity)')
    parser.add_argument('--sweep_count', type=int, default=None, help='Number of runs to execute if the W&B sweep method is "random"')
    parser.add_argument('--hidden_dim', type=int, default=None, help='Hidden dimension size (overrides config if set)')
    return parser.parse_args()


def main():
    # to make sure that we get deterministic results
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    args = parse_args()
    config_filename = args.model

    # Add sweep or normal suffix
    if args.sweep:
        config_filename = f"{config_filename}_sweep.yaml"
        if not args.use_wandb:
            print("Sweep config detected but W&B is disabled. Enabling it now.")
            args.use_wandb = True
    else:
        config_filename = f"{config_filename}.yaml"

    # Get the config
    if args.task == "pairs":
        if args.hidden_dim == None:
            args.hidden_dim = 16
            print(f"You did not specify a hidden dimension for the paris task. Using default hidden_dim {args.hidden_dim}.")
        config_path = os.path.join(f"configs/{args.task}/{args.dataset}/dim{args.hidden_dim}", config_filename)
        config = load_config(config_path)
    else:    
        config_path = os.path.join(f"configs/{args.task}/{args.dataset}/", config_filename)
        config = load_config(config_path)
        if args.hidden_dim is not None:
            print(f"Overriding hidden_dim in config: {config['parameters']['model.hidden_dim']} -> {args.hidden_dim}")
            config['parameters']['model.hidden_dim'] = args.hidden_dim
    if not args.sweep:
        config['parameters']["name"] = config_filename
    project = config.get("project", "default_project")
    use_wandb = args.use_wandb

    # Detect device
    cuda_idx = args.cuda
    global GLOBAL_CUDA_IDX
    GLOBAL_CUDA_IDX = cuda_idx
    global GLOBAL_SAVE_EMBEDDINGS
    GLOBAL_SAVE_EMBEDDINGS = args.save_embeddings


    # Main logic for sweep and single run
    if args.sweep:
        wandb.login()
        # wandb entity only used by some projects, so we only use if provided
        sweep_id = wandb.sweep(config, project=project, entity=args.wandb_entity)
        # If sweep method is random, specify number of runs
        if config["method"] == "random":
            wandb.agent(sweep_id, function=run_experiments, count = args.sweep_count)
        else:
            wandb.agent(sweep_id, function=run_experiments, count = args.sweep_count)
    else:
        param_config = config.get("parameters", config)
        # CLI flag is injected into config
        param_config["save_embeddings"] = args.save_embeddings
        device = f"cuda:{cuda_idx}" if cuda_idx >= 0 and torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        if use_wandb:
            # wandb entity only used by some projects, so we only use if provided
            wandb.init(config=param_config, project=project, entity=args.wandb_entity)

        print(f"Running {len(param_config['training.seeds'])}-seed experiment on {param_config['dataset.name'].upper()}...")
        results = None
        loader = GraphDatasetLoader(param_config['dataset.name'],
                                    task=param_config['dataset.task'],
                                    split=param_config['dataset.split'],
                                    num_val=param_config.get('dataset.num_val', None),
                                    num_test=param_config.get('dataset.num_test', None),
                                    normalize_feats=param_config.get('dataset.normalize_feats', False),
                                    normalize_adj=param_config.get('dataset.normalize_adj', False),
                                    add_self_loops=param_config.get('dataset.add_self_loops', False)
                                    )
        data = loader.load()
        seeds = param_config['training.seeds']

        for i, seed in enumerate(seeds):
            best_metrics, which_metrics = run_single_experiment(data, i, param_config, seed, device=device, use_wandb=use_wandb)
            if results is None:
                results = {
                    'val': {metric: [] for metric in which_metrics},
                    'test': {metric: [] for metric in which_metrics}
                }

            print_str = f"Seed {seed}: "
            for metric in which_metrics:
                print_str += f"Val {metric}: {best_metrics['val'][metric]:.4f}; "
                results['val'][metric].append(best_metrics['val'][metric])
                results['test'][metric].append(best_metrics['test'][metric])
            print(print_str)


        avg_metrics = {
            'val': {metric: 0 for metric in which_metrics},
            'test': {metric: 0 for metric in which_metrics}
        }
        std_metrics = {
            'val': {metric: 0 for metric in which_metrics},
            'test': {metric: 0 for metric in which_metrics}
        }
        for metric in which_metrics:
            avg_metrics['val'][metric] = np.mean([value for value in results['val'][metric]])
            std_metrics['val'][metric] = np.std([value for value in results['val'][metric]])
            avg_metrics['test'][metric] = np.mean([value for value in results['test'][metric]])
            std_metrics['test'][metric] = np.std([value for value in results['test'][metric]])
            print(f"\nAvg Val {metric}: {avg_metrics['val'][metric]:.4f} ± {std_metrics['val'][metric]:.4f}")
            print(f"Avg Test {metric}: {avg_metrics['test'][metric]:.4f} ± {std_metrics['test'][metric]:.4f}")

        if use_wandb:
            for metric in which_metrics:
                wandb.run.summary.update({
                    f"avg_val_{metric}": avg_metrics['val'][metric],
                    f"std_val_{metric}": std_metrics['val'][metric],
                    f"avg_test_{metric}": avg_metrics['test'][metric],
                    f"std_test_{metric}": std_metrics['test'][metric],
                })
            wandb.finish()

if __name__ == "__main__":
    main()
