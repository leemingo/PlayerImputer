# Standard library imports
import os
import json
import argparse
import pickle
from pathlib import Path
import ast
# Third-party library imports
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

# Local application imports
from datatools.preprocess import extract_match_id
from imputer.datasets import ImputerDataset
from imputer import phase2
from imputer.components import press
from imputer.components.model import SetTransformerModel, AgentImputer, XGBoostModel

# Set base path for PlayerImputer
base_path = os.path.abspath(os.path.join(os.getcwd()))  # PlayerImputer

component_dict = {
    "heatmap": press.HeatmapComponent,
    "transformer": press.SetTransformerComponent,
    "agentimputer": press.AgentImputerComponent,
    "xgboost": press.XGBoostComponent
}

model_dict = {
    "xgboost": XGBoostModel,
    "transformer": SetTransformerModel,
    "agentimputer": AgentImputer
}

def prepare_datasets(split, split_game_ids, args, data_path, version, transform=None):
    """
    Prepare dataset and save in structured folders:
    model/window/version_X/{train, valid, test}/dataset.pkl
    Also save info.txt with game IDs and yfns for reproducibility.
    
    Args:
        split (str): "train", "valid", or "test"
        split_game_ids (list): list of game IDs for this split
        args: command line arguments with model/window/yfns etc.
        base_data_path (str): root path where model/window folders exist
        version (int): version number for this split/task
        transform: optional transform function
    """

    version_path = os.path.join(base_path, data_path, args.model, f"window{args.window}", f"version_{version:03d}")
    os.makedirs(version_path, exist_ok=True)
    dataset_path = os.path.join(version_path, f"{split}_dataset.pkl")

    info_path = os.path.join(version_path, f"{split}_info.txt")

    # Create dataset if not exists
    if not os.path.exists(dataset_path):
        dataset = ImputerDataset(
            game_ids=split_game_ids,
            data_dir=args.data_dir,
            window=args.window,
            model=args.model,
            xfns=args.xfns,
            yfns=args.yfns,
            play_left_to_right=args.play_left_to_right,
            use_transform=args.use_transform,
            transform=transform
        )
        print(f"{split} dataset size: {len(dataset)}")
        
        # Save dataset
        with open(dataset_path, "wb") as f:
            pickle.dump(dataset, f)
        print(f"Saved {split} dataset to {dataset_path}")
        
        # Save info.txt
        with open(info_path, "w") as f:
            f.write(f"Game IDs: {split_game_ids}\n")
            f.write(f"yfns: {args.yfns}\n")
            f.write(f"xfns: {args.xfns}\n")
            f.write(f"window: {args.window}\n")
            f.write(f"play_left_to_right: {args.play_left_to_right}\n")
            f.write(f"use_transform: {args.use_transform}\n")
    else:
        print(f"Load {split} dataset from {dataset_path}")
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
    
    return dataset

def test(args, params, save_path, data_path, version):
    """Run testing for the specified model."""
    version_path = os.path.join(base_path, data_path, args.model, f"window{args.window}", f"version_{version:03d}")
    dataset_path = os.path.join(version_path, f"test_dataset.pkl")

    print(f"Load test dataset from {dataset_path}")
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    if "freeze_frame" not in args.xfns:
        data_dir = os.path.join(base_path, args.data_dir, args.model, f'window{args.window}', f'version_{args.data_version:03d}', 'test_info.txt')

        with open(data_dir, "r") as f:
            for line in f:
                if line.startswith("Game IDs:"):
                    # "Game IDs: " 뒷부분만 추출
                    list_str = line.split(":", 1)[1].strip()
                    # 문자열을 리스트로 변환 (ast.literal_eval 안전하게 사용)
                    test_game_ids = ast.literal_eval(list_str)

        freeze_frame_test_dataset = ImputerDataset(
            game_ids=test_game_ids,
            data_dir=args.data_dir,
            window=args.window,
            model=args.model,
            xfns=args.xfns + ["freeze_frame"],
            yfns=args.yfns,
            play_left_to_right=args.play_left_to_right,
            use_transform=args.use_transform,
            transform=dataset.transform
        )

    component_cls = component_dict[args.model]
    model_cls = model_dict[args.model]

    component = component_cls(
        model=model_cls(dataset, params.get("ModelConfig"), params.get("OptimizerConfig")),
        params=params
    )

    checkpoint_path = os.path.join(save_path, "best_model.ckpt")
    component.load(checkpoint_path)

    print(f"{'='*40}\nTest Metrics\n{'='*40}")
    metrics = component.test(dataset)
    print(metrics)
    
    print(f"{'='*40}\nPredict Metrics\n{'='*40}")
    component.predict(dataset)

    print(f"{'='*40}\Predict Metrics in freeze frame\n{'='*40}")
    if "freeze_frame" not in args.xfns:
        component.predict_freeze_frame1(dataset, freeze_frame_mask_lst=freeze_frame_test_dataset.freeze_frame_mask_lst)
    else:
        component.predict_freeze_frame1(dataset)

def train(args, params, game_ids, data_path, version):
    """Run training for the specified model."""

    # 변수 1개(fold_index)로 fold하는 과정
    fold_index = args.fold_index
    fold_index = fold_index % len(game_ids) # 리스트 길이를 넘어서도 순환하기 위해 사용
    rotated_ids = game_ids[fold_index:] + game_ids[:fold_index]

    splits = {
        "train": rotated_ids[:10],
        "valid": rotated_ids[10:12],
        "test": rotated_ids[12:13]
    }
    
    print("\nGame ID splits:", fold_index)
    for split_name, ids in splits.items():
        print(f"{split_name}: {ids}")

    mlflow.set_experiment(f"{args.model}_experiments")
    with mlflow.start_run(run_name=f"trial_{args.trial}"):
        # Prepare datasets
        train_dataset = prepare_datasets("train", splits["train"], args, data_path, version, transform=None)
        valid_dataset = prepare_datasets("valid", splits["valid"], args, data_path, version, transform=train_dataset.transform)
        test_dataset  = prepare_datasets("test", splits["test"], args, data_path, version, transform=train_dataset.transform)

        # Initialize component
        component_cls = component_dict[args.model]
        model_cls = model_dict[args.model]

        component = component_cls(
            model=model_cls(train_dataset, params.get("ModelConfig"), params.get("OptimizerConfig")),
            params=params
        )

        # Train component
        component.train(train_dataset, valid_dataset)

        # Evaluate and log metrics
        for name, dataset in [("train", train_dataset), ("valid", valid_dataset), ("test", test_dataset)]:
            print(f"{'='*40}\n{name.capitalize()} Metrics\n{'='*40}")
            metrics = component.test(dataset)
            print(metrics)
            for key, value in metrics.items():
                mlflow.log_metric(f"{name}_{key}", value)

        # Save model and component
        mlflow.pytorch.log_model(component.model, "model")

    return component

PHASE2_STATS_FILE = "phase2_stats.json"


def _phase2_loader(dataset, params, shuffle):
    return DataLoader(dataset, shuffle=shuffle, collate_fn=phase2.collate_phase2, **params["DataConfig"])


def _phase2_trainer(args, params):
    """Same callbacks as exPressPytorchComponent, but devices come from --devices (the component hardcodes devices=[3])."""
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=params["save_path"], filename="best_model", **params["ModelCheckpoint"])
    early_stop_callback = EarlyStopping(monitor="val_loss", **params["EarlyStopConfig"])
    return pl.Trainer(callbacks=[checkpoint_callback, early_stop_callback], enable_progress_bar=True, logger=True,
                      devices=args.devices, **params["TrainerConfig"])


def train_phase2(args, params, match_ids=None):
    """Train SetTransformerModel (dim_output 6) on the Phase 2 dataset. match_ids: optional {"train": [...], "valid": [...], "test": [...]} (default: splits.csv)."""
    root = args.data_dir or phase2.PHASE2_ROOT
    match_ids = match_ids or {}
    # normalisation statistics are fitted on the train matches only (fit_stats refuses any non-train match) and reused for valid / test
    train_dataset = phase2.Phase2Windows("train", args.window, match_ids=match_ids.get("train"), root=root)
    stats = train_dataset.stats
    stats_path = os.path.join(params["save_path"], PHASE2_STATS_FILE)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=1)
    print(f"Saved normalisation statistics (fit on {len(stats['fit_matches'])} train matches, window {stats['window']}) to {stats_path}")
    valid_dataset = phase2.Phase2Windows("valid", args.window, stats=stats, match_ids=match_ids.get("valid"), root=root)
    test_dataset = phase2.Phase2Windows("test", args.window, stats=stats, match_ids=match_ids.get("test"), root=root)
    print(f"samples: train {len(train_dataset)}, valid {len(valid_dataset)}, test {len(test_dataset)}")

    model = SetTransformerModel(train_dataset, params.get("ModelConfig"), params.get("OptimizerConfig"))
    trainer = _phase2_trainer(args, params)
    mlflow.set_experiment(f"{args.model}_experiments")
    with mlflow.start_run(run_name=f"trial_{args.trial}"):
        trainer.fit(model, _phase2_loader(train_dataset, params, True), _phase2_loader(valid_dataset, params, False))
        for name, dataset in (("valid", valid_dataset), ("test", test_dataset)):
            print(f"{'='*40}\n{name.capitalize()} Metrics\n{'='*40}")
            print(trainer.test(model, _phase2_loader(dataset, params, False), ckpt_path="best"))
    return model


def test_phase2(args, params):
    """Evaluate best_model.ckpt of a finished run on the test matches with the saved train statistics."""
    root = args.data_dir or phase2.PHASE2_ROOT
    with open(os.path.join(params["save_path"], PHASE2_STATS_FILE)) as f:
        stats = json.load(f)
    test_dataset = phase2.Phase2Windows("test", args.window, stats=stats, root=root)  # raises if args.window != stats["window"]
    model = SetTransformerModel(test_dataset, params.get("ModelConfig"), params.get("OptimizerConfig"))
    trainer = _phase2_trainer(args, params)
    print(trainer.test(model, _phase2_loader(test_dataset, params, False), ckpt_path=os.path.join(params["save_path"], "best_model.ckpt")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--dataset", type=str, choices=["imputer", "phase2"], default="imputer",
                        help="imputer: ImputerDataset (DFL/BEPRO csv, default); phase2: Phase 2 parquet windows (dim_output 6, data_dir defaults to the Phase 2 root)")
    parser.add_argument("--devices", type=int, default=1, help="phase2 path only: number of devices for the Lightning Trainer")
    parser.add_argument("--data_version", type=int, required=False)
    parser.add_argument("--fold_index", type=int, default=5, help="Rotation offset of the game id list that defines the train/valid/test split")
    parser.add_argument("--params_file", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=False)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--xfns", type=str, nargs="+", required=False)
    parser.add_argument("--yfns", type=str, nargs="+", required=False)
    parser.add_argument("--use_transform", action='store_true', default=True, help="Whether to apply transformation to the dataset")
    parser.add_argument("--play_left_to_right", action='store_true', default=False, help="Whether to play left to right")

    args = parser.parse_args()
    if args.dataset == "imputer":
        missing = [n for n in ("data_version", "data_dir", "xfns", "yfns") if getattr(args, n) is None]
        if missing:
            parser.error("the following arguments are required: " + ", ".join("--" + n for n in missing))
    elif args.model != "transformer":
        parser.error("--dataset phase2 supports --model transformer only")

    model_path = os.path.join(base_path, "stores", args.model)
    os.makedirs(model_path, exist_ok=True)
    
    save_path = os.path.join(model_path, f"{args.trial:03d}")
    os.makedirs(save_path, exist_ok=True)

    # Load parameters from JSON file
    with open(args.params_file, "r") as f:
        all_params = json.load(f)
    params = all_params.get(args.model, {})  # get model's parameter
    params["save_path"] = save_path
    
    if args.dataset == "phase2":
        if args.mode == "train":
            with open(os.path.join(save_path, "params.json"), "w") as f:
                json.dump(all_params[args.model], f)
            with open(os.path.join(save_path, "args.json"), "w") as f:
                json.dump(vars(args), f)
            train_phase2(args, params)
        else:
            test_phase2(args, params)
    elif args.mode=="train":
        # Save all parameters to JSON file
        with open(os.path.join(save_path, "params.json"), "w") as f:
            json.dump(all_params[args.model], f)

        # Save arguments to JSON file
        with open(os.path.join(save_path, "args.json"), "w") as f:
            json.dump(vars(args), f)

        mlflow.set_experiment(f"{args.model}_experiments")

        # Check if the data directory is valid and extract game IDs
        if "DFL" in args.data_dir:
            game_ids = sorted([extract_match_id(filename) for filename in os.listdir(args.data_dir) if filename.startswith("DFL")])
        elif "BEPRO" in args.data_dir:
            game_ids = sorted([extract_match_id(filename) for filename in os.listdir(args.data_dir) if filename.startswith("1")])
        else:
            raise ValueError("Unsupported data directory. Please use 'DFL' or 'BEPRO' data directories.")
        print(f"\nFound {len(game_ids)} games in {args.data_dir}: {game_ids}")

        component = train(args, params, game_ids, args.data_dir, args.data_version)
        component.save(Path(os.path.join(save_path, "component.pkl")))
    elif args.mode == "test":
        test(args, params, save_path, args.data_dir, args.data_version)
    else:
        raise ValueError("Invalid mode: Choose 'train' or 'test'.")