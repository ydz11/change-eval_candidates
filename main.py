import json
from pathlib import Path
from itertools import product as grid_product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import (
    load_ratings,
    filter_cold_start,
    reindex_ids,
    ratio_split,
    build_train_uir,
    build_ui,
    get_num_users_items,
)
from dataset import RatingTrainDataset, SasRecTrainDataset
from neighbor_retrieval import build_neighbor_dicts
from pretrain_sasrec import pretrain_sasrec
from mf_model import MF
from ncf_model import NCF
from sasrec_ncf import SASRecNCF
from neighbor_aware_model import NeighborAware
from evaluate import build_eval_candidates, evaluate_model


# =============================================================
# Configuration
# =============================================================

DATA_PATH = "ml-1m/ratings.dat"
OUTPUT_DIR = Path("outputs")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NEIGHBOR_K = 5

TRAIN_BATCH_SIZE = 256
TRAIN_EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 5

SASREC_MAXLEN = 50
SASREC_EPOCHS = 20
SASREC_BATCH_SIZE = 128

TOP_K = 10
NUM_NEG_EVAL = 99
RATING_THRESHOLD = 4

# Cold-start filtering thresholds
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5

# =============================================================
# GridSearch hyperparameter grid
# =============================================================
GRID = {
    "factor":         [16, 32, 64],
    "lr":             [1e-3],
    "weight_decay":   [1e-5, 1e-4],
    "dropout":        [0.2],
    "sasrec_num_neg": [1],
}


# =============================================================
# Training function
# =============================================================

def train_rating_model(model, train_dataset, valid_users, valid_candidates,
                       model_name, lr=LR, weight_decay=WEIGHT_DECAY):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    best_ndcg = -1.0
    best_state = None
    no_improve = 0

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        epoch_losses = []

        for user, item, rating in train_loader:
            user = user.to(DEVICE)
            item = item.to(DEVICE)
            rating = rating.to(DEVICE)

            pred = model(user, item)
            loss = loss_fn(pred, rating)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        hr, ndcg = evaluate_model(model, valid_users, valid_candidates, TOP_K, DEVICE)
        avg_loss = float(np.mean(epoch_losses))

        print(f"[{model_name}] epoch={epoch:02d}, "
              f"train_mse={avg_loss:.4f}, "
              f"valid_hr@{TOP_K}={hr:.4f}, "
              f"valid_ndcg@{TOP_K}={ndcg:.4f}")

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"[{model_name}] Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model


# =============================================================
# Plotting
# =============================================================

def plot_results(results, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    factors = sorted(results.keys())
    model_names = list(next(iter(results.values())).keys())

    plt.figure(figsize=(8, 5))
    for name in model_names:
        y = [results[f][name]["ndcg"] for f in factors]
        plt.plot(factors, y, marker="o", label=name)
    plt.xlabel("Factor")
    plt.ylabel(f"NDCG@{TOP_K}")
    plt.title("NDCG Comparison across Factors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "ndcg_compare.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name in model_names:
        y = [results[f][name]["hr"] for f in factors]
        plt.plot(factors, y, marker="o", label=name)
    plt.xlabel("Factor")
    plt.ylabel(f"HR@{TOP_K}")
    plt.title("HR Comparison across Factors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "hr_compare.png", dpi=200)
    plt.close()


# =============================================================
# Main
# =============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ===========================================================
    # 1. Load data
    # ===========================================================
    df = load_ratings(DATA_PATH)
    print(f"Raw data: {len(df)} interactions, "
          f"{df['user_id'].nunique()} users, {df['item_id'].nunique()} items")

    # ===========================================================
    # 2. Filter cold-start users/items BEFORE anything else
    # ===========================================================
    df = filter_cold_start(df,
                           min_user_interactions=MIN_USER_INTERACTIONS,
                           min_item_interactions=MIN_ITEM_INTERACTIONS)
    print(f"After cold-start filtering: {len(df)} interactions, "
          f"{df['user_id'].nunique()} users, {df['item_id'].nunique()} items")

    # ===========================================================
    # 3. Reindex user_id and item_id to be contiguous from 1
    # ===========================================================
    df = reindex_ids(df)

    # ===========================================================
    # 4. Get num users/items using nunique
    # ===========================================================
    n_users, n_items = get_num_users_items(df)
    print(f"After reindexing: n_users={n_users}, n_items={n_items}")

    # ===========================================================
    # 5. Split data 70:15:15 (original ratio_split, unchanged)
    # ===========================================================
    train_df, valid_df, test_df = ratio_split(df)
    print(f"Split: #train={len(train_df)}, #valid={len(valid_df)}, #test={len(test_df)}")

    # ===========================================================
    # 6. Prepare data structures
    # ===========================================================
    train_uir = build_train_uir(train_df)

    # Training dataset: use ALL observed ratings (no random neg sampling)
    train_dataset = RatingTrainDataset(train_df)

    # user_history: ordered item sequence per user from training (for SASRec)
    user_history = {}
    for user_id, group in train_df.groupby("user_id"):
        group = group.sort_values("timestamp")
        user_history[int(user_id)] = group["item_id"].astype(int).tolist()

    # ===========================================================
    # Step 1: Compute cosine-based neighbors
    # ===========================================================
    print("\n--- Step 1: Computing neighbors ---")
    user_neighbors, item_neighbors = build_neighbor_dicts(
        train_uir=train_uir[:, :3].astype(np.float32),
        n_users=n_users,
        n_items=n_items,
        k=NEIGHBOR_K,
        sim_threshold=-1.0,
    )

    # ===========================================================
    # Build evaluation candidates
    # Negatives = items user rated < threshold in TRAINING set
    # (no data leakage: only training ratings used for negatives)
    # ===========================================================
    valid_uir = valid_df[["user_id", "item_id"]].to_numpy(dtype=np.float64)
    test_uir = test_df[["user_id", "item_id"]].to_numpy(dtype=np.float64)

    valid_users, valid_candidates = build_eval_candidates(
        valid_uir, train_df, num_neg=NUM_NEG_EVAL, seed=42
    )
    test_users, test_candidates = build_eval_candidates(
        test_uir, train_df, num_neg=NUM_NEG_EVAL, seed=43
    )
    print(f"Eval users: valid={len(set(valid_users.tolist()))}, "
          f"test={len(set(test_users.tolist()))}")

    # ===========================================================
    # GridSearch over all hyperparameters
    # All tuning data is saved to disk for the experiment section.
    # ===========================================================
    all_tuning_results = []
    best_per_model = {}

    grid_keys = list(GRID.keys())
    grid_values = list(GRID.values())
    total_configs = 1
    for v in grid_values:
        total_configs *= len(v)
    print(f"\n{'='*60}")
    print(f"GridSearch: {total_configs} total configurations")
    print(f"{'='*60}")

    config_idx = 0
    for combo in grid_product(*grid_values):
        config = dict(zip(grid_keys, combo))
        config_idx += 1
        factor = config["factor"]
        lr = config["lr"]
        weight_decay = config["weight_decay"]
        dropout = config["dropout"]
        sasrec_num_neg = config["sasrec_num_neg"]

        emb_dim = factor * 2
        mlp_layers = [emb_dim, factor]

        print(f"\n{'='*60}")
        print(f"[Config {config_idx}/{total_configs}] {config}")
        print(f"{'='*60}")

        # --- Step 2: Pretrain SASRec ---
        print(f"\n--- Step 2: Pretraining SASRec (dim={emb_dim}, sasrec_num_neg={sasrec_num_neg}) ---")

        sasrec_train_dataset = SasRecTrainDataset(
            user_history=user_history,
            n_users=n_users,
            n_items=n_items,
            max_len=SASREC_MAXLEN,
            sasrec_num_neg=sasrec_num_neg,
            seed=42,
        )

        user_emb, item_emb = pretrain_sasrec(
            train_dataset=sasrec_train_dataset,
            user_history=user_history,
            n_users=n_users,
            n_items=n_items,
            device=DEVICE,
            hidden_units=emb_dim,
            max_len=SASREC_MAXLEN,
            num_blocks=2,
            num_heads=1,
            dropout_rate=0.2,
            batch_size=SASREC_BATCH_SIZE,
            lr=1e-3,
            epochs=SASREC_EPOCHS,
        )

        # --- Build models ---
        models = {
            "MF": MF(n_users, n_items, embedding_dim=emb_dim),

            "NCF": NCF(n_users, n_items, embedding_dim=emb_dim, hidden_dims=mlp_layers),

            "SASRec-NCF": SASRecNCF(user_emb, item_emb, hidden_dims=mlp_layers,
                                     freeze_pretrained=False),

            "NeighborAware": NeighborAware(
                user_emb, item_emb, user_neighbors, item_neighbors,
                n_users, n_items, k=NEIGHBOR_K,
                freeze_pretrained=False, dropout=dropout,
            ),
        }

        # --- Train all models ---
        for name, model in models.items():
            print(f"\n--- Training {name} (config {config_idx}) ---")
            trained_model = train_rating_model(
                model=model,
                train_dataset=train_dataset,
                valid_users=valid_users,
                valid_candidates=valid_candidates,
                model_name=f"{name}-cfg{config_idx}",
                lr=lr,
                weight_decay=weight_decay,
            )

            # --- Evaluate on validation and test ---
            valid_hr, valid_ndcg = evaluate_model(trained_model, valid_users,
                                                  valid_candidates,
                                                  TOP_K, DEVICE)
            test_hr, test_ndcg = evaluate_model(trained_model, test_users,
                                                test_candidates,
                                                TOP_K, DEVICE)

            result_row = {
                "config_idx": config_idx,
                "model": name,
                **config,
                "valid_hr": valid_hr,
                "valid_ndcg": valid_ndcg,
                "test_hr": test_hr,
                "test_ndcg": test_ndcg,
            }
            all_tuning_results.append(result_row)

            # Track best config per model (by validation NDCG)
            if name not in best_per_model or valid_ndcg > best_per_model[name]["valid_ndcg"]:
                best_per_model[name] = result_row.copy()

            print(f"  {name:15s}  valid_hr@{TOP_K}={valid_hr:.4f}  "
                  f"valid_ndcg@{TOP_K}={valid_ndcg:.4f}  "
                  f"test_hr@{TOP_K}={test_hr:.4f}  "
                  f"test_ndcg@{TOP_K}={test_ndcg:.4f}")

        # Save tuning results incrementally (in case of crash)
        with open(OUTPUT_DIR / "gridsearch_all_results.json", "w", encoding="utf-8") as f:
            json.dump(all_tuning_results, f, indent=2)

    # ===========================================================
    # Save final results
    # ===========================================================
    with open(OUTPUT_DIR / "gridsearch_all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_tuning_results, f, indent=2)

    with open(OUTPUT_DIR / "gridsearch_best_per_model.json", "w", encoding="utf-8") as f:
        json.dump(best_per_model, f, indent=2)

    # Build summary grouped by factor for plotting
    all_results = {}
    for row in all_tuning_results:
        f = row["factor"]
        name = row["model"]
        if f not in all_results:
            all_results[f] = {}
        if name not in all_results[f] or row["valid_ndcg"] > all_results[f][name]["ndcg"]:
            all_results[f][name] = {"hr": row["test_hr"], "ndcg": row["test_ndcg"]}

    with open(OUTPUT_DIR / "results_by_factor.json", "w", encoding="utf-8") as fp:
        json.dump(all_results, fp, indent=2)

    if all_results:
        plot_results(all_results, OUTPUT_DIR)

    print(f"\nGridSearch complete. All {len(all_tuning_results)} results saved to {OUTPUT_DIR}")
    print(f"Best config per model:")
    for name, best in best_per_model.items():
        print(f"  {name}: valid_ndcg={best['valid_ndcg']:.4f}, "
              f"test_ndcg={best['test_ndcg']:.4f}, config={best}")


if __name__ == "__main__":
    main()