import pandas as pd
import numpy as np


def load_ratings(path):
    cols = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(path, sep="::", engine="python", names=cols)
    df = df.sort_values(["user_id", "timestamp"]).copy()
    return df


def filter_cold_start(df, min_user_interactions=5, min_item_interactions=5):
    prev_len = 0
    while len(df) != prev_len:
        prev_len = len(df)

        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df["user_id"].isin(valid_users)]

        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        df = df[df["item_id"].isin(valid_items)]

    df = df.reset_index(drop=True)
    return df


def reindex_ids(df):
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["item_id"].unique())

    user_map = {old: new for new, old in enumerate(unique_users, start=1)}
    item_map = {old: new for new, old in enumerate(unique_items, start=1)}

    df = df.copy()
    df["user_id"] = df["user_id"].map(user_map)
    df["item_id"] = df["item_id"].map(item_map)

    return df


def get_num_users_items(df):
    n_users = int(df["user_id"].nunique())
    n_items = int(df["item_id"].nunique())
    return n_users, n_items


def ratio_split(df, train_ratio=0.70, valid_ratio=0.15, test_ratio=0.15,
                rating_threshold=4, min_pos_eval=2):
    train_rows, valid_rows, test_rows = [], [], []

    for user_id, group in df.groupby("user_id"):
        group = group.sort_values("timestamp")
        # split by time first (strict temporal)
        n = len(group)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)

        train_part = group.iloc[:n_train]
        valid_part = group.iloc[n_train:n_train + n_valid]
        test_part = group.iloc[n_train + n_valid:]

        # extract positives ONLY from eval splits
        valid_pos = valid_part[valid_part["rating"] >= rating_threshold]
        test_pos = test_part[test_part["rating"] >= rating_threshold]
        total_pos_eval = len(valid_pos) + len(test_pos)

        # ensure enough positives for evaluation
        if total_pos_eval < min_pos_eval:
            # fallback: move everything to train
            train_rows.append(group)
            continue

        # ensure both valid and test have at least 1 positive
        if len(valid_pos) == 0:
            # move earliest positive from test → valid
            valid_pos = test_pos.iloc[:1]
            test_pos = test_pos.iloc[1:]
        if len(test_pos) == 0:
            # move latest positive from valid → test
            test_pos = valid_pos.iloc[-1:]
            valid_pos = valid_pos.iloc[:-1]

        # final assignment
        train_rows.append(train_part)
        valid_rows.append(valid_pos)
        test_rows.append(test_pos)

    train_df = pd.concat(train_rows).reset_index(drop=True)
    valid_df = pd.concat(valid_rows).reset_index(drop=True) if valid_rows else pd.DataFrame()
    test_df = pd.concat(test_rows).reset_index(drop=True) if test_rows else pd.DataFrame()

    return train_df, valid_df, test_df



def build_train_uir(train_df):
    return train_df[["user_id", "item_id", "rating", "timestamp"]].to_numpy(dtype=np.float64)


def build_ui(df):
    return df[["user_id", "item_id", "timestamp"]].to_numpy(dtype=np.float64)