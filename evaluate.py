import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict


class EvalDataset(Dataset):
    def __init__(self, users, candidates, ratings):
        self.users = users.astype(np.int64)
        self.candidates = candidates.astype(np.int64)
        self.ratings = ratings.astype(np.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.candidates[idx], self.ratings[idx]


def build_eval_candidates(eval_user_item_pairs, train_df, num_neg=100, seed=42):
    rng = np.random.default_rng(seed)
    users = eval_user_item_pairs[:, 0].astype(np.int64)
    pos_items = eval_user_item_pairs[:, 1].astype(np.int64)
    pos_rats = eval_user_item_pairs[:, 2].astype(np.float32)

    from collections import defaultdict
    user_eval_pos = defaultdict(set)
    for u, i in zip(users.tolist(), pos_items.tolist()):
        user_eval_pos[u].add(i)

    user_neg_pool = defaultdict(list)
    for row in train_df.itertuples(index=False):
        if row.rating < 4:
            user_neg_pool[int(row.user_id)].append(
                (int(row.item_id), float(row.rating))
            )

    candidates = np.zeros((len(users), 1 + num_neg), dtype=np.int64)
    ratings = np.zeros((len(users), 1 + num_neg), dtype=np.float32)
    candidates[:, 0] = pos_items
    ratings[:, 0] = pos_rats

    for idx, (u, pos_i) in enumerate(zip(users.tolist(), pos_items.tolist())):
        pool = user_neg_pool.get(u, [])
        pool = [(item, rat) for item, rat in pool if item not in user_eval_pos[u]]

        if len(pool) == 0:
            continue

        pool = np.array(pool, dtype=object)
        indices = rng.choice(len(pool), size=num_neg, replace=(len(pool) < num_neg))
        sampled_items = np.array([pool[i][0] for i in indices], dtype=np.int64)
        sampled_rats = np.array([pool[i][1] for i in indices], dtype=np.float32)

        candidates[idx, 1:] = sampled_items
        ratings[idx, 1:] = sampled_rats

    return users, candidates, ratings

@torch.no_grad()
def evaluate_model(model, eval_users, eval_candidates, eval_ratings,
                   k, device, batch_size=4096):
    model.eval()

    ds = EvalDataset(eval_users, eval_candidates, eval_ratings)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    row_users = []
    row_hits = []
    row_ndcgs = []

    for u, items, act_ratings in dl:
        bsz, n_cand = items.shape
        u_exp = u.to(device).view(-1, 1).expand(bsz, n_cand).reshape(-1)
        i_flat = items.to(device).reshape(-1)
        scores = model(u_exp, i_flat).view(bsz, n_cand).cpu().numpy()
        act_ratings = act_ratings.numpy()

        for b in range(bsz):
            row_scores = scores[b]
            row_ratings = act_ratings[b]
            pos_item = items[b][0].item()

            binary_rel = np.zeros(len(row_scores), dtype=np.float64)
            binary_rel[0] = 1.0

            # Real Rank: 按 actual ratings 从高到低排序
            real_rank_order = np.argsort(-row_ratings)
            real_rank_relevances = binary_rel[real_rank_order]

            # Predict Rank: 按 predicted scores 从高到低排序
            predict_rank_order = np.argsort(-row_scores)
            predict_rank_relevances = binary_rel[predict_rank_order]
            predicted_items = items[b].numpy()[predict_rank_order]

            # NDCG(real_rank, predict_rank, k)
            ndcg = ndcg_at_k(real_rank_relevances, predict_rank_relevances, k)

            # HR
            hr = 1.0 if pos_item in predicted_items[:k] else 0.0

            row_users.append(u[b].item())
            row_hits.append(hr)
            row_ndcgs.append(ndcg)

    # 按用户聚合
    user_hits = defaultdict(list)
    user_ndcgs = defaultdict(list)
    for uid, h, n in zip(row_users, row_hits, row_ndcgs):
        user_hits[uid].append(h)
        user_ndcgs[uid].append(n)

    avg_hr = np.mean([np.mean(v) for v in user_hits.values()])
    avg_ndcg = np.mean([np.mean(v) for v in user_ndcgs.values()])

    return float(avg_hr), float(avg_ndcg)


def ndcg_at_k(real_rank_relevances, predict_rank_relevances, k):
    def dcg(relevances):
        relevances = np.array(relevances[:k], dtype=np.float64)
        if len(relevances) == 0:
            return 0.0
        positions = np.arange(1, len(relevances) + 1)
        return np.sum(relevances / np.log2(positions + 1))

    idcg = dcg(real_rank_relevances)
    if idcg == 0:
        return 0.0
    return dcg(predict_rank_relevances) / idcg