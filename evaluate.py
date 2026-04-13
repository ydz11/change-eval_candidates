import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict


class EvalDataset(Dataset):
    def __init__(self, users, candidates):
        self.users      = users.astype(np.int64)
        self.candidates = candidates.astype(np.int64)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.candidates[idx]


def build_eval_candidates(eval_user_item_pairs, train_df, num_neg=100, seed=42):
    rng = np.random.default_rng(seed)
    users     = eval_user_item_pairs[:, 0].astype(np.int64)
    pos_items = eval_user_item_pairs[:, 1].astype(np.int64)

    # 预先建好每个用户的neg pool，避免loop里重复扫描train_df
    user_neg_pool = defaultdict(list)
    for row in train_df.itertuples(index=False):
        if row.rating < 4:
            user_neg_pool[int(row.user_id)].append(int(row.item_id))

    candidates = np.zeros((len(users), 1 + num_neg), dtype=np.int64)
    candidates[:, 0] = pos_items

    for idx, (u, pos_i) in enumerate(zip(users.tolist(), pos_items.tolist())):
        pool = np.array(user_neg_pool[u], dtype=np.int64)
        if len(pool) == 0:
            continue
        sampled = rng.choice(pool, size=num_neg, replace=len(pool) < num_neg)
        candidates[idx, 1:] = sampled

    return users, candidates


def ndcg_from_rank(rank, k):
    if rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


@torch.no_grad()
def evaluate_model(model, eval_users, eval_candidates, k, device, batch_size=4096):
    model.eval()

    ds = EvalDataset(eval_users, eval_candidates)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    row_users = []
    row_hits  = []
    row_ndcgs = []

    for u, items in dl:
        bsz, n_cand = items.shape
        u_ids  = u.clone()
        u_exp  = u.to(device).view(-1, 1).expand(bsz, n_cand).reshape(-1)
        i_flat = items.to(device).reshape(-1)
        scores = model(u_exp, i_flat).view(bsz, n_cand).cpu().numpy()

        for b in range(bsz):
            row_scores = scores[b]
            pos_score  = row_scores[0]
            neg_scores = row_scores[1:]
            rank = 1 + int((neg_scores > pos_score).sum())
            row_users.append(u_ids[b].item())
            row_hits.append(1.0 if rank <= k else 0.0)
            row_ndcgs.append(ndcg_from_rank(rank, k))

    user_hits  = defaultdict(list)
    user_ndcgs = defaultdict(list)
    for uid, h, n in zip(row_users, row_hits, row_ndcgs):
        user_hits[uid].append(h)
        user_ndcgs[uid].append(n)

    avg_hr   = float(np.mean([np.mean(v) for v in user_hits.values()]))
    avg_ndcg = float(np.mean([np.mean(v) for v in user_ndcgs.values()]))

    return avg_hr, avg_ndcg