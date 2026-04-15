import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def build_rating_matrix(train_uir, n_users, n_items):
    rows = train_uir[:, 0].astype(np.int64) - 1
    cols = train_uir[:, 1].astype(np.int64) - 1
    vals = train_uir[:, 2].astype(np.float32)
    return csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))


def build_neighbor_dicts(train_uir, n_users, n_items, k=5, sim_threshold=-1.0):
    R = build_rating_matrix(train_uir, n_users, n_items)

    user_sim = cosine_similarity(R)
    item_sim = cosine_similarity(R.T)

    user_neighbors = {}
    item_neighbors = {}

    n_user_neighbors = min(k, n_users - 1)
    for user_idx in range(n_users):
        sims = user_sim[user_idx].copy()
        sims[user_idx] = -1.0
        topk_idx = np.argsort(-sims)[:n_user_neighbors]
        # --- Bug 7 fix: only keep neighbors above threshold ---
        filtered = [int(i + 1) for i in topk_idx if sims[i] > sim_threshold]
        user_neighbors[user_idx + 1] = filtered

    n_item_neighbors = min(k, n_items - 1)
    for item_idx in range(n_items):
        sims = item_sim[item_idx].copy()
        sims[item_idx] = -1.0
        topk_idx = np.argsort(-sims)[:n_item_neighbors]
        # --- Bug 7 fix: only keep neighbors above threshold ---
        filtered = [int(i + 1) for i in topk_idx if sims[i] > sim_threshold]
        item_neighbors[item_idx + 1] = filtered

    # Log neighbor coverage stats
    user_counts = [len(v) for v in user_neighbors.values()]
    item_counts = [len(v) for v in item_neighbors.values()]
    print(f"[Neighbors] user avg={np.mean(user_counts):.1f}, "
          f"min={np.min(user_counts)}, max={np.max(user_counts)}")
    print(f"[Neighbors] item avg={np.mean(item_counts):.1f}, "
          f"min={np.min(item_counts)}, max={np.max(item_counts)}")

    # 添加调试：检查邻居相似度
    print(f"[Debug] User 1 neighbors: {user_neighbors.get(1, [])}")
    print(f"[Debug] User 1 sim scores: {user_sim[0][:10]}")  # 前10个相似度

    return user_neighbors, item_neighbors