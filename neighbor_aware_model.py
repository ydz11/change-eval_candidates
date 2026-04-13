import torch
import torch.nn as nn


class NeighborAware(nn.Module):

    def __init__(self, user_emb, item_emb, user_neighbors, item_neighbors,
                 n_users, n_items, k=5, freeze_pretrained=False,
                 dropout=0.2):

        super().__init__()
        self.k = k
        emb_dim = user_emb.shape[1]

        # ============================================================
        # Part A: Load pretrained SASRec embeddings (from Step 2)
        # ============================================================
        # padding_idx=0 because user/item IDs start from 1;
        # index 0 is reserved as padding for neighbors that don't exist
        self.user_emb = nn.Embedding.from_pretrained(
            user_emb, freeze=freeze_pretrained, padding_idx=0
        )
        self.item_emb = nn.Embedding.from_pretrained(
            item_emb, freeze=freeze_pretrained, padding_idx=0
        )

        # ============================================================
        # Part B: Build neighbor lookup tables (from Step 1)
        # ============================================================
        user_topk = torch.zeros((n_users + 1, k), dtype=torch.long)
        for u, neigh_list in user_neighbors.items():
            neigh_list = neigh_list[:k]  # take at most k neighbors
            user_topk[u, :len(neigh_list)] = torch.tensor(neigh_list, dtype=torch.long)

        item_topk = torch.zeros((n_items + 1, k), dtype=torch.long)
        for i, neigh_list in item_neighbors.items():
            neigh_list = neigh_list[:k]
            item_topk[i, :len(neigh_list)] = torch.tensor(neigh_list, dtype=torch.long)

        self.register_buffer("user_topk_buf", user_topk)
        self.register_buffer("item_topk_buf", item_topk)

        # ============================================================
        # Part C: MLP for rating prediction (Step 4)
        # ============================================================
        mlp_input_dim = 2 * (k + 1) * emb_dim  # e.g., 2*6*16 = 192 for k=5, d=16

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_input_dim // 2, mlp_input_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_input_dim // 4, 1),
        )

    def forward(self, user, item):

        # ----------------------------------------------------------
        # Step 3a: Get target user/item embeddings
        # ----------------------------------------------------------
        # Shape: [batch_size, emb_dim]
        u_target = self.user_emb(user)
        i_target = self.item_emb(item)

        # ----------------------------------------------------------
        # Step 3b: Get neighbor embeddings
        # ----------------------------------------------------------
        # self.user_topk_buf[user] -> [batch_size, k] (neighbor IDs)
        # self.user_emb(...)        -> [batch_size, k, emb_dim]
        u_nei_ids = self.user_topk_buf[user]  # [batch, k]
        u_nei_emb = self.user_emb(u_nei_ids)  # [batch, k, emb_dim]

        i_nei_ids = self.item_topk_buf[item]  # [batch, k]
        i_nei_emb = self.item_emb(i_nei_ids)  # [batch, k, emb_dim]

        # ----------------------------------------------------------
        # Step 3c: Zero out padded neighbors
        # ----------------------------------------------------------
        u_pad_mask = u_nei_ids.eq(0).unsqueeze(-1)  # [batch, k, 1]
        u_nei_emb = u_nei_emb.masked_fill(u_pad_mask, 0.0)

        i_pad_mask = i_nei_ids.eq(0).unsqueeze(-1)  # [batch, k, 1]
        i_nei_emb = i_nei_emb.masked_fill(i_pad_mask, 0.0)

        # ----------------------------------------------------------
        # Step 3d: Concatenate target + neighbors (flatten neighbors)
        # ----------------------------------------------------------
        # u_nei_emb: [batch, k, emb_dim] -> flatten to [batch, k*emb_dim]
        batch_size = user.size(0)

        u_nei_flat = u_nei_emb.view(batch_size, -1)  # [batch, k*emb_dim]
        i_nei_flat = i_nei_emb.view(batch_size, -1)  # [batch, k*emb_dim]

        # user_side = [target_emb | nei_1_emb | nei_2_emb | ... | nei_k_emb]
        user_side = torch.cat([u_target, u_nei_flat], dim=-1)  # [batch, (k+1)*emb_dim]
        item_side = torch.cat([i_target, i_nei_flat], dim=-1)  # [batch, (k+1)*emb_dim]

        # ----------------------------------------------------------
        # Step 4: MLP rating prediction
        # ----------------------------------------------------------
        # Concatenate user-side and item-side, feed into MLP
        mlp_input = torch.cat([user_side, item_side], dim=-1)  # [batch, 2*(k+1)*emb_dim]
        output = self.mlp(mlp_input)  # [batch, 1]

        return output.squeeze(-1)  # [batch]


class NAMean(nn.Module):


    def __init__(self, user_emb, item_emb, user_neighbors, item_neighbors,
                 n_users, n_items, k=5, freeze_pretrained=False, dropout=0.2):
        super().__init__()
        self.k = k
        emb_dim = user_emb.shape[1]

        # Load pretrained embeddings
        self.user_emb = nn.Embedding.from_pretrained(
            user_emb, freeze=freeze_pretrained, padding_idx=0
        )
        self.item_emb = nn.Embedding.from_pretrained(
            item_emb, freeze=freeze_pretrained, padding_idx=0
        )

        # Build neighbor lookup tables (same as NeighborAware)
        user_topk = torch.zeros((n_users + 1, k), dtype=torch.long)
        for u, neigh_list in user_neighbors.items():
            neigh_list = neigh_list[:k]
            user_topk[u, :len(neigh_list)] = torch.tensor(neigh_list, dtype=torch.long)

        item_topk = torch.zeros((n_items + 1, k), dtype=torch.long)
        for i, neigh_list in item_neighbors.items():
            neigh_list = neigh_list[:k]
            item_topk[i, :len(neigh_list)] = torch.tensor(neigh_list, dtype=torch.long)

        self.register_buffer("user_topk_buf", user_topk)
        self.register_buffer("item_topk_buf", item_topk)

        # MLP input: [u_target(d), u_nei_mean(d), i_target(d), i_nei_mean(d)] = 4*d
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 4, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, user, item):
        # Target embeddings
        u_target = self.user_emb(user)  # [batch, d]
        i_target = self.item_emb(item)  # [batch, d]

        # Neighbor embeddings
        u_nei_ids = self.user_topk_buf[user]  # [batch, k]
        u_nei_emb = self.user_emb(u_nei_ids)  # [batch, k, d]

        i_nei_ids = self.item_topk_buf[item]  # [batch, k]
        i_nei_emb = self.item_emb(i_nei_ids)  # [batch, k, d]

        # Masked mean: average only non-padding neighbors
        u_mask = u_nei_ids.ne(0).float().unsqueeze(-1)  # [batch, k, 1]
        u_nei_mean = (u_nei_emb * u_mask).sum(dim=1) / u_mask.sum(dim=1).clamp(min=1.0)

        i_mask = i_nei_ids.ne(0).float().unsqueeze(-1)
        i_nei_mean = (i_nei_emb * i_mask).sum(dim=1) / i_mask.sum(dim=1).clamp(min=1.0)

        # Concatenate and predict
        mlp_input = torch.cat([u_target, u_nei_mean, i_target, i_nei_mean], dim=-1)
        output = self.mlp(mlp_input)

        return output.squeeze(-1)