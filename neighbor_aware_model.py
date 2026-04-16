import torch
import torch.nn as nn


class NeighborAware(nn.Module):

    def __init__(self, user_emb, item_emb, user_neighbors, item_neighbors,
                 n_users, n_items, k=5, freeze_pretrained=False, dropout=0.2):
        super().__init__()
        self.k = k
        emb_dim = user_emb.shape[1]

        # 添加调试标志
        self.debug_done = False

        self.user_emb = nn.Embedding.from_pretrained(
            user_emb, freeze=freeze_pretrained, padding_idx=0
        )
        self.item_emb = nn.Embedding.from_pretrained(
            item_emb, freeze=freeze_pretrained, padding_idx=0
        )

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

        # MLP input: [u_context(emb_dim), i_context(emb_dim)] = 2*emb_dim
        mlp_input_dim = 2 * (k + 1) * emb_dim

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_input_dim * 2 // 3),  # 384 → 256
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_input_dim * 2 // 3, mlp_input_dim // 3),  # 256 → 128
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_input_dim // 3, mlp_input_dim // 6),  # 128 → 64
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # separate prediction layer
        self.predict_layer = nn.Linear(mlp_input_dim // 6, 1)

        # Add the bias item (required for score prediction)
        self.user_bias = nn.Embedding(n_users + 1, 1, padding_idx=0)
        self.item_bias = nn.Embedding(n_items + 1, 1, padding_idx=0)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

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
        u_mask = u_nei_ids.eq(0).unsqueeze(-1)
        u_nei_emb = u_nei_emb.masked_fill(u_mask, 0.0)

        i_mask = i_nei_ids.eq(0).unsqueeze(-1)
        i_nei_emb = i_nei_emb.masked_fill(i_mask, 0.0)

        # Concatenate: target + neighbors
        u_concat = torch.cat([u_target, u_nei_emb.view(u_target.size(0), -1)], dim=-1)
        i_concat = torch.cat([i_target, i_nei_emb.view(i_target.size(0), -1)], dim=-1)

        # Final concat for MLP
        mlp_input = torch.cat([u_concat, i_concat], dim=-1)  # [batch, 2*(k+1)*emb_dim]
        hidden = self.mlp(mlp_input)
        pred = self.predict_layer(hidden).squeeze(-1)
        pred = pred + self.user_bias(user).squeeze(-1) + self.item_bias(item).squeeze(-1)

        # ===== 调试 =====
        if not self.debug_done:
            print(f"[Debug NeighborAware] u_nei_ids shape: {u_nei_ids.shape}")
            print(f"[Debug NeighborAware] u_nei_ids[0]: {u_nei_ids[0]}")
            print(f"[Debug NeighborAware] u_nei_emb[0] mean: {u_nei_emb[0].mean().item():.4f}")
            print(f"[Debug NeighborAware] u_target mean: {u_target.mean().item():.4f}")
            self.debug_done = True

        return pred