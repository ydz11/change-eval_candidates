import torch
import torch.nn as nn


class NeighborAware(nn.Module):

    def __init__(self, user_emb, item_emb, user_neighbors, item_neighbors,
                 n_users, n_items, k=5, freeze_pretrained=False, dropout=0.2):
        super().__init__()
        self.k = k
        emb_dim = user_emb.shape[1]

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

        # Self-attention for neighbor aggregation (Set-based, Order-invariant)
        self.user_attn = nn.MultiheadAttention(emb_dim, num_heads=1,
                                                dropout=dropout, batch_first=True)
        self.item_attn = nn.MultiheadAttention(emb_dim, num_heads=1,
                                                dropout=dropout, batch_first=True)

        # MLP input: [u_context(emb_dim), i_context(emb_dim)] = 2*emb_dim
        # 维度固定，不随k增大
        mlp_input_dim = 2 * emb_dim

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
        batch_size = user.size(0)

        # Target embeddings
        u_target = self.user_emb(user)  # [batch, emb_dim]
        i_target = self.item_emb(item)  # [batch, emb_dim]

        # Neighbor embeddings
        u_nei_ids = self.user_topk_buf[user]   # [batch, k]
        u_nei_emb = self.user_emb(u_nei_ids)   # [batch, k, emb_dim]

        i_nei_ids = self.item_topk_buf[item]   # [batch, k]
        i_nei_emb = self.item_emb(i_nei_ids)   # [batch, k, emb_dim]

        # Padding mask
        u_pad_mask = u_nei_ids.eq(0)  # [batch, k] True=padding
        i_pad_mask = i_nei_ids.eq(0)  # [batch, k]

        # Self-attention aggregation (Set-based, Order-invariant)
        # query=target, key=value=neighbors
        u_query = u_target.unsqueeze(1)  # [batch, 1, emb_dim]
        u_context, _ = self.user_attn(
            query=u_query,
            key=u_nei_emb,
            value=u_nei_emb,
            key_padding_mask=u_pad_mask,
        )  # [batch, 1, emb_dim]
        u_context = u_context.squeeze(1)  # [batch, emb_dim]

        i_query = i_target.unsqueeze(1)  # [batch, 1, emb_dim]
        i_context, _ = self.item_attn(
            query=i_query,
            key=i_nei_emb,
            value=i_nei_emb,
            key_padding_mask=i_pad_mask,
        )  # [batch, 1, emb_dim]
        i_context = i_context.squeeze(1)  # [batch, emb_dim]

        # Concat context-aware representations → MLP
        mlp_input = torch.cat([u_context, i_context], dim=-1)  # [batch, 2*emb_dim]
        output = self.mlp(mlp_input)

        return output.squeeze(-1)


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