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
        self.user_attn = nn.MultiheadAttention(
            emb_dim, num_heads=1, dropout=dropout, batch_first=True
        )
        self.item_attn = nn.MultiheadAttention(
            emb_dim, num_heads=1, dropout=dropout, batch_first=True
        )

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
        u_target = self.user_emb(user)  # [batch, emb_dim]
        i_target = self.item_emb(item)  # [batch, emb_dim]

        u_nei_ids = self.user_topk_buf[user]  # [batch, k]
        u_nei_emb = self.user_emb(u_nei_ids)  # [batch, k, emb_dim]
        i_nei_ids = self.item_topk_buf[item]  # [batch, k]
        i_nei_emb = self.item_emb(i_nei_ids)  # [batch, k, emb_dim]

        # Set-based self-attention: [target | neighbors] → context-aware representation
        # target作为第一个token，邻居作为后续tokens
        u_seq = torch.cat([u_target.unsqueeze(1), u_nei_emb], dim=1)  # [batch, 1+k, emb_dim]
        i_seq = torch.cat([i_target.unsqueeze(1), i_nei_emb], dim=1)  # [batch, 1+k, emb_dim]

        # Padding mask: target不mask，邻居padding位置mask
        u_pad = torch.cat([
            torch.zeros(u_nei_ids.size(0), 1, dtype=torch.bool, device=user.device),
            u_nei_ids.eq(0)
        ], dim=1)  # [batch, 1+k]
        i_pad = torch.cat([
            torch.zeros(i_nei_ids.size(0), 1, dtype=torch.bool, device=item.device),
            i_nei_ids.eq(0)
        ], dim=1)  # [batch, 1+k]

        # Self-attention (Set-based, Order-invariant)
        u_out, _ = self.user_attn(
            query=u_seq, key=u_seq, value=u_seq,
            key_padding_mask=u_pad,
        )  # [batch, 1+k, emb_dim]
        u_context = u_out[:, 0, :]  # 取target位置的输出作为context-aware representation

        i_out, _ = self.item_attn(
            query=i_seq, key=i_seq, value=i_seq,
            key_padding_mask=i_pad,
        )  # [batch, 1+k, emb_dim]
        i_context = i_out[:, 0, :]  # 取target位置的输出

        # Concat → MLP (Step d)
        mlp_input = torch.cat([u_context, i_context], dim=-1)  # [batch, 2*emb_dim]
        return self.mlp(mlp_input).squeeze(-1)