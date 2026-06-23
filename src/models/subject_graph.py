import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubjectGraphBase(nn.Module):
    def __init__(self, num_subjects, embed_dim=4096, hid_dim=256):
        super().__init__()
        self.num_subjects = int(num_subjects)
        self.embed_dim = int(embed_dim)
        self.hid_dim = int(hid_dim)

        self.subject_embeds = nn.Embedding(self.num_subjects, self.hid_dim)
        self.proj_to_hid = nn.Linear(self.embed_dim, self.hid_dim, bias=True)
        self.proj_from_hid = nn.Linear(self.hid_dim, self.embed_dim, bias=True)
        self.gate = nn.Linear(self.embed_dim * 2, 1, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.normal_(self.subject_embeds.weight, mean=0.0, std=0.02)
            nn.init.xavier_uniform_(self.proj_to_hid.weight)
            nn.init.zeros_(self.proj_to_hid.bias)
            nn.init.zeros_(self.proj_from_hid.weight)
            nn.init.zeros_(self.proj_from_hid.bias)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def forward(self, z, subject_labels):
        if subject_labels is None:
            return z
        subj = subject_labels.to(device=z.device)
        valid = subj >= 0
        if not bool(valid.any().item()):
            return z

        z_valid = z[valid]
        subj_valid = subj[valid].long()

        uniq, inv = torch.unique(subj_valid, sorted=True, return_inverse=True)
        node_z = self.proj_to_hid(z_valid.to(dtype=torch.float32))
        num_nodes = int(uniq.size(0))
        sums = torch.zeros(num_nodes, self.hid_dim, device=z.device, dtype=node_z.dtype)
        counts = torch.zeros(num_nodes, 1, device=z.device, dtype=node_z.dtype)
        sums.index_add_(0, inv, node_z)
        ones = torch.ones(inv.size(0), 1, device=z.device, dtype=node_z.dtype)
        counts.index_add_(0, inv, ones)
        proto = sums / counts.clamp(min=1.0)

        subj_embed = self.subject_embeds(uniq).to(dtype=proto.dtype)
        node_feat = proto + subj_embed

        norm = F.normalize(node_feat, p=2, dim=-1)
        sim = torch.matmul(norm, norm.t())
        adj = F.softmax(sim, dim=-1)
        agg = torch.matmul(adj, node_feat)

        g = agg.index_select(0, inv)
        g_out = self.proj_from_hid(g).to(dtype=z_valid.dtype)
        gate = torch.sigmoid(self.gate(torch.cat([z_valid, g_out], dim=-1))).to(dtype=z_valid.dtype)
        fused = z_valid + gate * g_out

        out = z.clone()
        out[valid] = fused
        return out

