import torch
import torch.nn as nn


# Multi-Head Self Attention (MSA)
class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super().__init__()
        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads

        self.q_mappings = nn.ModuleList([nn.Linear(self.d_head, self.d_head) for _ in range(n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(self.d_head, self.d_head) for _ in range(n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(self.d_head, self.d_head) for _ in range(n_heads)])

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        batch_size, seq_length, _ = sequences.shape
        result = []

        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q = self.q_mappings[head](sequence[:, head * self.d_head: (head + 1) * self.d_head])
                k = self.k_mappings[head](sequence[:, head * self.d_head: (head + 1) * self.d_head])
                v = self.v_mappings[head](sequence[:, head * self.d_head: (head + 1) * self.d_head])

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)

            result.append(torch.hstack(seq_result))

        return torch.stack(result)


# Positional Embeddings
def get_positional_embeddings(sequence_length, d, device="cpu"):
    pos_enc = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            pos_enc[i, j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return pos_enc.to(device)


# Vision Transformer (ViT)
class ViT(nn.Module):
    def __init__(self, input_shape, n_patches=7, hidden_d=8, n_heads=2, out_d=10):
        super().__init__()
        self.input_shape = input_shape
        self.n_patches = n_patches
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        assert input_shape[1] % n_patches == 0 and input_shape[2] % n_patches == 0, "Input shape must be divisible by number of patches"

        self.patch_size = (input_shape[1] // n_patches, input_shape[2] // n_patches)
        self.input_d = input_shape[0] * self.patch_size[0] * self.patch_size[1]

        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        self.ln1 = nn.LayerNorm((n_patches ** 2 + 1, hidden_d))
        self.msa = MSA(hidden_d, n_heads)
        self.ln2 = nn.LayerNorm((n_patches ** 2 + 1, hidden_d))
        self.enc_mlp = nn.Sequential(nn.Linear(hidden_d, hidden_d), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(hidden_d, out_d), nn.Softmax(dim=-1))

    def forward(self, images):
        batch_size, _, _, _ = images.shape
        patches = images.reshape(batch_size, self.n_patches ** 2, self.input_d)
        tokens = self.linear_mapper(patches)

        class_tokens = self.class_token.expand(batch_size, -1, -1)
        tokens = torch.cat((class_tokens, tokens), dim=1)

        pos_emb = get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_d, device=tokens.device)
        tokens += pos_emb.repeat(batch_size, 1, 1)

        out = tokens + self.msa(self.ln1(tokens))
        out = out + self.enc_mlp(self.ln2(out))

        return self.mlp(out[:, 0])
