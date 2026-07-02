import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# =========================================================================
# 🏗️ MODEL ARCHITECTURAL COMPONENT LAYERS
# =========================================================================
class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.c_attn = nn.Linear(d_model, d_model * 3)
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        return self.c_proj(out.transpose(1, 2).contiguous().view(B, T, C))

class FeedForward(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.c_fc   = nn.Linear(d_model, 4 * d_model)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalMultiHeadAttention(d_model, num_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn  = FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x

class MiniNanoGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.Sequential(*[TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.size()
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

# =========================================================================
# 📝 DATASET SETUP & REACTION LOOPS
# =========================================================================
training_corpus = """
attention is all you need. self-attention is a core component of transformers.
transformers are the heart of artificial intelligence ecosystems.
when a word speaks at a party, it uses queries, keys, and values to find neighbors.
the dog bit the man and the man bit the dog back.
"""

chars = sorted(list(set(training_corpus)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(text: str) -> list: return [stoi[c] for c in text]
def decode(ids: list) -> str: return "".join([itos[i] for i in ids])

data = torch.tensor(encode(training_corpus), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split: str, batch_size: int, block_size: int):
    dataset = train_data if split == "train" else val_data
    ix = torch.randint(len(dataset) - block_size, (batch_size,))
    x = torch.stack([dataset[i:i+block_size] for i in ix])
    y = torch.stack([dataset[i+1:i+block_size+1] for i in ix])
    return x, y

# =========================================================================
# 🏋️ OPTIMIZATION ENGINE RUNNER
# =========================================================================
def train_model():
    BATCH_SIZE   = 8
    BLOCK_SIZE   = 16
    EPOCHS       = 600
    LEARNING_RATE = 1e-3
    
    print(f"📊 Vocabulary Size: {vocab_size} distinct characters.")
    
    model = MiniNanoGPT(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=2,
        num_layers=2,
        max_seq_len=BLOCK_SIZE
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print("\n🚀 Commencing model training session...")
    print("-" * 50)
    
    model.train()
    for epoch in range(EPOCHS):
        xb, yb = get_batch("train", batch_size=BATCH_SIZE, block_size=BLOCK_SIZE)
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            print(f"🔄 Step {epoch:03d} | Current Cross-Entropy Training Loss: {loss.item():.4f}")

    print("-" * 50)
    print("✅ Training complete. Evaluating generation coherence...")
    
    model.eval()
    seed_prompt = "attention"
    encoded_seed = torch.tensor([encode(seed_prompt)], dtype=torch.long)
    generated_tokens = model.generate(encoded_seed, max_new_tokens=60)[0].tolist()
    
    print(f"\n🔮 Input Seed Prompt:  '{seed_prompt}'")
    print(f"🔮 Fine-Tuned Output:  '{decode(generated_tokens)}'")

if __name__ == "__main__":
    train_model()