import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-import or drop-in our verified structural Attention block from Stage 2
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
    """The MLP/FFN layer that processes information extracted by attention."""
    def __init__(self, d_model: int):
        super().__init__()
        # Standard scaling factor: expand by 4x, then compress back
        self.c_fc   = nn.Linear(d_model, 4 * d_model)
        self.gelu   = nn.GELU()  # Modern LLM non-linear activation
        self.c_proj = nn.Linear(4 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class TransformerBlock(nn.Module):
    """A full Transformer layer pairing Pre-LayerNorm, Attention, and FFN with residual connections."""
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalMultiHeadAttention(d_model, num_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn  = FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN combined with Residual addition (x + layer(LN(x)))
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class MiniNanoGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        # 1. Token & Position Embedding Tables
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(max_seq_len, d_model)
        
        # 2. Sequential stacking of Transformer blocks
        self.blocks = nn.Sequential(*[TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        
        # 3. Final layer norm and language modeling prediction head
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False) # Maps back to tokens

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.size()
        assert T <= self.max_seq_len, f"Cannot forward sequence of length {T}, max context window is {self.max_seq_len}"
        
        # Compute embeddings
        tok_emb = self.token_embedding_table(idx) # (B, T, d_model)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, d_model)
        
        x = tok_emb + pos_emb # (B, T, d_model)
        x = self.blocks(x)    # Pass through stacked Transformer Layers
        x = self.ln_f(x)      # Final normalization
        
        # Project hidden states to get vocabulary predictions (logits)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # Flatten tensors out for cross-entropy calculation
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """Auto-regressive generation loop: predicts the next token, appends it, repeats."""
        for _ in range(max_new_tokens):
            # If our growing sentence exceeds context window, crop it
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            
            logits, _ = self(idx_cond)
            # Focus exclusively on the very last predicted token position
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token from probability distribution
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

# =========================================================================
# 🧪 MODEL GENERATION TESTING TIMELINE
# =========================================================================
if __name__ == "__main__":
    # Hyperparameters for our micro model config
    VOCAB_SIZE   = 276  # Matches our Stage 1 custom tokenizer size boundary!
    D_MODEL      = 64
    NUM_HEADS    = 4
    NUM_LAYERS   = 3    # Stacking 3 identical blocks
    MAX_SEQ_LEN  = 32   # Tiny context window for simulation
    
    # Instantiate the model
    gpt_model = MiniNanoGPT(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN)
    print(f"🤖 Mini GPT Architecture instantiated successfully!")
    print(f"Total Parameter Channels: {sum(p.numel() for p in gpt_model.parameters())} weights.")

    # Simulating a user prompt token setup: [45, 112, 9, 201]
    prompt_tokens = torch.tensor([[45, 112, 9, 201]], dtype=torch.long)
    print(f"\n📥 Seed Prompt Input Token Stream: {prompt_tokens.tolist()}")

    # Generate 10 text tokens with untrained weights (will output scrambled random vocabulary characters)
    generated_stream = gpt_model.generate(prompt_tokens, max_new_tokens=10)
    print(f"📤 Auto-Regressive Generated Tokens: {generated_stream.tolist()}")
    print("\n✅ Stage 3 Network compiled, forward loops linked, and auto-generation functional!")