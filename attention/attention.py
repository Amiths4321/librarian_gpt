import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be perfectly divisible by num_heads!"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Combined projection layer for all Queries, Keys, and Values at once
        # (This is computationally faster than building 3 separate linear layers)
        self.c_attn = nn.Linear(d_model, d_model * 3)
        
        # Output projection layer to stitch multi-head outputs back together
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch Size B, Sequence Length T, Dimension C)
        B, T, C = x.size()
        
        # Step 1: Project input into Q, K, V grids simultaneously
        # Shape shifts from (B, T, C) -> (B, T, 3 * C)
        qkv = self.c_attn(x)
        
        # Split the matrix back into independent Q, K, and V chunks
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # Step 2: Reshape for Multi-Head Parallel Processing
        # Split dimension C into (num_heads, head_dim) and swap axis 1 and 2
        # Final shape for each: (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Step 3: Compute similarity matrix (Dot-Product Matchmaking)
        # Multiply Q by transposed K across the last two dimensions
        # Shape: (B, num_heads, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Step 4: Scale down to keep variance stable and prevent vanishing gradients
        scores = scores / math.sqrt(self.head_dim)
        
        # Step 5: Apply GPT-style Causal Masking (No Peeking at the Future)
        # Create a lower-triangular matrix of 1s
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        # Replace the 0 entries (the future tokens) with negative infinity
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Step 6: Convert raw scores into an Attention Budget probability map
        attn_weights = F.softmax(scores, dim=-1)
        
        # Step 7: Blend Values based on attention weights
        # Shape: (B, num_heads, T, head_dim)
        out = torch.matmul(attn_weights, v)
        
        # Step 8: Re-assemble (Concatenate) all heads back into one single vector space
        # Swap axes back and flatten heads dimension
        # Shape returns to original footprint: (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Step 9: Run through final linear transformation
        return self.c_proj(out)

# =========================================================================
# 🧪 LIVE SIMULATION RUNNER
# =========================================================================
if __name__ == "__main__":
    # Let's mock a standard input footprint
    BATCH_SIZE = 2
    SEQ_LEN = 5     # e.g., "Transformers use attention layers beautifully"
    D_MODEL = 64    # Hidden state dimensions
    NUM_HEADS = 4   # Split into 4 parallel brains (each head_dim = 16)
    
    # Generate random mock sentence tensor matrix
    mock_input = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    print(f"📥 Mock Input Matrix Shape: {mock_input.shape}")
    
    # Initialize our PyTorch module
    attention_layer = CausalMultiHeadAttention(d_model=D_MODEL, num_heads=NUM_HEADS)
    
    # Execute forward pass
    output = attention_layer(mock_input)
    print(f"📤 Attention Output Shape:  {output.shape}")
    print("\n✅ Stage 2 Attention Layer compiled and executed with zero compilation errors!")