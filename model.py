"""
somebody argued that human intelligence is not truly "general" intelligence, claiming it is only general regarding concepts we humans can comprehend and aware of, and that we are ignorant of anything beyond that.
i believe this is incorrect. they are referring to metaphysical ideas which aren't "real" and have no practical effect on human life. human intelligence is general regarding everything we comprehend—which encompasses all functional reality.
they also claimed we "suck" at chess. we don't. we may not be the absolute best, but we aren't bad either. chess is a closed system with fixed rules. being worse than stockfish doesn't mean our intelligence isn't general
 it simply means we aren't optimized for massive, brute-force search trees in perfect-information games

you know, the cia doesn't like hiring sociopaths, because they disregard the rules, they actually like people with sociopathic tendencies but not too much. they want someone who can switch off empathy when necessary but still respects the chain of command. 
it's a fine line between a useful asset and a liability
anyway, enough yapping. let's build a transformer
"""

from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1 
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str,theta_constant : float =  10000.0) -> torch.Tensor:
    assert head_dim % 2 == 0, "Dimension must be divisible by 2 you dumbfuck"
    #this is the the '2i' in theta
    theta_numerator = torch.arange(0, head_dim, 2).float()
    #theta = 10000**(-2i/head_dim) which we can simplify it to be 1/10000**(2i/head_dim)
    theta = 1.0/ theta_constant ** (theta_numerator / head_dim).to(device = device) #(head_dim/2, )
    # now lets compute m, which is basically the token position in the sentence (eg: first token will be 1 second will be 2 ...etc)
    # m is basicaly all the natural numbers between 1 and sequence length(seq_len)
    m = torch.arange(seq_len, device = device).float() #(seq_len, )
    frequencies = torch.outer(m, theta).float() # this is of shape (seq_len, head_dim / 2)
    # we convert them to polar form, r(cos( m * theta) + sin(m * theta)i) = eî*m*theta(euler formula)
    #torch.ones_like basically takes a matrix A and creates a new matrix b where b is of same shape as A and it only contains 1 on all rows and columns
    # so torch.ones_like is basically the magnitude r from the polar form, so it is basically unit circle(magnitude of 1)
    frequencies_complex = torch.polar(torch.ones_like(frequencies), frequencies)

    return frequencies_complex

def apply_rope(x: torch.Tensor, frequencies_complex: torch.Tensor, device: str):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = frequencies_complex.unsqueeze(0).unsqueeze(2)
    x_rotate = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotate)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # number of heads used for keys and values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # number of heads used for queries
        self.n_heads_q = args.n_heads
        # how many times we need to repeat k/v heads to match q heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # size of each head (the slice of the embedding each head looks at)
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # kv cache used during autoregressive inference
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_complex: torch.Tensor
    ):
        batch_size, seq_len, _ = x.shape  # (b, 1, dim)

        # project input to queries
        xq = self.wq(x)
        # project input to keys
        xk = self.wk(x)
        # project input to values
        xv = self.wv(x)

        # reshape queries into heads
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # reshape keys into kv heads
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # reshape values into kv heads
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # apply rotary embeddings to queries
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        # apply rotary embeddings to keys
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # write the new keys and values into the cache
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # grab all cached keys up to the current position
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        # grab all cached values up to the current position
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # q heads are grouped, so each group shares the same k/v
        # just repeat k/v heads to line up with q heads

        # repeat keys to match number of query heads
        keys = repeat_kv(keys, self.n_rep)
        # repeat values to match number of query heads
        values = repeat_kv(values, self.n_rep)

        # move head dimension forward for attention math
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # compute scaled dot-product attention scores
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # normalize scores with softmax
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # apply attention weights to values
        output = torch.matmul(scores, values)
        # merge heads back into a single embedding
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # final linear projection back to model dimension
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelArgs
    ):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self,args: ModelArgs):
        super().__init__()
        assert args.dim % args.n_heads ==0, "embedding dimension must be divisble by number of heads"
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        self.attention_norm = RMSNorm(args.dim, eps = args.norm_eps)
        self.feed_forward_norm = RMSNorm(args.dim, eps = args.norm_eps)

    def forward(self, x: torch.Tensor , start_pos: int, freqs_complex: torch.Tensor):
        # a skip connection + prenorom of x throughr self attention
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        # here we normalize the later output, and then the output goes throught a feedforward layer
        out = h + self.feed_forward(self.feed_forward_norm(h))
        return out 
     
class Transformer(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, Seq_Len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)

        # retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        
        # consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        # project to vocab size
        output = self.output(h).float()
        return output




"""
this is multi query attention, where we use the same key and value for all queries, in other words, we have a single kv head for all q heads, it is written here just for the purpose of learning
"""
class MultiQueryAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.dim % args.n_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)   
        self.register_buffer("cache_k", None, persistent = False)
        self.register_buffer("cache_v", None, persistent = False)

    def forward(self, x: torch.Tensor, use_cache: bool = True):
        B, T, C = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.head_dim).unsqueeze(1)
        v = self.wv(x).view(B, T, self.head_dim).unsqueeze(1)
        if use_cache: # we check if we are using cache, if we are we will use the cache to store the keys and values
            if self.cache_k is None: # if we don't have a cache, we will create one
                self.cache_k = k
                self.cache_v = v
            else: # in here we basically if we do have cache, we will append the new keys and values to the cache
                self.cache_k = torch.cat([self.cache_k, k ], dim = 2)
                self.cache_v = torch.cat([self.cache_v, v ], dim = 2)
            k = self.cache_k
            v = self.cache_v
        
        q = q.transpose(1, 2)   
        t_k = k.shape[2] # this is the number of tokens in the cache, we will use this to create the causal mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # this is of size (B, H, T, T) where T is the sequence length and H is the number of heads
        # now we create the causal mask so that we only consider the previous tokens
        causal_mask = torch.tril(torch.ones(T, t_k), device = x.device, dtype = torch.bool).unsqueeze(0).unsqueeze(0) # this is of size (1, 1, T, T)
        scores = scores.masked_fill(causal_mask == 0, float('-inf')) # this is of size (B, H,T, t_k). this will set all upper triangular values to -inf so that they are not considred in the softmax(past tokens can't attend to future tokens)

        scores = F.softmax(scores.float(), dim=-1).type_as(x)

        out = torch.matmul(scores, v) # this is of size (B, H, T, Head_Dim) where H is the number of heads, T is the sequence length, and head_dim is the dimension of each head
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim) # this is of size (B, T, C(in other words number of heads * head_dim))
        return self.wo(out)


class GroupedQueryAttention(nn.Module):
    def __init__(self, args: ModelArgs, max_seq_len: int):
        super().__init__()

        assert args.dim % args.n_heads == 0
        assert args.n_heads % args.n_kv_heads == 0

        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.group_size = self.n_heads // self.n_kv_heads
        self.max_seq_len = max_seq_len

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

        self.register_buffer("cache_k", torch.zeros(1, self.n_kv_heads, max_seq_len, self.head_dim), persistent=False)
        self.register_buffer("cache_v", torch.zeros(1, self.n_kv_heads, max_seq_len, self.head_dim), persistent=False)
        self.register_buffer("cache_len", torch.zeros(1, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor, use_cache: bool = True):
        B, T, C = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if use_cache:
            start = self.cache_len.item()
            end = start + T

            self.cache_k[:, :, start:end, :] = k
            self.cache_v[:, :, start:end, :] = v
            self.cache_len += T

            k = self.cache_k[:, :, :end, :]
            v = self.cache_v[:, :, :end, :]

        T_k = k.size(2)

        q = q.view(B, self.n_kv_heads, self.group_size, T, self.head_dim)

        scores = torch.matmul(q, k.unsqueeze(2).transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal_mask = torch.tril(torch.ones(T, T_k, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v.unsqueeze(2))

        out = out.view(B, self.n_heads, T, self.head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.wo(out)
