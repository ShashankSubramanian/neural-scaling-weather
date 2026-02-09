# Transformer Math
## Notation
### Model Dimensions

| Symbol | Description |
|--------|-------------|
| `B` | Batch size |
| `T` | Time dimension |
| `H` | Height (spatial dimension) |
| `W` | Width (spatial dimension) |
| `P` | Patch size |
| `V` | Number of input variables (includes invariants and features) |
| `E` | Embedding dimension |
| `Ah` | Number of attention heads |
| `Hd` | Head dimension = `E/Ah` |
| `Wh` | Window height |
| `Ww` | Window width |

 

### Parallelism Configuration

| Symbol | Description |
|--------|-------------|
| `dp` | Number of data parallel groups |
| `tp` | Number of tensor parallel groups |
| `sp1` | Number of spatial parallel groups in dimension 1 |
| `sp2` | Number of spatial parallel groups in dimension 2 |

**Total GPUs:** `dp × tp × sp1 × sp2`

## Split
Split batch among `dp`. Use `sp1` to split H, `sp2` to split W. We will use `tp` later
`B x T x H x W x V -> B/dp x T x H/sp1 x W/sp2 x V`

## Patch Embedding
**Patchify**

 `B/dp x T x H/sp1 x W/sp2 x V -> B/dp x T x H/(sp1 x P) x W/(sp2 x P) x P^2V`

**Embed** 

   `B/dp x T x H/(sp1 x P) x W/(sp2 x P) x P^2V, P^2V x E -> B/dp x T x H/(sp1 x P) x W/(sp2 x P) x E`

## Transformer Block
**LayerNorm**

 `B/dp x T x H/(sp1 x P) x W/(sp2 x P) x E -> B/dp x T x H/(sp1 x P) x W/(sp2 x P) x E`

(LayerNorm Math Omitted; But it has 2E weights that's shared in `sp1 x sp2 x tp`)


## Roll (if needed)
`B/dp x T x H/(sp1 x P) x W/(sp2 x P) x E -> B/dp x T x H/(sp1 x P) x W/(sp2 x P) x E`    [P2P]

## Build Windows
Given:
| Symbol | Description |
|--------|-------------|
| `Wh` | Window height |
| `Ww` | Window width |


Define:
| Symbol | Description |
|--------|-------------|
| `Nw` | Number of local windows = `H/(P x Wh x sp1) x W/(P x Ww x sp2)` |

**Window**

`B/dp x T x H/(sp1 x P) x W/(sp2 x P) x E -> (B x Nw)/dp x T x Wh x Ww x E`

Define:
| Symbol | Description |
|--------|-------------|
| `Lw` | Sequence length in window = `T x Wh x Wv` (spatiotemporal window volume) |

  
`(B x Nw)/dp x T x Wh x Ww x E -> (B x Nw)/dp x Lw x E`

  

## Self-Attention
**QKV**

`(B x Nw)/dp x Lw x E, E x 3E/tp -> (B x Nw)/dp x Lw x 3E/tp`

`(B x Nw)/dp x Lw x 3E/tp -> (B x Nw)/dp x Ah/tp x Lw x Hd x 3`


**QK^T**

`(B x Nw)/dp x Ah/tp x Lw x Hd, (B x Nw)/dp x Ah/tp x Hd x Lw -> (B x Nw)/dp x Ah/tp x Lw x Lw`

**Softmax**

`(B x Nw)/dp x Ah/tp x Lw x Lw -> (B x Nw)/dp x Ah/tp x Lw x Lw`

**Attend**

`(B x Nw)/dp x Ah/tp x Lw x Lw, (B x Nw)/dp x Ah/tp x Lw x Hd -> (B x Nw)/dp x Ah/tp x Lw x Hd`

**Project**

`(B x Nw)/dp x Ah/tp x Lw x Hd -> (B x Nw)/dp x Lw x E/tp`

`(B x Nw)/dp x Lw x E/tp, E/tp x E -> (B x Nw)/dp x Lw x E`   [AllReduce]

  

## Un-window

`(B x Nw)/dp x Lw x E -> (B x Nw)/dp x T x Wh x Ww x E`

`(B x Nw)/dp x T x Wh x Ww x E -> B/dp x T x H/(sp1 x P) x W/(sp2 x P) x E`

## Un-roll (if needed)

`B/dp x T x H/(sp1 x P) x W/(sp2 x P) x E -> B/dp x T x H/(sp1 x P) x W/(sp2 x P) x E` [P2P]

`B/dp x T x H/(sp1 x P) x W/(sp2 x P) x E -> B/dp x T x H/(sp1 x P) x W/(sp2 x P.tp) x E` [ReduceScatter]

 
## MLP
**LayerNorm**

` B/dp x T x H/(sp1 x P) x W/(sp2 x P) x E ->  B/dp x T x H/(sp1 x P) x W/(sp2 x P) x E` 
  
**FC1**

Define:
| Symbol | Description |
|--------|-------------|
| `L` | MLP Sequence Length = `T x H/(sp1 x P) x W/(sp2 x P)` |

`B/dp x T x H/(sp1 x P) x W/(sp2 x P) x E -> B/dp x L x E`
`B/dp x L x E, E x 4E/tp -> B/dp x L x 4E/tp`

 
**FC2**

`B/dp x L x 4E/tp, 4E/tp x E -> B/dp x L x E` [AllReduce]

`B/dp x L x E -> B/dp x T x H/(sp1 x P) x W/(sp2 x P) x E` 

--------
`B/dp x T x H/(sp1 x P) x W/(sp2 x P) x E` feeds into next transformer block
