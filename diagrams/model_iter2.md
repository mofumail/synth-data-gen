# Iteration 2 — SessionTransformer with RQ-VAE Item Tokenisation

## Key change from Iteration 1

Replaces the single 630k-way item embedding + projection with a two-stage system:
an **offline RQ-VAE** that tokenises each SKU into 3 discrete codes, and an
**online 3-step item head** that predicts those codes sequentially.
The transformer backbone, action head, and temporal head are unchanged.

---

## Offline Stage — Item Tokenisation (run once before training)

```mermaid
flowchart TD
    subgraph OFFLINE["Offline: Item Tokenisation"]
        direction TB

        PP["product_properties\n─────────────────\ncategory  integer ID · 6 912 unique\nprice     quantile bin 0–99\nname      16 × uint8 quantised LLM emb"]

        FE["Feature Encoder  MLP\n─────────────────────\ncat_emb 6912→32  ⊕\nprice / 99   →  1\nname  / 255  → 16\n─────────────────────\nconcat 49-dim  →  d_rq"]

        subgraph RQ["RQ-VAE  ·  3 residual levels  ·  codebook K=128"]
            direction LR
            L1["Level 1\nquantise z\n→ c₁ ∈ 0..127\nresidual r₁ = z − e₁"]
            L2["Level 2\nquantise r₁\n→ c₂ ∈ 0..127\nresidual r₂ = r₁ − e₂"]
            L3["Level 3\nquantise r₂\n→ c₃ ∈ 0..127"]
            L1 --> L2 --> L3
        end

        OUT["sku2codes  dict\n{sku → (c₁, c₂, c₃)}\nsaved to disk · loaded at dataset init"]

        PP --> FE --> RQ --> OUT
    end
```

**Coverage:** 128³ = 2 097 152 combinations → uniquely covers all 1.5M SKUs.
**Parameter cost:** 3 × 128 × d\_rq  ≪  630k × d (iteration 1).

---

## Online Stage — SessionTransformer (iteration 2)

```mermaid
flowchart TD
    subgraph INPUT["Input  ·  one token per event t"]
        direction LR
        AE["event_emb\naction_t → ℝᵈ"]
        CE["item_emb\nc₁_t  c₂_t  c₃_t\n3 × Embedding 128×d\nsummed → ℝᵈ"]
        DE["delta_emb\nbin_t → ℝᵈ"]
        PO["pos_emb\nt → ℝᵈ"]
        XS["⊕\nx_t ∈ ℝᵈ"]
        AE & CE & DE & PO --> XS
    end

    subgraph MEMORY["Cross-Session History  (unchanged)"]
        HST["prior session events\n→ CrossSessionHistory\n→ memory  ∈ ℝᴮˣ¹ˣᵈ"]
    end

    subgraph DECODER["Causal Transformer Decoder  (unchanged)"]
        DEC["TransformerDecoder\nn_layers · d_model · n_heads\nself-attn  causal\ncross-attn  ← memory"]
    end

    XS --> DECODER
    MEMORY --> DECODER

    HT["h_t  ∈ ℝᵈ\nhidden state at each output position"]
    DECODER --> HT

    subgraph HEADS["Prediction Heads  ·  teacher-forced  ·  single forward pass"]
        direction TB

        AH["ActionHead\nLinear d → 6\nCE loss  ·  all positions"]

        subgraph ITEM["Item Head  ·  3-step cascade"]
            direction TB
            IH1["ItemHead₁\nLinear  d → 128\npredict c₁\nCE loss"]
            IH2["ItemHead₂\nLinear  2d → 128\npredict c₂  given  h_t ⊕ emb(ĉ₁)\nCE loss"]
            IH3["ItemHead₃\nLinear  3d → 128\npredict c₃  given  h_t ⊕ emb(ĉ₁) ⊕ emb(ĉ₂)\nCE loss"]
            IH1 -->|"emb(ĉ₁) appended"| IH2
            IH2 -->|"emb(ĉ₂) appended"| IH3
        end

        TH["TemporalHead  (unchanged)\nLinear d → 64  bins\nCE loss  ·  all positions"]

        AH
        ITEM
        TH
    end

    HT --> AH
    HT --> ITEM
    HT --> TH
```

---

## Loss function

```
loss = action_loss
     + item_loss_c1 + item_loss_c2 + item_loss_c3
     + temporal_loss
```

Item loss is now three 128-way cross-entropy terms (one per code level) computed
only at item-bearing positions, identical masking logic to iteration 1.

---

## Parameter comparison

| Component            | Iteration 1          | Iteration 2             | Δ           |
|----------------------|----------------------|-------------------------|-------------|
| item\_emb            | Embedding 630k × d   | 3 × Embedding 128 × d   | −99.9%      |
| item\_head           | Linear d → 630k      | 3 × Linear ~3d → 128    | −99.9%      |
| Logit tensor / batch | M × 630k (≈ 1.2 GB)  | 3 × M × 128 (< 1 MB)   | −1 600×     |
| Action head          | unchanged            | unchanged               | —           |
| Temporal head        | unchanged            | unchanged               | —           |
| Transformer backbone | unchanged            | unchanged               | —           |

---

## Inference — autoregressive item sampling

```
h_t  →  ItemHead₁  →  sample ĉ₁  →  emb(ĉ₁)
                                          ↓
                    h_t ⊕ emb(ĉ₁)  →  ItemHead₂  →  sample ĉ₂  →  emb(ĉ₂)
                                                                          ↓
                                   h_t ⊕ emb(ĉ₁) ⊕ emb(ĉ₂)  →  ItemHead₃  →  sample ĉ₃
                                                                                      ↓
                                                                  lookup (ĉ₁,ĉ₂,ĉ₃) → SKU
```

Three 128-way multinomial samples replace one 630k-way sample.

---

## Citations

- van den Oord et al., *Neural Discrete Representation Learning*, NeurIPS 2017 (VQ-VAE)
- Lee et al., *Autoregressive Image Generation using Residual Quantization*, CVPR 2022 (RQ-VAE)
- Rajput et al., *Recommender Systems with Generative Retrieval*, NeurIPS 2023 (RQ-VAE for item tokenisation)
- Liu et al., *Multi-Behavior Generative Recommendation*, WWW 2024 (MBGen)
- Hou et al., *Generative Augmentation and Multi-lEvel behavior modeling for Recommendation*, 2024 (GAMER)
