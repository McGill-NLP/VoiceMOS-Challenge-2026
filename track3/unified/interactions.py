#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interaction vectors: how the two utterance embeddings are combined before the head.

The baseline concatenates [e_a, e_b, |e_a - e_b|, e_a * e_b]. That choice is load-bearing
and worth ablating, for two measured reasons (both on coral_commonaccent-moe, dev set):

  * It is strongly ROLE-DEPENDENT, not merely asymmetric. Swapping the two inputs takes
    SYS-SRCC from 0.932 to -0.164, and averaging f(a,b) with f(b,a) costs 0.10 SYS-SRCC.
    That is not a defect to be fixed: wav_a is always the system output and wav_b is always
    the sys019 reference, in train, dev and test, so f(b,a) is never asked for and the role
    genuinely carries information. Symmetric variants are provided to ablate this, not
    because symmetry is expected to win.

  * The reference is used LESS than the 1024-d vector suggests. Substituting a wrong
    reference from the same pool only drops SYS-SRCC from 0.932 to 0.786, so much of the
    system-level ranking comes from e_a alone -- "how degraded is this sample" -- rather
    than from comparing it to anything.

Two cheap consequences motivate most of the modes below. First, the embeddings are
L2-normalised, so sum(e_a * e_b) IS the cosine similarity: the information is present but
the head has to discover a uniform sum over 256 dimensions to recover it, while zero-shot
cosine alone scores SYS-SRCC 0.809. Handing it over explicitly costs nothing. Second, the
four blocks arrive on very different scales -- e_a and e_b on the unit sphere,
|e_a - e_b| in [0, 2], and the entries of e_a * e_b around 1/256 -- so the product block
is effectively attenuated before the first Linear ever sees it.

Modes, with d = embedding dim (256 by default):

    baseline         [a, b, |a-b|, a*b]                     4d    the official baseline
    scalars          baseline + [cos, ||a-b||]              4d+2  explicit similarity
    normed           LayerNorm each block, then baseline     4d    equalises block scales
    normed-scalars   both of the above                      4d+2
    signed           [a, b, a-b, a*b]                       4d    keeps the direction
    no-b             [a, |a-b|, a*b]                        3d    drops the reference block
    symmetric        [a+b, |a-b|, a*b]                      3d    f(a,b) == f(b,a) exactly
    bilinear         baseline + (Ua) * (Vb)                 4d+r  learned compared subspaces
    no-b-bilinear    no-b + (Ua) * (Vb)                     3d+r  the two that helped

`no-b-bilinear` combines the two modes that gained most in the first ablation (ECAPA + mlp
+ MSE, 20,000 steps, dev UTT-SRCC against the `baseline` control): `no-b` was +0.030 on
spk_sim and +0.002 on acc_sim, `bilinear` +0.017 and +0.030. They pull in compatible
directions -- one removes the raw reference block, the other adds a learned comparison of
projected subspaces -- so the combination keeps a route from b to the head while dropping
the 256 raw dimensions that measurably were not earning their place.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

INTERACTIONS = [
    "baseline", "scalars", "normed", "normed-scalars",
    "signed", "no-b", "symmetric", "bilinear", "no-b-bilinear",
]

# Modes that drop the raw e_b block, and modes that add the learned bilinear term.
_DROPS_B = ("no-b", "no-b-bilinear")
_USES_BILINEAR = ("bilinear", "no-b-bilinear")


class Interaction(nn.Module):
    """Combines two [B, d] embeddings into the [B, out_dim] vector the head consumes."""

    def __init__(self, dim: int, mode: str = "baseline", bilinear_rank: int = 64):
        super().__init__()
        if mode not in INTERACTIONS:
            raise ValueError(f"Unknown interaction '{mode}'. Available: {', '.join(INTERACTIONS)}")
        self.mode = mode
        self.dim = dim

        n_blocks = {"baseline": 4, "scalars": 4, "normed": 4, "normed-scalars": 4,
                    "signed": 4, "no-b": 3, "symmetric": 3, "bilinear": 4,
                    "no-b-bilinear": 3}[mode]
        self.out_dim = n_blocks * dim

        # Per-block LayerNorm. Applied to the d-wide blocks only -- never to the scalar
        # features, where normalising across two values would destroy them.
        self.norms = None
        if mode in ("normed", "normed-scalars"):
            self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_blocks)])

        if mode in ("scalars", "normed-scalars"):
            # cos and ||a-b|| are monotone transforms of each other once the embeddings are
            # L2-normalised (||a-b||^2 = 2 - 2cos), so this adds no information the vector
            # did not already contain. The point is that the head no longer has to *find*
            # it: as one explicit feature rather than a uniform sum over d dimensions.
            self.out_dim += 2

        self.bilinear_u = self.bilinear_v = None
        if mode in _USES_BILINEAR:
            # (Ua) * (Vb) generalises the elementwise product, which is the special case
            # U = V = I. Appended rather than replacing a*b, so a null result means the
            # term is genuinely useless and not that something was lost. In no-b-bilinear
            # it is also the only path from b to the head besides |a-b| and a*b.
            self.bilinear_u = nn.Linear(dim, bilinear_rank, bias=False)
            self.bilinear_v = nn.Linear(dim, bilinear_rank, bias=False)
            self.out_dim += bilinear_rank

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        diff_abs = torch.abs(emb_a - emb_b)
        prod = emb_a * emb_b

        if self.mode in _DROPS_B:
            blocks = [emb_a, diff_abs, prod]
        elif self.mode == "signed":
            blocks = [emb_a, emb_b, emb_a - emb_b, prod]
        elif self.mode == "symmetric":
            blocks = [emb_a + emb_b, diff_abs, prod]
        elif self.mode in ("baseline", "scalars", "normed", "normed-scalars", "bilinear"):
            blocks = [emb_a, emb_b, diff_abs, prod]
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(self.mode)

        if self.norms is not None:
            blocks = [norm(block) for norm, block in zip(self.norms, blocks)]

        if self.mode in _USES_BILINEAR:
            blocks.append(self.bilinear_u(emb_a) * self.bilinear_v(emb_b))

        if self.mode in ("scalars", "normed-scalars"):
            blocks.append(F.cosine_similarity(emb_a, emb_b, dim=-1).unsqueeze(-1))
            blocks.append(torch.linalg.vector_norm(emb_a - emb_b, dim=-1, keepdim=True))

        return torch.cat(blocks, dim=-1)

    def is_symmetric(self) -> bool:
        """True when forward(a, b) == forward(b, a) by construction."""
        return self.mode == "symmetric"
