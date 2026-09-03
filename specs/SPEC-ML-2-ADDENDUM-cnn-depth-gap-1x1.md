# SPEC-ML-2-ADDENDUM: CNN depth — channels between layers, receptive fields, GAP, and 1×1 convs

**Status:** approved
**Subject:** Machine Learning
**Section:** Theory (amends `02-machine-learning/01-theory/02-architectures.md`, SPEC-ML-2)
**Routing:** writer=Sonnet 4.6 · research=Haiku (light) · review=Sonnet (fresh) · architect=Opus 4.8

## Why this addendum
ML-2 already builds convolution from first principles (image as grid → global vs local transform →
convolution as a local weighted sum → output size → hand-set Sobel kernel → "from nine hand-picked
numbers to learned layers", `§2`). What it does **not** yet explain — and the owner asked for — is how
one conv layer *connects to the layer behind it*, plus two building blocks every modern CNN uses:
**Global Average Pooling** and **1×1 convolutions**. These slot in as new subsections at the end of
`§2` (after "From nine hand-picked numbers to a hundred learned layers", ~line 388), before `§3`
(Sequences). Do not disturb the existing kernel material — extend it.

## What to add (new subsections in §2)
1. **Channels — how layer L connects to layer L−1.** The missing link. A kernel is not 2-D once you
   have channels: a conv layer with `C_in` input channels and `C_out` output filters uses filters of
   shape `(C_out, C_in, kH, kW)`. Each output filter spans *all* `C_in` input channels and sums across
   them → one output feature map; stack `C_out` of them → the next layer's channels. Make it concrete
   with real small numbers (e.g. an RGB `3→16` conv, then `16→32`), show the parameter count
   arithmetic, and give the Java analogy (each layer's output *type* is the next layer's input *type*;
   channels are the "width" of the signal flowing through). A Mermaid diagram of the tensor shapes
   flowing layer to layer.
2. **Receptive field.** Why stacking small (3×3) kernels grows the region of the input each deep
   neuron "sees" — the receptive field grows with depth — and why that beats one big kernel (fewer
   params, more nonlinearity). Show the receptive-field growth with real numbers over 2–3 layers,
   with a small diagram.
3. **Global Average Pooling (GAP).** The problem it solves: a flatten+dense head ties the model to one
   input size and adds huge parameter counts. GAP averages each feature map to a single number →
   a `C`-length vector, no parameters, size-agnostic. Show the shape change `(C,H,W) → (C,)` with real
   numbers, contrast the parameter count vs a dense head, and note it's what lets a CNN accept variable
   input sizes and feed a `C→num_classes` linear layer. (Origin: Network-in-Network, Lin et al. 2013.)
4. **1×1 convolutions.** What a kernel that looks at *one pixel across all channels* actually does:
   it's a per-pixel linear mix of channels — a cheap way to change channel count (`C_in→C_out`,
   dimensionality reduction/expansion) and add a nonlinearity, with almost no spatial cost. Real
   numbers: `256→64` 1×1 conv param count vs a 3×3 doing the same. Note its role as the cheap
   "bottleneck" in Inception/ResNet. (Same NiN origin; popularised by GoogLeNet/ResNet.)

## Constraints
- **Extend, don't rewrite.** Preserve the existing prose and every existing code block byte-for-byte;
  add new subsections + diagrams. Any new snippet must be runnable (or fenced pseudocode) and pass the
  snippet gate. Keep the house style (problem-first, real numbers, "why we do it this way", Java
  analogy, plain language before notation).
- Visuals are GitHub-native Mermaid + LaTeX. Escape specials inside `\text{…}`; quote Mermaid labels
  containing parentheses.

## Claims to ground (Haiku — light)
- [ ] Confirm the shape/param arithmetic conventions are stated correctly (PyTorch `nn.Conv2d` weight
      shape `(out_channels, in_channels, kH, kW)`) — cite the PyTorch docs + date.
- [ ] Attribute GAP and 1×1 convs to Network-in-Network (Lin, Chen, Yan, 2013) — confirm the citation;
      confirm GoogLeNet/Inception (Szegedy et al. 2014/2015) and ResNet (He et al. 2015) use 1×1
      bottlenecks. Cite, don't assert from memory.

## Acceptance criteria
- [ ] AC1 — all four topics (channel connection, receptive field, GAP, 1×1 conv) added to §2 with
      real-number worked arithmetic and at least one diagram each (or a shared shapes diagram).
- [ ] AC2 — existing content unchanged; any new snippet compiles (`check_snippets.py`).
- [ ] AC3 — the two grounding items cited (NOTE or inline authoritative link + date).
- [ ] AC4 — renders on GitHub (`check_markdown_render.py` pass; diagrams/formulas eyeballed).

## Gates
Exit: ACs satisfied; fresh-Sonnet review; architect merge. (See `docs/definition-of-done.md`.)
