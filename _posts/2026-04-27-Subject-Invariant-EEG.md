---
layout: distill
title: "The Decoupling Hypothesis: Attempting Subject-Invariant EEG Representation Learning via Auxiliary Injection"
description: "We explore several ideas for learning subject-invariant EEG representations for reaction time and psychopathology prediction using only 2-second windows in the NeurIPS 2025 EEG Challenge. This blog discusses the ideas that seemed promising but ultimately did not work as intended — and why." 
date: 2026-02-01
future: true
htmlwidgets: true
hidden: false

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

authors:
  - name: [Dhruv Darda]
    url: "https://www.infocusp.com/"
    affiliations:
      name: [Infocusp Innovations]

bibliography: 2026-05-01-eeg-invariance.bib

toc:
  - name: Introduction
  - name: The HBN Dataset & Challenge
  - name: The Hypothesis
  - name: Methodology
    subsections:
      - name: The Encoder Design
      - name: Auxiliary Injection & Disentanglement
      - name: Spatial Scaling via Adjacency
      - name: Multi-Task Pseudo-Labeling
  - name: Loss Landscape
  - name: Results & Discussion
  - name: References
---

## Introduction

Electroencephalography (EEG) data is notoriously high-dimensional, noisy, and subject to massive inter-person variability. In the context of the [NeurIPS 2024 EEG Benchmark Competition](https://eeg2025.github.io/) <d-cite key="neurips2024eeg"></d-cite>, the goal was to predict reaction times and psychopathology factors using only 2-second windows of raw EEG signals.

The core challenge in such a task is **subject invariance**. A model trained on raw signals often overfits to subject-specific artifacts rather than learning the underlying neural dynamics required for downstream tasks like psychopathology prediction.

In this post, we discuss a "cool idea that didn't quite work out" (or at least, didn't reach State-of-the-Art in the limited compute window): **Forcing representation invariance by injecting "nuisance" variables (demographics, sequence position) only at the decoder stage.** The intuition was that if the decoder already has access to the subject's age, sex, and the temporal position of the window, the encoder is forced to strip that information from the latent space, leaving behind only the pure neural dynamics.

## The HBN Dataset & Challenge

The competition utilized the Healthy Brain Network (HBN) dataset <d-cite key="alexander2017healthy"></d-cite>, comprising recordings from over 3,000 participants.

The data spans six distinct cognitive tasks, categorized into Passive and Active states:

**Passive Tasks:**
* **Resting State (RS):** Eyes open/closed fixations.
* **Surround Suppression (SuS):** Flashing peripheral disks.
* **Movie Watching (MW):** Clips from *Despicable Me*, *The Present*, etc.

**Active Tasks:**
* **Contrast Change Detection (CCD):** Identifying contrast changes in flickering gratings.
* **Sequence Learning (SL):** Reproducing sequences of flashed circles.
* **Symbol Search (SyS):** Processing speed tasks.

Different tasks activate different brain regions; we hoped to exploit this diversity.

The competition required predicting:

- **Reaction Times** (RT)
- **Four Psychopathology Factors**

from a **2-second EEG window**, sampled at **100 Hz**, across **129 channels**, where the 129th channel is a reference channel.

**No preprocessing was allowed** — filtering, artifact removal, normalization, etc., had to be learned implicitly inside the model.

Thus the core challenge was:

> **Can we learn subject-invariant, task-invariant EEG embeddings from only 2 seconds of signal?**

We hypothesized **yes**, but only if representations incorporate:

1. task-level cognitive context,  
2. recording-specific demographic metadata,  
3. long-range position information missing from the short window.

## The Hypothesis

To learn robust representations, standard approaches include Masked Autoencoders (MAE) <d-cite key="he2022masked"></d-cite> or Contrastive Learning (SimCLR) <d-cite key="chen2020simple"></d-cite>. However, given the varying lengths of tasks, we hypothesized that the sequence position within a task is a strong bias we want to remove from the learned features.

**Our Core Idea:** Use an Autoencoder where the Encoder $E(x)$ sees only the EEG, but the Decoder $D(z, a)$ sees the latent $z$ *plus* an auxiliary embedding $a$ containing demographics and sequence position.

By applying dropout to $z$ but *not* to $a$ during training, we create an information bottleneck that incentivizes the decoder to rely on $a$ for static/positional information, forcing $z$ to encode only the dynamic, subject-invariant brain activity.

## Methodology

### Architectural Motivation

A single 2-second window is insufficient to predict behavioral or clinical outcomes.  
Thus, the architecture was designed to:

- **inject longer-term position information into the decoder, not the encoder**  
- **encourage invariance to demographics via an auxiliary encoder**
- **predict task-specific pseudo-labels to force richer latent structure**

This led to a hybrid architecture consisting of:

- A **multi-branch CNN encoder**  
- A **latent bottleneck** regularized with dropout  
- An **auxiliary encoder** (demographics + positional encoding)  
- A **decoder** reconstructing EEG  
- Six **task-specific MTL heads** predicting pseudo-labels  
- **Contrastive loss** across tasks  
- **Orthogonality constraints** on feature subspaces  
- **Electrode distance re-weighting**

### The Encoder Design

The encoder takes the raw EEG input $(B, 129, 200)$. We first broadcast the reference channel and stack it, resulting in $(B, 128, 200, 2)$.

$$
X' \in \mathbb{R}^{B \times 128 \times 200 \times 2}
$$


We employ a multi-scale CNN approach to capture temporal features at different frequencies:
1.  **Multi-Kernel Convolution:** Three parallel CNN branches with kernel sizes of $[1, 15, 45]$ (corresponding to immediate, alpha/beta, and delta/theta band scales).
$$
H_0 = \text{Concat}(H_{k=1}, H_{k=15}, H_{k=45})
$$
2.  **Orthogonal Feature Extraction:** We utilize two distinct convolutional branches with differing dilation rates $(1, 2, 4, 16)$ to capture long-range dependencies.
3.  **Orthogonality Constraint:** To ensure these branches learn distinct features, we minimize the Frobenius norm of their product.

$$
\mathcal{L}_{\text{ortho}}
= \left\| A^\top B \right\|_F^2
$$
where \( A, B \in \mathbb{R}^{B \times T \times d} \) are flattened feature maps.

{% include figure.liquid path="assets/img/2026-02-01-eeg-invariance/model_architecture.png" class="img-fluid" caption="Figure 1: The proposed architecture. Note the Auxiliary Encoder injecting demographics and sequence position directly into the latent space before decoding." %}

### Auxiliary Injection & Disentanglement

The decoder must know which part of the original EEG sequence the 2-second window corresponds to.

We manually constructed:

- **Linear ramp**
- **Exponential decay**
- **Sinusoidal positional encoding**

and concatenated them with demographic and task information:

$$
z_{\text{aux}} = f_{\theta}([d_{\text{demo}}, p_{\text{pos}}, t_{\text{task}}])
$$

where \( $ f_{\theta} $ \) is a small MLP producing a **32-D vector**.

This was the core of our "Cool Idea." We postulated that if we fed demographic data (age, sex, handedness), task information (one hot encoding representing the task) and the positional encoding (sequence index) into the decoder but not the encoder, we can force the encoder to focus on subject independent signals and we thought that the CNNs would be able to extract sequence based features without the use of RNNs.

$$
Encoder: z_{eeg} = E(x_{raw})
$$
$$
Aux Encoder: z_{aux} = E_{aux}(x_{demo}, x_{pos})
$$
$$
Decoder: \hat{x} = D(Concat(z_{eeg}, z_{aux}))
$$

**This auxiliary embedding was appended to the encoder without dropout**, forcing the model to use auxiliary signals for positioning. By applying dropout to $z_{eeg}$ but not to $z_{aux}$, we forced the Decoder to rely on $z_{aux}$ for the "easy" reconstruction details (like general signal amplitude related to age), thereby forcing the Encoder to focus purely on the residual neurological signals that are independent of demographics.

### Spatial Scaling via Adjacency

EEG electrodes have a physical geometry. Neighboring electrodes often capture redundant signals from the same brain lobe. To introduce spatial awareness, we scale the latent representations based on the physical distance of electrodes.

Let $Z \in \mathbb{R}^{B \times C \times F}$ be the latent representation and $D_{norm} \in \mathbb{R}^{C}$ be the normalized distance vector of electrodes from a center point (or inter-electrode adjacency). We apply an inverse scaling:

$$
Z'_{c} = Z_{c} \cdot \frac{1}{D_{norm, c} + \epsilon}
$$

This acts as a soft attention mechanism, upweighting signals from sparser regions or distinct lobes before they enter the dense layers of the decoder.

### Multi-Task Pseudo-Labeling

Each task contains additional metadata such as correctness, contrast values, movie identifiers, etc. To further regularize the latent space, we utilized these known task structures to generate "pseudo-labels." Even though the competition goal was reaction time, predicting the *state* of the experiment forces the model to recognize which neural circuits are active.

We attached 6 task-specific heads to the encoder output.

| Task | Pseudo-Labels Generated |
| :--- | :--- |
| **Contrast Change** | `Contrast_left_side` (binary), `Contrast_correct`, `Reaction_time` |
| **Symbol Search** | `Contrast_correct` (binary) |
| **Surround Supp.** | `Background_type`, `Foreground_contrast`, `Stimulus_cond` |
| **Seq. Learning** | `Correct_count`, `Target_count`, `Learning_phase` (binary) |
| **Resting State** | `Eyes_closed` (binary - derived from onset times) |
| **Movies** | One-hot encoded movie segment identifiers |

## Loss Landscape

The training objective was a complex balancing act of four distinct loss functions. We define the total loss $\mathcal{L}_{total}$ as:

$$
\mathcal{L}_{total} = \lambda_{recon}\mathcal{L}_{recon} + \lambda_{scl}\mathcal{L}_{scl} + \lambda_{mtl}\mathcal{L}_{mtl} + \lambda_{ortho}\mathcal{L}_{ortho}
$$

Where:
* $\mathcal{L}_{recon}$: MSE Reconstruction loss of the signal.
* $\mathcal{L}_{scl}$: Supervised Contrastive Loss <d-cite key="khosla2020supervised"></d-cite> to cluster embeddings of the same task type.
* $\mathcal{L}_{mtl}$: Multi-Task Learning loss (masked sum of BCE/MSE/CrossEntropy for pseudo-labels).
* $\mathcal{L}_{ortho}$: Orthogonality loss to force diverse feature extraction between the dilation branches.

Given two feature matrices $A$ and $B$ from the parallel branches, the orthogonal loss is defined as:

$$
\mathcal{L}_{ortho} = || A^T B ||_F^2 \approx 0
$$

The weighting was crucial. We used $\lambda_{recon}=1.0$, $\lambda_{mtl}=1.0$, but kept contrastive loss low ($\lambda_{scl}=0.001$) to prevent collapse, and $\lambda_{ortho}=0.1$.

## Results & Retrospective

We entered the competition late (4 weeks prior to deadline). Despite this, our approach achieved:
* **Rank 54** in Challenge 1
* **Rank 16** in Challenge 2

**What didn't work (and why):**
While many of our “clever tricks” did not work out, the process revealed useful lessons:

- Short-window EEG modeling requires **explicit temporal context modeling**, not manually constructed positional encodings.
- Demographic confounding remains difficult without strong adversarial training.
- Orthogonal subspace learning may require more structured architectural constraints.
- Spatial regularization should ideally incorporate **graph neural networks** or **eigenmodes** of the electrode layout.

The challenge highlighted the need for **foundation models for EEG** that integrate:

- multi-timescale structure  
- spatial geometry  
- subject/task metadata  
- noise-aware priors  

**Future Steps:**
We plan to benchmark this architecture on the test set (post-test dataset release) to verify if the spatial scaling and auxiliary injection offer robustness benefits on out-of-distribution subjects, even if they didn't top the leaderboard for the specific competition metrics.

# References

The full BibTeX is included in `2026-ICLR-EEG-AE.bib`.

---