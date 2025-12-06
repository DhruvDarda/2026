---
layout: distill
title: "The Decoupling Hypothesis: Attempting Subject-Invariant EEG Representation Learning via Auxiliary Injection"
description: "We explore several ideas for learning subject-invariant EEG representations for reaction time and psychopathology prediction using only 2-second windows in the NeurIPS 2025 EEG Challenge. The core of our approach is the Decoupling Hypothesis: an autoencoder framework where we attempt to disentangle subject-specific artifacts and long-term temporal trends (such as fatigue) from the neural signal by explicitly injecting 'nuisance' variables (like demographics and sequence position) into the decoder. This method aimed to force a purely convolutional encoder to learn slow, sequential features without relying on computationally expensive Recurrent or Attention mechanisms. This blog discusses the ideas that seemed promising but ultimately did not work as intended—and why." 
date: 2026-02-01
future: true
htmlwidgets: true
hidden: false

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

authors:
  - name: [Anonymous]
    url: "https://www.Anonymous.com/"
    affiliations:
      name: [Anonymous]

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

Decoding the human brain is difficult; decoding it from short, noisy snippets of electrical activity is even harder.

We recently participated in the NeurIPS 2025 EEG Benchmark Competition <d-cite key="neurips2025eeg"></d-cite>, a rigorous challenge designed to test the limits of EEG representation learning. The goal was to predict behavioral metrics (Reaction Times) and clinical diagnostics (Psychopathology Factors) using only 2-second windows of raw EEG signals.

Out of many participating teams, our experimental architecture achieved:

* **Rank 54** in Challenge 1 (Reaction Time Prediction)
* **Rank 16** in Challenge 2 (Psychopathology Prediction)

While we didn't take the top spot, our approach attempted to solve a fundamental problem in neuroscience: **Subject Invariance**. A model trained on raw signals often overfits to subject-specific artifacts (like skull thickness or hair density) rather than learning the underlying neural dynamics required for downstream tasks.

The core challenge with short, 2-second windows is capturing slow morphological trends in the EEG. While a 200-sample window (at 100 Hz) easily captures higher frequencies like Beta (13-30 Hz) or Alpha (8-12 Hz), it fundamentally limits the resolution of slow waves such as deep sleep Delta waves (~0.5–2 Hz).

{% include figure.liquid path="assets/img/2026-04-01-Subject-Invariant-EEG/brainwaves.png" class="img-fluid" caption="Figure 1: Visualization of different types of brainwaves, which are electrical pulses in the brain that communicate information, categorized by their frequency. Each wave type is depicted with its characteristic pattern, and a timeline at the bottom provides a scale for one second." %}
<d-cite key="neurips2025eeg"></d-cite>

Typically, capturing these long-range temporal dependencies requires explicit sequential models like Recurrent Neural Networks (RNNs) or Attention mechanisms.

In this post, we discuss a "cool idea that didn't quite work out": Forcing representation invariance and long-term trend learning by injecting "nuisance" variables (demographics, sequence position) only at the decoder stage. Our hypothesis was: If the decoder is explicitly given the window's temporal position within the overall sequence, the CNN Encoder might be forced to extract and encode the slow, sequential features needed for a proper reconstruction, thus effectively bypassing the need for computationally expensive RNNs to learn temporal context.

This attempt to decouple subject-specific noise (like age) and long-range sequential trends (like fatigue) from the pure neural dynamics forms the backbone of our Decoupling Hypothesis.

## The HBN Dataset & Challenge

The competition utilized the Healthy Brain Network (HBN) dataset <d-cite key="alexander2017healthy"></d-cite>, comprising recordings from over 3,000 participants.

The data spans six distinct cognitive tasks, categorized into Passive and Active states:

1.  **Passive Tasks (Sensory Processing):** The subject simply perceives stimuli.
    * *Resting State (RS):* Measuring baseline activity (eyes open/closed).
    * *Surround Suppression (SuS):* Visual processing of flashing disks.
    * *Movie Watching (MW):* Emotional and narrative processing (clips from *Despicable Me*, etc.).
2.  **Active Tasks (Cognitive Load + Motor Response):** The subject must think and react.
    * *Contrast Change Detection (CCD):* Attention vigilance.
    * *Sequence Learning (SL):* Working memory and pattern recognition.
    * *Symbol Search (SyS):* High-speed processing and executive function.

The tasks are very well explained in detail on the [competition page](https://eeg2025.github.io/data/).
Different tasks activate different brain regions; we hoped to exploit this diversity.

The competition required predicting:

- **Reaction Times** (RT)
- **Four Psychopathology Factors**

from a **2-second EEG window**, sampled at **100 Hz**, across **129 channels**, where the 129th channel is a reference channel.

**No preprocessing was allowed** — filtering, artifact removal, normalization, etc., had to be learned implicitly inside the model.

Thus the core challenge was:

> **Can we learn subject-invariant, task-invariant EEG embeddings from only 2 seconds of signal?**

## The Hypothesis

To learn robust representations, standard approaches include Masked Autoencoders (MAE) <d-cite key="he2022masked"></d-cite> or Contrastive Learning (SimCLR) <d-cite key="chen2020simple"></d-cite>. However, given the massive inter-subject variability and the varying, often irregular, lengths of the six different tasks in the HBN dataset, we hypothesized that the sequence position within a task is a strong and inconsistent bias we must account for.

The traditional approach to handling sequence-dependent biases and long-range dependencies (like fatigue) is to use Recurrent Neural Networks (RNNs) or Transformer-based Attention mechanisms. While Attention mechanisms are powerful, they are highly compute-intensive, a significant limitation given our restricted budget. Furthermore, the non-uniformity of task lengths made designing a single, robust RNN architecture challenging.

**Our Core Idea:** We proposed a computationally lighter solution: Use an Autoencoder where the Encoder $E(x)$ sees only the EEG, but the Decoder $D(z, a)$ sees the latent $z$ *plus* an auxiliary embedding $a$ containing demographics, task information and sequence position.

**The Crux of the Experiment**: By manually constructing and passing the sequence position $a$ to the decoder, we aimed to test whether this could *implicitly force the CNN Encoder to learn the required slow, long-term morphological features* that an RNN would typically capture, all while avoiding the high computational cost of a full attention model.

By applying dropout to $z$ but *not* to $a$ during training, we create an information bottleneck that incentivizes the decoder to rely on $a$ for static/positional information, forcing $z$ to encode only the **residual neural dynamics**—the pure brain activity we actually care about.

This led to our full hypothesis: Learning subject-invariant, task-invariant EEG embeddings requires incorporating:

1.  **Task-Level Cognitive Context:** A 2-second window of a "sad movie" looks different from a 2-second window of a "math problem." Without context, the model struggles to interpret the waveforms.
2.  **Recording-Specific Demographics:** A 5-year-old's brain waves are higher amplitude and slower than a 20-year-old's. If the model doesn't account for age, it might mistake "youth" for "pathology."
3.  **Manual Long-Range Position Information:** A 2-second window at the *start* of a task (fresh) is different from one at the *end* (fatigued). To substitute for computationally expensive RNNs and force the CNN encoder to learn slow morphological trends.


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
where \( $ A, B \in \mathbb{R}^{B \times T \times d} $ \) are flattened feature maps.


{% include figure.liquid path="assets/img/2026-04-01-Subject-Invariant-EEG/model_architecture.png" class="img-fluid" caption="Figure 2: The proposed architecture. Note the Auxiliary Encoder injecting demographics and sequence position directly into the latent space before decoding." %}

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

To introduce **spatial awareness**, we scale the latent representations based on the physical distance of electrodes. Let $D_{norm} \in \mathbb{R}^{C}$ be the normalized distance of electrodes from a center point. We apply an inverse scaling:


$$
Z'_{c} = Z_{c} \cdot \frac{1}{D_{norm, c} + \epsilon}
$$

This acts as a soft attention mechanism, upweighting signals from sparser regions or distinct lobes before they enter the dense layers of the decoder.

{% include figure.liquid path="assets/img/2026-04-01-Subject-Invariant-EEG/before-and-after_visualization_soft_attention_mechanism.png" class="img-fluid" caption="Figure 3: Stabilization effect  of the soft attention mechanism using spatial information; before-and-after visualizations." %}

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

{% include figure.liquid path="assets/img/2026-04-01-Subject-Invariant-EEG/mtl_head.png" class="img-fluid" caption="Figure 4: Individual head for each task for predicting those pseudo labels." %}

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

We entered the competition late (4 weeks prior to deadline), limiting our ability to tune hyperparameters. However, the results provided a clear picture of what works and what doesn't.

### Performance


{% include figure.liquid path="assets/img/2026-04-01-Subject-Invariant-EEG/Challenge1_score.png" class="img-fluid" caption="Figure 5: Our score and comparison to other near by scores in Challenge 1" %}

{% include figure.liquid path="assets/img/2026-04-01-Subject-Invariant-EEG/Challenge2_score.png" class="img-fluid" caption="Figure 6: Our score and comparison to other near by scores in Challenge 2" %}


| Metric | Our Score | Top scores | Rank |
| :--- | :--- | :--- | :--- |
| **Challenge 1 (Reaction Time)** | *0.95961* | *0.88668* | **54** |
| **Challenge 2 (Psychopathology)** | *0.99786* | *0.97843* | **16** |


### What didn't work (and why)

1.  **Manual Positional Encoding was too simple.** We assumed brain fatigue or habituation scales linearly or exponentially over a task. In reality, attention fluctuates in complex, non-linear waves. Our "Linear Ramp" auxiliary input likely confused the decoder more than it helped, as the brain states didn't align perfectly with simple time indices.

2.  **Demographics are harder to disentangle.**
    We hoped the auxiliary injection would strip age-related amplitude differences from the latent space. However, age impacts not just amplitude, but frequency coupling and signal complexity. A simple concatenation in the decoder wasn't expressive enough to capture these non-linear relationships, meaning the encoder still retained (and overfit to) demographic shifts.

3.  **Orthogonality constraints were too rigid.**
    Forcing the two CNN branches to be orthogonal ($\mathcal{L}_{ortho}$) was intended to encourage diversity. In practice, EEG features are highly correlated. By penalizing correlation, we may have forced the model to discard useful, albeit redundant, information.

### Future Steps

The challenge highlighted the need for **foundation models for EEG** that integrate multi-timescale structure and spatial geometry more organically.

Moving forward, we plan to:
1.  **Benchmark on the Test Set:** Once the full labels are released, we will verify if the spatial scaling offers robustness on out-of-distribution subjects.
2.  **Replace Manual Auxiliaries:** Instead of manually coding positions, we are exploring **Learnable Time Embeddings** (like in Transformers) that can adapt to the non-linear fatigue patterns of the brain.
3.  **Graph Neural Networks:** Replace the distance-based scaling with an explicit Graph Attention Network (GAT) to model the electrode topology dynamically.

# References

The full BibTeX is included in `2026-04-27-Subject-Invariant-EEG.bib`.

---