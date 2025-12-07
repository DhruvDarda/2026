---
layout: distill
title: "The Decoupling Hypothesis: Attempting Subject-Invariant EEG Representation Learning via Auxiliary Injection"
description: "We explore several ideas for learning subject-invariant EEG representations for reaction time and psychopathology prediction using only 2-second windows in the NeurIPS 2025 EEG Challenge. The core of our approach is the Decoupling Hypothesis: an autoencoder framework where we attempt to disentangle subject-specific artifacts and long-term temporal trends (such as fatigue) from the neural signal by explicitly injecting 'nuisance' variables (like demographics and sequence position) into the decoder. This method aimed to force a purely convolutional encoder to learn slow, sequential features without relying on computationally expensive Recurrent or Attention mechanisms. This blog discusses the ideas that seemed promising but ultimately did not work as intended—and why." 
date: 2026-04-27
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

bibliography: 2026-04-27-subject-invariant-eeg.bib

toc:
  - name: Introduction
  - name: The HBN Dataset & Challenge
  - name: Existing Self-Supervised Pretraining Approaches for EEG and Our Hypothesis
  - name: Methodology
    subsections:
      - name: Architectural Motivation
      - name: The Encoder Design
      - name: Auxiliary Injection & Disentanglement
      - name: Spatial Scaling via Adjacency
      - name: Multi-Task Pseudo-Labeling
  - name: Loss Landscape
  - name: Results & Retrospective
    subsections:
      - name: Performance
      - name: What didn't work (and why)
      - name: Future Steps
---

## Introduction

Decoding the human brain is difficult; decoding it from short, noisy snippets of electrical activity is even harder.

We recently participated in the NeurIPS 2025 EEG Benchmark Competition <d-cite key="aristimunha2025eeg"></d-cite>, a rigorous challenge designed to test the limits of EEG representation learning. The goal was to predict behavioral metrics (Reaction Times) and clinical diagnostics (Psychopathology Factors) using only 2-second windows of raw EEG signals.

Out of many participating teams, our experimental architecture achieved:

* **Rank 54** in Challenge 1 (Reaction Time Prediction)
* **Rank 16** in Challenge 2 (Psychopathology Prediction)

While we didn't take the top spot, our approach attempted to solve a fundamental problem in neuroscience: **Subject Invariance**. A model trained on raw signals often overfits to subject-specific artifacts (like skull thickness or hair density) rather than learning the underlying neural dynamics required for downstream tasks.

The core challenge with short, 2-second windows is capturing slow morphological trends in the EEG. While a 200-sample window (at 100 Hz) easily captures higher frequencies like Beta (13-30 Hz) or Alpha (8-12 Hz), it fundamentally limits the resolution of slow waves such as deep sleep Delta waves (~0.5–2 Hz).

{% include figure.liquid path="assets/img/2026-04-27-subject-invariant-eeg/brainwaves.png" class="img-fluid h-75" caption="Figure 0: Visualization of different types of brainwaves, which are electrical pulses in the brain that communicate information, categorized by their frequency. Each wave type is depicted with its characteristic pattern, and a timeline at the bottom provides a scale for one second." %}

Typically, capturing these long-range temporal dependencies requires explicit sequential models like Recurrent Neural Networks (RNNs) or Attention mechanisms.

In this post, we discuss a "cool idea that didn't quite work out": Forcing representation invariance and long-term trend learning by injecting "nuisance" variables (demographics, sequence position) only at the decoder stage. Our hypothesis was: If the decoder is explicitly given the window's temporal position within the overall sequence, the CNN Encoder might be forced to extract and encode the slow, sequential features needed for a proper reconstruction, thus effectively bypassing the need for computationally expensive RNNs to learn temporal context.

This attempt to decouple subject-specific noise (like age) and long-range sequential trends (like fatigue) from the pure neural dynamics forms the backbone of our Decoupling Hypothesis.

## The HBN Dataset & Challenge


{% include figure.liquid path="assets/img/2026-04-27-subject-invariant-eeg/eeg_challenge_2025.png" class="img-fluid h-80" caption="Figure 1: HBN-EEG Dataset and Data split. A. EEG is recorded using a 128-channel system during active tasks (i.e., with user input) or passive tasks. B. The psychopathology and demographic factors. C. The dataset split into Train, Test, and Validation." %}<d-cite key="aristimunha2025eeg"></d-cite>


The competition utilized the Healthy Brain Network (HBN) dataset <d-cite key="alexander2017hbn"></d-cite>, comprising recordings from over 3,000 participants.

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

## Existing Self-Supervised Pretraining Approaches for EEG and Our Hypothesis

Self-supervised learning (SSL) has become the standard approach for EEG representation learning, largely due to the scarcity of high-quality labeled datasets and the large inter-subject variability. Surveys on EEG and biomedical SSL <d-cite key="weng2023ssl_eeg_survey"></d-cite>, <d-cite key="ding2023ssl_biomedical_review"></d-cite> group existing methods into a few dominant paradigms.

**Masked reconstruction methods** adapt Masked Autoencoders (MAE) <d-cite key="he2022masked"></d-cite> to EEG by masking temporal segments or channels and reconstructing the missing signal. This strategy underpins several recent EEG foundation models, including EEGPT <d-cite key="eegpt2024arxiv"></d-cite>.

**Contrastive learning** remains the most widely used family of SSL techniques, extending SimCLR <d-cite key="chen2020simple"></d-cite> with EEG-specific augmentations such as channel dropout, temporal jittering, and filtering. Representative approaches include CPC <d-cite key="oord2018cpc"></d-cite>, TS-TCC <d-cite key="tstcc2021"></d-cite>, and CL-TCN <d-cite key="banville2021cltcn"></d-cite>.

**Bootstrap and teacher–student methods** (e.g., BYOL <d-cite key="byol2020"></d-cite> and VICReg <d-cite key="vicreg2022"></d-cite>) have also been adapted to avoid the negatives required in contrastive learning and stabilize training on noisy EEG.

**Clustering-driven and prototype-based SSL** <d-cite key="prototype2021"></d-cite> aim to discover latent neural states by grouping or assigning EEG segments to learned prototypes without labels.

Finally, **large-scale EEG foundation models** such as EEGPT <d-cite key="eegpt2024arxiv"></d-cite> and BC-SSL <d-cite key="bcssl2023"></d-cite> combine multiple SSL paradigms—masked prediction, temporal contrast, clustering, and cross-task invariance—to produce universal EEG encoders.


### Our Hypothesis: Context Injection via Decoupling

The traditional solution to modeling long-range dependencies is to employ Recurrent Neural Networks (RNNs) or Transformers. However, attention mechanisms are compute-intensive, and the non-uniformity of task lengths makes designing a robust RNN architecture difficult.


### Core Idea

We designed an autoencoder where:

- The **encoder** \(E(x)\) receives only the raw 2-second EEG window.  
- The **decoder** \(D(z, a)\) receives both the latent \(z\) and an auxiliary embedding \(a\) containing:
  - demographics,  
  - coarse task identity, and  
  - manually constructed sequence-position features.

The design intentionally applies **dropout to the latent \(z\)** but **not** to the auxiliary input **\(a\)**:

> This forces the decoder to rely on \(a\) for static, slow-varying information (subject factors, task stage), compelling the encoder to encode only the *residual fast neural dynamics*—the part we want to be subject- and task-invariant.


We hypothesized that subject-invariant embeddings require explicitly factoring out three specific biases:

1. **Task-level cognitive context**  
   A 2-second window of a "sad movie" implies different neural states than a "math problem."

2. **Demographic differences**  
   Isolating age-related amplitude shifts (e.g., high-amplitude waves in children) to prevent them from being confounded with pathology.

3. **Long-range sequence position**  
   Explicitly signaling the start vs. end of a task to force the CNN to learn slow morphological trends (like habituation) that usually require RNNs.

Our approach therefore aimed to approximate the benefits of long-range sequence modeling—**without** an RNN/Transformer—by making the decoder responsible for positional and demographic variation, and the encoder responsible for invariant neural representation learning.


## Methodology

### Architectural Motivation

A single 2-second EEG window contains only fast, local neural dynamics, while the behavioral and cognitive variables we aim to predict (reaction time, task state, fatigue, engagement) depend on **slower, longer-range processes** that unfold across minutes. This mismatch creates a fundamental challenge: any encoder that only sees isolated 2-second segments is forced to infer complex context without access to the underlying temporal structure.

The architecture addresses this by explicitly **separating what the encoder should learn (subject-invariant neural dynamics)** from what the decoder must reconstruct (context-dependent variations). This leads to three core design principles:


- **Inject longer-term position information into the decoder, not the encoder**  
- **Encourage invariance to demographics via an auxiliary encoder**
- **Predict task-specific pseudo-labels to force richer latent structure**

This led to a hybrid architecture consisting of:

- **Multi-branch CNN encoder** capturing multi-scale oscillatory features  
- **Latent bottleneck** with strong regularization  
- **Auxiliary encoder** injecting demographics and sequence-position info  
- **Decoder** conditioned on both EEG latent and auxiliary latent  
- **MTL heads** ensuring structured latent supervision  
- **Spatial reweighting** using electrode geometry  
- **Contrastive + orthogonality losses** ensuring disentangled representations

### The Encoder Design

We have the raw EEG input of size $(B, 129, 200)$, where B is the batch size, 129 are the signal-channels (last channel is the reference channel) and 200 is the time dimension of each channel. We split the channel dimension into size 128 and 1, the last reference channel. We then broadcast the reference channel to match the size of the other channels, i.e., 128, and stack the two components onto a new dimension resulting in a tensor of size $(B, 128, 200, 2)$.


$$
X' \in \mathbb{R}^{B \times 128 \times 200 \times 2}
$$


We employ a multi-scale CNN approach to capture temporal features at different frequencies:
1.  **Multi-Kernel Convolution:** Three parallel CNN branches with kernel sizes of $[1, 15, 45]$ (corresponding to immediate, alpha/beta, and delta/theta band scales).
$$
H_0 = \text{Concat}(H_{k=1}, H_{k=15}, H_{k=45})
$$

1.  **Orthogonal Feature Extraction:** We utilize two distinct convolutional branches with differing dilation rates $(1, 2, 4, 16)$ to capture long-range dependencies.
2.  **Orthogonality Constraint:** To ensure these branches learn distinct features, we minimize the Frobenius norm of their product.

$$
\mathcal{L}_{\text{ortho}}
= \left\| H_1^\top H_2 \right\|_F^2
$$

where \( $ H_1, H_2 \in \mathbb{R}^{B \times T \times d} $ \) are flattened feature maps.


{% include figure.liquid path="assets/img/2026-04-27-subject-invariant-eeg/model_architecture.png" class="img-fluid w-80 h-70" caption="Figure 2: The proposed architecture. Note the Auxiliary Encoder injecting demographics and sequence position directly into the latent space before decoding." %}

### Auxiliary Injection & Disentanglement

The decoder must know which part of the original EEG sequence the 2-second window corresponds to.

We manually constructed positional encodings that representing EEG’s temporal structure:

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

{% include figure.liquid path="assets/img/2026-04-27-subject-invariant-eeg/before-and-after_visualization_soft_attention_mechanism.png" class="img-fluid" caption="Figure 3: Stabilization effect  of the soft attention mechanism using spatial information; before-and-after visualizations." %}

### Multi-Task Pseudo-Labeling

Each task contains additional metadata such as correctness, contrast values, movie identifiers, etc. To further regularize the latent space, we utilized these known task structures to generate "pseudo-labels." Even though the competition goal was reaction time, predicting the *state* of the experiment forces the model to recognize which neural circuits are active.

<div class="row">

  <!-- Left column: Image (25%) -->
  <div class="col-sm-3 d-flex align-items-center justify-content-center">
    {% include figure.liquid 
         path="assets/img/2026-04-27-subject-invariant-eeg/mtl_head.png" 
         class="img-fluid" 
         caption="Figure 4: Task-specific prediction head." %}
  </div>


  <!-- Right column: Table (75%) -->
  <div class="col-sm-9">
    <p>We attached 6 task-specific heads to the encoder's latent.</p>

    <table>
      <thead>
        <tr>
          <th>Task</th>
          <th>Pseudo-Labels Generated</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Contrast Change</strong></td>
          <td><code>Contrast_left_side</code> (binary), <code>Contrast_correct</code>, <code>Reaction_time</code></td>
        </tr>
        <tr>
          <td><strong>Symbol Search</strong></td>
          <td><code>Contrast_correct</code> (binary)</td>
        </tr>
        <tr>
          <td><strong>Surround Supp.</strong></td>
          <td><code>Background_type</code>, <code>Foreground_contrast</code>, <code>Stimulus_cond</code></td>
        </tr>
        <tr>
          <td><strong>Seq. Learning</strong></td>
          <td><code>Correct_count</code>, <code>Target_count</code>, <code>Learning_phase</code> (binary)</td>
        </tr>
        <tr>
          <td><strong>Resting State</strong></td>
          <td><code>Eyes_closed</code> (binary - derived from onset times)</td>
        </tr>
        <tr>
          <td><strong>Movies</strong></td>
          <td>One-hot encoded movie segment identifiers</td>
        </tr>
      </tbody>
    </table>
  </div>

</div>


## Loss Landscape

The training objective was a complex balancing act of four distinct loss functions. We define the total loss $\mathcal{L}_{total}$ as:

$$
\mathcal{L}_{total} = \lambda_{recon}\mathcal{L}_{recon} + \lambda_{scl}\mathcal{L}_{scl} + \lambda_{mtl}\mathcal{L}_{mtl} + \lambda_{ortho}\mathcal{L}_{ortho}
$$

Where:
* $\mathcal{L}_{recon}$: MSE Reconstruction loss of the signal.
* $\mathcal{L}_{scl}$: Supervised Contrastive Loss <d-cite key="khosla2020supcon"></d-cite> to cluster embeddings of the same task type.
* $\mathcal{L}_{mtl}$: Multi-Task Learning loss (masked sum of BCE/MSE/CrossEntropy for pseudo-labels).
* $\mathcal{L}_{ortho}$: Orthogonality loss to force diverse feature extraction between the dilation branches.

The weighting was crucial. We used $\lambda_{recon}=1.0$, $\lambda_{mtl}=1.0$, but kept contrastive loss low ($\lambda_{scl}=0.001$) to prevent collapse, and $\lambda_{ortho}=0.1$.

## Results & Retrospective

We entered the competition late (4 weeks prior to deadline), limiting our ability to tune hyperparameters. However, the results provided a clear picture of what works and what doesn't.

### Performance


<div class="row">

  <div class="col-sm-6 d-flex justify-content-center">
    {% include figure.liquid 
         path="assets/img/2026-04-27-subject-invariant-eeg/challenge1_score.png" 
         class="img-fluid" 
         caption="Figure 5: Our score and comparison to near-by entries in Challenge 1." %}
  </div>

  <div class="col-sm-6 d-flex justify-content-center">
    {% include figure.liquid 
         path="assets/img/2026-04-27-subject-invariant-eeg/challenge2_score.png" 
         class="img-fluid" 
         caption="Figure 6: Our score and comparison to near-by entries in Challenge 2." %}
  </div>

</div>


The metric used for both regression challenges 1 and 2 were the normalized root mean square error for the response time prediction and the psychopathology factor prediction.

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
