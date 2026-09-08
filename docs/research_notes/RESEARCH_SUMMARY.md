# Adaptive Privacy and Noise for Federated Learning in IoT (QPrivIot-FL)

## Problem Statement
Internet of Things (IoT) devices generate distributed datasets critical for machine learning across healthcare, smart infrastructure, and industrial monitoring. **Federated Learning (FL)** enables model training on this distributed data without centralization, yet standard protocols remain vulnerable to gradient inversion and membership inference attacks. **Differential Privacy (DP)** provides mathematical safeguards, but traditional "one-size-fits-all" noise (e.g., DP-SGD) fails in heterogeneous IoT environments, leading to extreme utility loss on well-resourced devices or insufficient protection for vulnerable ones.

## Proposed Framework: Adaptive Client-Level DP
We propose a client-level differential privacy framework tailored for IoT. Our framework dynamically allocates noise based on:
1.  **Resource-Aware Scaling:** Noise scales inversely to a **Device Readiness Score** $(R_i)$, ensuring low-resource devices contribute in proportion to their reliability.
2.  **Layer-Wise Sensitivity:** Server tracks **EMA of gradient norms** to identify informative layers, prioritizing them for higher privacy protection.
3.  **Temporal Adaptation:** As the model reaches convergence (measured by **Coefficient of Variation**), the framework reduces the learning rate and noise to maximize final utility within the **Renyi Differential Privacy (RDP)** budget.

---

## Technical Algorithms

### Algorithm 1: Server-Side Orchestration (AdaPriv-Server)
**Goal:** Orchestrate rounds, manage device heterogeneity, and enforce global privacy via RDP.
1.  **Initialize** global model $\theta_0$ and sensitivity tracker $S$.
2.  **For** each round $t = 1 \dots T$:
    a.  **Calculate** round-level budget $\epsilon_t = \text{decay}(t, T) \times \frac{\epsilon}{\text{calibration\_factor}}$.
    b.  **Assess Convergence** using Coefficient of Variation over recent loss history.
    c.  **Sample Clients** $\mathbb{K}_t$: Prioritize high-resource clients ($R_i$) while maintaining random diversity.
    d.  **Broadcast** $\{\theta_{t-1}, \epsilon_t, \text{Sensitivities } s_{j}\}$ to selected clients.
    e.  **On Receive** updates $\Delta_i$:
        - If **SecAgg** enabled: Sum masked updates and dequantize.
        - Else: Apply **Resource-Weighted aggregation** based on $R_i$.
    f.  **Update Global Model** $\theta_t = \theta_{t-1} + \Delta_{avg}$.
    g.  **Privacy Accounting:** Add RDP cost $(\sigma_{avg}, q)$ to global budget. If $\epsilon_{spent} > \epsilon$, halt training.

### Algorithm 2: Resource-Aware Client Perturbation (AdaPriv-Client)
**Goal:** Train locally while adapting noise to hardware constraints and layer sensitivity.
1.  **Profile Device** to establish hardware Readiness Score $R_i$.
2.  **Fine-Tune Learning:** If convergence $> \text{threshold}$, reduce learning rate $\eta$ and noise.
3.  **Compute Update:** Train $\theta$ on local data $\mathcal{D}_i$ to get delta $\Delta_{raw}$.
4.  **Layer-wise Perturbation (DP-SGD):** For each layer $j$:
    a.  **Clip:** $\tilde{\Delta}_{i,j} = \Delta_{i,j}^{raw} \cdot \min(1, \frac{C_j}{\|\Delta_{i,j}^{raw}\|_2})$
    b.  **Noise:** $\Delta_{i,j} = \tilde{\Delta}_{i,j} + \mathcal{N}(0, (\frac{\sigma_j \cdot C_j}{R_i \cdot \sqrt{n_i}})^2)$
5.  **Secure Transmission:** Quantize $\Delta_{i}$ and apply zero-sum random masks if SecAgg is active.

### Algorithm 3: Sensitivity-Based Noise Allocation (AdaPriv-Brain)
**Goal:** Distribute privacy budget across layers based on information density and impact.
1.  **Impact Tracking:** Server maintains EMA of second moments of gradients to estimate layer influence.
2.  **Sensitivity Scoring:** $s_j = (1-\alpha) \cdot \text{normalized\_std}(\text{norms}) + \alpha \cdot \text{normalized\_moment}$.
3.  **Normalization:** Ensure $\text{mean}(s_{j \in S}) = 1.0$ for stable relative allocation.
4.  **Map to Parameters:**
    -   **Adaptive Clip Norm:** $C_j = C_{base} / \text{clamp}(s_j, 0.5, 2.0)$.
    -   **Adaptive Noise Multiplier:** $\sigma_j = \sigma_{base} \cdot s_j$.

---

## Algorithm-to-Code Mapping

| Algorithm | Step | Theoretical Action | Code Reference (File:Line) |
| :--- | :--- | :--- | :--- |
| **Server** | 1 | Global Initialization | `server_app.py`: 27-57 |
| | 2a | Round Epsilon ($\epsilon_t$) Calculation | `server_app.py`: 59-81 |
| | 2b | Convergence Assessment (CV) | `server_app.py`: 126-132 |
| | 2c | Resource-Aware Client Sampling | `server_app.py`: 88-112 |
| | 2e | SecAgg / Weighted Aggregation | `server_app.py`: 173-247 |
| | 2f | RDP Accounting & Budget Control | `server_app.py`: 271-282 |
| **Client** | 1 | Device Readiness Profiling ($R_i$) | `client_app.py`: 30-32 |
| | 2 | Convergence-Aware LR Tuning | `client_app.py`: 72-78 |
| | 4a | Adaptive Layer-wise Clipping | `client_app.py`: 95-125, 177-179 |
| | 4b | Readiness-Scaled Noise Injection | `client_app.py`: 109-110, 127, 305 |
| | 5 | Quantization & Zero-Sum Masking | `client_app.py`: 210-230 |
| **Brain** | 1 | Gradient Moment Tracking (EMA) | `privacy_utils.py`: 161-173 |
| | 2 | Sensitivity Score Computation ($s_j$) | `privacy_utils.py`: 181-227 |
| | 4 | Adaptive Clip & Multiplier Allocation | `privacy_utils.py`: 230-272 |
