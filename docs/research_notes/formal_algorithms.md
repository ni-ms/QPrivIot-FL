# Formal Algorithms for AdaPriv-FL

This document provides a formal, step-by-step description of the three core algorithms implemented in the **AdaPriv-FL** framework.

---

### Algorithm 1: Adaptive Global Server Protocol (AdaPriv-Server)
**Goal:** Orchestrate federated rounds, manage device heterogeneity, and enforce global privacy via RDP.

**Input:** Total rounds $T$, target $\epsilon$, model $\theta_0$, fraction $f$
**Output:** Trained global model $\theta_T$

1.  **Initialize** global model $\theta_0$ and sensitivity tracker $S$.
2.  **For** each round $t = 1, \dots, T$:
    a.  **Calculate** round-level privacy budget $\epsilon_t$:
        $$\epsilon_t = \text{decay}(t, T) \times \frac{\epsilon}{\sqrt{T}}$$
    b.  **Assess Convergence** $(\mathcal{C}_t)$ using Coefficient of Variation over recent loss history.
    c.  **Sample Clients** $\mathbb{K}_t$ (Size $m = f \cdot N$):
        *   If $t > 1$: Prioritize clients with high historical **Readiness Scores** ($R_i$).
        *   Else: Randomly sample $m$ clients.
    d.  **Broadcast** $\{\theta_{t-1}, \epsilon_t, \mathcal{C}_t, \text{Sensitivities } s_{j \in S}\}$ to selected clients $\mathbb{K}_t$.
    e.  **On Receive** updates $\{\Delta_i, R_i, \text{metrics}_i\}$ from clients:
        *   Update client resource registry with $R_i$.
    f.  **Aggregate Fit:**
        *   If **SecAgg** is enabled: Sum masked integer updates and dequantize.
        *   If **Resource-Weighted**:
            $$\Delta_{avg} = \frac{\sum_{i \in \mathbb{K}_t} n_i \cdot R_i \cdot \Delta_i}{\sum n_i \cdot R_i}$$
    g.  **Update Global Model:** $\theta_t = \theta_{t-1} + \Delta_{avg}$.
    h.  **Update Sensitivity Tracker:** $S.\text{update}(\text{EMA}(\text{norm}(\Delta_{avg})))$.
    i.  **RDP Accounting:** Add cost $(\text{noise}_{avg}, \text{sampling\_rate})$ to global budget.
    j.  **If** total $\epsilon_{\text{spent}} > \epsilon$: **Halt** training.
3.  **Return** Final Model $\theta_T$.

---

### Algorithm 2: Resource-Aware Client Training & Perturbation (AdaPriv-Client)
**Goal:** Train locally while adapting noise and learning intensity to hardware constraints and layer sensitivity.

**Input:** Global model $\theta$, round budget $\epsilon_t$, convergence $\mathcal{C}_t$, sensitivities $s_j$
**Output:** Perturbed update $\Delta_i$, Readiness Score $R_i$

1.  **Profile Device** to calculate **Readiness Score** $(R_i)$:
    $$R_i = \min(\text{CPU}, \text{Mem}, \text{Batt} \times \alpha_b, \text{Net} \times \alpha_n)$$
2.  **Adaptive Learning Rate** $(\eta)$:
    *   If $\mathcal{C}_t > \text{threshold}$: Set $\eta = \eta \times 0.5$ (Fine-tuning mode).
3.  **Noise Allocation (Algorithm 3):**
    *   Obtain layer-specific noise multipliers $\sigma_j$ and clip bounds $C_j$ using $s_j$.
    *   **Scale Noise by Readiness:** $\sigma_j = \sigma_j \times \text{clamp}(1/R_i, 1.0, 2.0)$.
4.  **Local Training:**
    *   Train $\theta$ on local data $\mathcal{D}_i$ for $E$ epochs to get $\theta'_{i}$.
    *   Compute raw delta: $\Delta_i^{raw} = \theta'_i - \theta$.
5.  **Layer-wise Perturbation (DP-SGD):**
    *   **For** each layer $j$:
        i.  **Clip:** $\tilde{\Delta}_{i,j} = \Delta_{i,j}^{raw} \cdot \min(1, \frac{C_j}{\|\Delta_{i,j}^{raw}\|_2})$
        ii. **Noise:** $\Delta_{i,j} = \tilde{\Delta}_{i,j} + \mathcal{N}(0, (\sigma_j C_j / \sqrt{n_i})^2)$
6.  **Secure Masking (Optional):**
    *   Quantize $\Delta_i$ to integers using bound $C_{\text{avg}}$.
    *   Apply zero-sum random masks $\mathcal{M}_i$.
7.  **Return** $\{\Delta_i, R_i\}$.

---

### Algorithm 3: Sensitivity-Based Noise Allocation (AdaPriv-Brain)
**Goal:** Distribute privacy budget across layers based on their information density and impact on convergence.

**Input:** Recent gradient norms $\{\mathbf{g}_{j,t-1}\}$, target budget $\epsilon_t$, base clip $C_{base}$
**Output:** Per-layer $\{\sigma_j, C_j\}$

1.  **Compute Sensitivity Components** for each layer $j$:
    *   **Volatility:** $v_j = \text{std}(\text{historical\_norms}_j)$
    *   **Impact:** $a_{j} = \text{EMA}(\text{second\_moment}_j)$
2.  **Calculate Sensitivity Score** $(s_j)$:
    $$s_j^{raw} = (1 - \alpha) \cdot \frac{v_j}{\text{mean}(v)} + \alpha \cdot \frac{a_j}{\text{mean}(a)}$$
3.  **Normalize Scores:** $s_j = s_j^{raw} / \text{mean}(s^{raw})$ (Ensure relative differences).
4.  **Calibrate Base Noise:** $\sigma_{base} = 1 / \epsilon_t$.
5.  **Map to Parameters:**
    *   **Adaptive Clip Norm:** $C_j = C_{base} / \text{clamp}(s_j, 0.5, 2.0)$
    *   **Adaptive Noise Multiplier:** $\sigma_j = \sigma_{base} \cdot s_j$
6.  **Return** $\{\sigma_j, C_j\}$.
