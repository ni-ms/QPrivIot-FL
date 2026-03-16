# Research Algorithms and Code Mapping

This document outlines the three core algorithms implemented in the **AdaPriv-FL** framework, mapping each theoretical step to its exact implementation in the codebase.

---

### Algorithm 1: Adaptive Global Server Protocol
*This algorithm manages the round-by-round orchestration, privacy budgeting, and global aggregation.*

| Step | Theoretical Action | Code Reference (File: Line) |
| :--- | :--- | :--- |
| **1** | **Privacy Budgeting:** Calculate per-round epsilon ($\epsilon_t$) using cosine annealing. | `server_app.py`: 58–72 (`_get_round_epsilon`) |
| **2** | **Convergence Assessment:** Compute $CS_t$ using Coefficient of Variation over recent loss. | `server_app.py`: 91–96 (`configure_fit`) |
| **3** | **Global Aggregation (SecAgg):** Sum masked integers and dequantize to floated deltas. | `server_app.py`: 156–174 (`aggregate_fit`) |
| **4** | **Sensitivity Tracking:** Update layer-wise gradients norms and distribution stats. | `server_app.py`: 191–206 (`aggregate_fit`) |
| **5** | **Privacy Accounting:** Aggregate Renyi DP costs to monitor global budget exhaustion. | `server_app.py`: 222–225 (`aggregate_fit`) |

---

### Algorithm 2: Adaptive Client-Side Training & Perturbation
*This algorithm handles local learning, resource-aware scaling, and individual data protection.*

| Step | Theoretical Action | Code Reference (File: Line) |
| :--- | :--- | :--- |
| **1** | **Resource Profiling:** Establish device "Readiness Score" ($R_i$) via local telemetry. | `client_app.py`: 30–32 (`__init__`) |
| **2** | **Temporal Fine-Tuning:** Detect convergence plateau and reduce learning rate ($\eta$). | `client_app.py`: 73–80 (`fit`) |
| **3** | **Spatial Noise Scaling:** Intensify noise for low-resource devices to protect privacy. | `client_app.py`: 102–103 (`fit`) |
| **4** | **Structural Selection:** Allocate layer-specific noise/clip metadata from server stats. | `client_app.py`: 105–110 (`fit`) |
| **5** | **Perturbation:** Apply clipped Gaussian noise per-layer based on combined scores. | `client_app.py`: 165 (`fit`) |
| **6** | **Secure Masking:** Quantize floating-point deltas and apply zero-sum random masks. | `client_app.py`: 196–213 (`fit`) |

---

### Algorithm 3: Sensitivity Analysis & Adaptive Allocation
*This algorithm describes the mathematical "brains" behind how noise levels are chosen.*

| Step | Theoretical Action | Code Reference (File: Line) |
| :--- | :--- | :--- |
| **1** | **Impact Tracking:** Maintain EMA of second moments to estimate layer influence. | `privacy_utils.py`: 171–172 (`SensitivityTracker.update`) |
| **2** | **Sensitivity Scoring:** Combine gradient volatility and impact into a unified score. | `privacy_utils.py`: 216–220 (`get_sensitivities`) |
| **3** | **Score Normalization:** Ensure layer scores stay relative (mean=1.0) for stable allocation. | `privacy_utils.py`: 222–226 (`get_sensitivities`) |
| **4** | **Adaptive Clipping:** Map higher sensitivity to tighter (lower) clipping bounds ($C_j$). | `privacy_utils.py`: 264–265 (`allocate_adaptive_noise`) |
| **5** | **Adaptive Noise:** Map higher sensitivity to larger noise standard deviations ($\sigma_j$). | `privacy_utils.py`: 268 (`allocate_adaptive_noise`) |
