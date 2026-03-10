---

# Modified sTDA Code for AO Integral Extraction

This directory contains a modified version of the **sTDA (simplified Tamm-Dancoff approximation)** code originally developed by the Grimme group.

The original implementation is available at:

https://github.com/grimme-lab/stda

and is distributed under the **GNU General Public License v3.0 (GPL-3.0)**.

---

# Purpose of This Modification

The original sTDA program is designed for excited-state calculations within the simplified Tamm-Dancoff approximation framework.

In this project, the code has been **significantly simplified and modified** for a different purpose:

- Only the routines related to **AO-based integral evaluation** are retained.
- The code computes **AO-related integrals required for subsequent calculations**.
- The computed integrals are exported as **sparse matrices in plain text format**.
- All other components of the original sTDA implementation (e.g., excited-state solvers, related workflow modules) have been removed.

Therefore, the current code should be viewed as a **specialized utility derived from the sTDA implementation**, rather than a full sTDA program.

---


