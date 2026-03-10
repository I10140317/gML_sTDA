# Modified sTDA Code for AO Integral Extraction

This directory contains a modified version of the **sTDA (simplified Tamm–Dancoff approximation)** code originally developed by the **Grimme group**.

The original implementation is available at:

https://github.com/grimme-lab/stda

and is distributed under the **GNU General Public License v3.0 (GPL-3.0)**.

---

## Purpose of This Modification

The original sTDA program is designed for excited-state calculations within the simplified Tamm–Dancoff approximation framework.

In this project, the code has been **significantly simplified and modified** for a different purpose.

The current implementation is intended only for **AO-based integral evaluation** and serves as a preprocessing tool in the gML-sTDA workflow.

Main modifications include:

- Retaining only routines related to **AO-based integral evaluation**
- Computing **AO-related integrals required for subsequent calculations**
- Exporting the computed integrals as **sparse matrices in plain text format**
- Removing other components of the original sTDA implementation, including excited-state solvers and related workflow modules

Therefore, this code should be viewed as a **specialized utility derived from the original sTDA implementation**, rather than a complete sTDA program.

---

## Output

The program outputs AO-related integrals in **sparse matrix format** written to text files.

A typical output format is:

row_index  column_index  value

This format allows convenient post-processing by external programs (e.g., Python scripts).

---

## License

This code is derived from the original **sTDA implementation** developed by the **Grimme group** and therefore remains distributed under the **GNU General Public License v3.0 (GPL-3.0)**.

The original copyright and license terms are preserved.


