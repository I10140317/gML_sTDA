import numpy as np
import re
import subprocess
import os
from pathlib import Path
from numba import njit
from fortio import FortranFile
from typing import Optional, Tuple, Union
from scipy.sparse import coo_matrix, csr_matrix
import time

def create_if_not_exists(filename, lines):
    path = Path(filename)
    if not path.exists():
        with open(path, "w") as f:
            for line in lines:
                f.write(line.rstrip() + "\n")
        print(f"Created: {path}")
    else:
        print(f"Existed {path}, Unchanged!")

def run_overlap_calculation():
    """Run Gaussian calculation to generate necessary files"""
    print("Running stda_overlap calculation...")
    
    try:
        subprocess.run(['./stda_overlap_v2 -f molden.input > stda_overlap.out'],
                              shell=True, check=True, capture_output=True, text=True)
        print("Overlap calculation completed")
    except subprocess.CalledProcessError as e:
        print(f"Overlap calculation error: {e}")
        return False
    
    return True

def molden4stda(filename,cart=True):
    with open("molden.input", 'w') as fout:
        with open(filename, 'r') as f:
            for raw_line in f:
                line=raw_line.rstrip('\n')
                if '[MO]' in line:
                    break
                else:
                    fout.write(line+"\n")

def get_ints(dat_file):
    """Main function: complete automated workflow"""
    print(f"=== get {dat_file} ===")
    
    if not os.path.exists(dat_file):
        # Check if required files exist
        required_files = ['file.molden']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"Missing required files: {missing_files}")
            print("Please ensure file.molden file exists")
            return
        
        # Step 1: Run Multiwfn calculation
        #molden4stda("file.molden")
        print("calculating ao ints...")
        t0=time.time()
        if not run_overlap_calculation():
            print("Calculation process failed, attempting to parse existing files...")
        t1=time.time()
        print(f"Time for calculating AO overlaps: {t1-t0}")
        
        # Step 2: Parse Overlap.dat file
        if not os.path.exists(dat_file):
            print(f"Error: Cannot find file {dat_file}")
            print("Please check if stda_overlap calculation completed successfully")
            return None
        S = read_fints(dat_file)
    else:
        S = read_fints(dat_file)

    return S

def read_fints(
    filename: str,
    shape: Optional[Tuple[int, int]] = None,
    n: Optional[int] = None,
    dtype=np.float64,
    assume_lower: bool = True,
) -> csr_matrix:

    data = np.loadtxt(filename, dtype=np.float64)  # (nnz, 3)
    if data.ndim == 1:
        data = data.reshape(1, 3)

    rows = data[:, 0].astype(np.int64) - 1
    cols = data[:, 1].astype(np.int64) - 1
    vals = data[:, 2].astype(dtype, copy=False)

    if shape is not None:
        nrows, ncols = shape
        if nrows != ncols:
            raise ValueError(f"shape must be square for symmetric matrix, got {shape}")
        dim = nrows
    elif n is not None:
        dim = int(n)
    else:
        dim = int(max(rows.max(), cols.max()) + 1)

    if assume_lower:
        mask = rows > cols
        r2 = cols[mask]
        c2 = rows[mask]
        v2 = vals[mask]
    else:
        mask = rows != cols
        r2 = cols[mask]
        c2 = rows[mask]
        v2 = vals[mask]

    rr = np.concatenate([rows, r2])
    cc = np.concatenate([cols, c2])
    vv = np.concatenate([vals, v2])

    A = coo_matrix((vv, (rr, cc)), shape=(dim, dim), dtype=dtype).tocsr()

    A.sum_duplicates()
    A.sort_indices()
    return A
