from pyscf import lib, ao2mo, gto, scf, dft, tdscf, tddft
import itertools
import numpy as np
import sys,os
from itertools import product
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import LinearOperator, eigsh
from pyscf.tdscf._lr_eig import eigh as lr_eigh
from pyscf.lib import logger
from rbm_train_sample_gpu_cpu import RBM, train_rbm, sample_from_rbm_constrained
from pysTDA_pairs import sTDA
import time
from scipy.sparse import csr_matrix, coo_matrix
from parse_molden import MoldenData
from parse_gto import parse_gto_aorange
import random
from pathlib import Path

def uniform_sample_compact(n, k=5):
    partition_size = n // k
    return [
        random.randint(i * partition_size, 
                      (i + 1) * partition_size - 1 if i < k - 1 else n - 1)
        for i in range(k)
    ]

def A_to_sparse(A, threshold=1e-5, mode="masked", fmt="csr"):

    if mode == "binary":
        data = (np.abs(A) > threshold).astype(float)
    elif mode == "masked":
        data = np.where(np.abs(A) > threshold, A, 0.0)
    elif mode == "normalized":
        mask = np.abs(A) > threshold
        data = np.zeros_like(A)
        if np.any(mask):
            data[mask] = A[mask] / np.max(np.abs(A[mask]))
    else:
        raise ValueError("mode should be 'binary' / 'masked' / 'normalized'")

    if fmt == "csr":
        sparse_matrix = csr_matrix(data)
    elif fmt == "coo":
        sparse_matrix = coo_matrix(data)
    else:
        raise ValueError("fmt should be 'csr' or 'coo'")

    return sparse_matrix

def diagonalize_sparse_Asub(Asub, pairs, n_states=5, weight_thresh=1e-4):
    n = len(pairs)
    A_sparse = A_to_sparse(Asub, threshold=1e-5, mode="masked", fmt="csr")
    w_sel, v_sel = eigsh(A_sparse, k=n_states, which='SA') 
    W_matrix = (v_sel.conj() * v_sel).real
    mask = np.any(W_matrix > weight_thresh, axis=1)
    selected_pairs = [p for p, keep in zip(pairs, mask) if keep]
    W_sel = W_matrix[mask, :]

    return w_sel, v_sel, W_sel, selected_pairs

def select_pairs(pairs, a, n_states=5, weight_thresh=1e-5):
    w_sub, v_sub = np.linalg.eigh(a)
    idx = np.argsort(w_sub)
    w_sub, v_sub = w_sub[idx], v_sub[:, idx]
    n_use = min(n_states, len(w_sub))
    w_sel = w_sub[:n_use]
    v_sel = v_sub[:, :n_use]
    W_matrix = (v_sel.conj() * v_sel).real
    mask = np.any(W_matrix > weight_thresh, axis=1)
    selected_pairs = [p for p, keep in zip(pairs, mask) if keep]
    W_sel = W_matrix[mask, :]
    return w_sel, v_sel, W_sel, selected_pairs

def ia_to_lin(i, a, nocc, nvir):
    return i * nvir + (a - nocc)

def lin_to_ia(idx, nocc, nvir, nmo):
    i = idx // nvir
    a = (idx % nvir) + nocc
    return i, a

def get_orbital_matrix(mol, method='RHF', n_occ_param=5, n_vir_param=5):
    if method.upper() == 'RHF':
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.kernel()
        aorange = mol.aoslice_by_atom()
    elif method.upper() == 'DFT':
        from pyscf import dft
        mf = dft.RKS(mol)
        mf.xc = 'PBE0'
        mf.verbose = 0
        mf.kernel()
        aorange = mol.aoslice_by_atom()
    elif method.upper() == 'MOLDEN':
        mol =None
        mf = MoldenData("./file.molden")
        aorange = parse_gto_aorange("./file.molden", cart=True)
    else:
        raise ValueError("method parameters should be RHF or DFT")

    nmo = mf.mo_coeff.shape[1]
    occ_num = mf.mo_occ[mf.mo_occ > 0].shape[0]
    vir_num = mf.mo_occ[mf.mo_occ == 0].shape[0]

    return occ_num, vir_num, mf, aorange

def generate_single_excitations(orbitals, occ_flags, occ_num, occ_ini, vir_ini):
    """
    Generate all singly excited configurations.
    Parameters:
        orbitals : 1D array, orbital nubmers
        occ_flags : 1D array, occupation flags
        occ_num : number of occupied orbitals
        n : lower limit for occ orbitals
        m : upper limits for vir orbitals
    return:
        orbitals : 1D orbitals
        configs  : 2D array, each row corresponds to a configuration
    """

    occ_indices = [i for i in orbitals if i < occ_num and i >= (occ_num - occ_ini)]
    vir_indices = [i for i in orbitals if i >= occ_num and i < (occ_num + vir_ini)]

    configs = []
    for i, a in product(occ_indices, vir_indices):
        new_flags = occ_flags.copy()
        idx_i = np.where(orbitals == i)[0][0]
        idx_a = np.where(orbitals == a)[0][0]
        new_flags[idx_i] = 0
        new_flags[idx_a] = 1
        configs.append(new_flags)

    configs = np.array(configs, dtype=int)
    print(f"\nGenerate {configs.shape[0]} single excitations with length of  {configs.shape[1]}")
    return configs

def get_excitation_pairs(orbitals, occ_flags, configs):
    """
    Generate (i, a) pairs according to orbital labels and singly excitaion matrix
    
    Parameters:
        orbitals : ndarray，
        occ_flags : 1D array, using 0/1 values to represent ground state
        configs : 2D array, using 0/1 arrays to represent singly excited configurations
    
    return:
        pairs : list[tuple[int, int]]，(i, a) pairs
    """
    pairs = []

    for cfg in configs:
        diff = cfg - occ_flags
        i_indices = np.where(diff == -1)[0]
        a_indices = np.where(diff == +1)[0]
        if len(i_indices) == 1 and len(a_indices) == 1:
            i = orbitals[i_indices[0]]
            a = orbitals[a_indices[0]]
            pairs.append((i, a))
        else:
            pairs.append(None)

    return pairs

def sample_config_vectors(selected_pairs, weights, occ_num, vir_num,
                          n_occ_keep, n_vir_keep, n_samples=1000):
    """
    Sampling n vectors

    Parameters：
        selected_pairs : list[(i,a)]   # configurations 
        weights        : list[float]   # corresponding weights
        occ_num        : int           # number of occupied orbitals
        vir_num        : int           # number of virtual orbitals
        n_occ_keep     : int           # considered number of occ
        n_vir_keep     : int           # considered number of vir
        n_samples      : int           # number of generated samples 

    return：
        configs         : np.ndarray (n_samples, n_occ_keep + n_vir_keep)
        unique_configs  : np.ndarray (len(selected_pairs), n_occ_keep + n_vir_keep)
        occ_indices     : list[int]   
        vir_indices     : list[int]   
        chosen_pairs    : list[(i,a)] 
    """

    # ---------- orbital range ----------
    occ_indices = list(range(occ_num - n_occ_keep, occ_num))
    vir_indices = list(range(occ_num, occ_num + n_vir_keep))
    n_total = n_occ_keep + n_vir_keep

    # ---------- normalize ----------
    weights = np.array(weights, dtype=float)
    prob = weights / np.sum(weights)

    # ---------- sampling by weights ----------
    sampled_idx = np.random.choice(len(selected_pairs), size=n_samples, p=prob, replace=True)
    chosen_pairs = [selected_pairs[i] for i in sampled_idx]

    # ---------- construct fixed unique_configs ----------
    unique_configs = np.zeros((len(selected_pairs), n_total), dtype=int)
    for idx, (i, a) in enumerate(selected_pairs):
        occ_vec = np.zeros(n_total, dtype=int)
        occ_vec[:n_occ_keep] = 1
        if i in occ_indices and a in vir_indices:
            i_local = occ_indices.index(i)
            a_local = vir_indices.index(a)
            occ_vec[i_local] = 0
            occ_vec[n_occ_keep + a_local] = 1
        unique_configs[idx] = occ_vec

    # ---------- sampling configs ----------
    configs = np.zeros((n_samples, n_total), dtype=int)
    for k, (i, a) in enumerate(chosen_pairs):
        if i not in occ_indices or a not in vir_indices:
            continue
        occ_vec = np.zeros(n_total, dtype=int)
        occ_vec[:n_occ_keep] = 1
        i_local = occ_indices.index(i)
        a_local = vir_indices.index(a)
        occ_vec[i_local] = 0
        occ_vec[n_occ_keep + a_local] = 1
        configs[k] = occ_vec

    return configs, unique_configs, occ_indices, vir_indices, chosen_pairs

def merge_single_excitation_configs(configs1, configs2, n_occ_keep, n_vir_keep):
    """
    Merge two arrays representing single excitation configurations.
    Discarding unphysical ones.
    
    Parameters:
        configs1, configs2 : np.ndarray (n, n_occ_keep+n_vir_keep)
        n_occ_keep         : int  
        n_vir_keep         : int  
    return:
        merged_unique : np.ndarray  merged unique configs
    """
    def is_single_excitation(vec):
        return (
            np.sum(vec[:n_occ_keep] == 0) == 1 and
            np.sum(vec[n_occ_keep:] == 1) == 1
        )

    # filter unphysical ones
    valid1 = np.array([is_single_excitation(v) for v in configs1])
    valid2 = np.array([is_single_excitation(v) for v in configs2])

    filtered1 = configs1[valid1]
    filtered2 = configs2[valid2]

    # merge and discarding
    merged = np.vstack((filtered1, filtered2))
    merged_unique = np.unique(merged, axis=0)

    return merged_unique

def restore_pairs_from_vector(vectors, occ_indices, vir_indices):
    """
    Restore the vectors to (i,a) pairs

    Parameters：
        vectors      : ndarray (N, n_occ_keep + n_vir_keep)
        occ_indices  : list[int]
        vir_indices  : list[int]
    return：
        pairs_global : list[(i,a)]
    """
    pairs_global = []

    for vec in vectors:
        i_local = np.where(vec[:len(occ_indices)] == 0)[0]
        a_local = np.where(vec[len(occ_indices):] == 1)[0]

        if len(i_local) == 1 and len(a_local) == 1:
            i_global = occ_indices[i_local[0]]
            a_global = vir_indices[a_local[0]]
            pairs_global.append((i_global, a_global))
        else:
            pairs_global.append(None)

    return pairs_global

def main_loop(occ_num, vir_num, occ_num_keep, vir_num_keep, occ_ini, vir_ini, 
              mf, nstate=5, max_iter=40, stable_steps=5,
              energy_thresh=2e-2, ratio_thresh=0.05, 
              aorange=None, ini_pairs = None, device= "cpu", 
              sample_roots=10, weight_thresh=1e-4,
              num_add_mos=20):

    lower_limit = occ_num - min(occ_num, occ_num_keep)
    upper_limit = occ_num + min(vir_num, vir_num_keep)

    prev_w_sel = None
    prev_pairs = None
    stable_counter = 0

    mol=None
    cache = {}
   
    t_ML=0
    t_diag=0

    for iteration in range(max_iter):

        if prev_w_sel is None:

            if ini_pairs is None:
                occ_min = occ_num - min(occ_num, occ_ini) - num_add_mos
                vir_max = occ_num + min(vir_num, vir_ini) + num_add_mos
                lower = max(lower_limit, occ_min)
                upper = min(upper_limit, vir_max)
                orbitals = np.arange(lower, upper)
                occ_flags = np.array([1 if i < occ_num else 0 for i in orbitals])
                configs = generate_single_excitations(orbitals, occ_flags, occ_num, occ_ini, vir_ini)
                restored_pairs = get_excitation_pairs(orbitals, occ_flags, configs)
            else:
                restored_pairs = ini_pairs
                occ_min=np.min(np.array(restored_pairs)[:, 0]) - num_add_mos
                vir_max=np.max(np.array(restored_pairs)[:, 1]) + num_add_mos
                lower = max(lower_limit, occ_min)
                upper = min(upper_limit, vir_max)

            stda = sTDA(mol, mf, aorange, ax=0.20, nocc=occ_num_keep, nvir=vir_num_keep, pairs=restored_pairs)

        else:
            stda.pairs = restored_pairs

        occ_num_update = occ_num - lower
        vir_num_update = upper - occ_num
        print(f'occ_min: {occ_min}, vir_max: {vir_max}')
        print(f'occ_num_as: {occ_num_update}, vir_num_as: {vir_num_update}')
        lower = max(lower_limit, occ_min)
        upper = min(upper_limit, vir_max)

        print(f"\n===== Iteration {iteration+1} =====")

        # === Step 1: construct TDA subspace ===
        t0=time.time()
        Asub = stda.get_Adict(prev_pairs=prev_pairs)
        w_sel, v_sel, W_sel, selected_pairs = diagonalize_sparse_Asub(Asub, stda.pairs, n_states=nstate, weight_thresh=weight_thresh)
        t1=time.time()
        t_diag += t1-t0
        print(t_diag)
        occ_min=np.min(np.array(selected_pairs)[:, 0]) - num_add_mos
        vir_max=np.max(np.array(selected_pairs)[:, 1]) + num_add_mos

        # === Step 2: convergence test ===
        converged = False

        if prev_w_sel is not None:
            #dE = np.max(np.abs(w_sel - prev_w_sel))
            dE = np.mean(np.abs(w_sel - prev_w_sel))
            overlap_ratio = len(set(selected_pairs) & set(prev_pairs)) / len(selected_pairs)
            new_ratio = 1 - overlap_ratio
           
            #print(f"ΔE_max = {dE:.6e} hartree | new CI ratio = {new_ratio:.2%}")
            print(f"ΔE_mean = {dE:.6e} hartree | new CI ratio = {new_ratio:.2%}")
           
            if dE < energy_thresh and new_ratio < ratio_thresh:
                stable_counter += 1
                print(f"Convergence Satisfied {stable_counter}/{stable_steps}.")
            else:
                stable_counter = 0 

            if stable_counter >= stable_steps:
                print(f"Times for t_ML and t_diag: {t_ML} {t_diag}")
                occ_off = stda.nocc - stda.nocc_stda
                vir_off = stda.nocc
                i_idx = np.array(stda.pairs)[:, 0] - occ_off
                a_idx = np.array(stda.pairs)[:, 1] - vir_off
                molintx=stda.molintx[i_idx,a_idx]
                molinty=stda.molinty[i_idx,a_idx]
                molintz=stda.molintz[i_idx,a_idx]

                with open("stda.out","w") as f:
                    print(f"\n===== Final Results =====")
                    for i in range(nstate):
                        dipole_moment=(v_sel[:,i].T @ molintx)**2+(v_sel[:,i].T @ molinty)**2+(v_sel[:,i].T @ molintz)**2
                        fl=w_sel[i]*4.0*dipole_moment/3.0
                        print(f"Excited State {i+1} 1 Ene: {w_sel[i]*27.21138:.4f} fL: {fl:.4f} diple_moment**2: {dipole_moment:.5f}")
                        f.write(f"Excited State {i+1} 1 Ene: {w_sel[i]*27.21138:.4f} fL: {fl:.4f} dipole_moment**2: {dipole_moment:.5f}\n")
                        sorted_indices = sorted(range(len(stda.pairs)), 
                                                key=lambda j: abs(v_sel[j,i]), 
                                                reverse=True)

                        for j in sorted_indices[0:3]:
                            if abs(v_sel[j,i]) > -1:
                                print(f"    {stda.pairs[j][0]+1} ->  {stda.pairs[j][1]+1}  {v_sel[j,i]:.6f}")
                                f.write(f"    {stda.pairs[j][0]+1} ->  {stda.pairs[j][1]+1}  {v_sel[j,i]:.6f} \n")
                        print("\n",end="")
                        f.write("\n")

                np.save('final_w_sel.npy', w_sel)
                np.save('final_v_sel.npy', v_sel)
                np.save('final_selected_pairs.npy', selected_pairs)
                break
                print(f"Converged! Stop Iteration.")

        # === Step 3: RBM training and sampling ===
        print("Training RBM... ")
        t0=time.time()
        numbers = uniform_sample_compact(nstate, sample_roots)

        k = 0
        for istate in numbers:
            configs1, unique_configs, occ_idx, vir_idx, chosen_pairs = sample_config_vectors(
                    selected_pairs, W_sel[:,istate],
                    occ_num, vir_num,
                    n_occ_keep=occ_num_update, n_vir_keep=vir_num_update,
                    n_samples=150)

            if k == 0:
                merged_unique = unique_configs

            rbm = RBM(n_visible=configs1.shape[1], n_hidden=configs1.shape[1])

            trained_rbm = train_rbm(rbm, configs1, batch_size=1000, epochs=50, lr=0.01, device=device)

            configs2 = sample_from_rbm_constrained(
                trained_rbm, n_samples=100,
                n_occ_keep=occ_num_update, n_vir_keep=vir_num_update,
                device=device
            )
  
            merged_unique = merge_single_excitation_configs(merged_unique, configs2, n_occ_keep=occ_num_update, n_vir_keep=vir_num_update)
            #print("Length of new configs:",len(merged_unique)-len(unique_configs))
            k += 1

        print("Length of Updated Configurations:",len(merged_unique))

        restored_pairs = restore_pairs_from_vector(merged_unique, occ_idx, vir_idx)

        # === Step 5: update ===
        prev_w_sel = w_sel.copy()
        prev_pairs = selected_pairs.copy()
        t1=time.time()
        t_ML += t1-t0
        print(t_ML)
    else:
        print("Error: Max Iter reaches!")


if __name__ == "__main__":
    print(time.time())

    occ_num_keep = 5
    vir_num_keep = 13
    occ_ini = 2
    vir_ini = 3
    nstate = 5
    max_iter = 500
    stable_steps = 3
    energy_thresh = 1e-4
    ratio_thresh = 0.03
    mol=None
    device="cpu"
    sample_roots=3
    weight_thresh=1e-3
    num_add_mos=20

    for p in Path(".").glob("*int"):
        if p.is_file():
            p.unlink()

    pair_file = 'ini_pairs.npy'

    if not os.path.exists(pair_file):
        ini_pairs = None
    else:
        ini_pairs = [tuple(x) for x in np.load(pair_file)]

    print("reading and parsing...")
    t0=time.time()
    occ_num, vir_num, mf, aorange = get_orbital_matrix(mol, 'molden')
    t1=time.time()
    print(f"Time for reading and parsing molden file: {t1-t0}")

    main_loop(occ_num, vir_num, occ_num_keep, vir_num_keep, occ_ini, vir_ini, 
              mf,nstate, max_iter, stable_steps,
              energy_thresh, ratio_thresh, aorange, 
              ini_pairs=ini_pairs, device=device,
              sample_roots=sample_roots,weight_thresh=weight_thresh,
              num_add_mos=num_add_mos)

    for p in Path(".").glob("*int"):
        if p.is_file():
            p.unlink()

    print(time.time())


