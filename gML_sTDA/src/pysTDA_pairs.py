import numpy as np
from scipy import linalg
from pyscf import lib
import os
import itertools
from parse_gto import parse_gto_aorange
from parse_molden import MoldenData
import time
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import  eigsh
from scipy.linalg import eigh,inv
import numpy as np
from get_lowdin_C import apply_S_half_adaptive
from get_ints import get_ints

def build_loewdin_ops(S, thr=1e-8):
    """
      X_apply(A)    = S^{-1/2} @ A
      invX_apply(A) = S^{+1/2} @ A
    """

    s, U = eigh(S, lower=True, overwrite_a=True, check_finite=False)

    s = np.clip(s, thr, None)
    inv_sqrt_s = 1.0/np.sqrt(s)
    sqrt_s     = np.sqrt(s)

    Ut = U.T

    def X_apply(A):
        # S^{-1/2} @ A  = U @ diag(inv_sqrt_s) @ U.T @ A
        return U @ (inv_sqrt_s[:,None] * (Ut @ A))

    def invX_apply(A):
        # S^{+1/2} @ A = U @ diag(sqrt_s) @ U.T @ A
        return U @ (sqrt_s[:,None] * (Ut @ A))

    return X_apply, invX_apply

def smart_sparse(name, M, thr_zero=1e-4, thr_sparse=0.8):

    if M is None:
        return None
    if issparse(M):
        nnz = M.nnz
        tot = np.prod(M.shape)
        print(f"[{name}] already sparse ({nnz/tot:.2%} nonzero, nnz={nnz})")
        return M

    M = np.where(np.abs(M) > thr_zero, M, 0.0)
    nonzero = np.count_nonzero(M)
    tot = M.size
    sparsity = 1 - nonzero / tot

    if sparsity >= thr_sparse:
        M_sparse = csr_matrix(M)
        print(f"[{name}] → sparse ({1-sparsity:.2%} nonzero, nnz={nonzero})")
        return M_sparse
    else:
        print(f"[{name}] kept dense ({1-sparsity:.2%} nonzero)")
        return M

class sTDA:
    def __init__(self, mol=None, mf=None, aorange=None, alpha=-100.0, beta=-100.0, ax=0.20, nocc=5, nvir=13, nstates=5, triplet=False, singlet=True, pairs=None):
        self.mol = mol

        if alpha <= -99.0:
            # \alpha = \alpha^{(1)} + \alpha^{(2)}a_x
            self.alpha = 1.42 + 0.48 * ax
        else:
            self.alpha = alpha

        if beta <= -99.0:
            # \beta = \beta^{(1)} + \beta^{(2)}a_x
            self.beta = 0.20 + 1.83 * ax
        else:
            self.beta = beta

        if mol is None:
            self.natom = mf.natm
            self.coord = mf.coord
            self.atom_charge = mf.atom_charge
        else:
            self.natom = mol.natm
            self.coord = mol.atom_coords()
            self.atom_charge = mol.atom_charges()

        self.ax = ax
        self.nocc_stda = nocc
        self.nvir_stda = nvir
        self.nstates = nstates
        self.nmo_stda = nocc + nvir
        self.occidx = np.where(mf.mo_occ == 2)[0]
        self.viridx = np.where(mf.mo_occ == 0)[0]
        self.nocc = len(self.occidx)
        self.nvir = len(self.viridx)

        self.occidx_stda = self.occidx[-self.nocc_stda:]
        self.viridx_stda = self.viridx[:self.nvir_stda]
        act_space = np.hstack((self.occidx_stda,self.viridx_stda))
        self.mo_coeff = mf.mo_coeff[:,act_space]

        self.q_ov = None
        self.q_oo = None
        self.q_vv = None
        self.gammaJ = None
        self.gammaK = None

        if self.q_ov is None or self.q_oo is None or self.q_vv is None:
            self.c_mo = self.coeff_ao2mo()

        print("preparing other data...")
        t0=time.time()
        self.aorange = aorange
        self.mo_energy = mf.mo_energy
        self.triplet = triplet
        self.singlet = singlet

        if pairs is None:
            self.pairs = list(itertools.product(self.occidx_stda, self.viridx_stda))
        else:
            self.pairs = pairs

        aolint=get_ints("xlint")
        aolint = smart_sparse("xlint", aolint, 1e-8, 0.90)
        temp = aolint @ self.mo_coeff[:,self.nocc_stda:]
        self.molintx=self.mo_coeff[:,:self.nocc_stda].T @ temp

        aolint=get_ints("ylint")
        aolint = smart_sparse("ylint", aolint, 1e-8, 0.90)
        temp = aolint @ self.mo_coeff[:,self.nocc_stda:]
        self.molinty=self.mo_coeff[:,:self.nocc_stda].T @ temp

        aolint=get_ints("zlint")
        aolint = smart_sparse("zlint", aolint, 1e-8, 0.90)
        temp = aolint @ self.mo_coeff[:,self.nocc_stda:]
        self.molintz=self.mo_coeff[:,:self.nocc_stda].T @ temp

        self.pairs2 = None

        #print('stda_ini',time.time())

        if self.q_ov is None:
            self.q_ov = self.get_qia()
            #print('q_ov',time.time())

        if self.q_oo is None:
            self.q_oo = self.get_qij()
            #print('q_oo',time.time())

        if self.q_vv is None:
            self.q_vv = self.get_qab()
            #print('q_vv',time.time())

        if self.gammaJ is None or self.gammaK is None:
            self.gammaJ, self.gammaK = self.get_gamma()
            #print('gamma_JK',time.time())

        self.Adict = None

        nA, nvir, _ = self.q_vv.shape
        Q = self.q_vv.reshape(nA, -1)  # (nA, nvir*nvir)
        self.T = (self.gammaJ @ Q).reshape(nA, nvir, nvir)
        #print('T_vv',time.time())
        t1=time.time()
        print(f"Time for calculating other data: {t1-t0}")

    def coeff_ao2mo(self):
        r'''
        X = S^{-1/2} = U s^{-1/2} U^\dagger
        C' = X^{-1} C
        '''
        S=get_ints("sint")

        t0=time.time()
        S = smart_sparse("Overlap", S, 1e-5, 0.90)

        if issparse(S):
            C, (lam_min, lam_max), used_deg = apply_S_half_adaptive(
                S, self.mo_coeff, deg0=70, step=10, tol=1e-3, maxdeg=200,
                bound_tol=1e-5,  verbose=True
            )
            #print('diag+dot_time1+dot_time2',time.time())
        else:
            X, invX = build_loewdin_ops(S, thr=1e-8)
            #print('diag+dot_time1',time.time())
            C = invX(self.mo_coeff)
            #print('dot_time2',time.time())
        t1=time.time()
        print(f"Time for calculating S^1/2 * C: {t1-t0}")
        return C

    def get_lowdin(self, i, j):
        r'''
        Lowdin population of i and j orbital centered on atom A
        q_{ij}^A = \sum_{\mu in A} C_{\mu i} C_{\mu j}
        '''
        qij = np.zeros(self.natom)
        for A in range(self.natom):
            ao_start = self.aorange[A,2]
            ao_end = self.aorange[A,3]
            qij[A] = np.dot(self.c_mo[ao_start:ao_end,i], self.c_mo[ao_start:ao_end,j])
        return qij

    def get_distance(self):
        r'''
        Calculate |R_A - R_B|, return a matrix
        '''
        dist = np.linalg.norm(self.coord[:, None] - self.coord[None, :], axis=-1)
        return dist

    def setrep(self):
        # Initialize an array with 94 elements (like the Fortran rep array)
        rep = np.zeros(94, dtype=np.float64)

        # Semiempirical Evaluation of the Global Hardness of the Atoms
        # The values are hardcoded as given in the original Fortran code
        rep[0] = 0.472592880
        rep[1] = 0.922033910
        rep[2] = 0.174528880
        rep[3] = 0.257007330
        rep[4] = 0.339490860
        rep[5] = 0.421954120
        rep[6] = 0.504381930
        rep[7] = 0.586918630
        rep[8] = 0.669313510
        rep[9] = 0.751916070
        rep[10] = 0.179641050
        rep[11] = 0.221572760
        rep[12] = 0.263485780
        rep[13] = 0.305396450
        rep[14] = 0.347340140
        rep[15] = 0.389247250
        rep[16] = 0.431156700
        rep[17] = 0.473082690
        rep[18] = 0.171054690
        rep[19] = 0.202762440
        rep[20] = 0.210073220
        rep[21] = 0.217396470
        rep[22] = 0.224710390
        rep[23] = 0.232015010
        rep[24] = 0.239339690
        rep[25] = 0.246656380
        rep[26] = 0.253982550
        rep[27] = 0.261288630
        rep[28] = 0.268594760
        rep[29] = 0.275925650
        rep[30] = 0.307629990
        rep[31] = 0.339315800
        rep[32] = 0.372359850
        rep[33] = 0.402735490
        rep[34] = 0.434457760
        rep[35] = 0.466117080
        rep[36] = 0.155850790
        rep[37] = 0.186493240
        rep[38] = 0.193562100
        rep[39] = 0.200633110
        rep[40] = 0.207705220
        rep[41] = 0.214772540
        rep[42] = 0.221846140
        rep[43] = 0.228918720
        rep[44] = 0.235986210
        rep[45] = 0.243056120
        rep[46] = 0.250130180
        rep[47] = 0.257199370
        rep[48] = 0.287847800
        rep[49] = 0.318486730
        rep[50] = 0.349124310
        rep[51] = 0.379765930
        rep[52] = 0.410408080
        rep[53] = 0.441057770
        rep[54] = 0.050193320
        rep[55] = 0.067625700
        rep[56] = 0.085044450
        rep[57] = 0.102477360
        rep[58] = 0.119911050
        rep[59] = 0.137327720
        rep[60] = 0.154762970
        rep[61] = 0.172182650
        rep[62] = 0.189612880
        rep[63] = 0.207047600
        rep[64] = 0.224467520
        rep[65] = 0.241896450
        rep[66] = 0.259325030
        rep[67] = 0.276760940
        rep[68] = 0.294182310
        rep[69] = 0.311595870
        rep[70] = 0.329022740
        rep[71] = 0.345922980
        rep[72] = 0.363880480
        rep[73] = 0.381305860
        rep[74] = 0.398774760
        rep[75] = 0.416142980
        rep[76] = 0.433645100
        rep[77] = 0.451040140
        rep[78] = 0.468489860
        rep[79] = 0.485845500
        rep[80] = 0.125267300
        rep[81] = 0.142686770
        rep[82] = 0.160116150
        rep[83] = 0.177558890
        rep[84] = 0.194975570
        rep[85] = 0.212407780
        rep[86] = 0.072635250
        rep[87] = 0.094221580
        rep[88] = 0.099202950
        rep[89] = 0.104186210
        rep[90] = 0.142356330
        rep[91] = 0.163942940
        rep[92] = 0.185519410
        rep[93] = 0.223701390

        return rep    

    def get_gamma(self):
        r'''
        Calculate the gamma matrix
        \gamma_{AB}^J = (\frac{1}{(R_{AB})^\beta + (a_x \eta_{AB})^{-\beta}})^{1/\beta}
        \gamma_{AB}^K = (\frac{1}{(R_{AB})^\alpha + (\eta_{AB})^{-\alpha}})^{1/\alpha}
        \eta_{AB} = (\eta_A + \eta_B) / 2
        '''
        dist = self.get_distance()
        eta = np.zeros(self.natom)
        for i in range(self.natom):
            idx = self.atom_charge[i] - 1
            eta[i] = self.setrep()[idx]
        gammaJ = np.zeros((self.natom, self.natom))
        gammaK = np.zeros((self.natom, self.natom))
        for i in range(self.natom):
            for j in range(self.natom):
                etaAB = (eta[i] + eta[j]) * 0.5
                gammaJ[i,j] = (1 / (np.power(dist[i,j], self.beta) + np.power(self.ax * etaAB, -self.beta)))**(1/self.beta)
                gammaK[i,j] = (1 / (np.power(dist[i,j], self.alpha) + np.power(etaAB, -self.alpha)))**(1/self.alpha)
        return gammaJ, gammaK

    def get_qia(self):
        """
        Calculate q_{ia}^A = sum_{μ in A} C_{μ i} C_{μ a} for all i, a, and A.
        """
        qia = np.zeros((self.natom, self.nocc_stda, self.nvir_stda))
        #C_occ = self.c_mo[:, self.occidx_stda]  # Shape: (n_ao, nocc_stda)
        #C_vir = self.c_mo[:, self.viridx_stda]  # Shape: (n_ao, nvir_stda)
        C_occ = self.c_mo[:, :self.nocc_stda]  # Shape: (n_ao, nocc_stda)
        C_vir = self.c_mo[:, self.nocc_stda:]  # Shape: (n_ao, nvir_stda)
        for A in range(self.natom):
            ao_start, ao_end = self.aorange[A, 2:4]
            C_A_occ = C_occ[ao_start:ao_end, :]  # Shape: (n_ao_A, nocc_stda)
            C_A_vir = C_vir[ao_start:ao_end, :]  # Shape: (n_ao_A, nvir_stda)
            qia[A, :, :] = np.dot(C_A_occ.T, C_A_vir)  # Shape: (nocc_stda, nvir_stda)
        return qia

    def get_qij(self):
        """
        Calculate q_{ij}^A = sum_{μ in A} C_{μ i} C_{μ j} for all i, j, and A.
        """
        qij = np.zeros((self.natom, self.nocc_stda, self.nocc_stda))
        #C_occ = self.c_mo[:, self.occidx_stda]  # Shape: (n_ao, nocc_stda)
        C_occ = self.c_mo[:, :self.nocc_stda]  # Shape: (n_ao, nocc_stda)
        for A in range(self.natom):
            ao_start, ao_end = self.aorange[A, 2:4]
            C_A_OCC = C_occ[ao_start:ao_end, :]  # Shape: (n_ao_A, nocc_stda)
            qij[A, :, :] = np.dot(C_A_OCC.T, C_A_OCC)  # Shape: (nocc_stda, nocc_stda)
        return qij

    def get_qab(self):
        """
        Calculate q_{ab}^A = sum_{μ in A} C_{μ a} C_{μ b} for all a, b, and A.
        """
        qab = np.zeros((self.natom, self.nvir_stda, self.nvir_stda))
        #C_vir = self.c_mo[:, self.viridx_stda]  # Shape: (n_ao, nvir_stda)
        C_vir = self.c_mo[:, self.nocc_stda:]  # Shape: (n_ao, nvir_stda)
        for A in range(self.natom):
            ao_start, ao_end = self.aorange[A, 2:4]
            C_A_vir = C_vir[ao_start:ao_end, :]  # Shape: (n_ao_A, nvir_stda)
            qab[A, :, :] = np.dot(C_A_vir.T, C_A_vir)  # Shape: (nvir_stda, nvir_stda)
        return qab

    def get_Amatrix(self,pairs1=None,pairs2=None):
        r'''
        Calculate A matrix
        A_{iajb}^{sTDA} = \delta_{ab}\delta_{ij}(E_a - E_i) + 2(ia|jb)' - (ij|ab)'
        '''
        if pairs1 is None and pairs2 is None:
            pairs1 = pairs2 = self.pairs
        elif pairs1 is not None and pairs2 is None:
            pairs2 = pairs1

        npairs1 = len(pairs1)
        npairs2 = len(pairs2)

        #print('npairs1',npairs1)
        #print('npairs2',npairs2)
        Amat = np.zeros((npairs1, npairs2)).astype(np.float32)
        Amat_s = np.zeros((npairs1, npairs2)).astype(np.float32)
        Amat_t = np.zeros((npairs1, npairs2)).astype(np.float32)
        nA = self.gammaJ.shape[0]

        for p, (i, a) in enumerate(pairs1):
            for q, (j, b) in enumerate(pairs2):
                if (i, a) == (j, b):
                    Amat[p, q] = self.mo_energy[a] - self.mo_energy[i]
 
        #print('diag_values',time.time())
        occ_off = self.nocc - self.nocc_stda
        vir_off = self.nocc
        i_idx = np.array(pairs1)[:, 0] - occ_off
        a_idx = np.array(pairs1)[:, 1] - vir_off

        j_idx = np.array(pairs2)[:, 0] - occ_off
        b_idx = np.array(pairs2)[:, 1] - vir_off
 
        Q = self.q_oo[:, i_idx[:, None], j_idx[None, :]].astype(np.float32)       # (nA, np1, np2)
        T = self.T[:, a_idx[:, None], b_idx[None, :]].astype(np.float32)       # (nA, np1, np2)
        nA, np1, np2 = Q.shape
        Q = Q.reshape(nA, -1)  # (nA, np1*np2)
        T = T.reshape(nA, -1)  # (nA, np1*np2)

        vJ_mat = np.sum(Q * T, axis=0).astype(np.float32).reshape(np1,np2)


        if self.singlet == True:
            Q1= self.q_ov[:, j_idx, b_idx].astype(np.float32)       # (nA, np2)
            Q2= self.q_ov[:, i_idx, a_idx].astype(np.float32)         # (nA, np1)
 
            temp = self.gammaK @ Q1
            vK_mat = 2.0 * (Q2.T @ temp)
 
            Amat_s = Amat + vK_mat - vJ_mat

        if self.triplet == True:
            Amat_t = Amat - vJ_mat

        if self.singlet == True and self.triplet == True:
            return Amat_s, Amat_t
        elif self.singlet == True and self.triplet == False:
            return Amat_s
        elif self.singlet == False and self.triplet == True:
            return Amat_t
        else:
            raise ValueError("Invalid combination of singlet and triplet")

    def get_Adict(self,prev_pairs=None):

        if self.Adict is None:
            self.Adict = {}
            npairs = len(self.pairs)

            A_keys = [
                (p, q, self.pairs[p], self.pairs[q])
                for p in range(npairs)
                for q in range(p,npairs)
            ]
 
            A = self.get_Amatrix()
            for (p, q, (i, a), (j, b)) in A_keys:
                key=(i,a,j,b)
                self.Adict[key] = A[p,q]
             
            return A

        else:
            npairs = len(self.pairs)

            pairs1 = self.pairs
            if prev_pairs is None:
                raise ValueError("Missing previous pairs!")
            else:
                pairs2 = list(set(self.pairs)-set(prev_pairs))

            Adict = {}
            if len(pairs2) > 0:
                A = self.get_Amatrix(pairs1=pairs1, pairs2=pairs2)
 
                for p, (i, a) in enumerate(pairs1):
                    for q, (j, b) in enumerate(pairs2):
                        key = (i, a, j, b)
                        Adict[key] = Adict.get(key,0) + A[p,q]
 
            self.Adict.update(Adict)

            A = np.zeros((npairs,npairs))
            for p in range(npairs):
                i, a = self.pairs[p]
                for q in range(p, npairs):
                    j, b = self.pairs[q]
                    if (i,a,j,b) in self.Adict.keys():
                        A[p, q] = A[q, p] = self.Adict[(i, a, j, b)]
                    elif (j,b,i,a) in self.Adict.keys():
                        A[p, q] = A[q, p] = self.Adict[(j, b, i, a)]
                    else:
                        raise ValueError("Missing values in A dict!")

            return A
          
    def analyze(self):
        r'''
        Diagonalize A matrix, print excitation energies and abs(configurations) > 0.1
        example:
        stda.analyze()
        output:
        State 1: 1.2356 eV
        16 -> 17 0.7071
        14 -> 17 0.1012
        State 2: 1.2399 eV
        15 -> 17 0.7071
        13 -> 18 -0.1012
        14 -> 19 -0.1002
        '''
        occ_off = self.nocc - self.nocc_stda
        vir_off = self.nocc
        i_idx = np.array(self.pairs)[:, 0] - occ_off
        a_idx = np.array(self.pairs)[:, 1] - vir_off
        molintx=self.molintx[i_idx,a_idx]
        molinty=self.molinty[i_idx,a_idx]
        molintz=self.molintz[i_idx,a_idx]

        A = self.get_Amatrix()
        #A = A.reshape(self.nocc_stda*self.nvir_stda, self.nocc_stda*self.nvir_stda)
        e, v = eigh(A)
        print('diag_A',time.time())
        e = e.real
        v = v.real

        print("Excitation energies (eV):")
        for i in range(self.nstates):
            dipole_moment=(v[:,i].T @ molintx)**2+(v[:,i].T @ molinty)**2+(v[:,i].T @ molintz)**2
            fl=e[i]*4.0*dipole_moment/3.0
            print(f"State {i+1}: {e[i]*27.2114:.4f} {fl:.4f}")
            for j in range(len(self.pairs)):
                if abs(v[j,i]) > 1e-3:
                    print(self.pairs[j][0]+1, ' -> ', self.pairs[j][1]+1, v[j,i])

    def analyze_pyscf(self, save_v=True, v_filename='eig_vec.dat', print_g=False):
        A = self.get_Amatrix()
        A = A.reshape(self.nocc_stda*self.nvir_stda, self.nocc_stda*self.nvir_stda)
        e, v = eigh(A)
        e = e.real[:self.nstates]
        v = v.real[:, :self.nstates]
        # turn v into (nstates, nocc_stda, nvir_stda)
        v = v.reshape(self.nocc_stda, self.nvir_stda, self.nstates)
        v = v.transpose(2, 0, 1)
        v *= np.sqrt(0.5)
        # expand v to (nstates, nocc, nvir)
        v_exp = []
        for i in range(self.nstates):
            v_exp.append([expand_matrix_lower_left(v[i], self.nocc, self.nvir), np.int64(0)])
        
        if save_v:
            np.savetxt(v_filename, np.array(v_exp))
        
        from pyscf import tddft
        mytd = tddft.TDA(self.mf)
        mytd.nstates = self.nstates
        mytd.singlet = self.singlet
        mytd.e = e
        mytd.xy = v_exp
        mytd._scf.mo_coeff = self.mf.mo_coeff
        mytd._scf.mo_occ = self.mf.mo_occ
        mytd.analyze()
        if print_g:
            trans_dip = mytd.transition_dipole()
            trans_m = mytd.transition_magnetic_dipole()
            print("State   absmu(10^-18 esu cm)   absm(10^-20 erg/G)   costheta         g              g*1000")
            print("-" * 90)  # Add a separator line
            for i in range(self.nstates):
                absmu_au = np.linalg.norm(trans_dip[i])
                absmu = absmu_au/0.393456 # Debye or 10^-18 esu cm
                absm_au = np.linalg.norm(trans_m[i])
                absm = absm_au*1.8548 #unit 10^-20 erg/G
                dotmium = np.dot(trans_dip[i], trans_m[i])
                if absmu_au < 1e-12 or absm_au < 1e-12:
                    costheta = 0.0
                    g = 0.0
                else:
                    costheta = dotmium / (absmu_au * absm_au)
                    g = 4*costheta*absm*(1E-20)/(absmu*(1E-18))
                print(f"{i+1:5d}   {absmu:20.6f}   {absm:18.6f}   {costheta:12.6f}   {g:12.6f}   {g*1000:12.6f}")

def expand_matrix_lower_left(submatrix, nocc, nvir):
    expanded = np.zeros((nocc, nvir), dtype=submatrix.dtype)
    nocc_stda, nvir_stda = submatrix.shape
    start_row = nocc - nocc_stda
    expanded[start_row : start_row + nocc_stda, :nvir_stda] = submatrix
    return expanded

if __name__ == '__main__':
    #from pyscf import gto
    #mol = gto.Mole()
    #mol.verbose = 0
    #mol.atom = "h2o.xyz"
    #mol.symmetry = False
    #mol.cart=True
    #mol.basis = '631g*'
    #mol.build()
    #mf = mol.RKS()
    #mf.xc = 'pbe0'
    #mf.kernel()
    #aorange = mol.aoslice_by_atom()
    #print(aorange)
    #stda = sTDA(mol, mf, aorange, ax=0.20, nocc=4, nvir=9)
    #stda.analyze()

    #pairs=[(2,5),(2,6),(2,7),(3,5),(3,6),(3,7),(4,5),(4,6),(4,7)]

    #print('ini',time.time())
    mol =None
    aorange = parse_gto_aorange("./file.molden", cart=True)
    #print('gto',time.time())
    mf = MoldenData("./file.molden")
    #print('molden',time.time())
    stda = sTDA(mol, mf, aorange, ax=0.20, nocc=5, nvir=14)
    stda.analyze()

