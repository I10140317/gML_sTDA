# molden_fast.py
import re
import numpy as np
from pathlib import Path

_use_c = False
_c = None
try:
    import ctypes
    so_candidates = list(Path(__file__).resolve().parent.glob("molden_parser*.so"))
    if so_candidates:
        _c = ctypes.CDLL(str(so_candidates[0]))
        class _CMoldenData(ctypes.Structure):
            _fields_ = [
                ("natm", ctypes.c_int),
                ("nmo", ctypes.c_int),
                ("nao", ctypes.c_int),
                ("atom_charge", ctypes.POINTER(ctypes.c_int)),
                ("coord", ctypes.POINTER(ctypes.c_double)),
                ("mo_energy", ctypes.POINTER(ctypes.c_double)),
                ("mo_occ", ctypes.POINTER(ctypes.c_double)),
                ("mo_coeff", ctypes.POINTER(ctypes.c_double)),
            ]
        _c.read_molden_file.restype = ctypes.POINTER(_CMoldenData)
        _c.free_molden.argtypes = [ctypes.POINTER(_CMoldenData)]
        _use_c = True
except Exception:
    _use_c = False
    _c = None


class MoldenData:
    """
      - coord: (natm,3) float64
      - mo_energy: (nmo,)
      - mo_coeff: (nao,nmo)  
      - mo_occ: (nmo,)       
      - natm: int
      - atom_charge: (natm,) int
    """
    def __init__(self, filename: str):
        self.filename = filename
        self.coord = None
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = None
        self.natm = 0
        self.atom_charge = None

        if _use_c:
            self._load_via_c(filename)
        else:
            self._load_via_python(filename)

    def _load_via_c(self, filename: str):
        ptr = _c.read_molden_file(filename.encode("utf-8"))
        if not ptr:
            raise RuntimeError(f"Failure to read: {filename}")
        data = ptr.contents
        try:
            natm, nmo, nao = data.natm, data.nmo, data.nao

            coord = np.ctypeslib.as_array(data.coord, shape=(natm*3,))
            atom_charge = np.ctypeslib.as_array(data.atom_charge, shape=(natm,))
            mo_energy = np.ctypeslib.as_array(data.mo_energy, shape=(nmo,))
            mo_occ = np.ctypeslib.as_array(data.mo_occ, shape=(nmo,))
            mo_coeff = np.ctypeslib.as_array(data.mo_coeff, shape=(nao*nmo,))

            self.coord = coord.reshape(natm, 3).copy()
            self.atom_charge = atom_charge.astype(np.int32, copy=True)
            self.mo_energy = mo_energy.astype(np.float64, copy=True)
            self.mo_occ = mo_occ.astype(np.float64, copy=True)
            self.mo_coeff = mo_coeff.reshape(nao, nmo).astype(np.float64, copy=True)

            self.natm = int(natm)

            if np.all(self.mo_occ == 0):
                n_occ = self.mo_energy.size // 2
                occ = np.zeros_like(self.mo_occ)
                occ[:n_occ] = 2.0
                self.mo_occ = occ
        finally:
            _c.free_molden(ptr)

    def _load_via_python(self, filename: str):
        with open(filename, 'r') as f:
            lines = f.readlines()

        atom_block = self._extract_block(lines, "[Atoms]")
        if not atom_block:
            raise ValueError("[Atoms] not found!")
        coord, charges = self._parse_atoms(atom_block)
        self.coord = coord
        self.atom_charge = charges.astype(np.int32, copy=False)
        self.natm = int(len(coord))

        mo_block = self._extract_block(lines, "[MO]")
        if not mo_block:
            raise ValueError("[MO] not found!")
        self.mo_energy, self.mo_coeff, self.mo_occ = self._parse_mo(mo_block)

    @staticmethod
    def _extract_block(lines, tag):
        start, end = None, None
        for i, line in enumerate(lines):
            if line.strip().startswith(tag):
                start = i + 1
                break
        if start is None:
            return []

        for j in range(start, len(lines)):
            if lines[j].strip().startswith('['):
                end = j
                break
        if end is None:
            end = len(lines)
        return lines[start:end]

    @staticmethod
    def _parse_atoms(atom_lines):
        coords, charges = [], []
        for line in atom_lines:
            s = line.strip()
            if not s or s.startswith('['):
                continue
            parts = s.split()
            if len(parts) < 6:
                continue
            try:
                z = int(parts[2])
                x, y, zc = map(float, parts[3:6])
            except Exception:
                continue
            charges.append(z)
            coords.append([x, y, zc])
        return np.array(coords, dtype=np.float64), np.array(charges, dtype=np.int32)

    @staticmethod
    def _parse_mo(mo_lines):
        mo_energy, mo_coeff, mo_occ = [], [], []
        current_coeff, current_occ = [], None

        for raw in mo_lines:
            line = raw.strip()
            if not line:
                continue

            if line.startswith('Ene='):
                if current_coeff:
                    mo_coeff.append(np.array(current_coeff, dtype=float))
                    mo_occ.append(2.0 if current_occ is None else current_occ)
                    current_coeff, current_occ = [], None
                mo_energy.append(float(line.split('=')[1]))
            elif line.startswith('Occup='):
                try:
                    current_occ = float(line.split('=')[1])
                except Exception:
                    current_occ = None
            elif re.match(r'^\d+', line):
                parts = line.split()
                if len(parts) == 2:
                    current_coeff.append(float(parts[1]))

        if current_coeff:
            mo_coeff.append(np.array(current_coeff, dtype=float))
            mo_occ.append(2.0 if current_occ is None else current_occ)

        C = np.array(mo_coeff, dtype=float).T  # (nao, nmo)
        E = np.array(mo_energy, dtype=float)
        OCC = np.array(mo_occ, dtype=float)

        if np.all(OCC == 0):
            n_occ = len(OCC) // 2
            OCC[:n_occ] = 2.0
        return E, C, OCC

    def summary(self):
        print(f"Molden file: {self.filename}")
        print(f" (natm): {self.natm}")
        print(f" atom_charge: {self.atom_charge}")
        print(f"Orbitals: {self.mo_energy.size}")
        print(f"AO × MO: {self.mo_coeff.shape}")
        print(f"occupation: {self.mo_occ[:10]}")


if __name__ == "__main__":
    m = MoldenData("h2o.molden")
    m.summary()




