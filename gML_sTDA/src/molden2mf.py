import re
import numpy as np

class MoldenData:
    def __init__(self, filename):
        self.filename = filename
        self.mo_energy = []
        self.mo_coeff = []
        self.mo_occ = []
        self.coord = []
        self.natm = 0              
        self.atom_charge = []      

        self._parse_molden()

    def _parse_molden(self):
        """Parse .molden file and extract coords, mo_energy, mo_coeff, mo_occ, atom_charge"""
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        # --- [Atoms] ---
        atom_block = self._extract_block(lines, "[Atoms]")
        if not atom_block:
            raise ValueError("[Atoms] not found!")
        self.coord, self.atom_charge = self._parse_atoms(atom_block)
        self.natm = len(self.coord)

        # --- [MO] ---
        mo_block = self._extract_block(lines, "[MO]")
        if not mo_block:
            raise ValueError("[MO] not found!")
        self.mo_energy, self.mo_coeff, self.mo_occ = self._parse_mo(mo_block)

    def _extract_block(self, lines, tag):
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

    def _parse_atoms(self, atom_lines):
        coords = []
        charges = []
        for line in atom_lines:
            if not line.strip() or line.startswith('['):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                charge = int(parts[2])
                x, y, z = map(float, parts[3:6])
            except Exception:
                continue
            charges.append(charge)
            coords.append([x, y, z])
        return np.array(coords), np.array(charges)

    def _parse_mo(self, mo_lines):
        mo_energy = []
        mo_coeff = []
        mo_occ = []
        current_coeff = []
        current_occ = None

        for line in mo_lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('Ene='):
                if current_coeff:
                    mo_coeff.append(np.array(current_coeff, dtype=float))
                    mo_occ.append(2.0 if current_occ is None else current_occ)
                    current_coeff = []
                    current_occ = None
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

        mo_coeff = np.array(mo_coeff, dtype=float).T  # shape = (nAO, nMO)
        mo_energy = np.array(mo_energy, dtype=float)
        mo_occ = np.array(mo_occ, dtype=float)

        if np.all(mo_occ == 0):
            n_occ = len(mo_occ) // 2  
            mo_occ[:n_occ] = 2.0

        return mo_energy, mo_coeff, mo_occ

    def summary(self):
        print(f"Molden file: {self.filename}")
        print(f"anatm: {self.natm}")
        print(f"atom_charge: {self.atom_charge}")
        print(f"orbitals: {len(self.mo_energy)}")
        print(f"AO × MO : {self.mo_coeff.shape}")
        print(f"occupation: {self.mo_occ[:10]}")

if __name__ == "__main__":
    molden = MoldenData("h2o.molden")
    molden.summary()

    print("\ncoord coord:")
    print(molden.coord)

    print("\n5 MO energies:")
    print(molden.mo_energy[:5])

    print("\n10 occupations:")
    print(molden.mo_occ[:10])

    print("\natom charges:")
    print(molden.atom_charge)

