import re
import numpy as np

def parse_gto_aorange(filename,cart=True):
    """
    Get AO range for each atom from molden file
    return list: [(atom_index, ao_start, ao_end), ...]
    """
    lines=[]
    recording = False
    with open("molden.input", 'w') as fout:
        with open(filename, 'r') as f:
            for raw_line in f:
                line=raw_line.rstrip('\n')
                fout.write(line+"\n")
                if '[GTO]' in line:
                    recording=True
                if recording:
                    lines.append(line)
                if '[MO]' in line:
                    break
    try:
        start = next(i for i, l in enumerate(lines) if '[GTO]' in l)
    except StopIteration:
        raise ValueError("[GTO] not found.")

    ao_counter = 0
    result = []
    current_atom = None
    atom_ao_start = 0

    for i in range(start + 1, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        if line.startswith('['):
            break

        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            if current_atom is not None:
                result.append((current_atom,current_atom, atom_ao_start, ao_counter))
            current_atom = int(parts[0])
            atom_ao_start = ao_counter
            continue

        if parts[0].lower() in ['s', 'p', 'd', 'f', 'g']:
            ltype = parts[0].lower()
            if cart:
               ncomp = {'s': 1, 'p': 3, 'd': 6, 'f': 10, 'g': 15}[ltype]
            else:
               ncomp = {'s': 1, 'p': 3, 'd': 5, 'f': 7, 'g': 9}[ltype]
            ao_counter += ncomp
            continue

    if current_atom is not None:
        result.append((current_atom, current_atom, atom_ao_start, ao_counter))

    return np.array(result)



