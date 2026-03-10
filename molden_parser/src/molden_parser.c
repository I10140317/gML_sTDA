#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef struct {
    int natm;
    int nmo;
    int nao;
    int *atom_charge;
    double *coord;
    double *mo_energy;
    double *mo_occ;
    double *mo_coeff;
} MoldenData;

static int is_valid_atom_line(const char *line) {
    char sym[8];
    int idx, z;
    double x, y, zc;
    int n = sscanf(line, "%7s %d %d %lf %lf %lf", sym, &idx, &z, &x, &y, &zc);
    return (n == 6 && isalpha(sym[0]));
}

static int is_coeff_line(const char *line) {
    while (isspace(*line)) line++;
    return isdigit(*line);
}

MoldenData* read_molden(const char* filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "无法打开文件: %s\n", filename);
        return NULL;
    }

    MoldenData *data = calloc(1, sizeof(MoldenData));
    char line[512];
    int in_atoms = 0, in_mo = 0;
    int atom_count = 0, nmo = 0, nao_total = 0;

    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "[Atoms]")) { in_atoms = 1; in_mo = 0; continue; }
        if (strstr(line, "[MO]"))    { in_atoms = 0; in_mo = 1; continue; }
        if (strstr(line, "[GTO]"))   { in_atoms = 0; in_mo = 0; }

        if (in_atoms && is_valid_atom_line(line))
            atom_count++;

        if (in_mo) {
            if (strncmp(line, "Ene=", 4) == 0)
                nmo++;
            else if (is_coeff_line(line))
                nao_total++;
        }
    }

    rewind(fp);

    int nao = (nmo > 0) ? nao_total / nmo : 0;
    data->natm = atom_count;
    data->nmo = nmo;
    data->nao = nao;

    data->atom_charge = malloc(sizeof(int) * atom_count);
    data->coord = malloc(sizeof(double) * 3 * atom_count);
    data->mo_energy = malloc(sizeof(double) * nmo);
    data->mo_occ = calloc(nmo, sizeof(double));
    data->mo_coeff = malloc(sizeof(double) * nao * nmo);

    in_atoms = in_mo = 0;
    int atom_idx = 0, mo_idx = -1, coeff_idx = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "[Atoms]")) { in_atoms = 1; in_mo = 0; continue; }
        if (strstr(line, "[MO]"))    { in_atoms = 0; in_mo = 1; continue; }
        if (strstr(line, "[GTO]"))   { in_atoms = 0; in_mo = 0; }

        // ---- Atoms ----
        if (in_atoms && is_valid_atom_line(line)) {
            char sym[8];
            int idx, z;
            double x, y, zc;
            sscanf(line, "%7s %d %d %lf %lf %lf", sym, &idx, &z, &x, &y, &zc);
            data->atom_charge[atom_idx] = z;
            data->coord[3*atom_idx+0] = x;
            data->coord[3*atom_idx+1] = y;
            data->coord[3*atom_idx+2] = zc;
            atom_idx++;
        }

        // ---- MO ----
        if (in_mo) {
            if (strncmp(line, "Ene=", 4) == 0) {
                mo_idx++;
                coeff_idx = 0;
                data->mo_energy[mo_idx] = atof(line + 4);
                continue;
            }
            if (strncmp(line, "Occup=", 6) == 0) {
                data->mo_occ[mo_idx] = atof(line + 6);
                continue;
            }
            if (is_coeff_line(line)) {
                int ao_idx;
                double coeff;
                if (sscanf(line, "%d %lf", &ao_idx, &coeff) == 2) {
                    if (coeff_idx < data->nao) {
                        int pos = coeff_idx * data->nmo + mo_idx;
                        data->mo_coeff[pos] = coeff;
                    }
                    coeff_idx++;
                }
            }
        }
    }

    fclose(fp);
    return data;
}

#ifdef _WIN32
#define API_EXPORT __declspec(dllexport)
#else
#define API_EXPORT
#endif

API_EXPORT MoldenData* read_molden_file(const char* filename) {
    return read_molden(filename);
}

API_EXPORT void free_molden(MoldenData* data) {
    if (!data) return;
    free(data->atom_charge);
    free(data->coord);
    free(data->mo_energy);
    free(data->mo_occ);
    free(data->mo_coeff);
    free(data);
}

