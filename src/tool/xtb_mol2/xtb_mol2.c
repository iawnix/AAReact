# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <ctype.h>

# define MAX_LINE 1024
# define MAX_ATOM_NAME 8
# define MAX_ATOM_TYPE 4

typedef struct mol_bond {
    int bond_idx;
    int atom_idx1;
    int atom_idx2;
    int bond_type;
} MOL2_BOND;

typedef struct mol_atom {
    char atom_name[MAX_ATOM_NAME];
    double x;
    double y;
    double z;
    char atom_type[MAX_ATOM_TYPE];
} MOL2_ATOM;


int read_mol2(const char *fpath, MOL2_BOND** bond_table){
    int n_atom = 0;
    int n_bond = 0;
    int i_bond = 0;
    // 0, 初始, 1找到molecule字段, 2, 找到bond字段, 3, 读取完毕
    int flag = 0;               
    char line[MAX_LINE];

    FILE * fp = fopen(fpath, "r");
    if (fp == NULL){
        printf("Error[iaw]>: can not open the file %s\n", fpath);
        return -1;
    }


    while (fgets(line, sizeof(line), fp) != NULL && flag != 3){

        if (strncmp(line, "@<TRIPOS>MOLECULE", 15) == 0) { flag = 1; continue; }
        if (strncmp(line, "@<TRIPOS>BOND", 12) == 0) { flag = 2; continue; }
        if (flag == 2 && strncmp(line, "@<TRIPOS>", 9) == 0){ flag = 3; continue; }

        if (flag == 1){
            static int line_count = 0;
            line_count++;
            if (line_count == 2) { 
                sscanf(line, "%d %d", &n_atom, &n_bond); 
                flag = 0;
                *bond_table = (MOL2_BOND* )malloc(n_bond * sizeof(MOL2_BOND));
                if (*bond_table == NULL) {
                    printf("Error[iaw]>: Fail to malloc the men for bond table!\n");
                    fclose(fp);
                    return -1;
                }
            }
            /*
            else
            {
                printf("Info[iaw]>: line_count=%d, line: %s\n", line_count, line);
            }
            */
            if (line_count > 3) { flag = 0; }
        }

        if (flag == 2 && i_bond < n_bond ) {
            MOL2_BOND* i_bond_table = &(*bond_table)[i_bond];
            sscanf(line, "%d %d %d %d", &i_bond_table->bond_idx, &i_bond_table->atom_idx1, &i_bond_table->atom_idx2, &i_bond_table->bond_type);

            i_bond++;
            if (i_bond == n_bond){ flag = 3; }
        }
    }

    fclose(fp);
    return n_bond;
}


void strrev(char *str) {
    if (str == NULL || *str == '\0') return;
    int len = strlen(str);
    int i = 0, j = len - 1;
    while (i < j) {
        char temp = str[i];
        str[i] = str[j];
        str[j] = temp;
        i++;
        j--;
    }
}

int read_traj(const char* fpath, MOL2_ATOM** atom_s){
    int n_atom = 0;
    char line[MAX_LINE];
    long file_size;
    long cur_pos;
    long last_atom_count_pos = -1; 
    int found = 0;

    FILE* fp = fopen(fpath, "rb"); 
    if (fp == NULL) {
        printf("Error[iaw]>: can not open the file %s\n", fpath);
        return -1;
    }


    fseek(fp, 0, SEEK_END);
    file_size = ftell(fp);
    cur_pos = file_size - 1;


    char buffer[MAX_LINE] = {0};
    int buf_idx = 0;
    while (cur_pos >= 0 && !found) {
        fseek(fp, cur_pos, SEEK_SET);
        char ch = fgetc(fp);


        if (ch == '\n' || cur_pos == 0) {
            if (cur_pos == 0) {
                buffer[buf_idx++] = ch;
            }
            buffer[buf_idx] = '\0';
            strrev(buffer);
          

            char *trim_buf = buffer;
            while (isspace((unsigned char)*trim_buf)) trim_buf++;
            if (*trim_buf != '\0') {
          
                int temp_n;
                if (sscanf(trim_buf, "%d", &temp_n) == 1 && temp_n > 0) {
            
                    char check_str[MAX_LINE];
                    sprintf(check_str, "%d", temp_n);
                    strrev(check_str); 
                    if (strcmp(trim_buf, check_str) == 0) {
                        n_atom = temp_n;
                        last_atom_count_pos = cur_pos + 1;
                        found = 1;
                        break;
                    }
                }
            }
            buf_idx = 0;
            memset(buffer, 0, sizeof(buffer));
        } else if (!isspace((unsigned char)ch)) {
            if (buf_idx < MAX_LINE - 1) {
                buffer[buf_idx++] = ch;
            }
        }
        cur_pos--;
    }

    if (!found || n_atom <= 0) {
        printf("Error[iaw]>: can not find n_atom lines: %s\n", fpath);
        fclose(fp);
        return -1;
    }


    fseek(fp, last_atom_count_pos, SEEK_SET);
    if (fgets(line, MAX_LINE, fp) == NULL) {
        printf("Error[iaw]>: can not read n_atom line\n");
        fclose(fp);
        return -3;
    }

 
    if (fgets(line, MAX_LINE, fp) == NULL) {
        printf("Error[iaw]>: can not read energy\n");
        fclose(fp);
        return -3;
    }

    // 3. 分配内存并读取最后一块的原子坐标
    *atom_s = (MOL2_ATOM *)malloc(n_atom * sizeof(MOL2_ATOM));
    if (*atom_s == NULL) {
        printf("Error[iaw]>: malloc failed for MOL2_ATOM array\n");
        fclose(fp);
        return -5;
    }

    int coord_count = 0;
    while (fgets(line, MAX_LINE, fp) != NULL && coord_count < n_atom) {
        char *start = line;
        while (isspace((unsigned char)*start)) start++;
        char *end = start + strlen(start) - 1;
        while (end > start && isspace((unsigned char)*end)) end--;
        *(end + 1) = '\0';

        if (*start == '\0') continue; 

        char atom_type[MAX_ATOM_TYPE];
        double x, y, z;
        if (sscanf(start, "%3s %lf %lf %lf", atom_type, &x, &y, &z) == 4) {
            snprintf((*atom_s)[coord_count].atom_name, MAX_ATOM_NAME, "%s%d", atom_type, coord_count+1);
            strncpy((*atom_s)[coord_count].atom_type, atom_type, MAX_ATOM_TYPE-1);
            (*atom_s)[coord_count].atom_type[MAX_ATOM_TYPE-1] = '\0';
            (*atom_s)[coord_count].x = x;
            (*atom_s)[coord_count].y = y;
            (*atom_s)[coord_count].z = z;
            coord_count++;
        } else {
            printf("Warning[iaw]>: 坐标行格式错误，跳过: %s\n", line);
        }
    }

    // 验证读取的原子数是否匹配
    if (coord_count != n_atom) {
        printf("Error[iaw]>: n_atom[%d] != n_line[%d]\n", n_atom, coord_count);
        free(*atom_s);
        *atom_s = NULL;
        fclose(fp);
        return -4;
    }

    fclose(fp);
    return n_atom;
}


int write_mol2(const char* fpath, MOL2_BOND* bond_table, MOL2_ATOM* atom_s, int n_bond, int n_atom){
    FILE* fp = NULL;

    fp = fopen(fpath, "w+");
    if (fp == NULL){
        printf("Error[iaw]>: can not open the file %s\n", fpath);
        return -1;
    }
    fprintf(fp, "# Title\n# Create by IAW[XTB+Gaussian16]\n# \n@<TRIPOS>MOLECULE\nMolecule Name\n%d %d\nSMALL\nNO_CHARGES\n\n\n", n_atom, n_bond);
    fprintf(fp, "@<TRIPOS>ATOM\n");
    for (int i = 0; i < n_atom; i++) {
        fprintf(fp, "%d %s   %.4f    %.4f    %.4f %s\n", i+1, atom_s[i].atom_name, atom_s[i].x, atom_s[i].y, atom_s[i].z, atom_s[i].atom_type);
    }
    fprintf(fp, "@<TRIPOS>BOND\n");
    for (int i = 0; i < n_bond; i++){
        fprintf(fp, "%d %d %d %d\n", bond_table[i].bond_idx, bond_table[i].atom_idx1, bond_table[i].atom_idx2, bond_table[i].bond_type);
    }
    fclose(fp);

    return 0;
}



void print_help(){
    printf("#++++++++++++++++++++++++++++++++++# xtb_mol2 Usage #++++++++++++++++++++++++++++++++++#\n");
    printf("Author: iawnix <ECNU>\n");
    printf("Email: iawhaha@163.com\n");
    printf("Compile[Linux C99]: gcc xtb_mol2.c -o xtb_mol2 -g \n");
    printf("#--------------------------------------------------------------------------------------#\n");
    printf("1. -h\n");
    printf("    help\n");
    printf("2. -imol2 xx.mol2 -itraj xxx.xyz -omol2 xx.mol2\n");
    printf("#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#\n");
}


int main(int argc, char* argv[]){

    MOL2_BOND *bond_table = NULL;
    MOL2_ATOM *atoms = NULL;

    if (argc == 7){
        if (strcmp(argv[1], "-imol2") == 0 && strcmp(argv[3], "-itraj") == 0 && strcmp(argv[5], "-omol2") == 0){
            int n_bond = read_mol2(argv[2], &bond_table);
            int n_atom = read_traj(argv[4], &atoms);
            write_mol2(argv[6], bond_table, atoms, n_bond, n_atom);

            free(bond_table);
            free(atoms);
        }
        else { print_help(); }
    }
    else { print_help(); printf("argc = %d", argc);}


    return 0;
}