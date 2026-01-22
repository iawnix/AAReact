def mol2_xtb(imol2: str, itraj: str, omol2: str) -> None:
    
    file = []
    with open(imol2, "r") as F:
        for lines in F.readlines():
            file.append(lines.rstrip("\n"))

    bond_table_block = None
    n_atom_bond_line_num = None
    for i,line in enumerate(file):

        if line.startswith("@<TRIPOS>MOLECULE"):
            n_atom_bond_line_num = i+2

        if line.startswith("@<TRIPOS>BOND"):
            bond_table_line_num = i
            bond_table_block = file[bond_table_line_num:]
            break
    if bond_table_block == None or n_atom_bond_line_num == None:
        print("Error[iaw]>: please check {}".format(imol2))
    n_atom_bond = [i for i in file[n_atom_bond_line_num].replace("\t", " ").replace("  "," ").split(" ") if i != ""]
    n_atom = n_atom_bond[0]
    n_bond = n_atom_bond[1]

    file = []
    with open(itraj, "r") as F:
        for lines in F.readlines():
            file.append(lines.rstrip("\n"))
    energy_line_num = []
    for i,line in enumerate(file):
        if line.startswith(" energy:"):
            energy_line_num.append(i)
    last_traj_block = file[energy_line_num[-1]-1:]

    with open(omol2, "w+") as F:
        F.writelines("# Title\n# Create by IAW[XTB+Gaussian16]\n# \n@<TRIPOS>MOLECULE\nMolecule Name\n{} {}\nSMALL\nNO_CHARGES\n\n\n".format(n_atom, n_bond))
        F.writelines("@<TRIPOS>ATOM\n")
        for i, i_ss in enumerate(last_traj_block[2:]):
            i_txt = [_ for _ in i_ss.replace("\t", " ").replace("  ", " ").split(" ") if _ != ""]
            i_type, i_x, i_y, i_z = i_txt
            F.writelines("{} {}{}    {:.4f}    {:.4f}    {:.4f}  {}\n".format(i+1, i_type,i+1, float(i_x), float(i_y), float(i_z), i_type))
        for i_ss in bond_table_block:
            F.writelines("{}\n".format(i_ss))

if __name__ == "__main__":

    imol2 = "/home/iaw/DATA2/AAReact/src/tool/xtb_mol2/CAT-9.mol2"
    itraj = "/home/iaw/DATA2/AAReact/src/tool/xtb_mol2/xtbtrj.xyz"
    omol2 = "/home/iaw/DATA2/AAReact/src/tool/xtb_mol2/CAT-9_new2.mol2"


    mol2_xtb(imol2, itraj, omol2)