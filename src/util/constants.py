
DF_COLUMNS = ["DATA_ID","REACTANT_SMI", "PRODUCT_R_SMI"
 , "PRODUCT_S_SMI", "SOLVENT_SMI", "ION_SMI"
 , "LIGAND_SMI", "TEMP", "PRESSURE", "EE"]

UNIMOL_EMBED_DIM = {
      "84m": 768
    , "164m": 768
    , "310m": 1024
    , "570m": 1536
    , "1.1B": 1536
}

METAL_TYPE = ['Rh','Ir','Co','Ru','Ni','Pd', 'Mn', "Fe"]
ELEMENT_LIST = ['Fe', 'Cr', 'Pd', 'Mn', 'Rh', 'C', 'P', 'Br', 'H', 'O', 'F', 'N', 'S', 'Cl', 'Co', 'Ni', 'Ru', 'Ir', "B"]
COORD_TYPE = ['N', 'O', 'S', 'P']
HOMO_LUMO_GAP_NUM_2 = 2
HOMO_LUMO_GAP_NUM_4 = 4