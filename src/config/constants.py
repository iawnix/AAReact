
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

METAL_TYPE = ['Rh','Ir','Co','Ru','Ni','Pd', 'Mn', "Fe", 'Cr']
ELEMENT_LIST = ['Fe', 'Cr', 'Pd', 'Mn', 'Rh', 'C', 'P', 'Br', 'H', 'O', 'F', 'N', 'S', 'Cl', 'Co', 'Ni', 'Ru', 'Ir', "B"]
COORD_TYPE = ['N', 'O', 'S', 'P']
HOMO_LUMO_GAP_NUM_2 = 2
HOMO_LUMO_GAP_NUM_4 = 4

XTB_BACHEND="/opt/xtb/6.7.1/bin/xtb"
OBABEL_BACHEND="/usr/bin/obabel"
XTB_WORK_SCRATCH="/home/iaw/DATA2/AAReact/DataSet/.xtb_tmp"


RF_PARAM_GRID = {
  'n_estimators': [1, 3, 5, 10, 20, 30, 50, 70, 100, 120, 130, 140, 145, 150, 155, 160, 200, 300], 
  'max_depth': [1, 3, 5, 8, 10, 20, 30, 40, 50, None], 
  'min_samples_split': [2, 5, 10, 30, 50],  
  'min_samples_leaf': [4, 5, 10, 30, 50], 
  "ccp_alpha": [0.00001, 0.001, 0.01, 0.05, 0.1, 0.2]
}

XGB_PARAM_GRID = {
  'n_estimators': [1, 3, 5, 7, 10, 15, 20, 25, 30, 50],  #1, 3, 5, 10, 20 , 30, 50, 70, 100, 120, 130, 140, 145, 150, 155, 160, 200, 太容易过拟合了
  'max_depth': [1, 3, 5, 8, 10, 20], # , 30, 40, 50
  "reg_alpha": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
  'reg_lambda': [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
  'learning_rate': [0.01, 0.1],      
  'min_child_weight': [5, 10],         
  'subsample': [0.5, 0.6, 0.8],  
  'colsample_bytree': [0.8, 0.9],
}

LGB_PARAM_GRID = {
  'n_estimators': [1, 3, 5, 7, 10, 15, 20, 25, 30, 50],
  'max_depth': [1, 3, 5, 8, 10, 20],
  'learning_rate': [0.01, 0.1, 0.2],    
  "reg_alpha": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
  'reg_lambda': [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
  'num_leaves': [31, 40, 45, 50],  
  'min_child_samples': [20, 50], 
  'subsample': [0.8, 0.9, 1.0], 
  'colsample_bytree': [0.8, 0.9, 1.0], 
}

SOAP_FIX_PARAMETER = {
      "rcut": 6.0
    , "nmax": 4
    , "lmax": 3
}

ACSF_FIX_PARAMETER = {
      "rcut": 6.0
    , "g2_params": [[1, 1], [1, 2], [1, 3]]
    , "g4_params": [[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]]
}

MBTR_FIX_PARAMETER = {
     "geometry": {"function": "inverse_distance"}
    , "grid": {"min": 1, "max": 5, "n": 200, "sigma": 0.05}
    , "weighting": {"function": "exp", "scale": 1, "threshold": 1e-2}
}

LMBTR_FIX_PARAMETER = {
      "geometry": {"function": "inverse_distance"}
    , "grid": {"min": 1, "max": 5, "n": 200, "sigma": 0.05}
    , "weighting": {"function": "exp", "scale": 1, "threshold": 1e-2}
}