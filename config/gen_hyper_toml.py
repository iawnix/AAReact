

def touch_toml(out_fp: str, data_for_train_path: str, model_name: str, desc_type : str, batch_type: str):
    with open(out_fp, "w+") as F:
        F.writelines("[Hyper]\n")
        F.writelines("data_x = \"{}/{}/{}_data_x.npy\"\n".format(data_for_train_path, desc_type, batch_type))
        F.writelines("data_y = \"{}/{}/{}_data_y.npy\"\n".format(data_for_train_path, desc_type, batch_type))
        F.writelines("x_label = \"{}/{}/{}_x_label.pkl\"\n".format(data_for_train_path, desc_type, batch_type))
        F.writelines("data_class = \"{}/{}/{}_data_class.pkl\"\n".format(data_for_train_path, desc_type, batch_type))
        F.writelines("seed = 1\n")
        F.writelines("test_size = 0.2\n")
        F.writelines("cv = 5\n")
        F.writelines("n_cpu = 5\n")
        F.writelines("\n")
        F.writelines("[Model]\n")
        F.writelines("name = \"{}\"\n".format(model_name))
        F.writelines("n_cpu = 5\n")
        F.writelines("params_save = \"/home/iaw/DATA2/AAReact/train/output/hyper_log/{}_{}_seed_0-1_test_0-2_cv_5_hyper.log\"\n".format(model_name, desc_type))



model_s = ["lgb", "xgb", "rf"]
comb_s = [("rdkit", "soap"), ("soap", "xtb"), ("rdkit", "xtb"), ("soap", "acsf"), ("acsf", "xtb"), ("rdkit", "acsf")
               , ("rdkit", "soap", "xtb"), ("rdkit", "soap", "acsf"),("rdkit", "xtb", "acsf"), ("soap", "xtb", "acsf") 
               , ("rdkit", "soap", "xtb", "acsf"), ("rdkit", ), ("xtb", ), ("soap", ), ("acsf", )]
for i_m in model_s:
    for i_c in comb_s:
        #print(i_c, type(i_c), len(i_c) == 1)
        if len(i_c) == 1:
            i_desc_type = i_c[0]
        else:
            i_desc_type = "_".join(i_c) 
        touch_toml(
              out_fp = "/home/iaw/DATA2/AAReact/config/hyper_ml_{}_{}.toml".format(i_m, i_desc_type)
            , data_for_train_path = "/home/iaw/DATA2/AAReact/DataSet/Data_All/4_data_for_train/"
            , model_name = i_m
            , desc_type = i_desc_type
            , batch_type = "train_test"
        )
