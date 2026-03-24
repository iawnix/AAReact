function model_predict() {
	rea=$1
	sol=$2
	cat=$3
	temp=$4
	press=$5
	save_feat=$6
	model=$7
	feat_label=$8
	python ../src/AHO_predict.py --task "ee" \
		--rea $rea \
		--sol $sol \
		--cat $cat \
		--temp $temp \
		--pressure $press \
		--model $model \
		--feat_label $feat_label \
		--verbose 0 \
		--save_feat $save_feat

}

echo "#+++++++++++++++++++++++++++++++++++++++++++++++++# Rea #++++++++++++++++++++++++++++++++++++++++++++++#"

rf_model_1_rdkit="/home/iaw/DATA2/AAReact/train/rf_model_seed_1_split_0-2_hyper1_rdkit.pkl"
xgb_model_1_rdkit="/home/iaw/DATA2/AAReact/train/xgb_model_seed_1_split_0-2_hyper1_rdkit.pkl"
xgb_model_2_rdkit="/home/iaw/DATA2/AAReact/train/xgb_model_seed_1_split_0-2_hyper2_rdkit.pkl"
xgb_model_2_rdkit_soap="/home/iaw/DATA2/AAReact/train/xgb_model_seed_1_split_0-2_hyper2_rdkit_soap.pkl"
xgb_model_2_rdkit_soap_xtb="/home/iaw/DATA2/AAReact/train/xgb_model_seed_1_split_0-2_hyper2_rdkit_soap_xtb.pkl"

rdkit_3_feat_label="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_3_x_label.pkl"
rdkit_soap_3_feat_label="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_soap_3_x_label.pkl"
rdkit_soap_xtb_3_feat_label="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_soap_xtb_3_x_label.pkl"

rea_smi_1="[H]OC(=O)C(C1=C([H])C([H])=C([H])C([H])=C1[H])=C([H])C([H])([H])[H]"
rea_smi_2="[H]OC(=O)C(=C([H])C1=C([H])C([H])=C([H])C([H])=C1[H])C1=C([H])C([H])=C([H])C([H])=C1[H]"
rea_smi_3="[H]OC(=O)C(C1=C([H])C([H])=C([H])C([H])=C1[H])=C(C([H])([H])[H])C([H])([H])[H]"
rea_smi_4="[H]OC(=O)C(C1=C([H])C([H])=C(Cl)C([H])=C1[H])=C(C([H])([H])[H])C([H])([H])[H]"

rea_sdf_1="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/3_opted_mol2/sdf/REA-1.sdf"
rea_sdf_2="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/3_opted_mol2/sdf/REA-2.sdf"
rea_sdf_3="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/3_opted_mol2/sdf/REA-3.sdf"
rea_sdf_4="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/3_opted_mol2/sdf/REA-4.sdf"

cat_smi_71="[H]c1c([H])c2[P](c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])(c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])->[Ru+2]34(<-[O]=C([O-]->3)C([H])([H])[H])(<-[O]=C([O-]->4)C([H])([H])[H])<-[P](c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])(c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])c3c([H])c([H])c4c(c3-c2c2c1OC([H])([H])O2)OC([H])([H])O4"
cat_smi_72="[H]c1c([H])c([H])c([P]2(c3c([H])c([H])c([H])c([H])c3[H])[C-]34->[Fe+2]56789%10%11(<-[C]%12([H])=[C]->5([H])[C-]->6([H])[C]->7([H])=[C]->8%12[H])<-[C]([H])(=[C]->93[H])[C]->%10([H])=[C]->%114[C@@]([H])(C([H])([H])[H])[N](C([H])([H])[H])(C([H])([H])[H])->[Rh+]<-2345<-[C]2([H])=[C]->3([H])C([H])([H])C([H])([H])[C]->4([H])=[C]->5([H])C([H])([H])C2([H])[H])c([H])c1[H]"
cat_smi_73="[H]C1=[C]2([H])->[Fe+2]3456789<-[C]%10([H])=[C]->3([H])[C-]->4([C@]([H])(c3c([H])c([H])c([H])c([H])c3[H])N(C([H])([H])[H])C([H])([H])[H])[C]->5([P](c3c([H])c([H])c([H])c([H])c3[H])(c3c([H])c([H])c([H])c([H])c3[H])->[Rh+]345(<-[C]%11([H])=[C]->3([H])C([H])([H])C([H])([H])[C]->4([H])=[C]->5([H])C([H])([H])C%11([H])[H])<-[P](c3c([H])c([H])c([H])c([H])c3[H])(c3c([H])c([H])c([H])c([H])c3[H])[C]->61=[C]->7([C@]([H])(c1c([H])c([H])c([H])c([H])c1[H])N(C([H])([H])[H])C([H])([H])[H])[C-]->82[H])=[C]->9%10[H]"

cat_sdf_71="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_IterativePredict_1/Final/CAT-71.sdf"
cat_sdf_72="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_IterativePredict_1/Final/CAT-72.sdf"
cat_sdf_73="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_IterativePredict_1/Final/CAT-73.sdf"

sol_smi_1="[H]OC([H])([H])[H]"
sol_smi_2="[H]C([H])([H])C([H])([H])N(C([H])([H])C([H])([H])[H])C([H])([H])C([H])([H])[H].[H]OC([H])([H])[H]"

sol_sdf_1="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/3_opted_mol2/sdf/SOL-1.sdf"
sol_sdf_2="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/3_opted_mol2/sdf/SOL-2.sdf"

temp=60
press_1=20
press_2=100

echo "cat_1 and raw 4 rea and sol: $sol_smi_1"
echo "Press: $press_1"
model_predict $rea_sdf_1 $sol_sdf_1 $cat_sdf_71 $temp $press_1 "./iter_1/cat_71_rea1_${press_1}.npy" $xgb_model_2_rdkit_soap_xtb $rdkit_soap_xtb_3_feat_label
model_predict $rea_sdf_2 $sol_sdf_1 $cat_sdf_71 $temp $press_1 "./iter_1/cat_71_rea2_${press_1}.npy" $xgb_model_2_rdkit_soap_xtb $rdkit_soap_xtb_3_feat_label
model_predict $rea_sdf_3 $sol_sdf_1 $cat_sdf_71 $temp $press_1 "./iter_1/cat_71_rea3_${press_1}.npy" $xgb_model_2_rdkit_soap_xtb $rdkit_soap_xtb_3_feat_label
model_predict $rea_sdf_4 $sol_sdf_1 $cat_sdf_71 $temp $press_1 "./iter_1/cat_71_rea4_${press_1}.npy" $xgb_model_2_rdkit_soap_xtb $rdkit_soap_xtb_3_feat_label
echo "Press: $press_2"
model_predict $rea_sdf_1 $sol_sdf_1 $cat_sdf_71 $temp $press_2 "./iter_1/cat_71_rea1_${press_2}.npy" $xgb_model_2_rdkit_soap_xtb $rdkit_soap_xtb_3_feat_label
model_predict $rea_sdf_2 $sol_sdf_1 $cat_sdf_71 $temp $press_2 "./iter_1/cat_71_rea2_${press_2}.npy" $xgb_model_2_rdkit_soap_xtb $rdkit_soap_xtb_3_feat_label
model_predict $rea_sdf_3 $sol_sdf_1 $cat_sdf_71 $temp $press_2 "./iter_1/cat_71_rea3_${press_2}.npy" $xgb_model_2_rdkit_soap_xtb $rdkit_soap_xtb_3_feat_label
model_predict $rea_sdf_4 $sol_sdf_1 $cat_sdf_71 $temp $press_2 "./iter_1/cat_71_rea4_${press_2}.npy" $xgb_model_2_rdkit_soap_xtb $rdkit_soap_xtb_3_feat_label

echo "cat_2 and raw 4 rea and sol: $sol_smi_2"
echo "Press: $press_1"
model_predict $rea_sdf_1 $sol_sdf_2 $cat_sdf_72 $temp $press_1 "./iter_1/cat_72_rea1_${press_1}.npy" $xgb_model_2_rdkit_soap_xtb $rdkit_soap_xtb_3_feat_label
model_predict $rea_sdf_2 $sol_sdf_2 $cat_sdf_72 $temp $press_1 "./iter_1/cat_72_rea2_${press_1}.npy" $xgb_model_2_rdkit_soap_xtb $rdkit_soap_xtb_3_feat_label
model_predict $rea_sdf_3 $sol_sdf_2 $cat_sdf_72 $temp $press_1 "./iter_1/cat_72_rea3_${press_1}.npy" $xgb_model_2_rdkit_soap_xtb $rdkit_soap_xtb_3_feat_label
model_predict $rea_sdf_4 $sol_sdf_2 $cat_sdf_72 $temp $press_1 "./iter_1/cat_72_rea4_${press_1}.npy" $xgb_model_2_rdkit_soap_xtb $rdkit_soap_xtb_3_feat_label

echo "cat_3 and raw 4 rea and sol: $sol_smi_2"
model_predict $rea_sdf_1 $sol_sdf_2 $cat_sdf_73 $temp $press_1 "./iter_1/cat_73_rea1_${press_1}.npy" $xgb_model_2_rdkit_soap_xtb $rdkit_soap_xtb_3_feat_label
model_predict $rea_sdf_2 $sol_sdf_2 $cat_sdf_73 $temp $press_1 "./iter_1/cat_73_rea2_${press_1}.npy" $xgb_model_2_rdkit_soap_xtb $rdkit_soap_xtb_3_feat_label
model_predict $rea_sdf_3 $sol_sdf_2 $cat_sdf_73 $temp $press_1 "./iter_1/cat_73_rea3_${press_1}.npy" $xgb_model_2_rdkit_soap_xtb $rdkit_soap_xtb_3_feat_label
model_predict $rea_sdf_4 $sol_sdf_2 $cat_sdf_73 $temp $press_1 "./iter_1/cat_73_rea4_${press_1}.npy" $xgb_model_2_rdkit_soap_xtb $rdkit_soap_xtb_3_feat_label

echo "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#"


