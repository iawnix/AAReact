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


#++++++++++++++++++++++++++++++++++++++++++++++++# Model #+++++++++++++++++++++++++++++++++++++++++++++#

declare -A model_s=(
	["rf_rdkit"]="/home/iaw/DATA2/AAReact/train/output/pt/rf_model_seed_1_split_0-2_hyper2_rdkit.pkl"
	["rf_rdkit_soap"]="/home/iaw/DATA2/AAReact/train/output/pt/rf_model_seed_1_split_0-2_hyper2_rdkit_soap.pkl"
	["rf_rdkit_soap_xtb"]="/home/iaw/DATA2/AAReact/train/output/pt/rf_model_seed_1_split_0-2_hyper2_rdkit_soap_xtb.pkl"
	["rf_soap_xtb"]="/home/iaw/DATA2/AAReact/train/pt/rf_model_seed_1_split_0-2_hyper2_rdkit.pkl"
	["xgb_rdkit"]="/home/iaw/DATA2/AAReact/train/output/pt/xgb_model_seed_1_split_0-2_hyper2_rdkit.pkl"
	["xgb_rdkit_soap"]="/home/iaw/DATA2/AAReact/train/output/pt/xgb_model_seed_1_split_0-2_hyper2_rdkit_soap.pkl"
	["xgb_rdkit_soap_xtb"]="/home/iaw/DATA2/AAReact/train/output/pt/xgb_model_seed_1_split_0-2_hyper2_rdkit_soap_xtb.pkl"
	["xgb_soap_xtb"]="/home/iaw/DATA2/AAReact/train/pt/xgb_model_seed_1_split_0-2_hyper2_rdkit.pkl"

)

declare -A feat_label_s=(
	["rdkit"]="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_3_x_label.pkl"
	["rdkit_soap"]="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_soap_3_x_label.pkl"
	["rdkit_soap_xtb"]="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_soap_xtb_3_x_label.pkl"
	["soap_xtb"]="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/soap_xtb_3_x_label.pkl"
)

declare -A model_feat_map=(
	["rf_rdkit"]="rdkit"
	["rf_rdkit_soap"]="rdkit_soap"
	["rf_rdkit_soap_xtb"]="rdkit_soap_xtb"
	["rf_soap_xtb"]="soap_xtb"
	["xgb_rdkit"]="rdkit"
	["xgb_rdkit_soap"]="rdkit_soap"
	["xgb_rdkit_soap_xtb"]="rdkit_soap_xtb"
	["xgb_soap_xtb"]="soap_xtb"

)

rea_smi_1="[H]OC(=O)C(C1=C([H])C([H])=C([H])C([H])=C1[H])=C([H])C([H])([H])[H]"
rea_smi_2="[H]OC(=O)C(=C([H])C1=C([H])C([H])=C([H])C([H])=C1[H])C1=C([H])C([H])=C([H])C([H])=C1[H]"
rea_smi_3="[H]OC(=O)C(C1=C([H])C([H])=C([H])C([H])=C1[H])=C(C([H])([H])[H])C([H])([H])[H]"
rea_smi_4="[H]OC(=O)C(C1=C([H])C([H])=C(Cl)C([H])=C1[H])=C(C([H])([H])[H])C([H])([H])[H]"
cat_smi_71="[H]c1c([H])c2[P](c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])(c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])->[Ru+2]34(<-[O]=C([O-]->3)C([H])([H])[H])(<-[O]=C([O-]->4)C([H])([H])[H])<-[P](c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])(c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])c3c([H])c([H])c4c(c3-c2c2c1OC([H])([H])O2)OC([H])([H])O4"
cat_smi_72="[H]c1c([H])c([H])c([P]2(c3c([H])c([H])c([H])c([H])c3[H])[C-]34->[Fe+2]56789%10%11(<-[C]%12([H])=[C]->5([H])[C-]->6([H])[C]->7([H])=[C]->8%12[H])<-[C]([H])(=[C]->93[H])[C]->%10([H])=[C]->%114[C@@]([H])(C([H])([H])[H])[N](C([H])([H])[H])(C([H])([H])[H])->[Rh+]<-2345<-[C]2([H])=[C]->3([H])C([H])([H])C([H])([H])[C]->4([H])=[C]->5([H])C([H])([H])C2([H])[H])c([H])c1[H]"
cat_smi_73="[H]C1=[C]2([H])->[Fe+2]3456789<-[C]%10([H])=[C]->3([H])[C-]->4([C@]([H])(c3c([H])c([H])c([H])c([H])c3[H])N(C([H])([H])[H])C([H])([H])[H])[C]->5([P](c3c([H])c([H])c([H])c([H])c3[H])(c3c([H])c([H])c([H])c([H])c3[H])->[Rh+]345(<-[C]%11([H])=[C]->3([H])C([H])([H])C([H])([H])[C]->4([H])=[C]->5([H])C([H])([H])C%11([H])[H])<-[P](c3c([H])c([H])c([H])c([H])c3[H])(c3c([H])c([H])c([H])c([H])c3[H])[C]->61=[C]->7([C@]([H])(c1c([H])c([H])c([H])c([H])c1[H])N(C([H])([H])[H])C([H])([H])[H])[C-]->82[H])=[C]->9%10[H]"
sol_smi_1="[H]OC([H])([H])[H]"
sol_smi_2="[H]C([H])([H])C([H])([H])N(C([H])([H])C([H])([H])[H])C([H])([H])C([H])([H])[H].[H]OC([H])([H])[H]"


#+++++++++++++++++++++++++++++++++++++++++++++++++# Rea #++++++++++++++++++++++++++++++++++++++++++++++#

rea_list=(
    "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/3_opted_mol2/sdf/REA-1.sdf"
    "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/3_opted_mol2/sdf/REA-2.sdf"
    "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/3_opted_mol2/sdf/REA-3.sdf"
    "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/3_opted_mol2/sdf/REA-4.sdf"
)

#++++++++++++++++++++++++++++++++++++++++++++++++# ComB #++++++++++++++++++++++++++++++++++++++++++++++#
combinations=(
    "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_IterativePredict_1/Final/CAT-71.sdf   /home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/3_opted_mol2/sdf/SOL-1.sdf   20"
    "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_IterativePredict_1/Final/CAT-71.sdf   /home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/3_opted_mol2/sdf/SOL-1.sdf   100"
    "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_IterativePredict_1/Final/CAT-72.sdf   /home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/3_opted_mol2/sdf/SOL-2.sdf   20"
    "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_IterativePredict_1/Final/CAT-73.sdf   /home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/3_opted_mol2/sdf/SOL-2.sdf   20"
)


#++++++++++++++++++++++++++++++++++++++++++++++++# Main #+++++++++++++++++++++++++++++++++++++++++++++#
temp=60
for model_name in "rf_rdkit" "rf_rdkit_soap" "rf_rdkit_soap_xtb" "rf_soap_xtb" "xgb_rdkit" "xgb_rdkit_soap" "xgb_rdkit_soap_xtb" "xgb_soap_xtb"
do
	model_pt=${model_s[$model_name]}
	feat_label_name=${model_feat_map[$model_name]}
	feat_label=${feat_label_s[$feat_label_name]}
	for combo in "${combinations[@]}"; do
		read cat_sdf sol_sdf press_num <<< "$combo"
		cat_name=$(basename "$cat_sdf" .sdf)
		sol_name=$(basename "$sol_sdf" .sdf)
		press="$press_num"
		for rea_sdf in "${rea_list[@]}"; do
			rea_name=$(basename "$rea_sdf" .sdf)
			save_path="./iter_1/${model_name}-${rea_name}_${sol_name}_${cat_name}_T${temp}_P${press}.npy"
			echo "${model_name}-${rea_name}_${sol_name}_${cat_name}_T${temp}_P${press}"
			model_predict $rea_sdf $sol_sdf $cat_sdf $temp $press $save_path $model_pt $feat_label
		done
	done
done
       
echo "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#"


