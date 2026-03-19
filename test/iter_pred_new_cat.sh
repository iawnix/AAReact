
function xgb_pred() {
	rea_smi=$1
	sol_smi=$2
	cat_smi=$3
	temp=$4
	press=$5
	save_feat=$6
	python ../src/AHO_predict.py --task "ee" \
		--rea_smi $rea_smi \
		--sol_smi $sol_smi \
		--cat_smi $cat_smi \
		--temp $temp \
		--pressure $press \
		--model "/home/iaw/DATA2/AAReact/train/xgb_model_seed_1_split_0-2_hyper2.pkl" \
		--feat_label "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_3_x_label.pkl" \
		--verbose 0 \
		--save_feat $save_feat

}

function rf_pred() {
	rea_smi=$1
	sol_smi=$2
	cat_smi=$3
	temp=$4
	press=$5
	save_feat=$6
	python ../src/AHO_predict.py --task "ee" \
		--rea_smi $rea_smi \
		--sol_smi $sol_smi \
		--cat_smi $cat_smi \
		--temp $temp \
		--pressure $press \
		--model "/home/iaw/DATA2/AAReact/train/rf_model_seed_1_split_0-2_hyper2.pkl" \
		--feat_label "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_3_x_label.pkl" \
		--verbose 0 \
		--save_feat $save_feat
}


echo "#+++++++++++++++++++++++++++++++++++++++++++++++++# Rea #++++++++++++++++++++++++++++++++++++++++++++++#"



rea_smi_1="[H]OC(=O)C(C1=C([H])C([H])=C([H])C([H])=C1[H])=C([H])C([H])([H])[H]"
rea_smi_2="[H]OC(=O)C(=C([H])C1=C([H])C([H])=C([H])C([H])=C1[H])C1=C([H])C([H])=C([H])C([H])=C1[H]"
rea_smi_3="[H]OC(=O)C(C1=C([H])C([H])=C([H])C([H])=C1[H])=C(C([H])([H])[H])C([H])([H])[H]"
rea_smi_4="[H]OC(=O)C(C1=C([H])C([H])=C(Cl)C([H])=C1[H])=C(C([H])([H])[H])C([H])([H])[H]"

cat_smi_71="[H]c1c([H])c2[P](c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])(c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])->[Ru+]34(<-[O]=C([O-]->3)C([H])([H])[H])(<-[O]=C([O-]->4)C([H])([H])[H])<-[P](c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])(c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])c3c([H])c([H])c4c(c3-c2c2c1OC([H])([H])O2)OC([H])([H])O4"
cat_smi_72="[H]/C1=C(\\[H])C([H])([H])C([H])([H])/C([H])=C(/[H])C([H])([H])C1([H])[H].[H]c1c([H])c([H])c(P(c2c([H])c([H])c([H])c([H])c2[H])[C]23->[Fe+]456789%10(<-[C]%11([H])[C]->4([H])=[C]->5([H])[C]->6([H])=[C]->7%11[H])<-[C]2([H])=[C]->8([H])[C]->9([H])=[C]->%103[C@]([H])(N(C([H])([H])[H])C([H])([H])[H])C([H])([H])[H])c([H])c1[H].[Rh+]"
cat_smi_73="[H]/C1=C(\\[H])C([H])([H])C([H])([H])/C([H])=C(/[H])C([H])([H])C1([H])[H].[H]C1=[C]2([H])->[Fe+]345678(<-[C]2([H])[C]->3([C@]([H])(c2c([H])c([H])c([H])c([H])c2[H])N(C([H])([H])[H])C([H])([H])[H])=[C]->41P(c1c([H])c([H])c([H])c([H])c1[H])c1c([H])c([H])c([H])c([H])c1[H])<-[C]1([C@]([H])(c2c([H])c([H])c([H])c([H])c2[H])N(C([H])([H])[H])C([H])([H])[H])[C]->5([H])=[C]->6([H])[C]->7([H])=[C]->81P(c1c([H])c([H])c([H])c([H])c1[H])c1c([H])c([H])c([H])c([H])c1[H].[Rh+]"

sol_smi_1="OC"
sol_smi_2="OC.CCN(CC)CC"

temp=60
press_1=20
press_2=100

echo "cat_1 and raw 4 rea and sol: $sol_smi_1"
echo "Press: $press_1"
xgb_pred $rea_smi_1 $sol_smi_1 $cat_smi_71 $temp $press_1 "./cat_71_rea1_${press_1}.npy"
xgb_pred $rea_smi_2 $sol_smi_1 $cat_smi_71 $temp $press_1 "./cat_71_rea2_${press_1}.npy"
xgb_pred $rea_smi_3 $sol_smi_1 $cat_smi_71 $temp $press_1 "./cat_71_rea3_${press_1}.npy"
xgb_pred $rea_smi_4 $sol_smi_1 $cat_smi_71 $temp $press_1 "./cat_71_rea4_${press_1}.npy"
echo "Press: $press_2"
xgb_pred $rea_smi_1 $sol_smi_1 $cat_smi_71 $temp $press_2 "./cat_71_rea1_${press_2}.npy"
xgb_pred $rea_smi_2 $sol_smi_1 $cat_smi_71 $temp $press_2 "./cat_71_rea2_${press_2}.npy"
xgb_pred $rea_smi_3 $sol_smi_1 $cat_smi_71 $temp $press_2 "./cat_71_rea3_${press_2}.npy"
xgb_pred $rea_smi_4 $sol_smi_1 $cat_smi_71 $temp $press_2 "./cat_71_rea4_${press_2}.npy"

echo "cat_2 and raw 4 rea and sol: $sol_smi_2"
echo "Press: $press_1"
xgb_pred $rea_smi_1 $sol_smi_2 $cat_smi_72 $temp $press_1 "./cat_72_rea1_${press_1}.npy"
xgb_pred $rea_smi_2 $sol_smi_2 $cat_smi_72 $temp $press_1 "./cat_72_rea2_${press_1}.npy"
xgb_pred $rea_smi_3 $sol_smi_2 $cat_smi_72 $temp $press_1 "./cat_72_rea3_${press_1}.npy"
xgb_pred $rea_smi_4 $sol_smi_2 $cat_smi_72 $temp $press_1 "./cat_72_rea4_${press_1}.npy"

echo "cat_3 and raw 4 rea and sol: $sol_smi_2"
xgb_pred $rea_smi_1 $sol_smi_2 $cat_smi_73 $temp $press_1 "./cat_73_rea1_${press_1}.npy"
xgb_pred $rea_smi_2 $sol_smi_2 $cat_smi_73 $temp $press_1 "./cat_73_rea2_${press_1}.npy"
xgb_pred $rea_smi_3 $sol_smi_2 $cat_smi_73 $temp $press_1 "./cat_73_rea3_${press_1}.npy"
xgb_pred $rea_smi_4 $sol_smi_2 $cat_smi_73 $temp $press_1 "./cat_73_rea4_${press_1}.npy"

echo "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#"


