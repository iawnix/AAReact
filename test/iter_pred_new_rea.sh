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

rdkit_3_feat_label="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_3_x_label.pkl"
rdkit_soap_3_feat_label="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_soap_3_x_label.pkl"


rea_smi_5="C/C(C)=C(C(O)=O)\C1=CC=C(F)C=C1"
rea_smi_6="CC1=CC=CC(/C(C(O)=O)=C(C)\C)=C1"
rea_smi_7="C/C(C)=C(C(O)=O)\C1=CC=C(C)C=C1"
rea_smi_8="C/C(C)=C(C(O)=O)\C1=CC=C(C(C)C)C=C1"
rea_smi_9="C/C(C)=C(C(O)=O)\C1=CC=C(CCCCC)C=C1"
rea_smi_10="O=C(/C(C1=CC=CC=C1)=C2CCCC/2)O"

rea_sdf_5="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_IterativePredict_1/Final/REA-5.sdf"
rea_sdf_6="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_IterativePredict_1/Final/REA-6.sdf"
rea_sdf_7="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_IterativePredict_1/Final/REA-7.sdf"
rea_sdf_8="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_IterativePredict_1/Final/REA-8.sdf"
rea_sdf_9="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_IterativePredict_1/Final/REA-9.sdf"
rea_sdf_10="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_IterativePredict_1/Final/REA-10.sdf"

sol_smi_2="[H]C([H])([H])C([H])([H])N(C([H])([H])C([H])([H])[H])C([H])([H])C([H])([H])[H].[H]OC([H])([H])[H]"
sol_sdf_2="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/3_opted_mol2/sdf/SOL-2.sdf"

cat_smi_41="[Fe+]123456789<-[C:3]%10([P:1]%11(->[Rh+:61]%12%13%14(<-[P:2]%15([C:12]->1%16=[C:17]->2([H:18])[C:15]->3([H:16])=[C:14]->4([H:53])[C:13]->5%16[H:52])[C@:29]([H:30])([C@@:62]([H:63])([H:64])[H:65])[C@:54]([H:57])([H:58])[C@:56]([H:59])([H:60])[C@@:51]%15([H:55])[C@@:66]([H:67])([H:68])[H:69])<-[C:31]1([H:32])=[C:33]->%12([H:34])[C@:35]([H:36])([H:37])[C@@:38]([H:39])([H:40])[C:41]->%13([H:42])=[C:43]->%14([H:44])[C@:45]([H:46])([H:47])[C@@:48]1([H:49])[H:50])[C@:19]([H:20])([C@@:70]([H:71])([H:72])[H:73])[C@@:21]([H:22])([H:23])[C@:24]([H:25])([H:26])[C@@:27]%11([H:28])[C@@:74]([H:75])([H:76])[H:77])=[C:4]->6([H:5])[C:6]->7([H:7])=[C:8]->8([H:9])[C:10]->9%10[H:11]"
cat_smi_42="[Fe+]123456789<-[C:3]%10([P:1]%11(->[Rh+:61]%12%13%14(<-[P:2]%15([C:12]->1%16[C:13]->2([H:52])=[C:14]->3([H:53])[C:15]->4([H:16])=[C:17]->5%16[H:18])[C@@:29]([H:30])([C@:66]([H:67])([H:68])[C@@:69]([H:70])([H:71])[H:72])[C@:54]([H:57])([H:58])[C@:56]([H:59])([H:60])[C@:51]%15([H:55])[C@:73]([C@@:62]([H:63])([H:64])[H:65])([H:74])[H:75])<-[C:31]1([H:32])=[C:33]->%12([H:34])[C@:35]([H:36])([H:37])[C@@:38]([H:39])([H:40])[C:41]->%13([H:42])=[C:43]->%14([H:44])[C@:45]([H:46])([H:47])[C@@:48]1([H:49])[H:50])[C@@:19]([H:20])([C@:83]([H:84])([H:85])[C@@:86]([H:87])([H:88])[H:89])[C@@:21]([H:22])([H:23])[C@:24]([H:25])([H:26])[C@:27]%11([H:28])[C@:76]([H:77])([H:78])[C@@:79]([H:80])([H:81])[H:82])=[C:10]->6([H:11])[C:8]->7([H:9])[C:6]->8([H:7])=[C:4]->9%10[H:5]"

cat_sdf_41="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/3_opted_mol2/sdf/CAT-41.sdf"
cat_sdf_42="/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/3_opted_mol2/sdf/CAT-42.sdf"
temp=60
press=20

echo "cat_41"
model_predict $rea_sdf_5  $sol_sdf_2 $cat_sdf_41 $temp $press "./iter_1/cat_71_rea5_${press_1}.npy"  $xgb_model_2_rdkit_soap $rdkit_soap_3_feat_label
model_predict $rea_sdf_6  $sol_sdf_2 $cat_sdf_41 $temp $press "./iter_1/cat_71_rea6_${press_1}.npy"  $xgb_model_2_rdkit_soap $rdkit_soap_3_feat_label
model_predict $rea_sdf_7  $sol_sdf_2 $cat_sdf_41 $temp $press "./iter_1/cat_71_rea7_${press_1}.npy"  $xgb_model_2_rdkit_soap $rdkit_soap_3_feat_label
model_predict $rea_sdf_8  $sol_sdf_2 $cat_sdf_41 $temp $press "./iter_1/cat_71_rea8_${press_1}.npy"  $xgb_model_2_rdkit_soap $rdkit_soap_3_feat_label
model_predict $rea_sdf_9  $sol_sdf_2 $cat_sdf_41 $temp $press "./iter_1/cat_71_rea9_${press_1}.npy"  $xgb_model_2_rdkit_soap $rdkit_soap_3_feat_label
model_predict $rea_sdf_10 $sol_sdf_2 $cat_sdf_41 $temp $press "./iter_1/cat_71_rea10_${press_1}.npy" $xgb_model_2_rdkit_soap $rdkit_soap_3_feat_label

echo "cat_41"
model_predict $rea_sdf_5  $sol_sdf_2 $cat_sdf_42 $temp $press "./iter_1/cat_71_rea5_${press_1}.npy"  $xgb_model_2_rdkit_soap $rdkit_soap_3_feat_label
model_predict $rea_sdf_6  $sol_sdf_2 $cat_sdf_42 $temp $press "./iter_1/cat_71_rea6_${press_1}.npy"  $xgb_model_2_rdkit_soap $rdkit_soap_3_feat_label
model_predict $rea_sdf_7  $sol_sdf_2 $cat_sdf_42 $temp $press "./iter_1/cat_71_rea7_${press_1}.npy"  $xgb_model_2_rdkit_soap $rdkit_soap_3_feat_label
model_predict $rea_sdf_8  $sol_sdf_2 $cat_sdf_42 $temp $press "./iter_1/cat_71_rea8_${press_1}.npy"  $xgb_model_2_rdkit_soap $rdkit_soap_3_feat_label
model_predict $rea_sdf_9  $sol_sdf_2 $cat_sdf_42 $temp $press "./iter_1/cat_71_rea9_${press_1}.npy"  $xgb_model_2_rdkit_soap $rdkit_soap_3_feat_label
model_predict $rea_sdf_10 $sol_sdf_2 $cat_sdf_42 $temp $press "./iter_1/cat_71_rea10_${press_1}.npy" $xgb_model_2_rdkit_soap $rdkit_soap_3_feat_label

echo "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#"

