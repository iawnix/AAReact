
function xgb_pred() {
	rea_smi=$1
	sol_smi=$2
	cat_smi=$3
	temp=$4
	press=$5
    model=$6
	python ../src/AHO_predict.py --task "ee" \
		--rea_smi $rea_smi \
		--sol_smi $sol_smi \
		--cat_smi $cat_smi \
		--temp $temp \
		--pressure $press \
		--model  \
		--feat_label "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_3_x_label.pkl" \
		--verbose 0
}

function rf_pred() {
	rea_smi=$1
	sol_smi=$2
	cat_smi=$3
	temp=$4
	press=$5
	python ../src/AHO_predict.py --task "ee" \
		--rea_smi $rea_smi \
		--sol_smi $sol_smi \
		--cat_smi $cat_smi \
		--temp $temp \
		--pressure $press \
		--model "/home/iaw/DATA2/AAReact/train/rf_model_seed_1_split_0-2_hyper2.pkl" \
		--feat_label "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_3_x_label.pkl" \
		--verbose 0
}


echo "#+++++++++++++++++++++++++++++++++++++++++++++++++# Rea #++++++++++++++++++++++++++++++++++++++++++++++#"



rea_smi_1="C/C(C)=C(C(O)=O)\C1=CC=C(F)C=C1"
rea_smi_2="CC1=CC=CC(/C(C(O)=O)=C(C)\C)=C1"
rea_smi_3="C/C(C)=C(C(O)=O)\C1=CC=C(C)C=C1"
rea_smi_4="C/C(C)=C(C(O)=O)\C1=CC=C(C(C)C)C=C1"
rea_smi_5="C/C(C)=C(C(O)=O)\C1=CC=C(CCCCC)C=C1"
rea_smi_6="O=C(/C(C1=CC=CC=C1)=C2CCCC/2)O"

sol_smi_1="CO.N(CC)(CC)CC"

cat_smi_41="[Fe+]123456789<-[C:3]%10([P:1]%11(->[Rh+:61]%12%13%14(<-[P:2]%15([C:12]->1%16=[C:17]->2([H:18])[C:15]->3([H:16])=[C:14]->4([H:53])[C:13]->5%16[H:52])[C@:29]([H:30])([C@@:62]([H:63])([H:64])[H:65])[C@:54]([H:57])([H:58])[C@:56]([H:59])([H:60])[C@@:51]%15([H:55])[C@@:66]([H:67])([H:68])[H:69])<-[C:31]1([H:32])=[C:33]->%12([H:34])[C@:35]([H:36])([H:37])[C@@:38]([H:39])([H:40])[C:41]->%13([H:42])=[C:43]->%14([H:44])[C@:45]([H:46])([H:47])[C@@:48]1([H:49])[H:50])[C@:19]([H:20])([C@@:70]([H:71])([H:72])[H:73])[C@@:21]([H:22])([H:23])[C@:24]([H:25])([H:26])[C@@:27]%11([H:28])[C@@:74]([H:75])([H:76])[H:77])=[C:4]->6([H:5])[C:6]->7([H:7])=[C:8]->8([H:9])[C:10]->9%10[H:11]"
cat_smi_42="[Fe+]123456789<-[C:3]%10([P:1]%11(->[Rh+:61]%12%13%14(<-[P:2]%15([C:12]->1%16[C:13]->2([H:52])=[C:14]->3([H:53])[C:15]->4([H:16])=[C:17]->5%16[H:18])[C@@:29]([H:30])([C@:66]([H:67])([H:68])[C@@:69]([H:70])([H:71])[H:72])[C@:54]([H:57])([H:58])[C@:56]([H:59])([H:60])[C@:51]%15([H:55])[C@:73]([C@@:62]([H:63])([H:64])[H:65])([H:74])[H:75])<-[C:31]1([H:32])=[C:33]->%12([H:34])[C@:35]([H:36])([H:37])[C@@:38]([H:39])([H:40])[C:41]->%13([H:42])=[C:43]->%14([H:44])[C@:45]([H:46])([H:47])[C@@:48]1([H:49])[H:50])[C@@:19]([H:20])([C@:83]([H:84])([H:85])[C@@:86]([H:87])([H:88])[H:89])[C@@:21]([H:22])([H:23])[C@:24]([H:25])([H:26])[C@:27]%11([H:28])[C@:76]([H:77])([H:78])[C@@:79]([H:80])([H:81])[H:82])=[C:10]->6([H:11])[C:8]->7([H:9])[C:6]->8([H:7])=[C:4]->9%10[H:5]"

temp=60
press=20

xgb_pred $rea_smi_1 $sol_smi_1 $cat_smi_41 $temp $press
xgb_pred $rea_smi_2 $sol_smi_1 $cat_smi_41 $temp $press
xgb_pred $rea_smi_3 $sol_smi_1 $cat_smi_41 $temp $press
xgb_pred $rea_smi_4 $sol_smi_1 $cat_smi_41 $temp $press
xgb_pred $rea_smi_5 $sol_smi_1 $cat_smi_41 $temp $press
xgb_pred $rea_smi_6 $sol_smi_1 $cat_smi_41 $temp $press

xgb_pred $rea_smi_1 $sol_smi_1 $cat_smi_42 $temp $press
xgb_pred $rea_smi_2 $sol_smi_1 $cat_smi_42 $temp $press
xgb_pred $rea_smi_3 $sol_smi_1 $cat_smi_42 $temp $press
xgb_pred $rea_smi_4 $sol_smi_1 $cat_smi_42 $temp $press
xgb_pred $rea_smi_5 $sol_smi_1 $cat_smi_42 $temp $press
xgb_pred $rea_smi_6 $sol_smi_1 $cat_smi_42 $temp $press

echo "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#"

