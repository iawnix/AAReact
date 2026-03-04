

#echo "#+++++++++++++++++++++++++++++++++++++++++++++++++# Cat #++++++++++++++++++++++++++++++++++++++++++++++#"
#echo 1
#python ../src/AHO_predict.py --task "ee" \
#	--rea_smi "[H]OC(=O)C(C1=C([H])C([H])=C([H])C([H])=C1[H])=C([H])C([H])([H])[H]" \
#	--sol_smi "CO" \
#	--cat_smi "[H]c1c([H])c2[P](c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])(c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])->[Ru+]34(<-[O]=C([O-]->3)C([H])([H])[H])(<-[O]=C([O-]->4)C([H])([H])[H])<-[P](c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])(c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])c3c([H])c([H])c4c(c3-c2c2c1OC([H])([H])O2)OC([H])([H])O4" \
#	--temp 60 \
#	--pressure 20 \
#	--model "/home/iaw/DATA2/AAReact/train/xgb_model_seed_1_split_0-2.pkl" \
#	--feat_label "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_3_x_label.pkl"
#
#python ../src/AHO_predict.py --task "ee" \
#	--rea_smi "[H]OC(=O)C(=C([H])C1=C([H])C([H])=C([H])C([H])=C1[H])C1=C([H])C([H])=C([H])C([H])=C1[H]" \
#	--sol_smi "CO" \
#	--cat_smi "[H]c1c([H])c2[P](c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])(c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])->[Ru+]34(<-[O]=C([O-]->3)C([H])([H])[H])(<-[O]=C([O-]->4)C([H])([H])[H])<-[P](c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])(c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])c3c([H])c([H])c4c(c3-c2c2c1OC([H])([H])O2)OC([H])([H])O4" \
#	--temp 60 \
#	--pressure 20 \
#	--model "/home/iaw/DATA2/AAReact/train/xgb_model_seed_1_split_0-2.pkl" \
#	--feat_label "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_3_x_label.pkl"
#
#python ../src/AHO_predict.py --task "ee" \
#	--rea_smi "[H]OC(=O)C(C1=C([H])C([H])=C([H])C([H])=C1[H])=C(C([H])([H])[H])C([H])([H])[H]" \
#	--sol_smi "CO" \
#	--cat_smi "[H]c1c([H])c2[P](c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])(c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])->[Ru+]34(<-[O]=C([O-]->3)C([H])([H])[H])(<-[O]=C([O-]->4)C([H])([H])[H])<-[P](c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])(c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])c3c([H])c([H])c4c(c3-c2c2c1OC([H])([H])O2)OC([H])([H])O4" \
#	--temp 60 \
#	--pressure 20 \
#	--model "/home/iaw/DATA2/AAReact/train/xgb_model_seed_1_split_0-2.pkl" \
#	--feat_label "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_3_x_label.pkl"
#
#python ../src/AHO_predict.py --task "ee" \
#	--rea_smi "[H]OC(=O)C(C1=C([H])C([H])=C(Cl)C([H])=C1[H])=C(C([H])([H])[H])C([H])([H])[H]" \
#	--sol_smi "CO" \
#	--cat_smi "[H]c1c([H])c2[P](c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])(c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])->[Ru+]34(<-[O]=C([O-]->3)C([H])([H])[H])(<-[O]=C([O-]->4)C([H])([H])[H])<-[P](c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])(c3c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c3[H])c3c([H])c([H])c4c(c3-c2c2c1OC([H])([H])O2)OC([H])([H])O4" \
#	--temp 60 \
#	--pressure 20 \
#	--model "/home/iaw/DATA2/AAReact/train/xgb_model_seed_1_split_0-2.pkl" \
#	--feat_label "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_3_x_label.pkl"
#
#echo 2
#python ../src/AHO_predict.py --task "ee" \
#	--rea_smi "[H]OC(=O)C(C1=C([H])C([H])=C([H])C([H])=C1[H])=C([H])C([H])([H])[H]" \
#	--sol_smi "CO.N(CC)(CC)CC" \
#	--cat_smi "[H]/C1=C(\\[H])C([H])([H])C([H])([H])/C([H])=C(/[H])C([H])([H])C1([H])[H].[H]c1c([H])c([H])c(P(c2c([H])c([H])c([H])c([H])c2[H])[C]23->[Fe+]456789%10(<-[C]%11([H])[C]->4([H])=[C]->5([H])[C]->6([H])=[C]->7%11[H])<-[C]2([H])=[C]->8([H])[C]->9([H])=[C]->%103[C@]([H])(N(C([H])([H])[H])C([H])([H])[H])C([H])([H])[H])c([H])c1[H].[Rh+]" \
#	--temp 60 \
#	--pressure 20 \
#	--model "/home/iaw/DATA2/AAReact/train/xgb_model_seed_1_split_0-2.pkl" \
#	--feat_label "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_3_x_label.pkl"
#
#python ../src/AHO_predict.py --task "ee" \
#	--rea_smi "[H]OC(=O)C(=C([H])C1=C([H])C([H])=C([H])C([H])=C1[H])C1=C([H])C([H])=C([H])C([H])=C1[H]" \
#	--sol_smi "CO.N(CC)(CC)CC" \
#	--cat_smi "[H]/C1=C(\\[H])C([H])([H])C([H])([H])/C([H])=C(/[H])C([H])([H])C1([H])[H].[H]c1c([H])c([H])c(P(c2c([H])c([H])c([H])c([H])c2[H])[C]23->[Fe+]456789%10(<-[C]%11([H])[C]->4([H])=[C]->5([H])[C]->6([H])=[C]->7%11[H])<-[C]2([H])=[C]->8([H])[C]->9([H])=[C]->%103[C@]([H])(N(C([H])([H])[H])C([H])([H])[H])C([H])([H])[H])c([H])c1[H].[Rh+]" \
#	--temp 60 \
#	--pressure 20 \
#	--model "/home/iaw/DATA2/AAReact/train/xgb_model_seed_1_split_0-2.pkl" \
#	--feat_label "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_3_x_label.pkl"
#
#python ../src/AHO_predict.py --task "ee" \
#	--rea_smi "[H]OC(=O)C(C1=C([H])C([H])=C([H])C([H])=C1[H])=C(C([H])([H])[H])C([H])([H])[H]" \
#	--sol_smi "CO.N(CC)(CC)CC" \
#	--cat_smi "[H]/C1=C(\\[H])C([H])([H])C([H])([H])/C([H])=C(/[H])C([H])([H])C1([H])[H].[H]c1c([H])c([H])c(P(c2c([H])c([H])c([H])c([H])c2[H])[C]23->[Fe+]456789%10(<-[C]%11([H])[C]->4([H])=[C]->5([H])[C]->6([H])=[C]->7%11[H])<-[C]2([H])=[C]->8([H])[C]->9([H])=[C]->%103[C@]([H])(N(C([H])([H])[H])C([H])([H])[H])C([H])([H])[H])c([H])c1[H].[Rh+]" \
#	--temp 60 \
#	--pressure 20 \
#	--model "/home/iaw/DATA2/AAReact/train/xgb_model_seed_1_split_0-2.pkl" \
#	--feat_label "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_3_x_label.pkl"
#
#python ../src/AHO_predict.py --task "ee" \
#	--rea_smi "[H]OC(=O)C(C1=C([H])C([H])=C(Cl)C([H])=C1[H])=C(C([H])([H])[H])C([H])([H])[H]" \
#	--sol_smi "CO.N(CC)(CC)CC" \
#	--cat_smi "[H]/C1=C(\\[H])C([H])([H])C([H])([H])/C([H])=C(/[H])C([H])([H])C1([H])[H].[H]c1c([H])c([H])c(P(c2c([H])c([H])c([H])c([H])c2[H])[C]23->[Fe+]456789%10(<-[C]%11([H])[C]->4([H])=[C]->5([H])[C]->6([H])=[C]->7%11[H])<-[C]2([H])=[C]->8([H])[C]->9([H])=[C]->%103[C@]([H])(N(C([H])([H])[H])C([H])([H])[H])C([H])([H])[H])c([H])c1[H].[Rh+]" \
#	--temp 60 \
#	--pressure 20 \
#	--model "/home/iaw/DATA2/AAReact/train/xgb_model_seed_1_split_0-2.pkl" \
#	--feat_label "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_3_x_label.pkl"
#
#echo 3
#python ../src/AHO_predict.py --task "ee" \
#	--rea_smi "[H]OC(=O)C(C1=C([H])C([H])=C([H])C([H])=C1[H])=C([H])C([H])([H])[H]" \
#	--sol_smi "CO.N(CC)(CC)CC" \
#	--cat_smi "[H]/C1=C(\\[H])C([H])([H])C([H])([H])/C([H])=C(/[H])C([H])([H])C1([H])[H].[H]C1=[C]2([H])->[Fe+]345678(<-[C]2([H])[C]->3([C@]([H])(c2c([H])c([H])c([H])c([H])c2[H])N(C([H])([H])[H])C([H])([H])[H])=[C]->41P(c1c([H])c([H])c([H])c([H])c1[H])c1c([H])c([H])c([H])c([H])c1[H])<-[C]1([C@]([H])(c2c([H])c([H])c([H])c([H])c2[H])N(C([H])([H])[H])C([H])([H])[H])[C]->5([H])=[C]->6([H])[C]->7([H])=[C]->81P(c1c([H])c([H])c([H])c([H])c1[H])c1c([H])c([H])c([H])c([H])c1[H].[Rh+]" \
#	--temp 60 \
#	--pressure 20 \
#	--model "/home/iaw/DATA2/AAReact/train/xgb_model_seed_1_split_0-2.pkl" \
#	--feat_label "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_3_x_label.pkl"
#
#python ../src/AHO_predict.py --task "ee" \
#	--rea_smi "[H]OC(=O)C(=C([H])C1=C([H])C([H])=C([H])C([H])=C1[H])C1=C([H])C([H])=C([H])C([H])=C1[H]" \
#	--sol_smi "CO.N(CC)(CC)CC" \
#	--cat_smi "[H]/C1=C(\\[H])C([H])([H])C([H])([H])/C([H])=C(/[H])C([H])([H])C1([H])[H].[H]C1=[C]2([H])->[Fe+]345678(<-[C]2([H])[C]->3([C@]([H])(c2c([H])c([H])c([H])c([H])c2[H])N(C([H])([H])[H])C([H])([H])[H])=[C]->41P(c1c([H])c([H])c([H])c([H])c1[H])c1c([H])c([H])c([H])c([H])c1[H])<-[C]1([C@]([H])(c2c([H])c([H])c([H])c([H])c2[H])N(C([H])([H])[H])C([H])([H])[H])[C]->5([H])=[C]->6([H])[C]->7([H])=[C]->81P(c1c([H])c([H])c([H])c([H])c1[H])c1c([H])c([H])c([H])c([H])c1[H].[Rh+]" \
#	--temp 60 \
#	--pressure 20 \
#	--model "/home/iaw/DATA2/AAReact/train/xgb_model_seed_1_split_0-2.pkl" \
#	--feat_label "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_3_x_label.pkl"
#
#python ../src/AHO_predict.py --task "ee" \
#	--rea_smi "[H]OC(=O)C(C1=C([H])C([H])=C([H])C([H])=C1[H])=C(C([H])([H])[H])C([H])([H])[H]" \
#	--sol_smi "CO.N(CC)(CC)CC" \
#	--cat_smi "[H]/C1=C(\\[H])C([H])([H])C([H])([H])/C([H])=C(/[H])C([H])([H])C1([H])[H].[H]C1=[C]2([H])->[Fe+]345678(<-[C]2([H])[C]->3([C@]([H])(c2c([H])c([H])c([H])c([H])c2[H])N(C([H])([H])[H])C([H])([H])[H])=[C]->41P(c1c([H])c([H])c([H])c([H])c1[H])c1c([H])c([H])c([H])c([H])c1[H])<-[C]1([C@]([H])(c2c([H])c([H])c([H])c([H])c2[H])N(C([H])([H])[H])C([H])([H])[H])[C]->5([H])=[C]->6([H])[C]->7([H])=[C]->81P(c1c([H])c([H])c([H])c([H])c1[H])c1c([H])c([H])c([H])c([H])c1[H].[Rh+]" \
#	--temp 60 \
#	--pressure 20 \
#	--model "/home/iaw/DATA2/AAReact/train/xgb_model_seed_1_split_0-2.pkl" \
#	--feat_label "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_3_x_label.pkl"
#
#python ../src/AHO_predict.py --task "ee" \
#	--rea_smi "[H]OC(=O)C(C1=C([H])C([H])=C(Cl)C([H])=C1[H])=C(C([H])([H])[H])C([H])([H])[H]" \
#	--sol_smi "CO.N(CC)(CC)CC" \
#	--cat_smi "[H]/C1=C(\\[H])C([H])([H])C([H])([H])/C([H])=C(/[H])C([H])([H])C1([H])[H].[H]C1=[C]2([H])->[Fe+]345678(<-[C]2([H])[C]->3([C@]([H])(c2c([H])c([H])c([H])c([H])c2[H])N(C([H])([H])[H])C([H])([H])[H])=[C]->41P(c1c([H])c([H])c([H])c([H])c1[H])c1c([H])c([H])c([H])c([H])c1[H])<-[C]1([C@]([H])(c2c([H])c([H])c([H])c([H])c2[H])N(C([H])([H])[H])C([H])([H])[H])[C]->5([H])=[C]->6([H])[C]->7([H])=[C]->81P(c1c([H])c([H])c([H])c([H])c1[H])c1c([H])c([H])c([H])c([H])c1[H].[Rh+]" \
#	--temp 60 \
#	--pressure 20 \
#	--model "/home/iaw/DATA2/AAReact/train/xgb_model_seed_1_split_0-2.pkl" \
#	--feat_label "/home/iaw/DATA2/AAReact/DataSet/AtropicAcid_416/4_gen_feature/rdkit_3_x_label.pkl"
#echo "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#"


function pred() {
	rea_smi=$1
	sol_smi="CO.N(CC)(CC)CC"
	cat_smi=$2
	temp=60
	press=20
	python ../src/AHO_predict.py --task "ee" \
		--rea_smi $rea_smi \
		--sol_smi $sol_smi \
		--cat_smi $cat_smi \
		--temp $temp \
		--pressure $press \
		--model "/home/iaw/DATA2/AAReact/train/xgb_model_seed_1_split_0-2.pkl" \
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
cat_smi_41="[Fe+]123456789<-[C:3]%10([P:1]%11(->[Rh+:61]%12%13%14(<-[P:2]%15([C:12]->1%16=[C:17]->2([H:18])[C:15]->3([H:16])=[C:14]->4([H:53])[C:13]->5%16[H:52])[C@:29]([H:30])([C@@:62]([H:63])([H:64])[H:65])[C@:54]([H:57])([H:58])[C@:56]([H:59])([H:60])[C@@:51]%15([H:55])[C@@:66]([H:67])([H:68])[H:69])<-[C:31]1([H:32])=[C:33]->%12([H:34])[C@:35]([H:36])([H:37])[C@@:38]([H:39])([H:40])[C:41]->%13([H:42])=[C:43]->%14([H:44])[C@:45]([H:46])([H:47])[C@@:48]1([H:49])[H:50])[C@:19]([H:20])([C@@:70]([H:71])([H:72])[H:73])[C@@:21]([H:22])([H:23])[C@:24]([H:25])([H:26])[C@@:27]%11([H:28])[C@@:74]([H:75])([H:76])[H:77])=[C:4]->6([H:5])[C:6]->7([H:7])=[C:8]->8([H:9])[C:10]->9%10[H:11]"
cat_smi_42="[Fe+]123456789<-[C:3]%10([P:1]%11(->[Rh+:61]%12%13%14(<-[P:2]%15([C:12]->1%16[C:13]->2([H:52])=[C:14]->3([H:53])[C:15]->4([H:16])=[C:17]->5%16[H:18])[C@@:29]([H:30])([C@:66]([H:67])([H:68])[C@@:69]([H:70])([H:71])[H:72])[C@:54]([H:57])([H:58])[C@:56]([H:59])([H:60])[C@:51]%15([H:55])[C@:73]([C@@:62]([H:63])([H:64])[H:65])([H:74])[H:75])<-[C:31]1([H:32])=[C:33]->%12([H:34])[C@:35]([H:36])([H:37])[C@@:38]([H:39])([H:40])[C:41]->%13([H:42])=[C:43]->%14([H:44])[C@:45]([H:46])([H:47])[C@@:48]1([H:49])[H:50])[C@@:19]([H:20])([C@:83]([H:84])([H:85])[C@@:86]([H:87])([H:88])[H:89])[C@@:21]([H:22])([H:23])[C@:24]([H:25])([H:26])[C@:27]%11([H:28])[C@:76]([H:77])([H:78])[C@@:79]([H:80])([H:81])[H:82])=[C:10]->6([H:11])[C:8]->7([H:9])[C:6]->8([H:7])=[C:4]->9%10[H:5]"

pred $rea_smi_1 $cat_smi_41
pred $rea_smi_2 $cat_smi_41
pred $rea_smi_3 $cat_smi_41
pred $rea_smi_4 $cat_smi_41
pred $rea_smi_5 $cat_smi_41
pred $rea_smi_6 $cat_smi_41

pred $rea_smi_1 $cat_smi_42
pred $rea_smi_2 $cat_smi_42
pred $rea_smi_3 $cat_smi_42
pred $rea_smi_4 $cat_smi_42
pred $rea_smi_5 $cat_smi_42
pred $rea_smi_6 $cat_smi_42

echo "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#"

