#!/usr/bin/bash

JOB="CAT-63"
CHARGE=0

#++++++++++++++++++++++++++++++++++# 中间变量 #+++++++++++++++++++++++++++++++++++#
cat <<- EOF > xtb_md.inp
	\$md
	 temp=500
	 time=1.0
	 dump=10.0
	 step=1.0
	 velo=false
	 nvt=true
	 hmass=1
	 shake=1
	\$end
	\$write
	 density=false
	 fod=false
	 charges=false
	 mulliken=false
	 geosum=false
	 inertia=false
	 mos=false
	 wiberg=false
	\$end
EOF
PYTHON=/home/iaw/soft/conda/2024.06.1/envs/python3.12/bin/python
XTBMOL2=/home/iaw/DATA2/AAReact/src/tool/xtb_mol2/xtb_mol2.py


#+++++++++++++++++++++++++++++++++# 运行程序  #++++++++++++++++++++++++++++++++++#
obabel -imol2 ${JOB}.mol2 -oxyz -O ${JOB}.xyz > obabel.log 2>&1
xtb ${JOB}.xyz \
	--gfn 1 --chrg ${CHARGE} --uhf 0 \
	--norestart \
	--omd \
	--input xtb_md.inp \
	> md.log 2>&1
mv xtb.trj xtbtrj.xyz

$PYTHON $XTBMOL2 -imol2 ${JOB}.mol2 \
			   -itraj xtbtrj.xyz \
			   -omol2 ${JOB}_new.mol2


