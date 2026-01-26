# IAWNIX
- 20260112
- iawhaha@163.com
- atropic acid 氢化还原反应的手性选择性预测

# TODO: 
## 数据确认
1. Catalyst_SMILES与Metal以及Ligand_SMILES之间的关系
2. Additive_SMILES代表的含义
3. 4种底物是一起反应, 相互之间是否有影响
   1. 以 r1,r2,r3,r4 ---[CataX,P,T,...]--->p1,p2,p3,p4为例子, 是否可以将r1---[CataX,P,T,...]--->p1视为独立数据?

## 符号补充
1. ee [-1, 1]
2. 
## 模型思路
1. 当前任务是构建一个Atropic acid[阿托品酸]加氢还原为Hydratropic acid[氢化阿托酸]的ee预测器, 以实现给定反应条件, 给出预测的ee
2. 需要考虑催化剂溶剂, 以及反应条件[P,T]对ee的影响
3. 输出头加入tanh, 使取值在-1, 1之间

```
## 反思
1. 小样本的时候, Unimol编码出来的分子潜入特征维度太大
2. 交叉注意力, 将B的特征融入到A上面, 除了缺少seq_len, 数据量太少，更难以捕捉A与B之间的关系，不如直接concat

3. gv修改键为正常的单双键, 同时保存为mol2, 可以让rdkit正常读取, 然后手动增加配位键, smiles可以保存
   * smiles可以保存
   * mogan指纹可以计算
   * unimol2的编码失败, 但是unimol2可以使用
 
4. 二茂铁的smiles

