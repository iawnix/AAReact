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
2. 需要考虑底物上取代基对反应的影响, 以及反应条件[P,T]对

```


 Reactant_smiles Product_SMILES     
       ||              ||
       \/              \/
             Unimol
               ||
               \/
 Reactant_embed  Product_embed
       ||              ||
       \/              \/
       MLP             MLP

```
