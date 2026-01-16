import rdkit
from rdkit import Chem
from rdkit.Chem import rdmolops




#+++++++++++++++++++++++++++++++++++++++# READ ME #++++++++++++++++++++++++++++++++++++++++++++++++#
# split_fe_complex_into_ligand_and_metal, 豆包构建用于拆分金属Fe与配体
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#



def split_fe_complex_into_ligand_and_metal(smi):
    """
    拆分含 Fe²⁺ 的配合物，得到有机配体和 Fe²⁺ 金属离子（修正 RemoveBond 参数错误）
    :param smi: 配合物完整 SMILES
    :return: 金属离子分子片段、有机配体分子片段（均为 RDKit Mol 对象）
    """
    # 步骤1：预处理分子（解决复杂 SMILES 解析失败问题，保留电荷和结构）
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        raise ValueError("SMILES 解析失败，无法加载分子")
    
    # 步骤2：手动选择性执行结构校验（修复基础结构，保留带电片段和配位连接）
    sanitize_ops = (
        rdmolops.SanitizeFlags.SANITIZE_FINDRADICALS
        | rdmolops.SanitizeFlags.SANITIZE_KEKULIZE
        | rdmolops.SanitizeFlags.SANITIZE_SETAROMATICITY
        | rdmolops.SanitizeFlags.SANITIZE_SETCONJUGATION
        | rdmolops.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
    )
    rdmolops.SanitizeMol(mol, sanitizeOps=sanitize_ops)
    
    # 步骤3：定位 Fe 金属中心的原子索引（找到所有 Fe 原子）
    fe_atom_indices = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "Fe":
            fe_atom_indices.append(atom.GetIdx())
            print(f"找到 Fe 金属中心，原子索引：{atom.GetIdx()}，电荷：{atom.GetFormalCharge()}")
    
    if not fe_atom_indices:
        raise ValueError("分子中未找到 Fe 金属原子")
    
    # 步骤4：收集所有需要切断的「原子对」（Fe 与配体的成键原子对，避免重复）
    atom_pairs_to_cut = set()  # 用集合避免重复的原子对
    for fe_idx in fe_atom_indices:
        fe_atom = mol.GetAtomWithIdx(fe_idx)
        # 遍历 Fe 原子的所有连接键，收集（Fe 索引，配体原子索引）原子对
        for bond in fe_atom.GetBonds():
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            # 确定 Fe 原子和配体原子（保证原子对顺序统一，避免 (a,b) 和 (b,a) 重复）
            metal_idx, ligand_idx = (idx1, idx2) if idx1 == fe_idx else (idx2, idx1)
            # 存入集合（自动去重），格式：(较小索引, 较大索引)，确保唯一性
            atom_pair = (min(metal_idx, ligand_idx), max(metal_idx, ligand_idx))
            atom_pairs_to_cut.add(atom_pair)
    
    # 步骤5：创建可编辑分子（RDKit 中 Mol 对象不可直接修改，需用 EditableMol）
    editable_mol = Chem.EditableMol(mol)
    
    # 步骤6：切断 Fe 与配体的所有连接键（传入原子对，匹配 RemoveBond 签名）
    cut_count = 0
    for (idx1, idx2) in atom_pairs_to_cut:
        editable_mol.RemoveBond(idx1, idx2)  # 正确调用：传入两个原子索引
        cut_count += 1
    print(f"成功切断 {cut_count} 根 Fe 与配体的连接键")
    
    # 步骤7：转换为普通 Mol 对象，分离独立分子片段
    modified_mol = editable_mol.GetMol()
    # 分离片段：sanitizeFrags=False 保留配体和金属的电荷，避免结构被篡改
    fragments = rdmolops.GetMolFrags(modified_mol, asMols=True, sanitizeFrags=False)
    
    # 步骤8：筛选金属离子片段和有机配体片段
    fe_fragment = None
    ligand_fragment = None
    for frag in fragments:
        has_fe = any(atom.GetSymbol() == "Fe" for atom in frag.GetAtoms())
        if has_fe:
            fe_fragment = frag
        else:
            ligand_fragment = frag
    
    # 步骤9：补充校验（避免拆分失败）
    if fe_fragment is None or ligand_fragment is None:
        raise RuntimeError("分子片段分离失败，未找到 Fe 或配体片段")
    
    return fe_fragment, ligand_fragment

# -------------- 执行拆分 --------------
try:
    # 执行拆分
    smi = "C(C)[C@@H]1[P@@]([C-]23[Fe+2]456789%10([C-]%11([CH]4=[CH]5[CH]6=[CH]7%11)[P@@]%12[C@@H](CC)CC[C@@H]%12CC)[CH]2=[CH]8[CH]9=[CH]%103)[C@@H](CC)CC1"
    fe_mol, ligand_mol = split_fe_complex_into_ligand_and_metal(smi)
    
    # 步骤10：结果验证与输出
    print("\n=== 拆分结果验证 ===")
    # 输出金属离子信息
    print("1. 金属离子片段：")
    print(f"   SMILES: {Chem.MolToSmiles(fe_mol)}")
    print(f"   原子数: {fe_mol.GetNumAtoms()}")
    print(f"   正式电荷: {sum(atom.GetFormalCharge() for atom in fe_mol.GetAtoms())}")
    
    # 输出有机配体信息
    print("\n2. 有机配体片段：")
    print(f"   SMILES: {Chem.MolToSmiles(ligand_mol)}")
    print(f"   原子数: {ligand_mol.GetNumAtoms()}")
    print(f"   正式电荷: {sum(atom.GetFormalCharge() for atom in ligand_mol.GetAtoms())}")
    
    ## 可选：保存为 SDF 文件（可用 PyMOL/Avogadro 打开查看）
    #Chem.MolToMolFile(fe_mol, "Fe_ion.sdf")
    #Chem.MolToMolFile(ligand_mol, "organic_ligand.sdf")
    #print("\n3. 结果已保存为 Fe_ion.sdf 和 organic_ligand.sdf")
    
except Exception as e:
    print(f"拆分失败：{e}")