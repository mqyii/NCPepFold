from Bio.PDB import PDBParser
from pathlib import Path

aa2atom = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB'],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG'],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
    'GLY': ['N', 'CA', 'C', 'O'],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG'],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],###
    "PTR": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ','OH', 'P','O1P', 'O2P', 'O3P'],
    "HYP": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OD1'],
    "NLE": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE'],
    "MLZ": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'CM'],
    "SEP": ['N', 'CA', 'C', 'O', 'CB', 'OG', 'P','O1P', 'O2P', 'O3P'],
    "TPO": ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', 'P','O1P', 'O2P', 'O3P'],
    "ALY": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'CH3', 'CH', 'OH'],
    "ABA": ['N', 'CA', 'C', 'O', 'CB', 'CG'],
    "AIB": ['N', 'CA', 'C', 'O', 'CB1', 'CB2'],
    "CSO": ['N', 'CA', 'C', 'O', 'CB', 'SG', 'OD'],
    "NVA": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
    "PCA": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE'],
    "PRK": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE','NZ','CAA','CAF','CAL','OAD'],
    "ALC": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD2','CE2','CZ','CE1','CD1'],
    "CCS": ['N', 'CA', 'C', 'O', 'CB', 'SG', 'CD','CE','OZ1'],
    "CGU": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD2','OE21','OE22','CD1','OE11','OE12'],
    "M3L": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'CM1', 'CM2', 'CM3'],
    "MLY": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'CH1', 'CH2'],
    "2MR": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'CQ1', 'NH2', 'CQ2'], ####下面是测试集
    "4BF": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ','BR'],
    "6CV": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ','BR'],
    "6DU": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ','BR'],
    "CIR": ['N', 'CA', 'C', 'O', 'CB', 'C4', 'C5','N6', 'C7', 'O7', 'N8'],
    "6CW": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'CLL'],
    "NAL": ['N', 'CA', 'C', 'O', 'CB','C1', 'C2', 'C3', 'C4', 'C4A', 'C5','C6','C7','C8','C8A'],
    "MEA": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ','C1'],
    "ORN": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
    "SET": ['N', 'CA', 'C', 'O', 'CB', 'OG', 'NT'],
    "ORQ": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'O1', 'C1', 'C2'],
    "4CF": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'C1', 'N1'],
    "3CF": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'C1', 'N1'],
    "4CY": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', 'NSC'],
    "DIV": ['N', 'CA', 'C', 'O', 'CB2', 'CB1', 'CG1'],
    "MK8": ['N', 'CA', 'C', 'O', 'CB1', 'CB', 'CG', 'CD', 'CE'],
    "1MH": ['N', 'CA', 'C', 'O', 'CB', 'C6', 'C7', 'N8', 'C9', 'C10', 'C11'],
    "PH8": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CI', 'CJ', 'CZ', 'CD1', 'CD2', 'CE1', 'CE2'],
    "PBF": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'CL', 'CK2', 'CI2', 'CT', 'CI1', 'CK1', 'CN1', 'ON2'],
    "EME": ['N', 'CA', 'C', 'O', 'CB', 'CG' , 'CD', 'OE1', 'OE2', 'C7'],
    "TYS": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', 'S', 'O1', 'O2', 'O3'],
    "AHP": ['N', 'CA', 'C', 'O', 'CB', 'CG' , 'CD', 'CE', 'CZ'],
    "GME": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CX', 'OE1', 'OE2'],
    "UN1": ['N', 'CA', 'C', 'O', 'C5', 'C6', 'C7', 'C8', 'O9'],
    "LYN": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ','NT'],
    "FME": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', 'CN', 'O1'],
    "HCS": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD'],
    "RVJ": ['N', 'CA', 'C', 'O', 'CB', 'NG', 'ND', 'NE'],
    "2AG": ['N', 'CA', 'C', 'O', 'CB', 'C1E', 'C1A'],
    "IML": ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', 'CN'],
    "BTK": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'CAA', 'OAD', 'CAF', 'CAJ', 'CAN'],
    "KCR": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'CH', 'OH', 'CX', 'CY', 'CH3'],
    "ORN1": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE']
}

standard_amino_acids = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 
        'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 
        'TYR', 'VAL'
    }

def sort_atoms(res_name, atom_list):
    atom_order = aa2atom.get(res_name, [])
    sorted_atom_list = sorted(atom_list, key=lambda x: atom_order.index(x) if x in atom_order else float('inf'))
    return sorted_atom_list

def select_chain(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", pdb_file)
    chain_length_min = float("inf")
    chains = []
    for chain in structure[0]:
        chains.append(chain.id)
        chain_length = len(list(residue.id for residue in chain.get_residues()))
        if chain_length < chain_length_min:
            chain_length_min = chain_length
            true_pep_chain_id = [chain.id]
    true_protein_chain_id = sorted([chain_id for chain_id in chains if chain_id != true_pep_chain_id[0]])  # monomer is []
    return true_pep_chain_id, true_protein_chain_id # 两个都是列表

def round_coords(coord_array):
        return [round(coord, 3) for coord in coord_array]

def extract_coordinates_and_nonstandard_indices(pdb_file,pep_chain_id, protein_chain_ids):
    '''
    原子顺序: [N, CA, C]
    '''
    mainchain_coordinates = [] # 全部的主链原子坐标
    non_standard_indices = [] # pep链中ncaa索引
    atom_num = [] # pep链中每个ncaa的原子数
    aa_num = 0 # 全部中aa的个数

    res_pep = [] # pep中的残基名,attn
    atoms_pep = [] # pep的所有原子标识符,attn

    pep_atom_count = [] # pep中每一个残基的原子数
    pdb_size = [] # pdb文件中每一条链中的标准氨基酸的数量

    # print(protein_chain_ids)

    if protein_chain_ids == []: # 0表示单体，1表示复合物
        flag_complex = 0
    else:
        flag_complex = 1

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('peptide', pdb_file)
    pep_aa_num = 0
    pep_len = len(list(structure[0][pep_chain_id].get_residues()))
    for res_index, residue in enumerate(structure[0][pep_chain_id]):
        res_name = residue.get_resname()
        res_pep.append(res_name)
        pep_atom_count.append(len([atom for atom in residue.get_atoms() if not atom.element == 'H']))
        
        if res_name not in standard_amino_acids:
            non_standard_indices.append(res_index)
            atom_num.append(len([atom for atom in residue.get_atoms() if not atom.element == 'H']))
        else:
            aa_num += 1
            pep_aa_num += 1

        try:
            atom_N = residue['N']
            atom_CA = residue['CA']
            atom_C = residue['C']

            n_coord = round_coords(atom_N.get_coord().tolist())
            ca_coord = round_coords(atom_CA.get_coord().tolist())
            c_coord = round_coords(atom_C.get_coord().tolist())

            mainchain_coordinates += n_coord, ca_coord, c_coord
        except KeyError as e:
            print(f"Missing atom {e} in residue {residue}")

        atoms_tmp = []
        for atom in residue:
            if atom.element != 'H' and atom.get_name() != 'OXT':
                atom_name = atom.get_name()
                atoms_tmp.append(atom_name)
        sorted_atoms = sort_atoms(res_name, atoms_tmp) # list
        atoms_pep.extend(sorted_atoms)

    ncaa_num = len(non_standard_indices)
    pdb_size.append(pep_aa_num)

    for protein_chain_id in protein_chain_ids:
        prot_aa_num = 0
        for res_index, residue in enumerate(structure[0][protein_chain_id]):
            aa_num += 1
            prot_aa_num += 1

            try:
                atom_N = residue['N']
                atom_CA = residue['CA']
                atom_C = residue['C']

                n_coord = round_coords(atom_N.get_coord().tolist())
                ca_coord = round_coords(atom_CA.get_coord().tolist())
                c_coord = round_coords(atom_C.get_coord().tolist())

                mainchain_coordinates += n_coord, ca_coord, c_coord
            except KeyError as e:
                print(f"Missing atom {e} in residue {residue}")
        pdb_size.append(prot_aa_num)

    total_info = [mainchain_coordinates, aa_num, non_standard_indices, ncaa_num, atom_num, pep_len, res_pep, atoms_pep, pep_atom_count, pdb_size, flag_complex]
    # pep链中：[主链坐标，全部aa数量，非标准氨基酸索引列表，ncaa数量，每个ncaa的原子个数列表，肽链长度, 肽链的残基名, 肽链的原子标识符, 
    #          pep中每一个残基的原子数(除H), pdb文件中每一条链中的标准氨基酸的数量, 复合物标识]
    # return total_info
    return mainchain_coordinates, aa_num, non_standard_indices, ncaa_num, atom_num, pep_len, res_pep, atoms_pep, pep_atom_count, pdb_size, flag_complex


def get_info(path):
    txt_all = {}
    if 'CB' in path.name or 'CP' in path.name:
        pdb_file = path / f"{path.name}.pdb"
    else:
        pdb_file = path / f"{path.name}_clean.pdb"
    pep_chain_id, protein_chain_id = select_chain(pdb_file) # 列表
    name = path.name.lower()
    mainchain_coordinates, aa_num, non_standard_indices, ncaa_num, atom_num, pep_len, res_pep, atoms_pep, pep_atom_count, pdb_size, flag_complex = extract_coordinates_and_nonstandard_indices(str(pdb_file), pep_chain_id[0], protein_chain_id)

    one_sample_info = [mainchain_coordinates, aa_num, non_standard_indices, ncaa_num, atom_num, pep_len, pep_chain_id[0], protein_chain_id, res_pep, atoms_pep, pep_atom_count, pdb_size, flag_complex]
    # [主链坐标，全部aa数量，非标准氨基酸索引列表，ncaa数量，每个ncaa的原子个数列表，肽链长度, 肽链chain id(字符），蛋白质chain id（列表），肽链的残基名，肽链的原子标识符]
    txt_all[name] = one_sample_info
    return txt_all
