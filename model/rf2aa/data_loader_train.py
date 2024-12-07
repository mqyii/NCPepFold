import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pickle
import copy

atom2num = {
    '<PAD>': 0, 'N': 1, 'CA': 2, 'C': 3, 'O': 4, 'CB': 5, 'CD': 6, 'CD1': 7, 'CD2': 8, 'CE': 9, 'CE1': 10, 'CE2': 11, 'CE3': 12, 'CG': 13, 'CG1':14,
    'CG2': 15, 'CH2': 16, 'CZ': 17, 'CZ2': 18, 'CZ3': 19, 'ND1': 20, 'ND2': 21, 'NH1': 22, 'NH2': 23, 'NE': 24, 'NE1': 25,
    'NE2':26, 'NZ': 27, 'OD1': 28, 'OD2': 29, 'OE1': 30, 'OE2': 31, 'OG': 32, 'OG1': 33, 'OH': 34, 'SD':35, 'SG':36, 
    'P':37, 'O1P':38, 'O2P':39, 'O3P':40,
    'CM':41, 'CH3':42, 'CH':43, 'CB1':44, 'CB2':45, 'OD':46, 'OE':47, 'CAA':48, 'CAF':49, 'CAL':50, 'OAD':51, 'OZ1':52, 'OE21':53, 'OE11':54, 'OE12':55, 'CH1':56,
    'CM1':57, 'CM2':58, 'CM3':59, 'CH1':60, 'CQ1':61, 'CQ2':62, 'OE22':63, ##下面开始是测试集的
    'NT':64, 'CX':65, 'S':66, 'O1':67, 'O2':68, 'O3':69, 'C1':70, 'N1':71, 'CL':72, 'C7':73, 'CI':74, 'CJ':75, 'CN':76, 'C6':77, 'N8':78, 'C9':79, 'C10':80, 'C11':81, 
    'NG2':82, 'NSC':83, 'CK2':84, 'CI2':85, 'CT':86, 'CI1':87, 'CK1':88, 'CN1':89, 'ON2':90,
    'CAE': 91, 'CAH': 92, 'CAI': 93, 'CAO': 94, 'CAM': 95, 'CLL': 96, 'BR': 97, 'C4': 98, 'C5': 99, 'N6': 100, 'O7': 101
    }

aa2num = {
    '<PAD>':0, '<SOP>':1, '<EOP>':2, '<UNK>':3, 'ALA':4, 'ARG':5, 'ASN':6, 'ASP':7, 'CYS':8, 'GLU':9, 'GLN':10, 'GLY':11, 'HIS':12, 'ILE':13,
    'LEU':14, 'LYS':15, 'MET':16, 'PHE':17, 'PRO':18, 'SER':19, 'THR':20, 'TRP':21, 'TYR':22, 'VAL':23, 
    'PTR':24, 'HYP':25, 'NLE':26, 'MLZ':27, 'SEP':28, 'TPO':29, 'ALY':30, 'ABA':31, 'AIB':32, 'CSO':33, 'NVA':34, 'PCA':35, 'PRK':36, 'ALC':37, 'CCS':38, 'CGU':39, 
    'M3L':40, 'MLY':41, '2MR':42, # 下面开始是测试集的
    '4BF':43, '6CV':44, '6DU':45, 'CIR':46, '6CW':47, 'NAL':48, 'MEA':49, 'ORN':50, 'SET':51, 'ORQ':52, '4CF':53, '3CF':54, '4CY':55, 'DIV':56, 'MK8':57, '1MH':58, 'PH8':59, 
    'PBF':60, 'EME':61, 'TYS':62, 'AHP':63, 'GME':64, 'UN1':65, 'LYN':66, 'FME':67, 'HCS':68, 'RVJ':69, '2AG':70, 'IML':71, 'BTK':72, 'KCR':73, 'ORN1':74, 'JKH':75,
    'CY3': 76, '2GX': 77

}

num2aa = {v: k for k, v in aa2num.items()}

def data_stat(data):
    atom_max = []
    res_max = []
    for name,sample in data.items():
        res_max.append(len(sample[8]))
        atom_max.append(len(sample[9]))
    return max(res_max), max(atom_max)

def padding_tmp(raw_data, atom_len_max, res_len_max):
    data = copy.deepcopy(raw_data)

    for name, sample in raw_data.items():
        res_num = len(sample[8])
        atom_num = len(sample[9])
        sample_tgt = sample[8].copy() + ['<EOP>']
        sample[8].extend(['<PAD>'] * max(res_len_max - res_num, 0))
        sample_tgt.extend(['<PAD>'] * max(res_len_max - res_num, 0))
        sample[9].extend(['<PAD>'] * max(atom_len_max - atom_num, 0))
        data[name].append(' '.join(sample[9])) #10
        data[name].append('<SOP> ' + ' '.join(sample[8])) #11
        data[name].append(' '.join(sample_tgt)) #12
    return data

def padding(raw_data, atom_len_max, res_len_max):
    sents = {}
    for name, sample in raw_data.items():
        res_num = len(sample[8])
        atom_num = len(sample[9])
        sample_tgt = sample[8].copy() + ['<EOP>']
        sample[8].extend(['<PAD>'] * max(res_len_max - res_num, 0))
        sample_tgt.extend(['<PAD>'] * max(res_len_max - res_num, 0))
        sample[9].extend(['<PAD>'] * max(atom_len_max - atom_num, 0))
        sents[name] = [' '.join(sample[9]),
                      '<SOP> ' + ' '.join(sample[8]),
                      ' '.join(sample_tgt)]
    return sents

def info_to_numberic(sents):
    info = {}
    for name, sample in sents.items():
        enc_input = [[atom2num[n] for n in sample[0].split()]]
        dec_input = [[aa2num[n] for n in sample[1].split()]]
        dec_output = [[aa2num[n] for n in sample[2].split()]]

        info[name] = [torch.LongTensor(enc_input), torch.LongTensor(dec_input), torch.LongTensor(dec_output)]

    return info  # 输出字典，包括三个张量列表，第一个：编码器输入（原子标识符-有填充），第二个：解码器输入（残基-有<sop>），第三个：解码器输出（残基-有<eop>）

def make_data(data_attn, raw_data):
    """

    data_attn: {"name":[tensor(enc_inputs), tensor(dec_inputs), tensor(dec_outputs)]}
    raw_data: {"name":[主链坐标(0), 全部aa数量(1), 非标准氨基酸索引列表(2), ncaa数量(3), 每个ncaa的原子个数列表(4), 肽链长度(5),  
                      肽链chain id(字符）(6), 蛋白质chain id(列表)(7), 肽链的残基名(8), 肽链的原子标识符(9)]}

    """
    data = {}
    for name, sample in data_attn.items():
        final_data = raw_data[name] + sample
        data[name] = final_data
    return data

def my_collate_fn(batch):
    names = [item[0] for item in batch]  
    configs = [item[1] for item in batch]
    rfaa_infos = [item[2] for item in batch]
    attn_infos = [item[3] for item in batch]  
    return names, configs, rfaa_infos, attn_infos

class MyDataset(Dataset):
    def __init__(self, root_path, data_final):
        self.data = data_final
        root_path = Path(root_path)
        self.config_path = {}
        self.names = []
        for subdir in root_path.iterdir():
            name = subdir.name.lower()
            yaml_file = subdir / "config_aa.yaml"
            if yaml_file.exists():
                self.config_path[name] = str(yaml_file)
                self.names.append(name)
            else:
                print(f"YAML file not found in {subdir.name}")
    def __len__(self):
        return len(self.config_path)

    def __getitem__(self, idx):
        name = self.names[idx]
        cfg_path = self.config_path[name]
        sample = self.data[name]

        coordinates = sample[0]
        aa_num = sample[1]
        ncaa_idx = sample[2]
        ncaa_num = sample[3]
        ncaa_atom_num = sample[4]
        pep_len = sample[5]
        pep_chain = sample[6]
        protein_chain = sample[7]

        pep_atom_count = sample[10]
        pdb_size = sample[11]
        flag_complex = sample[12]

        enc_inputs = sample[13]
        dec_inputs = sample[14]
        dec_outputs = sample[15]

        rfaa_info = [coordinates, aa_num, ncaa_idx, ncaa_num, ncaa_atom_num, pep_len, pep_chain, protein_chain, pep_atom_count, pdb_size, flag_complex]
        attn_info = [enc_inputs, dec_inputs, dec_outputs]

        return name, cfg_path, rfaa_info, attn_info


def data_loader(root_path, txt_file, batch_size=1, collate_fn=None):
    """
    txt_file: {名字：[主链坐标(0), 全部aa数量(1), 非标准氨基酸索引列表(2), ncaa数量(3), 每个ncaa的原子个数列表(4), 肽链长度(5), 肽链chain id(字符）(6), 
                     蛋白质chain id(列表)(7), 肽链的残基名(8), 肽链的原子标识符(9)]}
    
    """
    
    with open(txt_file, 'r') as f:
        true_data = f.read()
    raw_data = eval(true_data) # 字典
    res_len_max, atom_len_max = data_stat(raw_data) # 30 240
    sentences = padding(raw_data, atom_len_max, res_len_max) # 240 30 返回的是含有3个信息的字典
    data_attn = info_to_numberic(sentences) # 字典，attn信息转化为数字 {"name":[tensor(enc_inputs), tensor(dec_inputs), tensor(dec_outputs)]}
    data_final = make_data(data_attn, raw_data)
    # data_final: {"name":[主链坐标(0), 全部aa数量(1), 非标准氨基酸索引列表(2), ncaa数量(3), 每个ncaa的原子个数列表(4), 肽链长度(5), 肽链chain id(字符）(6), 
    #             蛋白质chain id(列表)(7), 肽链的残基名(8), 肽链的原子标识符(9), tensor([enc_inputs])(10), tensor([dec_inputs])(11), tensor([dec_outputs])(12)]}
    dataloader = DataLoader(MyDataset(root_path, data_final), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader

def data_loader_prediction(info):
    raw_data = info
    res_len_max, atom_len_max = data_stat(raw_data)
    sentences = padding(raw_data, atom_len_max, res_len_max)
    data_attn = info_to_numberic(sentences) # 字典，attn转化为数字 {"name":[tensor(enc_inputs), tensor(dec_inputs), tensor(dec_outputs)]}
    data_final = make_data(data_attn, raw_data) # 字典
    return data_final # 字典


def load_samples(pkl_file):
    with open(pkl_file, 'rb') as f:
        while True:
            try:
                all_feats = pickle.load(f)
                for name, value in all_feats.items():
                    feats, bond_feat, rfaa_info, attn_info = value
                    yield name, feats, bond_feat, rfaa_info, attn_info
            except EOFError:
                break       