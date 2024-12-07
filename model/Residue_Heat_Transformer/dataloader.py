from torch.utils.data import DataLoader, Dataset
import torch

atom2num = {
    '<PAD>': 0, 'N': 1, 'CA': 2, 'C': 3, 'O': 4, 'CB': 5, 'CD': 6, 'CD1': 7, 'CD2': 8, 'CE': 9, 'CE1': 10, 'CE2': 11, 'CE3': 12, 'CG': 13, 'CG1':14,
    'CG2': 15, 'CH2': 16, 'CZ': 17, 'CZ2': 18, 'CZ3': 19, 'ND1': 20, 'ND2': 21, 'NH1': 22, 'NH2': 23, 'NE': 24, 'NE1': 25,
    'NE2':26, 'NZ': 27, 'OD1': 28, 'OD2': 29, 'OE1': 30, 'OE2': 31, 'OG': 32, 'OG1': 33, 'OH': 34, 'SD':35, 'SG':36, 
    'P':37, 'O1P':38, 'O2P':39, 'O3P':40,
    'CM':41, 'CH3':42, 'CH':43, 'CB1':44, 'CB2':45, 'OD':46, 'OE':47, 'CAA':48, 'CAF':49, 'CAL':50, 'OAD':51, 'OZ1':52, 'OE21':53, 'OE11':54, 'OE12':55, 'CH1':56,
    'CM1':57, 'CM2':58, 'CM3':59, 'CH1':60, 'CQ1':61, 'CQ2':62, 'OE22':63, ##下面开始是测试集的
    'NT':64, 'CX':65, 'S':66, 'O1':67, 'O2':68, 'O3':69, 'C1':70, 'N1':71, 'CL':72, 'C7':73, 'CI':74, 'CJ':75, 'CN':76, 'C6':77, 'N8':78, 'C9':79, 'C10':80, 'C11':81, 
    'NG2':82, 'NSC':83, 'CK2':84, 'CI2':85, 'CT':86, 'CI1':87, 'CK1':88, 'CN1':89, 'ON2':90,
    # 'CAE': 91, 'CAH': 92, 'CAI': 93, 'CAO': 94
    }

aa2num = {
    '<PAD>':0, '<SOP>':1, '<EOP>':2, '<UNK>':3, 'ALA':4, 'ARG':5, 'ASN':6, 'ASP':7, 'CYS':8, 'GLU':9, 'GLN':10, 'GLY':11, 'HIS':12, 'ILE':13,
    'LEU':14, 'LYS':15, 'MET':16, 'PHE':17, 'PRO':18, 'SER':19, 'THR':20, 'TRP':21, 'TYR':22, 'VAL':23, 
    'PTR':24, 'HYP':25, 'NLE':26, 'MLZ':27, 'SEP':28, 'TPO':29, 'ALY':30, 'ABA':31, 'AIB':32, 'CSO':33, 'NVA':34, 'PCA':35, 'PRK':36, 'ALC':37, 'CCS':38, 'CGU':39, 
    'M3L':40, 'MLY':41, '2MR':42,
    '4BF':43, '6CV':44, '6DU':45, 'CIR':46, '6CW':47, 'NAL':48, 'MEA':49, 'ORN':50, 'SET':51, 'ORQ':52, '4CF':53, '3CF':54, '4CY':55, 'DIV':56, 'MK8':57, '1MH':58, 'PH8':59, 
    'PBF':60, 'EME':61, 'TYS':62, 'AHP':63, 'GME':64, 'UN1':65, 'LYN':66, 'FME':67, 'HCS':68, 'RVJ':69, '2AG':70, 'IML':71, 'BTK':72, 'KCR':73, 'ORN1':74, 'JKH':75,
    # 'CY3': 76, '2GX': 77

}

num2aa = {v: k for k, v in aa2num.items()}

def data_stat(data):
    atom_max = []
    res_max = []
    chain_max = []
    for sample in data:
        res_max.append(len(sample[0]))
        atom_max.append(len(sample[1]))
    return max(res_max), max(atom_max)

def padding(raw_data, atom_len_max, res_len_max):
    sents = []
    for sample in raw_data:
        res_num = len(sample[0])
        atom_num = len(sample[1])
        sample_tgt = sample[0].copy() + ['<EOP>']
        sample[0].extend(['<PAD>'] * max(res_len_max - res_num, 0))
        sample_tgt.extend(['<PAD>'] * max(res_len_max - res_num, 0))
        sample[1].extend(['<PAD>'] * max(atom_len_max - atom_num, 0))
        sents.append([' '.join(sample[1]),
                      '<SOP> ' + ' '.join(sample[0]),
                      ' '.join(sample_tgt)])
    return sents

def make_data(sents):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sents)):
#        print(sentences[i])
        enc_input = [[atom2num[n] for n in sents[i][0].split()]]
        dec_input = [[aa2num[n] for n in sents[i][1].split()]]
        dec_output = [[aa2num[n] for n in sents[i][2].split()]]
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    return [torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs),
            torch.LongTensor(dec_outputs)]  # 输出三个张量列表，第一个：编码器输入（原子标识符-有填充），第二个：解码器输入（残基-有<sop>），第三个：解码器输出（残基-有<eop>）

class MyDataset(Dataset):
    def __init__(self, data_final):
        self.enc_inputs = data_final[0]
        self.dec_inputs = data_final[1]
        self.dec_outputs = data_final[2]

    def __len__(self):
        return self.enc_inputs.shape[0]
    
    def __getitem__(self,idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]



def data_loader(txt_file, batch_size=1):
    with open(txt_file, 'r') as f:
        true_data = f.read()
    raw_data = eval(true_data)
    res_len_max, atom_len_max = data_stat(raw_data) # 30 240
    sentences = padding(raw_data, atom_len_max, res_len_max) # 240 30
    data_final = make_data(sentences)
    dataloader = DataLoader(MyDataset(data_final), batch_size=batch_size)
    return dataloader
