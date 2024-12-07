import os
import hydra
import torch
import torch.nn as nn
from Bio import SeqIO 
from pathlib import Path
from dataclasses import asdict
from omegaconf import DictConfig, OmegaConf
import torch.optim as optim
import datetime
from pympler.asizeof import asizeof

from rf2aa.data.merge_inputs import merge_all
from rf2aa.data.covale import load_covalent_molecules
from rf2aa.data.nucleic_acid import load_nucleic_acid
from rf2aa.data.protein import generate_msa_and_load_protein
from rf2aa.data.small_molecule import load_small_molecule
from rf2aa.ffindex import *
from rf2aa.chemical import initialize_chemdata, load_pdb_ideal_sdf_strings
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.model.RoseTTAFoldModel import RoseTTAFoldModule
from rf2aa.training.recycling import recycle_step_legacy
from rf2aa.util import writepdb, is_atom, Ls_from_same_chain_2d
from rf2aa.util_module import XYZConverter
from rf2aa.fix_pdb import fix
from rf2aa.loss.loss_train import RFAALoss
from rf2aa.data_loader_train import data_loader, load_samples, my_collate_fn, data_loader_prediction
from rf2aa.util_train import shared_state, split_and_concat_xyz_pred
from rf2aa.mapping import mapping_atom2aa, mapping_atom2aa_multimer
from rf2aa.attn_process import get_info

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from m_Transformer.m_transformer.model import Transformer


from rf2aa.util import ConfigPro
import traceback
import json
import logging
import networkx as nx
import pickle


runner_version = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
# getLogger获取日志logger
logger = logging.getLogger()

# 设置日志记录等级
logger.setLevel(logging.INFO)

# 创建输出格式：时间、日志等级、日志内容
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 设置控制台输出
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# 设置文件输出
# file_handler = logging.FileHandler(filename='log_'+runner_version+'.log', mode='a')
# file_handler.setFormatter(formatter)

# 将设置好的输出处理器添加到logger
logger.addHandler(console_handler)
# logger.addHandler(file_handler)

class ModelRunner:

    def __init__(self, config) -> None:
        self.config = config
        initialize_chemdata(self.config.chem_params)
        FFindexDB = namedtuple("FFindexDB", "index, data")
        self.ffdb = FFindexDB(read_index(config.database_params.hhdb+'_pdb.ffindex'),
                              read_data(config.database_params.hhdb+'_pdb.ffdata'))
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.xyz_converter = XYZConverter()
        self.deterministic = config.get("deterministic", False)
        self.molecule_db = load_pdb_ideal_sdf_strings()
        self.nc_cycle = self.config.cyclize
        self.seq_len = self.pep_length()


    def parse_inference_config(self):
        self.clean_path = Path(self.config.output_path)

        residues_to_atomize = [] # chain letter, residue number, residue name
        chains = []
        protein_inputs = {}
        if self.config.protein_inputs is not None:
            for chain in self.config.protein_inputs:
                if chain in chains:
                    raise ValueError(f"Duplicate chain found with name: {chain}. Please specify unique chain names")
                elif len(chain) > 1:
                    raise ValueError(f"Chain name must be a single character, found chain with name: {chain}")
                else:
                    chains.append(chain)
                protein_input = generate_msa_and_load_protein(
                    self.config.protein_inputs[chain]["fasta_file"],
                    chain,
                    self,
                    nc_cycle = self.nc_cycle  # 🌈
                ) 
                protein_inputs[chain] = protein_input
        
        na_inputs = {}
        if self.config.na_inputs is not None:
            for chain in self.config.na_inputs:
                na_input = load_nucleic_acid(
                    self.config.na_inputs[chain]["fasta"],
                    self.config.na_inputs[chain]["input_type"],
                    self
                )
                na_inputs[chain] = na_input

        sm_inputs = {} 
        # first if any of the small molecules are covalently bonded to the protein
        # merge the small molecule with the residue and add it as a separate ligand
        # also add it to residues_to_atomize for bookkeeping later on
        # need to handle atomizing multiple consecutive residues here too
        if self.config.covale_inputs is not None:
            covalent_sm_inputs, residues_to_atomize_covale = load_covalent_molecules(protein_inputs, self.config, self)
            for a, b in zip(covalent_sm_inputs, residues_to_atomize_covale):
                if len(a) != len(b): # 🌈🌈🌈 
                    pass
                else:
                    sm_inputs.update(a)
                    residues_to_atomize.extend(b)
            # sm_inputs.update(covalent_sm_inputs)
            # residues_to_atomize.extend(residues_to_atomize_covale)
        self.residues_to_atomize = residues_to_atomize  #🎀
        if self.config.sm_inputs is not None:
            for chain in self.config.sm_inputs:
                if self.config.sm_inputs[chain]["input_type"] not in ["smiles", "sdf"]:
                    raise ValueError("Small molecule input type must be smiles or sdf")
                if chain in sm_inputs: # chain already processed as covale
                    continue
                if "is_leaving" in self.config.sm_inputs[chain]:
                    raise ValueError("Leaving atoms are not supported for non-covalently bonded molecules")
                sm_input = load_small_molecule(
                   self.config.sm_inputs[chain]["input"],
                   self.config.sm_inputs[chain]["input_type"],
                   self
                )
                sm_inputs[chain] = sm_input

        if self.config.residue_replacement is not None:
            # add to the sm_inputs list
            # add to residues to atomize
            raise NotImplementedError("Modres inference is not implemented")
        
        raw_data = merge_all(protein_inputs, na_inputs, sm_inputs, residues_to_atomize, deterministic=self.deterministic)
        self.raw_data = raw_data

    def load_model(self):
        self.model = RoseTTAFoldModule(
            **self.config.legacy_model_param,
            aamask = ChemData().allatom_mask.to(self.device),
            atom_type_index = ChemData().atom_type_index.to(self.device),
            ljlk_parameters = ChemData().ljlk_parameters.to(self.device),
            lj_correction_parameters = ChemData().lj_correction_parameters.to(self.device),
            num_bonds = ChemData().num_bonds.to(self.device),
            cb_len = ChemData().cb_length_t.to(self.device),
            cb_ang = ChemData().cb_angle_t.to(self.device),
            cb_tor = ChemData().cb_torsion_t.to(self.device),
            nc_cycle = self.nc_cycle,  # 🌈
            seq_len = self.seq_len

        ).to(self.device)
        # checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        checkpoint = torch.load("/home/light/mqy/ncaa/outputs/rfaa_20241105101418_1e-05_192.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint)

        self.transformer_model = Transformer().to(self.device)
        checkpoint_attn = torch.load("/home/light/mqy/ncaa/outputs/attn_20241105101418_0.0001_192.pt", map_location=self.device)
        self.transformer_model.load_state_dict(checkpoint_attn)

    def construct_features(self):
        return self.raw_data.construct_features(self)

    def run_model_forward(self, input_feats, attn_feats=None):
        input_feats.add_batch_dim()
        input_feats.to(self.device)
        input_dict = asdict(input_feats)
        input_dict["bond_feats"] = input_dict["bond_feats"].long()
        input_dict["seq_unmasked"] = input_dict["seq_unmasked"].long()
        input_dict["attn_feats"] = attn_feats
        outputs = recycle_step_legacy(self.model, 
                                     input_dict, 
                                     self.config.loader_params.MAXCYCLE, 
                                     use_amp=False,
                                     nograds=True, ##################################################
                                     force_device=self.device)
        return outputs


    def write_outputs(self, input_feats, outputs):
        logits, logits_aa, logits_pae, logits_pde, p_bind, \
                xyz, alpha_s, xyz_allatom, lddt, _, _, _ \
            = outputs
        seq_unmasked = input_feats.seq_unmasked
        bond_feats = input_feats.bond_feats
        err_dict = self.calc_pred_err(lddt, logits_pae, logits_pde, seq_unmasked)
        print(f"PLDDT: {err_dict['mean_plddt']}")  # 🌈
        err_dict["same_chain"] = input_feats.same_chain
        plddts = err_dict["plddts"]
        
        if not self.residues_to_atomize:
            Ls = Ls_from_same_chain_2d(input_feats.same_chain)  # 🎀原  没有非标准的时候用这个 有非标准的时候用下面的
        else:
            _Ls = {k: v for k, v in self.raw_data.chain_lengths}
            for residue in self.residues_to_atomize:
                for chain in _Ls:
                    if chain == residue.original_chain:
                        _Ls[chain] -= 1
                Ls = []
                for k, v in _Ls.items():
                    Ls.append(v)
        # print(Ls)

        plddts = plddts[0]
        writepdb(os.path.join(f"{self.config.output_path}", f"predict_{self.config.job_name}_ckpt159.pdb"), 
                 xyz_allatom, 
                 seq_unmasked, 
                 bond_feats=bond_feats,
                 bfacts=plddts,
                 chain_Ls=Ls
                 )
        torch.save(err_dict, os.path.join(f"{self.config.output_path}", 
                                          f"{self.config.job_name}_aux_ckpt159.pt"))
    def fix_outputs(self):
        name = self.config.job_name
        clean_pdb = os.path.join(f"{self.config.output_path}", f"{self.config.job_name}_clean.pdb")
        predict_pdb = os.path.join(f"{self.config.output_path}", f"predict_{self.config.job_name}_ckpt159.pdb")
        fix(
            name = name,
            clean_pdb = clean_pdb,
            predict_pdb = predict_pdb,
            output_path= self.config.output_path
        )

    def infer(self):
        self.load_model()
        self.parse_inference_config()
        input_feats = self.construct_features()
        # print("aa", input_feats.same_chain.shape)
        outputs = self.run_model_forward(input_feats)
        self.write_outputs(input_feats, outputs)
        if self.config.covale_inputs is not None:
            self.fix_outputs()
    
    
    def load_to_pkl(self, dataloader, pkl_file, max_memory):
        # with open(pkl_file, 'rb') as f:
        #     data_tmp = pickle.load(f)
        # if isinstance(data_tmp, dict):
        #     id_have_loaded = set(data_tmp.keys())
        #     print(f"id_have_loaded: {id_have_loaded}")
        # else:
        #     print("The loaded data is not a dictionary.")
        
        all_feats = {}
        name_info = []
        for i,data in enumerate(dataloader):
            name = data[0][0]
            # print(name)
            # if name in id_have_loaded:
            #     print(f"{name} already loaded in pkl.")
            #     continue
    
            cfg_path =  data[1][0]
            rfaa_info = data[2][0]
            attn_info = data[3][0]
            self.config = OmegaConf.load(cfg_path)
            try:
                self.parse_inference_config()
                feats = self.construct_features()
                bond_feat = shared_state.computed_value
                all_feats[name] = [feats, bond_feat, rfaa_info, attn_info]
                # print(f"{name} has been load into all_feats")
                name_info.append(name)
            except Exception as e:
                print(f"{name} : {e}")
            

            size = asizeof(all_feats)
            size_gb = size / (1024 ** 3)
            print(f"size: {size}")
            if size >= max_memory:
                with open(pkl_file, 'ab') as f:
                    pickle.dump(all_feats, f)
                print(f"Saved {len(all_feats)} samples to {pkl_file}")
                print(f"name: {name_info}")
                all_feats.clear()
                name_info.clear()
            
            # if len(all_feats) == max_samples:
            #     with open(pkl_file, 'ab') as f:
            #         pickle.dump(all_feats, f)
            #     print(f"Saved {len(all_feats)} samples to {pkl_file}")
            #     all_feats.clear()


    def infer_prediction(self):
        weight_attn_feats = 1e-3
        self.load_model()
        self.parse_inference_config()
        input_feats = self.construct_features()
        info = get_info(self.clean_path) # dict{"name": info}
        info_all = data_loader_prediction(info) # 字典

        ncaa_idx = list(info_all.values())[0][2]
        ncaa_atom_num = list(info_all.values())[0][4]
        pep_atom_count = list(info_all.values())[0][10]
        pdb_size = list(info_all.values())[0][11]
        flag_complex = list(info_all.values())[0][12]
        enc_inputs = list(info_all.values())[0][13]
        dec_inputs = list(info_all.values())[0][14]
        dec_outputs = list(info_all.values())[0][15]

        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(self.device), dec_inputs.to(self.device), dec_outputs.to(self.device)
        outputs_attn, enc_self_attns, dec_self_attns, dec_enc_attns, enc_outputs = self.transformer_model.forward(enc_inputs, dec_inputs)
        print(enc_outputs.shape)
        attn_feats = mapping_atom2aa(enc_outputs, pep_atom_count, ncaa_idx, self.device)
        print(attn_feats.shape)
        attn_feats = mapping_atom2aa_multimer(attn_feats, flag_complex, pdb_size, ncaa_atom_num)
        print(attn_feats.shape)
        attn_feats = torch.mul(attn_feats, torch.ones_like(attn_feats) * weight_attn_feats)
        attn_feats = attn_feats.to(self.device)
        outputs = self.run_model_forward(input_feats,attn_feats)
        self.write_outputs(input_feats, outputs)
        if self.config.covale_inputs is not None:
            self.fix_outputs()

    def finetuning_infer_plus(self):
        self.load_model()

        # root_path_train = self.config.input_cfg #👿改
        # txt_path_train = "/home/light/mqy/ncaa/notebooks/tmp/train-2024-11-1.txt" #👿改
        # root_path_test = self.config.input_cfg_test #👿改
        # txt_path_test = "/home/light/mqy/ncaa/notebooks/tmp/test-2024-11-1.txt" #👿改

        # train_dataloader = data_loader(root_path=root_path_train, txt_file = txt_path_train, batch_size = 1, collate_fn=my_collate_fn)
        # test_dataloader = data_loader(root_path=root_path_test, txt_file = txt_path_test, batch_size = 1, collate_fn=my_collate_fn)

        # train_pkl_file = f"/home/light/mqy/ncaa/notebooks/tmp/{runner_version}_train.pkl" #👿改
        # test_pkl_file = f"/home/light/mqy/ncaa/notebooks/tmp/{runner_version}_test.pkl" #👿改
        # # max_memory = 3 * 1024 ** 3  # 4 GB
        # max_memory = 1 #1000000
        # self.load_to_pkl(train_dataloader, train_pkl_file, max_memory)
        # print("all training set loaded")
        # exit()
        # self.load_to_pkl(test_dataloader, test_pkl_file, max_memory)
        # print("all test set loaded")

        # exit()

        train_pkl_file = f"/home/light/mqy/ncaa/notebooks/tmp/20241104204213_train.pkl"
        test_pkl_file = f"/home/light/mqy/ncaa/notebooks/tmp/20241103001221_test.pkl"

        learning_rates = 1e-5  # 超参数 🤓
        learning_rates_attn = 1e-4  # 超参数 🤓
        optimizer = optim.Adam([
            {'params': self.transformer_model.parameters(), 'lr': learning_rates_attn},
            {'params': self.model.parameters(), 'lr': learning_rates},
        ])
        loss_rfaa = RFAALoss()

        weight_attn_feats = 1e-3  # 超参数 🤓

        with open(f'/home/light/mqy/ncaa/notebooks/tmp/loss_txt/FAPEloss_{runner_version}_training.txt', 'a') as f:
            f.write(f"learning_rate: {float(learning_rates)}, learning_rates_attn: {float(learning_rates_attn)}, weight_attn_feats: {float(weight_attn_feats)}\n")

        loss_list = []
        for epoch in range(500):
            if epoch == 0:
                pass_id = []
                exception_id = []
            print(f"Epoch: {epoch+1}")
            self.model.train()
            loss_epoch = 0.0
            n_sample_epoch = 0            
            for name, feats, bond_feat, true_info, attn_info in load_samples(train_pkl_file):
                print(f"    {name}")# ❤️
                true_coordinates = torch.tensor(true_info[0],dtype = torch.float32)
                true_coordinates = true_coordinates.reshape(-1, 3, 3).unsqueeze(0).to('cuda:0')
                aa_num = true_info[1]
                ncaa_idx = true_info[2]
                ncaa_num = true_info[3]
                ncaa_atom_num = true_info[4]
                pep_len = true_info[5]
                pep_chain = true_info[6]
                protein_chain = true_info[7]
                pep_atom_count = true_info[8] # [9,6,10,5,8,11...]
                pdb_size = true_info[9] # 所有链的标准氨基酸数
                flag_complex = true_info[10] # 是否是复合物，1表示复合物，2表示单体

                enc_inputs = attn_info[0]
                dec_inputs = attn_info[1]
                dec_outputs = attn_info[2]

                if aa_num >= 300:
                    print(f"training set {name} length >= 300, pass!!!")
                    if epoch == 0:
                        pass_id.append(name)
                    continue

                if epoch > 0:
                    feats.msa_latent = feats.msa_latent.squeeze(0)
                    feats.msa_full = feats.msa_full.squeeze(0)
                    feats.seq = feats.seq.squeeze(0)
                    feats.seq_unmasked = feats.seq_unmasked.squeeze(0)
                    feats.idx = feats.idx.squeeze(0)
                    feats.bond_feats = feats.bond_feats.squeeze(0)
                    feats.dist_matrix = feats.dist_matrix.squeeze(0)
                    feats.chirals = feats.chirals.squeeze(0)
                    feats.atom_frames = feats.atom_frames.squeeze(0)
                    feats.xyz_prev = feats.xyz_prev.squeeze(0)
                    feats.alpha_prev = feats.alpha_prev.squeeze(0)
                    feats.t1d = feats.t1d.squeeze(0)
                    feats.t2d = feats.t2d.squeeze(0)
                    feats.xyz_t = feats.xyz_t.squeeze(0)
                    feats.alpha_t = feats.alpha_t.squeeze(0)
                    feats.mask_t = feats.mask_t.squeeze(0)
                    feats.same_chain = feats.same_chain.squeeze(0)
                try:
                    enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(self.device), dec_inputs.to(self.device), dec_outputs.to(self.device)
                    outputs_attn, enc_self_attns, dec_self_attns, dec_enc_attns, enc_outputs = self.transformer_model.forward(enc_inputs, dec_inputs)
                    # print(f"enc_outputs.shape: {enc_outputs.shape}")
                    # with open("/home/light/mqy/m_Transformer/data/111.txt","w") as f:
                    #     f.write(f"{enc_outputs.detach().cpu().numpy().tolist()}\n")
                    # exit()
                    # enc_outputs   : [batch_size, 原子长度, d_model]
                    attn_feats = mapping_atom2aa(enc_outputs, pep_atom_count, ncaa_idx, self.device)
                    # print(f"attn_feats.shape: {attn_feats.shape}")
                    # tmp = torch.sum(attn_feats, dim=-1)
                    # with open("/home/light/mqy/m_Transformer/data/222.txt","w") as f:
                    #     f.write(f"{tmp.detach().cpu().numpy().astype(int).tolist()}\n")
                    attn_feats = mapping_atom2aa_multimer(attn_feats, flag_complex, pdb_size, ncaa_atom_num)
                    print(f"attn_feats_multimer.shape: {attn_feats.shape}")
                    # tmp = torch.sum(attn_feats, dim=-1)
                    # with open("/home/light/mqy/m_Transformer/data/111.txt","w") as f:
                    #     f.write(f"{tmp.detach().cpu().numpy().astype(int).tolist()}\n")
                    # exit()
                    attn_feats = torch.mul(attn_feats, torch.ones_like(attn_feats) * weight_attn_feats)
                    attn_feats = attn_feats.to(self.device)
                    
                    outputs = self.run_model_forward(feats,attn_feats)
                    xyz = outputs[5]
                    # print(f"        xyz.requires_grad: {xyz.requires_grad}")
                    # print(f"        xyz dimension: {xyz.shape}")# ❤️
                    pred = split_and_concat_xyz_pred(xyz, aa_num, ncaa_atom_num)
                    # ncaa_pred = xyz[:, :, -ncaa_atom_num:-ncaa_atom_num+3, 1, :]
                    # ncaa_pred = ncaa_pred.unsqueeze(2)
                    # print(f"        ncaa_pred dimension: {ncaa_pred.shape}")
                    # aa_pred = xyz[:, :, :pep_len-ncaa_num, :, :]
                    # print(f"        aa_pred dimension: {aa_pred.shape}")
                    # pred = torch.cat([aa_pred, ncaa_pred], dim=2) 
                    # print(f"        pred dimension: {pred.shape}")# ❤️

                    ncaa_true = true_coordinates[:,ncaa_idx,:,:]
                    mask = torch.ones(true_coordinates.shape[1], dtype=torch.bool)
                    mask[ncaa_idx] = False
                    aa_true = true_coordinates[:, mask, :, :]
                    # print(f"        aa_true: {aa_true.shape}, ncaa_true: {ncaa_true.shape}")

                    assert len(aa_true.shape) == len(ncaa_true.shape), "Shapes of aa_true and ncaa_true do not match"
                    true = torch.cat([aa_true, ncaa_true],dim=1) 
                    # print(f"        true dimension: {true.shape}")# ❤️
                    L = bond_feat.shape[0]
                    same_chain = torch.zeros((L,L))
                    G = nx.from_numpy_array(bond_feat.detach().cpu().numpy())
                    for idx in nx.connected_components(G):
                        idx = list(idx)
                        for i in idx:
                            same_chain[i,idx] = 1
                    same_chain = same_chain.unsqueeze(0)
                    # print(f"    same_chain dimension: {same_chain.shape}")# ❤️

                    mask_2d = torch.ones_like(same_chain)
                    # print(f"    mask_2d dimension: {mask_2d.shape}")# ❤️

                    loss = loss_rfaa(pred, true, mask_2d, same_chain)
                    print(f"    {name}: {loss:.5f}")
                    loss_epoch += loss
                    n_sample_epoch += 1

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                except Exception as e:
                    print(f"    {name} error: {e}")
                    if epoch == 0:
                        exception_id.append(name)
                    
            if epoch == 0:
                with open('/home/light/mqy/ncaa/notebooks/tmp/id_length_larger_than_300_20241104.txt', 'w') as f:
                    for i in pass_id:
                        f.write(f"{i}\n")
                with open('/home/light/mqy/ncaa/notebooks/tmp/id_exception_20241104.txt', 'w') as f:
                    for i in exception_id:
                        f.write(f"{i}\n")

            if (epoch+1) % 3 == 0:
                torch.save(self.model.state_dict(), '/home/light/mqy/ncaa/notebooks/tmp/ckpt/rfaa_' + runner_version + '_' + str(learning_rates) + '_' + str(epoch + 1) + '.pt')
                torch.save(self.transformer_model.state_dict(), '/home/light/mqy/ncaa/notebooks/tmp/ckpt/attn_' + runner_version + '_' + str(learning_rates_attn) + '_' + str(epoch + 1) + '.pt')

            loss_epoch /= n_sample_epoch
            loss_list.append(loss_epoch.item())

            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            logger.info(f'{current_time} - Epoch: {epoch + 1:04d} - loss = {loss_epoch.item():.6f}')
            each_epoch_info = f"{current_time} - Epoch: {epoch + 1:04d} - loss = {loss_epoch.item():.6f}"
            with open(f'/home/light/mqy/ncaa/notebooks/tmp/loss_txt/FAPEloss_{runner_version}_training.txt', 'a') as f: #👿改
                f.write(each_epoch_info + '\n')
            
            ########################################################################################################################
            self.model.eval()
            pass_id_test = []
            exception_id_test = []
            with torch.no_grad():
                loss_epoch_test = 0.0
                n_sample_epoch_test = 0
                for name, feats, bond_feat, true_info, attn_info in load_samples(test_pkl_file):
                    print(f"    {name}")# ❤️
                    if name in ['2fr9']:
                        continue
                    true_coordinates = torch.tensor(true_info[0],dtype = torch.float32)
                    true_coordinates = true_coordinates.reshape(-1, 3, 3).unsqueeze(0).to('cuda:0')
                    aa_num = true_info[1]
                    ncaa_idx = true_info[2]
                    ncaa_num = true_info[3]
                    ncaa_atom_num = true_info[4]
                    pep_len = true_info[5]
                    pep_chain = true_info[6]
                    protein_chain = true_info[7]

                    pep_atom_count = true_info[8] # [9,6,10,5,8,11...]
                    pdb_size = true_info[9] # 所有链的标准氨基酸数
                    flag_complex = true_info[10] # 是否是复合物，1表示复合物，2表示单体

                    enc_inputs = attn_info[0]
                    dec_inputs = attn_info[1]
                    dec_outputs = attn_info[2]

                    if aa_num >= 300:
                        print(f"test set {name} length >= 300, pass!!!")
                        pass_id_test.append(name)
                        continue

                    if epoch > 0:
                        feats.msa_latent = feats.msa_latent.squeeze(0)
                        feats.msa_full = feats.msa_full.squeeze(0)
                        feats.seq = feats.seq.squeeze(0)
                        feats.seq_unmasked = feats.seq_unmasked.squeeze(0)
                        feats.idx = feats.idx.squeeze(0)
                        feats.bond_feats = feats.bond_feats.squeeze(0)
                        feats.dist_matrix = feats.dist_matrix.squeeze(0)
                        feats.chirals = feats.chirals.squeeze(0)
                        feats.atom_frames = feats.atom_frames.squeeze(0)
                        feats.xyz_prev = feats.xyz_prev.squeeze(0)
                        feats.alpha_prev = feats.alpha_prev.squeeze(0)
                        feats.t1d = feats.t1d.squeeze(0)
                        feats.t2d = feats.t2d.squeeze(0)
                        feats.xyz_t = feats.xyz_t.squeeze(0)
                        feats.alpha_t = feats.alpha_t.squeeze(0)
                        feats.mask_t = feats.mask_t.squeeze(0)
                        feats.same_chain = feats.same_chain.squeeze(0)
                    try:
                        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(self.device), dec_inputs.to(self.device), dec_outputs.to(self.device)
                        outputs_attn, enc_self_attns, dec_self_attns, dec_enc_attns, enc_outputs = self.transformer_model.forward(enc_inputs, dec_inputs)
                        # print(enc_outputs.shape)
                        # enc_outputs   : [batch_size, 原子长度, d_model]
                        attn_feats = mapping_atom2aa(enc_outputs, pep_atom_count, ncaa_idx, self.device)
                        attn_feats = mapping_atom2aa_multimer(attn_feats, flag_complex, pdb_size, ncaa_atom_num)
                        attn_feats = torch.mul(attn_feats, torch.ones_like(attn_feats) * weight_attn_feats)
                        attn_feats = attn_feats.to(self.device)

                        outputs = self.run_model_forward(feats, attn_feats)
                        xyz = outputs[5]
                        # print(f"        xyz.requires_grad: {xyz.requires_grad}")
                        # print(f"        xyz dimension: {xyz.shape}")# ❤️
                        pred = split_and_concat_xyz_pred(xyz, aa_num, ncaa_atom_num)
                        # ncaa_pred = xyz[:, :, -ncaa_atom_num:-ncaa_atom_num+3, 1, :]
                        # ncaa_pred = ncaa_pred.unsqueeze(2)
                        # print(f"        ncaa_pred dimension: {ncaa_pred.shape}")
                        # aa_pred = xyz[:, :, :pep_len-ncaa_num, :, :]
                        # print(f"        aa_pred dimension: {aa_pred.shape}")
                        # pred = torch.cat([aa_pred, ncaa_pred], dim=2) 
                        # print(f"        pred dimension: {pred.shape}")# ❤️

                        ncaa_true = true_coordinates[:,ncaa_idx,:,:]
                        mask = torch.ones(true_coordinates.shape[1], dtype=torch.bool)
                        mask[ncaa_idx] = False
                        aa_true = true_coordinates[:, mask, :, :]
                        # print(f"        aa_true: {aa_true.shape}, ncaa_true: {ncaa_true.shape}")

                        assert len(aa_true.shape) == len(ncaa_true.shape), "Shapes of aa_true and ncaa_true do not match"
                        true = torch.cat([aa_true, ncaa_true],dim=1) 
                        # print(f"        true dimension: {true.shape}")# ❤️
                        L = bond_feat.shape[0]
                        same_chain = torch.zeros((L,L))
                        G = nx.from_numpy_array(bond_feat.detach().cpu().numpy())
                        for idx in nx.connected_components(G):
                            idx = list(idx)
                            for i in idx:
                                same_chain[i,idx] = 1
                        same_chain = same_chain.unsqueeze(0)
                        # print(f"    same_chain dimension: {same_chain.shape}")# ❤️

                        mask_2d = torch.ones_like(same_chain)
                        # print(f"    mask_2d dimension: {mask_2d.shape}")# ❤️

                        loss_test= loss_rfaa(pred, true, mask_2d, same_chain)
                        print(f"    {name}: {loss_test:.5f}")
                        loss_epoch_test += loss_test
                        n_sample_epoch_test += 1

                    except Exception as e:
                        print(f"    {name} error: {e}")
                        if epoch == 0:
                            exception_id_test.append(name)
                if epoch == 0:
                    with open('/home/light/mqy/ncaa/notebooks/tmp/test_id_length_larger_than_300_20241104.txt', 'w') as f:
                        for i in pass_id_test:
                            f.write(f"{i}\n")
                    with open('/home/light/mqy/ncaa/notebooks/tmp/test_id_exception_20241104.txt', 'w') as f:
                        for i in exception_id_test:
                            f.write(f"{i}\n")

                loss_epoch_test /= n_sample_epoch_test
                logger.info(f'{current_time} - Epoch: {epoch + 1:04d} - test_loss = {loss_epoch_test.item():.6f}')
                each_epoch_info_test = f"{current_time} - Epoch: {epoch + 1:04d} - test_loss = {loss_epoch_test.item():.6f}"
                with open(f'/home/light/mqy/ncaa/notebooks/tmp/loss_txt/FAPEloss_{runner_version}_test.txt', 'a') as f: #👿改
                    f.write(each_epoch_info_test + '\n')
                

    
    def pep_length(self):  # 🌈🌈🌈
        pep_path = Path(self.config.protein_inputs['A']['fasta_file'])
        with open(pep_path, 'r') as f:
            record = list(SeqIO.parse(f, 'fasta'))
        seq_len = len(record[0].seq)
        return seq_len

    def lddt_unbin(self, pred_lddt):
        # calculate lddt prediction loss
        nbin = pred_lddt.shape[1]
        bin_step = 1.0 / nbin
        lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=pred_lddt.dtype, device=pred_lddt.device)

        pred_lddt = nn.Softmax(dim=1)(pred_lddt)
        return torch.sum(lddt_bins[None,:,None]*pred_lddt, dim=1)

    def pae_unbin(self, logits_pae, bin_step=0.5):
        nbin = logits_pae.shape[1]
        bins = torch.linspace(bin_step*0.5, bin_step*nbin-bin_step*0.5, nbin,
                                dtype=logits_pae.dtype, device=logits_pae.device)
        logits_pae = torch.nn.Softmax(dim=1)(logits_pae)
        return torch.sum(bins[None,:,None,None]*logits_pae, dim=1)

    def pde_unbin(self, logits_pde, bin_step=0.3):
        nbin = logits_pde.shape[1]
        bins = torch.linspace(bin_step*0.5, bin_step*nbin-bin_step*0.5, nbin,
                                dtype=logits_pde.dtype, device=logits_pde.device)
        logits_pde = torch.nn.Softmax(dim=1)(logits_pde)
        return torch.sum(bins[None,:,None,None]*logits_pde, dim=1)

    def calc_pred_err(self, pred_lddts, logit_pae, logit_pde, seq):
        """Calculates summary metrics on predicted lDDT and distance errors"""
        plddts = self.lddt_unbin(pred_lddts)
        pae = self.pae_unbin(logit_pae) if logit_pae is not None else None
        pde = self.pde_unbin(logit_pde) if logit_pde is not None else None
        sm_mask = is_atom(seq)[0]
        sm_mask_2d = sm_mask[None,:]*sm_mask[:,None]
        prot_mask_2d = (~sm_mask[None,:])*(~sm_mask[:,None])
        inter_mask_2d = sm_mask[None,:]*(~sm_mask[:,None]) + (~sm_mask[None,:])*sm_mask[:,None]
        # assumes B=1
        err_dict = dict(
            plddts = plddts.cpu(),
            pae = pae.cpu(),
            pde = pde.cpu(),
            mean_plddt = float(plddts.mean()),
            mean_pae = float(pae.mean()) if pae is not None else None,
            pae_prot = float(pae[0,prot_mask_2d].mean()) if pae is not None else None,
            pae_inter = float(pae[0,inter_mask_2d].mean()) if pae is not None else None,
        )
        return err_dict


def load_status(status_path):
    if status_path.exists():
        with status_path.open("r") as f:
            return json.load(f)
    return {}

def save_status(status_path, status_info):
    with status_path.open("w") as f:
        json.dump(status_info, f, indent=4)


@hydra.main(version_base=None, config_path='/home/light/mqy/ncaa/RoseTTAFold-All-Atom/rf2aa/config/inference/', config_name='base_predict.yaml')
def main(config):
    ConfigPro.save_config(config)
    runner = ModelRunner(config)
    runner.infer_prediction()


if __name__ == "__main__":
    main()