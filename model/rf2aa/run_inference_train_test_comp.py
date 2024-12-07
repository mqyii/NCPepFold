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
from rf2aa.data_loader_train import data_loader
from rf2aa.util_train import shared_state, split_and_concat_xyz_pred


from util import ConfigPro
import traceback
import json
import logging
import networkx as nx



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
        # self.seq_len = self.pep_length()


    def parse_inference_config(self):
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
            # seq_len = self.seq_len

        ).to(self.device)
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        # checkpoint = torch.load("/home/light/mqy/ncaa/notebooks/tmp/ckpt/rfaa_20241012154108_90.pt", map_location=self.device)
        # print(checkpoint.keys())
        # exit()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.model.load_state_dict(checkpoint)

    def construct_features(self):
        return self.raw_data.construct_features(self)

    def run_model_forward(self, input_feats):
        input_feats.add_batch_dim()
        input_feats.to(self.device)
        input_dict = asdict(input_feats)
        input_dict["bond_feats"] = input_dict["bond_feats"].long()
        input_dict["seq_unmasked"] = input_dict["seq_unmasked"].long()
        outputs = recycle_step_legacy(self.model, 
                                     input_dict, 
                                     self.config.loader_params.MAXCYCLE, 
                                     use_amp=False,
                                     nograds=False, ##################################################
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
        writepdb(os.path.join(f"{self.config.output_path}", f"predict_{self.config.job_name}_ckpt95_ss.pdb"), 
                 xyz_allatom, 
                 seq_unmasked, 
                 bond_feats=bond_feats,
                 bfacts=plddts,
                 chain_Ls=Ls
                 )
        torch.save(err_dict, os.path.join(f"{self.config.output_path}", 
                                          f"{self.config.job_name}_aux_ckpt95_ss.pt"))
    def fix_outputs(self):
        name = self.config.job_name
        clean_pdb = os.path.join(f"{self.config.output_path}", f"{self.config.job_name}_clean.pdb")
        predict_pdb = os.path.join(f"{self.config.output_path}", f"predict_{self.config.job_name}_ckpt95_ss.pdb")
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
    


    def finetuning_infer(self):
        self.load_model()
        input_feats_all = {}
        bond_feat_dict = {}
        for subdir in Path(self.config.input_cfg).iterdir():
            print(subdir.name)
            file_path = subdir / "config_aa.yaml"
            cfg = OmegaConf.load(file_path)
            self.config = cfg
            self.parse_inference_config()
            # input_feats_all[subdir.name[:4].lower()] = self.construct_features()
            input_feats_all[subdir.name.lower()] = self.construct_features()
            bond_feat = shared_state.computed_value
            # bond_feat_dict[subdir.name[:4].lower()] = bond_feat
            bond_feat_dict[subdir.name.lower()] = bond_feat
        # a = [(key,value.shape) for key, value in bond_feat_dict.items()]
        # print(f"bond_feat_dict: {a}")
        print(input_feats_all.keys())

        ###############################################################
        input_feats_all_test = {}
        bond_feat_dict_test = {}
        cfg = OmegaConf.load("/home/light/mqy/ncaa/RoseTTAFold-All-Atom/rf2aa/config/inference/base.yaml")
        self.config = cfg
        for subdir in Path(self.config.input_cfg_test).iterdir():
            # print(subdir.name)
            file_path = subdir / "config_aa.yaml"
            cfg = OmegaConf.load(file_path)
            self.config = cfg
            self.parse_inference_config()
            # input_feats_all[subdir.name[:4].lower()] = self.construct_features()
            input_feats_all_test[subdir.name.lower()] = self.construct_features()
            bond_feat = shared_state.computed_value
            # bond_feat_dict[subdir.name[:4].lower()] = bond_feat
            bond_feat_dict_test[subdir.name.lower()] = bond_feat
        # a = [(key,value.shape) for key, value in bond_feat_dict.items()]
        # print(f"bond_feat_dict: {a}")
        print(input_feats_all_test.keys())
        
        loss_rfaa = RFAALoss()
        learning_rates = 1e-5

        with open(f'/home/light/mqy/ncaa/notebooks/tmp/loss_txt/FAPEloss_{runner_version}_ss_comp.txt', 'a') as f:
            f.write(f"learning_rate: {float(learning_rates)}\n")

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rates)
        weight_sample_in_loss = 1.0
        loader = data_loader('/home/light/mqy/ncaa/notebooks/tmp/train_ss_comp.txt', batch_size=1)
        loader_test = data_loader('/home/light/mqy/ncaa/notebooks/tmp/train_test.txt', batch_size=1)

        loss_list = []
        current_time_list = []
        # logger.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'lr_rfaa = %.6f, weight_sample_in_loss = %.2f' % (learning_rates, weight_sample_in_loss))
        # self.model.train()
        for epoch in range(400):
            self.model.train()
            print(f"Epoch: {epoch+1}")
            loss_epoch = 0.0
            n_sample_epoch = 0
            for feats in loader:
                print(len(feats))
                # name, true_coordinates, aa_num, ncaa_idx, ncaa_num, ncaa_atom_num, pep_len =  feats[0], feats[1], feats[2], feats[3], feats[4], feats[5], feats[6]
                name, true_coordinates, aa_num, ncaa_idx, ncaa_num, ncaa_atom_num, pep_len, pep_chain,  protein_chain =  feats[0], feats[1], feats[2], feats[3], feats[4], feats[5], feats[6], feats[7], feats[8]
                true_coordinates = true_coordinates.reshape(-1, 3, 3).unsqueeze(0).to('cuda:0')
                print(f"    {name}")# ❤️
                # print(f"        true_coordinates shape: {true_coordinates.shape}")
                name = name[0]
                aa_num = aa_num.tolist()[0]
                ncaa_idx = ncaa_idx.tolist()[0]
                ncaa_num = ncaa_num.tolist()[0]
                ncaa_atom_num = ncaa_atom_num.tolist()[0]
                pep_len = pep_len.tolist()[0]
                pep_chain = pep_chain[0] # ❤️复合物多加的
                protein_chain = list(protein_chain[0]) # ❤️复合物多加的
                # print(f"        name: {name}, aa_num: {aa_num}, ncaa_idx: {ncaa_idx}, ncaa_num: {ncaa_num}, ncaa_atom_num: {ncaa_atom_num}, pep_len: {pep_len}")

                if name not in input_feats_all.keys():
                    print(f"{name} is not exist in input_feats_all")
                    continue

                if epoch > 0:
                    input_feats_all[name].msa_latent = input_feats_all[name].msa_latent.squeeze(0)
                    input_feats_all[name].msa_full = input_feats_all[name].msa_full.squeeze(0)
                    input_feats_all[name].seq = input_feats_all[name].seq.squeeze(0)
                    input_feats_all[name].seq_unmasked = input_feats_all[name].seq_unmasked.squeeze(0)
                    input_feats_all[name].idx = input_feats_all[name].idx.squeeze(0)
                    input_feats_all[name].bond_feats = input_feats_all[name].bond_feats.squeeze(0)
                    input_feats_all[name].dist_matrix = input_feats_all[name].dist_matrix.squeeze(0)
                    input_feats_all[name].chirals = input_feats_all[name].chirals.squeeze(0)
                    input_feats_all[name].atom_frames = input_feats_all[name].atom_frames.squeeze(0)
                    input_feats_all[name].xyz_prev = input_feats_all[name].xyz_prev.squeeze(0)
                    input_feats_all[name].alpha_prev = input_feats_all[name].alpha_prev.squeeze(0)
                    input_feats_all[name].t1d = input_feats_all[name].t1d.squeeze(0)
                    input_feats_all[name].t2d = input_feats_all[name].t2d.squeeze(0)
                    input_feats_all[name].xyz_t = input_feats_all[name].xyz_t.squeeze(0)
                    input_feats_all[name].alpha_t = input_feats_all[name].alpha_t.squeeze(0)
                    input_feats_all[name].mask_t = input_feats_all[name].mask_t.squeeze(0)
                    input_feats_all[name].same_chain = input_feats_all[name].same_chain.squeeze(0)

                outputs = self.run_model_forward(input_feats_all[name.lower()])
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
                bond_feat = bond_feat_dict[name]
                # print(f"    bond_feat: {bond_feat.shape}")# ❤️
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
                loss_epoch += loss
                n_sample_epoch += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            

            if (epoch+1) % 10 == 0:
                torch.save(self.model.state_dict(), '/home/light/mqy/ncaa/notebooks/tmp/ckpt/rfaa_' + runner_version + '_' + str(epoch + 1) + '.pt')

            loss_epoch /= n_sample_epoch
            loss_list.append(loss_epoch.item())

            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            current_time_list.append(current_time)
            logger.info(f'{current_time} - Epoch: {epoch + 1:04d} - loss = {loss_epoch.item():.6f}')
            each_epoch_info = f"{current_time} - Epoch: {epoch + 1:04d} - loss = {loss_epoch.item():.6f}"
            with open(f'/home/light/mqy/ncaa/notebooks/tmp/loss_txt/FAPEloss_{runner_version}_ss_ncaa_mono_new.txt', 'a') as f:
                f.write(each_epoch_info + '\n')
            # for handler in logger.handlers:
            #     if isinstance(handler, logging.FileHandler):
            #         print("Flushing log to file")
            #         handler.flush()

            ###################################################################
            
            self.model.eval()
            with torch.no_grad():
                loss_epoch_test = 0.0
                n_sample_epoch_test = 0
                for feats in loader_test:
                    name, true_coordinates, aa_num, ncaa_idx, ncaa_num, ncaa_atom_num, pep_len =  feats[0], feats[1], feats[2], feats[3], feats[4], feats[5], feats[6]
                    true_coordinates = true_coordinates.reshape(-1, 3, 3).unsqueeze(0).to('cuda:0')
                    # print(f"    {name}")# ❤️
                    # print(f"        true_coordinates shape: {true_coordinates.shape}")
                    name = name[0]
                    aa_num = aa_num.tolist()[0]
                    ncaa_idx = ncaa_idx.tolist()[0]
                    ncaa_num = ncaa_num.tolist()[0]
                    ncaa_atom_num = ncaa_atom_num.tolist()[0]
                    pep_len = pep_len.tolist()[0]
                    # print(f"        name: {name}, aa_num: {aa_num}, ncaa_idx: {ncaa_idx}, ncaa_num: {ncaa_num}, ncaa_atom_num: {ncaa_atom_num}, pep_len: {pep_len}")

                    if name not in input_feats_all_test.keys():
                        print(f"{name} is not exist in input_feats_all")
                        continue

                    if epoch > 0:
                        input_feats_all_test[name].msa_latent = input_feats_all_test[name].msa_latent.squeeze(0)
                        input_feats_all_test[name].msa_full = input_feats_all_test[name].msa_full.squeeze(0)
                        input_feats_all_test[name].seq = input_feats_all_test[name].seq.squeeze(0)
                        input_feats_all_test[name].seq_unmasked = input_feats_all_test[name].seq_unmasked.squeeze(0)
                        input_feats_all_test[name].idx = input_feats_all_test[name].idx.squeeze(0)
                        input_feats_all_test[name].bond_feats = input_feats_all_test[name].bond_feats.squeeze(0)
                        input_feats_all_test[name].dist_matrix = input_feats_all_test[name].dist_matrix.squeeze(0)
                        input_feats_all_test[name].chirals = input_feats_all_test[name].chirals.squeeze(0)
                        input_feats_all_test[name].atom_frames = input_feats_all_test[name].atom_frames.squeeze(0)
                        input_feats_all_test[name].xyz_prev = input_feats_all_test[name].xyz_prev.squeeze(0)
                        input_feats_all_test[name].alpha_prev = input_feats_all_test[name].alpha_prev.squeeze(0)
                        input_feats_all_test[name].t1d = input_feats_all_test[name].t1d.squeeze(0)
                        input_feats_all_test[name].t2d = input_feats_all_test[name].t2d.squeeze(0)
                        input_feats_all_test[name].xyz_t = input_feats_all_test[name].xyz_t.squeeze(0)
                        input_feats_all_test[name].alpha_t = input_feats_all_test[name].alpha_t.squeeze(0)
                        input_feats_all_test[name].mask_t = input_feats_all_test[name].mask_t.squeeze(0)
                        input_feats_all_test[name].same_chain = input_feats_all_test[name].same_chain.squeeze(0)

                    outputs = self.run_model_forward(input_feats_all_test[name.lower()])
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
                    bond_feat = bond_feat_dict_test[name]
                    # print(f"    bond_feat: {bond_feat.shape}")# ❤️
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

                    loss_test = loss_rfaa(pred, true, mask_2d, same_chain)
                    loss_epoch_test += loss_test
                    n_sample_epoch_test += 1

                loss_epoch_test /= n_sample_epoch_test
                logger.info(f'{current_time} - Epoch: {epoch + 1:04d} - test_loss = {loss_epoch_test.item():.6f}')
                each_epoch_info_test = f"{current_time} - Epoch: {epoch + 1:04d} - test_loss = {loss_epoch_test.item():.6f}"
                with open(f'/home/light/mqy/ncaa/notebooks/tmp/loss_txt/FAPEloss_{runner_version}_ss_ncaa_mono_new_test.txt', 'a') as f:
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


@hydra.main(version_base=None, config_path='/home/light/mqy/ncaa/RoseTTAFold-All-Atom/rf2aa/config/inference/', config_name='base.yaml')
def main(config):
    runner = ModelRunner(config)
    runner.finetuning_infer()

# @hydra.main(version_base=None, config_path='config/inference')
# def main(config):
#     ConfigPro.save_config(config)
#     # runner = ModelRunner(config)
#     # runner.infer()

#     status_path = Path(config.output_path) / "status.json"

#     try:
#         status_info = load_status(status_path)
#         if status_info.get("success"):
#             print(f"PDB was predicted suceessed, skip {config.job_name}")
#             exit()
#         runner = ModelRunner(config)
#         runner.infer()
#         # 读取现有状态信息
#         # 更新状态为成功
#         status_info["success"] = True
#         if status_info.get("error"):
#             status_info["error"] = ""
#             status_info["traceback"] = ""
#         # 保存状态信息
#         save_status(status_path, status_info)

#     except Exception as e:
#         print(traceback.format_exc())
#         print(e)
#         # 构建错误信息
#         error_info = {
#             "error": str(e),
#             "traceback": traceback.format_exc(),
#             "success": False
#         }
#         # 读取现有状态信息
#         status_info = load_status(status_path)
#         # 更新状态信息为错误信息
#         status_info.update(error_info)
#         # 保存状态信息
#         save_status(status_path, status_info)


if __name__ == "__main__":
    main()