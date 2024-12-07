from pathlib import Path
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Entity import Entity
from Bio.PDB import PDBParser, Select, PDBIO
import re
import warnings
from loguru import logger

warnings.filterwarnings("ignore")

standard_amino_acids = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]


def is_aa(aa_str):
    if aa_str in standard_amino_acids:
        return True
    else:
        return False


class NonHydrogenSelect(Select):
    """只选择非氢原子的类，使用正则表达式匹配氢原子"""

    def accept_atom(self, atom):
        # 匹配 'H' 或以数字开头后跟 'H' 的原子名称
        flag = True
        if re.match(r"^\d*H", atom.get_id()):
            flag = False
        if re.match("OXT", atom.get_id()):
            flag = False
        return flag


class NoWatersSelect(Select):
    """选择器类，用于排除水分子。"""

    def accept_residue(self, residue):
        # 只接受非水分子的残基
        return residue.get_resname() not in ["HOH", "WAT"]


def remove_water(structure):
    # 遍历每个模型、每个链、每个残基，收集水分子的位置
    water_residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in ["HOH", "WAT"]:
                    water_residues.append((model.id, chain.id, residue.id))
    # 在结构中移除这些水分子
    for model_id, chain_id, res_id in water_residues:
        model = structure[model_id]
        chain = model[chain_id]
        del chain[res_id]
    logger.debug(f"Water molecules have been removed from the {str(structure)} in memory.")

def remove_ion(structure):
    # 遍历每个模型、每个链、每个残基，收集水分子的位置
    water_residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in ["ZN", "CD", "MG", "SO4", "NI", "CL", "CA", "PO4", "NA", "UNX","FAR", "ACT", "ANS", "EDO", "GOL", "K", "TBF", "NME", "BFC", "SIN", "ACT", "K", "NAG"]:
                    water_residues.append((model.id, chain.id, residue.id))
    # 在结构中移除这些水分子
    for model_id, chain_id, res_id in water_residues:
        model = structure[model_id]
        chain = model[chain_id]
        del chain[res_id]
    logger.debug(f"Zn molecules have been removed from the {str(structure)} in memory.")

def remove_dna(structure):
    # 遍历每个模型、每个链、每个残基，收集水分子的位置
    water_residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in ["A", "U", "C", "G", "T"]:
                    water_residues.append((model.id, chain.id, residue.id))
    # 在结构中移除这些水分子
    for model_id, chain_id, res_id in water_residues:
        model = structure[model_id]
        chain = model[chain_id]
        del chain[res_id]
    logger.debug(f"Zn molecules have been removed from the {str(structure)} in memory.")

def get_pos(chain, ncaa):
    print(chain)

