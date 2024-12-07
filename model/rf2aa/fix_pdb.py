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
import os
from copy import deepcopy

warnings.filterwarnings("ignore")

ncaa_dict = {
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
    "ACE???": [],
    "NH2???": [],
    "M3L": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'CM1', 'CM2', 'CM3'],
    "MLY": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'CH1', 'CH2'],
    "DA2": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH2', 'NH1', 'C1', 'C2'],
    "2MR": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'CQ1', 'NH2', 'CQ2'],
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
    "ORN1": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE'],
    "CY3": ['N', 'CA', 'C', 'O', 'CB', 'SG', 'N1'],
    "2GX": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'CAO', 'CAM', 'CAI', 'CAE', 'CAH', 'CAL']
    }

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

class NonHydrogenSelect(Select):
    def accept_atom(self, atom):
        flag = True
        if re.match(r"^\d*H", atom.get_id()):
            flag = False
        if re.match("OXT", atom.get_id()):
            flag = False
        return flag


class NoWatersSelect(Select):
    def accept_residue(self, residue):
        return residue.get_resname() not in ["HOH", "WAT"]


def remove_water(structure):
    water_residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in ["HOH", "WAT"]:
                    water_residues.append((model.id, chain.id, residue.id))
    for model_id, chain_id, res_id in water_residues:
        model = structure[model_id]
        chain = model[chain_id]
        del chain[res_id]
    print(f"Water molecules have been removed from the {str(structure)} in memory.")

def save_hetatm_residue(chain):
    residue_name = []
    pos = []
    residue = []
    for i, _residue in enumerate(chain):
        if _residue.id[0][0] == 'H':
            residue_name.append(_residue.resname)
            residue.append(_residue)
            pos.append(i)
    return residue_name, residue, pos

def mirrow_atom(atom: Atom):
    _atom = Atom(
        name=atom.get_name(),
        coord=None,
        bfactor=atom.get_bfactor(),
        occupancy=atom.get_occupancy(),
        altloc=atom.get_altloc(),
        fullname=atom.get_fullname(),
        serial_number=atom.get_serial_number(),
    )
    _atom.set_coord(atom.get_coord())
    return _atom

def mirror_residue(res: Residue, res_id):
    _res = Residue(res_id, res.resname, " ")
    for atom in res:
        _res.add(mirrow_atom(atom))
    return _res

def process_pdb_file(input_file, output_file):
    c_count = 0

    with open(input_file, 'r') as file:
        lines = file.readlines()
    atom_set = set()
    processed_lines = []
    for line in lines:
        if line.startswith("HETATM") and "LG" in line:
            parts = line.split()
            if parts[2] in atom_set:
                parts[2] = parts[2] + str(c_count)
                c_count += 1
            else:
                atom_set.add(parts[2])
            line = "{:<6}{:>5} {:>4} {:<3} {:<1} {:>3}     {:>7} {:>7} {:>7} {:>5} {:>5}           {:<1}\n".format(
                parts[0], parts[1], parts[2], parts[3], parts[4], parts[5],
                parts[6], parts[7], parts[8], parts[9], parts[10], parts[11]
            )
        processed_lines.append(line)

    with open(output_file, 'w') as file:
        file.writelines(processed_lines)

def modify_residue_identifiers_and_pdb_chains(input_file, output_file):
    chain_residue_map = {}
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    with open(output_file, 'w') as f:
        for line in lines:
            if line.startswith("HETATM") and "LG" in line:
                chain_id = line[21]
                if chain_id not in chain_residue_map:
                    chain_residue_map[chain_id] = f"LG{len(chain_residue_map) + 1}"
                new_residue_id = chain_residue_map[chain_id]
                line = line[:17] + new_residue_id + line[20:]
                line = line[:21] + 'A' + line[22:]
            f.write(line)


def fix(name, clean_pdb, predict_pdb, output_path):
    clean_pdb = Path(clean_pdb)
    predict_pdb = Path(predict_pdb)
    output_path = Path(output_path)
    
    pdb_tmp = output_path / f"{name}_tmp.pdb"
    pdb_output = output_path / f"{name}.fixed.pdb"
    process_pdb_file(predict_pdb, pdb_tmp)
    modify_residue_identifiers_and_pdb_chains(pdb_tmp, pdb_tmp)

    chain_refer_name = "A"
    structure_clean = PDBParser().get_structure("clean_pdb", clean_pdb)
    structure_refer = PDBParser(PERMISSIVE=False).get_structure("refer", pdb_tmp)
    chain_dict = {}
    for chain in structure_clean[0]:
        residue_count = len(list(chain.get_residues()))
        chain_dict[chain.id] = residue_count
    pep_chain_id = min(chain_dict,key=chain_dict.get)
    chain_true_name = pep_chain_id

    chain_true = structure_clean[0][chain_true_name]
    chain_refer = structure_refer[0][chain_refer_name]
    residue_name_true, residue_true, pos_true = save_hetatm_residue(chain_true)
    _, residue_refer, _ = save_hetatm_residue(chain_refer)

    assert len(chain_true) == len(chain_refer), f"The length of two chains are not equal."

    structure = Structure(name)
    model = Model(0)
    structure.add(model)

    for chain in structure_refer[0]:
        if chain.id == "A":
            chain_tmp = Chain("A")
            segid = " "

            i = len(chain)
            j = 0
            for residue in chain.get_residues():
                if i in [pos + len(chain) for pos in pos_true]:
                    residue_id = (f"H_{residue_name_true[j]}", i, " ")
                    _residue = Residue(residue_id, residue_name_true[j], segid)
                    residue_name = residue_true[j].get_resname()
                    atom_dict = ncaa_dict[residue_name]
                    count = 0
                    for x, y in zip(ncaa_dict[residue_name], residue_refer[j]):
                        y.name = x

                    #按照true对齐
                    residue_true_sorted = residue_true[j]
                    residue_refer_sorted = sorted(residue_refer[j], key=lambda x: x.get_name())
                    name_to_position = {res.get_name(): i for i, res in enumerate(residue_true_sorted)}
                    residue_refer_sorted = sorted(residue_refer_sorted, key=lambda x: name_to_position[x.get_name()])
                    
                    for atom_true, atom_refer in zip(residue_true_sorted, residue_refer_sorted):
                        _atom = mirrow_atom(atom_true)
                        _atom.set_coord(atom_refer.get_coord())
                        _residue.add(_atom)
                        count += 1
                        
                    chain_tmp.add(deepcopy(_residue))
                    i += 1
                    chain_tmp.add(mirror_residue(residue, res_id=(" ", i, " ")))
                    i += 1
                    j += 1
                else:
                    if residue.resname in standard_amino_acids:
                        chain_tmp.add(mirror_residue(residue, res_id=(" ", i, " ")))
                        i += 1
                    else:
                        pass
            model.add(chain_tmp)
        else:
            model.add(chain)

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_output), NonHydrogenSelect())
    pdb_tmp.unlink()
