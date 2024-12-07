
import torch

class SingletonState:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SingletonState, cls).__new__(cls)
            cls._instance.computed_value = None
        return cls._instance

shared_state = SingletonState()


def split_and_concat_xyz_pred(xyz, aa_num, ncaa_atom_num):  #xyz是pred [40,1,n,3,3],aa_num是标准氨基酸个数，ncaa_idx是非标准氨基酸索引,ncaa_atom_num是非标准氨基酸的原子列表
    standard_amino_acids = xyz[:, :, :aa_num, :, :]

    to_concat = [standard_amino_acids]

    start_idx = aa_num
    for count in ncaa_atom_num:
        end_idx = start_idx + count
        non_standard_part = xyz[:, :, start_idx:end_idx, :, :]
        non_standard_first_three = non_standard_part[:, :, :3, 1, :].unsqueeze(2)
        to_concat.append(non_standard_first_three)
        start_idx = end_idx

    concatenated_tensor = torch.cat(to_concat, dim=2)
    return concatenated_tensor