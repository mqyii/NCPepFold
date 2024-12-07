import torch

def block_diag_d1d2(aa_pair_feats, ncaa_pair_feats, batch_size=1, d_model=192, device = None):
    """
    :param aa_pair_feats: [batch_size, n_res, n_res, d_model]
    :param ncaa_pair_feats: [batch_size, n_atom, n_atom, d_model]
    :param batch_size:
    :param d_model:
    :return:
    """
    pair_feats_1 = torch.cat(
        (aa_pair_feats.to(device), torch.zeros(batch_size, ncaa_pair_feats.shape[1], aa_pair_feats.shape[2], d_model).to(device)), dim=1)
    pair_feats_2 = torch.cat(
        (torch.zeros(batch_size, aa_pair_feats.shape[1], ncaa_pair_feats.shape[2], d_model).to(device), ncaa_pair_feats.to(device)), dim=1)
    pair_feats = torch.cat((pair_feats_1, pair_feats_2), dim=2)
    return pair_feats


def mapping_atom2aa(enc_outputs, n_atom_each_aa, idx_ncaa, device):
    """
    enc_outputs:  [batch_size, src_len, d_model]
    n_atom_each_aa:  [9,8,6,10,11,5,6,6,...] 肽链中每一个残基的原子总数
    idx_ncaa:  [1,3] or []
    """
    index = 0
    n_atom_each_aa = [k for k in n_atom_each_aa if k > 0]
    # print(f"n_atom_each_aa: {n_atom_each_aa}")
    # print(f"ncaa_index: {idx_ncaa}")
    pair_feats_i = mapping_atom2aa_i(enc_outputs[index, ...].unsqueeze(0), n_atom_each_aa, idx_ncaa, device)
    return pair_feats_i


def mapping_atom2aa_i(enc_outputs, n_atom_each_aa, idx_ncaa, device):
    # enc_outputs   : [batch_size, src_len, d_model]
    # n_atom_each_aa = [4, 5, 8, 6]
    # idx_ncaa = [1, 3]

    # dec_enc_attn  : [n_layers, batch_size, tgt_len, src_len]   
    # pair          : [batch_size, 18, 18, 192]
       
    batch_size, src_len, d_model = enc_outputs.shape
    aa_feats_list = []
    ncaa_feats_list = []
    for i in range(len(n_atom_each_aa)):
        if i in idx_ncaa:
            ncaa_feats_i = enc_outputs[:, sum(n_atom_each_aa[:i]):sum(n_atom_each_aa[:i + 1]), :]
            ncaa_feats_list.append(ncaa_feats_i)
        else:
            aa_feats_i = enc_outputs[:, sum(n_atom_each_aa[:i]):sum(n_atom_each_aa[:i + 1]), :]
            aa_feats_list.append(torch.sum(aa_feats_i, dim=1))
    aa_feats = torch.stack(aa_feats_list, dim=1)
    # ncaa_feats = torch.cat(ncaa_feats_list, dim=1)
    # print(n_atom_each_aa)
    # print(idx_ncaa)
    # print(aa_feats.shape)
    # print(ncaa_feats.shape)

    pair_feats = torch.einsum('ijk,ilk->ijlk', [aa_feats, aa_feats])
    # print(pair_feats.shape)
    for ncaa_feats in ncaa_feats_list:
        ncaa_pair_feats = torch.einsum('ijk,ilk->ijlk', [ncaa_feats, ncaa_feats])
        # print(ncaa_pair_feats.shape)
        # pair_feats = torch.block_diag(pair_feats, ncaa_pair_feats)
        pair_feats = block_diag_d1d2(pair_feats, ncaa_pair_feats, batch_size, d_model, device)

    return pair_feats


# def mapping_atom2aa_multimer(attn_feats, flag_complex, all_chain_lens, ligand_position, n_atom_each_aa, idx_ncaa):
#     idx_batch = 0
#     if flag_complex == 0:
#         return attn_feats

#     attn_feats_complex_list = []
#     for i in range(ligand_position.item()):
#         protein_size = all_chain_lens[idx_batch][i].item()
#         matrix_protein = torch.zeros((1, protein_size, protein_size, attn_feats.shape[-1]))
#         attn_feats_complex_list.append(matrix_protein) #0
#     n_aa_ligand = all_chain_lens[idx_batch][ligand_position.item()].item() - torch.sum(idx_ncaa).item()
#     attn_feats_complex_list.append(attn_feats[:, :n_aa_ligand, :n_aa_ligand, :]) #1
#     for i in range(ligand_position.item()+1, len(all_chain_lens[idx_batch].tolist()), 1):
#         if all_chain_lens[idx_batch, i].item() == 0:
#             break
#         protein_size = all_chain_lens[idx_batch][i].item()
#         matrix_protein = torch.zeros((1, protein_size, protein_size, attn_feats.shape[-1]))
#         attn_feats_complex_list.append(matrix_protein) #2
#     attn_feats_complex_list.append(attn_feats[:, n_aa_ligand:, n_aa_ligand:, :]) #3
#     #
#     attn_feats_complex = block_diag_d1d2(attn_feats_complex_list[0], attn_feats_complex_list[1], attn_feats.shape[0], attn_feats.shape[-1])
#     for i in range(2, len(attn_feats_complex_list), 1):
#         attn_feats_complex = block_diag_d1d2(attn_feats_complex, attn_feats_complex_list[i], attn_feats.shape[0], attn_feats.shape[-1])

#     return attn_feats_complex

def mapping_atom2aa_multimer(attn_feats, flag_complex, pdb_size, ncaa_atom_num):
    """
    attn_feats: [1, pep_aa_num+pep_ncaa_num, pep_aa_num+pep_ncaa_num, 192]
    flag_complex: 1 is complex, and 0 is monomer
    pdb_size: [pep_aa_num, prot-1_aa_num, prot-2_aa_num, ...]
    ncaa_atom_num: ex.[6,9]
    """
    if flag_complex == 0:
        return attn_feats
    
    pep_aa_num = pdb_size[0]
    prot_aa_num = sum(pdb_size[1:])
    # print(f"ncaa_atom_num: {ncaa_atom_num}")
    pep_ncaa_num = sum(ncaa_atom_num)

    batch_size = 1
    matrix_zero = torch.zeros((batch_size, pep_aa_num+prot_aa_num+pep_ncaa_num, pep_aa_num+prot_aa_num+pep_ncaa_num, attn_feats.shape[-1]))
    # print(f"matrix_zero.shape: {matrix_zero.shape}")

    attn_feats_aa_res = attn_feats[:, :pep_aa_num, :pep_aa_num, :]
    # print(f"attn_feats_aa_res: {attn_feats_aa_res.shape}")
    matrix_zero[:, :pep_aa_num, :pep_aa_num, :] = attn_feats_aa_res

    if pep_ncaa_num == 0:
        matrix = matrix_zero
    else:
        attn_feats_ncaa_atom = attn_feats[:, pep_aa_num:, pep_aa_num:, :]
        # print(f"attn_feats_ncaa_atom: {attn_feats_ncaa_atom.shape}")
        matrix_zero[:, -pep_ncaa_num:, -pep_ncaa_num:, :] = attn_feats_ncaa_atom
        matrix = matrix_zero
    return matrix