import torch

from rf2aa.data.data_loader import RawInputData
from rf2aa.data.data_loader_utils import blank_template, TemplFeaturize
from rf2aa.data.parsers import parse_a3m, parse_templates_raw
from rf2aa.data.preprocessing import make_msa
from rf2aa.util import get_protein_bond_feats


def get_templates(
    qlen,
    ffdb,
    hhr_fn,
    atab_fn,
    seqID_cut,
    n_templ,
    pick_top: bool = True,
    offset: int = 0,
    random_noise: float = 5.0,
    deterministic: bool = False,
    chain = None
):
    (
        xyz_parsed,
        mask_parsed,
        qmap_parsed,
        f0d_parsed,
        f1d_parsed,
        seq_parsed,
        ids_parsed,
        score_parsed  # 🌈
    ) = parse_templates_raw(ffdb, hhr_fn=hhr_fn, atab_fn=atab_fn)
    # print(score_parsed)
    tplt = {
        "xyz": xyz_parsed.unsqueeze(0),
        "mask": mask_parsed.unsqueeze(0),
        "qmap": qmap_parsed.unsqueeze(0),
        "f0d": f0d_parsed.unsqueeze(0),
        "f1d": f1d_parsed.unsqueeze(0),
        "seq": seq_parsed.unsqueeze(0),
        "ids": ids_parsed,
        "score": score_parsed  # 🌈
    }
    params = {
        "SEQID": seqID_cut,
    }
    return TemplFeaturize(
        tplt,
        qlen,
        params,
        offset=offset,
        npick=n_templ,
        pick_top=pick_top,
        random_noise=random_noise,
        deterministic=deterministic,
        chain = chain
    )


def load_protein(msa_file, hhr_fn, atab_fn, model_runner, nc_cycle = False, chain = None):
    msa, ins, taxIDs = parse_a3m(msa_file)
    # NOTE: this next line is a bug, but is the way that
    # the code is written in the original implementation!
    ins[0] = msa[0]

    L = msa.shape[1]
    if hhr_fn is None or atab_fn is None:
    # if True:  # 若🌈
        print("No templates provided")
        xyz_t, t1d, mask_t, _ = blank_template(1, L)
    else:
        xyz_t, t1d, mask_t, id = get_templates(    # 🌈 id
            L,
            model_runner.ffdb,
            hhr_fn,
            atab_fn,
            seqID_cut=model_runner.config.loader_params.seqid,
            n_templ=model_runner.config.loader_params.n_templ,
            deterministic=model_runner.deterministic,
            chain = chain
        )

    bond_feats = get_protein_bond_feats(L, nc_cycle = nc_cycle )  # 🌈
    chirals = torch.zeros(0, 5)
    atom_frames = torch.zeros(0, 3, 2)
    return RawInputData(
        torch.from_numpy(msa),
        torch.from_numpy(ins),
        bond_feats,
        xyz_t,
        mask_t,
        t1d,
        chirals,
        atom_frames,
        taxids=taxIDs,
    )

def generate_msa_and_load_protein(fasta_file, chain, model_runner, nc_cycle = False):
    msa_file, hhr_file, atab_file = make_msa(fasta_file, chain, model_runner)
    return load_protein(str(msa_file), str(hhr_file), str(atab_file), model_runner, nc_cycle = nc_cycle, chain = chain)
