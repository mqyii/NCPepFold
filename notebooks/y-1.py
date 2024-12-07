import os
import yaml
import glob
from pathlib import Path

def update_yaml_config(dir_name, subdir, config_file, pep_fasta, protein_fastas):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        if config is None:
                raise ValueError(f"Failed to load YAML from {config_file}")

    config['output_path'] = str(subdir)
    config['job_name'] = dir_name
    config['protein_inputs'] = {
        'A': {
            'fasta_file': str(pep_fasta)
        }
    }
    if protein_fastas:
        letters = "BCDEFGHI"
        for i, protein_fasta in enumerate(protein_fastas):
            config['protein_inputs'][letters[i]] = {
                'fasta_file': str(protein_fasta)
            }

    with open(config_file, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)

def main(base_dir, monomers=False):

    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            dir_name = subdir.name
            config_file = subdir / "config_aa.yaml"
            pep_fasta = subdir / f"{dir_name}-pep.fasta"
            protein_fastas = sorted(Path(subdir).glob(f"{dir_name}-protein-*.fasta"))
            update_yaml_config(dir_name, subdir, config_file, pep_fasta, protein_fastas)
            print(f"Updated {config_file}")


if __name__ == "__main__":
    base_dir = Path("/home/light/mqy/ncaa/data/ncaa/cyc/complex/ss")  # 🌈
    main(base_dir)
