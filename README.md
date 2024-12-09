# NCPepFold

This is the official implementation for the paper titled 'NCPepFold: Accurate Prediction of Non-canonical Cyclic Peptide Structures via Cyclization Optimization with Multigranular Representation'.

![NCPepFold](./model/PNG/NCPepFold.png)

## Setups

1. Install Mamba

```
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh  # accept all terms and install to the default location
rm Mambaforge-$(uname)-$(uname -m).sh  # (optionally) remove installer after using it
source ~/.bashrc  # alternatively, one can restart their shell session to achieve the same result
```

2. Clone the package

```
git clone https://github.com/mqyii/NCPepFold.git
cd NCPepFold
```

3. Create Mamba environment

```
mamba env create -f environment.yaml
conda activate NCPepFold
cd model/rf2aa/SE3Transformer/
pip3 install --no-cache-dir -r requirements.txt
python3 setup.py install
cd ../../
```

4. Configure signalp6 after downloading a licensed copy of it from https://services.healthtech.dtu.dk/services/SignalP-6.0/

```
signalp6-register signalp-6.0h.fast.tar.gz
mv $CONDA_PREFIX/lib/python3.10/site-packages/signalp/model_weights/distilled_model_signalp6.pt $CONDA_PREFIX/lib/python3.10/site-packages/signalp/model_weights/ensemble_model_signalp6.pt
```

5. Install input preparation dependencies

```
bash install_dependencies.sh
```

6. Download the model weights.

7. Download sequence databases for MSA and template generation.

```
# uniref30 [46G]
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz
mkdir -p UniRef30_2020_06
tar xfz UniRef30_2020_06_hhsuite.tar.gz -C ./UniRef30_2020_06

# BFD [272G]
wget https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz
mkdir -p bfd
tar xfz bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz -C ./bfd

# structure templates [81G] (including *_a3m.ffdata, *_a3m.ffindex)
wget https://files.ipd.uw.edu/pub/RoseTTAFold/pdb100_2021Mar03.tar.gz
tar xfz pdb100_2021Mar03.tar.gz
```

8. Download BLAST [39M]

```
wget https://ftp.ncbi.nlm.nih.gov/blast/executables/legacy.NOTSUPPORTED/2.2.26/blast-2.2.26-x64-linux.tar.gz
mkdir -p blast-2.2.26
tar -xf blast-2.2.26-x64-linux.tar.gz -C blast-2.2.26
cp -r blast-2.2.26/blast-2.2.26/ blast-2.2.26_bk
rm -r blast-2.2.26
mv blast-2.2.26_bk/ blast-2.2.26
```

## Experiments

1. Before starting the training, please use NCPepFold to perform multiple sequence alignment and store the alignment results in the corresponding folder for each sequence.

2. Training for NCPepFold

```
python run_inference_train_test_pkl_attn.py
```

3. Prediction for NCPepFold

```
python run_inference_train_test_pkl_attn_prediction.py
```

4. Result analysis

The result analysis of NCPepFold are demonstrated in the `notebooks` and `tools` directory:

​	1. `ca_rmsd.ipynb`: Calculate peptide RMSD

​	2. `modification rmsd.ipynb`: Calculate modification RMSD

​	3. `pep_plddt.ipynb`:  Calculate peptide pLDDT

​	4. `sequence_similarity.ipynb`: Calculate sequence similarity

​	5. `tools/DockQ`: Calculate DockQ and Fnat