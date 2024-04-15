##Overview
Folding-Docking-Affinity (FDA) is a framework which folds proteins, determines protein-ligand binding conformations, and predicts binding affinities from computed three-dimensional protein-ligand binding structures.
<p align="center">
    <img src="figure/FDA_fig1.pdf">
##Dependencies
The Folding part was tested with Python 3.10.13 and CUDA 12.3 on Ubuntu 20.04, with access to Nvidia Tesla V100 (32GB RAM), Intel(R) Xeon(R) Platinum 8168 CPU @ 2.70GHz, and 1.5TB RAM. Please follow [localcolabfold](https://github.com/YoshitakaMo/localcolabfold) to install the working environment. 

The Docking and Affinity parts were tested with Python 3.9.18 and CUDA 11.5 on CentOS Linux 7 (Core), with access to Nvidia A100 (80GB RAM), AMD EPYC 7352 24-Core Processor, and 1TB RAM. Run the following to create two conda environments (diffdock, pymol).

```
conda env create -f environment_diffdock.yml # create an environment, diffdock
conda env create -f environment_pymol.yml # create an environment, pymol
```
##Datasets
Download the processed data from and decompress the files

```
tar -xvzf benchmark.tar.gz
tar -xvzf ablation_study.tar.gz
```

and place them in `/data` directory.
###File structure

```
|--data
   |--ablation_study
      |--crystal_crystal
      |--crystal_diffdock
      |--colabfold_diffdock
      |--train.csv
      |--valid.csv
      |--test.csv
   |--benchmark
      |--complex
      |--davis_colabfold_protein
      |--davis_ligand
      |--davis_data.tsv
```

 
##Replicate results
###Affinity prediction benchmark
####Folding 
Use ColabFold to generate three-dimensional protein structures. Please follow [localcolabfold](https://github.com/YoshitakaMo/localcolabfold) to install the working environment. Or directly download the processed data from and place them in `/data` directory and jump to the last step.

```python
python folding/create_davis_protein_input.py
colabfold_batch --templates --amber folding/input/davis_protein.csv folding/output/davis_colabfold_protein --use-gpu-relax --num-relax 1 --gpu 10

```
####Docking
Create protein-ligand complex directories

```
conda activate diffdock
python docking/create_dir.py
```
Implement DiffDock to generate ligand binding poses

```
cd docking/DiffDock
python -m affinity.dataset_davis_colabfold --run_name davis_colabfold --inference_steps 20 --samples_per_complex 10 --batch_size 10 --ns 12 --nv 6 --num_conv_layers 3 --dynamic_max_cross --scale_by_sigma --dropout 0.2 --remove_hs --c_alpha_max_neighbors 24 --receptor_radius 15 --gpu_num 6
```

Add DiffDock-generated ligand poses into original complex directories.

```
cd ../../
python docking/update_dir.py
```

####Affinity
Pre-process protein-ligand complexes and generate inputs for GIGN.

```
conda activate pymol
python affinity/GIGN/preprocessing.py
conda activate diffdock
cd affinity/GIGN
python dataset_GIGN_benchmark.py
```

Train GIGN to predict binding affinity under different split_methods (drug, protein, both, and seqid).

```
python train_GIGN_benchmark.py --split_method drug
```
###Ablation study
Download the processed data from and place them in `/data`. Train GIGN to predict binding affinity under three different scenarios (crystal\_crystal, crystal\_diffdock, and colabfold\_diffdock) 

```
python train_GIGN_ablation.py --scenario crystal_crystal
```
