## Overview

```diff
- Please note that the  repository is for reference purposes only. We do not guarantee its active functionality. A user-friendly version is currently under development.
```

Folding-Docking-Affinity (FDA) is a framework which folds proteins, determines protein-ligand binding conformations, and predicts binding affinities from computed three-dimensional protein-ligand binding structures.
<p align="center">
    <img src="figure/abstract_diagram.png">
    
## Dependencies
The Folding part was tested with Python 3.10.13 and CUDA 12.3 on Ubuntu 20.04, with access to Nvidia Tesla V100 (32GB RAM), Intel(R) Xeon(R) Platinum 8168 CPU @ 2.70GHz, and 1.5TB RAM. Please follow [localcolabfold](https://github.com/YoshitakaMo/localcolabfold) to install the working environment. 

The Docking and Affinity parts were tested with Python 3.9.18 and CUDA 11.5 on CentOS Linux 7 (Core), with access to Nvidia A100 (80GB RAM), AMD EPYC 7352 24-Core Processor, and 1TB RAM. Run the following to create a conda environment, FDA.

```
conda create --name FDA python=3.9
conda activate FDA
conda install conda-forge::pymol-open-source
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install scipy
pip install --no-index pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install torch_geometric
python -m pip install PyYAML scipy "networkx[default]" biopython rdkit-pypi e3nn spyrmsd pandas biopandas
!!!! need to install openbabel, but seems to have conflicts.
```
## Datasets
Create a directory `/data`, and download the processed data for replicating benchmark, ablation study, and binding pose augmentation results from [zenodo](https://zenodo.org/records/15058571) and decompress the files

```
git clone git@github.com:ZhiGroup/FDA.git
cd FDA
mkdir data
cd data
wget https://zenodo.org/records/15058571/files/benchmark.tar.gz?download=1
wget https://zenodo.org/records/15058571/files/ablation_study.tar.gz?download=1
tar -xvzf benchmark.tar.gz?download=1
tar -xvzf ablation_study.tar.gz?download=1
cd ../
```
### File structure

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
      |--davis_colabfold_protein
      |--davis_colabfold_protein_rank_2
      |--davis_colabfold_protein_rank_3
      |--davis_complex_colabfold_diffdock
      |--davis_complex_colabfold_rank2_diffdock
      |--davis_complex_colabfold_rank3_diffdock
      |--davis_ligand
      |--davis_data.tsv
      |--davis_cluster_id50_cluster.tsv
      |--kiba_colabfold_protein
      |--kiba_colabfold_protein_rank_2
      |--kiba_colabfold_protein_rank_3
      |--kiba_complex_colabfold_diffdock
      |--kiba_complex_colabfold_rank2_diffdock
      |--kiba_complex_colabfold_rank3_diffdock
      |--kiba_ligand
      |--kiba_data.tsv
      |--kiba_cluster_id50_cluster.tsv
```

 
## Replicate results
### Affinity prediction benchmark
#### Folding 
Use ColabFold to generate three-dimensional protein structures. Please follow [localcolabfold](https://github.com/YoshitakaMo/localcolabfold) to install the working environment. Or directly download the processed data from [zenodo](https://zenodo.org/records/15058571/files/benchmark.tar.gz?download=1) and place them in `/data` directory and jump to the [last step](#train-gign-to-predict-binding-affinity-under-different-split_methods-drug-protein-both-and-seqid).

```python
python folding/create_davis_protein_input.py
colabfold_batch --templates --amber folding/input/davis_protein.csv folding/output/davis_colabfold_protein --use-gpu-relax --num-relax 3 --gpu 0
```
#### Docking
Create protein-ligand complex directories

```
python docking/create_dir.py --data_dir data/benchmark/ --complex_dir_name davis_complex_colabfold_diffdock --protein_dir_name davis_colabfold_protein --ligand_dir_name davis_ligand
```
Download ESM2 embedding from [zenodo](https://zenodo.org/records/15058571/files/esm2_3billion_embeddings_davis_colabfold_protein.pt.tar.gz?download=1) and place the file in `docking/DiffDock/data/` for the input of DiffDock. The process of generating ESM2 embedding could refer [DiffDock](https://github.com/gcorso/DiffDock/tree/v1.0) or
follow the following commands.

```
cd docking/DiffDock
python datasets/esm_embedding_preparation.py --protein_path ../../data/benchmark/davis_colabfold_protein --out_file data/davis_colabfold_protein.fasta
git clone https://github.com/facebookresearch/esm.git
cd esm
python scripts/extract.py esm2_t33_650M_UR50D ../data/davis_colabfold_protein.fasta ../data/davis_colabfold_protein_embedding_output --repr_layers 33 --include per_tok --truncation_seq_length 4096
cd ../
python datasets/esm_embeddings_to_pt.py --esm_embeddings_path data/davis_colabfold_protein_embedding_output --output_path data/esm2_3billion_embeddings_davis_colabfold_protein.pt
```

Implement DiffDock to generate ligand binding poses
```
cd docking/DiffDock
python -m affinity.dataset_davis_colabfold --run_name davis_colabfold --split_train ../../data/benchmark/davis_data.tsv --data_dir ../../data/benchmark --protein_dir_name davis_colabfold_protein --ligand_dir_name davis_ligand --esm_embeddings_path data/esm2_3billion_embeddings_davis_colabfold_protein.pt --inference_steps 20 --samples_per_complex 10 --batch_size 10 --ns 12 --nv 6 --num_conv_layers 3 --dynamic_max_cross --scale_by_sigma --dropout 0.2 --remove_hs --c_alpha_max_neighbors 24 --receptor_radius 15 --gpu_num 0
```

Update DiffDock-generated ligand poses into original complex directories.

```
cd ../../
python docking/update_dir.py --diffdock_ligand_path docking/DiffDock/data/cacheNew/davis_colabfold_protein_davis_data_split_train_limit_0/ligand_positions_rank1.pkl --complex_path data/benchmark/davis_complex_colabfold_diffdock
```

#### Affinity
Pre-process protein-ligand complexes and generate inputs for [GIGN](https://github.com/guaguabujianle/GIGN).

```
python affinity/GIGN/preprocessing.py --data_df data/benchmark/davis_data.tsv --complex_path data/benchmark/davis_complex_colabfold_diffdock
cd affinity/GIGN
python affinity/GIGN/dataset_GIGN_benchmark.py --data_df data/benchmark/davis_data.tsv --complex_path data/benchmark/davis_complex_colabfold_diffdock --mmseqs_seq_clus_df data/benchmark/davis_cluster_id50_cluster.tsv
```
##### Train GIGN to predict binding affinity under different split_methods (drug, protein, both, and seqid).

```
cd affinity/GIGN
python train_GIGN_benchmark.py --job_name benchmark_davis_drug --split_method drug --gpu 0 --data_df ../../data/benchmark/davis_data.tsv --complex_path ../../data/benchmark/davis_complex_colabfold_diffdock
```
For the KIBA dataset, the process remains the same, with only the input being changed. The input can also be found in `/data/benchmark`.
### Ablation study
Download the processed data from [zenodo](https://zenodo.org/records/15058571/files/ablation_study.tar.gz?download=1) and place them in `/data`. Train GIGN to predict binding affinity under three different scenarios (crystal\_crystal, crystal\_diffdock, and colabfold\_diffdock).

```
cd affinity/GIGN
python train_GIGN_ablation.py --scenario crystal_crystal --gpu 0
```

### Binding poses augmentation
Use the same dataset as the affinity prediction benchmark. The dataset can be found in `/data/benchmark`. The following commands are used to train GIGN with different binding poses augmentation strategies.

F-5D-A: The top five binding poses generated by
DiffDock are used to augment the training set.
```
python train_GIGN_benchmark.py --split_method drug --job_name benchmark_davis_pose_augmentation5_protein_rank1_drug --gpu 0 --data_df ../../data/benchmark/davis_data.tsv --complex_path ../../data/benchmark/davis_complex_colabfold_diffdock --mmseqs_seq_clus_df ../../data/benchmark/davis_cluster_id50_cluster.tsv --seeds 0 1 2 3 4 --top_n 5
```
F-10D-A: The top ten binding poses generated
by DiffDock are used to augment the training set
```
python train_GIGN_benchmark.py --split_method drug --job_name benchmark_davis_pose_augmentation10_protein_rank1_drug --gpu 0 --data_df ../../data/benchmark/davis_data.tsv --complex_path ../../data/benchmark/davis_complex_colabfold_diffdock --mmseqs_seq_clus_df ../../data/benchmark/davis_cluster_id50_cluster.tsv --seeds 0 1 2 3 4 --top_n 10
```
2F-5D-A: For each protein-ligand pair, five
binding poses were selected from both the rank-1 and rank-2 protein conformations generated by
ColabFold, resulting in a total of 10 distinct binding poses
```
python train_GIGN_benchmark.py --split_method drug --job_name benchmark_davis_pose_augmentation10_protein_rank1_2_drug --gpu 0 --data_df ../../data/benchmark/davis_data.tsv --complex_path ../../data/benchmark/davis_complex_colabfold_diffdock --mmseqs_seq_clus_df ../../data/benchmark/davis_cluster_id50_cluster.tsv --seeds 0 1 2 3 4 --top_n 5 --protein_rank 1 2 
```
3F-5D-A: For each protein-ligand
pair, five binding poses were selected from the rank-1, rank-2, and rank-3 protein conformations
generated by ColabFold, resulting in a total of 15 distinct binding poses.
```
python train_GIGN_benchmark.py --split_method drug --job_name benchmark_davis_pose_augmentation15_protein_rank1_2_3_drug --gpu 0 --data_df ../../data/benchmark/davis_data.tsv --complex_path ../../data/benchmark/davis_complex_colabfold_diffdock --mmseqs_seq_clus_df ../../data/benchmark/davis_cluster_id50_cluster.tsv --seeds 0 1 2 3 4 --top_n 5 --protein_rank 1 2 3
```
For the KIBA dataset, the process remains the same, with only the input being changed. The input can also be found in /data/benchmark.
### Citation
Wu, MH., Xie, Z., & Zhi, D. Protein-ligand binding affinity prediction: Is 3D binding pose needed?. bioRxiv (2024). [https://doi.org/10.1101/2024.04.16.589805](https://doi.org/10.1101/2024.04.16.589805) 
