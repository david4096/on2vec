# Cross-Ontology Alignment Evaluation
During the BioHackathon, we started evaluating on2vec on the ontology alignment use case.

## Experiment Design
We designed an experiment for the alignment of two commonly used ontologies in the biomedical domain: OMIM and ORDO. 

Our method was based on the [OAEI 2025 challenge](https://liseda-lab.github.io/OAEI-Bio-ML/2025/index.html). 

For the evaluation, we downloaded the omim-ordo benchmark datasets Version OAEI Bio-ML 2025 from [Zenodo](https://zenodo.org/records/13119437).

### Workflow
#### 1. Merge ontologies
```bash
python ontology_merger.py omim-ordo/omim.owl omim-ordo/ordo.owl ./ontology_merged.owl
```

#### 2. Train `on2vec` model 
```bash
on2vec train ontology_merged.owl --output omim_ordo_alignment_g cn_model.pt --model-type gcn --epochs 100
```

#### 3. Create embeddings
```
on2vec embed --output omim_ordo_alignment_gcn_embeddings.parquet omim_ordo_alignment_gcn_model.pt ontology_merged.owl
```

#### 4. Evaluate alignment _in progress_
