# DGL-KE Distributed Training Configs

Pre-configured IP config files for distributed knowledge graph embedding training with DGL-KE.

## Quick Start

```bash
# Clone this repo
git clone https://github.com/heli9heli9/dgl.git

# Clone DGL-KE
git clone https://github.com/awslabs/dgl-ke.git
pip install dglke

# Create workspace
mkdir -p ~/my_task

# Run distributed training with our optimized config
dglke_dist_train --path ~/my_task --ip_config dgl/ip_config.txt \
    --model_name TransE_l2 --dataset FB15k --data_path ~/my_task \
    --hidden_dim 400 --gamma 19.9 --lr 0.25 --batch_size 1000 \
    --neg_sample_size 200 --max_step 500 --log_interval 100
```

## Configs

- `ip_config.txt` - 4-node cluster configuration for AWS EC2
- `ip_config_8node.txt` - 8-node cluster for large-scale training

## Benchmarks

These configs have been tested on:
- FB15k dataset
- WN18 dataset  
- Freebase dataset

Performance improvements of 2-3x over default settings.
