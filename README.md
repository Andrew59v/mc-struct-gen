# MC-Struct-Gen: 3D Voxel Structure Generation with Conditional Diffusion

A PyTorch-based 3D generative model for Minecraft structure generation using conditional diffusion and prototype-based classification, built with the [Hugging Face Diffusers](https://github.com/huggingface/diffusers) library.

> **⚠️ Work in Progress**: This is an experimental project created for learning and exploration. The implementation not fully functional or complete.

## Overview

MC-Struct-Gen implements a sophisticated 3D voxel generation system combining:

- **3D UNet Architecture**: Conditional diffusion model with cross-attention and feature extraction
- **Prototype-Based Classification**: Embedding-based voxel head for block type prediction using learned prototypes and cosine similarity
- **Cross-Attention Mechanism**: Text-to-voxel attention for prompt-guided generation
- **Metadata Prediction**: Per-voxel binary metadata channels (e.g., rotation, orientation) via binary cross-entropy loss
- **Multi-task Training**: Combined denoising, block classification, and metadata prediction losses

## Project Structure

```
mc-struct-gen/
├── model.py          # Core 3D UNet diffusion model with residual and attention blocks
├── struct_head.py    # Prototype-based classification head for block ID and metadata prediction
├── generate.py       # Diffusers pipeline for text-to-structure generation
├── dataset.py        # Data loading utilities for voxel structures and text embeddings
├── train.py          # Training script using Diffusers for conditional diffusion
├── legacy.json       # Legacy minecraft block/item ids for .schematic parser
├── blocks.json       # Block dictionary for model
├── metadata.json     # Metadata dictionary for model
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

legacy.json file is sourced from [minecraft-data](https://github.com/PrismarineJS/minecraft-data).

## Key Components

### model.py
Implements the complete 3D conditional diffusion model with timestep and text conditioning:

- **Timesteps**: Sinusoidal positional embeddings from [diffusers.models.embeddings](https://huggingface.co/docs/diffusers)
- **ResidualBlock3D**: 3D residual blocks with group normalization and conditional projection
- **DownBlock3D & UpBlock3D**: Down/upsampling blocks for UNet encoder-decoder architecture
- **CrossAttention3D**: Cross-attention mechanism for text-to-voxel interaction (77 CLIP tokens)
- **UNet3DConditional**: Full 3D UNet with timestep, text embeddings, and feature extraction
  - Returns (denoised_voxels, features) when `return_features=True`
  - Features feed into struct_head for classification

### struct_head.py
Prototype-based embedding head for voxel classification:

- **MCStructEmbedHead**: Classification head with:
  - Learned block embeddings (prototypes) for K block types
  - Feature projection (single 1x1 conv or MLP bottleneck) from UNet to embedding space
  - Cosine similarity-based block logits computation
  - Per-voxel metadata prediction via 1x1 convolution
  - `forward_from_features()`: Returns raw logits for training
  - `logits_to_outputs()`: Converts logits to discrete block IDs and metadata flags (0/1)
  - Configurable temperature, embedding normalization, and optional freezing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Andrew59v/mc-struct-gen.git
cd mc-struct-gen
```

2. Create and activate a virtual environment (recommended):
```bash
python3 -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

3. Install dependencies:
```bash
pip3 install -r requirements.txt
```

## Usage

1. Download dataset (.litematic / .schematic files) into './dataset'
2. Make dataset manifest
```python
manifest: List[Dict[str, str]] = [
  {
    "file": "path_to.litematic",
    "name": "title or description"
  },{
    "file": "path_to.schematic",
    "name": "title or description"
  }
]
with open('./dataset/dataset.json', 'w', encoding="utf-8") as fout:
  json.dump(manifest, fout, ensure_ascii=False)
```
3. Prepare dataset:
```bash
python3 dataset.py
```
4. Train model:
```bash
python3 train.py
```
5. Generate structure:
```bash
python3 generate.py
```
