# MC-Struct-Gen: 3D Voxel Structure Generation with Conditional Diffusion

A PyTorch-based 3D generative model for Minecraft structure generation using conditional diffusion and prototype-based classification.

> **⚠️ Work in Progress**: This is an experimental project created for learning and exploration. The implementation not be fully functional or complete.

## Overview

MC-Struct-Gen implements a sophisticated 3D voxel generation system combining:

- **3D UNet Architecture**: Conditional diffusion-based model with sinusoidal positional embeddings. Implementation is similar to Unet2D in [diffusers](https://github.com/huggingface/diffusers) library
- **Prototype-Based Classification**: Embedding-based voxel head for block type prediction using learned prototypes
- **Cross-Attention Mechanism**: Context-aware 3D feature processing
- **Metadata Prediction**: Per-voxel binary metadata channels (e.g., rotation, orientation)

## Project Structure

```
mc-struct-gen/
├── model.py          # Core 3D UNet diffusion model with residual and attention blocks
├── struct_head.py    # Prototype-based classification head for voxel prediction
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Key Components

### model.py
Implements the complete 3D conditional diffusion model:

- **SinusoidalPosEmb**: Sinusoidal positional embeddings for time/condition encoding
- **ResidualBlock3D**: 3D residual blocks with group normalization and conditional projection
- **DownBlock3D & UpBlock3D**: Down/upsampling blocks for UNet encoder-decoder
- **CrossAttention3D**: Cross-attention mechanism for context integration
- **UNet3DConditional**: Full 3D UNet architecture for conditional generation

### struct_head.py
Prototype-based embedding head for voxel classification:

- **MCStructEmbedHead**: Classification head with:
  - Learned block embeddings (prototypes) for each block type
  - Feature projection from UNet output to embedding space
  - Cosine similarity-based classification
  - Optional MLP projector with bottleneck design
  - Per-voxel metadata prediction (rotation, state, etc.)

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
