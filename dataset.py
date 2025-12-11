	
from typing import Optional

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from model import get_text_embedding

class MCStructureDataset(Dataset):
	"""Dataset for Minecraft structure voxel data with text conditioning."""

	def __init__(self, data_dir: str, clip_model, max_samples: Optional[int] = None):
		self.data_dir = data_dir
		self.clip_model = clip_model
		self.samples = []

		# Load dataset metadata
		# This assumes you have a dataset with voxel data and text prompts
		# You'll need to adapt this to your actual data format
		self._load_dataset(max_samples)

	def _load_dataset(self, max_samples):
		# Placeholder - implement based on your data format
		# Expected format: each sample has voxel_data.npy and prompt.txt
		data_files = [f for f in os.listdir(self.data_dir) if f.endswith('_voxels.npy')]
		if max_samples:
			data_files = data_files[:max_samples]

		for voxel_file in data_files:
			base_name = voxel_file.replace('_voxels.npy', '')
			prompt_file = f"{base_name}_prompt.txt"

			if os.path.exists(os.path.join(self.data_dir, prompt_file)):
				self.samples.append({
					'voxel_path': os.path.join(self.data_dir, voxel_file),
					'prompt_path': os.path.join(self.data_dir, prompt_file)
				})

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]

		# Load voxel data (expected shape: C, D, H, W)
		voxels = np.load(sample['voxel_path'])
		voxels = torch.from_numpy(voxels).float()

		# Load and encode text prompt
		with open(sample['prompt_path'], 'r') as f:
			prompt = f.read().strip()

		# Get CLIP text embedding
		text_embedding = get_text_embedding(self.clip_model, [prompt])
		with torch.no_grad():
			text_embedding = text_embedding.squeeze(0)	# Remove batch dim

		return {
			'voxels': voxels,
			'text_embedding': text_embedding,
			'prompt': prompt
		}

if __name__ == "__main__":
	pass