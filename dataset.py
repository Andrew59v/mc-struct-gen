	
import nbt
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Tuple
from pathlib import Path
import logging

from generate import get_text_embedding

logger = logging.getLogger(__name__)

class SchematicParser:
	"""Parse Minecraft .schematic files (NBT format)."""
	
	@staticmethod
	def parse(
		file_path: str,
		block_dict: Dict[str, int],
		metadata_dict: Dict[str, Dict[str, int]]
		) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Parse .schematic file and extract block IDs and metadata flags.
		
		Args:
			file_path: Path to .schematic file
			block_dict: Mapping {block_name: block_id}
			metadata_dict: Mapping {attribute: {value: bit_index}}
			
		Returns:
			(block_ids, metadata_flags): 
				block_ids shape (1, D, H, W)
				metadata_flags shape (M, D, H, W) where M = max bit index + 1
		"""
		
		with open(file_path, 'rb') as f:
			nbt_data = nbt.load(f)
		
		# Extract dimensions
		width = nbt_data['Width'].value
		height = nbt_data['Height'].value
		length = nbt_data['Length'].value

		if width > 256 or height > 256 or length > 256:
			raise ValueError(f"Structure dimensions are too large: ({length}, {height}, {width})")
		
		# Extract block data
		blocks = np.frombuffer(nbt_data['Blocks'].value, dtype=np.uint8)
		blocks = blocks.reshape((length, height, width))
		blocks = np.transpose(blocks, (0, 2, 1))  # Reorder to (D, H, W)
		
		# Extract add (high bits) if present
		if 'Add' in nbt_data:
			add = np.frombuffer(nbt_data['Add'].value, dtype=np.uint8)
			add = add.reshape((length, height, width))
			add = np.transpose(add, (0, 2, 1))  # (D, H, W)
			blocks = blocks | (add << 8)  # Combine with high bits
		
		# Extract block data (metadata/damage values)
		block_data = None
		if 'Data' in nbt_data:
			block_data = np.frombuffer(nbt_data['Data'].value, dtype=np.uint8)
			block_data = block_data.reshape((length, height, width))
			block_data = np.transpose(block_data, (0, 2, 1))  # (D, H, W)
		
		# Convert to tensors
		block_ids = torch.from_numpy(blocks).long().unsqueeze(0)  # (1, D, H, W)
		
		# Calculate num metadata channels from metadata_dict
		num_metadata_channels = 0
		for attr_map in metadata_dict.values():
			num_metadata_channels = max(num_metadata_channels, max(attr_map.values()) + 1)
		
		D, H, W = blocks.shape
		metadata_flags = torch.zeros(num_metadata_channels, D, H, W, dtype=torch.float32)
		
		# Extract metadata from block_data if available
		if block_data is not None:
			# TODO: Parse block_data to extract specific attributes
			# This requires mapping block state bits to metadata_dict attributes
			pass
		
		return block_ids, metadata_flags


class LitematicParser:
	"""Parse Minecraft .litematic files (structure format)."""
	
	@staticmethod
	def parse(
		file_path: str,
		block_dict: Dict[str, int],
		metadata_dict: Dict[str, Dict[str, int]]
		) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Parse .litematic file and extract block IDs and metadata flags.
		
		Args:
			file_path: Path to .litematic file
			block_dict: Mapping {block_name: block_id}
			metadata_dict: Mapping {attribute: {value: bit_index}}
			
		Returns:
			(block_ids, metadata_flags):
				block_ids shape (1, D, H, W)
				metadata_flags shape (M, D, H, W) where M = max bit index + 1
		"""
		
		with open(file_path, 'rb') as f:
			nbt_data = nbt.load(f)
		
		# Litematic format stores regions
		regions = nbt_data['Regions']
		if not regions:
			raise ValueError("No regions found in litematic file")
		
		# Get first region (main structure)
		region_tag = regions[0]
		
		# Extract dimensions
		size = region_tag['Size']
		width = size['x'].value
		height = size['y'].value
		length = size['z'].value

		if width > 256 or height > 256 or length > 256:
			raise ValueError(f"Structure dimensions are too large: ({length}, {height}, {width})")
		
		# Extract block palette and block states
		palette = region_tag['BlockStatePalette']
		block_states = np.frombuffer(region_tag['BlockStates'].value, dtype=np.uint64)
		
		# Decode block states (variable-length encoding)
		blocks = LitematicParser._decode_block_states(block_states, palette, length, height, width)
		
		# Convert to tensors
		block_ids = torch.from_numpy(blocks).long().unsqueeze(0)  # (1, D, H, W)
		
		# Calculate num metadata channels from metadata_dict
		num_metadata_channels = 0
		for attr_map in metadata_dict.values():
			num_metadata_channels = max(num_metadata_channels, max(attr_map.values()) + 1)
		
		D, H, W = blocks.shape
		metadata_flags = torch.zeros(num_metadata_channels, D, H, W, dtype=torch.float32)
		
		# Extract metadata from palette block states
		# TODO: Parse palette block properties to extract metadata attributes
		
		return block_ids, metadata_flags
	
	@staticmethod
	def _decode_block_states(block_states: np.ndarray, palette, length: int, height: int, width: int) -> np.ndarray:
		"""Decode block states from variable-length integer array."""
		blocks = np.zeros((length, width, height), dtype=np.uint16)
		
		# Simplified decoding (full implementation depends on palette size)
		# This is a placeholder - real implementation needs proper bit manipulation
		palette_size = len(palette)
		bits_per_block = max(4, (palette_size - 1).bit_length())
		
		# TODO: Implement proper bit-level decoding for block states
		logger.warning("Litematic decoding is simplified. Full implementation pending.")
		
		return blocks


class MCStructureDataset(Dataset):
	"""Dataset for prepared Minecraft structures with CLIP embeddings."""
	
	def __init__(self, prepared_dir: str, max_samples: Optional[int] = None):
		"""
		Load prepared dataset from prepared folder.
		
		Args:
			prepared_dir: Path to ./prepared folder containing .pt files
			max_samples: Maximum number of samples to load
		"""
		self.prepared_dir = prepared_dir
		self.samples = []
		
		# Find all .pt files
		pt_files = sorted(Path(prepared_dir).glob('*.pt'))
		if max_samples:
			pt_files = pt_files[:max_samples]
		
		self.samples = [str(f) for f in pt_files]
		
		if not self.samples:
			raise ValueError(f"No prepared samples found in {prepared_dir}")
		
		logger.info(f"Loaded {len(self.samples)} prepared samples from {prepared_dir}")
	
	def __len__(self):
		return len(self.samples)
	
	def __getitem__(self, idx):
		sample_path = self.samples[idx]
		sample = torch.load(sample_path)
		
		return {
			'block_ids': sample['block_ids'],
			'metadata_flags': sample['metadata_flags'],
			'text_embedding': sample['text_embedding'],
		}


class DatasetGenerator:
	"""Generate prepared dataset from raw schematics/litematic files."""
	
	def __init__(
		self,
		dataset_dir: str,
		block_dict: Dict[str, int],
		metadata_dict: Dict[str, Dict[str, int]],
		clip_model,
	):
		"""
		Initialize dataset generator.
		
		Args:
			dataset_dir: Path to ./dataset folder containing dataset.json
			block_dict: Mapping {block_name: block_id}
			metadata_dict: Mapping {metadata_name: bit_index}
			clip_model: CLIP model for text encoding
			num_metadata_channels: Number of binary metadata channels (M)
		"""
		self.dataset_dir = Path(dataset_dir)
		self.block_dict = block_dict
		self.metadata_dict = metadata_dict
		self.clip_model = clip_model

		largest_bit = 0
		for attr_map in metadata_dict.values():
			largest_bit = max(largest_bit, max(attr_map.values()))
		self.num_metadata_channels = largest_bit + 1
		
		self.prepared_dir = self.dataset_dir / 'prepared'
		self.prepared_dir.mkdir(exist_ok=True)
		
		# Load manifest
		self.manifest_path = self.dataset_dir / 'dataset.json'
		if not self.manifest_path.exists():
			raise FileNotFoundError(f"dataset.json not found in {self.dataset_dir}")
		
		with open(self.manifest_path, 'r') as f:
			self.manifest = json.load(f)
		
		if not isinstance(self.manifest, list):
			raise ValueError("dataset.json must contain an array of structures")
		
		logger.info(f"Loaded manifest with {len(self.manifest)} structures")
	
	def _parse_structure(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
		"""Parse .schematic or .litematic file."""
		file_path = str(self.dataset_dir / file_path)
		
		if file_path.endswith('.schematic'):
			return SchematicParser.parse(file_path, self.block_dict, self.metadata_dict)
		elif file_path.endswith('.litematic'):
			return LitematicParser.parse(file_path, self.block_dict, self.metadata_dict)
		else:
			raise ValueError(f"Unsupported file format: {file_path}")
	
	def process(self, sample_idx: Optional[int] = None):
		"""
		Process single structure or all structures.
		
		Args:
			sample_idx: Index of sample to process, or None for all
		"""
		samples = [self.manifest[sample_idx]] if sample_idx is not None else self.manifest
		
		for i, structure_info in enumerate(samples):
			try:
				file_path = structure_info['file']
				description = structure_info.get('description', structure_info['title'])
				
				logger.info(f"Processing {file_path}...")
				
				# Parse structure
				block_ids, metadata_flags = self._parse_structure(file_path)
				
				# Get text embedding using generate.py's function
				text_embedding = get_text_embedding(
					self.clip_model, [description]
				).squeeze(0)  # Remove batch dim: (77, 512)
				
				# Save prepared sample
				sample_name = Path(file_path).stem
				output_path = self.prepared_dir / f'{sample_name}.pt'
				
				torch.save({
					'text_embedding': text_embedding,
					'block_ids': block_ids,
					'metadata_flags': metadata_flags,
				}, output_path)
				
				logger.info(f"Saved prepared sample to {output_path}")
				logger.info(f"  block_ids: {block_ids.shape}, metadata_flags: {metadata_flags.shape}, text_emb: {text_embedding.shape}")
				
			except Exception as e:
				logger.error(f"Failed to process {structure_info.get('file', 'unknown')}: {e}")
				continue


if __name__ == "__main__":
	# Example usage
	logging.basicConfig(level=logging.INFO)
	
	# Define block dictionary: block_name -> block_id
	block_dict = {
		'air': 0,
		'stone': 1,
		'dirt': 2,
		'grass_block': 3,
		'cobblestone': 4,
		# ... more blocks
	}
	
	metadata_dict = {
		'rotation': {
			'0': 0,
			'1': 1,
			'2': 2,
			'3': 3,
			'4': 4,
			'5': 5,
			'6': 6,
			'7': 7,
			'8': 8,
			'9': 9,
			'10': 10,
			'11': 11,
			'12': 12,
			'14': 14,
			'15': 15,
		},
		"axis": {
			"x": 16,
			"y": 17,
			"z": 18,
		},
		'facing': {
			'down': 19,
			'east': 20,
			'north': 21,
			'south': 22,
			'up': 23,
			'west': 24,
		},
		'waterlogged': {
			'true': 25,
			'false': 26,
		},
		'half': {
			'lower': 8,
			'upper': 9
		},
		# ... more attributes
	}
	
	# Initialize generator
	generator = DatasetGenerator(
		dataset_dir='./dataset',
		block_dict=block_dict,
		metadata_dict=metadata_dict
	)
	
	# Process all structures
	generator.process()