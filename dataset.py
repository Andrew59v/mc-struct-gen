	
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Tuple, List
from pathlib import Path
import logging
import clip
from nbt import nbt
import re

from generate import get_text_embedding

logger = logging.getLogger(__name__)

class StructureParser:
	def __init__(
			self,
			block_dict: Dict[str, int],
			metadata_dict: Dict[str, Dict[str, int]],
			metadata_dim: int = 32):
		"""
		Initialize structure parser.
		
		Args:
			block_dict: Mapping {block_name: block_id}
			metadata_dict: Mapping {metadata_name: bit_index}
			metadata_dim: Number of binary metadata channels (M)
		"""
		self.block_dict = block_dict
		self.metadata_dict = metadata_dict
		self.metadata_dim = metadata_dim
	
	def metadata_to_array(self, metadata: List[Tuple[str, str]]):
		result = np.zeros(metadata_dim, dtype=np.bool)
		if metadata:
			for key, value in metadata:
				if key in metadata_dict and value in metadata_dict[key]:
					bit_index = self.metadata_dict[key][value]
					result[bit_index] = 1
		return result

	def __call__(self, file_path: str) -> Tuple[str, torch.Tensor, torch.Tensor]:
		"""Parse structure file and extract block IDs and metadata flags.
		Args:
			file_path: Path to structure file
			legacy_map: Mapping from legacy block IDs to new block IDs + metadata
		Returns:
			(description, block_ids, metadata_flags): 
				description: text description of the structure
				block_ids: tensor of shape (H, W, D) with block IDs
				metadata_flags: tensor of shape (M, H, W, D) with metadata binary flags
		"""
		raise NotImplementedError

class SchematicParser(StructureParser):
	"""Parse Minecraft .schematic files (NBT format).
	
	Format (only used fields) (from https://minecraft.fandom.com/wiki/Schematic_file_format):
	- Width
	- Height
	- Length
	- Blocks: byte array of block IDs, the index of the block at X,Y,Z is (Y*length + Z)*width + X
	- Data: optional byte array of block metadata, full byte per block, but only lower 4 bits used
	- Add: optional byte array of high bits for block IDs > 255 (4 bits per block, packed two per byte)
	"""
	
	def __init__(
			self,
			block_dict: Dict[str, int],
			metadata_dict: Dict[str, Dict[str, int]],
			metadata_dim: int = 32,
			legacy_file: str = "./legacy.json"):
		super().__init__(block_dict, metadata_dict, metadata_dim)

		with open(legacy_file, 'r') as f:
			legacy_dict: Dict[str, str] = json.load(f)["blocks"]

		# Legacy block ID mapping from old Minecraft versions to new.
		# self.legacy_map[mc_block_id][subid] = (block_id, metadata_flags)
		self.legacy_map: List[List[Tuple[int, np.ndarray]]] = [None] * 256

		for legacy_id, block_state in legacy_dict.items():
			args = legacy_id.split(':')
			mc_block_id = int(args[0])
			subid = int(args[1])
			
			block_name = block_state.split('[')[0].strip()
			if block_name not in self.block_dict:
				continue
			
			m = re.search(r'\[(.*)\]', block_state)
			metadata: List[Tuple[str, str]] = []
			if m:
				args = m.group(1).split(',')
				for pair in args:
					key, value = pair.split("=")
					metadata.append((key, value.lower()))
			
			map_item = (self.block_dict[block_name], self.metadata_to_array(metadata))
			if self.legacy_map[mc_block_id] is None: # usually first item is subid=0
				self.legacy_map[mc_block_id] = [map_item] * 16
			else:
				self.legacy_map[mc_block_id][subid] = map_item
	
	# see StructureParser.__call__
	def __call__(self, file_path: str) -> Tuple[str, torch.Tensor, torch.Tensor]:
		if self.legacy_map is None:
			raise RuntimeError("Legacy map not initialized in SchematicParser")
		
		nbt_data = nbt.NBTFile(file_path, 'rb')

		# Extract dimensions
		width = nbt_data['Width'].value
		height = nbt_data['Height'].value
		length = nbt_data['Length'].value
		volume = width * height * length
		
		if width > 256 or height > 256 or length > 256:
			raise ValueError(f"Structure dimensions are too large: ({width}, {height}, {length})")
		if 'AddBlocks' in nbt_data:
			raise NotImplementedError("AddBlocks format not supported yet")
		if 'Add' in nbt_data:
			raise NotImplementedError("Add format not supported yet")

		# nbt_data['Blocks'].value == block IDs as byte array
		# nbt_data['Data'].value == block metadata as byte array (4 bits per block)
		
		# Extract block data
		block_ids_array = np.array(nbt_data['Blocks'].value, dtype=np.uint8)
		
		# Extract metadata (4 bits per block)
		data_array = None
		if 'Data' in nbt_data:
			data_array = np.array(nbt_data['Data'].value, dtype=np.uint8)
		
		# Map legacy block IDs to new block IDs and extract metadata
		block_id_array = np.zeros(volume, dtype=np.uint16)
		metadata_tensor = np.zeros((self.metadata_dim, volume), dtype=np.bool)
		
		for idx in range(volume):
			mc_block_id = block_ids_array[idx]
			mc_subid = (data_array[idx] & 0x0F) if data_array is not None else 0
			
			# Look up in legacy map
			if self.legacy_map[mc_block_id] is not None and self.legacy_map[mc_block_id][mc_subid] is not None:
				block_id, metadata_flags = self.legacy_map[mc_block_id][mc_subid]
			else:
				# Default to air if not found
				block_id, metadata_flags = self.legacy_map[0][0]
			
			block_id_array[idx] = block_id
			metadata_tensor[:, idx] = metadata_flags
		
		# Reshape to (H, L, W) = (height, width, length)
		shape = (height, width, length)
		block_ids_reshaped = block_id_array.reshape(shape)
		metadata_reshaped = metadata_tensor.reshape((self.metadata_dim, *shape))
		
		# TODO: swap Z and Y here
		
		# Convert to tensors
		block_ids = torch.from_numpy(block_ids_reshaped)
		metadata_flags = torch.from_numpy(metadata_reshaped)
		
		return None, block_ids, metadata_flags


class LitematicParser(StructureParser):
	"""Parse Minecraft .litematic files (Sponge Litematic format).
	
	Format (from litematica mod source):
	- Version: schematic version (int)
	- MinecraftDataVersion: data version (int) 
	- Metadata: {Name, Author, Description, RegionCount, TimeCreated, TimeModified, TotalVolume, TotalBlocks}
	- Regions: {region_name: {
	    Position: {x, y, z},
	    Size: {x, y, z},
	    BlockStatePalette: [BlockState compounds],
	    BlockStates: long array (packed palette indices),
	    TileEntities: [tile entity compounds],
	    Entities: [entity compounds],
	    PendingBlockTicks: [pending tick compounds]
	  }}
	"""

	def __init__(
			self,
			block_dict: Dict[str, int],
			metadata_dict: Dict[str, Dict[str, int]],
			metadata_dim: int = 32):
		super().__init__(block_dict, metadata_dict, metadata_dim)
	
	# see StructureParser.__call__
	def __call__(self, file_path: str) -> Tuple[str, torch.Tensor, torch.Tensor]:
		nbt_data = nbt.NBTFile(file_path, 'rb')

		
		# Validate litematic format
		if 'Version' not in nbt_data or 'Regions' not in nbt_data:
			raise ValueError("Invalid litematic file: missing Version or Regions tag")
		
		regions = nbt_data['Regions']
		if not regions or len(regions) == 0:
			raise ValueError("No regions found in litematic file")

		# Get first region (main structure) - regions is a dict-like compound
		region_name = list(regions.keys())[0] if hasattr(regions, 'keys') else None
		if region_name is None:
			raise ValueError("Unable to read region name from litematic")
		
		region_tag = regions[region_name]
		
		# Extract region position and size
		if 'Position' not in region_tag or 'Size' not in region_tag:
			raise ValueError(f"Region '{region_name}' missing Position or Size tags")
		
		size_tag = region_tag['Size']
		# Size is stored as BlockPos compound with x, y, z tags
		width = abs(size_tag['z'].value)
		height = abs(size_tag['y'].value)
		depth = abs(size_tag['x'].value)
		volume = width * height * depth
		
		if width > 256 or height > 256 or depth > 256:
			raise ValueError(f"Structure dimensions are too large: ({height}, {width}, {depth})")

		# Extract block states and palette
		if 'BlockStates' not in region_tag or 'BlockStatePalette' not in region_tag:
			raise ValueError(f"Region '{region_name}' missing BlockStates or BlockStatePalette")
		
		block_states_raw = region_tag['BlockStates'].value
		block_states = np.array(block_states_raw, dtype=np.int64)

		palette: List[Tuple[int, np.ndarray]] = []
		palette_tag = region_tag['BlockStatePalette']
		for palette_item in palette_tag:
			name = palette_item["Name"].value
			if name not in self.block_dict:
				raise RuntimeError("Palette contains unknown block:" + name)
			metadata = []
			if "Properties" in palette_item:
				properties = palette_item["Properties"]
				for key, value in properties.items():
					metadata.append((key, str(value).lower()))
			palette.append((self.block_dict[name], self.metadata_to_array(metadata)))
		
		# Decode block states to palette indices
		blocks = LitematicParser._decode_block_states(block_states, len(palette_tag), volume)

		shape = (height, width, depth)
		block_ids = torch.zeros(shape, dtype=torch.uint16)
		metadata_flags = torch.zeros((self.metadata_dim, *shape), dtype=torch.bool)

		# Extract real block_ids and metadata_flags from palette indices
		blocks_reshaped = blocks.reshape(shape)
		for h in range(height):
			for w in range(width):
				for l in range(depth):
					palette_idx = blocks_reshaped[h, w, l]
					if palette_idx < len(palette):
						block_id, metadata = palette[palette_idx]
						block_ids[h, w, l] = block_id
						metadata_flags[:, h, w, l] = torch.from_numpy(metadata)
		
		return None, block_ids, metadata_flags
	
	@staticmethod
	def _decode_block_states(block_states: np.ndarray, palette_size: int, volume: int) -> np.ndarray:
		"""Decode block states from long array (packed palette indices).
		
		Litematica stores blocks as palette indices in a packed bit array stored in longs (int64).
		Each block's palette index takes up bits_per_block bits, where:
		  bits_per_block = max(4, ceil(log2(palette_size)))
		
		The bits are stored LSB-first within each long, and the longs are read in order.
		"""
		blocks = np.zeros(volume, dtype=np.uint16)
		
		# Calculate bits per block based on palette size
		if palette_size <= 1:
			bits_per_block = 1
		else:
			bits_per_block = max(4, (palette_size - 1).bit_length())
		
		# Extract palette indices from packed block states
		bit_offset = 0
		for block_idx in range(volume):
			# Extract bits_per_block bits starting at bit_offset
			palette_idx = 0
			for b in range(bits_per_block):
				current_long_idx = (bit_offset + b) // 64
				current_bit_idx = (bit_offset + b) % 64
				
				if current_long_idx < len(block_states):
					bit_value = (block_states[current_long_idx] >> current_bit_idx) & 1
					palette_idx |= (bit_value << b)
			
			blocks[block_idx] = palette_idx
			bit_offset += bits_per_block
		
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
			'block_ids': sample['block_ids'].unsqueeze(0),  # (H, W, D) -> (1, H, W, D)
			'metadata_flags': sample['metadata_flags'],
			'text_embedding': sample['text_embedding'],
		}


class DatasetGenerator:
	"""Generate prepared dataset from raw schematics/litematic files."""
	
	def __init__(
		self,
		block_dict: Dict[str, int],
		metadata_dict: Dict[str, Dict[str, int]],
		metadata_dim: int,
		clip_model,
		dataset_dir: str = "./dataset",
		output_dir: str = "./prepared",
		legacy_file: str = "./legacy.json"
	):
		"""
		Initialize dataset generator.
		
		Args:
			block_dict: Mapping {block_name: block_id}
			metadata_dict: Mapping {metadata_key: {metadata_value: bit_index}}
			clip_model: CLIP model for text encoding
			dataset_dir: Path to ./dataset folder containing dataset.json
			output_dir: Path to ./prepared folder to save processed samples
			legacy_file: Path to legacy block ID mapping file
		"""

		self.dataset_dir = Path(dataset_dir)
		self.clip_model = clip_model
		
		self.prepared_dir = Path(output_dir)
		self.prepared_dir.mkdir(exist_ok=True)
		
		# Load manifest
		self.manifest_path = self.dataset_dir / 'dataset.json'
		if not self.manifest_path.exists():
			raise FileNotFoundError(f"dataset.json not found in {self.dataset_dir}")
		
		with open(self.manifest_path, 'r', encoding='utf-8') as f:
			self.manifest = json.load(f)
		
		if not isinstance(self.manifest, list):
			raise ValueError("dataset.json must contain an array of structures")
		
		logger.info(f"Loaded manifest with {len(self.manifest)} structures")

		self.schematic_parser = SchematicParser(
			block_dict=block_dict,
			metadata_dict=metadata_dict,
			metadata_dim=metadata_dim,
			legacy_file=legacy_file
		)
		self.litematic_parser = LitematicParser(
			block_dict=block_dict,
			metadata_dict=metadata_dict,
			metadata_dim=metadata_dim
		)
	
	def _parse_structure(self, file_path: str) -> Tuple[str, torch.Tensor, torch.Tensor]:
		"""Parse .schematic or .litematic file."""
		file_path = str(self.dataset_dir / file_path)
		
		if file_path.endswith('.schematic'):
			return self.schematic_parser(file_path)
		elif file_path.endswith('.litematic'):
			return self.litematic_parser(file_path)
		else:
			raise ValueError(f"Unsupported file format: {file_path}")
	
	def print_slices(self, block_ids, metadata_flags):
		"""
		Prints vertical slices of the structure to console.
		Just for debug purposes
		"""
		import numpy as np

		# Convert to numpy if tensor
		if hasattr(block_ids, 'cpu'):
			block_ids = block_ids.cpu().numpy()

		# Get dimensions
		height, width, depth = block_ids.shape
		print(f"\nStructure dimensions: {height}x{width}x{depth}")
		print("=" * 50)

		# Print horizontal slices (top to bottom)
		for y in range(height):
			print(f"\nLayer Y={y} (height {y}/{height-1}):")
			print("-" * 30)

			for x in range(width):
				row = ""
				for z in range(depth):
					block_id = int(block_ids[y, x, z])
					row += f"{block_id:4d}"
				print(f"X={x:2d}: {row}")

			print()

	def process(self, sample_idx: Optional[int] = None):
		"""
		Process single structure or all structures.
		
		Args:
			sample_idx: Index of sample to process, or None for all
		"""
		samples = [self.manifest[sample_idx]] if sample_idx is not None else self.manifest
		
		success = 0
		failed = 0

		for i, structure_info in enumerate(samples):
			try:
				file_path = structure_info['file']
				logger.info(f"Processing {file_path}...")
				
				# Parse structure
				description, block_ids, metadata_flags = self._parse_structure(file_path)
				if description is None:
					description = structure_info.get('description', structure_info['title'])

				#self.print_slices(block_ids, metadata_flags)
				
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
				#logger.info(f"  block_ids: {block_ids.shape}, metadata_flags: {metadata_flags.shape}, text_emb: {text_embedding.shape}")
				success += 1
			except Exception as e:
				logger.error(f"Failed to process {structure_info.get('file', 'unknown')}: {e}")
				failed += 1
				
		logger.info(f"Processed {len(samples)} samples")
		logger.info(f"Successful: {success}")
		logger.info(f"Failed: {failed}")


if __name__ == "__main__":
	# Example usage
	logging.basicConfig(level=logging.INFO)

	# {"<block_id>": <index>, ...}
	# {"<metadata_key>": {"metadata_value": <index>, ...}, ...}
	with open("./dictionary.json", 'r', encoding='utf-8') as f:
		dictionary = json.load(f)
		metadata_dict = dictionary["metadata"]
		block_dict = dictionary["blocks"]

	metadata_dim = 32
	for attr_map in metadata_dict.values():
		metadata_dim = max(metadata_dim, max(attr_map.values()) + 1)
	
	assert metadata_dim == 32, "Metadata dictionary contains indicies > 31"
	
	clip_model, _ = clip.load("ViT-B/32", device="cpu")

	# Initialize generator
	generator = DatasetGenerator(
		block_dict=block_dict,
		metadata_dict=metadata_dict,
		clip_model=clip_model,
		metadata_dim=metadata_dim
	)
	
	# Process all structures
	generator.process()