"""
Generation script using diffusers pipeline for 3D Minecraft structure generation.
Replaces the custom generation loop in main.py with diffusers' professional inference pipeline.
"""

import os
import clip
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, Any, Tuple
import argparse

# Diffusers imports for generation
from diffusers import DiffusionPipeline, DDPMScheduler
from accelerate import Accelerator

from model import get_text_embedding, print_slices, UNet3DConditional
from struct_head import MCStructEmbedHead

#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

class MCStructurePipeline(DiffusionPipeline):
	"""
	Custom diffusers pipeline for 3D Minecraft structure generation.
	Wraps your custom UNet and struct head in a diffusers-compatible interface.
	"""

	def __init__(
		self,
		unet: UNet3DConditional,
		struct_head: MCStructEmbedHead,
		scheduler: DDPMScheduler,
		clip_model,
		device: str = "cpu",
	):
		super().__init__()
		self.register_modules(
			unet=unet,
			struct_head=struct_head,
			scheduler=scheduler,
		)
		self.clip_model = clip_model
		self._device = torch.device(device)

	@property
	def device(self):
		"""Return the device of the pipeline."""
		return self._device

	@torch.no_grad()
	def __call__(
		self,
		prompt: str,
		height: int = 16,
		width: int = 16,
		depth: int = 16,
		num_inference_steps: int = 50,
		guidance_scale: float = 5.0,
		num_images_per_prompt: int = 1,
		latents: Optional[torch.Tensor] = None,
		return_dict: bool = True,
		**kwargs
	) -> Dict[str, Any]:
		"""
		Generate 3D Minecraft structures using diffusers pipeline.

		Args:
			prompt: Text description of the structure
			height/width/depth: Spatial dimensions of the structure
			num_inference_steps: Number of denoising steps
			guidance_scale: Classifier-free guidance scale
			num_images_per_prompt: Number of structures to generate
			latents: Optional pre-generated noise latents
			return_dict: Whether to return a dictionary or tuple

		Returns:
			Dictionary containing generated structures and metadata
		"""

		# Validate inputs
		if num_images_per_prompt > 1:
			raise ValueError("num_images_per_prompt > 1 not yet supported for 3D generation")

		# Set dimensions
		batch_size = num_images_per_prompt
		shape = (batch_size, self.unet.in_channels, height, width, depth)

		# Get text embeddings using CLIP (token-level)
		text_emb = get_text_embedding(self.clip_model, [prompt], self.device)
		print("text_emb shape (tokens):", text_emb.shape)  # torch.Size([1, 77, 512])
		
		if guidance_scale > 1.0:
			# For classifier-free guidance, we need unconditional embeddings
			uncond_emb = get_text_embedding(self.clip_model, [""], self.device)
			# Concatenate along batch dimension for both conditional and unconditional
			text_emb = torch.cat([uncond_emb, text_emb], dim=0)  # (2, 77, 512)

		# Generate latents if not provided
		if latents is None:
			latents = torch.randn(shape, device=self.device, dtype=next(self.unet.parameters()).dtype)
		else:
			latents = latents.to(device=self.device, dtype=next(self.unet.parameters()).dtype)

		# Set scheduler timesteps
		self.scheduler.set_timesteps(num_inference_steps, device=self.device)

		# Denoising loop (diffusers style)
		for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Generating")):
			# Create timesteps tensor for the batch
			timesteps = torch.tensor([t] * batch_size, device=self.device, dtype=torch.long)

			# Expand latents and timesteps for classifier-free guidance
			latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
			timesteps_input = torch.cat([timesteps] * 2) if guidance_scale > 1.0 else timesteps
			latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

			# Predict noise
			if guidance_scale > 1.0:
				# Separate conditional and unconditional predictions
				noise_pred_uncond, noise_pred_cond = torch.chunk(
					self.unet(latent_model_input, timesteps_input, text_emb), 2
				)
				# Apply classifier-free guidance
				noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
			else:
				noise_pred = self.unet(latent_model_input, timesteps_input, text_emb)

			# Compute previous noisy sample
			latents = self.scheduler.step(noise_pred, t, latents).prev_sample

		# Use final denoised latents as features
		features = latents

		# Convert features to block IDs and metadata using struct head
		block_logits, meta_logits = self.struct_head.forward_from_features(features)
		block_ids, meta_bits = self.struct_head.logits_to_outputs(block_logits, meta_logits)

		if return_dict:
			return { "block_ids": block_ids, "meta_bits": meta_bits }
		else:
			return (block_ids, meta_bits)

def create_diffusers_pipeline(
	model_path: Optional[str] = None,
	device: str = DEVICE
) -> MCStructurePipeline:
	"""
	Create a diffusers pipeline from saved models.

	Args:
		model_path: Path to directory containing saved models (unet.pth, struct_head.pth)
		device: Device to load models on

	Returns:
		MCStructurePipeline ready for generation
	"""
	# Initialize models
	unet = UNet3DConditional()
	struct_head = MCStructEmbedHead(unet, temperature=0.01)

	# Load CLIP model
	clip_model, _ = clip.load("ViT-B/32", device=device)

	# Create diffusers scheduler
	scheduler = DDPMScheduler(
		num_train_timesteps=1000,
		beta_start=0.0001,
		beta_end=0.02,
		prediction_type="epsilon",
	)

	# Move to device
	unet = unet.to(device)
	struct_head = struct_head.to(device)

	# Load saved weights if provided
	if model_path is not None and os.path.exists(model_path):
		checkpoint = torch.load(model_path, map_location=device)
		
		# Load UNet and StructHead state dicts
		if 'unet_state_dict' in checkpoint:
			unet.load_state_dict(checkpoint['unet_state_dict'])
		if 'struct_head_state_dict' in checkpoint:
			struct_head.load_state_dict(checkpoint['struct_head_state_dict'])
		
		print(f"Loaded checkpoint from {model_path} (step {checkpoint.get('step', 'unknown')})")

	# Create pipeline
	pipeline = MCStructurePipeline(
		unet=unet,
		struct_head=struct_head,
		scheduler=scheduler,
		clip_model=clip_model,
		device=device,
	)

	return pipeline

def main():
	"""Demo generation using diffusers pipeline."""
	parser = argparse.ArgumentParser(description="Generate 3D Minecraft Structures with Diffusers")
	parser.add_argument("--model_path", type=str, default="./checkpoints/checkpoint-25192/checkpoint.pt", help="Path to trained model weights")
	parser.add_argument("--prompt", type=str, default="Oak Tree", help="Text description of structure")
	parser.add_argument("--steps", type=int, default=100, help="Number of inference steps")
	parser.add_argument("--guidance_scale", type=float, default=1.0, help="Classifier-free guidance scale")
	parser.add_argument("--height", type=int, default=16, help="Structure height")
	parser.add_argument("--width", type=int, default=16, help="Structure width")
	parser.add_argument("--depth", type=int, default=16, help="Structure depth")

	args = parser.parse_args()

	print("Creating diffusers pipeline...")
	pipeline = create_diffusers_pipeline(args.model_path, DEVICE)

	print(f"Generating structure: '{args.prompt}'")
	output = pipeline(
		prompt=args.prompt,
		height=args.height,
		width=args.width,
		depth=args.depth,
		num_inference_steps=args.steps,
		guidance_scale=args.guidance_scale,
	)

	block_ids = output["block_ids"].squeeze(0).squeeze(0)  # (D, H, W)
	meta_bits = output["meta_bits"].squeeze(0)  # (M, D, H, W)
	print(f"Generated structure shape: {block_ids.shape}")
	print(f"Block ID range: {block_ids.min().item()} - {block_ids.max().item()}")

	print_slices(block_ids, meta_bits)

if __name__ == "__main__":
	main()
