
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from typing import Optional, Dict, Any, List
import argparse
import json

# Diffusers imports for training infrastructure
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from accelerate import Accelerator
from accelerate.logging import get_logger

from model import UNet3DConditional
from struct_head import MCStructEmbedHead
from dataset import MCStructureDataset
import clip

logger = get_logger(__name__)

class DiffusersTrainer:
	"""Trainer that leverages diffusers training infrastructure for 3D voxel generation."""

	def __init__(
		self,
		model_dir: str,
		from_checkpoint: Optional[str] = None,
		num_blocks: int = 12,
		meta_dim: int = 4,
		embed_dim: int = 64,
		projector_hidden: int = 128,
	):
		self.model_dir = model_dir
		self.from_checkpoint = from_checkpoint

		# Initialize accelerator for distributed training
		self.accelerator = Accelerator(
			gradient_accumulation_steps=1,
			mixed_precision="fp16",  # Enable mixed precision
		)

		# Initialize models
		self.unet = UNet3DConditional(
			in_channels=32,
			out_channels=32,
			cross_attn=True,
			text_dim=512,  # CLIP embedding dim
		)

		self.struct_head = MCStructEmbedHead(
			self.unet,
			num_blocks=num_blocks,
			meta_dim=meta_dim,
			embed_dim=embed_dim,
			projector_hidden=projector_hidden
		)

		# Diffusers noise scheduler (DDPM)
		self.noise_scheduler = DDPMScheduler(
			num_train_timesteps=1000,
			beta_start=0.0001,
			beta_end=0.02,
			prediction_type="epsilon",  # Predict noise
		)

		# EMA model for stable training
		self.ema_unet = EMAModel(self.unet, inv_gamma=1.0, power=2/3, max_value=0.9999)

		# Optimizer
		self.optimizer = torch.optim.AdamW(
			list(self.unet.parameters()) + list(self.struct_head.parameters()),
			lr=1e-4,
			weight_decay=1e-6,
		)

		# Learning rate scheduler
		self.lr_scheduler = get_scheduler(
			"cosine",
			optimizer=self.optimizer,
			num_warmup_steps=500,
			num_training_steps=100000,  # Will be updated based on actual steps
		)

		# Prepare with accelerator
		self.unet, self.struct_head, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
			self.unet, self.struct_head, self.optimizer, self.lr_scheduler
		)

		# Load checkpoint if specified
		if from_checkpoint:
			self._load_checkpoint(from_checkpoint)

	def _load_checkpoint(self, checkpoint_path: str):
		"""Load model checkpoint."""
		checkpoint = torch.load(checkpoint_path, map_location='cpu')

		# Load model states
		self.unet.load_state_dict(checkpoint['unet_state_dict'])
		self.struct_head.load_state_dict(checkpoint['struct_head_state_dict'])
		self.ema_unet.load_state_dict(checkpoint['ema_unet_state_dict'])

		# Load optimizer and scheduler
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

		logger.info(f"Loaded checkpoint from {checkpoint_path}")

	def save_checkpoint(self, step: int):
		"""Save model checkpoint."""
		checkpoint_path = os.path.join(self.model_dir, f"checkpoint-{step}")
		os.makedirs(checkpoint_path, exist_ok=True)

		# Prepare state dicts
		unet_state = self.accelerator.unwrap_model(self.unet).state_dict()
		struct_head_state = self.accelerator.unwrap_model(self.struct_head).state_dict()
		ema_state = self.ema_unet.state_dict()

		checkpoint = {
			'step': step,
			'unet_state_dict': unet_state,
			'struct_head_state_dict': struct_head_state,
			'ema_unet_state_dict': ema_state,
			'optimizer_state_dict': self.optimizer.state_dict(),
			'scheduler_state_dict': self.lr_scheduler.state_dict(),
			'config': {
				'num_blocks': self.struct_head.num_blocks,
				'meta_dim': self.struct_head.meta_dim,
				'embed_dim': self.struct_head.embed_dim,
				'projector_hidden': self.struct_head.projector_hidden,
			}
		}

		torch.save(checkpoint, os.path.join(checkpoint_path, "checkpoint.pth"))
		logger.info(f"Saved checkpoint to {checkpoint_path}")

	def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
		"""Single training step: denoise + classify blocks and metadata."""
		block_ids = batch['block_ids']  # (B, 1, D, H, W)
		metadata_flags = batch['metadata_flags']  # (B, M, D, H, W)
		text_embeddings = batch['text_embedding']  # (B, 77, 512)

		B = block_ids.shape[0]
		device = block_ids.device

		# Generate random noise as initial unstructured voxels
		noise = torch.randn(B, 32, *block_ids.shape[2:], device=device)
		
		# Sample random timesteps for diffusion
		timesteps = torch.randint(
			0, self.noise_scheduler.config.num_train_timesteps,
			(B,), device=device
		)
		
		# Add noise at timestep t (simulates diffusion forward process)
		noisy_input = self.noise_scheduler.add_noise(noise, noise, timesteps)
		
		# UNet predicts noise and extracts features
		unet_out = self.unet(noisy_input, timesteps, text_embeddings, return_features=True)
		
		if isinstance(unet_out, (tuple, list)):
			noise_pred, features = unet_out
		else:
			raise RuntimeError("UNet.forward did not return (out, features)")

		# Get logits from struct_head
		block_logits, meta_logits = self.struct_head.forward_from_features(features)

		# Loss 1: Denoising loss (predict the noise that was added)
		denoise_loss = F.mse_loss(noise_pred, noise)

		# Loss 2: Block classification loss
		K, D, H, W = block_logits.shape[1:]
		logits_flat = block_logits.permute(0, 2, 3, 4, 1).reshape(-1, K)
		block_ids_flat = block_ids.squeeze(1).reshape(-1).long()
		block_loss = F.cross_entropy(logits_flat, block_ids_flat)

		# Loss 3: Metadata binary loss
		meta_loss = F.binary_cross_entropy_with_logits(meta_logits, metadata_flags)

		# Combine losses with weights
		loss = 0.5 * denoise_loss + 0.3 * block_loss + 0.2 * meta_loss

		# Backpropagate
		self.accelerator.backward(loss)

		# Clip gradients
		if self.accelerator.sync_gradients:
			self.accelerator.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
			self.accelerator.clip_grad_norm_(self.struct_head.parameters(), max_norm=1.0)

		# Optimizer step
		self.optimizer.step()
		self.lr_scheduler.step()
		self.optimizer.zero_grad()

		# Update EMA model
		if self.accelerator.sync_gradients:
			self.ema_unet.step(self.unet.parameters())

		return {
			"loss": loss.item(),
			"denoise_loss": denoise_loss.item(),
			"block_loss": block_loss.item(),
			"meta_loss": meta_loss.item(),
		}

	def train(
		self,
		dataset_dir: str,
		num_epochs: int = 100,
		batch_size: int = 4,
		save_steps: int = 1000,
		max_samples: Optional[int] = None,
		gradient_accumulation_steps: int = 1,
	):
		"""Main training loop using diffusers-style training."""

		# Load CLIP model for dataset
		clip_model, _ = clip.load("ViT-B/32", device=self.accelerator.device)

		# Create dataset and dataloader
		dataset = MCStructureDataset(dataset_dir, clip_model, max_samples)
		dataloader = DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=True,
			num_workers=4,
			pin_memory=True,
		)

		# Prepare dataloader with accelerator
		dataloader = self.accelerator.prepare(dataloader)

		# Calculate total training steps
		total_steps = len(dataloader) * num_epochs // gradient_accumulation_steps

		# Update scheduler with correct number of steps
		self.lr_scheduler = get_scheduler(
			"cosine",
			optimizer=self.optimizer,
			num_warmup_steps=500,
			num_training_steps=total_steps,
		)

		logger.info(f"Starting training for {num_epochs} epochs, {total_steps} total steps")
		logger.info(f"Dataset size: {len(dataset)} samples")

		global_step = 0
		for epoch in range(num_epochs):
			self.unet.train()
			self.struct_head.train()

			epoch_losses = []

			for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
				# Training step
				loss_dict = self.train_step(batch)
				epoch_losses.append(loss_dict["loss"])

				global_step += 1

				# Save checkpoint
				if global_step % save_steps == 0:
					self.save_checkpoint(global_step)

				# Log progress
				if step % 100 == 0 and self.accelerator.is_main_process:
					avg_loss = np.mean(epoch_losses[-100:]) if len(epoch_losses) >= 100 else np.mean(epoch_losses)
					logger.info(f"Step {global_step}: loss={avg_loss:.4f}, denoise={loss_dict['denoise_loss']:.4f}, block={loss_dict['block_loss']:.4f}, meta={loss_dict['meta_loss']:.4f}")

			# End of epoch logging
			if self.accelerator.is_main_process:
				avg_epoch_loss = np.mean(epoch_losses)
				logger.info(f"Epoch {epoch+1}/{num_epochs} completed. Average loss: {avg_epoch_loss:.4f}")

		# Save final model
		self.save_checkpoint(global_step)
		logger.info("Training completed!")

def main():
	parser = argparse.ArgumentParser(description="Train 3D Minecraft Structure Generation Model")
	parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset directory")
	parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
	parser.add_argument("--from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
	parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
	parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
	parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N steps")
	parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use")
	parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")

	args = parser.parse_args()

	# Create output directory
	os.makedirs(args.output_dir, exist_ok=True)

	# Initialize trainer
	trainer = DiffusersTrainer(
		model_dir=args.output_dir,
		from_checkpoint=args.from_checkpoint,
	)

	# Start training
	trainer.train(
		dataset_dir=args.dataset_dir,
		num_epochs=args.num_epochs,
		batch_size=args.batch_size,
		save_steps=args.save_steps,
		max_samples=args.max_samples,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
	)

if __name__ == "__main__":
	main()
		
