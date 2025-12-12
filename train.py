
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

logger = get_logger(__name__)

class DiffusersTrainer:
	"""Trainer that leverages diffusers training infrastructure for 3D voxel generation."""

	def __init__(
		self,
		model_dir: str,
		from_checkpoint: Optional[str] = None
	):
		self.model_dir = model_dir
		self.from_checkpoint = from_checkpoint

		# Initialize accelerator for distributed training
		self.accelerator = Accelerator(
			gradient_accumulation_steps=1,
			mixed_precision="fp16",  # Enable mixed precision
		)

		# Initialize models
		self.unet = UNet3DConditional()
		self.struct_head = MCStructEmbedHead(self.unet)

		# Diffusers noise scheduler (DDPM)
		self.noise_scheduler = DDPMScheduler(
			num_train_timesteps=1000,
			beta_start=0.0001,
			beta_end=0.02,
			prediction_type="epsilon",  # UNet predicts noise (better training stability)
		)

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

		# EMA model for stable training (create after accelerator.prepare)
		self.ema_unet = EMAModel(self.unet, inv_gamma=1.0, power=2/3, max_value=0.9999)

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
		"""Single training step: encode blocks, add noise, denoise, classify."""
		block_ids = batch['block_ids']  # (B, 1, H, W, D)
		metadata_flags = batch['metadata_flags']  # (B, M, H, W, D)
		text_embeddings = batch['text_embedding']  # (B, 77, 512)

		B = block_ids.shape[0]
		device = block_ids.device

		# Encode blocks and metadata into latent space: (B, 1, H, W, D) + (B, M, H, W, D) -> (B, 32, H, W, D)
		# Initially will be noise (no block embeddings trained)
		# but after several steps should approximate structure features
		block_logits, metadata_logits = self.struct_head.logits_from_outputs(block_ids, metadata_flags)
		encoded_blocks = self.struct_head.backward_to_features(block_logits, metadata_logits)
		
		# Generate random noise with same shape as encoded blocks
		noise = torch.randn_like(encoded_blocks)
		
		# Sample random timesteps for diffusion
		timesteps = torch.randint(
			0, self.noise_scheduler.config.num_train_timesteps,
			(B,), device=device
		)
		
		# Add noise to the encoded block structure
		noisy_input = self.noise_scheduler.add_noise(encoded_blocks, noise, timesteps)
		
		# UNet predicts the noise (epsilon prediction)
		noise_pred = self.unet(noisy_input, timesteps, text_embeddings)

		# Loss 1: Denoising loss (predict the noise that was added)
		denoise_loss = F.mse_loss(noise_pred, noise)
		
		# For classification, reconstruct denoised latent from noise prediction
		# x_0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon_pred) / sqrt(alpha_bar_t)
		alpha_prod_t = self.noise_scheduler.alphas_cumprod[timesteps]
		alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1, 1)  # Reshape for broadcasting
		denoised_latent = (noisy_input - torch.sqrt(1 - alpha_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)

		# Get logits from struct_head using denoised latent
		block_logits, meta_logits = self.struct_head.forward_from_features(denoised_latent)

		# Loss 2: Block classification loss
		K, H, W, D = block_logits.shape[1:]
		logits_flat = block_logits.permute(0, 2, 3, 4, 1).reshape(-1, K)
		block_ids_flat = block_ids.squeeze(1).reshape(-1).long()
		block_loss = F.cross_entropy(logits_flat, block_ids_flat)

		# Loss 3: Metadata binary loss
		meta_loss = F.binary_cross_entropy_with_logits(meta_logits, metadata_flags.float())

		# Combine losses with weights
		# denoise_loss as regularization (lower weight), block+meta losses are primary
		loss = 0.1 * denoise_loss + 0.5 * block_loss + 50 * meta_loss

		# Backpropagate
		self.accelerator.backward(loss)

		# Clip gradients
		if self.accelerator.sync_gradients:
			# Combine all model parameters for gradient clipping
			all_params = list(self.unet.parameters()) + list(self.struct_head.parameters())
			self.accelerator.clip_grad_norm_(all_params, max_norm=1.0)

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
		num_epochs: int = 5,
		batch_size: int = 4,
		max_samples: Optional[int] = None,
		gradient_accumulation_steps: int = 1,
	):
		"""Main training loop using diffusers-style training."""

		# Create dataset and dataloader
		dataset = MCStructureDataset(dataset_dir, max_samples)

		def collate_fn(batch):
			"""Custom collate function to pad structures to batch max size (ceiled to multiple of 8)."""
			# Find maximum dimensions in the batch
			max_H = max(sample['block_ids'].shape[1] for sample in batch)
			max_W = max(sample['block_ids'].shape[2] for sample in batch)
			max_D = max(sample['block_ids'].shape[3] for sample in batch)

			# Ceil to nearest multiple of 8
			target_H = ((max_H + 7) // 8) * 8
			target_W = ((max_W + 7) // 8) * 8
			target_D = ((max_D + 7) // 8) * 8

			# First pass: identify valid samples (those with valid 3D dimensions)
			valid_samples = []
			for sample in batch:
				block_ids = sample['block_ids']
				B, H, W, D = block_ids.shape

				# Skip samples with invalid dimensions
				if H <= 0 or W <= 0 or D <= 0:
					tqdm.write(f"Skipping sample with invalid dimensions: {(B, H, W, D)}")
					continue

				valid_samples.append(sample)

			# If no valid samples, skip this batch
			if not valid_samples:
				return None

			# Second pass: process valid samples only
			collated = {}
			for key in batch[0].keys():
				if key in ['block_ids', 'metadata_flags']:
					processed_tensors = []
					for sample in valid_samples:
						tensor = sample[key]  # Shape: (1, H, W, D) for block_ids, (M, H, W, D) for metadata_flags
						B, H, W, D = tensor.shape

						# Calculate padding needed
						pad_h = target_H - H  # Pad only upward (keep grounded)
						pad_w = target_W - W  # Pad equally left/right
						pad_d = target_D - D  # Pad equally front/back

						# For Y (height): pad only on top (pad_H_before = 0, pad_H_after = pad_h)
						pad_H_before = 0
						pad_H_after = pad_h

						# For X (width) and Z (depth): pad equally on both sides
						pad_W_before = pad_w // 2
						pad_W_after = pad_w - pad_W_before
						pad_D_before = pad_d // 2
						pad_D_after = pad_d - pad_D_before

						# Apply padding: (pad_D_before, pad_D_after, pad_W_before, pad_W_after, pad_H_before, pad_H_after, pad_B_before, pad_B_after)
						padding = (pad_D_before, pad_D_after, pad_W_before, pad_W_after, pad_H_before, pad_H_after, 0, 0)
						result = torch.nn.functional.pad(tensor, padding)

						processed_tensors.append(result)

					collated[key] = torch.stack(processed_tensors)
				else:
					# Stack other tensors normally (text embeddings) - only for valid samples
					collated[key] = torch.stack([sample[key] for sample in valid_samples])
			return collated

		dataloader = DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=True,
			num_workers=0,  # Set to 0 - nested collate_fn can't be pickled for multiprocessing
			pin_memory=True,
			collate_fn=collate_fn,
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
				# Skip invalid batches (collate_fn returned None)
				if batch is None:
					continue

				# Training step
				loss_dict = self.train_step(batch)
				epoch_losses.append(loss_dict["loss"])

				global_step += 1

				# Log progress
				if step % 100 == 0 and self.accelerator.is_main_process:
					avg_loss = np.mean(epoch_losses[-100:]) if len(epoch_losses) >= 100 else np.mean(epoch_losses)
					tqdm.write(f"Step {global_step}: loss={avg_loss:.4f}, denoise={loss_dict['denoise_loss']:.4f}, block={loss_dict['block_loss']:.4f}, meta={loss_dict['meta_loss']:.4f}")

			self.save_checkpoint(global_step)

			# End of epoch logging
			if self.accelerator.is_main_process:
				avg_epoch_loss = np.mean(epoch_losses)
				tqdm.write(f"Epoch {epoch+1}/{num_epochs} completed. Average loss: {avg_epoch_loss:.4f}")

		# Save final model
		# self.save_checkpoint(global_step)
		logger.info("Training completed!")

def main():
	parser = argparse.ArgumentParser(description="Train 3D Minecraft Structure Generation Model")
	parser.add_argument("--dataset_dir", type=str, default="./prepared", help="Path to dataset directory")
	parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
	parser.add_argument("--from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
	parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
	parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
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
		max_samples=args.max_samples,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
	)

if __name__ == "__main__":
	main()
		
