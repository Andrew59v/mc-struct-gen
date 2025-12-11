
import clip
import torch
import numpy as np
from tqdm import tqdm

from model import get_text_embedding, UNet3DConditional
from struct_head import MCStructEmbedHead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class NoiseScheduler:
	def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
		self.num_timesteps = num_timesteps
		self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
		self.alphas = 1.0 - self.betas
		self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
		self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alpha_cumprod[:-1]])
		self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
		self.sqrt_alphas_cumprod = torch.sqrt(self.alpha_cumprod)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
		
	def to(self, device):
		self.betas = self.betas.to(device)
		self.alphas = self.alphas.to(device)
		self.alpha_cumprod = self.alpha_cumprod.to(device)
		self.alpha_cumprod_prev = self.alpha_cumprod_prev.to(device)
		self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
		self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
		self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
		return self

def generate_structure(clip_model, unet, scheduler, shape, prompt, steps=10):
	text_emb = get_text_embedding(clip_model, [prompt,], device=DEVICE)
	notext_emb = get_text_embedding(clip_model, ["",], device=DEVICE)
	guidance_scale = 5.0

	latents = torch.randn(shape, device=DEVICE)
	for t in tqdm(reversed(range(scheduler.num_timesteps - 1, -1, -(scheduler.num_timesteps // steps))), desc="Generating"):
		timesteps = torch.tensor([t] * shape[0], device=DEVICE)

		pred_noise_cond = unet(latents, timesteps, text_emb)
		pred_noise_uncond = unet(latents, timesteps, notext_emb)
		noise_pred = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
		
		alpha_t, alpha_cumprod_t, beta_t = scheduler.alphas[t], scheduler.alpha_cumprod[t], scheduler.betas[t]
		latents = (1/torch.sqrt(alpha_t)) * (latents - ((1-alpha_t)/torch.sqrt(1-alpha_cumprod_t)) * noise_pred)
		if t > 0: latents += torch.sqrt(beta_t) * torch.randn_like(latents)

	_, features = unet(latents, timesteps, text_emb, return_features=True)
	return features

if __name__ == "__main__":
	shape = (1, 32, 8, 8, 8)

	unet = UNet3DConditional(in_channels=shape[1], out_channels=shape[1]).to(DEVICE)
	head = MCStructEmbedHead(unet, num_blocks=12, meta_dim=4, feat_channels=None, embed_dim=shape[1]*2, projector_hidden=128).to(DEVICE)
	clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
	scheduler = NoiseScheduler(num_timesteps=1000).to(DEVICE)
	print("Models loaded successfully.")
	
	features = generate_structure(clip_model, unet, scheduler, shape, "A simple house", steps=50)
	block_id, meta_bits, block_logits, meta_logits = head.forward_from_features(features, return_logits=True)

	print(block_id)