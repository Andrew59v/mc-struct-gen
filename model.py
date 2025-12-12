
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.embeddings import Timesteps

import clip

# ----------------------------- Utils -----------------
def get_text_embedding(clip_model, prompts: list, device = "cpu") -> torch.Tensor:
		with torch.no_grad():
				text_tokens = clip.tokenize(prompts, truncate=True)
				text_tokens = text_tokens.to(device)
				
				# We need to access the intermediate token embeddings
				# CLIP's encode_text processes tokens through transformer and returns pooled output
				# To get token-level embeddings, we need to access the transformer directly
				
				# Get token embeddings by passing through text encoder's transformer
				text = clip_model.token_embedding(text_tokens)  # (B, L, 512)
				text = text + clip_model.positional_embedding
				text = text.permute(1, 0, 2)  # (L, B, 512)
				text = clip_model.transformer(text)
				text = text.permute(1, 0, 2)  # (B, L, 512)
				text = clip_model.ln_final(text)
				return text.float()
		
def print_slices(block_ids, metadata_flags):
	"""
	Prints vertical slices of the structure to console.
	Just for debug purposes
	"""
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

# ----------------------------- Basic Blocks -------------------------------
class ResidualBlock3D(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, cond_dim: int, groups: int = 4, dropout: float = 0.0):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels

		self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=1e-6)
		self.act = nn.SiLU()
		self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

		self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=1e-6)
		self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

		if in_channels != out_channels:
			self.skip_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
		else:
			self.skip_conv = nn.Identity()

		self.cond_proj = nn.Linear(cond_dim, out_channels)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
		h = self.conv1(self.act(self.norm1(x)))
		if cond is not None:
			proj: torch.Tensor = self.cond_proj(cond)
			cond_out = proj.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
			h = h + cond_out
		h = self.conv2(self.dropout(self.act(self.norm2(h))))
		return h + self.skip_conv(x)

class DownBlock3D(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, cond_dim: int,
				 num_res_blocks: int = 1, groups: int = 4, dropout: float = 0.0):
		super().__init__()
		self.blocks = nn.ModuleList()
		cur_in = in_channels
		for _ in range(num_res_blocks):
			blk = ResidualBlock3D(cur_in, out_channels, cond_dim, groups=groups, dropout=dropout)
			self.blocks.append(blk)
			cur_in = out_channels
		self.downsample = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

	def forward(self, x: torch.Tensor, cond: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
		skips = []
		h = x
		for blk in self.blocks:
			h = blk(h, cond=cond)
			skips.append(h)
		return self.downsample(h), skips

class UpBlock3D(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, cond_dim: int,
				 num_res_blocks: int = 1, groups: int = 4, dropout: float = 0.0):
		super().__init__()
		self.blocks = nn.ModuleList()
		self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
		in_ch_for_blocks = out_channels + out_channels
		for _ in range(num_res_blocks):
			blk = ResidualBlock3D(in_ch_for_blocks, out_channels, cond_dim, groups=groups, dropout=dropout)
			self.blocks.append(blk)

	def forward(self, x: torch.Tensor, skips: List[torch.Tensor], cond: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
		h = self.upsample(x)
		for blk, skip in zip(self.blocks, skips[::-1]):
			h = torch.cat([h, skip], dim=1)
			h = blk(h, cond=cond)
		return h

# ----------------------------- Cross-Attention ---------------------------
class CrossAttention3D(nn.Module):
	def __init__(self, query_dim: int, context_dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
		super().__init__()
		self.heads = heads
		self.scale = dim_head ** -0.5
		self.inner_dim = dim_head * heads

		# projections
		self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
		self.to_k = nn.Linear(context_dim, self.inner_dim, bias=False)
		self.to_v = nn.Linear(context_dim, self.inner_dim, bias=False)
		self.to_out = nn.Linear(self.inner_dim, query_dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
		b, cq, d, h, w = x.shape
		n = d * h * w
		# flatten spatial
		x_flat = x.view(b, cq, n).permute(0, 2, 1)  # (B, N, Cq)

		# ensure context is (B, L, C)
		if context.dim() == 2:
			# single vector -> single token
			context = context.unsqueeze(1)
		b2, l, _ = context.shape
		assert b2 == b, f"Batch mismatch between x and context: {b} vs {b2}"

		q: torch.Tensor = self.to_q(x_flat)  # (B, N, inner)
		k: torch.Tensor = self.to_k(context)  # (B, L, inner)
		v: torch.Tensor = self.to_v(context)

		# split heads
		q = q.view(b, n, self.heads, -1).permute(0, 2, 1, 3)  # (B, heads, N, dim_head)
		k = k.view(b, l, self.heads, -1).permute(0, 2, 1, 3)  # (B, heads, L, dim_head)
		v = v.view(b, l, self.heads, -1).permute(0, 2, 1, 3)

		attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, N, L)
		attn = torch.softmax(attn, dim=-1)
		attn: torch.Tensor = self.dropout(attn)

		out = torch.matmul(attn, v)  # (B, heads, N, dim_head)
		out = out.permute(0, 2, 1, 3).contiguous().view(b, n, self.inner_dim)
		out: torch.Tensor = self.to_out(out)  # (B, N, Cq)

		out = out.permute(0, 2, 1).view(b, cq, d, h, w)
		return out

# ----------------------------- Cross-Attn Blocks ------------------------
class CrossAttnBlock3D(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, cond_dim: int, context_dim: int, *,
				 num_heads: int = 8, dim_head: int = 64, groups: int = 8, dropout: float = 0.0):
		super().__init__()
		self.res = ResidualBlock3D(in_channels, out_channels, cond_dim, groups=groups, dropout=dropout)
		self.attn = CrossAttention3D(out_channels, context_dim, heads=num_heads, dim_head=dim_head, dropout=dropout)

	def forward(self, x: torch.Tensor, cond: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
		h = self.res(x, cond=cond)
		if context is not None:
			h = self.attn(h, context)
		return h

class CrossAttnDownBlock3D(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, cond_dim: int, context_dim: int,
				 num_res_blocks: int = 1, num_heads: int = 8, dim_head: int = 64, groups: int = 4, dropout: float = 0.0):
		super().__init__()
		self.blocks = nn.ModuleList()
		cur_in = in_channels
		for _ in range(num_res_blocks):
			blk = CrossAttnBlock3D(cur_in, out_channels, cond_dim, context_dim, num_heads=num_heads, dim_head=dim_head, groups=groups, dropout=dropout)
			self.blocks.append(blk)
			cur_in = out_channels
		self.downsample = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

	def forward(self, x: torch.Tensor, cond: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
		skips = []
		h = x
		for blk in self.blocks:
			h = blk(h, cond=cond, context=context)
			skips.append(h)
		return self.downsample(h), skips

class CrossAttnUpBlock3D(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, cond_dim: int, context_dim: int,
				 num_res_blocks: int = 1, num_heads: int = 8, dim_head: int = 64, groups: int = 4, dropout: float = 0.0):
		super().__init__()
		self.blocks = nn.ModuleList()
		self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
		in_ch_for_blocks = out_channels + out_channels
		for _ in range(num_res_blocks):
			blk = CrossAttnBlock3D(in_ch_for_blocks, out_channels, cond_dim, context_dim, num_heads=num_heads, dim_head=dim_head, groups=groups, dropout=dropout)
			self.blocks.append(blk)

	def forward(self, x: torch.Tensor, skips: List[torch.Tensor], cond: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
		h = self.upsample(x)
		for blk, skip in zip(self.blocks, skips[::-1]):
			h = torch.cat([h, skip], dim=1)
			h = blk(h, cond=cond, context=context)
		return h

# ----------------------------- Token Expander ---------------------------
class TokenExpander(nn.Module):
	def __init__(self, in_dim: int = 512, num_tokens: int = 8, token_dim: int = 256):
		super().__init__()
		self.num_tokens = num_tokens
		self.token_dim = token_dim
		self.proj = nn.Linear(in_dim, num_tokens * token_dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = self.proj(x)
		out = out.view(x.shape[0], self.num_tokens, self.token_dim)
		return out

# ----------------------------- UNet 3D w/ optional cross-attn ------------
class UNet3DConditional(nn.Module):
	def __init__(
		self,
		in_channels: int = 32,
		out_channels: int = 32,
		down_block_types: Tuple[str, ...] = ("CrossAttnDownBlock3D", "DownBlock3D"),
		up_block_types: Tuple[str, ...] = ("UpBlock3D", "CrossAttnUpBlock3D"),
		block_out_channels: Tuple[int, ...] = (64, 128),
		num_res_blocks: int = 1,
		text_dim: int = 512,
		time_emb_dim: int = 512,
		groups: int = 4,
		dropout: float = 0.0,
		# cross-attn related
		cross_attn: bool = True,
		cross_attn_heads: int = 8,
		cross_attn_dim_head: int = 64,
		token_length: int = 8,
	):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels

		# initial conv
		self.init_conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

		# time & text embedding processors
		self.time_embed = Timesteps(
			num_channels=512,
			flip_sin_to_cos=False,
			downscale_freq_shift=1.0,
			scale=1
		)

		self.text_proj = nn.Sequential(nn.Linear(text_dim, time_emb_dim), nn.SiLU())

		# token expander for single-vector text embeddings
		self.cross_attn = cross_attn
		self.token_length = token_length
		self.token_dim = min(time_emb_dim, cross_attn_dim_head * cross_attn_heads) if cross_attn else 0
		if cross_attn:
			self.token_expander = TokenExpander(in_dim=text_dim, num_tokens=token_length, token_dim=self.token_dim)
		else:
			self.token_expander = None

		assert len(down_block_types) == len(up_block_types), "down_block_types and up_block_types must have same number of stages"
		assert len(block_out_channels) == len(down_block_types), "block_out_channels"

		# build down blocks
		self.down_blocks = nn.ModuleList()
		in_ch = in_channels
		for i, block_type in enumerate(down_block_types):
			out_ch = block_out_channels[i]
			if cross_attn and block_type == "CrossAttnDownBlock3D":
				blk = CrossAttnDownBlock3D(in_ch, out_ch, time_emb_dim, self.token_dim, num_res_blocks=num_res_blocks,
										   num_heads=cross_attn_heads, dim_head=cross_attn_dim_head, groups=groups, dropout=dropout)
			elif not cross_attn or block_type == "DownBlock3D":
				blk = DownBlock3D(in_ch, out_ch, time_emb_dim, num_res_blocks=num_res_blocks,
										   groups=groups, dropout=dropout)
			else:
				raise ValueError(f"Unknown down block type: {block_type}")
			self.down_blocks.append(blk)
			in_ch = out_ch

		# bottleneck
		self.mid_block1 = ResidualBlock3D(in_ch, in_ch, time_emb_dim, groups=groups, dropout=dropout)
		self.mid_attn = CrossAttention3D(in_ch, self.token_dim, heads=cross_attn_heads, dim_head=cross_attn_dim_head, dropout=dropout) if cross_attn else None
		self.mid_block2 = ResidualBlock3D(in_ch, in_ch, time_emb_dim, groups=groups, dropout=dropout)

		# build up blocks (mirror order)
		self.up_blocks = nn.ModuleList()
		# up_block_types provided in stage order from top->bottom; we need reversed to match skip ordering
		for i, (out_ch, block_type) in enumerate(zip(reversed(block_out_channels), up_block_types)):
			if cross_attn and block_type == "CrossAttnUpBlock3D":
				blk = CrossAttnUpBlock3D(in_ch, out_ch, time_emb_dim, self.token_dim, num_res_blocks=num_res_blocks,
										 num_heads=cross_attn_heads, dim_head=cross_attn_dim_head, groups=groups, dropout=dropout)
			elif not cross_attn or block_type == "UpBlock3D":
				blk = UpBlock3D(in_ch, out_ch, time_emb_dim, num_res_blocks=num_res_blocks,
										 groups=groups, dropout=dropout)
			else:
				raise ValueError(f"Unknown up block type: {block_type}")
			self.up_blocks.append(blk)
			in_ch = out_ch

		# final conv
		self.final_norm = nn.GroupNorm(num_groups=groups, num_channels=in_ch, eps=1e-6)
		self.final_act = nn.SiLU()
		self.final_conv = nn.Conv3d(in_ch, out_channels, kernel_size=3, padding=1)

		self._init_weights()

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
				nn.init.kaiming_normal_(m.weight, a=0.0, mode="fan_in")
				if m.bias is not None:
					nn.init.zeros_(m.bias)
			elif isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.zeros_(m.bias)

	def _prepare_context(self, encoder_hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
		if encoder_hidden_states is None:
			return None
		if encoder_hidden_states.dim() == 3:
			return encoder_hidden_states
		if encoder_hidden_states.dim() == 2:
			if self.token_expander is not None:
				return self.token_expander(encoder_hidden_states)
			else:
				return encoder_hidden_states.unsqueeze(1)
		raise ValueError("encoder_hidden_states must be shape (B, C) or (B, L, C)")

	def forward(self, x: torch.Tensor, timesteps: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
		assert x.dim() == 5, "Input must be (B, C, D, H, W)"
		# initial conv
		h = self.init_conv(x)

		# prepare conditioning
		t_emb = self.time_embed(timesteps)
		# text_proj used only to create a global text contribution to residual blocks
		text_emb_global = None
		if encoder_hidden_states is not None and encoder_hidden_states.dim() == 2:
			text_emb_global = self.text_proj(encoder_hidden_states)
		cond = t_emb + (text_emb_global if text_emb_global is not None else 0)

		# prepare context tokens for cross-attn
		context = self._prepare_context(encoder_hidden_states)

		# Down path
		all_skips: List[List[torch.Tensor]] = []
		h_cur = h
		for blk in self.down_blocks:
			h_cur, skips = blk(h_cur, cond=cond, context=context)
			all_skips.append(skips)

		# bottleneck
		h_cur = self.mid_block1(h_cur, cond=cond)
		if self.mid_attn is not None and context is not None:
			h_cur = h_cur + self.mid_attn(h_cur, context)
		h_cur = self.mid_block2(h_cur, cond=cond)

		# Up path
		for blk, skips in zip(self.up_blocks, reversed(all_skips)):
			h_cur = blk(h_cur, skips, cond=cond, context=context)

		h_cur = self.final_act(self.final_norm(h_cur))
		out = self.final_conv(h_cur)

		# Return denoised latent representation (B, 32, D, H, W)
		return out


# ----------------------------- Example / Smoke test ---------------------
if __name__ == "__main__":
	# build model where we explicitly provide block type lists (mirrors diffusers style)
	model = UNet3DConditional()

	B = 1
	C = 32
	D, H, W = 16, 16, 16
	x = torch.randn(B, C, D, H, W)
	timesteps = torch.randint(0, 1000, (B,))
	text_emb = torch.randn(B, 10, 512)

	out = model(x, timesteps, text_emb)
	print("out shape", out.shape)
