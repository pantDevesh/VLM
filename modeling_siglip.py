


from typing import Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6, # Layer norm epsilon: A small value added to the denominator for numerical stability
        attention_dropout_rate=0.0,
        num_image_tokens: int = None, 
        **kwargs
        ) -> None:
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout_rate = attention_dropout_rate
        self.num_image_tokens = num_image_tokens
      
      
class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.attn = SiglipAttention(config)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states):
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.attn(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        return hidden_states
        
class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipVisionLayer(config) for _ in range(config.num_hidden_layers)])
         
    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embed = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid"
        )
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.postition_embeddings = nn.Embedding(self.num_positions, self.embed_dim)
        
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)),
                                persistent=False)
         
    def forward(self, pixel_values):
        _, _, height, width = pixel_values.shape
        
        patch_embeds = self.patch_embed(pixel_values) # [Batch_size, hidden_size, num_patches, num_patches]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2) # [Batch_size, num_patches, hidden_size]
        
        embeddings = patch_embeds + self.postition_embeddings(self.position_ids)
        
        return embeddings
        
                             


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SigLipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        
    def forward(self, pixel_values):
        embeddings = self.embeddings(pixel_values)
        hidden_states = self.encoder(embeddings)
        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states
        
          
    
class SiglipVisionModel(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
        
    def forward(self, pixel_values) -> Tuple:
        # [Batch_size, num_channels, height, weight] -> [Batch_size, num_patches, Embed_dim]
        return self.vision_model(pixel_values)
    

