


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
        
class SigLipEncoder(nn.Module):
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
        # to be continued ...



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
    

