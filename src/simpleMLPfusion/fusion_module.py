import torch
import torch.nn as nn
from .parameters import IMAGE_EMB_DIM, TEXT_EMB_DIM, FUSION_HIDDEN_DIM, OUTPUT_TEXT_DIM, PREFIX_LENGTH

class FusionModule(nn.Module):
    def __init__(self,
                 image_dim=IMAGE_EMB_DIM,
                 text_dim=TEXT_EMB_DIM,
                 fusion_hidden_dim=FUSION_HIDDEN_DIM,
                 output_text_dim=OUTPUT_TEXT_DIM,
                 prefix_length=PREFIX_LENGTH,
                 llama_hidden_size=None):
        """
        :param llama_hidden_size: Must be provided (usually read from the LLaMA model config)
        """
        super(FusionModule, self).__init__()
        if llama_hidden_size is None:
            raise ValueError("llama_hidden_size must be provided")
        self.llama_hidden_size = llama_hidden_size  # store as an attribute
        total_input_dim=(image_dim * 2 + text_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_input_dim, fusion_hidden_dim),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim, output_text_dim)
        )

        self.prefix_length = prefix_length
        # Project predicted text embedding into a sequence of soft prefix tokens
        self.prefix_projector = nn.Linear(output_text_dim, prefix_length * llama_hidden_size)

    def forward(self, image_emb1, image_emb2, text_emb1):
        fusion_input = torch.cat([image_emb1, image_emb2, text_emb1], dim=-1)
        predicted_text_emb = self.fusion_mlp(fusion_input)  # shape: [batch, output_text_dim]
        prefix_tokens = self.prefix_projector(predicted_text_emb)  # shape: [batch, prefix_length * llama_hidden_size]
        # Reshape to [batch, prefix_length, llama_hidden_size]
        prefix_tokens = prefix_tokens.view(-1, self.prefix_length, self.llama_hidden_size)
        return prefix_tokens
