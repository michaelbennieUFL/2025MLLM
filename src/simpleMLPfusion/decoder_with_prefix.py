import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from .parameters import PREFIX_LENGTH

class LlamaDecoderWithPrefix(nn.Module):
    def __init__(self, llama_model, prefix_length=PREFIX_LENGTH):
        """
        :param llama_model: a pretrained LlamaForCausalLM model (with resized token embeddings if needed)
        """
        super(LlamaDecoderWithPrefix, self).__init__()
        self.llama = llama_model
        self.prefix_length = prefix_length
        self.embed_tokens = self.llama.model.embed_tokens  # LLaMA's token embeddings

    def forward(self, prefix_embeds, target_input_ids):
        """
        During training, we prepend the soft prefix embeddings to the token embeddings
        for the target caption and compute the LM loss.
        We mask out the prefix tokens in the labels by setting them to -100.
        """
        # Get target token embeddings [batch, seq_len, hidden_size]
        target_embeds = self.embed_tokens(target_input_ids)
        # Concatenate soft prefix and target token embeddings
        inputs_embeds = torch.cat([prefix_embeds, target_embeds], dim=1)
        # Build an attention mask (all ones)
        attention_mask = torch.ones(inputs_embeds.size()[:-1], device=inputs_embeds.device)

        # Create labels for the entire sequence:
        # For prefix positions, use -100 (ignore index); for target tokens, use target_input_ids.
        batch_size = target_input_ids.shape[0]
        prefix_labels = torch.full((batch_size, self.prefix_length), -100, dtype=target_input_ids.dtype,
                                     device=target_input_ids.device)
        full_labels = torch.cat([prefix_labels, target_input_ids], dim=1)

        outputs = self.llama(inputs_embeds=inputs_embeds,
                             attention_mask=attention_mask,
                             labels=full_labels)
        return outputs.loss

    def generate_with_prefix(self, prefix_embeds, max_length=50, **generate_kwargs):
        """
        For inference: use the soft prefix to condition LLaMA generation.
        """
        generated_ids = self.llama.generate(inputs_embeds=prefix_embeds,
                                            max_length=self.prefix_length + max_length,
                                            **generate_kwargs)
        return generated_ids
