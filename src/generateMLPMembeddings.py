import os
import csv
import torch
import sys
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar
from transformers import LlamaTokenizer
from simpleMLPfusion.fusion_module import FusionModule
from simpleMLPfusion.decoder_with_prefix import LlamaDecoderWithPrefix
from src.simpleMLPfusion.parameters import PREFIX_LENGTH
from testMLPModel import load_llama_model_and_tokenizer, load_checkpoint

# Constants
CHECKPOINT_DIR = "checkpoints/MLP/"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "fusion_model_checkpoint_v2.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pretrained_model():
    """
    Loads the pretrained Fusion module and Decoder with the tokenizer.
    """
    # Load LLaMA model and tokenizer
    llama_model, tokenizer = load_llama_model_and_tokenizer()

    # Get LLaMA hidden size
    llama_hidden_size = llama_model.config.hidden_size

    # Load Fusion module and Decoder
    fusion_module = FusionModule(prefix_length=PREFIX_LENGTH, llama_hidden_size=llama_hidden_size)
    decoder = LlamaDecoderWithPrefix(llama_model, prefix_length=PREFIX_LENGTH)

    # Load checkpoint
    optimizer = torch.optim.Adam(list(fusion_module.parameters()), lr=1e-4)  # Placeholder optimizer
    load_checkpoint(CHECKPOINT_PATH, fusion_module, decoder, optimizer, DEVICE)

    # Move models to device
    fusion_module.to(DEVICE)
    decoder.to(DEVICE)
    llama_model.to(DEVICE)

    # Set evaluation mode
    fusion_module.eval()
    decoder.eval()

    return fusion_module, decoder, tokenizer


def parse_embedding(embedding_str):
    """
    Converts a comma-separated string back into a torch tensor (float32).
    """
    return torch.tensor(np.fromstring(embedding_str, sep=","), dtype=torch.float32).to(DEVICE)


def process_tsv(input_tsv, output_tsv, fusion_module, decoder, tokenizer, max_length=20):
    """
    Reads the input TSV, generates captions using the model, and writes the output to a new TSV.
    """
    with open(input_tsv, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile, delimiter="\t")
        rows = list(reader)  # Read all rows into memory for progress tracking

    # Define new column order (place 'generated_caption' after 'positive_prime')
    fieldnames = reader.fieldnames[:reader.fieldnames.index("positive_prime") + 1] + \
                 ["generated_caption"] + \
                 [col for col in reader.fieldnames if col not in ["positive_prime", "generated_caption"]]

    with open(output_tsv, "w", newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        # Use tqdm for progress tracking
        for row in tqdm(rows, desc="Processing Rows", unit="row"):
            # Extract necessary embeddings from TSV
            image_emb1 = parse_embedding(row["next_filename_image_embedding"]).unsqueeze(0)  # Image 1 Embedding
            image_emb2 = parse_embedding(row["prev_image_embedding"]).unsqueeze(0)  # Image 2 Embedding
            text_emb1 = parse_embedding(row["next_filename_positive_prime_embedding"]).unsqueeze(0)  # Image 1 Text Embedding

            # Generate prefix embedding and caption
            with torch.no_grad():
                prefix_embeds = fusion_module(image_emb1, image_emb2, text_emb1)
                generated_ids = decoder.generate_with_prefix(prefix_embeds, max_length=max_length)
                generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip().split(".")[0] + "."

            # Insert generated caption at the correct position
            row["generated_caption"] = generated_caption

            # Write updated row to TSV
            writer.writerow(row)

    print(f"\n✅ Generated captions saved to: {output_tsv}")


if __name__ == "__main__":
    input_tsv = "../data/preprocessed/embeddings_prismatic.tsv"
    output_tsv = "../data/out/embeddings_prismatic.tsv"

    # Load the pretrained model
    fusion_module, decoder, tokenizer = load_pretrained_model()

    # Process the TSV file with progress bar
    process_tsv(input_tsv, output_tsv, fusion_module, decoder, tokenizer)
