import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer
import random
from tqdm import tqdm
from simpleMLPfusion.parameters import LLAMA_MODEL_NAME, PREFIX_LENGTH
from simpleMLPfusion.fusion_module import FusionModule
from simpleMLPfusion.decoder_with_prefix import LlamaDecoderWithPrefix
from simpleMLPfusion.dataset_handler import CocoFusionTSVDataset  # renamed dataset file
import torch

import os

CHECKPOINT_DIR = "checkpoints/MLP/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "fusion_model_checkpoint_v2.pth")

def save_checkpoint(epoch, fusion_module, decoder, optimizer, avg_loss, checkpoint_path=CHECKPOINT_PATH):
    checkpoint = {
         'epoch': epoch,
         'fusion_module_state_dict': fusion_module.state_dict(),
         'decoder_state_dict': decoder.state_dict(),
         'optimizer_state_dict': optimizer.state_dict(),
         'loss': avg_loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(checkpoint_path, fusion_module, decoder, optimizer, device):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))
    fusion_module.load_state_dict(checkpoint['fusion_module_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Then move the model parameters to the GPU
    fusion_module.to(device)
    decoder.to(device)

    start_epoch = checkpoint['epoch'] + 1
    avg_loss = checkpoint.get('loss', None)
    print(f"Checkpoint loaded, resuming from epoch {start_epoch}")
    return start_epoch, avg_loss



def load_llama_model_and_tokenizer():
    # Load tokenizer and model; add special tokens and resize embeddings.
    tokenizer = LlamaTokenizer.from_pretrained(LLAMA_MODEL_NAME)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = LlamaForCausalLM.from_pretrained(LLAMA_MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()  # set model to evaluation mode (for generation)
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer

def load_data(tsv_path, batch_size=5, max_examples=sys.maxsize):
    dataset = CocoFusionTSVDataset(tsv_path, max_examples=max_examples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    return dataset, dataloader


def train_model(train_dataloader, fusion_module, decoder, tokenizer, val_dataset, num_epochs=10, device="cuda"):
    optimizer = optim.Adam(list(fusion_module.parameters()), lr=1e-4)

    # 檢查 checkpoint 是否存在，若存在則從中恢復
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        start_epoch, _ = load_checkpoint(CHECKPOINT_PATH, fusion_module, decoder, optimizer, device)

    fusion_module.train()
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
        for batch in pbar:
            optimizer.zero_grad()
            batch_size = len(batch)
            image_emb1_list = []
            text_emb1_list = []
            image_emb2_list = []
            candidate_captions_list = []  # list of lists
            for sample in batch:
                image_emb1_list.append(sample["image_emb1"])
                text_emb1_list.append(sample["text_emb1"])
                selected_emb2 = random.choice(sample["image_emb2_candidates"])
                image_emb2_list.append(selected_emb2)
                candidate_captions_list.append(sample["target_captions"])
            image_emb1_batch = torch.stack(image_emb1_list).to(device)
            text_emb1_batch = torch.stack(text_emb1_list).to(device)
            image_emb2_batch = torch.stack(image_emb2_list).to(device)

            prefix_embeds = fusion_module(image_emb1_batch, image_emb2_batch, text_emb1_batch)

            batch_loss = 0.0
            for i in range(batch_size):
                sample_loss = None
                for caption in candidate_captions_list[i]:
                    tokenized = tokenizer(caption, return_tensors="pt", padding=True, truncation=True, max_length=256)
                    input_ids = tokenized.input_ids.to(device)
                    loss_i = decoder(prefix_embeds[i].unsqueeze(0), input_ids)
                    if sample_loss is None or loss_i.item() < sample_loss.item():
                        sample_loss = loss_i
                batch_loss += sample_loss
            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            pbar.set_postfix(loss=batch_loss.item())

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}", end=" || ")

        validate_loss(fusion_module, decoder, tokenizer, val_dataset, device=device, max_length=256)
        if (epoch + 1) % 1 == 0 or (epoch == num_epochs - 1):
            validate_results(fusion_module, decoder, tokenizer, val_dataset, device=device, max_length=20,
                             max_samples=2)

        # 保存 checkpoint
        save_checkpoint(epoch, fusion_module, decoder, optimizer, avg_loss)


def validate_loss(fusion_module, decoder, tokenizer, val_dataset, device="cuda", max_length=256):
    fusion_module.eval()
    decoder.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for sample in val_dataset:
            image_emb1 = sample["image_emb1"].unsqueeze(0).to(device)
            image_emb2 = random.choice(sample["image_emb2_candidates"]).unsqueeze(0).to(device)
            text_emb1 = sample["text_emb1"].unsqueeze(0).to(device)
            prefix_embeds = fusion_module(image_emb1, image_emb2, text_emb1)
            sample_loss = None
            for caption in sample["target_captions"]:
                tokenized = tokenizer([caption], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                input_ids = tokenized.input_ids.to(device)
                loss = decoder(prefix_embeds, input_ids)
                if sample_loss is None or loss.item() < sample_loss.item():
                    sample_loss = loss
            total_loss += sample_loss.item()
            count += 1
    avg_loss = total_loss / count if count > 0 else float('nan')
    print(f"Validation Loss: {avg_loss:.4f}")

def validate_results(fusion_module, decoder, tokenizer, test_dataset, device="cuda", max_length=20, max_samples=30):
    fusion_module.eval()
    decoder.eval()
    num_samples = min(max_samples, len(test_dataset))
    selected_indices = random.sample(range(len(test_dataset)), num_samples)
    print("\n--- Validation Results ---")
    for count, i in enumerate(selected_indices):
        sample = test_dataset[i]
        image_emb1 = sample["image_emb1"].unsqueeze(0).to(device)
        image_emb2 = random.choice(sample["image_emb2_candidates"]).unsqueeze(0).to(device)
        text_emb1 = sample["text_emb1"].unsqueeze(0).to(device)
        prefix_embeds = fusion_module(image_emb1, image_emb2, text_emb1)
        generated_ids = decoder.generate_with_prefix(prefix_embeds, max_length=max_length)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Sample {count + 1}:")
        print("Generated Caption:", generated_text)
        print("Candidate Captions:", sample["target_captions"])
        print("-----")

def run_inference(fusion_module, decoder, tokenizer, dataset, device="cuda", max_length=20):
    fusion_module.eval()
    decoder.eval()
    with torch.no_grad():
        sample = dataset[0]
        image_emb1 = sample["image_emb1"].unsqueeze(0).to(device)
        image_emb2 = random.choice(sample["image_emb2_candidates"]).unsqueeze(0).to(device)
        text_emb1 = sample["text_emb1"].unsqueeze(0).to(device)
        prefix_embeds = fusion_module(image_emb1, image_emb2, text_emb1)
        generated_ids = decoder.generate_with_prefix(prefix_embeds, max_length=max_length)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("\n--- Inference Example ---")
        print("Generated Caption:", generated_text)
        print("Candidate Captions:", sample["target_captions"])

def test_MLP(num_epochs=20, batch_size=8,
             train_tsv_path="../data/preprocessed/train2017_preprocessed_image_text_pairs.tsv",
             val_tsv_path="../data/preprocessed/val2017_preprocessed_image_text_pairs.tsv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load LLaMA model and tokenizer
    llama_model, tokenizer = load_llama_model_and_tokenizer()
    # Load training dataset and validation dataset
    train_dataset, train_dataloader = load_data(train_tsv_path, batch_size=batch_size, max_examples=7*10**4)
    val_dataset, _ = load_data(val_tsv_path, batch_size=1, max_examples=8*10**2)  # For validation, batch size can be 1

    # Create Fusion module and Decoder wrapper
    llama_hidden_size = llama_model.config.hidden_size
    fusion_module = FusionModule(prefix_length=PREFIX_LENGTH, llama_hidden_size=llama_hidden_size)
    decoder = LlamaDecoderWithPrefix(llama_model, prefix_length=PREFIX_LENGTH)
    fusion_module.to(device)
    decoder.to(device)
    llama_model.to(device)

    # Train the fusion module (with periodic validation)
    train_model(train_dataloader, fusion_module, decoder, tokenizer, val_dataset, num_epochs=num_epochs, device=device)

    # Run a final inference example on the training dataset
    run_inference(fusion_module, decoder, tokenizer, train_dataset, device=device, max_length=20)
    validate_results(fusion_module, decoder, tokenizer, val_dataset, device=device, max_length=20, max_samples=15)
    validate_loss(fusion_module, decoder, tokenizer, val_dataset, device=device, max_length=256)

if __name__ == "__main__":
    test_MLP(num_epochs=30, batch_size=13)
