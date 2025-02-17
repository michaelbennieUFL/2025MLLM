
# Hyperparameters and Dimensions
IMAGE_EMB_DIM = 512         # e.g. CLIP embedding dimension
TEXT_EMB_DIM = 768          # e.g. BERT [CLS] embedding dimension
PREFIX_LENGTH = 10          # number of soft prompt tokens to generate
FUSION_HIDDEN_DIM = 1024
OUTPUT_TEXT_DIM = TEXT_EMB_DIM  # output size of fusion MLP

# Huggingface LLM name
LLAMA_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
