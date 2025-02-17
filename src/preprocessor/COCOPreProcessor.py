import os
import csv
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading

from pycocotools.coco import COCO
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel

# Create a thread-local storage for CUDA streams.
thread_local = threading.local()


def get_cuda_stream():
    """Return a CUDA stream unique to the current thread."""
    if not hasattr(thread_local, "stream"):
        thread_local.stream = torch.cuda.Stream()
    return thread_local.stream


class CocoDataProcessor:
    def __init__(self, coco_instance_file, captions_file, bert_model_name, clip_model_name, num_threads=5):
        """
        Initializes the processor with:
          - a COCO instance (for instance annotations),
          - a captions file (for caption annotations),
          - a Hugging Face BERT model name,
          - a Hugging Face CLIP model name,
          - num_threads: Number of threads to use for generating embeddings.
        """
        # Store the COCO instance for object/instance annotations
        self.coco = COCO(coco_instance_file)
        # Create a COCO instance for captions using the provided captions file
        self.coco_caps = COCO(captions_file)

        # Initialize the BERT tokenizer and model (for text encoding)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        self.bert_model.eval()  # set to evaluation mode

        # Initialize the CLIP processor and model (for image encoding)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_model.eval()  # set to evaluation mode

        # (Optional) Move models to GPU if available.
        if torch.cuda.is_available():
            self.bert_model.to("cuda")
            self.clip_model.to("cuda")

        # Create caches to avoid re-computing embeddings
        self.image_embedding_cache = {}
        self.text_embedding_cache = {}

        # Number of threads for multi-threaded embedding generation
        self.num_threads = num_threads

    # -------------------------------
    # Helper functions
    # -------------------------------

    def download_image(self, url):
        """Download an image from a URL and return a PIL Image."""
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image

    def _array_to_str(self, arr):
        """Convert a numpy array into a comma-separated string."""
        return ",".join(map(str, arr.flatten()))

    # -------------------------------
    # Function 1: generateImageTextPairs
    # -------------------------------

    def generateImageTextPairs(self, image_id):
        """
        For a given image_id, generate a list of dictionaries.
        Each dictionary contains:
          - 'image_url': URL (or file name) of the image.
          - 'description': A caption from the captions annotations.
        """
        imgs = self.coco.loadImgs([image_id])
        if not imgs:
            return []
        image_info = imgs[0]
        image_url = image_info.get('coco_url') or image_info.get('file_name')

        annIds = self.coco_caps.getAnnIds(imgIds=image_id)
        anns = self.coco_caps.loadAnns(annIds)

        pairs = []
        for ann in anns:
            pair = {
                'image_url': image_url,
                'description': ann['caption']
            }
            pairs.append(pair)
        return pairs

    # -------------------------------
    # Function 2: generateAllDataPairs
    # -------------------------------

    def generateAllDataPairs(self):
        """
        Generate a list of data pairs for all unique image ids in the instances COCO object.
        Each entry is a dictionary containing 'image_url' and 'description'.
        Each (image, caption) pair is a separate entry.
        """
        all_pairs = []
        imgIds = self.coco.getImgIds()
        for image_id in tqdm(imgIds, desc="Generating Image-Text Pairs"):
            pairs = self.generateImageTextPairs(image_id)
            all_pairs.extend(pairs)
        return all_pairs

    # -------------------------------
    # Function 3: generateImageEmbeddings (Multi-threaded with separate CUDA streams)
    # -------------------------------

    def compute_image_embedding(self, image_url):
        """
        Compute the image embedding using the CLIP model.
        Uses caching so that repeated image_urls do not incur extra computation.
        Each thread uses its own CUDA stream.
        """
        if image_url in self.image_embedding_cache:
            return self.image_embedding_cache[image_url]

        image = self.download_image(image_url)
        inputs = self.clip_processor(images=image, return_tensors="pt")
        # Move inputs to GPU if available.
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        stream = get_cuda_stream()
        with torch.cuda.stream(stream):
            with torch.no_grad():
                embedding = self.clip_model.get_image_features(**inputs)
            stream.synchronize()
        embedding_np = embedding.squeeze().cpu().numpy()
        self.image_embedding_cache[image_url] = embedding_np
        return embedding_np

    def _process_image_item(self, item):
        """Helper to process one data pair for image embedding."""
        url = item['image_url']
        item['image_embedding'] = self.compute_image_embedding(url)
        return item

    def generateImageEmbeddings(self, data_pairs):
        """
        Compute the image embedding for each dictionary in data_pairs using multithreading.
        """
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(tqdm(executor.map(self._process_image_item, data_pairs),
                                total=len(data_pairs),
                                desc="Generating Image Embeddings"))
        return results

    # -------------------------------
    # Function 4: generateTextEmbeddings (Multi-threaded with separate CUDA streams)
    # -------------------------------

    def compute_text_embedding(self, text):
        """
        Compute the text embedding using the BERT model.
        Uses caching based on the input text.
        Each thread uses its own CUDA stream.
        """
        if text in self.text_embedding_cache:
            return self.text_embedding_cache[text]

        inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        # Move inputs to GPU if available.
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        stream = get_cuda_stream()
        with torch.cuda.stream(stream):
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            stream.synchronize()
        # Use the [CLS] token embedding (first token in the sequence)
        embedding = outputs.last_hidden_state[:, 0, :]
        embedding_np = embedding.squeeze().cpu().numpy()
        self.text_embedding_cache[text] = embedding_np
        return embedding_np

    def _process_text_item(self, item):
        """Helper to process one data pair for text embedding."""
        text = item['description']
        item['text_embedding'] = self.compute_text_embedding(text)
        return item

    def generateTextEmbeddings(self, data_pairs):
        """
        Compute the text embedding for each dictionary in data_pairs using multithreading.
        """
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(tqdm(executor.map(self._process_text_item, data_pairs),
                                total=len(data_pairs),
                                desc="Generating Text Embeddings"))
        return results

    # -------------------------------
    # Function 5: generateDataList
    # -------------------------------

    def generateDataList(self, max_ids=None):
        """
        Generate the complete data list by:
          1. Generating all data pairs (or only for the first 'max_ids' image ids if provided).
          2. Generating image embeddings for each pair.
          3. Generating text embeddings for each pair.
        """
        if max_ids is not None:
            imgIds = self.coco.getImgIds()[:max_ids]
            all_pairs = []
            for image_id in tqdm(imgIds, desc="Processing Limited Images"):
                pairs = self.generateImageTextPairs(image_id)
                all_pairs.extend(pairs)
        else:
            all_pairs = self.generateAllDataPairs()

        all_pairs = self.generateImageEmbeddings(all_pairs)
        all_pairs = self.generateTextEmbeddings(all_pairs)
        return all_pairs

    # -------------------------------
    # Function 6: saveDataAsTSV
    # -------------------------------

    def saveDataAsTSV(self, file_location, data_list):
        """
        Save the data (list of dictionaries) to a TSV file.
        Each row contains:
          - image_url
          - description
          - image_embedding (as a comma-separated string)
          - text_embedding (as a comma-separated string)
        """
        with open(file_location, "w", newline="") as tsvfile:
            writer = csv.writer(tsvfile, delimiter="\t")
            writer.writerow(["image_url", "description", "image_embedding", "text_embedding"])
            for item in data_list:
                image_url = item.get("image_url", "")
                description = item.get("description", "")
                image_embedding = self._array_to_str(item.get("image_embedding", np.array([])))
                text_embedding = self._array_to_str(item.get("text_embedding", np.array([])))
                writer.writerow([image_url, description, image_embedding, text_embedding])

    # -------------------------------
    # Cleanup method to delete CUDA streams in thread-local storage
    # -------------------------------

    def cleanup_cuda_streams(self):
        """
        Spawns a temporary ThreadPoolExecutor to run a cleanup function on each worker thread
        that deletes its CUDA stream from thread-local storage.
        """

        def cleanup_func(_):
            if hasattr(thread_local, "stream"):
                del thread_local.stream

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            list(executor.map(cleanup_func, range(self.num_threads)))

    def __del__(self):
        # Ensure CUDA streams in thread-local storage are cleaned up when the instance is deleted.
        self.cleanup_cuda_streams()


def generate_all_embedding_files(num_threads=5):
    """
    Generates TSV files for both validation and training datasets and stores them in:
    ../data/preprocessed/{type}_preprocessed_image_text_pairs.tsv
    :param num_threads: Number of threads to use for embedding generation.
    """
    base_dir = "../../data/coco_ann2017/annotations/"
    output_dir = "../../data/preprocessed/"
    os.makedirs(output_dir, exist_ok=True)

    dataset_types = ["val2017", "train2017"]

    for dataset_type in dataset_types:
        print(f"Processing {dataset_type} dataset with {num_threads} threads...")
        instances_file = os.path.join(base_dir, f"instances_{dataset_type}.json")
        captions_file = os.path.join(base_dir, f"captions_{dataset_type}.json")
        output_file = os.path.join(output_dir, f"{dataset_type}_preprocessed_image_text_pairs.tsv")

        processor = CocoDataProcessor(instances_file, captions_file,
                                      bert_model_name="bert-base-uncased",
                                      clip_model_name="openai/clip-vit-base-patch32",
                                      num_threads=num_threads)

        data_list = processor.generateDataList(max_ids=100)
        processor.saveDataAsTSV(output_file, data_list)
        print(f"Saved {dataset_type} preprocessed data to {output_file}")
        # Optionally, clean up CUDA streams after processing this dataset.
        processor.cleanup_cuda_streams()


if __name__ == "__main__":
    # Optionally, pass the number of threads (default is 5)
    generate_all_embedding_files(num_threads=15)
