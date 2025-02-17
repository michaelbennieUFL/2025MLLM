#!/usr/bin/env python
import os
import csv
import random
import torch
import numpy as np
from PIL import Image
import threading
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel

# 如果 preprocessor 中已有该函数，也可以直接导入
thread_local = threading.local()
def get_cuda_stream():
    """Return a CUDA stream unique to the current thread."""
    if not hasattr(thread_local, "stream"):
        thread_local.stream = torch.cuda.Stream()
    return thread_local.stream

# 从 preprocessor（COCOPreProcessor.py）中导入 CocoDataProcessor
import COCOPreProcessor as preprocessor

# 本地处理器继承自 CocoDataProcessor，但我们只需要加载模型，不需要 COCO 标注
class LocalProcessor(preprocessor.CocoDataProcessor):
    def __init__(self, bert_model_name, clip_model_name,num_threads=2):
        # 本地版本不需要传 COCO 文件，因此直接初始化模型
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        self.bert_model.eval()

        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_model.eval()

        if torch.cuda.is_available():
            self.bert_model.to("cuda")
            self.clip_model.to("cuda")

        # 初始化缓存
        self.image_embedding_cache = {}
        self.text_embedding_cache = {}
        self.num_threads=num_threads

    def download_image(self, filename):
        """
        重写 download_image 方法，从 selected_image 文件夹加载本地图片。
        """
        path = os.path.join("../../data/test_set/selected_images", filename)
        return Image.open(path).convert("RGB")

    def compute_image_embedding(self, filename):
        """
        重写 compute_image_embedding，直接以文件名为 key，从本地加载图片计算 CLIP image embedding。
        """
        if filename in self.image_embedding_cache:
            return self.image_embedding_cache[filename]

        image = self.download_image(filename)
        inputs = self.clip_processor(images=image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        stream = get_cuda_stream()
        with torch.cuda.stream(stream):
            with torch.no_grad():
                embedding = self.clip_model.get_image_features(**inputs)
            stream.synchronize()

        embedding_np = embedding.squeeze().cpu().numpy()
        self.image_embedding_cache[filename] = embedding_np
        return embedding_np

    def compute_text_embedding(self, text):
        """
        重写 compute_text_embedding，直接计算文本的 BERT embedding。
        """
        if text in self.text_embedding_cache:
            return self.text_embedding_cache[text]

        inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        stream = get_cuda_stream()
        with torch.cuda.stream(stream):
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            stream.synchronize()

        # 使用 [CLS] token 作为句子的 embedding
        embedding = outputs.last_hidden_state[:, 0, :]
        embedding_np = embedding.squeeze().cpu().numpy()
        self.text_embedding_cache[text] = embedding_np
        return embedding_np

    def generate_caption(self, filename):
        """
        用于生成第二张图片的 caption。
        这里不调用专门的 caption 模型，而是简单地根据文件名返回一个伪 caption。
        你可以根据实际需求修改该逻辑。
        """
        # 例如：根据文件名生成描述
        return f"Caption for {os.path.splitext(filename)[0]}"

    def _array_to_str(self, arr):
        """将 numpy 数组转换为逗号分隔的字符串（复用父类方法）"""
        return ",".join(map(str, arr.flatten()))


def processData(input_csv, output_csv):
    # 初始化 LocalProcessor，只加载 BERT 与 CLIP 模型
    processor = LocalProcessor(bert_model_name="bert-base-uncased",
                               clip_model_name="openai/clip-vit-base-patch32")

    # 读取输入 CSV（格式：label,filename,img_id,positive_sentence,negative_sentence）
    with open(input_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data_rows = list(reader)

    # 获取 input_csv 的父级目录，并定位到 selected_image 文件夹
    parent_folder = Path(input_csv).parent
    selected_image_folder = parent_folder / "selected_images"
    all_images = [img for img in os.listdir(selected_image_folder)
                  if img.lower().endswith((".jpg", ".jpeg", ".png"))]

    output_rows = []

    for row in data_rows:
        # 读取第一张图片的信息及其描述
        prev_label = row['\ufefflabel']
        prev_filename = row["filename"]
        prev_img_id = row["img_id"]
        positive_prime = row["positive_sentence"]
        negative_prime = row["negative_sentence"]

        # 计算第一张图片的 embedding（使用 CLIP 模型）
        prev_image_embedding = processor.compute_image_embedding(prev_filename)
        # 计算 positive_sentence 的文本 embedding（使用 BERT 模型）
        positive_prime_embedding = processor.compute_text_embedding(positive_prime)

        # 从 selected_image 文件夹中随机选择另一张图片（排除当前图片）
        candidate_images = [img for img in all_images if img != prev_filename]
        if not candidate_images:
            candidate_images = all_images  # 如果只有一张图片，则退化为所有图片均可选
        next_filename = random.choice(candidate_images)

        # 计算第二张图片的 image embedding
        next_image_embedding = processor.compute_image_embedding(next_filename)
        # 生成第二张图片的 caption（通过重写的 generate_caption 方法）
        next_caption = processor.generate_caption(next_filename)
        # 计算生成 caption 的文本 embedding
        next_caption_embedding = processor.compute_text_embedding(next_caption)

        # 将 embedding 数组转换为逗号分隔的字符串
        output_row = {
            "prev_label": prev_label,
            "prev_filename": prev_filename,
            "prev_img_id": prev_img_id,
            "positive_prime": positive_prime,
            "negative_prime": negative_prime,
            "next_filename": next_filename,
            "prev_image_embedding": processor._array_to_str(prev_image_embedding),
            "positive_prime_embedding": processor._array_to_str(positive_prime_embedding),
            "next_filename_image_embedding": processor._array_to_str(next_image_embedding),
            "next_filename_positive_prime_embedding": processor._array_to_str(next_caption_embedding),
        }
        output_rows.append(output_row)

    # 写入最终 CSV，包含 10 列信息
    fieldnames = [
        "prev_label",
        "prev_filename",
        "prev_img_id",
        "positive_prime",
        "negative_prime",
        "next_filename",
        "prev_image_embedding",
        "positive_prime_embedding",
        "next_filename_image_embedding",
        "next_filename_positive_prime_embedding",
    ]
    with open(output_csv, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    print(f"最终输出 CSV 已保存至 {output_csv}")

if __name__ == "__main__":
    # 修改为实际文件路径
    processData(input_csv="../../data/test_set/data.csv",
                output_csv="../../data/preprocessed/embeddings_prismatic.tsv")
