import sys
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class CocoFusionTSVDataset(Dataset):
    def __init__(self, tsv_file, max_examples=sys.maxsize):
        self.samples = []
        groups = {}
        with open(tsv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            count = 0
            # 逐行讀取，只讀取 max_examples 行
            for row in reader:
                if count >= max_examples:
                    break
                url = row["image_url"]
                groups.setdefault(url, []).append(row)
                count += 1

        # 對每一組資料，預先計算好所有行的 image/text 嵌入以及描述
        for url, group in groups.items():
            if len(group) == 0:
                continue
            image_embs = []
            text_embs = []
            target_captions = []
            for row in group:
                try:
                    # 將 image_embedding 與 text_embedding 從字串轉換成 numpy 陣列，再轉成 tensor
                    img_emb = torch.tensor(np.array(list(map(float, row["image_embedding"].split(',')))), dtype=torch.float32)
                    txt_emb = torch.tensor(np.array(list(map(float, row["text_embedding"].split(',')))), dtype=torch.float32)
                except Exception as e:
                    print(f"Error parsing row in group {url}: {e}")
                    continue
                image_embs.append(img_emb)
                text_embs.append(txt_emb)
                target_captions.append(row["description"])
            if len(image_embs) == 0:
                continue
            # 儲存整組資料
            self.samples.append({
                "image_embs": image_embs,  # 此群組所有的 image_embedding
                "text_embs": text_embs,    # 此群組所有的 text_embedding
                "image_emb2_candidates": image_embs,  # 作為 image2 的候選，這裡直接使用 image_embs
                "target_captions": target_captions      # 此群組所有的描述
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        num_candidates = len(sample["image_embs"])
        # 每次隨機選擇一筆作為基礎行 (base row)，使得 image_emb1 與 text_emb1 每次取樣都不同
        rand_idx = random.randint(0, num_candidates - 1)
        return {
            "image_emb1": sample["image_embs"][rand_idx],
            "text_emb1": sample["text_embs"][rand_idx],
            "image_emb2_candidates": sample["image_emb2_candidates"],
            "target_captions": sample["target_captions"]
        }
