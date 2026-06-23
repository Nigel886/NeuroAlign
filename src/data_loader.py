import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np
from pathlib import Path
import re

def _infer_subject_id_from_path(path_str):
    match = re.search(r"Z[A-Z]{2}", str(path_str).upper())
    if match:
        return match.group(0)
    return None

def _ensure_2d_eeg(arr):
    eeg = np.asarray(arr, dtype=np.float32)
    if eeg.ndim == 1:
        eeg = np.expand_dims(eeg, axis=0)
    if eeg.ndim != 2:
        raise ValueError(f"Expected EEG with 2 dims (seq_len, channels), got shape {eeg.shape}.")
    return eeg

def _compute_subject_zscore_stats(samples_by_subject, eps=1e-6):
    stats = {}
    for subject_id, items in samples_by_subject.items():
        ch_sum = None
        ch_sumsq = None
        count = 0
        for item in items:
            eeg = _ensure_2d_eeg(item["eeg_features"])
            if ch_sum is None:
                ch_sum = np.zeros((eeg.shape[1],), dtype=np.float64)
                ch_sumsq = np.zeros((eeg.shape[1],), dtype=np.float64)
            ch_sum += eeg.sum(axis=0, dtype=np.float64)
            ch_sumsq += (eeg.astype(np.float64) ** 2).sum(axis=0)
            count += int(eeg.shape[0])
        if count == 0:
            mean = np.zeros((105,), dtype=np.float32)
            std = np.ones((105,), dtype=np.float32)
        else:
            mean = (ch_sum / count).astype(np.float32)
            var = (ch_sumsq / count - (ch_sum / count) ** 2).astype(np.float32)
            var = np.maximum(var, 0.0)
            std = np.sqrt(var + eps).astype(np.float32)
        stats[subject_id] = (mean, std)
    return stats

class ZuCoDataset(Dataset):
    def __init__(
        self,
        pt_path,
        tokenizer_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
        max_len=512,
        split="all",
        test_subject_id=None,
        subject_wise_normalize=True,
    ):
        """
        pt_path: 预处理后的 .pt 文件路径 (包含 content, word_list, eeg_features)
        tokenizer_name_or_path: LLM 的分词器名称
        max_len: 文本最大长度
        """
        pt_paths = pt_path if isinstance(pt_path, (list, tuple)) else [pt_path]
        resolved_pt_paths = []
        for p in pt_paths:
            p = Path(p)
            if p.is_dir():
                resolved_pt_paths.extend(sorted(p.glob("*.pt")))
            else:
                resolved_pt_paths.append(p)
        if not resolved_pt_paths:
            raise FileNotFoundError(f"No .pt files found from: {pt_path}")
        for p in resolved_pt_paths:
            if not p.exists():
                raise FileNotFoundError(f"Processed data not found at {p}. Please run preprocess.py first.")

        raw_items = []
        for p in resolved_pt_paths:
            items = torch.load(p, weights_only=False)
            inferred = _infer_subject_id_from_path(p.name)
            for it in items:
                subject_id = (
                    str(it.get("subject_id")).upper()
                    if isinstance(it, dict) and it.get("subject_id") is not None
                    else inferred
                )
                if subject_id is None:
                    subject_id = "UNK"
                raw_items.append(
                    {
                        "subject_id": subject_id,
                        "content": it["content"],
                        "eeg_features": it["eeg_features"],
                    }
                )

        test_subject = str(test_subject_id).upper() if test_subject_id is not None else None
        if split not in {"all", "train", "test"}:
            raise ValueError(f"split must be one of: all/train/test, got: {split}")
        if split in {"train", "test"} and not test_subject:
            raise ValueError("test_subject_id is required when split is train/test.")

        samples_by_subject = {}
        for it in raw_items:
            samples_by_subject.setdefault(it["subject_id"], []).append(it)

        if split in {"train", "test"} and test_subject not in samples_by_subject:
            available = sorted(samples_by_subject.keys())
            raise ValueError(f"test_subject_id={test_subject} not found. Available subjects: {available}")

        self.subject_stats = (
            _compute_subject_zscore_stats(samples_by_subject)
            if subject_wise_normalize
            else {}
        )

        if split == "all":
            self.data = raw_items
        elif split == "train":
            self.data = [it for it in raw_items if it["subject_id"] != test_subject]
        else:
            self.data = [it for it in raw_items if it["subject_id"] == test_subject]

        subjects = sorted({str(it["subject_id"]).upper() for it in raw_items if it.get("subject_id") not in (None, "UNK")})
        self.subject_to_idx = {sid: i for i, sid in enumerate(subjects)}

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, local_files_only=False)
        except Exception:
            print("Warning: Network error. Ensure $env:HF_HUB_OFFLINE=1 is set.")
            raise
        
        # Llama-3 等模型需要手动设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 获取文本并 Tokenize
        text = item['content']
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding=False, # 我们在 collate_fn 中统一处理 padding
            truncation=True,
            return_tensors=None
        )
        
        # 2. 获取 EEG 特征 (seq_len, 105)
        # 这里的 seq_len 是单词数
        eeg_np = _ensure_2d_eeg(item["eeg_features"])
        if self.subject_stats:
            mean, std = self.subject_stats[item["subject_id"]]
            eeg_np = (eeg_np - mean) / std
        eeg_features = torch.from_numpy(eeg_np)
        subject_label = self.subject_to_idx.get(str(item["subject_id"]).upper(), -1)
        
        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'eeg': eeg_features,
            'text': text,
            'subject_id': item["subject_id"],
            'subject_label': torch.tensor(subject_label, dtype=torch.long),
        }

def zuco_collate_fn(batch, pad_token_id):
    """
    自定义 Collate Function，处理变长的 EEG 和 Token 序列
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    eegs = [item['eeg'] for item in batch]
    subject_labels = [item.get('subject_label') for item in batch]
    
    # 对 input_ids 进行 Padding (文本)
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(
        attention_masks, batch_first=True, padding_value=0
    )
    
    # 对 EEG 进行 Padding (脑电)
    # eeg shape: (seq_len, 105) -> Padding 后的 shape: (batch, max_seq_len, 105)
    eeg_padded = torch.nn.utils.rnn.pad_sequence(
        eegs, batch_first=True, padding_value=0.0
    )
    
    # 创建 EEG 的 attention mask (记录哪些位置是真实的脑电单词)
    eeg_masks = []
    max_eeg_len = eeg_padded.shape[1]
    for eeg in eegs:
        mask = torch.zeros(max_eeg_len, dtype=torch.long)
        mask[:eeg.shape[0]] = 1
        eeg_masks.append(mask)
    eeg_masks = torch.stack(eeg_masks)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'eeg': eeg_padded,
        'eeg_mask': eeg_masks,
        'texts': [item['text'] for item in batch],
        'subject_ids': [item.get('subject_id') for item in batch],
        'subject_labels': torch.stack(subject_labels) if all(x is not None for x in subject_labels) else None,
    }

def get_dataloader(
    pt_path,
    tokenizer_name,
    batch_size=8,
    shuffle=True,
    split="all",
    test_subject_id=None,
    subject_wise_normalize=True,
    max_len=512,
):
    dataset = ZuCoDataset(
        pt_path,
        tokenizer_name_or_path=tokenizer_name,
        max_len=max_len,
        split=split,
        test_subject_id=test_subject_id,
        subject_wise_normalize=subject_wise_normalize,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: zuco_collate_fn(b, dataset.tokenizer.pad_token_id)
    )
    
    return loader

def get_loso_loaders(
    pt_path,
    tokenizer_name,
    test_subject="ZAB",
    train_batch_size=2,
    test_batch_size=4,
    max_len=512,
    subject_wise_normalize=True,
):
    train_loader = get_dataloader(
        pt_path,
        tokenizer_name,
        batch_size=train_batch_size,
        shuffle=True,
        split="train",
        test_subject_id=test_subject,
        subject_wise_normalize=subject_wise_normalize,
        max_len=max_len,
    )
    test_loader = get_dataloader(
        pt_path,
        tokenizer_name,
        batch_size=test_batch_size,
        shuffle=False,
        split="test",
        test_subject_id=test_subject,
        subject_wise_normalize=subject_wise_normalize,
        max_len=max_len,
    )
    return train_loader, test_loader

if __name__ == "__main__":
    # 测试代码
    PT_PATH = "./data/preprocessed/processed_zuco_cleaned.pt"
    # 注意：本地测试时如果没有下载 Llama-3，可以换成 gpt2 或其他轻量级分词器
    TOKENIZER = "gpt2" 
    
    try:
        loader = get_dataloader(PT_PATH, TOKENIZER, batch_size=2, shuffle=False)
        for batch in loader:
            print(f"Input IDs Shape: {batch['input_ids'].shape}")
            print(f"EEG Shape: {batch['eeg'].shape}") # (batch, max_word_len, 105)
            print(f"EEG Mask Shape: {batch['eeg_mask'].shape}")
            break
    except Exception as e:
        print(f"DataLoader Test Info: {e}")
