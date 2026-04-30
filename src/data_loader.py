import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np
from pathlib import Path

class ZuCoDataset(Dataset):
    def __init__(self, pt_path, tokenizer_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct", max_len=512):
        """
        pt_path: 预处理后的 .pt 文件路径 (包含 content, word_list, eeg_features)
        tokenizer_name_or_path: LLM 的分词器名称
        max_len: 文本最大长度
        """
        if not Path(pt_path).exists():
            raise FileNotFoundError(f"Processed data not found at {pt_path}. Please run preprocess.py first.")
        
        self.data = torch.load(pt_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        
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
        eeg_features = torch.tensor(item['eeg_features'], dtype=torch.float32)
        
        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'eeg': eeg_features,
            'text': text
        }

def zuco_collate_fn(batch, pad_token_id):
    """
    自定义 Collate Function，处理变长的 EEG 和 Token 序列
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    eegs = [item['eeg'] for item in batch]
    
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
        'texts': [item['text'] for item in batch]
    }

def get_dataloader(pt_path, tokenizer_name, batch_size=8, shuffle=True):
    dataset = ZuCoDataset(pt_path, tokenizer_name)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: zuco_collate_fn(b, dataset.tokenizer.pad_token_id)
    )
    
    return loader

if __name__ == "__main__":
    # 测试代码
    PT_PATH = "./data/preprocessed/processed_zuco_cleaned.pt"
    # 注意：本地测试时如果没有下载 Llama-3，可以换成 gpt2 或其他轻量级分词器
    TOKENIZER = "gpt2" 
    
    try:
        loader = get_dataloader(PT_PATH, TOKENIZER, batch_size=2)
        for batch in loader:
            print(f"Input IDs Shape: {batch['input_ids'].shape}")
            print(f"EEG Shape: {batch['eeg'].shape}") # (batch, max_word_len, 105)
            print(f"EEG Mask Shape: {batch['eeg_mask'].shape}")
            break
    except Exception as e:
        print(f"DataLoader Test Info: {e}")
