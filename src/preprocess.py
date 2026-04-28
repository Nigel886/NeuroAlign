import scipy.io
import numpy as np
import torch
import os
from pathlib import Path
from tqdm import tqdm

def process_word_eeg(eeg_data):
    """
    处理单个单词的脑电数据。
    ZuCo 中 rawEEG 可能是:
    1. 一个包含多个 fixation 矩阵的 object array: [(105, T1), (105, T2), ...]
    2. 一个直接的矩阵: (105, T)
    目标: 转换为 (105,) 的向量 (通过对时间维度取平均)
    """
    if isinstance(eeg_data, np.ndarray) and eeg_data.dtype == np.object_:
        # 情况 1: 多个 fixation，先对每个 fixation 取平均，再对所有 fixation 取平均
        fixation_averages = []
        for fix in eeg_data:
            if isinstance(fix, np.ndarray) and fix.size > 0:
                fixation_averages.append(np.mean(fix, axis=1))
        if len(fixation_averages) > 0:
            return np.mean(fixation_averages, axis=0)
        else:
            return np.zeros(105)
    elif isinstance(eeg_data, np.ndarray) and eeg_data.ndim == 2:
        # 情况 2: 直接是 (105, T)
        return np.mean(eeg_data, axis=1)
    else:
        # 异常情况，返回零向量
        return np.zeros(105)

def run_preprocessing(input_mat, output_pt):
    print(f"Loading {input_mat}...")
    mat = scipy.io.loadmat(input_mat, squeeze_me=True, struct_as_record=False)
    sentence_data = mat['sentenceData']
    
    processed_data = []
    
    print("Preprocessing sentences...")
    for sent_obj in tqdm(sentence_data):
        content = sent_obj.content
        words_obj = sent_obj.word
        
        # 提取该句所有单词的 EEG 特征
        sentence_eeg_features = []
        word_list = []
        
        # 处理每个单词
        # 注意: 有些单词可能没有有效的脑电记录（注视缺失）
        for w in words_obj:
            word_text = w.content
            # 尝试获取 rawEEG
            eeg_raw = getattr(w, 'rawEEG', None)
            
            if eeg_raw is not None:
                eeg_vector = process_word_eeg(eeg_raw)
                sentence_eeg_features.append(eeg_vector)
                word_list.append(word_text)
        
        if len(sentence_eeg_features) > 0:
            processed_data.append({
                'content': content,
                'word_list': word_list,
                'eeg_features': np.array(sentence_eeg_features) # (seq_len, 105)
            })

    print(f"Saving {len(processed_data)} processed sentences to {output_pt}...")
    torch.save(processed_data, output_pt)
    print("Done!")

if __name__ == "__main__":
    # 自动处理 data 目录下的 resultsZAB_SR.mat
    input_file = "./data/resultsZAB_SR.mat"
    output_file = "./data/processed_zuco.pt"
    
    if os.path.exists(input_file):
        run_preprocessing(input_file, output_file)
    else:
        print(f"Input file {input_file} not found. Please check the path.")
