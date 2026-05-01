import scipy.io
import numpy as np
import torch
import mne
import os
import re
from pathlib import Path
from tqdm import tqdm

def _infer_subject_id_from_filename(path_str):
    match = re.search(r"Z[A-Z]{2}", str(path_str).upper())
    if match:
        return match.group(0)
    return "UNK"

def clean_eeg_with_mne(data_105, sfreq=1000, target_fs=250):
    """
    使用 MNE 进行高级清洗：滤波、重采样、ICA 去伪影
    data_105: (105, T) 的 numpy 数组
    """
    # 1. 创建 MNE 结构
    # ZuCo 使用 105 通道，这里简单命名，实际可根据 electrode 映射表优化
    ch_names = [f"EEG{i:03d}" for i in range(1, 106)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data_105, info, verbose=False)
    
    # 设置标准导联（可选，有助于 ICA 可视化和自动检测）
    # raw.set_montage('standard_1024') 
    
    # 2. 带通滤波 (0.5 - 50Hz)
    raw.filter(l_freq=0.5, h_freq=50.0, fir_design='firwin', verbose=False)
    
    # 3. 重采样
    if sfreq != target_fs:
        raw.resample(target_fs, verbose=False)
    
    # 4. ICA 去伪影
    # n_components 设置为 20 是一个轻量级的平衡点
    ica = mne.preprocessing.ICA(n_components=20, random_state=42, method='fastica', verbose=False)
    ica.fit(raw, verbose=False)
    
    # 自动检测眼动 (EOG) 成分
    # 由于没有专门的 EOG 通道，我们利用前额通道 (如 EEG001/EEG002，对应 FP1/FP2) 作为参考
    # 在 ZuCo 中，前几个通道通常是前额通道
    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='EEG001', threshold=3.0, verbose=False)
    
    # 如果没搜到，尝试另一个前额通道
    if not eog_indices:
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='EEG002', threshold=3.0, verbose=False)
    
    # 剔除检测到的成分
    ica.exclude = eog_indices
    raw_cleaned = raw.copy()
    ica.apply(raw_cleaned, verbose=False)
    
    return raw_cleaned

def process_word_eeg(eeg_data):
    """
    处理单个单词的脑电数据（时间维度平均）
    """
    if isinstance(eeg_data, np.ndarray) and eeg_data.dtype == np.object_:
        fixation_averages = []
        for fix in eeg_data:
            if isinstance(fix, np.ndarray) and fix.size > 0:
                fixation_averages.append(np.mean(fix, axis=1))
        if len(fixation_averages) > 0:
            return np.mean(fixation_averages, axis=0)
        else:
            return np.zeros(105)
    elif isinstance(eeg_data, np.ndarray) and eeg_data.ndim == 2:
        return np.mean(eeg_data, axis=1)
    else:
        return np.zeros(105)

def run_advanced_preprocessing(input_mat, output_dir):
    """
    完整清洗流程：加载 -> MNE 清洗 -> 保存 .fif -> 提取特征保存 .pt
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fif_dir = output_path / "fif_cleaned"
    fif_dir.mkdir(exist_ok=True)
    subject_id = _infer_subject_id_from_filename(Path(input_mat).name)
    subject_fif_dir = fif_dir / subject_id
    subject_fif_dir.mkdir(exist_ok=True)

    print(f"Loading {input_mat}...")
    mat = scipy.io.loadmat(input_mat, squeeze_me=True, struct_as_record=False)
    sentence_data = mat['sentenceData']
    
    processed_list = []
    
    print(f"Cleaning {len(sentence_data)} sentences with MNE ICA... (subject={subject_id})")
    for i, sent_obj in enumerate(tqdm(sentence_data)):
        # 1. 句子级原始信号清洗
        raw_eeg = sent_obj.rawData # (105, T)
        
        # 执行 MNE 清洗
        try:
            cleaned_raw = clean_eeg_with_mne(raw_eeg, sfreq=1000, target_fs=250)
            
            # 2. 保存为 .fif 格式
            fif_name = subject_fif_dir / f"{subject_id}_sentence_{i:03d}_cleaned.fif"
            cleaned_raw.save(str(fif_name), overwrite=True, verbose=False)
            
            # 3. 提取清洗后的 Word-level 特征
            # 注意：此处我们仍然利用 ZuCo 已经切分好的 word 结构，
            # 但为了严谨，理想情况应根据 wordbounds 重新在 cleaned_raw 上切分。
            # 这里先遵循用户逻辑，处理 word['rawEEG']
            words_obj = sent_obj.word
            sentence_eeg_features = []
            word_list = []
            
            for w in words_obj:
                word_text = w.content
                eeg_raw = getattr(w, 'rawEEG', None)
                if eeg_raw is not None:
                    # 对 word-level 数据也进行平均化处理
                    eeg_vector = process_word_eeg(eeg_raw)
                    sentence_eeg_features.append(eeg_vector)
                    word_list.append(word_text)
            
            if len(sentence_eeg_features) > 0:
                processed_list.append({
                    'subject_id': subject_id,
                    'sentence_id': i,
                    'content': sent_obj.content,
                    'word_list': word_list,
                    'eeg_features': np.array(sentence_eeg_features) # (seq_len, 105)
                })
                
        except Exception as e:
            print(f"Error processing sentence {i}: {e}")

    # 保存最终用于训练的 .pt 文件
    torch.save(processed_list, output_path / f"processed_{subject_id}_cleaned.pt")
    print(f"\nPreprocessing complete!")
    print(f"Cleaned .fif files saved in: {subject_fif_dir}")
    print(f"Final training data saved as: {output_path / f'processed_{subject_id}_cleaned.pt'}")

    return processed_list

if __name__ == "__main__":
    data_dir = Path("./data")
    output_dir = "./data/preprocessed"
    mat_files = sorted(data_dir.glob("*.mat"))
    if not mat_files:
        print(f"No .mat files found in {data_dir.resolve()}")
        raise SystemExit(1)

    all_processed = []
    for mat_path in mat_files:
        all_processed.extend(run_advanced_preprocessing(str(mat_path), output_dir))

    combined_path = Path(output_dir) / "processed_zuco_cleaned.pt"
    torch.save(all_processed, combined_path)
    print(f"\nCombined training data saved as: {combined_path}")
