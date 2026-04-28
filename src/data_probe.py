import scipy.io
import numpy as np
import os
from pathlib import Path

def probe_zuco_refined(file_path):
    print(f"\n{'#'*40}")
    print(f"Refined Probing: {os.path.basename(file_path)}")
    print(f"{'#'*40}")
    
    try:
        # 按照建议：squeeze_me=True 移除冗余维度, struct_as_record=False 将结构体转换为对象
        mat = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
        
        if not hasattr(mat, 'keys'):
            # 当 struct_as_record=False 时，loadmat 返回的是 dict，但内部结构体是对象
            pass
        
        # 1. 查找 sentenceData
        if 'sentenceData' not in mat:
            print(f"Error: 'sentenceData' not found in {list(mat.keys())}")
            return
        
        sentence_data = mat['sentenceData']
        print(f"Total Sentences Found: {len(sentence_data)}")
        
        # 2. 遍历前 3 个句子
        for i in range(min(3, len(sentence_data))):
            sentence = sentence_data[i]
            print(f"\n--- Sentence {i+1} ---")
            
            # 文本内容 (content)
            # 在 struct_as_record=False 模式下，通过属性访问
            content = sentence.content
            print(f"Content: {content}")
            
            # 包含的单词数量
            words = sentence.word
            num_words = len(words)
            print(f"Number of Words: {num_words}")
            
            # 探索单词级的 EEG 维度
            if num_words > 0:
                first_word = words[0]
                # 根据之前的探测，ZuCo 1.0/2.0 的单词脑电通常在 rawEEG 字段
                # 我们检查几个可能的字段名
                eeg_field = None
                for field in ['rawEEG', 'EEG_RAW', 'mean_EEG']:
                    if hasattr(first_word, field):
                        eeg_field = field
                        break
                
                if eeg_field:
                    eeg_data = getattr(first_word, eeg_field)
                    # rawEEG 通常是 object 数组（cell array），每个元素是一个 fixation 的 EEG
                    if isinstance(eeg_data, np.ndarray) and eeg_data.dtype == np.object_:
                        print(f"EEG Field Found: '{eeg_field}' (Object Array)")
                        if eeg_data.size > 0:
                            sample_eeg = eeg_data[0] # 取第一个 fixation 的 EEG
                            print(f"  Shape of first fixation EEG: {sample_eeg.shape} (Channels, Timepoints)")
                    else:
                        print(f"EEG Field Found: '{eeg_field}' | Shape: {getattr(eeg_data, 'shape', 'N/A')}")
                else:
                    # 如果单词级没找到，看看句子级是否有 rawData
                    if hasattr(sentence, 'rawData'):
                        print(f"Note: Word-level EEG not found, Sentence-level 'rawData' shape: {sentence.rawData.shape}")
                    else:
                        print("Warning: No recognized EEG fields found.")

    except Exception as e:
        print(f"Error during refined probing: {e}")
        import traceback
        traceback.print_exc()

def main():
    # 针对你提供的具体文件进行探测
    target_file = Path("./data/resultsZAB_SR.mat")
    
    if target_file.exists():
        probe_zuco_refined(str(target_file))
    else:
        # 回退到扫描 data 目录
        data_dir = Path("./data")
        mat_files = list(data_dir.glob("*.mat"))
        if not mat_files:
            print(f"File {target_file} not found and no other .mat files in ./data")
            return
        for f in mat_files:
            probe_zuco_refined(str(f))

if __name__ == "__main__":
    main()
