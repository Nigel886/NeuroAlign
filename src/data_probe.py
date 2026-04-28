import os
import scipy.io
import numpy as np
from pathlib import Path

def print_structure(data, name="", indent=0):
    """递归打印 .mat 文件的结构和数据形状"""
    spacing = "  " * indent
    
    if isinstance(data, dict):
        print(f"{spacing}{name} (dict):")
        for key, value in data.items():
            if not key.startswith('__'):
                print_structure(value, key, indent + 1)
    
    elif isinstance(data, np.ndarray):
        # 处理 MATLAB struct 数组 (dtype=object)
        if data.dtype.names is not None:
            print(f"{spacing}{name} (struct array): shape={data.shape}, fields={data.dtype.names}")
            # 打印第一个元素的字段结构作为示例
            if data.size > 0:
                first_elem = data.flatten()[0]
                for field in data.dtype.names:
                    print_structure(first_elem[field], f"{field}", indent + 1)
        # 处理嵌套的 object 数组
        elif data.dtype == np.object_:
            print(f"{spacing}{name} (object array): shape={data.shape}")
            if data.size > 0 and indent < 3: # 限制递归深度
                print_structure(data.flatten()[0], f"{name}[0]", indent + 1)
        else:
            print(f"{spacing}{name} (ndarray): shape={data.shape}, dtype={data.dtype}")
            
    elif isinstance(data, (str, bytes)):
        print(f"{spacing}{name}: {type(data).__name__} = {data[:100]}...")
    
    else:
        # 其他基本类型
        val_str = str(data)[:100]
        print(f"{spacing}{name}: {type(data).__name__} = {val_str}")

def probe_zuco_file(file_path):
    print(f"\n{'='*20} Probing File: {os.path.basename(file_path)} {'='*20}")
    try:
        # mat_dtype=True 保持结构体，squeeze_me=True 减少多余维度
        data = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        
        # 查找可能的关键字段，如 'sentenceData'
        keys = [k for k in data.keys() if not k.startswith('__')]
        print(f"Top-level keys: {keys}")
        
        for key in keys:
            val = data[key]
            if hasattr(val, 'dtype') and val.dtype.names is not None:
                # 处理 MATLAB 结构体对象
                print(f"Key: {key} (MATLAB struct)")
                print(f"  Fields: {val.dtype.names}")
                # 探查第一个句子的结构
                if val.size > 0:
                    print(f"  Probing first element of {key}...")
                    first_trial = val[0]
                    for field in val.dtype.names:
                        field_val = getattr(first_trial, field)
                        shape = getattr(field_val, 'shape', 'N/A')
                        print(f"    - {field}: shape={shape}, type={type(field_val)}")
            else:
                print_structure(val, key, 1)
                
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

def main():
    data_dir = Path("./data")
    mat_files = list(data_dir.glob("*.mat"))
    
    if not mat_files:
        print(f"No .mat files found in {data_dir.absolute()}")
        print("Please place ZuCo .mat files in the 'data' directory.")
        return

    for mat_file in mat_files:
        probe_zuco_file(str(mat_file))

if __name__ == "__main__":
    main()
