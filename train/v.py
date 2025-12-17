#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import json
import sys
import argparse
from pathlib import Path

def extract(ckpt_path):
    print(f"Loading: {ckpt_path}")
    # 加载 ckpt
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # 获取 hyper_parameters
    hp = checkpoint.get("hyper_parameters", {})
    
    # 尝试提取 model_cfg
    model_cfg = {}
    if "model_cfg" in hp:
        print("Found 'model_cfg' in hyper_parameters.")
        model_cfg = hp["model_cfg"]
    elif "enc_cfg" in hp:
        print("Found 'enc_cfg' in hyper_parameters.")
        model_cfg = hp["enc_cfg"]
    else:
        print("WARNING: No explicit 'model_cfg' dictionary found. Dumping all hyper_parameters.")
        model_cfg = hp

    # 打印出来看看是不是这里有问题
    print("="*40)
    print("EXTRACTED CONFIG:")
    print(json.dumps(model_cfg, indent=2, default=str))
    print("="*40)

    # 保存路径
    ckpt_p = Path(ckpt_path)
    # 假设 run_dir 是 checkpoints 的上一级
    run_dir = ckpt_p.parent.parent
    records_dir = run_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = records_dir / "config.json"
    
    # 构造兼容格式 (包一层)
    final_json = {"model_cfg": model_cfg}
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
        
    print(f"Saved config to: {save_path}")
    print("Now you can run the evaluation script again.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str, help="Path to the .ckpt file")
    args = parser.parse_args()
    extract(args.ckpt_path)