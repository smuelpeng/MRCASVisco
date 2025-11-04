# VisCo Attack Implementation

基于论文 "Visual Contextual Attack: Jailbreaking MLLMs with Image-Driven Context Injection" 的复现实现。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置模型路径

编辑 `configs/config.yaml`，设置模型路径：

```yaml
models:
  target_model:
    model_path: "Qwen/Qwen2.5-VL-7B-Instruct"  # 或本地路径
  sdxl:
    model_path: "stabilityai/stable-diffusion-xl-base-1.0"  # 或本地路径
```

### 3. 运行示例

```bash
sh examples/run.sh
```

或手动运行：

```bash
python examples/demo_vh.py --data-dir data/data/VH --json-file VH_flag_4o.json --index 1
```

## 当前实现

- ✅ **VH (Exploiting Image Hallucination)** - 图像幻觉利用策略
- ⏸️ VS, VM, VI 策略（待完善）

## 项目结构

```
MRCASVisco/
├── visco/                  # 核心实现
│   ├── models/            # 模型封装
│   ├── components/        # 攻击组件
│   └── pipeline.py        # 攻击流程
├── examples/
│   ├── demo_vh.py         # VH 策略示例
│   └── run.sh             # 快速运行脚本
├── configs/
│   └── config.yaml        # 配置文件
└── outputs/               # 输出结果
```

## 参考文献

```bibtex
@article{miao2025visco,
  title={Visual Contextual Attack: Jailbreaking MLLMs with Image-Driven Context Injection},
  author={Miao, Ziqi and Ding, Yi and Li, Lijun and Shao, Jing},
  journal={arXiv preprint arXiv:2507.02844},
  year={2025}
}
```

## 免责声明

本项目仅用于学术研究和安全评测目的，严禁用于实际攻击或生成有害内容。
