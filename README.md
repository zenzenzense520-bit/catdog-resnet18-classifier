# CatDog ResNet18 Classifier

基于 ImageNet 预训练 ResNet18 的猫狗图片二分类项目，包含数据增强、分层训练/验证划分、迁移学习、训练曲线、最佳模型保存和单图推理。

## 快速上手

需要 [uv](https://docs.astral.sh/uv/)；Windows 用户建议安装 [Git for Windows](https://git-scm.com/download/win) 后在 Git Bash 中执行脚本。

```bash
git clone https://github.com/zenzenzense520-bit/catdog-resnet18-classifier.git
cd catdog-resnet18-classifier

bash scripts/download_data.sh   # 下载 Kaggle Cats vs Dogs 数据（Microsoft 官方镜像）
bash scripts/setup.sh           # uv sync 创建 .venv 并安装固定版本依赖
bash scripts/run_train.sh --data-dir data --epochs 10 --batch-size 32   # 训练
bash scripts/run_predict.sh --image path/to/test.jpg --model outputs/best_model.pt  # 单图推理
bash scripts/run_tests.sh       # 运行测试
```

训练产物位于 `outputs/`（`best_model.pt` 最佳模型、`training_curves.png` 训练曲线、`history.json` 每轮指标），日志位于 `logs/`。训练与推理自动选择 CUDA、MPS 或 CPU。

详细的数据结构、训练和推理命令见 [`docs/使用说明.md`](docs/使用说明.md)。
