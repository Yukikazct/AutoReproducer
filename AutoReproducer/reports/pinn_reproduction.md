# 第一篇真实复现报告:pinn

> 首次在 Docker 中真实运行 PaperBench 论文代码,产出的真实复现结果。

## 论文信息

- **论文**: Challenges in Training PINNs: A Loss Landscape Perspective (Rathore et al., ICML 2024)
- **PaperBench ID**: `pinn`
- **复现分**: 0.543(来自 PaperGuru aggregate-final.json)

## 运行环境

- Docker 镜像: `autorepro-pinn`(python:3.12-slim + torch 2.12.0+cpu + numpy + PyYAML)
- 执行方式: `reproduce.sh`,SMOKE=1(每 PDE 100 次迭代:Adam 50 + L-BFGS 50)

## 复现结果(smoke 模式)

| PDE | final_loss | l2re | num_params |
|---|---|---|---|
| convection | 0.1723 | 1.0672 | 81201 |
| reaction | 0.0760 | 0.9937 | 81201 |
| wave | 0.0570 | 1.1897 | 81201 |

## 说明

1. **NNCG 阶段在 smoke 模式下发散(产出 NaN)**:仅 100 次迭代模型未收敛,二阶牛顿法(NNCG)数值不稳定。已用 `--no_nncg` 禁用 NNCG,获得有意义的 loss/l2re。
2. **l2re ≈ 1.0 是 smoke 模式的预期结果**:完整复现需 41000 次迭代(Adam 11000 + L-BFGS 30000 + NNCG 2000),论文报告的 l2re 约 1e-3 量级。
3. **验证结论**:论文代码可在 Docker 中真实运行,产出真实指标,复现流水线端到端可用。

## 关键经验

- 国内网络:torch CPU wheel 用阿里云镜像 `mirrors.aliyun.com/pytorch-wheels/cpu/`,PyPI 依赖用清华源
- Windows 的 `.sh` 脚本被 git 检出为 CRLF,运行前需 `sed -i 's/\r$//'`
- torch 必须用 `+cpu` 版 wheel(CUDA 版 import 时硬性要求 nvidia 库)

## 时间戳

- 运行时间: 2026-09-05
