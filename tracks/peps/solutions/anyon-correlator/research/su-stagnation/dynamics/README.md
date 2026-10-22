# SU 闭式更新与非基态固定点

对明确的 D=2 态族 |ψ(K)⟩∝∏ₚexp(KₚBₚ)|+⟩，推导原 L 形 SU 的分段更新映射。无拟合参数的公式预测旧 dt=0.05 停滞能量到约 2×10⁻¹²/spin。在所分析的表示、权重、门序和截断分支内，小步长极限满足 dK/dτ=1−2tanh(2K)，稳定点为 K*=ln(3)/4、⟨B⟩=1/2、e=−3/4。

推导、8 条轨迹、截断通道比较、独立无限 CTMRG 测量及适用范围见 [REPORT.md](REPORT.md)。[fixed_points.csv](fixed_points.csv)、[dt_scaling.csv](dt_scaling.csv)、[summary.json](summary.json) 保存主要数值，`results/` 保存本地原始数据。

本部分使用 `../single-step/` 的小 torus 和 Pauli 算符函数，以及项目共同模型。基础测试和完整重算命令见 [上级说明](../README.md)。
