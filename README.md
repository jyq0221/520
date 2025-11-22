# 物理实验数据处理（OLS 回归）

使用 Streamlit 编写的交互式工具，用于复现截图中的三类电容实验数据处理：

1. **电桥法测固体电介质相对介电常数**：录入样品直径、厚度、杂散电容与测量电容，自动计算面积、真空电容 \(C_0\) 和相对介电常数 \(\varepsilon_r\)。
2. **回归法测定空气介电常数**：对不同电极间距下的电容进行一元线性回归（自变量为 \(1/d\)），给出斜率、截距、\(R^2\) 与 \(\varepsilon_0\) 估计值，并绘制散点与拟合曲线。
3. **同轴线电容法**（扩展验证）：输入同轴线几何参数和电容值，利用解析公式计算介电常数。

## 快速开始

```bash
# 可选：创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

启动后浏览器访问 `http://localhost:8501`，即可按需编辑表格并查看自动更新的计算、回归结果与图表。

## 主要公式

- 平板真空电容：\(C_0 = \varepsilon_0 S / h\)，单位换算后 \(C_0/\mathrm{pF} \approx 0.008854\, S_{\mathrm{mm}^2}/h_{\mathrm{mm}}\)
- 平板介质常数：\(\varepsilon_r = (C - C_k)/C_0\)
- 回归模型：\(C = B(1/d) + A\)，\(B \approx 1000\,\varepsilon_0 S\)，故 \(\varepsilon_0 \approx B \times 10^{-9} / S_{\mathrm{mm}^2}\)
- 同轴线：\(\varepsilon_r = \dfrac{C\,\ln(R/r_0)}{2\pi\varepsilon_0 L}\)

## 项目结构

- `app.py`：Streamlit 前端与全部计算逻辑。
- `requirements.txt`：运行依赖列表。
- `proj/index.html`：原始示例文件，保留以防后续参考。
