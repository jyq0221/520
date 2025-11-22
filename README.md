# 物理实验数据处理（OLS 回归）

双击运行即会弹出桌面窗口，也可用命令行完成电桥法、回归法和同轴线法的数据处理。

## 快速开始

```bash
# 可选：创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate

pip install -r requirements.txt

# 双击 app.py 或在命令行运行打开图形界面
python app.py

# 继续支持命令行：电桥法示例（使用内置样例）
python app.py bridge

# 回归法示例（使用内置样例，指定电极面积并输出拟合图）
python app.py air --area 500 --plot fit.png

# 同轴线法示例
python app.py coax
```

也可以通过 `--csv` 传入自定义数据，CSV 列顺序应与内置表格一致：

- 电桥法：`样品, D/mm, h/mm, C/pF, Ck/pF, C0/pF`
- 回归法：`d/mm, C/pF`
- 同轴线法：`r0/mm, R/mm, L/cm, C/pF, 备注`

> 提示：界面和命令行仅接受 UTF-8 编码的纯文本 CSV。若选择 Excel 工作簿或二进制文件会弹出“文件似乎不是文本或 CSV 格式”的提示，请在 Excel 中另存为 CSV 后再加载。

## 打包 Windows 可执行文件

1. 安装打包依赖（需联网环境）：`pip install pyinstaller`
2. 在项目目录下执行：`pyinstaller --noconsole --onefile app.py --name dielectric-tool`
3. 打包完成后，在 `dist/dielectric-tool.exe` 直接双击即可弹出与上图类似的窗口。

## 主要公式

- 平板真空电容：\(C_0 = \varepsilon_0 S / h\)，单位换算后 \(C_0/\mathrm{pF} \approx 0.008854\, S_{\mathrm{mm}^2}/h_{\mathrm{mm}}\)
- 平板介质常数：\(\varepsilon_r = (C - C_k)/C_0\)
- 回归模型：\(C = B(1/d) + A\)，\(B \approx 1000\,\varepsilon_0 S\)，故 \(\varepsilon_0 \approx B \times 10^{-9} / S_{\mathrm{mm}^2}\)
- 同轴线：\(\varepsilon_r = \dfrac{C\,\ln(R/r_0)}{2\pi\varepsilon_0 L}\)

## 项目结构

- `app.py`：核心计算逻辑与命令行入口。
- `requirements.txt`：运行依赖列表。
- `proj/index.html`：原始示例文件，保留以防后续参考。
