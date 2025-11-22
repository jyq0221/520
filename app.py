import argparse
import math
import sys
from io import StringIO
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import math
from typing import Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Constants
EPSILON_0 = 8.854187817e-12  # F/m


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric, coercing errors to NaN."""
    return pd.to_numeric(series, errors="coerce")


def compute_bridge_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived values for the bridge method table."""
    df = raw.copy()
    df.columns = [
        "样品",
        "D/mm",
        "h/mm",
        "C/pF",
        "Ck/pF",
        "C0/pF",
    ]

    df["D/mm"] = _safe_numeric(df["D/mm"])
    df["h/mm"] = _safe_numeric(df["h/mm"])
    df["C/pF"] = _safe_numeric(df["C/pF"])
    df["Ck/pF"] = _safe_numeric(df["Ck/pF"])
    df["C0/pF"] = _safe_numeric(df["C0/pF"])

    area_mm2 = math.pi * (df["D/mm"] / 2) ** 2
    df["S/mm²"] = area_mm2

    calculated_c0 = 0.008854187817 * area_mm2 / df["h/mm"]
    df["C0/pF"] = df["C0/pF"].fillna(calculated_c0)

    df["εr"] = (df["C/pF"] - df["Ck/pF"]) / df["C0/pF"]

    return df


def regression_with_intercept(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return slope, intercept, and R² for y = slope * x + intercept."""
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else math.nan
    r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    return slope, intercept, r2


def compute_air_regression(raw: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    df = raw.copy()
    df.columns = ["d/mm", "C/pF"]
    df["d/mm"] = _safe_numeric(df["d/mm"])
    df["C/pF"] = _safe_numeric(df["C/pF"])

    df = df.dropna(subset=["d/mm", "C/pF"])
    if len(df) < 2:
        return df, {}

    df["1/d (1/mm)"] = 1 / df["d/mm"]
    slope, intercept, r2 = regression_with_intercept(df["1/d (1/mm)"].to_numpy(), df["C/pF"].to_numpy())
    df["拟合C/pF"] = slope * df["1/d (1/mm)"] + intercept

    # ε0 estimation from C = ε0 * S / d + Ck; slope uses x = 1/d_mm
    # ε0 = slope_pF * 1e-9 / S_mm2  (derived from unit conversions)
    return df, {"slope": slope, "intercept": intercept, "r2": r2}


def estimate_epsilon0(slope_pf: float, area_mm2: float) -> float:
    return slope_pf * 1e-9 / area_mm2 if area_mm2 else math.nan


def compute_coaxial_results(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = ["r0/mm", "R/mm", "L/cm", "C/pF", "备注"]
    for col in ["r0/mm", "R/mm", "L/cm", "C/pF"]:
        df[col] = _safe_numeric(df[col])

    ln_ratio = np.log(df["R/mm"] / df["r0/mm"])
    length_m = df["L/cm"] * 0.01
    capacitance_f = df["C/pF"] * 1e-12
    df["εr"] = capacitance_f * ln_ratio / (2 * math.pi * EPSILON_0 * length_m)
    return df


def _load_or_default(path: Optional[str], default: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if path is None:
        return default
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    if list(df.columns) != columns:
        df.columns = columns
    return df


def run_bridge(csv_path: Optional[str]) -> None:
    default = pd.DataFrame(
        [
            ["样品A", 50, 2.0, 60.0, 2.5, None],
            ["样品B", 45, 1.5, 55.2, 2.5, None],
        ],
        columns=["样品", "D/mm", "h/mm", "C/pF", "Ck/pF", "C0/pF"],
    )
    df = _load_or_default(csv_path, default, list(default.columns))
    results = compute_bridge_results(df)
    print("电桥法计算结果：")
    print(results.to_string(index=False))


def run_air(csv_path: Optional[str], area: float, plot_path: Optional[str]) -> None:
    default = pd.DataFrame(
        [
            [2.5, 42.6],
            [3.0, 35.1],
            [3.5, 30.8],
            [4.0, 27.0],
            [4.5, 24.1],
            [5.0, 21.8],
        ],
        columns=["d/mm", "C/pF"],
    )
    df = _load_or_default(csv_path, default, list(default.columns))
    results, stats = compute_air_regression(df)
    if not stats:
        print("需要至少两组有效 (d, C) 数据进行回归。")
        return

    epsilon0_est = estimate_epsilon0(stats["slope"], area)
    print("回归结果：")
    print(results.to_string(index=False))
    print(
        f"\n斜率 B (pF·mm): {stats['slope']:.4f}\n"
        f"截距 A (pF): {stats['intercept']:.4f}\n"
        f"R²: {stats['r2']:.4f}\n"
        f"估计 ε0 (F/m): {epsilon0_est:.3e}"
    )

    if plot_path:
        plt.figure(figsize=(6, 4))
        plt.scatter(results["1/d (1/mm)"], results["C/pF"], label="测量值")
        plt.plot(results["1/d (1/mm)"], results["拟合C/pF"], color="orange", label="拟合")
        plt.xlabel("1/d (1/mm)")
        plt.ylabel("C/pF")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        print(f"拟合图已保存到 {plot_path}")


def run_coax(csv_path: Optional[str]) -> None:
    default = pd.DataFrame(
        [
            [1.2, 4.8, 30.0, 68.0, ""],
            [1.2, 4.8, 45.0, 72.5, ""],
        ],
        columns=["r0/mm", "R/mm", "L/cm", "C/pF", "备注"],
    )
    df = _load_or_default(csv_path, default, list(default.columns))
    results = compute_coaxial_results(df)
    print("同轴线法计算结果：")
    print(results.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="物理实验数据处理（OLS 回归）")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_bridge = subparsers.add_parser("bridge", help="电桥法计算固体介电常数")
    p_bridge.add_argument("--csv", help="包含样品, D/mm, h/mm, C/pF, Ck/pF, C0/pF 列的CSV")

    p_air = subparsers.add_parser("air", help="回归法估计空气介电常数")
    p_air.add_argument("--csv", help="包含 d/mm, C/pF 列的CSV")
    p_air.add_argument("--area", type=float, default=500.0, help="电极有效面积 S (mm²)")
    p_air.add_argument("--plot", help="保存拟合图的路径，如 output.png")

    p_coax = subparsers.add_parser("coax", help="同轴线电容法计算")
    p_coax.add_argument("--csv", help="包含 r0/mm, R/mm, L/cm, C/pF, 备注 列的CSV")

    return parser


def main():
    if len(sys.argv) == 1:
        launch_gui()
        return

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "bridge":
        run_bridge(args.csv)
    elif args.command == "air":
        run_air(args.csv, args.area, args.plot)
    elif args.command == "coax":
        run_coax(args.csv)


def _string_to_df(content: str, columns: list[str]) -> pd.DataFrame:
    """Parse CSV-like text into a DataFrame with the desired columns."""

    data = pd.read_csv(StringIO(content))
    if list(data.columns) != columns:
        data.columns = columns[: len(data.columns)]
    return data


def _load_csv_via_dialog(text_widget, columns: list[str]) -> None:
    from tkinter import filedialog, messagebox

    path = filedialog.askopenfilename(filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")])
    if not path:
        return
    try:
        content = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        messagebox.showerror(
            "读取失败",
            "文件似乎不是文本或 CSV 格式（可能是 Excel 或二进制文件），请先另存为 UTF-8 编码的 CSV 再加载。",
        )
        return
    except OSError as exc:
        messagebox.showerror("读取失败", str(exc))
        return
    text_widget.delete("1.0", "end")
    text_widget.insert("1.0", content)


def _create_text_block(parent, default_text: str):
    from tkinter import scrolledtext

    text = scrolledtext.ScrolledText(parent, height=10, width=80)
    text.insert("1.0", default_text)
    text.pack(fill="x", padx=8, pady=(0, 4))
    return text


def _create_result_box(parent):
    from tkinter import scrolledtext

    text = scrolledtext.ScrolledText(parent, height=12, width=80, state="disabled")
    text.pack(fill="both", expand=True, padx=8, pady=(4, 8))
    return text


def _render_results(box, content: str):
    box.configure(state="normal")
    box.delete("1.0", "end")
    box.insert("1.0", content)
    box.configure(state="disabled")


def launch_gui():
    import tkinter as tk
    from tkinter import ttk, messagebox

    root = tk.Tk()
    root.title("物理实验数据处理 (OLS 回归)")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # Bridge tab
    bridge_tab = ttk.Frame(notebook)
    notebook.add(bridge_tab, text="电桥法")

    ttk.Label(
        bridge_tab,
        text="输入或粘贴 CSV (样品,D/mm,h/mm,C/pF,Ck/pF,C0/pF)，可留空 C0/pF 自动计算",
    ).pack(anchor="w", padx=8, pady=4)

    bridge_default = """样品,D/mm,h/mm,C/pF,Ck/pF,C0/pF
样品A,50,2.0,60.0,2.5,
样品B,45,1.5,55.2,2.5,
"""
    bridge_input = _create_text_block(bridge_tab, bridge_default)
    ttk.Button(bridge_tab, text="选择CSV", command=lambda: _load_csv_via_dialog(bridge_input, [])).pack(
        anchor="e", padx=8, pady=(0, 4)
    )
    bridge_result = _create_result_box(bridge_tab)

    def compute_bridge_cb():
        try:
            df = _string_to_df(bridge_input.get("1.0", "end"), ["样品", "D/mm", "h/mm", "C/pF", "Ck/pF", "C0/pF"])
            results = compute_bridge_results(df)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("计算失败", str(exc))
            return
        _render_results(bridge_result, results.to_string(index=False))

    ttk.Button(bridge_tab, text="计算", command=compute_bridge_cb).pack(anchor="e", padx=8, pady=4)

    # Air regression tab
    air_tab = ttk.Frame(notebook)
    notebook.add(air_tab, text="回归法 (空气)")
    ttk.Label(air_tab, text="输入 CSV (d/mm,C/pF)，可选电极面积与拟合图保存路径").pack(anchor="w", padx=8, pady=4)

    air_default = """d/mm,C/pF
2.5,42.6
3.0,35.1
3.5,30.8
4.0,27.0
4.5,24.1
5.0,21.8
"""
    air_input = _create_text_block(air_tab, air_default)

    area_frame = ttk.Frame(air_tab)
    area_frame.pack(fill="x", padx=8)
    ttk.Label(area_frame, text="电极面积 S (mm²)：").pack(side="left")
    area_var = tk.StringVar(value="500")
    ttk.Entry(area_frame, textvariable=area_var, width=10).pack(side="left", padx=(0, 12))
    ttk.Label(area_frame, text="拟合图保存路径 (可空)：").pack(side="left")
    plot_var = tk.StringVar(value="")
    ttk.Entry(area_frame, textvariable=plot_var, width=24).pack(side="left")

    air_result = _create_result_box(air_tab)

    def compute_air_cb():
        try:
            df = _string_to_df(air_input.get("1.0", "end"), ["d/mm", "C/pF"])
            area_val = float(area_var.get()) if area_var.get() else 0.0
            results, stats = compute_air_regression(df)
            if not stats:
                raise ValueError("需要至少两组有效 (d, C) 数据进行回归")
            epsilon0_est = estimate_epsilon0(stats["slope"], area_val)
            report = (
                results.to_string(index=False)
                + "\n\n"
                + f"斜率 B (pF·mm): {stats['slope']:.4f}\n"
                + f"截距 A (pF): {stats['intercept']:.4f}\n"
                + f"R²: {stats['r2']:.4f}\n"
                + f"估计 ε0 (F/m): {epsilon0_est:.3e}"
            )
            _render_results(air_result, report)

            plot_path = plot_var.get().strip()
            if plot_path:
                plt.figure(figsize=(6, 4))
                plt.scatter(results["1/d (1/mm)"], results["C/pF"], label="测量值")
                plt.plot(results["1/d (1/mm)"], results["拟合C/pF"], color="orange", label="拟合")
                plt.xlabel("1/d (1/mm)")
                plt.ylabel("C/pF")
                plt.legend()
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150)
                messagebox.showinfo("已保存", f"拟合图保存到: {plot_path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("计算失败", str(exc))

    ttk.Button(air_tab, text="计算", command=compute_air_cb).pack(anchor="e", padx=8, pady=4)

    # Coaxial tab
    coax_tab = ttk.Frame(notebook)
    notebook.add(coax_tab, text="同轴线法")
    ttk.Label(coax_tab, text="输入 CSV (r0/mm,R/mm,L/cm,C/pF,备注)").pack(anchor="w", padx=8, pady=4)

    coax_default = """r0/mm,R/mm,L/cm,C/pF,备注
1.2,4.8,30.0,68.0,
1.2,4.8,45.0,72.5,
"""
    coax_input = _create_text_block(coax_tab, coax_default)
    coax_result = _create_result_box(coax_tab)

    def compute_coax_cb():
        try:
            df = _string_to_df(coax_input.get("1.0", "end"), ["r0/mm", "R/mm", "L/cm", "C/pF", "备注"])
            results = compute_coaxial_results(df)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("计算失败", str(exc))
            return
        _render_results(coax_result, results.to_string(index=False))

    ttk.Button(coax_tab, text="计算", command=compute_coax_cb).pack(anchor="e", padx=8, pady=4)

    root.mainloop()


if __name__ == "__main__":
    main()
st.set_page_config(page_title="物理实验数据处理", layout="wide")
st.title("物理实验数据处理（OLS回归）")

with st.expander("使用说明", expanded=True):
    st.markdown(
        """
        - 参考图片中的实验流程，提供三种典型场景：
          1. **电桥法测固体电介质相对介电常数**：输入几何尺寸与电容读数，自动计算 $S$、$C_0$ 与 $\varepsilon_r$。
          2. **回归法测定空气介电常数**：录入不同电极间距下的电容值，对 $C$ 与 $1/d$ 做线性回归，给出斜率、截距与 $R^2$，并估计 $\varepsilon_0$。
          3. **同轴线电容法**：根据同轴参数计算介电常数，便于与前两种方法交叉验证。
        - 可直接在表格中双击编辑数据，更新后所有结果会自动刷新。
        """
    )

# 1) Bridge method for solid dielectrics
st.header("电桥法测固体电介质相对介电常数数据")
bridge_default = pd.DataFrame(
    [
        ["样品A", 50, 2.0, 60.0, 2.5, None],
        ["样品B", 45, 1.5, 55.2, 2.5, None],
    ],
    columns=["样品", "D/mm", "h/mm", "C/pF", "Ck/pF", "C0/pF"],
)
bridge_df = st.data_editor(
    bridge_default,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "样品": st.column_config.TextColumn("样品"),
        "D/mm": st.column_config.NumberColumn("D/mm", help="平板电极直径(mm)"),
        "h/mm": st.column_config.NumberColumn("h/mm", help="样品厚度(mm)"),
        "C/pF": st.column_config.NumberColumn("C/pF", help="含样品测得的电容"),
        "Ck/pF": st.column_config.NumberColumn("Ck/pF", help="杂散电容"),
        "C0/pF": st.column_config.NumberColumn("C0/pF", help="真空电容(留空则自动计算)"),
    },
)
bridge_results = compute_bridge_results(bridge_df)
st.dataframe(bridge_results, use_container_width=True, height=220)

st.caption(
    """
    计算公式：$S = \pi (D/2)^2$；$C_0 = \varepsilon_0 S / h$（单位换算后 $C_0/\mathrm{pF}=0.008854 S_{\mathrm{mm}^2}/h_{\mathrm{mm}}$）；
    $\varepsilon_r = (C-C_k)/C_0$。
    """
)

# 2) Air permittivity via regression
st.header("回归法测定空气介电常数数据")
air_default = pd.DataFrame(
    [
        [2.5, 42.6],
        [3.0, 35.1],
        [3.5, 30.8],
        [4.0, 27.0],
        [4.5, 24.1],
        [5.0, 21.8],
    ],
    columns=["d/mm", "C/pF"],
)
air_df = st.data_editor(
    air_default,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "d/mm": st.column_config.NumberColumn("d/mm", help="电极间距"),
        "C/pF": st.column_config.NumberColumn("C/pF", help="测得电容"),
    },
)

col_area, col_display = st.columns([1, 3])
with col_area:
    area_mm2 = st.number_input("电极有效面积 S (mm²)", value=500.0, min_value=1.0)
    st.markdown("使用 $C = \varepsilon_0 S / d + C_k$ 近似模型")

air_results, air_stats = compute_air_regression(air_df)
if air_stats:
    epsilon0_est = estimate_epsilon0(air_stats["slope"], area_mm2)
    with col_display:
        st.metric("斜率 B (pF·mm)", f"{air_stats['slope']:.4f}")
        st.metric("截距 A (pF)", f"{air_stats['intercept']:.4f}")
        st.metric("R²", f"{air_stats['r2']:.4f}")
        st.metric("估计 ε0 (F/m)", f"{epsilon0_est:.3e}")

    st.dataframe(air_results, use_container_width=True)

    chart = (
        alt.Chart(air_results)
        .mark_circle(size=90, color="#1f77b4")
        .encode(x="1/d (1/mm)", y="C/pF")
        .interactive()
    )
    line = (
        alt.Chart(air_results)
        .mark_line(color="#ff7f0e")
        .encode(x="1/d (1/mm)", y="拟合C/pF")
    )
    st.altair_chart(chart + line, use_container_width=True)
else:
    st.info("请至少输入两组有效的 (d, C) 数据以进行回归。")

st.caption(
    """
    线性模型：$C = B \cdot (1/d) + A$，其中 $B \approx 1000\,\varepsilon_0 S$；
    因而 $\varepsilon_0 \approx B \times 10^{-9} / S_{\mathrm{mm}^2}$。
    """
)

# 3) Coaxial cable method
st.header("同轴线电容法（扩展验证）")
coax_default = pd.DataFrame(
    [
        [1.2, 4.8, 30.0, 68.0, ""],
        [1.2, 4.8, 45.0, 72.5, ""],
    ],
    columns=["r0/mm", "R/mm", "L/cm", "C/pF", "备注"],
)
coax_df = st.data_editor(
    coax_default,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "r0/mm": st.column_config.NumberColumn("r0/mm", help="内导体半径"),
        "R/mm": st.column_config.NumberColumn("R/mm", help="外导体内径"),
        "L/cm": st.column_config.NumberColumn("L/cm", help="线长"),
        "C/pF": st.column_config.NumberColumn("C/pF", help="测得电容"),
        "备注": st.column_config.TextColumn("备注"),
    },
)
coax_results = compute_coaxial_results(coax_df)
st.dataframe(coax_results, use_container_width=True)

st.caption(
    """
    同轴线公式：$\varepsilon_r = \dfrac{C\,\ln(R/r_0)}{2\pi\varepsilon_0 L}$，其中 $C$ 以法拉计，$L$ 以米计。
    """
)
