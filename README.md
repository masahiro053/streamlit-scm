# SCM (Synthetic Control Method) Learning Dashboard

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![SciPy](https://img.shields.io/badge/SciPy-Optimization-green.svg)](https://scipy.org/)

## 📌 概要 (Overview)
本アプリケーションは、因果推論手法の一つである「**合成コントロール法（Synthetic Control Method: SCM）**」の論理構造を理解し、手元で検証（PoC）を行うための個人学習用Streamlitダッシュボードです。

「特定の施策（介入）が行われなかった場合の仮想的な結果（反事実）」を、介入を受けていない複数のドナー群の重み付き平均によって合成し、プロモーション等の純効果を推定する一連のプロセスを直感的に実行・確認することができます。

## ✨ 主な機能 (Key Features)

1. **シミュレーションデータの生成機能 (Data Generation)**
   * トレンドと季節性を持つ時系列データをベースに、介入効果（正解リフト）やノイズを指定してダミーデータを自動生成
   * アルゴリズムが真の効果量を正しく推定できるか、正解のある環境で挙動テストが可能
2. **外部データのアップロード分析 (Custom Data Analysis)**
   * 手元の時系列データ（CSV/XLSXのWide形式）をアップロード可能
   * UI上で「時間軸」「ターゲット」「ドナー群」「介入時点」を柔軟にマッピングして分析を実行
3. **ハイパーパラメータの調整 (Hyperparameter Tuning)**
   * L2ペナルティ（Ridgeペナルティ）をスライダーで動的に調整
   * 少数ドナーへの重みの極端な偏り（過学習）を防ぐ、バリアンスとバイアスのトレードオフ検証
4. **主要KPIの算出と可視化 (KPI & Visualization)**
   * **Model Fit**: Pre期間のMAPE (%) による事前予測精度の確認
   * **Causal Effect**: 平均リフト効果、合計リフト効果、相対リフト効果 (%) の自動計算
   * **Weights Distribution**: 各ドナーへの重み割り当て状況をグラフ化

## 🛠 技術スタック (Tech Stack)
- **Frontend / Backend**: Streamlit
- **Optimization Engine**: SciPy (`scipy.optimize.minimize`)
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, japanize-matplotlib
- **File I/O**: openpyxl (Excel読み込み用)

## 🚀 デプロイと環境構築の注意点 (Deployment Notes)
本アプリケーションを実行する際は、Python 3.9以上の環境を推奨します。
日本語フォントの文字化けを防ぐため、`japanize-matplotlib` を含めた以下のライブラリ群をインストールしてください。

```text
streamlit
numpy
pandas
scipy
matplotlib
japanize-matplotlib
openpyxl
```

ローカル環境で立ち上げる場合は、ディレクトリ内で以下のコマンドを実行します。

Bash
```
pip install -r requirements.txt
streamlit run app.py
```