import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- 1. SCMコアロジック ---
def optimize_scm(df, target_col, donor_cols, t_intervention_idx, l2_weight):
    """SCMの重み最適化を実行する関数"""
    df_pre = df.iloc[:t_intervention_idx]
    X_pre = df_pre[donor_cols].values
    y_pre = df_pre[target_col].values

    def objective_function(w, X, y, l2_pen):
        synthetic_y = np.dot(X, w)
        mse = np.sum((y - synthetic_y)**2)
        return mse + l2_pen * np.sum(w**2)

    n_donors = len(donor_cols)
    w0 = np.ones(n_donors) / n_donors
    bounds = [(0, 1) for _ in range(n_donors)]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    result = minimize(
        objective_function, 
        w0, 
        args=(X_pre, y_pre, l2_weight),
        bounds=bounds,
        constraints=constraints
    )
    return result.x

# --- 2. UIセットアップ ---
st.set_page_config(page_title="SCM PoC Dashboard", layout="wide")
st.title("合成コントロール法 (SCM) 分析ダッシュボード")

st.sidebar.header("最適化パラメータ")
l2_penalty = st.sidebar.slider("L2ペナルティ (正則化)", 0.0, 1000.0, 0.0, 10.0)

# データ保持用の変数
df = None
target_col = None
donor_cols = []
t_intervention_idx = None
time_col = None

# --- 3. データソースの選択 ---
tab1, tab2 = st.tabs(["テストデータの生成", "ファイルのアップロード (CSV/XLSX)"])

with tab1:
    st.subheader("シミュレーションデータの生成パラメータ")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        gen_n_periods = st.number_input("全期間 (時点数)", min_value=50, max_value=500, value=100)
        gen_n_donors = st.number_input("ドナー数", min_value=2, max_value=20, value=5)
    with col_b:
        gen_t_intervention = st.number_input("介入開始時点", min_value=10, max_value=gen_n_periods-10, value=int(gen_n_periods*0.7))
        gen_effect_size = st.number_input("介入効果 (リフト量)", value=20.0)
    with col_c:
        gen_noise = st.number_input("ノイズの大きさ", min_value=0.0, value=2.0)

    if st.button("データを生成"):
        np.random.seed(42)
        time_arr = np.arange(gen_n_periods)
        base_trend = time_arr * 0.5
        seasonality = np.sin(time_arr / 5) * 10
        
        tmp_df = pd.DataFrame({'Time': time_arr})
        gen_donor_names = [f'Donor_{i}' for i in range(1, gen_n_donors + 1)]
        
        for donor in gen_donor_names:
            tmp_df[donor] = base_trend + seasonality + np.random.normal(0, 5, gen_n_periods) + np.random.uniform(10, 50)
            
        true_weights = np.random.dirichlet(np.ones(gen_n_donors))
        tmp_df['Target'] = tmp_df[gen_donor_names].dot(true_weights) + np.random.normal(0, gen_noise, gen_n_periods)
        tmp_df.loc[gen_t_intervention:, 'Target'] += gen_effect_size
        
        st.session_state['df'] = tmp_df
        st.session_state['target_col'] = 'Target'
        st.session_state['donor_cols'] = gen_donor_names
        st.session_state['time_col'] = 'Time'
        st.session_state['t_idx'] = gen_t_intervention

with tab2:
    st.subheader("分析データのアップロード")
    uploaded_file = st.file_uploader("ファイルをアップロードしてください", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                uploaded_df = pd.read_csv(uploaded_file)
            else:
                uploaded_df = pd.read_excel(uploaded_file)
                
            st.dataframe(uploaded_df.head())
            
            # カラム設定UI
            all_columns = uploaded_df.columns.tolist()
            sel_time = st.selectbox("時間カラム (X軸用)", all_columns)
            sel_target = st.selectbox("ターゲットカラム (介入を受けたユニット)", all_columns)
            
            # ターゲットと時間以外の数値カラムをドナー候補とする
            numeric_cols = uploaded_df.select_dtypes(include=[np.number]).columns.tolist()
            default_donors = [c for c in numeric_cols if c not in [sel_time, sel_target]]
            sel_donors = st.multiselect("ドナーカラム (比較対象ユニット)", numeric_cols, default=default_donors)
            
            sel_intervention_val = st.selectbox("介入開始時点", uploaded_df[sel_time].tolist())
            
            if st.button("アップロードデータで分析実行"):
                st.session_state['df'] = uploaded_df
                st.session_state['target_col'] = sel_target
                st.session_state['donor_cols'] = sel_donors
                st.session_state['time_col'] = sel_time
                st.session_state['t_idx'] = uploaded_df[uploaded_df[sel_time] == sel_intervention_val].index[0]
        except Exception as e:
            st.error(f"ファイルの読み込みに失敗しました: {e}")

# --- 4. 分析と可視化の実行 ---
if 'df' in st.session_state and st.session_state['df'] is not None:
    df = st.session_state['df']
    target_col = st.session_state['target_col']
    donor_cols = st.session_state['donor_cols']
    time_col = st.session_state['time_col']
    t_intervention_idx = st.session_state['t_idx']

    st.markdown("---")
    st.header("分析結果")
    
    if len(donor_cols) < 1:
        st.error("ドナーを1つ以上選択してください。")
    else:
        # 最適化実行
        optimal_weights = optimize_scm(df, target_col, donor_cols, t_intervention_idx, l2_penalty)
        
        # 反事実と効果の計算
        df['Synthetic_Target'] = df[donor_cols].dot(optimal_weights)
        df['Causal_Effect'] = df[target_col] - df['Synthetic_Target']
        
        post_effect = df.loc[t_intervention_idx:, 'Causal_Effect'].mean()
        st.metric(f"介入後 ({df.loc[t_intervention_idx, time_col]} 以降) の平均効果", f"{post_effect:.2f}")

        # グラフ描画
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        x_values = df[time_col] if time_col else df.index

        # 上段
        ax1.plot(x_values, df[target_col], label='Actual Target', color='black', linewidth=2)
        ax1.plot(x_values, df['Synthetic_Target'], label='Synthetic Target', color='blue', linestyle='--')
        ax1.axvline(x=x_values.iloc[t_intervention_idx], color='red', linestyle=':')
        ax1.set_title(f'SCM Result: {target_col}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 下段
        ax2.plot(x_values, df['Causal_Effect'], label='Causal Effect (Gap)', color='green')
        ax2.axhline(y=0, color='black', linestyle='-')
        ax2.axvline(x=x_values.iloc[t_intervention_idx], color='red', linestyle=':')
        ax2.set_title('Estimated Effect')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        st.pyplot(fig)

        # 重み分布の表示
        st.subheader("算出された重み (W)")
        weight_df = pd.DataFrame({
            'Donor': donor_cols,
            'Weight': optimal_weights
        }).sort_values('Weight', ascending=False)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(weight_df.style.format({'Weight': '{:.4f}'}))
        with col2:
            st.bar_chart(weight_df.set_index('Donor'))