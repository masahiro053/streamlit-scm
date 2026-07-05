import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import japanize_matplotlib

# --- 1. SCMコアロジック ---
def optimize_scm(df, target_col, donor_cols, t_intervention_idx, l2_weight):
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
st.info("このアプリは合成コントロール法を用いて、特定の施策（介入）が行われなかった場合の仮想的な結果（反事実）を予測し、施策の純効果を推定する学習用ツールです。")

st.sidebar.header("最適化パラメータ")
l2_penalty = st.sidebar.slider(
    "L2ペナルティ (正則化)", 
    0.0, 1000.0, 0.0, 10.0,
    help="値を大きくすると、少数のドナーに重みが極端に偏る（過学習する）のを防ぎます。Pre期間のMAPEが大きくなりすぎない範囲で調整してください。"
)

# --- 3. データソースの選択 ---
tab1, tab2 = st.tabs(["テストデータの生成", "ファイルのアップロード (CSV/XLSX)"])

with tab1:
    st.subheader("シミュレーションデータの生成")
    st.caption("SCMのアルゴリズムが正しく機能するかを検証するため、正解がわかっているダミーデータを生成します。")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        gen_n_periods = st.number_input("全期間 (時点数)", min_value=50, max_value=500, value=100, help="時系列の全体の長さを指定します。")
        gen_n_donors = st.number_input("ドナー数", min_value=2, max_value=20, value=5, help="比較対象となる未介入ユニットの数です。")
    with col_b:
        gen_t_intervention = st.number_input("介入開始時点", min_value=10, max_value=gen_n_periods-10, value=int(gen_n_periods*0.7), help="施策が開始されたタイミング（インデックス番号）です。")
        gen_effect_size = st.number_input("介入効果 (平均リフト量)", value=20.0, help="介入後に上乗せされる真の効果量です。モデルがこの値に近い結果を出せば成功です。")
    with col_c:
        gen_noise = st.number_input("ノイズの大きさ", min_value=0.0, value=2.0, help="ターゲットデータに付与するランダムノイズの分散です。値が小さいほど、アルゴリズムは正解を見つけやすくなります。")

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
        
        st.session_state['generated_df'] = tmp_df
        st.session_state['gen_target_col'] = 'Target'
        st.session_state['gen_donor_cols'] = gen_donor_names
        st.session_state['gen_time_col'] = 'Time'
        st.session_state['gen_t_idx'] = gen_t_intervention

    if 'generated_df' in st.session_state:
        st.markdown("### 生成されたデータのプレビュー")
        st.dataframe(st.session_state['generated_df'].head())
        
        col_dl, col_run = st.columns(2)
        with col_dl:
            csv = st.session_state['generated_df'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="生成されたデータをCSVとしてダウンロード",
                data=csv,
                file_name='scm_simulation_data.csv',
                mime='text/csv'
            )
        with col_run:
            if st.button("このデータでSCM分析を実行"):
                st.session_state['df'] = st.session_state['generated_df']
                st.session_state['target_col'] = st.session_state['gen_target_col']
                st.session_state['donor_cols'] = st.session_state['gen_donor_cols']
                st.session_state['time_col'] = st.session_state['gen_time_col']
                st.session_state['t_idx'] = st.session_state['gen_t_idx']

with tab2:
    st.subheader("分析データのアップロード")
    st.caption("お手持ちのデータ（CSVまたはXLSX）を読み込みます。データはWide形式（行が時間、列が各ユニット）であることを前提としています。")
    uploaded_file = st.file_uploader("ファイルをアップロード", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                uploaded_df = pd.read_csv(uploaded_file)
            else:
                uploaded_df = pd.read_excel(uploaded_file)
                
            st.dataframe(uploaded_df.head())
            
            all_columns = uploaded_df.columns.tolist()
            sel_time = st.selectbox("時間カラム (X軸用)", all_columns, help="時系列を表す列を選択してください。")
            sel_target = st.selectbox("ターゲットカラム (介入を受けたユニット)", all_columns, help="効果測定を行いたい対象の列を選択してください。")
            
            numeric_cols = uploaded_df.select_dtypes(include=[np.number]).columns.tolist()
            default_donors = [c for c in numeric_cols if c not in [sel_time, sel_target]]
            sel_donors = st.multiselect("ドナーカラム (比較対象ユニット)", numeric_cols, default=default_donors, help="反事実を作成するための材料となる、介入を受けていない列を複数選択してください。")
            
            sel_intervention_val = st.selectbox("介入開始時点", uploaded_df[sel_time].tolist(), help="施策が実行されたタイミングを選択してください。これより前のデータでモデルを学習します。")
            
            if st.button("アップロードデータで分析実行"):
                st.session_state['df'] = uploaded_df
                st.session_state['target_col'] = sel_target
                st.session_state['donor_cols'] = sel_donors
                st.session_state['time_col'] = sel_time
                st.session_state['t_idx'] = uploaded_df[uploaded_df[sel_time] == sel_intervention_val].index[0]
        except Exception as e:
            st.error(f"ファイルの読み込みに失敗しました: {e}")

# --- 4. 分析と可視化の実行 ---
if 'df' in st.session_state and st.session_state.get('target_col') is not None:
    df = st.session_state['df'].copy()
    target_col = st.session_state['target_col']
    donor_cols = st.session_state['donor_cols']
    time_col = st.session_state['time_col']
    t_intervention_idx = st.session_state['t_idx']

    st.markdown("---")
    st.header("分析結果")
    
    if len(donor_cols) < 1:
        st.error("エラー: ドナーを1つ以上選択してください。")
    else:
        # 最適化実行
        optimal_weights = optimize_scm(df, target_col, donor_cols, t_intervention_idx, l2_penalty)
        
        # 予測値と効果の算出
        df['Synthetic_Target'] = df[donor_cols].dot(optimal_weights)
        df['Causal_Effect'] = df[target_col] - df['Synthetic_Target']
        
        # --- 指標の計算 ---
        # Pre期間 (モデルの適合度)
        df_pre = df.iloc[:t_intervention_idx]
        mape_pre = np.mean(np.abs(df_pre['Causal_Effect'] / df_pre[target_col].replace(0, np.nan))) * 100
        
        # Post期間 (推定効果量)
        df_post = df.iloc[t_intervention_idx:]
        mean_effect = df_post['Causal_Effect'].mean()
        cumulative_effect = df_post['Causal_Effect'].sum()
        relative_effect = (cumulative_effect / df_post['Synthetic_Target'].sum()) * 100
        
        intervention_label = df.loc[t_intervention_idx, time_col] if time_col else t_intervention_idx
        
        st.subheader("主要KPI指標")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("平均リフト効果", f"{mean_effect:.2f}", help="介入期間の1時点あたりの純増平均値です。")
        with col2:
            st.metric("合計リフト効果", f"{cumulative_effect:.2f}", help="介入期間全体で発生した純増の総和です。")
        with col3:
            st.metric("相対リフト効果 (%)", f"{relative_effect:.2f}%", help="反事実（Synthetic）に対して、何パーセントの増加があったかを示します。")
        with col4:
            st.metric("Pre期間 MAPE (%)", f"{mape_pre:.2f}%", help="介入前の予測誤差の割合です。この値が小さいほど、精度の高い反事実が作れていることを示します。")

        # グラフ描画
        st.markdown("### トレンドの推移と推定効果")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        x_values = df[time_col] if time_col else df.index

        ax1.plot(x_values, df[target_col], label='Actual Target (実測値)', color='black', linewidth=2)
        ax1.plot(x_values, df['Synthetic_Target'], label='Synthetic Target (反事実)', color='blue', linestyle='--')
        ax1.axvline(x=x_values.iloc[t_intervention_idx], color='red', linestyle=':', label='介入開始')
        ax1.set_title('Target vs Synthetic Target')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(x_values, df['Causal_Effect'], label='Causal Effect (純増分)', color='green')
        ax2.axhline(y=0, color='black', linestyle='-')
        ax2.axvline(x=x_values.iloc[t_intervention_idx], color='red', linestyle=':')
        ax2.set_title('Gap (推定効果の推移)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        st.pyplot(fig)

        # 重み分布の表示
        st.subheader("最適化された重み (W)")
        st.caption("反事実を作成するために、各ドナーがどの程度の割合でブレンドされたかを示します。")
        weight_df = pd.DataFrame({
            'Donor': donor_cols,
            'Weight': optimal_weights
        }).sort_values('Weight', ascending=False)
        
        col_w1, col_w2 = st.columns([1, 2])
        with col_w1:
            st.dataframe(weight_df.style.format({'Weight': '{:.4f}'}))
        with col_w2:
            st.bar_chart(weight_df.set_index('Donor'))