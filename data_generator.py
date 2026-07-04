import numpy as np
import pandas as pd

def generate_scm_data(n_periods=100, t_intervention=70, effect_size=20.0):
    """
    SCM検証用のテストデータを生成する関数。
    TargetはDonor_1~3の線形結合にノイズを加えたもので構成される。
    """
    np.random.seed(42)
    time = np.arange(n_periods)
    
    # 共通のベーストレンドと季節性
    base_trend = time * 0.5
    seasonality = np.sin(time / 5) * 10
    
    donor_names = [f'Donor_{i}' for i in range(1, 6)]
    df = pd.DataFrame(index=time)
    
    # ドナープールのデータ生成
    for donor in donor_names:
        df[donor] = base_trend + seasonality + np.random.normal(0, 5, n_periods) + np.random.uniform(10, 50)
        
    # Targetのデータ生成 (真の重み: Donor_1=0.5, Donor_2=0.3, Donor_3=0.2)
    true_weights = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
    df['Target'] = df[donor_names].dot(true_weights) + np.random.normal(0, 2, n_periods)
    
    # 介入効果の付与
    df.loc[t_intervention:, 'Target'] += effect_size
    df.index.name = 'Time'
    
    return df, t_intervention, donor_names, true_weights