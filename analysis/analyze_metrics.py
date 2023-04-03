import os
import numpy as np
import pandas as pd


def get_model_metrics(model_dir: str):
    test_metrics_path = os.path.join(model_dir, 'test_metrics.csv')
    case_0_metrics_path = os.path.join(model_dir, 'case_0_metrics.csv')
    case_1_metrics_path = os.path.join(model_dir, 'case_1_metrics.csv')
    test_metrics_df = pd.read_csv(test_metrics_path)
    case_0_metrics_df = pd.read_csv(case_0_metrics_path)
    case_1_metrics_df = pd.read_csv(case_1_metrics_path)
    return test_metrics_df, case_0_metrics_df, case_1_metrics_df


def concat_model_metrics(metrics: dict):       
    metrics_60min = pd.concat(metrics.values())
    metrics_60min = metrics_60min.drop(columns=['POD_20.0', 'FAR_20.0', 'CSI_20.0', 
                                                'POD_30.0', 'FAR_30.0', 'CSI_30.0'])
    metrics_60min.index = metrics.keys()
    print(metrics_60min)
    return metrics_60min


def analyze_ablation_metrics():
    model_names = ['AGAN(g)', 'AGAN(g)+SVRE', 'AGAN', 'AGAN+SVRE']
    model_dirs = ['results/AttnUNet', 'results/AttnUNet_SVRE', 'results/AGAN', 'results/AGAN_SVRE']
    test_metrics, case_0_metrics, case_1_metrics = {}, {}, {}
    for name, dir_ in zip(model_names, model_dirs):
        test_metrics[name], case_0_metrics[name], case_1_metrics[name] = get_model_metrics(dir_)
    test_metrics = concat_model_metrics(test_metrics)
    case_0_metrics = concat_model_metrics(case_0_metrics)
    case_1_metrics = concat_model_metrics(case_1_metrics)
    with pd.ExcelWriter('results/metrics_ablation.xlsx') as writer:
        test_metrics.to_excel(writer, sheet_name='test', index_label='Model', float_format='%.4g')
        case_0_metrics.to_excel(writer, sheet_name='case_0', index_label='Model', float_format='%.4g')
        case_1_metrics.to_excel(writer, sheet_name='case_1', index_label='Model', float_format='%.4g')


def analyze_comparison_metrics():
    model_names = ['PySTEPS', 'SmaAt-UNet', 'MotionRNN', 'AGAN+SVRE']
    model_dirs = ['results/PySTEPS', 'results/SmaAt_UNet', 'results/MotionRNN', 'results/AGAN_SVRE']
    test_metrics, case_0_metrics, case_1_metrics = {}, {}, {}
    for name, dir_ in zip(model_names, model_dirs):
        test_metrics[name], case_0_metrics[name], case_1_metrics[name] = get_model_metrics(dir_)
    test_metrics = concat_model_metrics(test_metrics)
    case_0_metrics = concat_model_metrics(case_0_metrics)
    case_1_metrics = concat_model_metrics(case_1_metrics)
    with pd.ExcelWriter('results/metrics_comparison.xlsx') as writer:
        test_metrics.to_excel(writer, sheet_name='test', index_label='Model', float_format='%.4g')
        case_0_metrics.to_excel(writer, sheet_name='case_0', index_label='Model', float_format='%.4g')
        case_1_metrics.to_excel(writer, sheet_name='case_1', index_label='Model', float_format='%.4g')


if __name__ == '__main__':
    analyze_ablation_metrics()
    analyze_comparison_metrics()
