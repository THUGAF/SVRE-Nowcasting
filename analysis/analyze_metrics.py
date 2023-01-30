import os
import pandas as pd


def get_model_metrics(model_dir):
    test_metrics_path = os.path.join(model_dir, 'test_metrics.csv')
    sample_0_metrics_path = os.path.join(model_dir, 'sample_0_metrics.csv')
    sample_1_metrics_path = os.path.join(model_dir, 'sample_1_metrics.csv')
    test_metrics_df = pd.read_csv(test_metrics_path)
    sample_0_metrics_df = pd.read_csv(sample_0_metrics_path)
    sample_1_metrics_df = pd.read_csv(sample_1_metrics_path)
    return test_metrics_df, sample_0_metrics_df, sample_1_metrics_df


def concat_model_metrics(metrics, idx):
    metrics_60min = pd.concat(metrics.values())
    metrics_60min = metrics_60min.drop(columns=['Time', 'POD-40dBZ', 'FAR-40dBZ', 'CSI-40dBZ'])
    metrics_60min = metrics_60min.loc[idx]
    metrics_60min.index = metrics.keys()
    print(metrics_60min)
    return metrics_60min


def analyze_ablation_metrics():
    model_names = ['AGAN(g)', 'AGAN(g)+SVRE', 'AGAN', 'AGAN+SVRE']
    model_dirs = ['results/AttnUNet', 'results/AttnUNet_SVRE', 'results/AttnUNet_GA', 'results/AttnUNet_GASVRE']
    test_metrics, sample_0_metrics, sample_1_metrics = {}, {}, {}
    for name, dir_ in zip(model_names, model_dirs):
        test_metrics[name], sample_0_metrics[name], sample_1_metrics[name] = get_model_metrics(dir_)
    test_metrics = concat_model_metrics(test_metrics, 9)
    sample_0_metrics = concat_model_metrics(sample_0_metrics, 9)
    sample_1_metrics = concat_model_metrics(sample_1_metrics, 9)
    with pd.ExcelWriter('results/metrics_ablation.xlsx') as writer:
        test_metrics.to_excel(writer, sheet_name='test', index_label='Model', float_format='%.4g')
        sample_0_metrics.to_excel(writer, sheet_name='sample_0', index_label='Model', float_format='%.4g')
        sample_1_metrics.to_excel(writer, sheet_name='sample_1', index_label='Model', float_format='%.4g')


def analyze_comparison_metrics():
    model_names = ['PySTEPS', 'SmaAt-UNet', 'MotionRNN', 'AGAN+SVRE']
    model_dirs = ['results/PySTEPS', 'results/SmaAt_UNet', 'results/MotionRNN', 'results/AttnUNet_GASVRE']
    test_metrics, sample_0_metrics, sample_1_metrics = {}, {}, {}
    for name, dir_ in zip(model_names, model_dirs):
        test_metrics[name], sample_0_metrics[name], sample_1_metrics[name] = get_model_metrics(dir_)
    test_metrics = concat_model_metrics(test_metrics, 9)
    sample_0_metrics = concat_model_metrics(sample_0_metrics, 9)
    sample_1_metrics = concat_model_metrics(sample_1_metrics, 9)
    with pd.ExcelWriter('results/metrics_comparison.xlsx') as writer:
        test_metrics.to_excel(writer, sheet_name='test', index_label='Model', float_format='%.4g')
        sample_0_metrics.to_excel(writer, sheet_name='sample_0', index_label='Model', float_format='%.4g')
        sample_1_metrics.to_excel(writer, sheet_name='sample_1', index_label='Model', float_format='%.4g')


if __name__ == '__main__':
    analyze_ablation_metrics()
    analyze_comparison_metrics()
