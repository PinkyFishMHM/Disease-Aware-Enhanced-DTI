import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和全局样式
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# 定义颜色方案
COLORS = {
    'primary': '#2E8B57',
    'secondary': '#4169E1', 
    'accent': '#FF6347',
    'warning': '#FFD700',
    'info': '#20B2AA',
    'success': '#32CD32',
    'methods': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
}

def load_data():
    """加载所有数据文件"""
    try:
        data = {
            'performance': pd.read_csv('data/performance_comparison.csv'),
            'ablation': pd.read_csv('data/ablation_study.csv'),
            'disease': pd.read_csv('data/disease_specific_performance.csv'),
            'pathway': pd.read_csv('data/pathway_attention_weights.csv'),
            'case_study': pd.read_csv('data/case_study_data.csv'),
            'metrics': pd.read_csv('data/computational_metrics.csv'),
            'attention': pd.read_csv('data/attention_heatmap_data.csv', index_col=0)
        }
        print("✓ 所有数据文件加载成功")
        return data
    except FileNotFoundError as e:
        print(f"❌ 数据文件未找到: {e}")
        return None

def create_performance_comparison(data):
    """创建图3: 整体性能比较"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    methods = ['DeepDTA', 'MolTrans', 'AttentionDTI', 'DLM-DTI', 'Ours']
    datasets = ['BindingDB', 'DAVIS', 'BIOSNAP']

    x = np.arange(len(methods))
    width = 0.25

    # AUC-ROC
    for i, dataset in enumerate(datasets):
        dataset_data = data['performance'][data['performance']['Dataset'] == dataset]
        auc_values = [dataset_data[dataset_data['Method'] == method]['AUC_ROC'].iloc[0] 
                     for method in methods]

        bars = ax1.bar(x + i*width, auc_values, width, 
                      label=dataset, alpha=0.8, color=COLORS['methods'][i])

        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax1.set_xlabel('Methods')
    ax1.set_ylabel('AUC-ROC')
    ax1.set_title('(a) AUC-ROC Comparison Across Datasets')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(methods, rotation=15)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # AUPR
    for i, dataset in enumerate(datasets):
        dataset_data = data['performance'][data['performance']['Dataset'] == dataset]
        aupr_values = [dataset_data[dataset_data['Method'] == method]['AUPR'].iloc[0] 
                      for method in methods]

        bars = ax2.bar(x + i*width, aupr_values, width, 
                      label=dataset, alpha=0.8, color=COLORS['methods'][i])

        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax2.set_xlabel('Methods')
    ax2.set_ylabel('AUPR')
    ax2.set_title('(b) AUPR Comparison Across Datasets')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(methods, rotation=15)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Figure3_Performance_Comparison.png', bbox_inches='tight', dpi=300)
    plt.show()
    print("✓ 创建 Figure 3: 整体性能比较")

def create_ablation_study(data):
    """创建图4: 消融研究"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 累积性能改进
    configs = data['ablation']['Configuration'].tolist()
    auc_values = data['ablation']['AUC_ROC'].tolist()
    aupr_values = data['ablation']['AUPR'].tolist()

    x = np.arange(len(configs))
    width = 0.35

    ax1.bar(x - width/2, auc_values, width, label='AUC-ROC', color=COLORS['primary'], alpha=0.8)
    ax1.bar(x + width/2, aupr_values, width, label='AUPR', color=COLORS['secondary'], alpha=0.8)

    ax1.set_xlabel('Model Variants')
    ax1.set_ylabel('Performance Score')
    ax1.set_title('(a) Cumulative Performance Improvement')
    ax1.set_xticks(x)
    ax1.set_xticklabels([config.replace('+ ', '') for config in configs], rotation=15, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 组件贡献饼图
    components = ['Knowledge Integration', 'Attention Mechanism', 'Pathway Scoring', 'GO Embedding']
    contributions = [43.6, 24.5, 17.3, 14.5]

    ax2.pie(contributions, labels=components, autopct='%1.1f%%', 
           colors=COLORS['methods'][:4], startangle=90)
    ax2.set_title('(b) Component Contribution (AUC-ROC)')

    # 疾病特异性性能
    disease_data = data['disease']
    bars = ax3.bar(disease_data['Disease'], disease_data['AUC_ROC'], 
                   color=COLORS['methods'][:len(disease_data)], alpha=0.8)

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.3f}', ha='center', va='bottom')

    ax3.set_ylabel('AUC-ROC Score')
    ax3.set_title('(c) Disease-Specific Performance')
    ax3.set_xticklabels(disease_data['Disease'], rotation=45, ha='right')

    # 注意力热图
    attention_data = data['attention']
    im = ax4.imshow(attention_data.values, cmap='YlOrRd', aspect='auto')

    ax4.set_xticks(range(len(attention_data.columns)))
    ax4.set_xticklabels(attention_data.columns)
    ax4.set_yticks(range(len(attention_data.index)))
    ax4.set_yticklabels(attention_data.index)
    ax4.set_title('(d) Pathway Attention Heatmap')

    plt.tight_layout()
    plt.savefig('Figure4_Ablation_Study.png', bbox_inches='tight', dpi=300)
    plt.show()
    print("✓ 创建 Figure 4: 消融研究")

def main():
    """主函数"""
    print("开始生成DLM-DTI论文图表...")

    data = load_data()
    if data is None:
        return

    create_performance_comparison(data)
    create_ablation_study(data)

    print("✅ 所有图表生成完成！")

if __name__ == "__main__":
    main()