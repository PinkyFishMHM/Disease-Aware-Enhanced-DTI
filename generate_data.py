import pandas as pd
import numpy as np
import json
import os

# 设置随机种子确保可重复性
np.random.seed(42)

def create_performance_data():
    """创建模型性能比较数据"""

    # 基础模型性能数据
    methods = ['DeepDTA', 'MolTrans', 'AttentionDTI', 'DLM-DTI', 'Ours']
    datasets = ['BindingDB', 'DAVIS', 'BIOSNAP']

    # 根据论文结果设定的性能数据
    auc_roc_data = {
        'BindingDB': [0.852, 0.867, 0.871, 0.871, 0.934],  # 我们的模型提升7.2%
        'DAVIS': [0.833, 0.845, 0.856, 0.846, 0.912],      # 我们的模型提升7.8%
        'BIOSNAP': [0.801, 0.815, 0.823, 0.823, 0.898]     # 我们的模型提升9.1%
    }

    aupr_data = {
        'BindingDB': [0.723, 0.741, 0.756, 0.756, 0.828],  # 我们的模型提升9.5%
        'DAVIS': [0.718, 0.735, 0.748, 0.745, 0.817],      # 我们的模型提升9.6%
        'BIOSNAP': [0.701, 0.719, 0.728, 0.723, 0.804]     # 我们的模型提升11.2%
    }

    # 创建DataFrame
    performance_data = []
    for dataset in datasets:
        for i, method in enumerate(methods):
            performance_data.append({
                'Dataset': dataset,
                'Method': method,
                'AUC_ROC': auc_roc_data[dataset][i],
                'AUPR': aupr_data[dataset][i]
            })

    df = pd.DataFrame(performance_data)
    df.to_csv('performance_comparison.csv', index=False)
    return df

def create_ablation_study_data():
    """创建消融研究数据"""

    # 根据论文表格1的数据
    ablation_data = [
        {'Configuration': 'Base Model', 'AUC_ROC': 0.823, 'AUPR': 0.756, 
         'AUC_ROC_Improvement': 0.0, 'AUPR_Improvement': 0.0,
         'Component_Contribution_AUC': 0.0, 'Component_Contribution_AUPR': 0.0},

        {'Configuration': '+ Knowledge Integration', 'AUC_ROC': 0.871, 'AUPR': 0.798,
         'AUC_ROC_Improvement': 0.048, 'AUPR_Improvement': 0.042,
         'Component_Contribution_AUC': 43.6, 'Component_Contribution_AUPR': 38.5},

        {'Configuration': '+ Attention Mechanism', 'AUC_ROC': 0.898, 'AUPR': 0.825,
         'AUC_ROC_Improvement': 0.027, 'AUPR_Improvement': 0.027,
         'Component_Contribution_AUC': 24.5, 'Component_Contribution_AUPR': 24.8},

        {'Configuration': '+ Pathway Scoring', 'AUC_ROC': 0.917, 'AUPR': 0.847,
         'AUC_ROC_Improvement': 0.019, 'AUPR_Improvement': 0.022,
         'Component_Contribution_AUC': 17.3, 'Component_Contribution_AUPR': 20.2},

        {'Configuration': '+ GO Embedding', 'AUC_ROC': 0.933, 'AUPR': 0.865,
         'AUC_ROC_Improvement': 0.016, 'AUPR_Improvement': 0.018,
         'Component_Contribution_AUC': 14.5, 'Component_Contribution_AUPR': 16.5},

        {'Configuration': 'Full Model', 'AUC_ROC': 0.934, 'AUPR': 0.869,
         'AUC_ROC_Improvement': 0.111, 'AUPR_Improvement': 0.113,
         'Component_Contribution_AUC': 100.0, 'Component_Contribution_AUPR': 100.0}
    ]

    df = pd.DataFrame(ablation_data)
    df.to_csv('ablation_study.csv', index=False)
    return df

def create_disease_specific_data():
    """创建疾病特异性性能数据"""

    disease_data = [
        {'Disease': 'Cancer', 'AUC_ROC': 0.945, 'AUPR': 0.892, 'Sample_Size': 12500},
        {'Disease': 'Cardiovascular', 'AUC_ROC': 0.923, 'AUPR': 0.876, 'Sample_Size': 8900},
        {'Disease': 'Neurological', 'AUC_ROC': 0.912, 'AUPR': 0.854, 'Sample_Size': 6700},
        {'Disease': 'Metabolic', 'AUC_ROC': 0.907, 'AUPR': 0.849, 'Sample_Size': 5400},
        {'Disease': 'Immunological', 'AUC_ROC': 0.901, 'AUPR': 0.841, 'Sample_Size': 4200}
    ]

    df = pd.DataFrame(disease_data)
    df.to_csv('disease_specific_performance.csv', index=False)
    return df

def create_pathway_attention_data():
    """创建通路注意力权重数据"""

    # 癌症药物的通路注意力分布
    cancer_pathways = [
        {'Pathway': 'PI3K-Akt', 'Attention_Weight': 0.89, 'Drug_Type': 'Cancer'},
        {'Pathway': 'MAPK', 'Attention_Weight': 0.82, 'Drug_Type': 'Cancer'},
        {'Pathway': 'p53', 'Attention_Weight': 0.78, 'Drug_Type': 'Cancer'},
        {'Pathway': 'Cell Cycle', 'Attention_Weight': 0.76, 'Drug_Type': 'Cancer'},
        {'Pathway': 'Apoptosis', 'Attention_Weight': 0.72, 'Drug_Type': 'Cancer'},
        {'Pathway': 'DNA Repair', 'Attention_Weight': 0.68, 'Drug_Type': 'Cancer'}
    ]

    # 心血管药物的通路注意力分布
    cardio_pathways = [
        {'Pathway': 'Cholesterol Biosynthesis', 'Attention_Weight': 0.94, 'Drug_Type': 'Cardiovascular'},
        {'Pathway': 'PPAR Signaling', 'Attention_Weight': 0.88, 'Drug_Type': 'Cardiovascular'},
        {'Pathway': 'Fatty Acid Metabolism', 'Attention_Weight': 0.82, 'Drug_Type': 'Cardiovascular'},
        {'Pathway': 'Calcium Signaling', 'Attention_Weight': 0.79, 'Drug_Type': 'Cardiovascular'},
        {'Pathway': 'Renin-Angiotensin', 'Attention_Weight': 0.74, 'Drug_Type': 'Cardiovascular'},
        {'Pathway': 'Cardiac Muscle', 'Attention_Weight': 0.71, 'Drug_Type': 'Cardiovascular'}
    ]

    pathway_data = cancer_pathways + cardio_pathways
    df = pd.DataFrame(pathway_data)
    df.to_csv('pathway_attention_weights.csv', index=False)
    return df

def create_case_study_data():
    """创建案例研究数据"""

    # 药物-靶点交互案例
    case_studies = [
        {
            'Drug': 'Imatinib',
            'Target': 'BCR-ABL1',
            'Disease': 'Cancer',
            'Predicted_Score': 0.94,
            'Primary_Pathway': 'PI3K-Akt',
            'Secondary_Pathway': 'MAPK',
            'Pathway_Importance': 0.89,
            'Chemical_Structure': 'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5'
        },
        {
            'Drug': 'Atorvastatin',
            'Target': 'HMGCR',
            'Disease': 'Cardiovascular',
            'Predicted_Score': 0.92,
            'Primary_Pathway': 'Cholesterol Biosynthesis',
            'Secondary_Pathway': 'PPAR Signaling',
            'Pathway_Importance': 0.94,
            'Chemical_Structure': 'CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4'
        }
    ]

    df = pd.DataFrame(case_studies)
    df.to_csv('case_study_data.csv', index=False)
    return df

def create_computational_metrics():
    """创建计算性能指标数据"""

    metrics_data = [
        {'Model': 'DeepDTA', 'VRAM_Usage_GB': 6.2, 'Inference_Time_sec': 0.08, 'Training_Time_hours': 24},
        {'Model': 'MolTrans', 'VRAM_Usage_GB': 7.1, 'Inference_Time_sec': 0.12, 'Training_Time_hours': 32},
        {'Model': 'AttentionDTI', 'VRAM_Usage_GB': 7.8, 'Inference_Time_sec': 0.15, 'Training_Time_hours': 28},
        {'Model': 'DLM-DTI', 'VRAM_Usage_GB': 7.9, 'Inference_Time_sec': 0.03, 'Training_Time_hours': 18},
        {'Model': 'Ours', 'VRAM_Usage_GB': 8.0, 'Inference_Time_sec': 0.02, 'Training_Time_hours': 20}
    ]

    df = pd.DataFrame(metrics_data)
    df.to_csv('computational_metrics.csv', index=False)
    return df

def create_attention_distribution_data():
    """创建注意力分布数据用于热图"""

    # 为热图创建注意力权重矩阵
    pathways = ['PI3K-Akt', 'MAPK', 'p53', 'Cholesterol', 'Cardiac', 'Apoptosis']
    drug_categories = ['Cancer', 'Cardiovascular']

    # 创建注意力矩阵
    attention_matrix = np.array([
        [0.89, 0.25],  # PI3K-Akt
        [0.82, 0.30],  # MAPK  
        [0.78, 0.20],  # p53
        [0.15, 0.94],  # Cholesterol
        [0.12, 0.88],  # Cardiac
        [0.72, 0.18]   # Apoptosis
    ])

    # 转换为DataFrame
    attention_df = pd.DataFrame(attention_matrix, 
                               index=pathways, 
                               columns=drug_categories)
    attention_df.to_csv('attention_heatmap_data.csv')
    return attention_df

def main():
    """主函数：创建所有数据文件"""
    print("开始创建DLM-DTI论文数据文件...")

    # 创建输出目录
    os.makedirs('data', exist_ok=True)
    os.chdir('data')

    # 生成所有数据文件
    create_performance_data()
    create_ablation_study_data()
    create_disease_specific_data()
    create_pathway_attention_data()
    create_case_study_data()
    create_computational_metrics()
    create_attention_distribution_data()

    print("✅ 所有数据文件创建完成！")

if __name__ == "__main__":
    main()