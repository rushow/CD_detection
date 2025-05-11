# # utils/visualization.py
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd


# # def plot_accuracy(t, m, title):
# #     plt.figure(figsize=(10, 6))
# #     plt.plot(t, m, '-b', label='Avg Accuracy: %.2f%%'%(m[-1]))
# #     plt.xlabel('Time')
# #     plt.ylabel('Accuracy (%)')
# #     plt.title(title)
# #     plt.legend()
# #     plt.show()


# def plot_summary(t_array, m_array, title_array, title):
#     plt.rcParams.update({'font.size': 8})
#     plt.figure(figsize=(10, 6))
#     sns.set_style("darkgrid")
#     plt.clf()

#     colors = ['-r', '-g', '-b', '-k', '-y', '-m', '-c']  

#     # Plot each t and m in the array with a unique color
#     for i in range(len(t_array)):
#         color = colors[i % len(colors)]  
#         plt.plot(t_array[i], m_array[i], color, label=f'{title_array[i]}: Accuracy: %.2f%%' % (m_array[i][-1]))

#     plt.legend(loc='best')
#     plt.title(title, fontsize=15)
#     plt.xlabel('Number of samples')
#     plt.ylabel('Accuracy (%)')
#     plt.show()


# # def plot_violin_auc(data):
# #     plt.figure(figsize=(10, 8))
# #     sns.violinplot(x='Algorithm', y='AUC', data=data)
# #     plt.title('AUC Distribution by Drift Detector Algorithm')
# #     plt.xlabel('Drift Detector Algorithm')
# #     plt.ylabel('AUC')
# #     plt.show()

# def plot_heatmap_auc(data):
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(data, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'AUC'})
#     plt.title('AUC Heatmap for Models and Drift Detectors')
#     plt.xlabel('Drift Detector')
#     plt.ylabel('Model')
#     plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional
import pandas as pd

def plot_heatmap_auc(
    data: pd.DataFrame,
    figsize: tuple = (12, 8),
    cmap: str = "RdYlBu_r",
    title_fontsize: int = 14,
    label_fontsize: int = 12,
    annotation_fontsize: int = 10,
    rotation_xticks: int = 45,
    vmin: Optional[float] = 0.5,
    vmax: Optional[float] = 1.0,
    center: Optional[float] = 0.75,
) -> None:
    """
    Create an enhanced heatmap visualization for AUC scores comparison.
    
    Args:
        data: DataFrame with models as index and drift detectors as columns
        figsize: Figure size as (width, height)
        cmap: Colormap for heatmap (default: RdYlBu_r)
        title_fontsize: Font size for title
        label_fontsize: Font size for axis labels
        annotation_fontsize: Font size for cell annotations
        rotation_xticks: Rotation angle for x-axis labels
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling
        center: Center value for colormap
    """
    # Set the style using seaborn's set_style instead of plt.style.use
    sns.set_style("whitegrid")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mask for missing values
    mask = np.isnan(data)
    
    # Create heatmap
    sns.heatmap(
        data=data,
        ax=ax,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        mask=mask,
        annot_kws={'size': annotation_fontsize},
        cbar_kws={
            'label': 'AUC Score',
            'orientation': 'vertical',
            'pad': 0.01
        }
    )
    
    # Customize appearance
    ax.set_title('AUC Performance Comparison:\nModels vs Drift Detectors', 
                fontsize=title_fontsize, 
                pad=20)
    
    ax.set_xlabel('Drift Detectors', fontsize=label_fontsize, labelpad=10)
    ax.set_ylabel('Models', fontsize=label_fontsize, labelpad=10)
    
    # Rotate x-axis labels
    plt.xticks(rotation=rotation_xticks, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax

def plot_detailed_auc_comparison(
    data: pd.DataFrame,
    highlight_threshold: float = 0.8
) -> None:
    """
    Create a detailed AUC comparison with multiple visualizations.
    
    Args:
        data: DataFrame with models as index and drift detectors as columns
        highlight_threshold: Threshold for highlighting high-performing combinations
    """
    # Set the style
    sns.set_style("whitegrid")
    
    # Create a figure with two subplots
    fig = plt.figure(figsize=(18, 8))
    
    # 1. Heatmap
    ax1 = plt.subplot(121)
    sns.heatmap(
        data=data,
        ax=ax1,
        annot=True,
        fmt='.3f',
        cmap='RdYlBu_r',
        center=0.75,
        vmin=0.5,
        vmax=1.0,
        cbar_kws={'label': 'AUC Score'}
    )
    ax1.set_title('AUC Heatmap')
    plt.xticks(rotation=45, ha='right')
    
    # 2. Top Performers Bar Plot
    ax2 = plt.subplot(122)
    
    # Calculate mean AUC for each detector
    mean_aucs = data.mean().sort_values(ascending=False)
    
    # Create bar plot
    bars = ax2.bar(range(len(mean_aucs)), mean_aucs.values)
    ax2.set_xticks(range(len(mean_aucs)))
    ax2.set_xticklabels(mean_aucs.index, rotation=45, ha='right')
    
    # Customize bars
    for i, bar in enumerate(bars):
        if mean_aucs.values[i] >= highlight_threshold:
            bar.set_color('green')
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    ax2.set_title('Average AUC by Drift Detector')
    ax2.set_ylabel('Mean AUC Score')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

