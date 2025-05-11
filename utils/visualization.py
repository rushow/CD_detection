import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
import os

def plot_heatmap_auc(
    data: pd.DataFrame,
    figsize: tuple = (12, 8),
    cmap: str = "viridis",
    title_fontsize: int = 16,
    label_fontsize: int = 14,
    annotation_fontsize: int = 12,
    rotation_xticks: int = 45,
    vmin: Optional[float] = 0.5,
    vmax: Optional[float] = 1.0,
    center: Optional[float] = 0.75,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create an enhanced heatmap visualization for AUC scores comparison.
    
    Args:
        data: DataFrame with models as index and drift detectors as columns
        figsize: Figure size as (width, height)
        cmap: Colormap for heatmap (default: viridis)
        title_fontsize: Font size for title
        label_fontsize: Font size for axis labels
        annotation_fontsize: Font size for cell annotations
        rotation_xticks: Rotation angle for x-axis labels
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling
        center: Center value for colormap
        save_path: Path to save the figure (optional)
        
    Returns:
        Tuple of (figure, axis)
    """
    # Set the style
    sns.set_style("white")
    
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
        annot_kws={'size': annotation_fontsize, 'weight': 'bold'},
        linewidths=0.5,
        linecolor='white',
        cbar_kws={
            'label': 'AUC Score',
            'orientation': 'vertical',
            'shrink': 0.8,
            'pad': 0.02,
            'aspect': 30
        }
    )
    
    # Customize appearance
    ax.set_title('AUC Performance Comparison:\nModels vs Drift Detectors', 
                fontsize=title_fontsize, 
                weight='bold',
                pad=20)
    
    ax.set_xlabel('Drift Detectors', fontsize=label_fontsize, weight='bold', labelpad=10)
    ax.set_ylabel('Models', fontsize=label_fontsize, weight='bold', labelpad=10)
    
    # Rotate x-axis labels and make them bold
    plt.xticks(rotation=rotation_xticks, ha='right', fontsize=label_fontsize-2, fontweight='bold')
    plt.yticks(fontsize=label_fontsize-2, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_multi_dataset_heatmaps(
    results: Dict[str, pd.DataFrame],
    figsize: tuple = (18, 14),  # Increased figure size for better spacing
    cmap: str = "viridis",
    save_path: Optional[str] = None,
    wspace: float = 0.3,  # Horizontal space between subplots
    hspace: float = 0.4   # Vertical space between subplots
) -> plt.Figure:
    """
    Create a figure with subplots for AUC heatmaps from multiple datasets with improved spacing.
    
    Args:
        results: Dictionary mapping dataset names to their AUC DataFrames
        figsize: Figure size for the entire plot
        cmap: Colormap to use for heatmaps
        save_path: Path to save the figure (optional)
        wspace: Width space between subplots
        hspace: Height space between subplots
        
    Returns:
        Figure object
    """
    # Calculate number of rows and columns for subplots
    n_datasets = len(results)
    n_cols = 2  # Fixed to 2 columns for better layout
    n_rows = (n_datasets + 1) // 2
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Create a GridSpec layout with proper spacing
    gs = fig.add_gridspec(n_rows, n_cols, wspace=wspace, hspace=hspace)
    
    # Process each dataset
    for idx, (dataset_name, auc_df) in enumerate(results.items()):
        row = idx // n_cols
        col = idx % n_cols
        
        # Create subplot with proper position
        ax = fig.add_subplot(gs[row, col])
        
        # Create heatmap for this dataset
        sns.heatmap(
            data=auc_df,
            annot=True,
            fmt='.3f',
            cmap=cmap,
            center=0.75,
            vmin=0.5,
            vmax=1.0,
            ax=ax,
            annot_kws={'size': 11, 'weight': 'bold'},
            linewidths=0.5,
            cbar_kws={'label': 'AUC Score', 'shrink': 0.8}
        )
        
        # Set title with increased size an