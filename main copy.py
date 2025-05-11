# main.py
import pandas as pd
from river import naive_bayes, tree
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io.arff import loadarff 

from concept_drift.ddm import DDMDriftDetector
from concept_drift.eddm import EDDMDriftDetector
from concept_drift.fhddm import FHDDMDriftDetector
from concept_drift.rddm import RDDMDriftDetector
from concept_drift.ftdd import FTDDDriftDetector
from concept_drift.arf import ARFDriftDetector
from concept_drift.ace import ACEDriftDetector
from concept_drift.dwm import DWMDriftDetector
from concept_drift.d3 import D3DriftDetector
from concept_drift.fpdd import FPDDDriftDetector
from concept_drift.adwin import ADWINDriftDetector
from concept_drift.kswin import KSWINDriftDetector
from concept_drift.mddm import MDDMDriftDetector
from concept_drift.wstd import WSTDDriftDetector
from concept_drift.cusum import CUSUMDriftDetector
from concept_drift.ewma import EWMADriftDetector
from concept_drift.aue import AUEDriftDetector
from concept_drift.awe import AWEDriftDetector
from concept_drift.kappa import KappaDriftDetector
from utils.load_data import load_rt_8873985678962563_abrupto_data, load_rt_8873985678962563_gradual_data, load_sine_0123_abrupto_data, load_sine_0123_gradual_data, load_mixed_0101_abrupto_data, load_mixed_0101_gradual_data, load_elec_data, load_kdd_data, load_iot_data, load_cic_data
from utils.evaluation import evaluate_learner
# from utils.visualization import plot_accuracy
from utils.visualization import plot_multi_dataset_heatmaps



# df_name, X, y = load_sine_0123_abrupto_data()
# df_name, X, y = load_sine_0123_gradual_data()
# df_name, X, y = load_rt_8873985678962563_abrupto_data()
# df_name, X, y = load_rt_8873985678962563_gradual_data() 
# df_name, X, y = load_mixed_0101_abrupto_data()
# df_name, X, y = load_mixed_0101_gradual_data()
# df_name, X, y = load_elec_data()
# df_name, X, y = load_iot_data()
# df_name, X, y = load_cic_data()


# Define datasets with their loading functions and names
datasets = [
    (load_sine_0123_abrupto_data, "Sine Abrupto"),
    (load_sine_0123_gradual_data, "Sine Gradual"),
    (load_rt_8873985678962563_abrupto_data, "RT Abrupto"),
    (load_rt_8873985678962563_gradual_data, "RT Gradual"),
    (load_mixed_0101_abrupto_data, "Mixed Abrupto"),
    (load_mixed_0101_gradual_data, "Mixed Gradual"),
    (load_elec_data, "Electricity"),
    (load_kdd_data, "KDD"),
    (load_iot_data, "IoT"),
    (load_cic_data, "CIC")
]



# List of models and drift detectors
models = {
    'Naive Bayes': naive_bayes.GaussianNB(),
    'HT': tree.HoeffdingTreeClassifier()
}

drift_detectors = {
    # Statistical-Based Drift Detectors
    # 'FTDD': FTDDDriftDetector(), # 2018
    # 'RDDM': RDDMDriftDetector(), # 2017
    # 'FHDDM': FHDDMDriftDetector(), # 2016 == River
    # 'EWMA': EWMADriftDetector(), # 2012
    # 'EDDM': EDDMDriftDetector(), # 2006 == River
    # 'DDM': DDMDriftDetector(), # 2004 == River
    # 'CUSUM': CUSUMDriftDetector(), #1950 == Accuracy 0% - HT
    # Window-Based Drift Detectors
    'KSWIN': KSWINDriftDetector(), #2020 == River
    'FPDD': FPDDDriftDetector(), # 2018
    'WSTD': WSTDDriftDetector(), # 2018 
    'MDDM': MDDMDriftDetector(), #2018
    'ADWIN': ADWINDriftDetector(), # 2007 == River
    # Ensemble-Based Drift Detectors
    # 'ARF': ARFDriftDetector(), # 2017
    # 'D3': D3DriftDetector(), #2015
    # 'AUE': AUEDriftDetector(), # 2011
    # 'DWM': DWMDriftDetector(), # 2007  
    # 'AWE': AWEDriftDetector(), # 2003 
    # 'Kappa': KappaDriftDetector() #2016
    # 'ACE': ACEDriftDetector(), # 2005 
}

# t_array = []
# m_array = []
# title_array = []
# data_auc = {}

# # plt.rcParams.update({'font.size': 10})
# # plt.figure(1,figsize=(10,6)) 
# # sns.set_style("darkgrid")
# # plt.clf() 

# # Iterate through each model and drift detector combination
# for model_name, model in models.items():
#     for detector_name, drift_detector in drift_detectors.items():
#         print(f"{detector_name} Drift Detection with {model_name} Running...")
        
#         # Evaluate the model
#         t, m, metric, metric_f1, metric_precision, auc_value, auroc_value = evaluate_learner(df_name, model, drift_detector, X, y)
#         print(metric)
#         print(metric_precision)
#         print(metric_f1)
#         print(f'AUC: {auc_value}')
#         print(f'AUROC: {auroc_value}')
        
#         # plt.plot(t,m,'-b',label='Avg Accuracy: %.2f%%'%(m[-1]))
#         t_array.append(t)
#         m_array.append(m)
#         title_array.append(f'{detector_name} - {model_name}')
 
#         # Store AUC values
#         if model_name not in data_auc:
#             data_auc[model_name] = {}
#         data_auc[model_name][detector_name] = auc_value


#         # # Plot the results
#         # plot_title = f'{detector_name} Drift Detection with {model_name} - Accuracy'
#         # plot_accuracy(t, m, plot_title)

        
# # plt.legend(loc='best')
# # plt.title('Accuracy Summary', fontsize=15)
# # plt.xlabel('Number of samples')
# # plt.ylabel('Accuracy (%)')

# # plt.draw()
# # plot_summary(t_array, m_array, title_array, "Accuracy Summary for All Models and Drift Detectors")

# # Convert data_auc to DataFrame
# auc_df = pd.DataFrame.from_dict(data_auc, orient='index')

# # # Convert the DataFrame to a long format for violin plot
# # auc_long_df = auc_df.reset_index().melt(id_vars='index', var_name='Algorithm', value_name='AUC')
# # auc_long_df.rename(columns={'index': 'Model'}, inplace=True)

# # # Plot the violin plot for AUC
# # plot_violin_auc(data=auc_long_df)

# # Plot the heatmap for AUC
# plot_heatmap_auc(data=auc_df)


# # Or detailed comparison
# # plot_detailed_auc_comparison(auc_df)

# # Show the plot
# plt.show()

cd_detector_name = ''

# Dictionary to store results for each dataset
results = {}

# Process each dataset
for load_func, dataset_name in datasets:
    print(f"\nProcessing {dataset_name} dataset...")
    
    # Load dataset
    try:
        df_name, X, y = load_func()
        
        # Initialize storage for this dataset
        data_auc = {}
        
        # Run models and detector
        for model_name, model in models.items():
            for detector_name, drift_detector in drift_detectors.items():
                cd_detector_name = detector_name
                print(f"Running {detector_name} with {model_name}...")
                
                # Evaluate the model
                _, _, metric, metric_f1, metric_precision, auc_value, auroc_value = evaluate_learner(
                    df_name, model, drift_detector, X, y
                )
                
                # Store results
                if model_name not in data_auc:
                    data_auc[model_name] = {}
                data_auc[model_name][detector_name] = auc_value
                
                # Print metrics
                print(f"Accuracy: {metric}")
                print(f"Precision: {metric_precision}")
                print(f"F1: {metric_f1}")
                print(f"AUC: {auc_value}")
                print(f"AUROC: {auroc_value}")
        
        # Store results for this dataset
        results[dataset_name] = pd.DataFrame.from_dict(data_auc, orient='index')
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        continue

# Create combined visualization
plt.figure(figsize=(15, 10))

# Calculate number of rows and columns for subplots
n_datasets = len(results)
n_cols = 2
n_rows = (n_datasets + 1) // 2

# Create subplots for each dataset
for idx, (dataset_name, auc_df) in enumerate(results.items(), 1):
    plt.subplot(n_rows, n_cols, idx)
    
    # Create heatmap for this dataset
    sns.heatmap(
        data=auc_df,
        annot=True,
        fmt='.3f',
        cmap='RdYlBu_r',
        center=0.75,
        vmin=0.5,
        vmax=1.0,
        cbar_kws={'label': 'AUC Score'}
    )
    
    plt.title(f'{dataset_name} Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

# Create summary DataFrame
summary_df = pd.DataFrame({
    dataset_name: df[cd_detector_name].mean() 
    for dataset_name, df in results.items()
}, index=['Average AUC']).T

# Plot summary heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    summary_df,
    annot=True,
    fmt='.3f',
    cmap='RdYlBu_r',
    center=0.75,
    vmin=0.5,
    vmax=1.0,
    cbar_kws={'label': 'Average AUC Score'}
)
plt.title(f'{cd_detector_name} Performance Across Datasets')
plt.tight_layout()
plt.show()

print("Creating improved multi-dataset visualization...")
multi_fig = plot_multi_dataset_heatmaps(
    results=results,
    figsize=(18, 14),     # Larger figure size
    cmap="RdYlBu_r",      # Using a color scheme that highlights differences better
    wspace=0.3,           # More horizontal space between subplots
    hspace=0.4,           # More vertical space between subplots
    save_path="results/improved_all_datasets_heatmaps.png"
)
plt.show()

# Print summary statistics
print("\nSummary Statistics:")
print("==================")
for dataset_name, df in results.items():
    print(f"\n{dataset_name} Dataset:")
    print(f"Average AUC: {df[cd_detector_name].mean():.3f}")
    print(f"Std Dev AUC: {df[cd_detector_name].std():.3f}")