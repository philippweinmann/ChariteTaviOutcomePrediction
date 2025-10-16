# %%
from tavidifficulty.data.parse_results_files import parse_all_results_and_targets_files_to_patients
import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest
# %%
patients = parse_all_results_and_targets_files_to_patients()
patients = [patient for patient in patients if not patient.incomplete]

print(f"patient length after filtering out incomplete ones: {len(patients)}")
# %%
# for outlier detection we require the data in a dataframe.
# I know it sounds stupid to move it from a dataframe to a class and back to a dataframe again, but the class mostly there for validation purposes.

# the patient should have a function that converts its attributes to a dataframe

display(patients[0].params_dataframe)
# %%
# let's build the entire dataframe
def get_full_dataframe(patients):
    patient_dfs = []
    for i, patient in enumerate(patients):
        patient_df = patient.params_dataframe
        patient_dfs.append(patient_df)
    
    patient_dfs = pd.concat(patient_dfs, axis=0, ignore_index=False)

    return patient_dfs

patients_df = get_full_dataframe(patients)

display(patients_df.head())
# %%
# one issue, how will I identify which row is which patient? Use the pid.
# Using it as index, feels like a smart thing to do.

# %%
# 'contamination' parameter is the expected proportion of outliers in the data.
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outlier_labels = iso_forest.fit_predict(patients_df)
# outlier_labels: -1 for outliers, 1 for inliers

# %%
patients_df_outlier = patients_df.copy()
scores = iso_forest.decision_function(patients_df_outlier)
patients_df_outlier['anomaly_score'] = scores
patients_df_outlier = patients_df_outlier.sort_values('anomaly_score')
display(patients_df_outlier)
# %%
# okay cool, however there are too many features, I need to know why those were outliers.
# Create a TreeExplainer for your Isolation Forest model.
explainer = shap.TreeExplainer(iso_forest)

# %%
# let's visualize which features are responsible for one patient
top_10_indexes = patients_df_outlier.index[:10] # top outlier score

def visualize_shap_value(pid, patients_df_outlier):
    print(f"Visualizing outlier shap scores for patient with id: {pid}")

    # Extract the row for the given patient ID
    patient_row = patients_df_outlier.loc[[pid]]
    
    print(f"anomaly_score: {patient_row['anomaly_score'].item()}")


    # Compute SHAP values
    shap_values = explainer.shap_values(patient_row)

    # Ensure SHAP values match feature count
    feature_names = patient_row.columns  # Extract column names dynamically

    # Convert SHAP values into a DataFrame
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values[0],  # First instance (since we selected only one row)
        'feature_value': patient_row.iloc[0].values,  # Extract values from patient row
    })

    # Sort by absolute SHAP value to get the top features
    shap_df = shap_df.reindex(shap_df['shap_value'].abs().sort_values(ascending=False).index)

    # Get the top 5 features with the highest impact
    top_5_shap = shap_df.head(5)

    

    # Compute the average and std values of these features across the entire dataset
    std_values = patients_df[top_5_shap['feature']].std()
    avg_values = patients_df[top_5_shap['feature']].mean()
    

    # Add the average values to the shap_df
    shap_df['average_value'] = shap_df['feature'].map(avg_values)
    shap_df['std'] = shap_df['feature'].map(std_values)

    # Print results
    print("\nTop 5 Features with Highest SHAP Impact:")
    display(shap_df.head(5))

    # SHAP beeswarm plot
    shap.summary_plot(shap_values, patient_row, feature_names=feature_names)

    print("-------PATIENT DONE; NEXT PATIENT-----------")



for i, top_10_index in enumerate(top_10_indexes):
    print(f"visualizing the top 10 outliers: {i + 1} / 10")
    visualize_shap_value(str(top_10_index), patients_df_outlier)
# %%
print(f"top 10 outliers: {top_10_indexes}")
# %%
