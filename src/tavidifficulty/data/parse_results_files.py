# %%
import os
import pandas as pd
from tavidifficulty.data.patient import Patient, Segment
from tavidifficulty.data.excel_extraction_utils import get_excel_sheet_selection, get_most_recent_file
from tavidifficulty.utils import extract_patient_id_from_results_excel_fp
from tavidifficulty.config import anomalous_patient_ids, old_johanna_targets_excel_fp, new_isaac_targets_excel_fp
from tavidifficulty.data.parse_targets import get_targets_dataframe
from tavidifficulty.data.parse_metadata import parse_metadata
from tavidifficulty.config import p_id_column_name
from tavidifficulty.data.resource_config import gender_fp, results_folder_fp, important_sheet_name
import logging
import pickle
from tavidifficulty.preprocessing.preprocess_metadata import fix_metadata
# %%
def parse_results_file_and_targets_to_patients(results_fp, target_df):
    # first we parse the targets

    p_id = int(extract_patient_id_from_results_excel_fp(results_fp))

    results_exel = pd.ExcelFile(results_fp)
    condensed_df = get_excel_sheet_selection(results_exel, important_sheet_name, skiprows=3, cols="A:J", nrows=5)
    detailed_segments_df = get_excel_sheet_selection(results_exel, important_sheet_name, skiprows=10, cols="A:H") # we do not specify the amt of rows, get all from row 11

    segments = []
    for _, row in condensed_df.iterrows():
        segment_values = []
        for _, value in row.items():
            segment_values.append(value)
        
        segment = Segment(*segment_values)
        segments.append(segment)
    
    # now we get the target for this patient
    if p_id in target_df.index:
        target = target_df.loc[p_id]
        se_device = target.get('device_se', None)
        optimal = target.get('classification_optimal', None)
    else:
        logging.info(f"patient {p_id} not found in targets")
        se_device = None  # or another fallback value
        optimal = None   # or another fallback value
    
    patient = Patient(p_id, *segments, detailed_segments_df, se_device, optimal)
    return patient
# %%
# now we do it for all files.
def parse_all_results_and_targets_files_to_patients(use_old_targets = True, use_new_targets = False, parent_folder = results_folder_fp, remove_incomplete=False, remove_outliers=False, remove_without_targets=False):
    p_counter = 0
    skipped_folders = 0
    patients = []

    target_df = get_targets_dataframe(old_data=use_old_targets, new_data=use_new_targets)
    metadata_df, _ = parse_metadata()
    metadata_df = fix_metadata(metadata_df, target_df)

    for item in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, item)
        # Check if the item is a directory (subfolder)
        if os.path.isdir(subfolder_path):
            # List all items in the subfolder
            files = os.listdir(subfolder_path)
            
            # Count Excel files with "result" in their filename (case-insensitive)
            count_results = 0
            results_files = []
            for f in files:
                file_path = os.path.join(subfolder_path, f)
                if os.path.isfile(file_path):
                    # Check if the file has an Excel extension and contains "result" in the name
                    if f.lower().endswith(('.xls', '.xlsx', '.xlsm')) and 'result' in f.lower():
                        count_results += 1
                        results_files.append(file_path)
            
            # Determine if at least one results file is present
            results_files_present = count_results > 0

            if not results_files_present:
                skipped_folders += 1
                continue

            if results_files_present:
                # only then do we parse the results file.
                latest_filepath, _ = get_most_recent_file(excel_fps=results_files)
                
                patient = parse_results_file_and_targets_to_patients(latest_filepath, target_df)
                p_counter += 1
                logging.info(f"patient counter: {p_counter}")
                patients.append(patient)
    
    logging.info(f"parsed {len(patients)} result files")
    logging.info(f"skipped {skipped_folders} because no results_file was found")

    if remove_incomplete:
        amt_patients_before = len(patients)
        complete_patients = []
        for patient in patients:
            if patient.incomplete:
                logging.warning(f"incomplete patient id: {patient.id}")
                continue
            complete_patients.append(patient)
        
        patients = complete_patients
        logging.info(f"removed {amt_patients_before - len(patients)} patients because I believe that the annotation process did not finalize")

    if remove_outliers:
        amt_patients_before = len(patients)
        patients = [patient for patient in patients if not str(patient.id) in anomalous_patient_ids]
        logging.info(f"removed {amt_patients_before - len(patients)} patients because the anomaly scores were deemed too high. We are investivating")

    if remove_without_targets:
        amt_patients_before = len(patients)
        patients_with_targets, patients_without_targets = [], []
        for patient in patients:
            if patient.se_device is None:
                patients_without_targets.append(patient.id)
                continue
            patients_with_targets.append(patient)
            
        patients = patients_with_targets
        logging.warning(f"removed {len(patients_without_targets)} patients because they did not have a target. Here is the list of patients without targets but where we have results files: \n{patients_without_targets}")

    # add the gender to the patients
    for patient in patients:
        # raise error is there is no gender or target
        if ((patient.se_device is not None) and (int(patient.id) not in metadata_df.index)):
            print(f"truth value: (int(patient.id) not in metadata_df) {(int(patient.id) not in metadata_df.index)}")
            print(f"patient without metadata: {patient.id}")

        if int(patient.id) in metadata_df.index:
            patient.gender = metadata_df["gender"].loc[int(patient.id)]
            patient.age = metadata_df["age"].loc[int(patient.id)]
            patient.preop_log_euroscore = metadata_df["preop_log_euroscore"].loc[int(patient.id)]
            patient.preop_euroscore_ii = metadata_df["preop_euroscore_ii"].loc[int(patient.id)]
            patient.preop_sts_prom = metadata_df["preop_sts_prom"].loc[int(patient.id)]
            patient.preop_sts_morb_or_mort = metadata_df["preop_sts_morb_or_mort"].loc[int(patient.id)]
            patient.prosthesis_size = metadata_df["prosthesis_size"].loc[int(patient.id)]

        else:
            logging.info(f"patient id not found in metadata file: {patient.id}")

    print(f"final patient count: {len(patients)}")
    return patients

def save_patients_to_pickle():
    patients = parse_all_results_and_targets_files_to_patients(remove_incomplete=True, remove_without_targets=True)

    patient_cached_fp = "/srv/data/TAVIDifficulty/tavidifficulty/src/tavidifficulty/cached_objects/patients_default.pkl"
    with open(patient_cached_fp, mode='wb') as f:
        pickle.dump(patients, f)
        print(f"patients list saved at: {patient_cached_fp}, with remove_incomplete=True, remove_without_targets=True.")

def get_patients_from_cache():
    logging.warning("do not overuse this function if you're not sure that the saved object was saved properly")

    patient_cached_fp = "/srv/data/TAVIDifficulty/tavidifficulty/src/tavidifficulty/cached_objects/patients_default.pkl"
    with open(patient_cached_fp, mode='rb') as f:
        patients = pickle.load(f)
        print(f"patients list loaded from: {patient_cached_fp}. Patient count: {len(patients)}")

    return patients


def parse_results_files_to_dfs():
    # function that returns a dataframe.
    # 1. get all patients
    patients = parse_all_results_and_targets_files_to_patients(remove_incomplete=True)
    patients_dataframes = [patient.params_dataframe for patient in patients]
    combined_dataframe = pd.concat(patients_dataframes)
    return combined_dataframe