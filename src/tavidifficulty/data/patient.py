# %%
from tavidifficulty.config import min_aorta_lengths
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging
# %%
class Patient:
    def __init__(self, id, lvot_segment, aorta_asc_segment, aorta_desc_segment, abdominal_aorta_segment, detailed_sections_df, se_device, optimal):
        self.id = id
        self.lvot_segment = lvot_segment
        self.aorta_asc_segment = aorta_asc_segment
        self.aorta_desc_segment = aorta_desc_segment
        self.abdominal_aorta_segment = abdominal_aorta_segment
        self.detailed_sections_df = detailed_sections_df

        self.segments = [self.lvot_segment, self.aorta_asc_segment, self.aorta_desc_segment, self.abdominal_aorta_segment]

        # operational decisions
        self.se_device = se_device # if this is false, then be_device was used

        # patient metadata
        self.gender = None
        self.age = None
        self.preop_log_euroscore = None
        self.preop_euroscore_ii = None
        self.preop_sts_prom = None
        self.preop_sts_morb_or_mort = None
        self.prosthesis_size = None

        # label
        self.optimal = optimal

    @property
    def incomplete(self):
        # we determine this using the aorta lengths
        for key, value in min_aorta_lengths.items():
            if value >= getattr(self, key).length:
                logging.warning(f"incomplete because of: {key} length: {getattr(self, key).length}. This is considered too short to be possible")
                return True
        
        return False

    @property
    def params_condensed_dataframe(self):
        # Define segment names in order.
        # note that only the two first segments seem to be relevant
        # this might be due the model overfitting, but I need to verify this.
        segment_names = ['lvot_segment', 'aorta_asc_segment', 'aorta_desc_segment', 'abdominal_aorta_segment']
        
        # Create a list to hold modified series.
        series_list = []
        
        for name, segment in zip(segment_names, self.segments):
            # Copy to avoid modifying the original series.
            s = segment.params_series.copy()
            # Prefix the index with the segment name.
            s.index = [f"{name}_{param}" for param in s.index]
            series_list.append(s)
        
        # Concatenate the series into one long series.
        flat_series = pd.concat(series_list)

        # add the device used
        flat_series['se_device'] = 1 if self.se_device else 0
        
        # Wrap the flattened series in a DataFrame with a single row.
        return pd.DataFrame([flat_series], index=[self.id])
    
    @property
    def get_extended_params_without_abdominal_aorta(self):
        length_aorta_asc = self.aorta_asc_segment.length
        length_aorta_desc = self.aorta_desc_segment.length
        cutoff_length = length_aorta_asc + length_aorta_desc

        detailed_section_df_copy = self.detailed_sections_df.copy()
        detailed_section_df_without_abdominal_aorta = detailed_section_df_copy[detailed_section_df_copy["Distance from Valve[mm]"] < cutoff_length]

        return detailed_section_df_without_abdominal_aorta
    
    @property
    def metadata_series(self):
        metadatas_dict = {
            "gender": self.gender,
            "age": self.age,
            "preop_log_euroscore": self.preop_log_euroscore,
            "preop_euroscore_ii": self.preop_euroscore_ii,
            "preop_sts_prom": self.preop_sts_prom,
            "preop_sts_morb_or_mort": self.preop_sts_morb_or_mort,
            "prosthesis_size": self.prosthesis_size,
        }
        metadatas = pd.DataFrame(metadatas_dict, index=[int(self.id)])

        return metadatas

                     
class Segment:
    def __init__(self, seg_area, length, tortuosity, curvature, avg_curvature, acc_curvature, avg_acc_curvature, min_diameter, avg_diameter, calcification):
        self.seg_area = seg_area
        self.length = length
        self.tortuosity = tortuosity
        self.curvature = curvature
        self.avg_curvature = avg_curvature
        self.acc_curvature = acc_curvature
        self.avg_acc_curvature = avg_acc_curvature
        self.min_diameter = min_diameter
        self.avg_diameter = avg_diameter
        self.calcification = calcification

    @property
    def params_series(self):
        # Create a copy of the instance dictionary and remove 'seg_area'
        params = self.__dict__.copy()
        params.pop('seg_area', None)
        return pd.Series(params)

def detect_boolean_columns(df):
    return [col for col in df.columns if df[col].nunique() == 2]

# TODO find the best percentile value
def clip_outliers(df, percentile=0.05):
    lower = df.quantile(percentile)
    upper = df.quantile(1 - percentile)
    return df.clip(lower=lower, upper=upper, axis=1)

def scale_values(df):
    # don't scale boolean columns
    columns_to_scale = [col for col in df.columns if df[col].nunique() > 2]
    df_scaled = df.copy()
    df_scaled[columns_to_scale] = StandardScaler().fit_transform(df[columns_to_scale])

    return df_scaled

def get_dataset(patients, clip_values=True, percentile=0.05, scale=False):
    # Create a list to hold the dataframes for each patient.
    dataframes = []
    optimal_outcomes = []
    
    # Iterate over the patients.
    for patient in patients:
        # Get the patient's parameters as a dataframe.
        df = patient.params_condensed_dataframe
        # Append the dataframe to the list.
        dataframes.append(df)
        optimal_outcomes.append({'id': patient.id, 'optimal': patient.optimal})
    
    # Concatenate the dataframes into one large dataframe.
    input_df = pd.concat(dataframes)

    # output is if the patient had an optimal outcome or not
    expected_output_df = pd.DataFrame(optimal_outcomes)
    expected_output_df.set_index('id', inplace=True)
    expected_output_series = expected_output_df.squeeze()

    if clip_values:
        input_df = clip_outliers(input_df, percentile=percentile)
    
    if scale:
        input_df = scale_values(input_df)

    return input_df, expected_output_series
# %%