# %%
from tavidifficulty.data.parse_results_files import parse_all_results_and_targets_files_to_patients
import matplotlib.pyplot as plt
from tavidifficulty.config import min_aorta_lengths
# %%
patients = parse_all_results_and_targets_files_to_patients()
# %%
segments = ["lvot_segment", "aorta_asc_segment", "aorta_desc_segment", "abdominal_aorta_segment"]

lengths = []

for patient in patients:
    patient_segment_lengths = []
    for segment in segments:
        patient_segment_lengths.append(getattr(patient, segment).length)
    
    lengths.append(patient_segment_lengths)

lengths = [list(t) for t in zip(*lengths)]
print(lengths)
# %%
def hist_plot_seg_lengths(segment_lengths, type_str):
    plt.hist(segment_lengths, bins=30)

    plt.xlabel("length of segment [mm]")
    plt.ylabel("amount of patients")
    plt.title(f"{type_str} lengths")
    plt.show()
# %%
for type_str, seg_lengths in zip(segments, lengths):
    hist_plot_seg_lengths(seg_lengths, type_str)
# %%
for patient in patients:
    if patient.incomplete:
        print(patient.id)
        for key, value in min_aorta_lengths.items():
            print(getattr(patient, key).length)
# %%