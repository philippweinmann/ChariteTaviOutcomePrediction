# %%
from tavidifficulty.data.parse_results_files import parse_all_results_and_targets_files_to_patients
# %%
patients = parse_all_results_and_targets_files_to_patients()
# %%
patients_with_gender = [patient for patient in patients if (patient.gender is not None)]
print(f"patients with gender: {len(patients_with_gender)}")
# %%
# let's look at the non optimal ones
non_optimal_gender_patients = [patient for patient in patients if not patient.optimal]
gender1 = len([patient for patient in non_optimal_gender_patients if patient.gender])
gender2 = len([patient for patient in non_optimal_gender_patients if not patient.gender])

print(gender1)
print(gender2)

# %%
# let's look at the optimal ones
non_optimal_gender_patients = [patient for patient in patients if patient.optimal]
gender1 = len([patient for patient in non_optimal_gender_patients if patient.gender])
gender2 = len([patient for patient in non_optimal_gender_patients if not patient.gender])

print(gender1)
print(gender2)
# %%
