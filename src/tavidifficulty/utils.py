# %%
from functools import wraps
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
import os
import re
import pickle
import time
# %%
def convert_elements_to_str(data):
    """Convert all elements in a nested list to strings."""
    return [[str(item) for item in row] for row in data]

def build_styled_table(data, style=None):
    """Create a table with a given style."""
    table = Table(data)
    # Apply default style if none provided
    if style is None:
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])
    table.setStyle(style)
    return table

output_dir_path = "/srv/data/TAVIDifficulty/tavidifficulty/output"
def create_pdf_from_dataframe(df, pdf_filename, output_dir=output_dir_path, pagesize=landscape(letter)):
    """Converts a DataFrame to a styled PDF table."""
    # Ensure the output directory exists
    # os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, pdf_filename)

    # include the index
    df = df.reset_index()

    # Convert DataFrame (including headers) to a nested list and ensure all data is a string
    data = [df.columns.tolist()] + df.values.tolist()
    data = convert_elements_to_str(data)

    # Create the document template
    document = SimpleDocTemplate(pdf_path, pagesize=pagesize)

    # Build the styled table
    table = build_styled_table(data)

    # Build the PDF document with the table
    document.build([table])
    print(f"PDF created successfully: {pdf_path}")


# %%
def extract_patient_id(filename):
    filename = re.sub(r"\s+", "", filename)

    # Extract the first sequence of digits
    match = re.match(r"(\d+)_", filename)

    if match:
        extracted_number = match.group(1)
        # print(extracted_number)  # Output: 236105

        return extracted_number
    else:
        raise ValueError(f"filename: {filename}")
    
def extract_patient_id_from_results_excel_fp(filepath):
    # Extract the patient ID using regex
    match = re.search(r"/(\d+)\s*_+(?:\d{2}[._]\d{2}[._]\d{2,4})_", filepath)

    if match:
        patient_id = match.group(1)
        return patient_id
    else:
        raise ValueError(f"filepath: {filepath}")
# %%
def save_df_as_pkl(df, fn):
    fp = "saved_dataframes/" + fn

    with open(fp, mode="wb") as f:
        pickle.dump(df, f)
    
    print(f"dataframe saved at: {fp}")

# decorator function to measure time taken of fct.
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        print(f'Function {func.__name__} took {total_time:.4f}s to run')
        return result
    return timeit_wrapper
# %%
