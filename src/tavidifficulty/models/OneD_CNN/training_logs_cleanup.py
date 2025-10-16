# %%
from pathlib import Path
from datetime import datetime
import shutil
from tavidifficulty.config import training_logs_folder

fstring = "%Y-%m-%d_%H-%M-%S"
def cleanup_trainingLogs():
    datetime_objects = []
    datetime_filenames = []
    for item in training_logs_folder.iterdir():
        name_to_parse = item.stem if item.is_file() else item.name

        try:
            datetime_object = datetime.strptime(name_to_parse, fstring)
            datetime_objects.append(item)
            datetime_filenames.append(name_to_parse)
        except:
            continue
    
    print("Files or folders:")
    for filename in datetime_filenames:
        print(f" • {filename}")
    user_input = input("Delete? Y/N: ")

    if user_input == "Y":
        for item in datetime_objects:
            if item.is_file():
                item.unlink()
                print(f"Deleted file: {item.name}")
            else:
                # If you want to handle directories too
                shutil.rmtree(item)
                print(f"Deleted directory: {item.name}")
    else:
        print("aborting")


cleanup_trainingLogs()
# %%
>