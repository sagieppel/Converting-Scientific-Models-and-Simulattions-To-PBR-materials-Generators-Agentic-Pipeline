import os
# check average running time for all models
def creation_time_difference(folder_path: str) -> float:
    """
    Returns the difference in seconds between the creation time
    of the first (oldest) and last (newest) file in a given folder.

    Args:
        folder_path (str): Path to the folder to check.

    Returns:
        float: Difference in seconds between oldest and newest file creation time.
               Returns 0 if there are fewer than 2 files.
    """
    # Get all files in the folder (ignore subdirectories)
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path, f))]

    if len(files) < 2:
        return 0.0  # Not enough files to compare

    # Get creation times for all files
    creation_times = [os.path.getctime(f) for f in files]

    # Find oldest and newest
    oldest = min(creation_times)
    newest = max(creation_times)

    return newest - oldest
##################################################################################3

main_dir = r"Textures_Final_100/endless_textures_Final_Selection_All_Good//"
all_tm=0
all_tsts=0
for dr in os.listdir(main_dir):


    if os.path.isdir(main_dir+"//"+dr+"//"+"new_textures_C//") and len(main_dir+"//"+dr+"//"+"new_textures_C//")>=10:

        tm=creation_time_difference(main_dir+"//"+dr+"//"+"new_textures_C//")
        all_tm +=tm
        all_tsts +=1
        print(all_tsts,") all time hours=",all_tm/60/60,"  time sec",tm)