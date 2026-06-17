import os
import shutil
if __name__ == "__main__":



    main_dirs=["/media/fogbrain/6TB/python_project/PBR_Materials_Generator/pbr_general",
"/media/fogbrain/6TB/python_project/PBR_Materials_Generator/pbr_earth_science",
"/media/fogbrain/6TB/python_project/PBR_Materials_Generator/pbr_art",
"/media/fogbrain/6TB/python_project/PBR_Materials_Generator/pbr_math",
"/media/fogbrain/6TB/python_project/PBR_Materials_Generator/pbr_biology",
"/media/fogbrain/6TB/python_project/PBR_Materials_Generator/pbr_social_science",
"/media/fogbrain/6TB/python_project/PBR_Materials_Generator/pbr_chemistry_materials",
"/media/fogbrain/6TB/python_project/PBR_Materials_Generator/pbr_phsyics_engineering",
"/media/fogbrain/6TB/python_project/PBR_Materials_Generator/pbr_creative",
               "/media/fogbrain/6TB/python_project/PBR_Materials_Generator/pbr_complex_original"]

    out_dir="/media/fogbrain/6TB/python_project/PBR_Materials_Generator/renders"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    itr=0
    for main_dir in main_dirs:
        for sdr in os.listdir(main_dir):
             workdir = os.path.join(main_dir, sdr)

             if os.path.exists(os.path.join(workdir, "sphere1.jpg")):
                     shutil.copyfile(os.path.join(workdir, "sphere1.jpg"), os.path.join(out_dir, sdr+".jpg"))