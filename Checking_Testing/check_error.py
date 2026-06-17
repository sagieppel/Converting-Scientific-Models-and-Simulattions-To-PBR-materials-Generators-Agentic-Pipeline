import os
import sys
import shutil
import cv2
import json
import pickle
maindir = r"/media/fogbrain/6TB/EmergenTexture_Full_Code/all_dirs/"
out_sub_dir = r"Samples_512x512_10/"
num_samples = 10
sz = 512
counter=0
errors=0
with open("/media/fogbrain/6TB/EmergenTexture_Full_Code/fail_runs.json","r") as fl:
    fail_runs = json.load(fl)


for topic_subdir in os.listdir(maindir):
    topicdir  = os.path.join(maindir, topic_subdir)

    for ii,sdir  in enumerate(os.listdir(topicdir)):
        counter += 1
        print(counter,":",topicdir,")",ii,sdir)
        code_dir = os.path.join(topicdir,sdir)
        out_dir = os.path.join(code_dir,out_sub_dir)

        if not os.path.isdir(out_dir): os.mkdir(out_dir)
        if len(os.listdir(out_dir))>=num_samples: continue
        errors+=1
    print(counter,":",topicdir,",",sdir, "Errors:",errors, "Total",counter)
