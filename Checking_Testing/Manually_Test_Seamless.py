import os
import importlib, os, sys
import time
import json
import cv2
import numpy as np
#################################################################################################################

if __name__ == "__main__":


    maindir = "/home/infinipig/Documents/EmergenTexture_Full_Code/all_dirs"
    fl_name="textures/sample_000//BaseColor.jpg"
    counter = 0
    for topic_subdir in os.listdir(maindir):
        topicdir  = os.path.join(maindir, topic_subdir)
        for ii,sdir  in enumerate(os.listdir(topicdir)):
            counter+=1
            print(counter,":",topicdir,")",ii,sdir)
            code_dir = os.path.join(topicdir,sdir)
            if os.path.exists(os.path.join(code_dir,"NOT_SEAMLESS")) or os.path.exists(os.path.join(code_dir,"SEAMLESS")) or os.path.exists(os.path.join(code_dir, "PROBLEM")): continue
            img_path = os.path.join(code_dir,fl_name)
            im=cv2.imread(img_path)
            im = np.hstack((im,im))
            im = np.vstack((im,im))
            print(counter,code_dir)
            while True:
                cv2.imshow("Seamless: Yes/No/problematic",im)
                ky=cv2.waitKey(0)
                if chr(ky) == "n":
                    with open(os.path.join(code_dir,"NOT_SEAMLESS"),"w") as fl: fl.close()
                    break
                if chr(ky) == "y":
                    with open(os.path.join(code_dir,"SEAMLESS"),"w") as fl: fl.close()
                    break
                if chr(ky) == "p":
                    with open(os.path.join(code_dir, "PROBLEM"), "w") as fl: fl.close()
                    break