import os
import importlib, os, sys
import shutil
import time
import json
import cv2
import numpy as np
#################################################################################################################

if __name__ == "__main__":


    maindir = "/media/fogbrain/6TB/EmergenTexture_Full_Code/all_dirs"
    fl_name="textures/sample_000//BaseColor.jpg"
    fl_name_seamless ="Samples_512x512_10_seamless/sample_000//BaseColor.png"
    counter = 0
    for topic_subdir in os.listdir(maindir):
        topicdir  = os.path.join(maindir, topic_subdir)
        for ii,sdir  in enumerate(os.listdir(topicdir)):
            counter+=1

            print(counter,":",topicdir,")",ii,sdir)
            code_dir = os.path.join(topicdir,sdir)
            if not os.path.exists(os.path.join(code_dir, fl_name_seamless)): continue
            img_path = os.path.join(code_dir,fl_name)
            im=cv2.imread(img_path)
            im = np.hstack((im,im))
            im = np.vstack((im,im))

            img_path_seamless = os.path.join(code_dir, fl_name_seamless)
            im2 = cv2.imread(img_path_seamless)
            im2 = np.hstack((im2, im2))
            im2 = np.vstack((im2, im2))


            print(counter,code_dir)
            while True:
                cv2.imshow("Replace: Yes/No/problematic",np.hstack([im,im2]))
                ky=cv2.waitKey(0)

                if chr(ky) == "y":
                    print(code_dir)
                    with open(os.path.join(code_dir,"SEAMLESS"),"w") as fl: fl.close()
                    os.remove(os.path.join(code_dir,"NOT_SEAMLESS"))
                    os.remove(os.path.join(code_dir,"generate.py"))
                    shutil.move(os.path.join(code_dir,"generate_seamless.py"),os.path.join(code_dir,"generate.py"))

                    shutil.rmtree(os.path.join(code_dir,"Samples_512x512_10"))
                    shutil.move(os.path.join(code_dir,"Samples_512x512_10_seamless"),os.path.join(code_dir,"Samples_512x512_10"))

                    shutil.rmtree(os.path.join(code_dir, os.path.join(code_dir, "Samples_1024x1024_10")))
                    shutil.move(os.path.join(code_dir, "Samples_1024x1024_10_seamless"), os.path.join(code_dir, "Samples_1024x1024_10"))


                    break
                if   chr(ky) == "n":
                    print(code_dir)
                    os.remove(code_dir+"//generate_seamless.py")
                    shutil.rmtree(os.path.join(code_dir, "Samples_512x512_10_seamless"))
                    shutil.rmtree(os.path.join(code_dir, "Samples_1024x1024_10_seamless"))
                    break
