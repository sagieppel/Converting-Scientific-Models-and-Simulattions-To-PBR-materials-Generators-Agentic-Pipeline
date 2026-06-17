import os
import importlib, os, sys
import time
import json
import cv2
import numpy as np
import tools.VisualQuestion as VQ
#################################################################################################################

if __name__ == "__main__":


    maindir = "/home/infinipig/Documents/EmergenTexture_Full_Code/all_dirs"
    fl_name="textures/sample_000//BaseColor.jpg"
    counter = 0
    not_seamless=0
    code_model = "google/gemini-3.5-flash"
    for topic_subdir in os.listdir(maindir):
        topicdir  = os.path.join(maindir, topic_subdir)
        for ii,sdir  in enumerate(os.listdir(topicdir)):
            counter+=1
            print(counter,":",topicdir,")",ii,sdir)
            code_dir = os.path.join(topicdir,sdir)
            if os.path.exists(os.path.join(code_dir,"generate_seamless.py")) or os.path.exists(os.path.join(code_dir, "no_seamless_fix")): continue
            if os.path.exists(os.path.join(code_dir,"NOT_SEAMLESS")):
                 with open(os.path.join(code_dir,"generate.py"),"r") as fl:
                     code = fl.read()
                 txt=("The code below generate PBR texture which is not seamless. Or not completely seamless (tillable).\n"
                      "Look at the code is there a way to make the code generated seamless texture.\n"
                      "In general if the method does not support seamless texture, or you cant see a simple way to change the code to generate seamless texture. Answer no.\n"
                      "If there is simple way to change the code so it will generate the same textures but seamless answer yes and supply fix code.\n"
                      "Here is the code:****\n"+code+"\n****\n"
                      "Provide your  answer as a parsable json dictionary of the following format:\n"
                      "{'seamless':'yes'/'no' can you make the code generate seamless textures,'code': fixed code (Same format as the original code) no mark down or anything else}\n"
                      "Only add the code if you answer yes for seamless")
                 try:
                     code_dic = VQ.get_reponse(text=txt, as_json=True, model=code_model)
                     print(ii,sdir)
                     if "seamless" in code_dic and code_dic["seamless"].lower()=="yes" and "code" in code_dic:
                          with open(os.path.join(code_dir,"generate_seamless.py"),"w") as fl:
                                  fl.write(code_dic["code"])
                     elif "seamless" in code_dic and code_dic["seamless"].lower()=="no":
                         with open(os.path.join(code_dir, "no_seamless_fix"), "w") as fl: fl.close()
                 except:
                     continue






