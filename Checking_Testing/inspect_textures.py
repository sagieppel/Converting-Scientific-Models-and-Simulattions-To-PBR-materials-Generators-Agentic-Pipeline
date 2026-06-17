
import shutil

import os

import cv2
import numpy as np

import json_pkl

# manual check and filter benchmarks display several images from each model and ask user to validate or mark them
# note this will update in the data.pkl file and next time you run Create_SciTextures_Dataset.py with this file it will redo all marked files

#################################################################################################################
def display(tx_dir,sz=1200,grd=3,name=""):
    x=y=0
    ps=int(sz/grd)
    all=0
    fail=0


    for fl in os.listdir(tx_dir):
        for tp in [".png",".jpg",".PNG",".JPG",".tiff"]:
              if tp in fl:
                  im=cv2.imread(tx_dir+"/"+fl)
                  if im.max() < 10 or len(np.unique(im)) == 1:
                      fail+=1
                  all+=1
                  im=cv2.resize(im,[ps,ps])
                  if x==grd:
                      x=0
                      y+=1
                      if y==1:
                          full_im=im_line.copy()
                      else:
                          full_im=np.concatenate([full_im,im_line],axis=0)
                      if y>grd: break
                  if x==0:
                      im_line=im.copy()
                  else:
                      im_line=np.concatenate([im_line,im],axis=1)
                  x+=1
    cv2.destroyAllWindows()
    if fail/all>0.5: return 'f'
    if fail / all > 0.25: return 'm'

    cv2.imshow(name+ " [a]ccept, [f]ail, [m]edium, [r]return/prev",full_im)
    while(True):
          ky = cv2.waitKey()
          ky = chr(ky)
          if ky in ['a','f','m','r']: return ky
#############################################################################################################################
def update_folder_name(main_dir):
    if not os.path.exists(fail_dir): os.mkdir(fail_dir)
    if not os.path.exists(med_dir): os.mkdir(med_dir)
    data=json_pkl.read_pkl(main_dir+"//data.pkl")
    x=0
    for ky in data['benchmarks']:
       print(ky,os.path.exists(main_dir+"/"+ky))
       if 'dir' in data['benchmarks'][ky]:
            data['benchmarks'][ky]['dir']=data['benchmarks'][ky]['dir'].replace("endless_textures", "").replace("/", "")
       else:
           print(ky, os.path.exists(main_dir + "/" + ky),"\n",data['benchmarks'][ky]['overlap'])
           x+=1
           print(x)
    json_pkl.save_pkl(data,main_dir+"//data.pkl")
    json_pkl.save_pkl(data,main_dir+"//data_back.pkl")
    print(x)

##########################################################################################################33
def check_all_set(main_dir, fail_dir, med_dir, min_im):
    if not os.path.exists(fail_dir): os.mkdir(fail_dir)
    if not os.path.exists(med_dir): os.mkdir(med_dir)

    data = json_pkl.read_pkl(main_dir + "//data.pkl")
    x = 0

    lst_dr= os.listdir(main_dir)
    data['abandoned_ideas']={}
    kys=list(data["benchmarks"].keys())
    for ky in kys:
        if "dir" not in data["benchmarks"][ky]:
            data['abandoned_ideas'][ky]=data["benchmarks"][ky]
            del data["benchmarks"][ky]
            print(data['abandoned_ideas'][ky])
    json_pkl.save_pkl(data, main_dir + "//data.pkl")
    json_pkl.save_pkl(data, main_dir + "//data_back.pkl")
    # for k in kys:
    #     data['benchmarks']
    indx=-1
    ch='a'
    while(True):
        indx += 1
        if indx>=len(lst_dr): break
        dr=lst_dr[indx]
        print(indx, dr)
        if not os.path.isdir(main_dir + "//" + dr): continue
        pth = main_dir + "//" + dr + "//textures//"
        # if not os.path.isdir(pth):
        #     shutil.move(main_dir + "//" + dr,fail_dir + "//" + dr)
        #     continue

#  find dir name  and index in data file
        for bnm in data['benchmarks']:
            ent = data['benchmarks'][bnm]
            if dr == ent['simple name']: break
            ent=None
        if ent==None:
            print("cant find ", dr, " record") # can find folder in data file
            continue
        if "checked" in ent  and ch!='r':
            print("Exists")
            continue  # if already done  continue

       # ent=kys[indx]
       # if not 'dir' in ent: c

        if os.path.exists(pth) and len(os.listdir(pth))>min_im:
           ch=display(pth,sz=1200,grd=3,name=str(indx)+")"+dr) # display and ask for reply
        else:
            ch="f"
        print(ch)
        ent["checked"] = "pass"
        if ch == 'r':
              indx-=2
        if ch == 'f':
              shutil.move(main_dir+"//"+dr,fail_dir+"//"+dr)
              ent["checked"] = "redo"
        if ch == 'm':
              shutil.move(main_dir+"//"+dr,med_dir+"//"+dr)
              ent["checked"] = "redo"



        if indx%4==0:
            json_pkl.save_pkl(data, main_dir + "//data.pkl")
        if indx%10==0:
            json_pkl.save_pkl(data, main_dir + "//data_back.pkl")
    json_pkl.save_pkl(data, main_dir + "//data.pkl")
    json_pkl.save_pkl(data, main_dir + "//data_back.pkl")


    for ky in data['benchmarks']:
        print(ky, os.path.exists(main_dir + "/" + ky))
        if 'dir' in data['benchmarks'][ky]:
            data['benchmarks'][ky]['dir'] = data['benchmarks'][ky]['dir'].replace("endless_textures", "").replace("/","")

    json_pkl.save_pkl(data, main_dir + "//data.pkl")
    json_pkl.save_pkl(data, main_dir + "//data_back.pkl")
    print(x)

############################################################################################################################
main_dir= r"../Scitextures/" # Input dir of models to inspect, models generated by Create_Textures_Models_Code.py
fail_dir= r"../Scitextures_fail/" # Move all fail model in thi dir
med_dir= r"../Scitextures_medium/" # put models which are problematic but not complete fail
min_im=6 # Minimum number of images in folder
check_all_set(main_dir,fail_dir,med_dir,min_im=min_im)

#
# lst_dr= os.listdir(main_dir)
# indx=0
# while(True):
#     dr=lst_dr[indx]
#     pth=main_dir+"//"+dr+"//textures//"
#     if os.path.exists(pth):
#        x=display(pth,sz=1200,grd=3,name=dr)
#        print(x)
#     indx+=1
#     if x=='r':
#       indx-=2