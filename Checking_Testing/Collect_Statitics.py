# collect statitics regardin accuract
import json
import os
import numpy
import  json_pkl
benchmark_dir = "Textures_Final_100//endless_textures_Final_Selection_All_Good//"#

# acc_d={}
# for dr in os.listdir(benchmark_dir):
#     pth= benchmark_dir + "//" + dr +"//model_accuracy.json"
#     if os.path.exists(pth):
#        res = json_pkl.read_json(pth)
#        if not res['choice'] in acc_d:
#            acc_d[res['choice']]=0
#        acc_d[res['choice']] +=1
#     print(acc_d)
acc_d={}
for dr in os.listdir(benchmark_dir):
    pth= benchmark_dir + "//" + dr +"//Errors.json"
    if os.path.exists(pth):
       res = json_pkl.read_json(pth)
       if not res['choice'] in acc_d:
           acc_d[res['choice']]=0
       acc_d[res['choice']] +=1
    print(acc_d)
