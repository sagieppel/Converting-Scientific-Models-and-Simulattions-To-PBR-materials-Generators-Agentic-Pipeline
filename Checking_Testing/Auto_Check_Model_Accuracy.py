# Classify model accuracy level 'accurate', 'good approximation', 'toy model','weak model','inspired'
# Will add model accuracy (with discussion) in each model folder in model_accuracy.txt
# Run after Create_Textures_Models_Code.py
# see main __main__ for parameters
import os
import json_pkl

import tools.VisualQuestion as VQ
def check_accuracy(main_dir,model):
    for dr in os.listdir(main_dir):
        code_file = main_dir + "//" + dr + "//generate.py"
        if os.path.exists(main_dir + "//" + dr + "//model_accuracy.txt"): continue
        if not os.path.exists(code_file): continue
        with open(code_file,"r") as fl:
            code=fl.read()
        txt=("The following script model or claim to model some system to generate visual pattern."
             "\nLook carefully at the code and tell me does this code actually simulate or model what it claim it does."
             "\nIs it:"
             "\n1) Accurate simulation, of the system."
             "\n2) Good approximation  might be flowed or miss some details but still capture the general process, and is  a good approximation to the system behavior."
             "\n3) Toy model, that capture the core idea of the system but doesnt actually simulate the real process."
             "\n4) Weak model, that capture some aspect of the system but miss most of the important aspects."
             "\n5) Inspired, the code doesnt model anything related to the system and just generate pretty pattern that might be inspired by it or try to capture the general look."
             "Your answer should come as ready to load json dictionary of the following format:"
             "{'choice': <one of  the following in single letter: 'accurate', 'good approximation', 'toy model','weak model','inspired'>"
             " 'explain': explain your choice in free language}"
             "\n\n\n\nHere is the code:\n\n"+code)
        print(txt)
        response=VQ.get_response_image_txt_json(txt,model=model)
        json_pkl.save_json(response,main_dir + "//" + dr + "//model_accuracy.json")
        txt="Generat model accuracy: "+response['choice']
        txt+="\nExplanation:\n\n"+response['explain']
        with open(main_dir + "//" + dr + "//model_accuracy.txt", "w") as fl:
            fl.write(txt)
        print(txt+"\n\n\n\n\n*********************************************************\n\n\n\n\n\n")
        # with open(main_dir + "//" + dr + "//"+response['choice'].replace(" ","_") +".txt", "w") as fl:
        #         fl.write(txt)
#--------------------------------------------------------------------------------------------------------------------
if __name__=="__main__":
    model = "claude-sonnet-4-5-20250929" # model use for evaluation (you must have API key in API_KEY.py)
    main_dir=r"..//Scitextures" # Input Folder with models generated in Create_Textures_Models_Code.py
    check_accuracy(main_dir=main_dir,model=model)