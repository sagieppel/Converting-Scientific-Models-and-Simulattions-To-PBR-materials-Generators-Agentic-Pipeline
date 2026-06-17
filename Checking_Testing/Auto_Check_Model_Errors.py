# Identify and classify model error level 'Accurate', 'Minor errors', 'Major errors','Undecided'
# Will automatically mark the error level in each model folder in Errors.json and will discuss the errors found
# You can also instruct to fix major errors
# Run after Create_Textures_Models_Code.py
# see main __main__ for parameters

import os
import json_pkl
import tools.VisualQuestion as VQ
import time

def check_accuracy(main_dir,model,fix=True):
    for dr in os.listdir(main_dir):

        code_file = main_dir + "//" + dr + "//generate.py"# file path for  for fixed model code (if fix==True and model have major errors)
        if not os.path.exists(code_file): continue
        with open(code_file,"r") as fl:
            code=fl.read()

        if not os.path.exists(main_dir + "//" + dr + "//Errors.txt"):
            txt=("The following script model some system to generate visual pattern."
                 "\nLook carefully and identify what it claim  to do and see if you can find any errors."
                 "\nIs it:"
                 "\n1) No clear error in the code. The code might be approximation or toy model but do what it claim to do with no obvious errors."
                 "\n2) Minor fixable errors but the code still mostly do what it say (note that suggestion for improvement doesnt count as error)."
                 "\n3) Major errors or bugs, that mean the code will not actually do what it claim it does even as approximation or toy model."
                 "\n4) Undecided, its unclear what to code try to do."
          
                 "\nYour answer should come as ready to load json dictionary of the following format:"
                 "\n{\n'choice': <one of  the following in single letter: 'Accurate', 'Minor errors', 'Major errors','Undecided'>,"
                 "\n'explain': <explain your choice in free language>\n}"
                 "\n\n\n\nHere is the code:\n\n"+code)
            print(txt)
            response=VQ.get_response_image_txt_json(txt,model=model)
            json_pkl.save_json(response,main_dir + "//" + dr + "//Errors.json")
            txt="Generat model Errors level : "+response['choice']
            txt+="\nDescription:\n\n"+response['explain']
            with open(main_dir + "//" + dr + "//Errors.txt", "w") as fl:
                fl.write(txt)
            print(txt+"\n\n\n\n\n*********************************************************\n\n\n\n\n\n")
            # with open(main_dir + "//" + dr + "//"+response['choice'].replace(" ","_") +"_for_fixed_cod.txt", "w") as fl:
            #         fl.write(txt)
            # with open("errors//" + dr + "_"+response['choice'].replace(" ","_") +"_for_fixed_code.txt", "w") as fl:
            #         fl.write(txt)
        else:
            response = json_pkl.read_json(main_dir + "//" + dr + "//Errors.json")
        if fix and (response['choice']=='Minor errors' or  response['choice']=='Major errors'):
            if os.path.exists(main_dir + "//" + dr + "//generate_fixed.py"): continue
            txt = ("The following code generate some visual patterns however it contain some "+ response['choice']+".\n"
                   "Look at the code and try to fix the errors to make the model more correct.\n"
                   "\n\n"+ code+ "\n\n"
                   "\nHere is the description of the errors:\n"
                   +response['explain']+
                   "\nYou are not allowed to add new packages."
                   "\nYour output should come as parsable json format with the following format:\n"
                   "\n{'code':<The fixed code ready to run>, 'success':<True/False> did you manage to fix all or some of the issues, 'comments':<free text comments>}"
                   )
            response = VQ.get_response_image_txt_json(txt, model=model)
            if 'success' in response and response['success']==True and 'code' in response:
                with open(main_dir + "//" + dr + "//generate_fixed.py","w") as fl:
                    fl.write(response['code'])
                    print("fixed "+dr, " time "+time.strftime("%H:%M:%S"))
                json_pkl.save_json(response,main_dir + "//" + dr + "//second_debug_log.json")





if __name__=="__main__":
    main_dir=r"..//Scitextures"  # The model folder generated in Create_Textures_Models_Code.py
    model="gpt-5" # model use for evaluation (you must have API key in API_KEY.py)
    fix=False # if errors were found should you fix them if true will generate fixed code in generate_fixed.py (in model dir)
    check_accuracy(main_dir=main_dir,model="gpt-5",fix=fix)#claude-sonnet-4-5-20250929 # "gemini-2.5-pro" #"deepseek-ai/DeepSeek-R1-0528"