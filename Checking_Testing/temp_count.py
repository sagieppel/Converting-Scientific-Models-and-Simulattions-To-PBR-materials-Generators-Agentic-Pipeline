import os
if __name__ == "__main__":
    x=0

    maindir = "/home/infinipig/Documents/EmergenTexture_Full_Code/all_dirs"
    log_error_file="/home/infinipig/Documents/EmergenTexture_Full_Code/fail_runs.json"
    log_attempts_file = "/home/infinipig/Documents/EmergenTexture_Full_Code/attempt_file.json"


    out_sub_dir = r"Samples_1024x1024_10/"
    time_file = r"Time_1024x1024_10.txt"
    num_samples = 10
    sz = 1024
    counter=0
    for topic_subdir in os.listdir(maindir):
        topicdir  = os.path.join(maindir, topic_subdir)
        for ii,sdir  in enumerate(os.listdir(topicdir)):
            counter+=1
            print(counter,":",topicdir,")",ii,sdir)
            code_dir = os.path.join(topicdir,sdir)
            out_dir = os.path.join(code_dir,out_sub_dir)

            if not os.path.isdir(out_dir): os.mkdir(out_dir)
            if len(os.listdir(out_dir))==num_samples: x+=1
            print(x)