import os
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import xarray as xr




# def create_oar_file(region_number):
#     with open('oar_template.txt', 'r') as template_file:
#         oar_template = template_file.read()

#     print(f'OAR Template content:\n{oar_template}')  # Add this line for debugging
#     oar_content = oar_template.format(region_number)

#     with open(f'Job_Region{region_number}.oar', 'w') as oar_file:
#         oar_file.write(oar_content)


def create_bat_file(region_number):
    with open('bat_region.py', 'r') as bat_template_file:
        bat_template = bat_template_file.read()

    with open(f'bat_region.py', 'w') as bat_file:
        bat_file.write(bat_template)

    print(f'bat_region.py was read and written for region {region_number}')



def create_oar_file(model_number):
    
    f = open(f'Job_{model_number}.oar', 'w')
    f.write('#!/bin/bash\n')
    f.write(f'#OAR -n Job\n')
    f.write('#OAR -l /nodes=1/core=48,walltime=8:00:00\n')
    f.write('#OAR -O result_prep.%jobid%.out\n')
    f.write('#OAR -E result_prep.%jobid%.err\n')
    f.write('#OAR -t idempotent\n')
    f.write('#OAR -q default\n')
    f.write('#OAR --project ice_speed\n')
    f.write('\n')
    f.write('source /home/moncadaf/.bashrc\n')
    f.write('eval "$(/home/moncadaf/miniconda3/bin/conda shell.bash hook)"\n')
    f.write('conda activate myenv\n')
    f.write('cd /bettik/moncadaf/machine_learning_calving_project/U-net/\n')
    f.write(f'python u_net.py {model_number}\n')
    f.close()
    
    print(f'Job_{model_number}.oar was created')
    
    os.system(f'chmod 777 Job_{model_number}.oar')
    #os.system('source /applis/environments/cuda_env.sh 12.1')
    #os.system(f'oarsub -S ./Job_{model_number}.oar')

def main():       

    for i in range(1,6):
        print(f'Creating job for region {i}')
        create_oar_file(i)
        os.system(f'oarsub -S ./Job_{i}.oar')

if __name__ == "__main__":
    main()
