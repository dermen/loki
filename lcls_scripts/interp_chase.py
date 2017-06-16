import os
import time
import glob

import argparse
######
# parse parameters
#######


parser = argparse.ArgumentParser(description='Look for interpolated intensities\
 in specifice run directory and submit them for correlation jobs')
parser.add_argument('-r', '--run_dir', type=str, 
    required=True, help='directory in which the interpolated data is saved')
parser.add_argument('-m', '--max_files', type=int, 
    default=100, help='max number of files to submit')
parser.add_argument('-w', '--max_wait', 
    default=100, type=float, help='max time in seconds to wait for new files before quitting')
parser.add_argument('-s','--save_dir',type=str,
    required=True, help='directory in which to save the correlation data')


args=parser.parse_args()


max_files = args.max_files # quit if this many files have been created.
max_wait = args.max_wait # in seconds
run_dir = args.run_dir# dir to check for interpolated intensity files
save_dir = args.save_dir

print("Saving correlation data in %s"%save_dir)
def get_current_filelist(run_dir):
    current_filelist = glob.glob( os.path.join(run_dir,'*.h5') ) 

    while len(current_filelist) == 0:
        time.sleep(1) # wait for 1 second
        current_filelist = glob.glob( os.path.join(run_dir,'*.h5') )

    return current_filelist

# keep track of all the interpolated file submitted for paring and diff cor
master_list = []

current_filelist = get_current_filelist(run_dir)
total_waittime = 0

#if no more new files are created after max_waitime, stop the chaser
while True:
    # loop forever until master_list exceeds length 
    if total_waittime>max_wait or len(master_list)>max_files:
        break
    # check if current files are already in the master_list
    files_to_submit = [f for f in current_filelist if f not in master_list]
    if len(files_to_submit)>0:
        print("submitting the following files for correlating...")
        print files_to_submit
        #############################################################
        # insert code for submitting pair and diff cor jobs here!!!!
        #############################################################

        master_list.extend(files_to_submit)
        # record the job ids, file names, and time of submission

    else:
        print("Nothing more to submit for yet. Let's wait for 10 seconds")
        time.sleep(10) # wait to 10 seconds 
        total_waittime+=10 # add to total wait time for files to be created
        current_filelist = get_current_filelist(run_dir)

        continue

print("Chaser has stopped submitting correlation jobs. EXIT")

