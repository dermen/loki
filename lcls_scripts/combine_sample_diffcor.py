import os
import tables
from shutil import copyfile
import argparse

parser = argparse.ArgumentParser(description='Combining data from serparat h5 files to one big one')
parser.add_argument('-r', '--sample', type=str, required=True, help='run number to process')
parser.add_argument('-d', '--data_dir', type=str, required = True, help='where the data is')
parser.add_argument('-s', '--save_dir', type=str, required = True, help='where to save the data')
args = parser.parse_args()

sample = args.sample

def combine_files(sample, dirpath,save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    newfname = os.path.join(save_dir, '%s_diffcorr.tbl'%sample)

    if os.path.exists(newfname):
        return None

#   load files and sort numerically
    fnames = [ os.path.join(dirpath, f) for f in os.listdir(dirpath)
        if f.startswith('run') and f.endswith('all_diffcorr.h5') ]

    if not fnames:
        print("No filenames for sample %s"%sample)
        return None
    
    # fnames = sorted(fnames, 
    #     key=lambda x: int(x.split('run')[-1].split('_')[-1].split('.h5')[0] ))
    
    for f in fnames:
        try:
            tables.File(f)
        except tables.exceptions.HDF5ExtError:
            print ("Cannot open file %s"%f)
            return None

    print fnames
#   copy first file as a start
    tbl = tables.File(newfname,'w')
    old_tbl = tables.File(fnames[0],'r')
    a = old_tbl.get_node('/diff_corr').atom
    tbl.create_earray(tbl.root, 'diff_corr', a,(0,35,354))
    print tbl.get_node('/diff_corr')
    # copyfile( fnames[0], newfname )
    # print "here1 \n"


    # print "here2 \n"
#   this is the master file we will append to
    with tables.File(newfname, 'r+') as tbl:
        # print "here3 \n"
        for fname in fnames[1:]:
            print "here loop \n"
            next_tbl = tables.File( fname, 'r')
            
            tbl.get_node('/diff_corr').append( next_tbl.get_node('/diff_corr')[:] )

# combines files
combine_files(sample, args.data_dir, args.save_dir)


