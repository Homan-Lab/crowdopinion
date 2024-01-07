# This script will run any prepping needed for experiments such as removing temp folders
import os
import matplotlib as mpl
from numpy.lib.shape_base import expand_dims
if os.environ.get('DISPLAY','') == '':
 	mpl.use('Agg')
import shutil
import argparse
import sys

def remove_folder(foldername):
    try:
        shutil.rmtree(foldername)
        print("Removed folder "+foldername)
    except:
        pass
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", help="Folder to delete",default=None)
    args = parser.parse_args()
    foldername = args.folder_name
    remove_folder(foldername)

if __name__ == '__main__':
	main()
