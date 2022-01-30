#!/usr/bin/env python
#title           :LegendCodeMonomer_run.py
#description     :Call methods for analyzing MD trajectories
#                 All output is formatted consistently
#python_version  :3.7
#creater		 :Suman Samantray
#email			 :suman.rishis@gmail.com
#web			 :suman-samantray.github.io
#==============================================================================
import mdtraj as mdt
import LegendCodeMonomer as lcm

#Define file variables
xtc_file = r"your\path\file.xtc"
pdb_file = r"your\path\initial_state.pdb"
output_file = r"your\path\output_file.txt"

#Read in xtc file
trajectory = mdt.load(xtc_file, top=pdb_file)

#Calculate Secondary Structure
#Calling Secondary Structure method from analysis_lib.py --example
result_matrix = lcm.calc_secondary_structure(trajectory)

#Write data to file
with open(output_file) as output:
    for line in result_matrix:
        output.write('\t'.join(map(str, line)) + '\n')