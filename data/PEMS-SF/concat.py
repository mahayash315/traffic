#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from os import listdir, chdir
from os.path import isdir, isfile, join
from string import atoi

input_folder="days"
output_file="PEMS_sorted"

line = None
with open("randperm") as fin:
    line = fin.readline()

list = line.strip()[1:-1].split(' ')

if not isdir(input_folder):
    raise "{} is not a directory".format(input_folder)

# list files
input_files = [ f for f in listdir(input_folder) if isfile(join(input_folder,f)) ]

with open(output_file, 'w') as fout:
    chdir(input_folder)

    for file in input_files:
        print("input: {}/{}".format(input_folder, file))
        with open(file) as fin:
            line = fin.readline()
            while line:
                fout.write(line)
                line = fin.readline()