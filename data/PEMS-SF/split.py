#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from string import atoi

input_file="PEMS_all"
output_folder="days"

line = None
with open("randperm") as fin:
    line = fin.readline()

list = line.strip()[1:-1].split(' ')

if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

with open(input_file) as fin:
    os.chdir(output_folder)

    i=0
    line = fin.readline()
    while line:
        day = atoi(list[i])
        file = "{0:03d}".format(day)
        print("output: {}/{}".format(output_folder, file))
        with open(file, 'w') as fout:
            fout.write(line)
        i=i+1
        line = fin.readline()