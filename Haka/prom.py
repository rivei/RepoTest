# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 00:12:03 2019
https://github.com/harrywang/pyprom
http://www.promtools.org/doku.php
https://fluxicon.com/disco/
https://graphviz.gitlab.io/_pages/Download/Download_windows.html
https://plot.ly/dash/
https://arxiv.org/abs/1905.06169
http://www.processmining.org/event_logs_and_models_used_in_book
@author: Wei
"""

import os
import shutil
import sys
from algo import alpha


def main(argv):
    log_dir = "logs"  # log folder name
    output_dir = "output"  # output folder name

    # clean the output folder
    # for the_file in os.listdir(output_dir):
    #     file_path = os.path.join(output_dir, the_file)
    #     try:
    #         if os.path.isfile(file_path):
    #             os.remove(file_path)
    #     except Exception as e:
    #         print(e)

    # read the log file
    log = []
    input_file = log_dir + "/" + argv[0]
    output_file = os.path.splitext(output_dir +
                                   "/" + os.path.basename(argv[0]))[0]
    with open(input_file, "r") as f:
            for line in f.readlines():
                line = line.split()
                if line not in log:  # some sequence only counts once
                    log.append(line)

    print(log, input_file, output_file)

    alpha.apply(log, input_file, output_file)


if __name__ == "__main__":
    main(sys.argv)