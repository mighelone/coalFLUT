"""
coalFLUT
========
"""

import pyFLUT.ulf as ulf
import yaml
import cantera
import numpy as np
import shutil
import os
import glob

backup_dir = os.path.join(os.path.curdir, 'backup')

def runUlf(ulf_settings, Y, chist, fuel, ox):
    ulf_basename = ulf_settings['basename'] + "_Y{:4.3f}_chist{:1.1f}".format(Y, chist)
    ulf_result = ulf_basename + ".ulf"
    ulf_basename_run = ulf_basename+"run"
    ulf_input = ulf_basename_run + ".ulf"
    shutil.copy(ulf_settings['basename'] + ".ulf", ulf_input)
    runner = ulf.UlfRun(ulf_input, ulf_settings["solver"])
    runner.set("BASENAME", ulf_basename_run)

    list_of_species = list(set(ox['Y'].keys() + fuel['Y'].keys()))
    for i, sp in enumerate(list_of_species):
        runner.set('SPECIES{}'.format(i), sp)
        runner.set('FUELVAL{}'.format(i), fuel['Y'].get(sp, 0))
        runner.set('OXIVAL{}'.format(i), ox['Y'].get(sp, 0))

    runner.set('CHIST', chist)

    try:
        print("Run {}".format(ulf_basename))
        runner.run()
        shutil.copy(ulf_basename_run+'final.ulf', ulf_result)
        print("End run {}".format(ulf_basename))
        if not os.path.exists(backup_dir):
            os.mkdir(backup_dir)
        for f in glob.glob(ulf_basename_run+ "*"):
            shutil.move(f, backup_dir)
        return ulf.read_ulf(ulf_result)
    except:
        print("Error running {}".format(ulf_basename))
        return None







class coalFLUT(ulf.UlfDataSeries):
    def __init__(self, input_yaml):
        with open(input_yaml, "r") as f:
            inp = yaml.load(f)
        self.coal = inp['coal']
        self.oxidizer = inp['oxidizer']
        # define self.Y
        if inp['mixture_fraction']['Y']['method'] == 'list':
            self.Y = np.array(inp['mixture_fraction']['Y']['values'])
        else:
            self.Y = getattr(np, inp['mixture_fraction']['Y']['method'])\
                (*inp['mixture_fraction']['Y']['values'])
        self.ulf_settings = inp['ulf']