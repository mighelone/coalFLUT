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
import pyFLUT.ulf.equilibrium as eq

backup_dir = os.path.join(os.path.curdir, 'backup')

def runUlf(ulf_settings, Y, chist, fuel, ox):
    """
    Run ULF solver for a given Y and chist

    Parameters
    ----------
    ulf_settings: {'solver': SOLVER, 'basename': BASENAME}
        solver is the path of the ulf solver
        basename is the basename of the ulf files. A file named BASENAME.ulf should be provided
    Y: float
        Ratio Z1/(Z1+Z2)
    chist:
        stoichiometric scalar dissipation rate
    fuel: {'T': T, 'Y': {sp0:y0, sp1:y1, ...}}
        fuel dictionary
    ox: {'T': T, 'Y': {sp0:y0, sp1:y1, ...}}
        oxidizer dictionary
    Returns
    -------
    ulf_result: pyFLUT.ulf.UlfData
        result object of the ULF simulation
    """
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
    runner.set('TOXIDIZER', ox['T'])
    runner.set('TFUEL', ox['TFUEL'])

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

def convert_mole_to_mass(X, gas):
    """
    Convert
    Parameters
    ----------
    X: {sp0:x0, sp1:x1, ...}
    gas: cantera.Solution

    Returns
    -------
    Y: {sp0:x0, sp1:x1, ...}
    """
    gas.X = [X.get(sp, 0) for sp in gas.species_names]
    return {sp:gas.Y[gas.species_index(sp)] for sp in X.keys()}

def read_dict_list(method, values):
    """
    Read a dictionary list and return the array

    Parameters
    ----------
    method: {'list', 'linspace', 'logspace', 'arange'}
    values: list

    Returns
    -------
    numpy.ndarray
    """
    if method == 'list':
        return np.array(values)
    elif method in ('linspace', 'logspace', 'arange'):
        return getattr(np, method)(*values)

class coalFLUT(ulf.UlfDataSeries):
    def __init__(self, input_yaml):
        with open(input_yaml, "r") as f:
            inp = yaml.load(f)
        self.mechanism = inp['mechanism']
        self.gas = cantera.Solution(self.mechanism)
        self.coal = inp['coal']
        self.volatiles = inp['coal']['volatiles']
        self.oxidizer = inp['oxidizer']
        if 'Y' not in self.oxidizer:
            self.oxidizer['Y'] = convert_mole_to_mass(self.oxidizer['X'], self.gas)
        # define self.Y
        self.Y = read_dict_list(**inp['mixture_fraction']['Y'])
        self.Tf = read_dict_list(**inp['coal']['T'])
        self.ulf_settings = inp['ulf']

        self.chargas = self._define_chargas()


    def mix_fuels(self, Y):
        """
        Mix volatile and char fuels

        Parameters
        ----------
        Y: float
            Y=Z1/(Z1+Z2), where Z1 is the mixture fraction of volatiles and Z2 of char burnoff gas

        Returns
        -------
        mix_fuel: {'T': T, 'Y': {sp0:y0, sp1:y1, ...}}
            mixed fuel dictionary
        """
        pass

    def _define_chargas(self):
        chargas = {}
        mw = self.gas.molecular_weights
        mass = mw[self.gas.species_index('CO')]
        x_o2 = self.oxidizer['X']['O2']
        for sp, x in self.oxidizer['X'].items():
            if not sp == 'O2':
                mass += 0.5*x/x_o2 * mw[self.gas.species_index(sp)]
        for sp, x in self.oxidizer['X'].items():
            index = self.gas.species_index(sp)
            if sp == 'O2':
                chargas['O2'] = 0
            else:
                chargas[sp] = 0.5 * mw[index] * x/x_o2/mass
        chargas['CO'] = (mw[self.gas.species_index('CO')] * (1 + 0.5 * self.oxidizer['X'].get(
                'CO', 0)/x_o2))/mass
        return chargas