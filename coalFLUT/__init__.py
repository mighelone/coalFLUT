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
import multiprocessing as mp

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
    ulf_basename = ulf_settings['basename'] + "_Tf{:4.1f}_Y{:4.3f}_chist{:1.1f}".format(fuel[
                                                                                             'T'],Y,
                                                                                        chist)
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
    runner.set('TFUEL', fuel['T'])

    try:
        print("Run {}".format(ulf_basename))
        runner.run()
        shutil.copy(ulf_basename_run+'final.ulf', ulf_result)
        print("End run {}".format(ulf_basename))
    except:
        print("Error running {}".format(ulf_basename))
        return None
    if not os.path.exists(backup_dir):
        os.mkdir(backup_dir)
    for f in glob.glob(ulf_basename_run+ "*"):
        shutil.move(f, os.path.join(backup_dir, f))
    return ulf.read_ulf(ulf_result)


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

def convert_mass_to_mole(Y, gas):
    """
    Convert
    Parameters
    ----------
    Y: {sp0:x0, sp1:x1, ...}
    gas: cantera.Solution

    Returns
    -------
    X: {sp0:x0, sp1:x1, ...}
    """
    gas.Y = [Y.get(sp, 0) for sp in gas.species_names]
    return {sp:gas.X[gas.species_index(sp)] for sp in Y.keys()}

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

def normalize(x):
    sumx = sum(x.values())
    return {sp:xi/sumx for sp, xi in x.items()}

class coalFLUT(ulf.UlfDataSeries):
    def __init__(self, input_yaml):
        with open(input_yaml, "r") as f:
            inp = yaml.load(f)
        self.mechanism = inp['mechanism']
        self.gas = cantera.Solution(self.mechanism)
        self.coal = inp['coal']
        # TODO volatiles can be defined as C, H, O
        self.volatiles = inp['coal']['volatiles']
        self.volatiles['Y'] = normalize(self.volatiles['Y'])
        self.oxidizer = inp['oxidizer']
        if 'Y' in self.oxidizer:
            self.oxidizer['Y'] = normalize(self.oxidizer['Y'])
            self.oxidizer['X'] = convert_mass_to_mole(self.oxidizer['Y'], self.gas)
        else:
            self.oxidizer['X'] = normalize(self.oxidizer['X'])
            self.oxidizer['Y'] = convert_mole_to_mass(self.oxidizer['X'], self.gas)
        # define self.Y
        self.chist = read_dict_list(**inp['mixture_fraction']['chist'])
        self.Y = read_dict_list(**inp['mixture_fraction']['Y'])
        self.Tf = read_dict_list(**inp['coal']['T'])
        self.ulf_settings = inp['ulf']
        self.z_points = inp['mixture_fraction']['Z']['points']
        self.chargas = {'Y': self._define_chargas()}


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
        return {sp: (Y*self.volatiles['Y'].get(sp, 0) + (1-Y) * self.chargas['Y'].get(sp, 0))
                for sp in list(set(self.chargas['Y'].keys() + self.volatiles['Y'].keys()))}

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

    def run(self, n_p=1):
        # define here common settings to all cases
        runner = ulf.UlfRun(self.ulf_settings['basename']+".ulf", self.ulf_settings['solver'])
        runner.set('MECHANISM', self.mechanism)
        runner.set('AXISLENGHTREFINED', self.z_points)

        if n_p > 1:
            p = mp.Pool(processes=n_p)
            procs = [p.apply_async(runUlf,
                               args=(self.ulf_settings, Y, chist, {'T': Tf, 'Y': self.mix_fuels(
                                       Y)},
                                     self.oxidizer))
                 for Tf in self.Tf for Y in self.Y for chist in self.chist]
            results = [pi.get() for pi in procs]
        else:
            results = [runUlf(self.ulf_settings, Y, chist, {'T': Tf, 'Y': self.mix_fuels(Y)},
                              self.oxidizer)
                 for Tf in self.Tf for Y in self.Y for chist in self.chist]
        super(coalFLUT, self).__init__(input_data=results, key_variable='Z')