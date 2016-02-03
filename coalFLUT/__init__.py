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
import pyFLUT.ulf.equilibrium as equilibrium

#TODO calculate Tf from the normalized enthalpy. The actual value of enthalpy and the normalized
# should be passed to the run function

#TODO set the new composition of the fuel (vol+char) considering the oxygen consumed

backup_dir = os.path.join(os.path.curdir, 'backup')

def runUlf(ulf_settings, Y, chist, Hnorm, fuel, ox):
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
    # ulf_basename = ulf_settings['basename'] + "_Tf{:4.1f}_Y{:4.3f}_chist{:4.3f}".format(fuel[
    #                                                                                         'T'],Y,
    #                                                                                    chist)
    ulf_basename = ulf_settings['basename'] + "_Hnorm{:4.3f}_Y{:4.3f}_chist{:4.3f}".format(Hnorm, Y, chist)
    ulf_result = ulf_basename + ".ulf"
    ulf_basename_run = ulf_basename+"run"
    ulf_input = ulf_basename_run + ".ulf"
    shutil.copy(ulf_settings['basename'] + ".ulf", ulf_input)
    runner = ulf.UlfRun(ulf_input, ulf_settings["solver"])
    runner.set("BASENAME", ulf_basename_run)
    pressure = float(runner['PRESSURE'])

    fuel['T'] = 300
    ox['T'] = 300
    # dumb fuel and oxidizer temperature equilibrium necessary only for the stoichiomtric
    # conditions
    eq = equilibrium.EquilibriumSolution(fuel=fuel, oxidizer=ox, mechanism=runner['MECHANISM'])
    runner.set('ZST', eq.z_stoich())


    list_of_species = list(set(ox['Y'].keys() + fuel['Y'].keys()))
    for i, sp in enumerate(list_of_species):
        runner.set('SPECIES{}'.format(i), sp)
        runner.set('FUELVAL{}'.format(i), fuel['Y'].get(sp, 0))
        runner.set('OXIVAL{}'.format(i), ox['Y'].get(sp, 0))

    runner.set('CHIST', chist)

    #Hf = fuel['H'].min() + Hnorm * (fuel['H'].max() - fuel['H'].min())
    runner.set('TFUEL', calc_tf(eq.gas, fuel['H'].min() + Hnorm * (fuel['H'].max() - fuel['H'].min()),
                                pressure, fuel['Y']))
    Ho = ox['H'].min() + Hnorm * (ox['H'].max() - ox['H'].min())
    runner.set('TOXIDIZER', calc_tf(eq.gas, Ho, pressure, ox['Y']))
    print("Run {}".format(ulf_basename))
    runner.run()
    shutil.copy(ulf_basename_run+'final.ulf', ulf_result)
    print("End run {}".format(ulf_basename))
    #try:
    #    print("Run {}".format(ulf_basename))
    #    runner.run()
    #    shutil.copy(ulf_basename_run+'final.ulf', ulf_result)
    #    print("End run {}".format(ulf_basename))
    #except:
    #    print("Error running {}".format(ulf_basename))
    #    return None
    if not os.path.exists(backup_dir):
        os.mkdir(backup_dir)
    for f in glob.glob(ulf_basename_run+ "*"):
        shutil.move(f, os.path.join(backup_dir, f))
    return ulf.read_ulf(ulf_result)

def calc_tf(gas, H, pressure, Y):
    """
    Calculate the fuel temperature for a given total enthalpy

    Parameters
    ----------
    fuel
    gas
    pressure

    Returns
    -------

    """
    gas.HPY = H, pressure, Y
    return gas.T

def calc_hf(gas, T, pressure, Y):
    """
    Calculate the fuel total enthalpy for a given temperature

    Parameters
    ----------
    fuel
    gas
    pressure

    Returns
    -------

    """
    gas.TPY = T, pressure, species_string(Y)
    return gas.enthalpy_mass

def species_string(X_dict):
    return ''.join('{}:{},'.format(sp, value) for sp, value in X_dict.iteritems())[:-1]

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
        # TODO volatiles can be defined as C, H, O
        self.volatiles = {}
        self.volatiles['Y'] = normalize(inp['volatiles']['Y'])
        self.volatiles['T'] = read_dict_list(**inp['volatiles']['T'])

        self.oxidizer = {}
        #self.oxidizer[] = inp['oxidizer']
        if 'Y' in inp['oxidizer']:
            self.oxidizer['Y'] = normalize(inp['oxidizer']['Y'])
            self.oxidizer['X'] = convert_mass_to_mole(self.oxidizer['Y'], self.gas)
        else:
            self.oxidizer['X'] = normalize(inp['oxidizer']['X'])
            self.oxidizer['Y'] = convert_mole_to_mass(self.oxidizer['X'], self.gas)
        self.oxidizer['T'] = read_dict_list(**inp['oxidizer']['T'])
        # define self.Y
        self.chist = read_dict_list(**inp['mixture_fraction']['chist'])
        self.Y = read_dict_list(**inp['mixture_fraction']['Y'])

        n_H = len(self.volatiles['T'])
        # check if volatile and oxidizer temperature levels are the same
        # if not use volatiles as refence number
        if len(self.oxidizer['T']) != n_H:
            self.oxidizer['T'] = np.linspace(self.oxidizer['T'].min(), self.oxidizer['T'].max(),
                                             n_H)

        self.ulf_settings = inp['ulf']
        runner = ulf.UlfRun(self.ulf_settings['basename']+".ulf", self.ulf_settings['solver'])
        pressure = float(runner['PRESSURE'])
        self.chargas = self._define_chargas()
        self.chargas['T'] = self.volatiles['T']

        # define H normalized between 0 and 1
        # Hnorm = 0 corresponds to Tf min
        # Hnorm = 1 corresponds to Tf max
        #self.Hnorm = (self.Tf - self.Tf.min())/(self.Tf.max()-self.Tf.min())
        self.Hnorm = np.linspace(0, 1, n_H)

        H = [np.linspace(calc_hf(self.gas, fuel['T'].min(), pressure, fuel['Y']),
                         calc_hf(self.gas, fuel['T'].max(), pressure, fuel['Y']),
                         n_H)
             for fuel in [self.volatiles, self.chargas, self.oxidizer]]
        self.volatiles['H'] = H[0]
        self.chargas['H'] = H[1]
        self.oxidizer['H'] = H[2]

        self.z_points = inp['mixture_fraction']['Z']['points']

    def mix_fuels(self, Y):
        """
        Mix volatile and char fuels.

        Notes
        -----
        The function is updated for considering the oxygen consumption in the mixing of the species.
        See Watanabe et al. PCI 2014

        Parameters
        ----------
        Y: float
            Y=Z1/(Z1+Z2), where Z1 is the mixture fraction of volatiles and Z2 of char burnoff gas

        Returns
        -------
        mix_fuel: {'T': T, 'Y': {sp0:y0, sp1:y1, ...}, 'H':[Hmin, Hmax]}
            mixed fuel dictionary
        """
        onealphac = 1 + self.chargas['alphac']
        yf = {sp: (Y*self.volatiles['Y'].get(sp, 0) + (1-Y) * onealphac *
                   self.chargas['Y'].get(sp, 0)) / (Y + (1-Y) * onealphac)
                for sp in list(set(self.chargas['Y'].keys() + self.volatiles['Y'].keys()))}
        Hf = (Y*self.volatiles['H'] + (1-Y) * onealphac * self.chargas['H']) / \
             (Y + (1-Y) * onealphac)
        return {'Y': yf, 'H': Hf}

    def _define_chargas(self):
        Yc = {}
        mw = self.gas.molecular_weights
        mc = self.gas.atomic_weight('C')
        mass = mw[self.gas.species_index('CO')]
        alphac = mw[self.gas.species_index('O2')]
        x_o2 = self.oxidizer['X']['O2']
        for sp, x in self.oxidizer['X'].items():
            if not sp == 'O2':
                prod = x/x_o2 * mw[self.gas.species_index(sp)]
                mass += 0.5 * prod
                alphac += prod
        alphac *= 0.5/ mc
        for sp, x in self.oxidizer['X'].items():
            index = self.gas.species_index(sp)
            if sp == 'O2':
                Yc['O2'] = 0
            else:
                Yc[sp] = 0.5 * mw[index] * x/x_o2/mass
        Yc['CO'] = (mw[self.gas.species_index('CO')] * (1 + 0.5 * self.oxidizer['X'].get(
                'CO', 0)/x_o2))/mass
        return {'Y': Yc, 'alphac': alphac}

    def run(self, n_p=1):
        # define here common settings to all cases
        runner = ulf.UlfRun(self.ulf_settings['basename']+".ulf", self.ulf_settings['solver'])
        runner.set('MECHANISM', self.mechanism)
        runner.set('AXISLENGHTREFINED', self.z_points)

        if n_p > 1:
            p = mp.Pool(processes=n_p)
            procs = [p.apply_async(runUlf,
                               args=(self.ulf_settings, Y, chist, Hnorm,
                                     self.mix_fuels(Y),
                                     self.oxidizer))
                 for Hnorm in self.Hnorm for Y in self.Y for chist in self.chist]
            results = [pi.get() for pi in procs]
        else:
            results = [runUlf(self.ulf_settings, Y, chist, Hnorm,
                                     self.mix_fuels(Y),
                                     self.oxidizer)
                       for Hnorm in self.Hnorm
                       for Y in self.Y for chist in self.chist]
        super(coalFLUT, self).__init__(input_data=results, key_variable='Z')