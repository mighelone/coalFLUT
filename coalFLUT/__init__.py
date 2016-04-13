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
from termcolor import colored

T_limit = 200

backup_dir = os.path.join(os.path.curdir, 'backup')


def runUlf(ulf_settings, Y, chist, Hnorm, fuel, ox, z_DHmin):
    """
    Run ULF solver for a given Y and chist

    Parameters
    ----------
    ulf_settings: {'solver': SOLVER, 'basename': BASENAME}
        solver is the path of the ulf solver
        basename is the basename of the ulf files.
        A file named BASENAME.ulf should be provided
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
    ulf_basename = ulf_settings['basename'] + \
        "_Hnorm{:5.4f}_Y{:5.4f}_chist{:5.4f}".format(Hnorm, Y, chist)
    ulf_result = ulf_basename + ".ulf"
    ulf_basename_run = ulf_basename+"run"
    ulf_input = ulf_basename_run + ".ulf"
    shutil.copy(ulf_settings['basename'] + ".ulf", ulf_input)
    runner = ulf.UlfRun(ulf_input, ulf_settings["solver"])
    runner.set("BASENAME", ulf_basename_run)
    pressure = float(runner['PRESSURE'])

    fuel['T'] = 300
    ox['T'] = 300
    # dumb fuel and oxidizer temperature equilibrium necessary only
    # for the stoichiomtric conditions
    eq = equilibrium.EquilibriumSolution(fuel=fuel, oxidizer=ox,
                                         mechanism=runner['MECHANISM'])
    runner.set('ZST', eq.z_stoich())

    list_of_species = list(set(ox['Y'].keys() + fuel['Y'].keys()))
    for i, sp in enumerate(list_of_species):
        runner.set('SPECIES{}'.format(i), sp)
        runner.set('FUELVAL{}'.format(i), fuel['Y'].get(sp, 0))
        runner.set('OXIVAL{}'.format(i), ox['Y'].get(sp, 0))

    runner.set('CHIST', chist)
    runner.set('ZMAX', z_DHmin)
    if Hnorm > 0:
        DH_max = 0
    else:
        H_1 = z_DHmin * fuel['H'].max() + (1-z_DHmin) * ox['H'].max()
        H_0 = z_DHmin * fuel['H'].min() + (1-z_DHmin) * ox['H'].min()
        DH_max = -Hnorm*(H_1 - H_0)
    runner.set('DHMAX', DH_max)

    # Hf = fuel['H'].min() + Hnorm * (fuel['H'].max() - fuel['H'].min())
    Hnorm_t = Hnorm if Hnorm > 0 else 0
    runner.set('TFUEL', calc_tf(eq.gas,
                                fuel['H'].min() + Hnorm_t *
                                (fuel['H'].max() - fuel['H'].min()),
                                pressure, fuel['Y']))
    Ho = ox['H'].min() + Hnorm_t * (ox['H'].max() - ox['H'].min())
    runner.set('TOXIDIZER', calc_tf(eq.gas, Ho, pressure, ox['Y']))
    try:
        print("Run {}".format(ulf_basename))
        results = runner.run()
        shutil.copy(ulf_basename_run+'final.ulf', ulf_result)
        print("End run {}".format(ulf_basename))
    except:
        print(colored("Error running {}".format(ulf_basename), 'red'))
        results = None
    if not os.path.exists(backup_dir):
        os.mkdir(backup_dir)
    for f in glob.glob(ulf_basename_run + "*"):
        shutil.move(f, os.path.join(backup_dir, f))
    return results


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
    return ''.join('{}:{},'.format(sp, value)
                   for sp, value in X_dict.iteritems())[:-1]


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
    return {sp: gas.Y[gas.species_index(sp)] for sp in X.keys()}


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
    return {sp: gas.X[gas.species_index(sp)] for sp in Y.keys()}


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
    return {sp: xi/sumx for sp, xi in x.items()}

def fix_composition(stream, gas):
    """
    Given a stream dictionary, normalize the composition and return a new
    string dictionary with normalized values, mole and mass fractions
    Parameters
    ----------
    stream: dict
        {'Y':{'CH4':1}} or {'X':{'CH4':1}}
    gas: cantera.Solution
        cantera solution object

    Returns
    -------
    dict
    """
    if "Y" in stream:
        stream["Y"] = normalize(stream["Y"])
        stream["X"] = convert_mass_to_mole(stream["Y"], gas)
    else:
        stream["X"] = normalize(stream["X"])
        stream["Y"] = convert_mole_to_mass(stream["X"], gas)
    return stream



class coalFLUT(ulf.UlfDataSeries):
    def __init__(self, input_yaml):
        with open(input_yaml, "r") as f:
            inp = yaml.load(f)
        self.mechanism = inp['mechanism']
        self.gas = cantera.Solution(self.mechanism)
        # TODO volatiles can be defined as C, H, O
        self.volatiles = {}
        self.volatiles['Y'] = normalize(inp['volatiles']['Y'])
        # self.volatiles['T'] = read_dict_list(**inp['volatiles']['T'])
        self.volatiles['T'] = np.array([inp['volatiles']['T']['min'],
                                        inp['volatiles']['T']['max']])


        self.oxidizer = {}
        if 'Y' in inp['oxidizer']:
            self.oxidizer['Y'] = normalize(inp['oxidizer']['Y'])
            self.oxidizer['X'] = convert_mass_to_mole(self.oxidizer['Y'],
                                                      self.gas)
        else:
            self.oxidizer['X'] = normalize(inp['oxidizer']['X'])
            self.oxidizer['Y'] = convert_mole_to_mass(self.oxidizer['X'],
                                                      self.gas)
        # self.oxidizer['T'] = read_dict_list(**inp['oxidizer']['T'])
        self.oxidizer['T'] = np.array([inp['oxidizer']['T']['min'],
                                        inp['oxidizer']['T']['max']])
        # define self.Y
        self.chist = read_dict_list(**inp['flut']['chist'])
        self.Y = read_dict_list(**inp['flut']['Y'])

        # n_H = len(self.volatiles['T'])
        n_H = inp['flut']['Hnorm']['positive']['points']
        # check if volatile and oxidizer temperature levels are the same
        # if not use volatiles as refence number
        # if len(self.oxidizer['T']) != n_H:
        #     self.oxidizer['T'] = np.linspace(self.oxidizer['T'].min(),
        #                                     self.oxidizer['T'].max(), n_H)
        self.ulf_settings = inp['ulf']
        runner = ulf.UlfRun(self.ulf_settings['basename']+".ulf",
                            self.ulf_settings['solver'])
        pressure = float(runner['PRESSURE'])
        self.chargas = self._define_chargas()
        self.chargas['T'] = self.volatiles['T']

        # define H normalized between 0 and 1
        # Hnorm = 0 corresponds to Tf min
        # Hnorm = 1 corresponds to Tf max
        Hnorm = np.linspace(0, 1, n_H)
        Hnorm_negative = read_dict_list(**inp['flut']['Hnorm']['negative'])
        self.z_DHmin = inp['flut']['Hnorm']['Z']
        self.Hnorm = np.concatenate([Hnorm_negative, Hnorm])

        H = [np.array([calc_hf(self.gas, fuel['T'].min(), pressure, fuel['Y']),
                         calc_hf(self.gas, fuel['T'].max(), pressure, fuel['Y'])])
             for fuel in [self.volatiles, self.chargas, self.oxidizer]]
        self.volatiles['H'], self.chargas['H'], self.oxidizer['H'] = \
            (H[i] for i in range(3))

        self.z_points = inp['flut']['Z']['points']

    def mix_fuels(self, Y):
        """
        Mix volatile and char fuels.

        Notes
        -----
        The function is updated for considering the oxygen consumption in the
        mixing of the species.
        See Watanabe et al. PCI 2014

        Parameters
        ----------
        Y: float
            Y=Z1/(Z1+Z2), where Z1 is the mixture fraction of volatiles and Z2
            of char burnoff gas

        Returns
        -------
        mix_fuel: {'T': T, 'Y': {sp0:y0, sp1:y1, ...}, 'H':[Hmin, Hmax]}
            mixed fuel dictionary
        """
        onealphac = 1 + self.chargas['alphac']
        den = Y + (1-Y) * onealphac
        alpha_v, alpha_c = Y / den, (1-Y)*onealphac
        yf = {sp: (alpha_v*self.volatiles['Y'].get(sp, 0) + alpha_c *
                   self.chargas['Y'].get(sp, 0)) for sp in
              list(set(self.chargas['Y'].keys() + self.volatiles['Y'].keys()))}
        Hf = alpha_v * self.volatiles['H'] + alpha_c * self.chargas['H']
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
        alphac *= 0.5 / mc
        for sp, x in self.oxidizer['X'].items():
            index = self.gas.species_index(sp)
            if sp == 'O2':
                Yc['O2'] = 0
            else:
                Yc[sp] = 0.5 * mw[index] * x/x_o2/mass
        Yc['CO'] = (mw[self.gas.species_index('CO')] *
                    (1 + 0.5 * self.oxidizer['X'].get('CO', 0)/x_o2))/mass
        return {'Y': Yc, 'alphac': alphac}

    def run(self, n_p=1):
        # define here common settings to all cases
        runner = ulf.UlfRun(self.ulf_settings['basename']+".ulf",
                            self.ulf_settings['solver'])
        runner.set('MECHANISM', self.mechanism)
        runner.set('AXISLENGHTREFINED', self.z_points)

        if n_p > 1:
            p = mp.Pool(processes=n_p)
            procs = [p.apply_async(runUlf,
                                   args=(self.ulf_settings, Y, chist, Hnorm,
                                         self.mix_fuels(Y),
                                         self.oxidizer, self.z_DHmin))
                     for Hnorm in self.Hnorm for Y in self.Y
                     for chist in self.chist]
            results = [pi.get() for pi in procs]
        else:
            results = [runUlf(self.ulf_settings, Y, chist, Hnorm,
                              self.mix_fuels(Y),
                              self.oxidizer, self.z_DHmin)
                       for Hnorm in self.Hnorm
                       for Y in self.Y for chist in self.chist]
        # super(coalFLUT, self).__init__(input_data=results, key_variable='Z')
        return results

    def extend_enthalpy_range(self, h_levels):
        '''
        Extend the enthalpy range

        Parameters
        ----------
        h_levels: array of levels (they must be negative)

        '''
        from pyFLUT.ulf.ulfSeriesReader import ulf_to_cantera

        def add_levels(stream, h_levels):
            return np.insert(stream['H'], 0, (stream['H'].min() +
                                              h_levels*(stream['H'].max() -
                                                        stream['H'].min())))
        if isinstance(h_levels, float):
            h_levels = np.array([h_levels])
        elif isinstance(h_levels, list):
            h_levels = np.array(h_levels)

        self.volatiles['H'], self.chargas['H'], self.oxidizer['H'] =  \
            (add_levels(stream, h_levels)
             for stream in [self.volatiles, self.chargas, self.oxidizer])

        data_hmin = np.take(self.data, 0,
                            axis=self.input_variable_index('Hnorm'))

        i_species = [self.output_variable_index(sp)
                     for sp in self.gas.species_names]
        # i_T = self.output_variable_index('T')
        i_Z = self.output_variable_index('Z')
        i_Y = self.output_variable_index('Y')
        # i_hMean = self.output_variable_index('hMean')
        # i_Hnorm_o = self.output_variable_index('Hnorm')

        shape = list(self.data.shape)
        shape[self.input_variable_index('Hnorm')] = len(h_levels)
        data_new = np.empty(shape)

        n_l = len(h_levels)
        pressure = self['p'].ravel()[0]

        i_Hnorm = self.input_variable_index('Hnorm')

        for index in np.ndindex(data_hmin.shape[:-1]):
            datai = data_hmin[index]
            y = np.take(datai, i_species)
            Z = np.take(datai, i_Z)
            Y = np.take(datai, i_Y)
            oneY_onealphac = (1-Y)*(1+self.chargas['alphac'])
            Hf = (Y * self.volatiles['H'][:n_l] + oneY_onealphac *
                  self.chargas['H'][:n_l])/(Y + oneY_onealphac)
            H = Z * Hf + (1-Z)*self.oxidizer['H'][:n_l]

            for i_H, Hi in enumerate(H):
                self.gas.HPY = Hi, pressure, y
                T = self.gas.T
                index_new = list(index)
                index_new.insert(i_Hnorm, i_H)
                index_new = tuple(index_new)

                for var, i in self.output_dict.iteritems():
                    if var == 'hMean':
                        value = H[i_H]
                    elif var in ulf_to_cantera.keys():

                        value = getattr(self.gas, ulf_to_cantera.get(var, var))\
                            if T > T_limit else 0
                    elif 'reactionRate_' in var:
                        if T > T_limit:
                            sp_index = self.gas.species_index(var.split('_')[1])
                            value = self.gas.net_production_rates[sp_index] / \
                                self.gas.density
                        else:
                            value = 0
                    elif var == 'alpha':
                        value = self.gas.thermal_conductivity / \
                            self.gas.density / self.gas.cp \
                            if T > T_limit else 0
                    elif var == 'Hnorm':
                        value = h_levels[i_H]
                    else:
                        value = data_hmin[index][i]
                    data_new[index_new][i] = value

        self.data = np.concatenate((data_new, self.data), axis=i_Hnorm)
        self.input_dict['Hnorm'] = np.concatenate((h_levels,
                                                   self.input_dict['Hnorm']))
