from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import pyFLUT.ulf.dflut
import numpy as np

from pyFLUT.ulf.dflut import run_sldf
from scoop import futures
import functools
from autologging import logged
# New implementation #


@logged
class CoalFLUT(pyFLUT.ulf.dflut.DFLUT_2stream):
    streams = ['volatiles', 'oxidizer']

    def __init__(self, input_yml):
        super(CoalFLUT, self).__init__(input_yml=input_yml)
        self.set_chargas()

    def mix_streams(self, Y):
        '''
        Mix volatiles and char gas for the given Y value.
        The following mixing rules are used (see Doc.):

        YFi = ((1+alphac) (1-Y) * YPi + Y * Yvi ) /
                ((1+alphac) (1-Y) + Y)

        Parameters
        ----------
        Y: float
           Volatiles fraction (note is defined as mv/(mv+mc)
           with mp = mv * (1+alphac) or Z* = Z[Y + (1-Y) (1+alphac)]

        Return
        ------
        mix_fuel: dict
           Dictionary with the mix fuel
        '''
        alphac_one = 1 + self.alphac

        def mix(p, v):
            return ((alphac_one * (1 - Y) * p + Y * v) /
                    (alphac_one * (1 - Y) + Y))

        yp, Hp = self.chargases['Y'], self.chargases['H']
        yv, Hv = self.volatiles['Y'], self.volatiles['H']

        y = {sp: mix(yp.get(sp, 0), yv.get(sp, 0))
             for sp in set(list(yp.keys()) + list(yv.keys()))}
        H = mix(Hp, Hv)
        mix_fuel = {}
        mix_fuel['Y'] = y
        mix_fuel['H'] = H
        mix_fuel['T'] = np.array(
            [pyFLUT.utilities.calc_Tf(
                self.gas, H_i, self.pressure, y)
             for H_i in H])
        mix_fuel['z_st'] = pyFLUT.utilities.calc_zstoich(
            fuel=mix_fuel, oxidizer=self.oxidizer, gas=self.gas)
        return mix_fuel

    def set_chargas(self):
        """
        Define the char gas composition
        """
        def mw(sp):
            return self.gas.molecular_weights[self.gas.species_index(sp)]

        Yc = {}
        mc = self.gas.atomic_weight('C')
        mass = mw('CO')
        alphac = mw('O2')
        X_ox = self.oxidizer['X']
        x_o2 = X_ox['O2']
        for sp, x in X_ox.items():
            if not sp == 'O2':
                prod = x / x_o2 * mw(sp)
                mass += 0.5 * prod
                alphac += prod
        alphac *= 0.5 / mc
        self.alphac = alphac
        for sp, x in X_ox.items():
            if sp == 'O2':
                Yc['O2'] = 0
            else:
                Yc[sp] = 0.5 * mw(sp) * x / x_o2 / mass

        Yc['CO'] = (
            mw('CO') * (1. + 0.5 * X_ox.get('CO', 0) / x_o2)) / mass
        chargases = {}
        chargases['Y'] = Yc
        chargases['T'] = self.volatiles['T'].copy()
        self.chargases = pyFLUT.utilities.fix_composition_T(
            chargases, self.gas)

    def run_scoop(self):
        """
        Run in parallel using scoop
        """
        def fuel_gen(Y, Hnorm, chist):
            for Yi in Y:
                for H in Hnorm:
                    for chi in chist:
                        mix = self.mix_streams(Yi)
                        self.__log.debug(
                            'Y=%s H_mix=%s', Y, mix['H'])
                        fuel = mix.copy()
                        H_fuel = ((mix['H'][1] - mix['H'][0]) *
                                  H + mix['H'][0])
                        self.__log.debug('Y=%s Hnorm=%s', Y, Hnorm)
                        fuel['T'] = pyFLUT.utilities.calc_Tf(
                            self.gas, H_fuel, self.pressure, mix['Y'])
                        self.__log.debug('Tf=%s', fuel['T'])
                        yield fuel

        def oxid_gen(Y, Hnorm, chist):
            for Yi in Y:
                for H in Hnorm:
                    for chi in chist:
                        oxid = self.oxidizer.copy()
                        H_oxid = (oxid['H'][1] - oxid['H'][0]) * \
                            H + oxid['H'][0]
                        oxid['T'] = pyFLUT.utilities.calc_Tf(
                            self.gas, H_oxid, self.pressure, oxid['Y'])
                        self.__log.debug('Toxidizer=%s', oxid['T'])
                        yield oxid

        def parameters_gen(Y, Hnorm, chist):
            for Yi in Y:
                for H in Hnorm:
                    for chi in chist:
                        yield {'Hnorm': H, 'Y': Yi, 'chist': chi}

        fuel = fuel_gen(self.Y, self.Hnorm, self.chist)
        self.__log.debug('Create fuel generator')
        oxid = oxid_gen(self.Y, self.Hnorm, self.chist)
        self.__log.debug('Create oxid generator')
        parameters = parameters_gen(self.Y, self.Hnorm, self.chist)
        self.__log.debug('Create parameters generator')
        results = list(
            futures.map(functools.partial(
                run_sldf,
                par_format=self.format,
                ulf_reference=self.ulf_reference,
                solver=self.solver,
                species=self.gas.species_names,
                key_names=self.keys,
                basename=self.basename,
                rerun=self.rerun),
                fuel,
                oxid,
                parameters)
        )

        self.assemble_data(results)
