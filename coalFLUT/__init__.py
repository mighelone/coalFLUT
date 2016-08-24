"""
coalFLUT
========
"""
import pyFLUT.ulf.dflut
import numpy as np


# New implementation #

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
             for sp in set(yp.keys() + yv.keys())}
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
        for sp, x in X_ox.iteritems():
            if not sp == 'O2':
                prod = x / x_o2 * mw(sp)
                mass += 0.5 * prod
                alphac += prod
        alphac *= 0.5 / mc
        self.alphac = alphac
        for sp, x in X_ox.iteritems():
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
