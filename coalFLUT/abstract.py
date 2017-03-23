from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import pyFLUT
import pyFLUT.utilities
import pyFLUT.ulf.abstract

from autologging import logged

@logged
class AbstractCoalFLUT(pyFLUT.ulf.abstract.AbstractFlut2Stream):
    """
    Abstract class for coal flut
    """
    _streams = ('volatiles', 'chargases', 'oxidizer')

    def _read_streams(self, settings):
        """
        Set the streams

        :param settings:
        :return:
        """
        fix_composition = pyFLUT.utilities.fix_composition_T

        for stream in ('volatiles', 'oxidizer'):
            setattr(self, stream, fix_composition(settings[stream],
                                                  self.gas))
            self.__log.debug('Set {}:{}', stream, getattr(self, stream))

        # create chargas
        self.set_chargas()
        # read pressure
        self.pressure = settings['pressure']
        self.__log.debug('Pressure %s', self.pressure)


    def mix_fuels(self, Y, Hnorm, gas):
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
        def mix(p, v):
            return ((alphac_one * (1 - Y) * p + Y * v) /
                    (alphac_one * (1 - Y) + Y))

        alphac_one = 1 + self.alphac
        yp, Hp = self.chargases['Y'], self.chargases['H']
        yv, Hv = self.volatiles['Y'], self.volatiles['H']

        y = {sp: mix(yp.get(sp, 0), yv.get(sp, 0))
             for sp in set(list(yp.keys()) + list(yv.keys()))}

        H = mix(Hp, Hv)
        mix_fuel = {'H': H, 'Y': y,
                    'T': [pyFLUT.utilities.calc_Tf(
                    gas, H_i, self.pressure, y)
             for H_i in H]}

        mix_fuel['Zst'] = pyFLUT.utilities.calc_zstoich(
            fuel=mix_fuel, oxidizer=self.oxidizer, gas=gas)
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