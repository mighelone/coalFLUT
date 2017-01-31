from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

from .flut import CoalFLUT
from scoop import futures
import functools
import numpy as np
from autologging import logged
import pyFLUT

mapf = map


def get_equilibrium(fuel, oxidizer, variables, P, mechanism, Z, Zmin, Zmax):
    """
    Estimate equilibrium solution for the given fuel and oxidizer
    """
    eq = pyFLUT.equilibrium.EquilibriumSolution(
        fuel=fuel, oxidizer=oxidizer, pressure=P, mechanism=mechanism,
        Z=Z)
    eq.calc_equilibrium(Zmin=Zmin, Zmax=Zmax)
    return pyFLUT.Flame1D(data=eq.data, output_dict=eq.output_dict,
                          input_var='Z',
                          variables=variables)


@logged
class CoalFLUTEq(CoalFLUT):
    """
    Coal FLUT with equilibrium
    """

    def run_scoop(self):
        def fuel_gen():
            for Yi in self.Y:
                for H in self.Hnorm:
                    mix = self.mix_streams(Yi)
                    self.__log.debug(
                        'Y=%s H_mix=%s', Yi, mix['H'])
                    fuel = mix.copy()
                    H_fuel = ((mix['H'][1] - mix['H'][0]) *
                              H + mix['H'][0])
                    self.__log.debug('Y=%s Hnorm=%s', Yi, H)
                    fuel['T'] = pyFLUT.utilities.calc_Tf(
                        self.gas, H_fuel, self.pressure, mix['Y'])
                    self.__log.debug('Tf=%s', fuel['T'])
                    yield fuel

        def oxid_gen():
            for Yi in self.Y:
                for H in self.Hnorm:
                    oxid = self.oxidizer.copy()
                    H_oxid = (oxid['H'][1] - oxid['H'][0]) * \
                        H + oxid['H'][0]
                    oxid['T'] = pyFLUT.utilities.calc_Tf(
                        self.gas, H_oxid, self.pressure, oxid['Y'])
                    self.__log.debug('Toxidizer=%s', oxid['T'])
                    yield oxid

        def parameters_gen():
            for Yi in self.Y:
                for H in self.Hnorm:
                    yield {'Y': Yi, 'Hnorm': H}

        Z = np.linspace(0, 1, 101)
        results = list(mapf(
            functools.partial(
                get_equilibrium,
                P=self.pressure,
                mechanism=self.mechanism,
                Z=self.z_points,
                Zmin=0,
                Zmax=1),
            fuel_gen(),
            oxid_gen(),
            parameters_gen()))

        self.assemble_data(results)
