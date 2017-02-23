from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

from .flut import CoalFLUT
from scoop import futures
import functools
import numpy as np
from autologging import logged
import pyFLUT

mapf = futures.map


def z_from_phi(phi, zst):
    """
    Calculate Z from phi
    """
    return zst * phi / (zst * phi + (1 - zst))


@logged
def get_equilibrium(fuel, oxidizer, variables, P, mechanism, Z):
    """
    Estimate equilibrium solution for the given fuel and oxidizer
    """
    eq = pyFLUT.equilibrium.EquilibriumSolution(
        fuel=fuel, oxidizer=oxidizer, pressure=P, mechanism=mechanism,
        Z=Z)
    Zmin = z_from_phi(0.1, eq.Z_st)
    Zmax = z_from_phi(3, eq.Z_st)
    eq.calc_equilibrium(Zmin=Zmin, Zmax=Zmax)
    get_equilibrium._log.debug('var=%s Zmin=%s Zmax=%s, T_fuel=%s - %s',
                               variables, Zmin, Zmax, eq['T'][-1],
                               fuel['T'])
    return pyFLUT.Flame1D(data=eq.data, output_dict=eq.output_dict,
                          input_var='Z',
                          variables=variables)


@logged
class CoalFLUTEq(CoalFLUT, pyFLUT.Flut):
    """
    Coal FLUT with equilibrium
    """
    export_variables = ['T', 'rho', 'p', 'MMean', 'cpMean',
                        'hMean', 'visc', 'alpha', 'lambda']

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
                    H_oxid = ((oxid['H'][1] - oxid['H'][0]) *
                              H + oxid['H'][0])
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
                Z=self.z_points),
            fuel_gen(),
            oxid_gen(),
            parameters_gen()))

        self.assemble_data(results)

    def write_hdf5(self, file_name='FLUT.h5', turbulent=False,
                   n_proc=1, verbose=False):
        output_variables = list(set(self.export_variables +
                                    self.gas.species_names))

        self.__log.debug('out variables: %s', output_variables)

        return pyFLUT.Flut.write_hdf5(self, file_name=file_name,
                                      cantera_file=self.mechanism,
                                      # regular_grid=True,
                                      # shape=tuple(shape),
                                      output_variables=output_variables,
                                      turbulent=turbulent,
                                      n_var=len(
                                          self.varZ),
                                      solver=self.output_solver,
                                      n_proc=n_proc,
                                      verbose=True)
