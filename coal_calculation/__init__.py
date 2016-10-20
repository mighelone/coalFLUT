"""
Coal calculation
================

Utilities for coal1D calculation
"""
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import pyFLUT
from pyFLUT.flame1D import extract_from_filename
import matplotlib.pyplot as plt
from autologging import logged


@logged
class Coal1D(pyFLUT.Flame1D):

    def __init__(self, file_input):
        split_type = None
        with open(file_input, 'r') as f:
            read_data = f.readlines()
            index = read_data[0].split(split_type)
            read_data.pop(0)
            data = np.empty((len(read_data), len(index)))
            for i, d in enumerate(read_data):
                data[i] = d.split(split_type)
        file_vars = extract_from_filename(file_input)
        super(Coal1D, self).__init__(
            output_dict=index, data=data, variables=file_vars,
            input_var='X')
        # calc Z and Y
        Z = self['Z'] + self['ZCfix']
        Y = np.zeros_like(Z) + 1.
        cond = Z > 0
        Y[cond] = self['Z'][cond] / Z[cond]

        self['Zsum'] = Z
        self['Y'] = Y

    def calc_Zstar(self, flut):
        self['Zstar'] = (self['Zsum'] *
                         (self['Y'] + (1 - self['Y']) * (1 + flut.alphac)))

    def calc_Hnorm_premix(self, flut):
        """
        Calculate the normalized enthalpy levels for a coal1D solution

        Parameters
        ----------
        flut: pyFLUT.Flut
        Lookup table. It has to be defined with Z and Y
        """
        Hc, Ho, Hv = (getattr(flut, stream)['H'][:, np.newaxis]
                      for stream in ('chargases', 'oxidizer',
                                     'volatiles'))
        self.Ho = Ho
        self.Hv = Hv
        self.Hc = Hc

        Y = self['Y']
        Z = self['Zsum']
        Hf = (Y * Hv + (1 - Y) * (1 + flut.alphac) * Hc)

        # min and max enthalpy
        H = Ho * (1 - Z) + Hf * Z

        self.calc_Zstar(flut)
        X = self['X']
        points = np.empty((len(X), len(flut.input_dict)))
        for i, inp_var in enumerate(flut.input_variables):
            if inp_var == 'cc':
                self.__log.debug('Set %s=0 to index %s', inp_var, i)
                points[:, i] = 0
            elif inp_var == 'Z':
                self.__log.debug('Set Zstar / %s to index %s', inp_var, i)
                points[:, i] = self['Zstar']
            elif inp_var == 'Hnorm':
                self.__log.debug('Set %s=0 to index %s', inp_var, i)
                points[:, i] = 0
            else:
                self.__log.debug('Set %s to index %s', inp_var, i)
                points[:, i] = self[inp_var]
        self['Hmin'] = flut.getvalue(points, 'hMin')
        self['Hmax'] = flut.getvalue(points, 'hMax')
        #self['Hmin'] = H[0]
        #self['Hmax'] = H[1]

        self['Hnorm'] = (self['hMean'] - self['Hmin']) / (self['Hmax'] - self['Hmin'])

    def calc_Hnorm(self, flut):
        """
        Calculate the normalized enthalpy levels for a coal1D solution

        Parameters
        ----------
        flut: pyFLUT.Flut
        Lookup table. It has to be defined with Z and Y
        """
        Hc, Ho, Hv = (getattr(flut, stream)['H'][:, np.newaxis]
                      for stream in ('chargases', 'oxidizer',
                                     'volatiles'))
        self.Ho = Ho
        self.Hv = Hv
        self.Hc = Hc

        Y = self['Y']
        Z = self['Zsum']
        Hf = (Y * Hv + (1 - Y) * (1 + flut.alphac) * Hc)

        # min and max enthalpy
        H = Ho * (1 - Z) + Hf * Z

        self['Hmin'] = H[0]
        self['Hmax'] = H[1]

        self['Hnorm'] = (self['hMean'] - H[0]) / (H[1] - H[0])

    def calc_a_priori_analysis(self, scl, flut, output='simple', correct=False):
        """
        Calculate scl array from a-priori analysis

        Parameters
        ----------
        scl: str
            scalar string name
        flut: UlfDataSeries
            flut table
        output: str {'simple', 'all'}
            Define which output return.
            simple: return only the a-priori data of scl
            points: return X, points, y
            all: return X, points, yc_min, yc_max, y
        correct: Bool (default: False)
            Correct temperature and density if Hnorm < 1

        Returns
        -------
        np.array(N), np.array(N, 4), np.array(N)
            X, points, scl arrays
        """
        if 'Hnorm' not in self:
            self.calc_Hnorm(flut)
        if 'Zstar' not in self:
            self.calc_Zstar(flut)
        X = self['X']
        points = np.empty((len(X), len(flut.input_dict)))
        for i, inp_var in enumerate(flut.input_variables):
            if inp_var == 'cc':
                self.__log.debug('Set %s=0 to index %s', inp_var, i)
                points[:, i] = 0
            elif inp_var == 'Z':
                self.__log.debug('Set Zstar / %s to index %s', inp_var, i)
                points[:, i] = self['Zstar']
            else:
                self.__log.debug('Set %s to index %s', inp_var, i)
                points[:, i] = self[inp_var]
        yc_max = flut.getvalue(points, 'yc_max')
        yc_min = flut.getvalue(points, 'yc_min')
        cc = np.zeros_like(yc_max)
        dyc = yc_max - yc_min
        cond = dyc > 0
        cc[cond] = (self['yc'][cond] - yc_min[cond]) / dyc[cond]
        # check if cc > 0 and cc < 1
        cc[cc < 0] = 0
        cc[cc > 1] = 1
        points[:, flut.input_variable_index('cc')] = cc
        self.__log.debug('Set cc to index %s')

        if correct and scl in ['T', 'rho']:
            dH = self.dH(flut)
            T = flut.getvalue(points, 'T')
            T_corr = T - dH / flut.getvalue(points, 'cpMean')
            if scl == 'T':
                y = T_corr
            elif scl == 'rho':
                y = flut.getvalue(points, 'rho') * T / T_corr
        else:
            y = flut.getvalue(points, scl)

        # # correct what is necessary to correct
        # if scl == 'rho':
        if output == 'simple':
            return y
        elif output == 'points':
            return X, points, y
        elif output == 'all':
            return X, np.insert(points, -1, [yc_min, yc_max], axis=-1), y

    def dH(self, flut):
        """
        Return DH for a given FLUT
        DH=0 for hMean, otherwise < 0

        Parameters
        ----------
        flut: ulf.UlfDataSeries
            FLUT table
        """
        hMean = self['hMean']
        dH = self['Hmin'] - hMean
        dH[dH < 0] = 0
        return dH

    def plot_a_priori_analysis(self, scl, flut, ax=None):
        """
        Plot a-priori analysis for scl variable

        Parameters
        ----------
        scl: str
            scalar string name
        coal1D: UlfData
            ulf solution
        flut: UlfDataSeries
            flut table

        Returns
        -------
        ax
        """
        if not ax:
            fig, ax = plt.subplots()
        X, _, y = self.calc_a_priori_analysis(scl, flut, output='all')
        self.plotdata('X', scl, label='Coal1D', ax=ax)
        ax.plot(X, y, label='FLUT', marker='o',
                markevery=20, linestyle='')
        ax.legend(loc='best')
        ax.set_xlim([0, 0.02])
        return ax

    def stagnation_X(self):
        """
        Calculate stagnation point
        """
        return self['X'][self.stagnation_index()]

    def stagnation_index(self):
        """
        Calculate stagnation index
        """
        return np.where(self['n_p'] == 0)[0][0]
