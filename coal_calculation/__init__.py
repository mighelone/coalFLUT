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

    def flame_index(self, species=['CH4', 'C2H4', 'C6H6']):
        X = self['X']
        y_fuel = np.zeros_like(X)
        for sp in species:
            y_fuel += self[sp]
        y_fuel /= y_fuel.max()
        y_oxid = self['O2'] / self['O2'].max()
        self['fuel'] = y_fuel
        self['oxid'] = y_oxid
        grad_fuel = self.gradient('fuel', along='X')
        grad_oxid = self.gradient('oxid', along='X')
        self['FI'] = grad_fuel * grad_oxid
        self['FIn'] = 0.5 * (1 + self['FI'] / np.abs(self['FI']))

    def calc_Hnorm(self, H_o, H_f):
        """
        Calculate the normalized enthalpy levels for a coal1D solution

        Parameters
        ----------
        H_o: np.array
            array containing minimum and max oxidizer enthalpy in the table
        H_f: np.array
            array containing minimum and max fuel enthalpy in the table
        z_DH: float, default=0.1
            minimum values of Z for negative enthalpy levels
        use_zb: bool, default=True
            use the bilger mixture fraction (defined before)
        """
        z = self['Z']
        hMean = self['hMean']
        if isinstance(H_o, list):
            H_o = np.array(H_o)
        if isinstance(H_f, list):
            H_f = np.array(H_f)
        H_z = (z[:, np.newaxis] * H_f[np.newaxis] +
               (1 - z[:, np.newaxis]) * H_o[np.newaxis])

        # index_neg = (hMean < H_z[:, 0]) & (z != 0) & (z != 1)
        # index_lean = z < z_DH
        # index_neg_lean = index_neg & index_lean
        # index_neg_rich = index_neg & (~ index_lean)
        Hnorm = (hMean - H_z[:, 0]) / (H_z[:, 1] - H_z[:, 0])
        # H_DH_z = np.zeros_like(z)
        # H_DH_z[index_neg_lean] = (z_DH / z[index_neg_lean] *
        #                          (hMean[index_neg_lean] -
        #                           H_o[0]) + H_o[0])
        # H_DH_z[index_neg_rich] = ((1 - z_DH) / (1 - z[index_neg_rich]) *
        #                          (hMean[index_neg_rich] - H_f[0]) +
        #                          H_f[0])

        # Hnorm[index_neg] = (H_DH_z[index_neg] -
        #                    H_DH[0]) / (H_DH[1] - H_DH[0])
        self['Hnorm'] = Hnorm
        self.H_o = H_o
        self.H_f = H_f

    def calc_zN2(self, N2f=0):
        """
        Calculate Z for coal1D solution using N2

        Parameters
        ----------
        coal1D: UlfData
            ulf solution
        N2f: float, default=0
            N2 contents in fuel
        """
        N2 = self['N2']
        N2o = N2[0]
        self['Zb'] = (N2o - N2) / (N2o - N2f)

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
        X = self['X']
        points = np.empty((len(X), len(flut.input_dict)))
        for i, inp_var in enumerate(flut.input_variables):
            if inp_var == 'Y':
                points[:, i] = 1
            elif inp_var == 'cc':
                points[:, i] = 0
            else:
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
        z = self['Z']
        H_f = self.H_f
        H_o = self.H_o
        H_min = (1 - z) * H_o[0] + z * H_f[0]
        dH = H_min - hMean
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
