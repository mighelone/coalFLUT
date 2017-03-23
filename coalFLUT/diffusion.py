from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


from pyFLUT.ulf.sldf_2stream import DiffusionFlut2Stream
from .abstract import AbstractCoalFLUT
from autologging import logged
import h5py


@logged
class CoalDiffusionFLUT(AbstractCoalFLUT, DiffusionFlut2Stream):

    def write_hdf5(self, file_name='FLUT.h5', turbulent=False,
                   n_proc=1):
        # add AlphaC to h5 file_input
        self_h5 = super(CoalDiffusionFLUT, self).write_hdf5(file_name=file_name,
                                                            turbulent=turbulent,
                                                            n_proc=n_proc)
        self.__log.debug('Add AlphaC=%s to FLUT file', self.alphac)
        h5 = h5py.File(file_name, 'r+')
        h5['Input'].create_dataset(name='AlphaC',
                                   shape=(1,),
                                   dtype=float,
                                   data=[self.alphac])
        h5.close()
        return self_h5
