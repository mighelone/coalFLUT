"""
Coal Premix FLUT
"""
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

from ..abstract import AbstractCoalFLUT
from autologging import logged


@logged
class CoalPremixFLUT(AbstractCoalFLUT):
    """
    Coal FLUT using premix flames
    """

    _files = ()
    pass
