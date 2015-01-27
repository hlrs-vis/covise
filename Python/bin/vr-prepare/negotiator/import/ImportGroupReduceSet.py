
# Part of the vr-prepare 

# Copyright (c) 2008 Visenso GmbH


""" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
"""                                                                                             """
"""      simplify the content of the ImportGroup with AssembleUsg                               """
"""                                                                                             """
""" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """


from ImportGroupSimpleFilter import ImportGroupSimpleFilter
from VRPCoviseNetAccess import ReduceSetModule


class ImportGroupReduceSet(ImportGroupSimpleFilter):
    def __init__(self, iGroupModule):
        ImportGroupSimpleFilter.__init__(self, iGroupModule, ReduceSetModule)
        self.__factor = 1 
        
    def setReductionFactor( self, factor):
        self.__factor = factor
        
    def _update(self, varname=None):
        if varname==None:
            self._filterGeo.setReductionFactor(self.__factor)
        else:
            self._filterVar[varname].setReductionFactor(self.__factor)
        