
# Part of the vr-prepare 

# Copyright (c) 2008 Visenso GmbH


""" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
"""                                                                                             """
"""      simplify the content of the ImportGroup with AssembleUsg                               """
"""                                                                                             """
""" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """


from ImportGroupSimpleFilter import ImportGroupSimpleFilter
from VRPCoviseNetAccess import AssembleUsgModule


class ImportGroupAssembleUsg(ImportGroupSimpleFilter):
    def __init__(self, iGroupModule):
        ImportGroupSimpleFilter.__init__(self, iGroupModule, AssembleUsgModule)