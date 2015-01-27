
# Part of the vr-prepare program for dc

# Copyright (c) 2006-2007 Visenso GmbH

from KeydObject import coKeydObject, TYPE_CAD_PRODUCT, TYPE_CAD_PART
from coCADPartMgr import coCADPartMgrParams
from VRPCoviseNetAccess import theNet
from coPyModules import ReadCAD
from Utils import ParamsDiff
from printing import InfoPrintCapable
import copy

_infoer = InfoPrintCapable()
_infoer.doPrint = False # True


class coCADMgr(coKeydObject):
    """ class handling cad files """
    def __init__(self):
        coKeydObject.__init__(self, TYPE_CAD_PRODUCT, 'TYPE_CAD_PRODUCT')
        self.params = coCADMgrParams()
        self.receivedParts = False
        self.partList = []

    def initPartTree( self, negMsgHandler ):
        readCAD = ReadCAD()
        theNet().add(readCAD)
        negMsgHandler.registerParamsNotfier( readCAD, self.key, ['SelectPart'] )
        readCAD.set_catia_server('obie')
        readCAD.set_catia_server_port('7000')
        readCAD.set_file_path( self.params.filename )
        self.neg = negMsgHandler

    def setParamsByModule( self, mparam, mvalue):
        """ receives parameter changes from the readCAD module
            return a list of objKey and their parameters to be set by the Neg2Gui class
        """
        _infoer.function = str(self.setParamsByModule)
        _infoer.write(" ")
        pChangeList = []
        self.partList = mvalue[:len(mvalue)-1]
        self.receivedParts = True
        partCnt = 2
        for part in self.partList[2:]:
            print("Part ", part)
            if not part==None:
                pP = coCADPartMgrParams()
                pP.name = part[part.find('|')+1:]
                pP.filename = self.params.filename
                pP.index = partCnt
                #pP.featureAngle = self.params.featureAngle
                #pP.max_Dev_mm   = self.params.max_Dev_mm
                #pP.max_Size_mm  = self.params.max_Size_mm
                partCnt = partCnt+1
                partVis = self.neg.internalRequestObject( TYPE_CAD_PART, self.key, pP )
                partVis.setParams(pP, None)
                self.neg.sendParams(partVis.key, pP)

        return pChangeList

    def setParams( self, params, negMsgHandler):
        _infoer.function = str(self.setParams)
        _infoer.write(" ")
        realChange = ParamsDiff( self.params, params )
        coKeydObject.setParams( self, params)

        if 'filename' in realChange:
            self.initPartTree(negMsgHandler)

    def __getstate__(self):
        """ __getstate__ returns a cleaned dictionary
            only called while class is pickled
        """
        mycontent = copy.copy(self.__dict__)
        del mycontent['neg']
        return mycontent

    def recreate(self, negMsgHandler, parentKey, offset):
        """ recreate is called after all classes of the session have been unpickled """
        self.receivedParts = False
        self.partList = []
        coKeydObject.recreate(self, negMsgHandler, parentKey, offset)


class coCADMgrParams(object):
    def __init__(self):
        self.name = 'testCAD'
        self.filename = None
        self.featureAngleDefault = True
        self.featureAngle = 30
        self.max_Dev_mm_Default = True
        self.max_Dev_mm   = 20
        self.max_Size_mm_Default = True
        self.max_Size_mm  = 400

# eof
