
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

import KeydObject
from KeydObject import globalColorMgrKey, globalKeyHandler, coKeydObject, TYPE_COLOR_MGR

from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False

class coColorMgr(coKeydObject):
    def __init__(self):
        coKeydObject.__init__(self, TYPE_COLOR_MGR, 'ColorMgr')
        self.params = coColorMgrParams()

    def getKeyOfTable( self, species ):
        if species==None:
            return None
        for obj in self.objects:
            if obj.params.species==species:
                return obj.key
        return None

    def getRightName(self, species):
        cnt=0
        _infoer.function = str(self.run)
        _infoer.write("looking for %s" % species)
        for obj in self.objects:
            _infoer.write("found %s, %s " % ( cnt, obj.params.species) )
            if obj.params.species==species:
                cnt=cnt+1
        if cnt==0:
            return species
        else:
            return "%s_%d" % (species, cnt)
    
    def recreate(self, negMsgHandler, parentKey, offset):        
        coKeydObject.recreate(self, negMsgHandler, parentKey, offset)
        if offset>0 :
            globalKeyHandler().getObject(globalColorMgrKey).merge(self)

    def run(self, runmode, negMsgHandler=None):
        #interrupt recursive run
        return
     
class coColorMgrParams(object):
    def __init__(self):
        self.name  = 'ColorMgrParams'
