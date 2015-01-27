
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH


from KeydObject import coKeydObject, RUN_ALL, globalKeyHandler, TYPE_JOURNAL, TYPE_JOURNAL_STEP
from Utils import CopyParams, ParamsDiff

STEP_PARAM = 0
STEP_ADD   = 1
STEP_DEL   = 2

from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False # True

class coJournalMgr(coKeydObject):
    """ class to handle project files """
    def __init__(self):
        coKeydObject.__init__(self, TYPE_JOURNAL, 'Journal')
        globalKeyHandler().globalJournalMgrKey = self.key
        self.params = coJournalMgrParams()
        self.name = self.params.name
        # children build history of params: coJournalStep    

    def recreate(self, negMsgHandler, parentKey, offset):
        coKeydObject.recreate(self, negMsgHandler, parentKey, offset)
        globalKeyHandler().globalJournalMgrKey = self.key

    def addObject( self, obj ):
        _infoer.function = str(self.addObject)
        _infoer.write("name: %s" % obj.params.key)
        coKeydObject.addObject(self,obj)
        self.params.maxIdx = self.__numSteps()-1
        self.params.currentIdx = self.params.maxIdx        
        
    def __numSteps(self):
        return len(self.objects)

    def hasStatusForKey(self, key):
        for obj in self.objects:
            if obj.params.key==key:
                return True
        return False
                
    def setParams( self, params, negMsgHandler=None, sendToCover=False):
        _infoer.function = str(self.setParams)
        _infoer.write("old index: %d, new_index: %d, max_index:%d " % ( self.params.currentIdx, params.currentIdx, self.params.maxIdx) )
        # change of maxIdx means shorten the history
        changedParams = ParamsDiff( self.params, params)
        if 'maxIdx' in changedParams:
            self.params.maxIdx = params.maxIdx
            for idx in range(self.__numSteps()-self.params.maxIdx-1):
                del self.objects[0]
            self.params.currentIdx=self.params.maxIdx
            if negMsgHandler:
                negMsgHandler.sendParams( self.key, self.params )
            return

        if self.params.currentIdx==params.currentIdx:
            _infoer.write("currentIdx did not change")
            return

        old_index = self.params.currentIdx
        if params.currentIdx<-1 or params.currentIdx>self.params.maxIdx:
            _infoer.write("currentIdx out of range")
            return

        coKeydObject.setParams( self, params )

        #restore settings
        objsToRefresh = {}                
        if self.params.currentIdx>old_index:
            inc = 1
        else:
            inc = -1
        for currentStepDiff in range(inc, self.params.currentIdx-old_index+inc, inc):                
            pStepParams = self.objects[old_index+currentStepDiff].params
            obj = globalKeyHandler().getObject( pStepParams.key )
            #print "Setting params of object ", obj.name, pStepParams.key, pStepParams.param.__dict__
            
            if pStepParams.action==STEP_PARAM:
                obj.setParams( CopyParams(pStepParams.param), negMsgHandler )
                objsToRefresh[obj] = True
                if negMsgHandler:                    
                    negMsgHandler.sendParams( obj.key, obj.params )
                else:
                    _infoer.write("param change not send to gui")    

        #auto apply
        for obj in objsToRefresh:
            obj.run(RUN_ALL, negMsgHandler)

class coJournalMgrParams(object):
    def __init__(self):
        self.name       = 'History'
        self.currentIdx = -1
        self.maxIdx     = -1        

class coJournalStep(coKeydObject):
    """ class to handle session history
        TODO handle add/del of objects """
    def __init__(self):
        coKeydObject.__init__(self, TYPE_JOURNAL_STEP, 'JournalStep')
        self.params = coJournalStepParams()

class coJournalStepParams(object):
    def __init__(self):
        self.name   = 'JournalStep'
        self.action  = STEP_PARAM
        self.key     = 0
        self.param   = None
        
