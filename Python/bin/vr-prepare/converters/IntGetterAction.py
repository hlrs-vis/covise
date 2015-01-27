from paramAction import NotifyAction

class IntGetterAction(NotifyAction):

    """One-time-use-action for Ints"""

    def __init__(self):
        self.readingDone_ = False
        
    def run(self):
        self.readingDone_ = True

    def waitForInts(self):
        while not self.readingDone_: pass

    def resetWait(self):
        self.readingDone_ = False

    def getInt(self):
        #print "IntGetterAction.getInt"
        #print self.value_
        if self.readingDone_ :                
            return int(self.value_[0])
        else :
            return None    
