##########################################################################################
#
# Python-Module: GlobalChoiceDict.py
#
# Description: python module holding a global dictionary of dynamic choices
#
# IMPORTANT WARNING:  N E V E R    F O R M A T    T H I S    F I L E !!!!              
# =======================================================================
# ( if you don't know what youre' doing )
#
# (C) 2002 VirCinity GmbH, Stuttgart   info@vircinity.com
#
# 08.12.2003 [RM]
#
##########################################################################################


class _GlobalChoiceDict:
    instance_=None
    def __init__(self):
        self.globalChoiceDict_ = {}

    #
    # a is param-msg from covise packed to a list
    def set( self, a):
        
        if len(a) == 9:
            name = str(a[0]) + str(a[1])
            #print("got name ",name)

            paramName = a[4]
            paramName.replace(' ','_')
            allChoices = a[8]

            if not self.globalChoiceDict_.has_key(name) :
                params = {}
                params[paramName] = allChoices
                self.globalChoiceDict_[name] = params

            else:
                params = self.globalChoiceDict_[name]
                #if params.has_key(paramName):
                params[paramName] = allChoices


    def printIt(self):
        print("GCDict> ",  self.globalChoiceDict_)

        
    def getChoices(self, modName, paramName):
        if not self.globalChoiceDict_.has_key( modName ) :
            return None
        else:
           modDict = self.globalChoiceDict_[ modName ]
           if not modDict.has_key( paramName ):
               return None
           else:
               a = str(modDict[ paramName ])
               la = a.split(' ')
               return la
               
           
#
# create access ( disguise the the class )
#
def GlobalChoiceDict():
    if _GlobalChoiceDict.instance_ == None:
        _GlobalChoiceDict.instance_ = _GlobalChoiceDict()
    return _GlobalChoiceDict.instance_

