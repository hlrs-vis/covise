##########################################################################################
#
# Python-Module: GlobalParamDict.py
#
# Description: python module holding a global dictionary of parameters
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


class _GlobalParamDict:
    instance_=None
    def __init__(self):
        self.globalParamDict_ = {}

    #
    # a is param-msg from covise packed to a list
    def set( self, a):
        name = str(a[1]) + str(a[2])
        paramName = a[4]
        paramName.replace(' ','_')

        if not self.globalParamDict_.has_key(name) :
            params = {}
            params[paramName] = a[6]
            self.globalParamDict_[name] = params

        else:
            params = self.globalParamDict_[name]
            params[paramName] = a[6]


    def printIt(self):
        print("GCDict> ",  self.globalParamDict_)

        
    def getParam(self, modName, paramName):
        if not self.globalParamDict_.has_key( modName ) :
           return None
        else:
           modDict = self.globalParamDict_[ modName ]
           if not modDict.has_key( paramName ):
               return None
           else:
               a = str(modDict[ paramName ])
               la = a.split(' ')
               return la
               
           
#
# create access ( disguise the the class )
#
def GlobalParamDict():
    if _GlobalParamDict.instance_ == None:
        _GlobalParamDict.instance_ = _GlobalParamDict()
    return _GlobalParamDict.instance_

