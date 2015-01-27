"""File: paramAction.py

Description: Python module defining an action class to react on PARINFO covise
             messages.

TODO: EXPLAIN: WHAT IS 'PARINFO covise messages'?

This module contains

o ParamAction


Initial version 05.07.2006 [RM]

(C) 2004-2006 Visenso GmbH, Stuttgart   info@visenso.de

"""

from CoviseMsgLoop import CoviseMsgLoopAction


class NotifyAction:

    """Thought for use with coivse-modules to listen to parameter-changes."""

    param_ = None
    value_ = None

    def run(self):
        pass


class ParamAction(CoviseMsgLoopAction):

    UI=6
    PARAM=47
    def __init__(self, modname=None, instance=-1, params=None):
        CoviseMsgLoopAction.__init__(
            self,
            "PARAMACTION" + modname + str(instance),
            (self.UI,self.PARAM), "action to react on parameter changes" )
        self.__modname_  = modname
        self.__instance_ = instance
        self.__params_ = {}
        self.__notifyFunc_ = None
        # transform parm list to dict
        if params:
            for p in params:
                self.__params_[p.name_] = p

        self.__infoPrintIsOn = False # Goes together with self.__printInfo.

    def run(self, param):

        self.__printInfo('entering method run(param == "%s")' % (str(param),))
        if int(self.__instance_) > -1:

            if self.runType_== self.PARAM:
                if str(self.__modname_) == str(param[0]) and \
                       str(self.__instance_) == str(param[1]):
                    paramObj = self.__params_[param[3]] #4
                    value=self.__getValue(paramObj, param)
                    paramObj.setValue(value)
                    # run notifier if present
                    self.__runNotifier(paramObj, param, value, 3)
                else :
                    raise NameError

            if self.runType_==self.UI:
                if str(param[0])=='PARAM' and str(self.__modname_)==str(param[1]) and \
                       str(self.__instance_)==str(param[2]):
                    #print('   ++++ running  UI-PARAM')
                    paramObj = self.__params_[param[4]]
                    value=self.__getValue(paramObj, param[1:])
                    #print('   ++++ UI-PARAM param ',param)
                    paramObj.setValue(value)
                    # run notifier if present
                    self.__runNotifier(paramObj, param, value)
                else: raise NameError

    def __getValue(self, paramObj, param, startIdx=7):
        retVal=()
        if paramObj.type_ == 'xxxChoice':
            retVal = retVal + (param[startIdx],)
            ss = str(param[startIdx+1],)
            choiceLabels = ss.split(' ')
            #choiceLabels=choiceLabels[0:-1] # here we assume that the last entry
            #is always an emty string
            # check if we got no lables (i.e.) we have a UI-PARAM msg
            if len(choiceLabels)<2 and choiceLabels[0]=='':
                choiceLabels=paramObj.choiceLabels_
            else: # if we got choiceLabels we store it in the parameter 
                paramObj.choiceLabels_=choiceLabels
            if choiceLabels:
                for item in choiceLabels:
                    retVal = retVal + (item,)
        elif paramObj.type_ == 'Browser':
            try:
                retVal = param[6]
            except IndexError:
                assert False
        elif paramObj.type_ == "String":
            # dont split up strings at whitespaces, return them unchanged
            retVal = param[5]
        elif paramObj.type_ == "ColormapChoice":
            retVal = param[5]
        else:
            try:
                ss = param[5]
                vals = ss.split(' ')
                for item in vals:
                    if (len(item) > 0) and (item != " "):
                        retVal = retVal + (item,)
            except IndexError:
                assert False

            #for cc in range(0,int(param[6])):
            #    retVal = retVal + (param[7+cc],)
        return retVal



    def __runNotifier(self,paramObj, param, value, paramIdx=4):
        if paramObj.notifyAction_:
            act = paramObj.notifyAction_
            try:
                if isinstance(act, NotifyAction):
                    act.param_ = str(param[paramIdx])
                    act.value_ = value
                    act.run()
                else:
                    print('Warning: python-module paramAction: class ParamAction: method run:  There is an unexpected object for delegation of the run-call.  Doing nothing.')
            except (Exception, a):
                print('ParamAction - EXCEPTION in running notifier ', a)



    def __printInfo(self, printable):
        if self.__infoPrintIsOn:
            print('(info) python-module paramAction: class %s: %s' % (self.__class__.__name__, str(printable)))

