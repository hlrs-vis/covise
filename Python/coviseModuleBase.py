##########################################################################################
#
# Python-Module: coviseModuleBase.py
#
# Description: python module defining basics of COVISE scripting
#
# IMPORTANT WARNING:  N E V E R    F O R M A T    T H I S    F I L E !!!!              
# =======================================================================
# ( if you don't know what youre' doing )
#
# (C) 2002 VirCinity GmbH, Stuttgart   info@vircinity.com
#
# 31.10.2002 [RM]
#
##########################################################################################
import socket
import string
import sys, time

import covise
from covise import *
from threading import *
from GlobalChoiceDict import GlobalChoiceDict
from GlobalParamDict import GlobalParamDict
from paramAction import ParamAction
from CoviseMsgLoop import CoviseMsgLoop, CoviseMsgLoopAction

#
# thread safe queue
#
class BoundedQueue :
    def __init__(self, limit):
        self.mon = RLock()
        self.rc = Condition(self.mon)
        self.wc = Condition(self.mon)
        self.limit = 8000#limit
        self.queue = []
        
    def put(self, item):
        self.mon.acquire()
        while len(self.queue) >= self.limit:
            self.wc.wait()
        self.queue.append(item)
        self.rc.notify()
        self.mon.release()
            
    def get(self):
        assert False
        self.mon.acquire()
        while not self.queue:
            self.rc.wait(2.0)
        item = self.queue[0]
        del self.queue[0]
        self.wc.notify()
        self.mon.release()
        return item

    def size(self):
        return len( self.queue_ )



class BoundedStack :
    def __init__(self, limit):
        self.mon_ = RLock()
        self.limit_ = limit
        self.stack_ = []


    def put(self, item):
        self.mon_.acquire()
        if len(self.stack_) <= self.limit_:
            self.stack_.append(item)
        else:
            self.stack_.pop(0)
            self.stack_.append(item)
        self.mon_.release()       

    def get(self):
        self.mon_.acquire()
        ret=None
        if self.stack_:
            if len( self.stack_ ) > 0:
                ret = self.stack_.pop(0)
        self.mon_.release()
        return ret

    def __str__(self):
        return str( self.stack_ )

    def size(self):
        return len( self.stack_ )
        
    def clear(self):
        self.mon_.acquire()
        del self.stack_[:]
        self.mon_.release()
        
globalMessageQueue = BoundedQueue(180)


#
# utility class to store hostname
#
class HostCont:
    hn_ = ""
    def __init__(self,x):
        self.hn_=x
    def getName(self):
        return self.hn_
    def setName(self,x):
        self.hn_ = x
    


globalHostInfo= HostCont("")   

############################# CLASS PORT #####################################
class CovisePort:
    name_ = ""
    dataType_ = ""
    portType_ = ""
    coObjName_ = ""
    
    def __init__(self, name, dtype, ptype):
        coObjName = ""
        self.name_ = name
        self.dataType_ = dtype
        self.portType_ = ptype

def float_tuple(s, sep='\n'):
    ret = ()
    sl = s.split(sep)
    for f in sl:
        if len(f)>0:
            ret = ret + (float(f),)
    return ret


def int_tuple(s, sep='\n'):
    ret = ()
    sl = s.split(sep)
    for f in sl:
        if len(f)>0:
            ret = ret + ( int(f), )
    return ret
        
############################# CLASS PARAM #####################################
class CoviseParam:
    name_  = ""
    type_  = ""
    value_ = ""
    mode_  = ""
    notifyAction_ = None
    choiceLabels_=None
    
    def __init__(self, name, type, mode = "START"):
        self.name_ = name
        self.type_ = type
        self.mode_ = mode

    def setNotifyAction(self, act):
        self.notifyAction_ = act
        
    # set the value of a parameter    
    def setValue(self,value):
        self.value_ = value
    
    def setValueStr(self, s):

        """Set the value of a parameter given as string.

        The string is formated to fit in directly in the
        messages send to the controller.

        TBD: remove this method and switch to the more concise
        setValue - Method

        """

        # print('%s self.type_ == "%s", s == "%s"' %)
        # (repr(self.setValueStr), self.type_, s)

        # Parse functions dependent on type:
        pDict = {"Scalar"   : float,
                 "FloatScalar" : float,
                 "IntScalar" : int,
                 "Vector"   : float_tuple,
                 "IntVector": int_tuple,
                 "FloatVector": float_tuple,
                 "Choice"   : int,
                 "choice"   : int,
                 "Browser"  : str,
                 "String"   : str,
                 "Boolean"  : str,
                 "Colormap" : str,
                 "Timer"    : int,
                 "Slider"   : float_tuple,
                 "IntSlider"   : int_tuple,
                 "FloatSlider": float_tuple}

        # Remove first entry i.e. the number of following items.
        reducedStr = '\n'.join(s.split('\n')[1:])
        print("bla ", reducedStr)
        try:
            xx = pDict[self.type_](reducedStr)
        except ValueError:
            xx = 0
            
        if isinstance(xx, tuple):
            self.value_ = xx
        else:
            self.value_ = (xx,)
        print("Set ", self.name_, "to ", self.value_)
        
    def isChoice(self):
        return self.type_=='Choice'
    
############################# BASE CLASS MODULE #####################################
class CoviseModule:
    
    name_ = ""
    nr_   = "0"
    host_ = ""
    posx_ = 10
    posy_ = 10
    ports = []
    params_ = []
    coObjName_ = ""
    choiceDict_ = {}
    __paramAction_ = None
    
    def __init__(self):
        self.ports = []
        self.params_ = []           
        self.host_ = globalHostInfo.getName()
        self.title_ = None
    
    def setTitle( self, title ):        
        if not int(self.nr_) == 0:
            self.title_ = title            
            cn = "\n"
            titleStr = "MODULE_TITLE\n" + self.name_ + cn +                  \
                      self.nr_ + cn + self.host_  + cn + self.title_
            covise.sendCtrlMsg(titleStr)
                
    def createParamAction(self):
        if not self.__paramAction_:
            self.__paramAction_ = ParamAction(self.name_, self.nr_, self.params_)
            CoviseMsgLoop().register(self.__paramAction_)

    def addNotifier(self, paraname, act):

        """Add a notifier action for covise-parameter paraname.

        The action act should be a NotifyAction and is
        triggered when the parameter changes.

        Requirement: Parameter paraname must be an
        immediate-parameter.

        """
        foundParam = None
        for x in self.params_:
            if x.name_ == paraname:
                foundParam = x
                break
        if foundParam:
            foundParam.setNotifyAction(act)
            
    # init string to be send to the controller  
    def getInitStr(self):
    #==================== 
        cn = "\n"
        return "INIT\n" + self.name_ + cn + self.nr_ + cn + self.host_ + cn + "%d" % (self.posx_) + cn + "%d" % (self.posy_) + cn


    # delete string to be send to the controller
    def getDelStr(self):
    #==================== 
        cn = "\n"
        return "DEL\n1\n" + self.name_ + cn + self.nr_ + cn + self.host_ + cn
        

    # add a port to the covise-module
    def addPort(self, name, dtype, ptype):
    #=====================================    
        a = CovisePort(name, dtype, ptype)
        self.ports.append(a)
            

    # send DOCONN msg
    def finishPorts(self):
    #=====================    
        i = 0
        for x in self.ports:            
            if ( int(self.nr_) > 0):
                cn = "\n"
                if (x.portType_ == "OUT"):
                    x.coObjName_ = self.name_ + "_" + "%s" % (self.nr_) + "_OUT_" + "%2.2d" % i 
                    execStr = "OBJCONN2\n" + self.name_ + cn +  "%s" % (self.nr_) + cn + self.host_ + cn + x.name_ + cn + x.coObjName_ + cn
                    sendCtrlMsg( execStr )
                    i = i + 1        


    # get COVISE obj name
    def getCoObjName(self, name):
    #============================
        for x in self.ports:
            if ((x.name_ == name) and (x.portType_ == "OUT")):
                return x.coObjName_
        return ""    


    # show all ports
    def showPorts(self):        
    #===================
        for x in self.ports:
            print(x.name_)


    # show all parameters
    def showParams(self):      
    #===================
        for x in self.params_:
            print(x.name_,"(",x.type_,")")


    # add parameter to the module        
    def addParam(self, name, type, mode="START"):
    #==============================
        a = CoviseParam(name, type, mode)
        self.params_.append(a)
  
    # set the value directly in the param class
    # send no param msg to the controller
    def setParamValueWithoutMsg( self, name, value):
        foundParam = None
        for x in self.params_:
            if x.name_ == name:
                foundParam = x
                x.setValue(value)
                break
        return foundParam
        
    def setParamValue(self, name, value):

        """Set the value of a covise-parameter name.

        Value must be in the format given by the covise controller.

        """

        foundParam = self.setParamValueWithoutMsg( name, value)
        if int(self.nr_) > 0:
            execStr=" "

            if foundParam.type_ == 'Boolean':
                value = value.upper()

            #if foundParam.type_ == 'Browser':
                #foundParam.setValueStr( value )
                #cn = "\n"
                #execStr = "PARAM" + cn              \
                      #+ self.name_ + cn         \
                      #+ self.nr_ + cn \
                      #+ self.host_  + cn        \
                      #+ name + cn               \
                      #+ foundParam.type_ + cn   \
                      #+ foundParam.value_[0] + cn
            #else :
            cn = "\n"
            execStr = "PARAM_INIT" + cn              \
                    + self.name_ + cn         \
                    + self.nr_ + cn \
                    + self.host_  + cn        \
                    + name + cn               \
                    + foundParam.type_ + cn   \
                    + str(value)

            covise.sendCtrlMsg(execStr)
            if  foundParam.mode_ == "IMM" :
                cn = "\n"
                execStr = "PARAM_INIT" + cn                 \
                          + self.name_ + cn             \
                          +  self.nr_ + cn     \
                          + self.host_ + cn             \
                          + name + cn                   \
                          + foundParam.type_ + cn       \
                          + str(value)
                covise.sendCtrlMsg(execStr)


    def getParamValue(self, name):
    #====================================
        """Return covise-module-parameter name.

        Return None if the covise-module-parameter name doesn't exist.

        """

        foundParam = None
        for x in self.params_:
            if x.name_ == name:
                foundParam = x
                break
        if not foundParam:
            print("getParamValue - parameter does not belong to module")
            return None
        return foundParam.value_
     
    

    # set position value
    def setPos(self, x, y):
    #====================================    
        # search parameter
        cn = "\n"
        moveStr = "MOV\n1\n" + self.name_ + cn +  self.nr_ + cn + self.host_  + cn  + "%d" % (x) + cn  + "%d" % (y)
        sendCtrlMsg( moveStr )     
            

            
    # execute a module
    def execute(self):
    #=================          
        cn = "\n"    
        execStr = "EXEC\n" + self.name_ + cn +  self.nr_ + cn + self.host_  + cn
        sendCtrlMsg( execStr )
        


class ModuleCont:
    dict_={}
    def add( self, x ):
        self.dict_[x]=0

    def getDict( self ):
        return self.dict_
        

globalModules = ModuleCont()
                

############################# CLASS NET #####################################    
class net :

    def printNet(self, change=None):
        if self.debug_printNetChange or not change:
            print("")
            if change:
                print("########################################################################")
                print("NET MODIFIED * NET MODIFIED * NET MODIFIED * NET MODIFIED * NET MODIFIED")
                print(change)
                print("########################################################################")
                if self.debug_printNetChange_includeTraceback:
                    import traceback
                    traceback.print_stack()
            print("+-------------------------------------------------------------+")
            print("|                       The Net (Modules)                     |")
            print("+-------------------------------------------------------------+")
            self.actualModules_.sort(lambda x, y: cmp(x.name_+str(x.nr_), y.name_+str(y.nr_)))
            for m in self.actualModules_:
                print("| " + m.name_ + "_" + str(m.nr_))
            print("+-------------------------------------------------------------+")
            print("|                       The Net (Connections)                 |")
            print("+-------------------------------------------------------------+")
            self.__connections.sort(lambda x, y: cmp(x[0].name_+str(x[0].nr_)+str(x[1]),\
                                                        y[0].name_+str(y[0].nr_)+str(y[1])))
            for (m1, p1, m2, p2) in self.__connections:
                print("| " + m1.name_ + "_" + str(m1.nr_) + " (" + str(p1) + ")" + "   ->   " + m2.name_ + "_" + str(m2.nr_) + " (" + str(p2) + ")")
            print("+-------------------------------------------------------------+")
            print("")

    def __init__( self ):
        self.debug_printNetChange = False
        self.debug_printNetChange_includeTraceback = False

        self.knownModules_ = globalModules.getDict()
        self.actualModules_ = []

        """ Hosts are always associated with ip adresses ? """
        self.__knownHosts = []
        self.__knownHosts.append(globalHostInfo.getName())
        self.__defaultHost = globalHostInfo.getName()
        self.__connections = [] # use tuples of (m1,p1, m2,p2) as entries

        CoviseMsgLoop().register( FinishedAction() )
        CoviseMsgLoop().register( ConnectAction() )

    # add module and wait until an INIT msg comes back        
    def add(self,m,  host=""):
        if ( m.name_ in self.knownModules_ ):
            i = self.knownModules_[m.name_]
            #increment the number of the actual module
            self.knownModules_[m.name_] = i+1
            m.nr_ = str(i+1)
            #m.nr_ = "-1"
            self.actualModules_.append(m)
            
            """ start module on <host> if host is given """
            if host!="" and host in self.__knownHosts:
                m.host_ = host
            elif self.__defaultHost != self.__knownHosts[0]:
               m.host_ = self.__defaultHost
               
            iaMod = InitAction(m.name_)
            CoviseMsgLoop().register( iaMod )
            deMod = DescAction(m)
            CoviseMsgLoop().register( deMod )
            # send init msg to controller
            sendCtrlMsg(m.getInitStr())
            m.finishPorts()
            while not deMod.stop:
                time.sleep(0.08)
            m.nr_ = iaMod.modNr
            CoviseMsgLoop().unregister( iaMod )
            CoviseMsgLoop().unregister( deMod )
            m.createParamAction()
        else:    
            print("Scripting Interface: module not recognized by covise scripting! add")
            return

        self.printNet("Add: " + m.name_ + "_" + str(m.nr_))
# add module and wait until an INIT msg comes back        
    def addExisting(self,m, nr=1, host=""):
        if ( m.name_ in self.knownModules_ ):
            i = self.knownModules_[m.name_]
            #increment the number of the actual module
            self.knownModules_[m.name_] = i+1
            m.nr_ = str(nr)
            self.actualModules_.append(m)
            
            """ start module on <host> if host is given """
            if host!="" and host in self.__knownHosts:
                m.host_ = host
            elif self.__defaultHost != self.__knownHosts[0]:
               m.host_ = self.__defaultHost
            m.createParamAction()
        else:    
            print("Scripting Interface: module not recognized by covise scripting! addExisting")
            return

        self.printNet("Add: " + m.name_ + "_" + str(m.nr_))

    #remove module from net
    def remove(self,m):
        if ( not  m.name_ in self.knownModules_ ):
            print("Scripting Interface: module not recognized by covise scripting! remove")
            return
        if ( m not in self.actualModules_ ):
            print("Scripting Interface: module not recognized by current covise-net")
            print("                     probably it was never added or already removed!")
            return
        sendCtrlMsg( m.getDelStr() )
        self.actualModules_.remove( m )
        # this does not remove the module globally !!
        self.printNet("Remove: " + m.name_ + "_" + str(m.nr_))
        del m
        

    # save a net        
    def show(self):
    #=================   
        for m in self.actualModules_:
            print(m.name_ + "_" + m.nr_)
        

    # save a net        
    def save(self,fn):
    #=================    
        cn = "\n"    
        execStr = "SAVE\n" + fn +cn
        sendCtrlMsg( execStr )


    # connect two modules
    def connect(self, module1, port1, module2, port2):
    #=================================================    
        cn = "\n"
        # no longer needed
        #con = module1.getCoObjName(port1)
        #if (con == ""):
        #    print("error: connection not possible")
        #    print(port1, " is not a output-port")
        #    return
        
        # send OBJCONN msg
        execStr = "OBJCONN\n" +\
                  module1.name_ + cn + module1.nr_ + cn + module1.host_ + cn + port1 + cn +\
                  module2.name_ + cn + module2.nr_ + cn + module2.host_ + cn + port2 + cn     
        ConnectAction().reset()
        sendCtrlMsg( execStr )
        while not ConnectAction().stop:
            time.sleep(0.01)
                    
        self.__connections.append((module1, port1, module2, port2))
        self.printNet("Connect: " + module1.name_ + "_" + str(module1.nr_) + " (" + str(port1) + ")" + "   ->   " + module2.name_ + "_" + str(module2.nr_) + " (" + str(port2) + ")")

    def getAllConnectionsFromModulePort(self, module, port):
        allConnections = []
        for (m1, p1, m2, p2) in self.__connections:
            if (m1, p1) == (module, port):
                allConnections.append((m1, p1))
            if (m2, p2) == (module, port):
                allConnections.append((m2, p2))
        return allConnections


    def disconnectAllFromModule(self, module):
        for (m1, p1, m2, p2) in list(self.__connections):
            if m1 == module or m2 == module:
                self.disconnect(m1, p1, m2, p2)

    def disconnectAllFromModulePort(self, module, port):
        for (m1, p1, m2, p2) in list(self.__connections):
            if (m1, p1) == (module, port) or (m2, p2) == (module, port):
                self.disconnect(m1, p1, m2, p2)

    def disconnectModuleModule(self, module1, module2):
        for (m1, p1, m2, p2) in list(self.__connections):
            if (m1, m2) == (module1, module2) or (m2, m1) == (module1, module2):
                self.disconnect(m1, p1, m2, p2)

    def disconnect(self, module1, port1, module2, port2):

        """Disconnect two Covise-modules."""

        if not (module1, port1, module2, port2) in self.__connections:
            return

        cn = "\n"
        con = module1.getCoObjName(port1)
        if con == "":
            print("error: disconnection not possible")
            print(port1, " is not a output-port")
            return
        execStr = "DIDEL\n" +                  \
                  module2.name_ + cn +          \
                  "%s" % (module2.nr_) + cn +   \
                  module2.host_ + cn +          \
                  port2 + cn +                  \
                  con + cn
        sendCtrlMsg(execStr)
        execStr = "DELETE_LINK\n" +                 \
                  module1.name_ + cn +          \
                  "%s" % (module1.nr_) + cn +   \
                  module1.host_ + cn +          \
                  port1 + cn +                  \
                  module2.name_ + cn +          \
                  "%s" % (module2.nr_) + cn +   \
                  module2.host_ + cn +          \
                  port2 + cn
        sendCtrlMsg(execStr)

        self.__connections.remove((module1, port1, module2, port2))

        self.printNet("Disconnect: " + module1.name_ + "_" + str(module1.nr_) + " (" + str(port1) + ")" + "   ->   " + module2.name_ + "_" + str(module2.nr_) + " (" + str(port2) + ")")

    def finishedBarrier(self):
        FinishedAction().reset()
        died=0
        while not (FinishedAction().isStopped()==True):
            time.sleep(0.01)
        return (died==0)
                    
    def setDefaultHost( self, hostip ):
        if hostip in self.__knownHosts:                            
            self.__defaultHost = hostip
    
    def getLocalHost( self ):
        return  self.__knownHosts[0]
               
    def moveToHost(self, module, hostip):

        """Move a module to different host"""
        if module not in self.__actualModules:
            print("Scripting Interface moveToHost: module not recognized by current covise-net")
            print("                     probably it was never added or already removed!")
            return
        if hostip not in self.__knownHosts:
            print("Host %s is not activated yet. Use addHost first. " % hostip)
            return
                
        module.moveToHost(hostip)   
        
        for (m1, p1, m2, p2) in self.__connections:
            if m1 == module and m1.host_ != module.host_:
                m1.host_ = module.host_
            if m2 == module and m2.host_ != module.host_:
                m2.host_ = module.host_

class _FinishedAction(CoviseMsgLoopAction):

    UI=6
    instance_ = None
    
    def __init__(self):
        CoviseMsgLoopAction.__init__( self, "FinishedACTION", self.UI, "action to react on Finished msg of modules" )
        self.stop = False
    
    def reset(self):
        self.stop = False
                
        
    def isStopped(self):
        return self.stop
        
    def run(self, param):
        if 'FINISHED'==param[0]:
            self.stop=True
        else : raise nameError
        
def FinishedAction():
    if _FinishedAction.instance_ == None:
        _FinishedAction.instance_ = _FinishedAction()
    return _FinishedAction.instance_
    
class _ConnectAction(CoviseMsgLoopAction):

    UI=6
    instance_ = None
    
    def __init__(self):
        CoviseMsgLoopAction.__init__( self, "ConnectACTION", self.UI, "action to react on Connect msg of modules" )
        self.stop = False
    
    def reset(self):
        self.stop = False
                
    def run(self, param):
        if param[0]=="PARCONN" or param[0]=="OBJCONN":
            self.stop=True
        else : raise nameError
        
def ConnectAction():
    if _ConnectAction.instance_ == None:
        _ConnectAction.instance_ = _ConnectAction()
    return _ConnectAction.instance_
    
class InitAction(CoviseMsgLoopAction):

    UI=6
    def __init__(self, modName):
        CoviseMsgLoopAction.__init__( self, "InitACTION for "+ modName, self.UI, "action to react on Init msg of modules" )
        self.stop = False
        self.modName = modName
        self.modNr = None
        
    def run(self, param):
        if 'INIT'==param[0] and self.modName==param[1]:
            self.modNr = param[2]
            self.stop=True
        else : raise nameError

# action to parse the DESC message of module. The message contains all default parameters set by the module
# currently only the colormap choice parameter is parsed
class DescAction(CoviseMsgLoopAction):

    UI=6
    def __init__(self, mod):
        CoviseMsgLoopAction.__init__( self, "DescACTION for "+ mod.name_, self.UI, "action to react on Init msg of modules" )
        self.stop = False
        self.mod = mod
        
    def run(self, param):
        if 'DESC'==param[0] and self.mod.name_==param[1]:
            for i in range(len(param)-1):
                if param[i]=="IMM":
                    if i+1<len(param)-1 and i+4<len(param)-1 : self.mod.setParamValueWithoutMsg(param[i+1], param[i+4] )
            self.stop=True
        else : raise nameError
