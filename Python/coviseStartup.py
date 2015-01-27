##########################################################################################
#
# Python-Module: coviseStartup.py
#
# Description: python module defining receiver thread internal startup and exit functions
#
# IMPORTANT WARNING:  N E V E R    F O R M A T    T H I S    F I L E !!!!              
# =======================================================================
# ( if you don't know what youre' doing )
#
# (C) 2002/2003 VirCinity GmbH, Stuttgart   info@vircinity.com
#
# 14.03.2003 [RM]
#
##########################################################################################

from math import *
from threading import *
from coPyModules import *
from CoviseMsgLoop import CoviseMsgLoop, CoviseMsgLoopAction
from coviseModuleBase import *

import time

#--------
udpMode = False
try:
    import covise
    from PyQt5.QtNetwork import QHostAddress, QUdpSocket 

    # Set the socket parameters
    secondUDP_IP = covise.getCoConfigEntry("vr-prepare.SecondUDP.IP")
    if secondUDP_IP:
        host = QHostAddress(secondUDP_IP)
    else:
        host = QHostAddress("127.0.0.1")
    secondUDP_Port = covise.getCoConfigEntry("vr-prepare.SecondUDP.Port")
    if secondUDP_Port:
        port = int(secondUDP_Port)
    else:
        port = 6666
    buf = 1024
    addr = (host,port)

    # Create socket and bind to address
    if covise.coConfigIsOn("vr-prepare.SecondUDP"):
        UDPSock = QUdpSocket() 
        UDPSock.bind(port)
        udpMode = True
except:
    pass
#--------


#
# receiver thread filling up the global message queue
#
class Rec(Thread):
#================
    lastInfo_ = ""
    
    def __init__(self,queue):        
        Thread.__init__(self, name="CoviseRec")
        self.quit = 1
        self.numUi = 0
        self.queue = queue
        self.infoQueue_ = BoundedStack(50)
        self.choiceParamEvent_ = Event()
        self.infoEvent_ = Event()

    def waitForParam(self,time=None):
        self.choiceParamEvent_.wait(time)
        self.choiceParamEvent_.clear()


    def waitForInfo(self,time=None):
        self.infoEvent_.wait(time)
        self.infoEvent_.clear()

                            
    def run(self):
        cnt=0;
        while (self.quit>0) :
            if udpMode:
                if UDPSock.hasPendingDatagrams():
                    (data, sender, sendport) = UDPSock.readDatagram(buf)
                    if data:     
                        CoviseMsgLoop().run(1000, data.split())         
        
            msg=CoMsg(-1,"")
            msg = getSingleMsg()
            type = msg.type;
            data = msg.data
            if ( type > 0 ):
                #print(" received message ", type)
                #print(data)
                
                dl = data.split('\n')
           
                if '.exe' in dl[0]:
                        dl[0]  = dl[0][0:dl[0].find('.exe') ]
                
                # QUIT message
                if ( type == 34 ):
                    self.quit=0

                # we must be sure that nothing kills us especially indexError exceptions
                try:
                    if ( type == 56 ):
                        # print("INFO: ", dl[0]+dl[1] + "@" + dl[2], "  ", dl[3])
                        x = dl[3]
                        self.infoQueue_.put(x);
                        self.infoEvent_.set()
                        
                    if ( type == 47 ):
                        #print("<<<< type 47", dl[0])
                        GlobalChoiceDict().set( dl )
                        self.choiceParamEvent_.set()
                           
                except IndexError:
                            print("Receiver index error ", dl )
                except KeyError:
                    print("Rec.run(): Receiver key error ", dl)
                except MemoryError as val:
                    print("Rec.run(): Receiver memory error xxx", dl)
                    print("VALUE: ", str(val))
                except NameError as val:
                    print("Rec.run(): Receiver name error ", dl)
                    print("VALUE: ", str(val)                            )

                CoviseMsgLoop().run(type, dl)
                    # message was handled by one of the registered actions
                    # do not put it into the queue
                    #if (len(dl) > 1):                    
                    #    outStr = "Rec:  got Msg with type: %s " % (type)
                    #    for xx in dl:
                    #        outStr = outStr + "  " + str(xx)
                    #    print(outStr )
                    #self.queue.put(data)
                dl=None 
            time.sleep(0.0001)


globalReceiverThread = Rec( globalMessageQueue )


def getLastInfo():
    a=[]
#    for i in range(0,maxLen):
    x= globalReceiverThread.infoQueue_.get()
    while ( x != None ):
        a.append( x )
        x= globalReceiverThread.infoQueue_.get()
    
    return a


#
# obtain all available Modules from controller
# ... and hope that we really have a python representation
#     for all of them...
#
def getListOfAllModules():
    import time
    while not ListAction().stop:
        time.sleep(0.1)

#
# Action to print warnings
#
class WarnAct( CoviseMsgLoopAction ):
    def __init__(self):
        CoviseMsgLoopAction.__init__(self, "Warning", 55, "action to print warnings to stdout" )

    def run(self, param):
        print("WARNING")
        idx=len(param)-1
        print(param[idx])

#
# Action to print errors
#
class ErrorAct( CoviseMsgLoopAction ):
    def __init__(self):
        CoviseMsgLoopAction.__init__(self, "Error", 36, "action to print warnings to stdout" )

    def run(self, param):
        print("COVISE-ERROR: ", param[0]+param[1]+"@"+param[2],"  ",param[3])

class _ListAction(CoviseMsgLoopAction):

    UI=6
    instance_ = None
    
    def __init__(self):
        CoviseMsgLoopAction.__init__( self, "LISTACTION", self.UI, "action to react on the list of modules" )
        self.stop = False
         
    def run(self, param):
        if 'LIST'==param[0]:
            ii=4
            while ( ii < len(param)-2 ):
                modName = param[ii]
                if (len(modName) > 0):
                    if (modName[0] != '.'):
                        globalModules.add(modName)
                ii=ii+2
            self.stop=True
        else : raise nameError
        
def ListAction():
    if _ListAction.instance_ == None:
        _ListAction.instance_ = _ListAction()
    return _ListAction.instance_
    

def coviseExitFunc():
    #===============
    clean()
    quit()
    print(" COVISE PYTHON INTERFACE finished")
    globalReceiverThread .join();



def coviseStartupFunc():
    #===================
    sys.exitfunc = coviseExitFunc
    CoviseMsgLoop().register( WarnAct() )
    CoviseMsgLoop().register( ErrorAct() )
    CoviseMsgLoop().register( ListAction() )
    globalReceiverThread .start();
    time.sleep(0.1)
    getListOfAllModules()
    print("  COVISE PYTHON INTERFACE ready\n")
