#-----------------------------------------------------------------------------
# Name:        SharedCovise.py
# Purpose:     This is the COVISE Accessgrid integration. 
#
# Author:      Uwe Woessner
#
# Created:     2003/10/01
# RCS-ID:      $Id: SharedCovise.py,v 1.10 2003/09/16 17:18:22 turam Exp $
# Copyright:   (c) 2003
# Licence:     See COPYING.TXT
#-----------------------------------------------------------------------------
# Normal import stuff
import os
import sys
import getopt
import logging
from threading import Thread
import Queue
import shutil
import string
import socket

from wxPython.wx import *


# Imports we need from the Access Grid Module
from AccessGrid.hosting.pyGlobus import Client
from AccessGrid.EventClient import EventClient
from AccessGrid.Events import ConnectEvent, Event
from AccessGrid import Platform
from AccessGrid import DataStore


# This function registers the application with the users environment
# There is no longer support for shared application installations
# that might be reintroduced later.

def registerApp(fileName):
    import AccessGrid.Toolkit as Toolkit
    app = Toolkit.CmdlineApplication()
    appdb = app.GetAppDatabase()

    fn = os.path.basename(fileName)

    exeCmd = sys.executable + " " + fn + " -a %(appUrl)s"

    # Call the registration method on the applications database
    #appdb.RegisterApplication("Shared Covise",
    #                          "application/x-ag-shared-covise",
    #                          "sharedcovise",
    #                          {"Open" : exeCmd },
    #                          [fn], os.getcwd())
    # old init have to be changed soon    
    uad = Platform.GetUserAppPath()
    src = os.path.abspath(fn)
    dest = os.path.join(uad,fn)

    exeCmd = sys.executable + " \"" + dest + "\" -a %(appUrl)s"
    print("hallo" + src +" to "  +  dest)
    try:
        shutil.copyfile(src, dest)
    except IOError:
        print "could not copy app into place"
        sys.exit(1)

    # Call the registration method on the applications database
    appdb.RegisterApplication("Shared Covise",
                              "application/x-ag-covise",
                              "sharedcovise",
                              {"Open" : exeCmd })

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# GUI code
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class SharedCoviseFrame(wxFrame):

    ID_JOIN = wxNewId()
    ID_EXIT = wxNewId()

    def __init__(self, parent, ID, title,covise):
        wxFrame.__init__(self, parent, ID, title,
                         wxDefaultPosition, wxSize(450, 300))


        #
        # Create UI controls
        #
        
        # - Create menu bar
        self.covise=covise
        menubar = wxMenuBar()
        fileMenu = wxMenu()
        fileMenu.Append(self.ID_JOIN,"&Join", "Join Covise session")
        fileMenu.AppendSeparator()
        fileMenu.Append(self.ID_EXIT,"&Exit", "Exit")
     	menubar.Append(fileMenu, "&File")
        self.SetMenuBar(menubar)

        # - Create main sizer
        sizer = wxBoxSizer(wxVERTICAL)
        self.SetSizer(sizer)

        # - Create sizer for remaining ctrls
        staticBoxSizer = wxStaticBoxSizer(wxStaticBox(self, -1, ""), wxVERTICAL)
        gridSizer = wxFlexGridSizer(4, 2, 5, 5)
        gridSizer.AddGrowableCol(1)
        staticBoxSizer.Add(gridSizer, 1, wxEXPAND)
        sizer.Add(staticBoxSizer, 1, wxEXPAND)

        # - Create textctrl for vis host
        staticText = wxStaticText(self, -1, "Visualization Host")
        self.visHostText = wxTextCtrl(self,-1)
        gridSizer.Add( staticText, 0, wxALIGN_LEFT)
        gridSizer.Add( self.visHostText, 1, wxEXPAND)

        # - Create textctrl for daemon port
        staticText = wxStaticText(self, -1, "Daemon Port")
        self.daemonPortText = wxTextCtrl(self,-1)
        gridSizer.Add( staticText, wxALIGN_LEFT)
        gridSizer.Add( self.daemonPortText )
        
        # - Create textctrl for master host
        staticText = wxStaticText(self, -1, "Master Host: ")
        self.masterHostText = wxStaticText(self, -1, "")
        gridSizer.Add( staticText,0, wxALIGN_LEFT)
        gridSizer.Add( self.masterHostText,0, wxALIGN_LEFT)

        # - Create buttons for control 
        rowSizer = wxBoxSizer(wxHORIZONTAL)
        self.joinButton = wxButton(self,-1,"Join")
        self.cleanButton = wxButton(self,-1,"Clean")
        self.quitButton = wxButton(self,-1,"Quit")
        rowSizer.Add( self.joinButton )
        rowSizer.Add( self.cleanButton )
        rowSizer.Add( self.quitButton )
        gridSizer.Add( wxStaticText(self,-1,"") )
        gridSizer.Add( rowSizer, 0, wxALIGN_RIGHT )
        
        # Set up event callbacks
        EVT_TEXT_ENTER(self, self.visHostText.GetId(), self.HostCB)
        EVT_TEXT_ENTER(self, self.daemonPortText.GetId(), self.PortCB)
        EVT_BUTTON(self, self.joinButton.GetId(), self.JoinCB)
        EVT_BUTTON(self, self.cleanButton.GetId(), self.CleanCoviseCB)
        EVT_BUTTON(self, self.quitButton.GetId(), self.ExitCB)

        EVT_MENU(self, self.ID_JOIN, self.JoinCB)
        EVT_MENU(self, self.ID_EXIT, self.ExitCB)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # Callback stubs for the UI
    #

    def PortCB(self,event):
        """
        Callback for "enter" presses in the port text field
        """
        self.checkFields();


    def HostCB(self,event):
        """
        Callback for "enter" presses in the slide URL text field
        """
        self.checkFields();
        
    def checkFields(self):
        
        daemonPort = int(self.daemonPortText.GetValue())
        daemonHost = self.visHostText.GetValue()
        if self.covise.serverPort != daemonPort:
            self.covise.serverPort = daemonPort
            self.covise.saveConfig()
        if self.covise.hostName != daemonHost:
            self.covise.hostName = daemonHost
            self.covise.saveConfig()
        #
        
    def JoinCB(self,event):
        """
        Callback for "previous" button
        """
        self.checkFields();
        self.covise.DoJoin();
        
    def ExitCB(self,event):
        """
        Callback for "exit" menu item
        """
        self.checkFields();
        self.Close();
        
    def CleanCoviseCB(self,event):
        """
        Callback for "clean_covise" menu item
        """
        self.checkFields();
        self.covise.eventClient.Send(Event(SharedCovEvent.CLEAN, self.covise.channelId,(self.covise.publicId, self.covise.publicId)))

        
    def SetPort(self, port):
        """
        This method is used to set the slide number
        """
        self.daemonPortText.SetValue('%s' % port)

    def SetHost(self, name):
        """
        This method is used to set the slide URL
        """
        self.visHostText.SetValue(name)
        
    def SetMasterHost(self, name):
        """
        This method is used to set the slide URL
        """
        if name != ":0":
            self.masterHostText.SetLabel(name)
        else:
            self.masterHostText.SetLabel("")



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Shared covise constants classes
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class SharedCovEvent:
    QUIT = "quit"
    MASTER = "master"
    CLEAN = "clean"

class SharedCovKey:
    MASTERHOST = "masterHost"
    MASTERPORT = "masterPort"
    MASTERID = "masterID"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Shared Covise class itself
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class SharedCovise(wxApp):
    """
    The SharedCovise is an Access Grid 2 Application.
    """
    appName = "Shared Covise"
    appDescription = "A shared covise is a set of slides that someone presents to share an idea, plan, or activity with a group."
    appMimetype = "application/x-ag-shared-covise"
    
    def __init__(self,arg,url, log=None):
        wxApp.__init__(self,arg)
        """
        This is the constructor for the Shared Covise.
        """
        # Initialize state in the shared covise
        self.url = url
        self.eventQueue = Queue.Queue(5)
        self.log = log
        self.masterId = None
        self.numSteps = 0
        self.hostName = "vision.rus.uni-stuttgart.de"
        self.serverPort = 31098
        self.masterSocket = None

        # Get a handle to the application object in the venue
        self.log.debug("Getting application proxy (%s).", url)
        self.appProxy = Client.Handle(self.url).GetProxy()

        # Join the application object, get a private ID in response
        self.log.debug("Joining application.")
        (self.publicId, self.privateId) = self.appProxy.Join()

        # Get the information about our Data Channel
        self.log.debug("Retrieving data channel information.")
        (self.channelId, esl) = self.appProxy.GetDataChannel(self.privateId)

        # Connect to the Data Channel, using the EventClient class
        # The EventClient class is a general Event Channel Client, but since
        # we use the Event Channel for a Data Channel we can use the
        # Event Client Class as a Data Client. For more information on the
        # Event Channel look in AccessGrid\EventService.py
        # and AccessGrid\EventClient.py
        self.log.debug("Connecting to data channel.")
        self.eventClient = EventClient(self.privateId, esl, self.channelId)
        self.eventClient.start()
        self.eventClient.Send(ConnectEvent(self.channelId, self.privateId))

        # Register callbacks with the Data Channel to handle incoming
        # events.
        self.log.debug("Registering for events.")
        self.eventClient.RegisterCallback(SharedCovEvent.MASTER, self.RecvMaster)
        self.eventClient.RegisterCallback(SharedCovEvent.CLEAN, self.CleanCovise)


        self.masterHost = ""
        self.masterPort = 0
        self.masterHost = self.appProxy.GetData(self.privateId,SharedCovKey.MASTERHOST)
        self.masterPort = self.appProxy.GetData(self.privateId,SharedCovKey.MASTERPORT)
        self.frame.SetMasterHost(self.masterHost+":"+str(self.masterPort))

        self.log.debug("fit.")

        self.log.debug("SharedCovise V1.0")
        
        # read configueation file
        userapp = Platform.GetUserAppPath()
        configFileName = userapp+"\\SharedCovise.config"
        configFile = None
        try:
            configFile = file(configFileName,'r')
        except:
            print "could no open config file"
        if configFile != None:
            entry = configFile.readline(200)
            if entry != "":
                self.hostName = entry.rstrip("\n")
            entry = configFile.readline(200)
            if entry != "":
                self.serverPort = int(entry)
            
        self.frame.SetPort(self.serverPort)
        self.frame.SetHost(self.hostName)
        
        if self.masterHost == self.hostName and self.serverPort == self.serverPort:
            #connect to our Daemon and find out if covise is still running
            message = "check"
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((HOST, PORT))
                s.send(message)
                self.masterSocket = s;
                self.appProxy.SetData(self.privateId,SharedCovKey.MASTERHOST,self.hostName)
                self.appProxy.SetData(self.privateId,SharedCovKey.MASTERPORT,self.serverPort)
                self.eventClient.Send(Event(SharedCovEvent.MASTER, self.channelId,(self.publicId, self.publicId)))
                Thread(target=self.monitorMaster).start()
            except:
                self.appProxy.SetData(self.privateId,SharedCovKey.MASTERHOST,"")
                self.appProxy.SetData(self.privateId,SharedCovKey.MASTERPORT,0)
                self.eventClient.Send(Event(SharedCovEvent.MASTER, self.channelId,(self.publicId, self.publicId)))
                print "could not connect to AccessGridDaemon\n"

        

    def OnInit(self):
        # Create the GUI

        self.frame = SharedCoviseFrame(NULL, -1, "Shared Covise",self)
        self.frame.Fit()
        self.frame.Show(true)
        self.SetTopWindow(self.frame)


        return true
    
    def Start(self):
        """
        Start the UI app loop
        """
        self.MainLoop()
        
        # When the quit event gets processed, the running flag gets cleared
        self.log.debug("Shutting down...")
        self.Quit()



    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # Methods registered as callbacks with the UI
    # These methods typically put an event in the event queue.
    #

    def DoJoin(self):
        """
        This method handles NEXT button presses in the UI.
        The event is only sent if the local user is the "master"
        """

        
        
        self.log.debug("Method DoJoin called")
        # Put the event on the queue
        if self.masterHost == "":
            #become master and start covise
            #tell the master which client wants to join
            HOST = self.hostName    # Our AccessGrid Daemon
            PORT = self.serverPort  # Our port number
            message = "startCovise\n"
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((HOST, PORT))
                s.send(message)
                self.masterSocket = s;
                self.appProxy.SetData(self.privateId,SharedCovKey.MASTERHOST,self.hostName)
                self.appProxy.SetData(self.privateId,SharedCovKey.MASTERPORT,self.serverPort)
                self.eventClient.Send(Event(SharedCovEvent.MASTER, self.channelId,(self.publicId, self.publicId)))
                Thread(target=self.monitorMaster).start()
            except:
                print "could not connect to AccessGridDaemon\n"
            ##data = s.recv(1024)
        else:
            #tell the master which client wants to join
            HOST = self.masterHost  # The master AccessGrid Daemon
            PORT = self.masterPort  # The master Port
            message = "join "+self.hostName+":"+str(self.serverPort)+"\n"
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((HOST, PORT))
                s.send(message)
            except:
                print "could not connect to masters AccessGridDaemon\n"
            ##data = s.recv(1024)
            s.close()
            #print 'Received', `data`

        #print "join done\n"
    def monitorMaster(self):
        data = self.masterSocket.recv(1024)
        while data != "" and data.find("masterLeft")<0:
            #print "XX"+data+"XX"+str(data.find("masterLeft"))+"\n"
            data = self.masterSocket.recv(1024)
        
        #print "XX"+data+"XX\n"
        self.appProxy.SetData(self.privateId,SharedCovKey.MASTERHOST,"")
        self.appProxy.SetData(self.privateId,SharedCovKey.MASTERPORT,0)
        self.eventClient.Send(Event(SharedCovEvent.MASTER, self.channelId,(self.publicId, self.publicId)))
        self.masterSocket.close();
        #print "Master Covise left\n"
        self.masterSocket = None


    def CleanCovise(self,event):        
        self.log.debug("Method CleanCovise called")
        # Put the event on the queue
        HOST = self.hostName    # Our AccessGrid Daemon
        PORT = self.serverPort  # Our port number
        message = "cleanCovise\n"
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((HOST, PORT))
            s.send(message)
            s.close()
        except:
            print "could not connect to AccessGridDaemon\n"
            ##data = s.recv(1024)
            
    def monitorMaster(self):
        data = self.masterSocket.recv(1024)
        while data != "" and data.find("masterLeft")<0:
            #print "XX"+data+"XX"+str(data.find("masterLeft"))+"\n"
            data = self.masterSocket.recv(1024)
        
        #print "XX"+data+"XX\n"
        self.appProxy.SetData(self.privateId,SharedCovKey.MASTERHOST,"")
        self.appProxy.SetData(self.privateId,SharedCovKey.MASTERPORT,0)
        self.eventClient.Send(Event(SharedCovEvent.MASTER, self.channelId,(self.publicId, self.publicId)))
        self.masterSocket.close();
        #print "Master Covise left\n"
        self.masterSocket = None        
        
    def saveConfig(self):
        
        userapp = Platform.GetUserAppPath()
        configFileName = userapp+"\\SharedCovise.config"
        configFile = None
        try:
            configFile = file(configFileName,'w')
        except:
            print "could no save config file"
        if configFile != None:
            configFile.write(self.hostName+"\n")
            configFile.write(str(self.serverPort)+"\n")



    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # Methods registered as callbacks with EventClient
    # These methods typically put an event in the event queue.
    #

    def RecvMaster(self, event):
        """
        This callback puts a "master" event from the network
        on the event queue
        """
        
        self.log.debug("Method RecvMaster called")
        self.masterHost = ""
        self.masterPort = 0
        self.masterHost = self.appProxy.GetData(self.privateId,SharedCovKey.MASTERHOST)
        self.masterPort = self.appProxy.GetData(self.privateId,SharedCovKey.MASTERPORT)
        self.frame.SetMasterHost(self.masterHost+":"+str(self.masterPort))

        


    def Quit(self, data=None):
        """
        This is the _real_ Quit method that tells the viewer to quit
        """
        # Turn off the main loop
        self.log.debug("Method Quit called")
        if self.masterSocket != None:
            self.masterSocket.close()
        if self.masterHost == self.hostName:
            self.appProxy.SetData(self.privateId,SharedCovKey.MASTERHOST,"")
            self.appProxy.SetData(self.privateId,SharedCovKey.MASTERPORT,0)
            self.eventClient.Send(Event(SharedCovEvent.MASTER, self.channelId,(self.publicId, self.publicId)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Utility functions
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#
# This gets logging started given the log name passed in
# For more information about the logging module, check out:
# http://www.red-dove.com/python_logging.html
#
def InitLogging(appName, debug=0):
    """
    This method sets up logging so you can see what's happening.
    If you want to see more logging information use the appName 'AG',
    then you'll see logging information from the Access Grid Module.
    """
    logFormat = "%(name)-17s %(asctime)s %(levelname)-5s %(message)s"


    # Set up a venue client log, too, since it's used by the event client
    log = logging.getLogger("AG.VenueClient")
    log.setLevel(logging.DEBUG)
    hdlr = logging.StreamHandler()
    hdlr.setFormatter(logging.Formatter(logFormat))
    log.addHandler(hdlr)

    log = logging.getLogger(appName)
    log.setLevel(logging.DEBUG)


    # Log to file
    logFile = appName + ".log"
    fileHandler = logging.FileHandler(logFile)
    fileHandler.setFormatter(logging.Formatter(logFormat))
    log.addHandler(fileHandler)

    # If debugging, log to command window too
    if debug:
        hdlr = logging.StreamHandler()
        hdlr.setFormatter(logging.Formatter(logFormat))
        log.addHandler(hdlr)
    return log

def Usage():
    """
    Standard usage information for users.
    """
    print "%s:" % sys.argv[0]
    print "    -a|--applicationURL : <url to application in venue>"
    print "    -d|--data : <url to data in venue>"
    print "    -h|--help : print usage"
    print "    -r|--register : register application with the Accessgrid Client"
    print "    -i|--information : <print information about this application>"
    print "    -l|--logging : <log name: defaults to SharedCovise>"
    print "    --debug : print debugging output"



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# MAIN block
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
    # Initialization of variables
    venueURL = None
    appURL = None
    venueDataUrl = None
    logName = "SharedCovise"
    debug = 0

    # Here we parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:v:a:l:ihr",
                                   ["venueURL=", "applicationURL=",
                                    "information=", "logging=", 
                                    "data=", "debug", "help", "register"])
    except getopt.GetoptError:
        Usage()
        sys.exit(2)

    for o, a in opts:
        if o in ("-v", "--venueURL"):
            venueURL = a
        elif o in ("-a", "--applicationURL"):
            appURL = a
        elif o in ("-l", "--logging"):
            logName = a
        elif o in ("-i", "--information"):
            print "App Name: %s" % SharedCovise.appName
            print "App Description: %s" % SharedCovise.appDescription
            print "App Mimetype: %s" % SharedCovise.appMimetype
            sys.exit(0)
        elif o in ("-d", "--data"):
            venueDataUrl = a
        elif o in ("--debug",):
            debug = 1
        elif o in ("-r", "--register"):
            registerApp(sys.argv[0])
            sys.exit(0)
        elif o in ("-h", "--help"):
            Usage()
            sys.exit(0)
   
    # Initialize logging
    #debug=1
    log = InitLogging(logName, debug)
    # If we're not passed some url that we can use, bail showing usage
    if appURL == None and venueURL == None:
        Usage()
        sys.exit(0)

    # If we got a venueURL and not an applicationURL
    # This is only in the example code. When Applications are integrated
    # with the Venue Client, the application will only be started with
    # some applicatoin URL (it will never know about the Venue URL)
    if appURL == None and venueURL != None:
        venueProxy = Client.Handle(venueURL).get_proxy()
        appURL = venueProxy.CreateApplication(SharedCovise.appName,
                                              SharedCovise.appDescription,
                                              SharedCovise.appMimetype)
        log.debug("Application URL: %s", appURL)

    # This is all that really matters!
    covise = SharedCovise(0,appURL, log)

    covise.Start()

    # This is needed because COM shutdown isn't clean yet.
    # This should be something like:
    #sys.exit(0)
    os._exit(0)

