#!/usr/bin/python2.7

# -----------------------------------------------------------------------------
# Description:
#
# -----------------------------------------------------------------------------
# Dependencies:
#     python   == 2.7.3
#     wxPython >= 2.8.11.0
#     pyCrypto >= 2.6.1
#     paramiko >= 1.7.7.1
#
# -----------------------------------------------------------------------------
# TODO:
# - print "please wait x s" to status bar, so that the user knows that he was
#   not allowed to send commands to the projectors
# - use backport of python3.4 enum
# - remove global stuff 
# -----------------------------------------------------------------------------

import wx
import functools
import threading
import subprocess
import time
import xml.etree.ElementTree as ET

#wxPython < 2.8.11.0
#from wx.lib.pubsub import Publisher as publish

#wxPython >= 2.8.11.0
from wx.lib.pubsub import setupkwargs
from wx.lib.pubsub import pub as publish

# ssh
import paramiko
import cmd

# sasuctrl
import socket
import sys
import getopt

# =============================================================================
# definitions
# =============================================================================

# -----------------------------------------------------------------------------
# global stuff
# will be removed ... someday
# -----------------------------------------------------------------------------
CMD_TIME = [0.0, 0.0, 0.0, 0.0, 0.0]
CMD_WAIT = [0.0, 0.0, 0.0, 0.0, 0.0]

# =============================================================================
# helper functions
# =============================================================================

# -----------------------------------------------------------------------------
# X_is_running()
# if we are running linux, this methods checks, if there is an X running or not
# -----------------------------------------------------------------------------
def X_is_running():
    from subprocess import Popen, PIPE

    p = Popen(["xset", "-q"], stdout=PIPE, stderr=PIPE)
    p.communicate()
    return p.returncode == 0

# -----------------------------------------------------------------------------
# enum()
# python 2.7 doesn't support enum, so we have do define something similar
# but with python 3.4 there is a backport available, we should use it someday
# -----------------------------------------------------------------------------
def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    reverse = dict((value, key) for key, value in enums.iteritems())
    enums['reverse_mapping'] = reverse
    return type('Enum', (), enums)

e_os   = enum('unknown', 'linux', 'windows')
e_env  = enum('unknown', 'desktop', 'embedded')
e_xenv = enum('unknown', 'yes', 'no')

# =============================================================================
# exceptions
# =============================================================================

# -----------------------------------------------------------------------------
# TimeLimitException()
# used as a time limit within we are not allowed to send further commands to
# the f35 projectors
# -----------------------------------------------------------------------------
class  TimeLimitException(Exception):
    def __init__(self, value):
        self.value = value
        
    def __str__(self):
        return repr(self.value)

# =============================================================================
# sasuctrl
# =============================================================================

# -----------------------------------------------------------------------------
# sasuctrl()
# send a udp diagram with command to UDP_IP
# commands the rpi knows: on off 3d_on 3d_off
# -----------------------------------------------------------------------------
def sasuctrl(command):
    global config

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(command, (config.tidyip, int(config.tidyport)))

# -----------------------------------------------------------------------------
# f35ctrl()
# start a tcp session with TCP_IP, transmit a command and close it again
# commands the rpi knows: on off 3d_on 3d_off
# -----------------------------------------------------------------------------
def f35ctrl(address, command, wait):
    global config

    if (time.time() > CMD_TIME[address] + CMD_WAIT[address]):

        TCP_IP = config.projectors[address]
        TCP_PORT = 1025
        BUFFER_SIZE = 1024
        MESSAGE = command

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((TCP_IP, TCP_PORT))
        s.send(MESSAGE)
        data = s.recv(BUFFER_SIZE)
        print "received data:", data

        CMD_TIME[address] = time.time()
        CMD_WAIT[address] = wait;
        s.close()
        return (0, data)

    else:
    
        return ((CMD_TIME[address] + CMD_WAIT[address]) - time.time(),"")

# =============================================================================
# structs
# =============================================================================

# -----------------------------------------------------------------------------
# Config()
# this struct holds config data
# -----------------------------------------------------------------------------
class Config():
    target = ''            # ip address or hostname of master to start
                           # if target == 'localhost' ? shell : ssh
    targetaccount = 'none' # login id for ssh
    xenv = e_xenv.unknown  # X or windows available yes or no
    env = e_env.unknown    # desktop or embedded, how large is the display
    os = e_os.unknown      # detect OS for execute commands properly
    tidyip = "127.0.0.1"   # ip address of raspberry pi controlling tidy
    tidyport = 5005        # ip address of raspberry pi controlling tidy
    projectors = []        # list of projectors hostname or ip

# -----------------------------------------------------------------------------
# Status()
# status of devices 
# -----------------------------------------------------------------------------
class Status():
    projectors = []        # status of projectors, 0-unknown
    
    def __init__(self): 
        for i in range(10):
            self.projectors.append(-1)


# =============================================================================
# classes
# =============================================================================
 
# -----------------------------------------------------------------------------
# InfoUpdate()
# this thread periodically updates the status information of the projectors
# -----------------------------------------------------------------------------
class InfoUpdate(threading.Thread): 

  def __init__(self): 
      threading.Thread.__init__(self) 
      #self.t = threading.Timer(60.0, self.run)
      #self.t.start()
 
  def run(self): 
      global root
      global config
      global status
   
      for i in range(len(config.projectors)):
          
          res = f35ctrl(i, ":LST1?\n\r", 0.1)
          res = res[1].strip()
          status.projectors[i*2] = ord(res[15]) - ord('0')
          time.sleep(0.1)
            
          res = f35ctrl(i, ":LST2?\n\r", 0.1)
          res = res[1].strip()
          status.projectors[i*2+1] = ord(res[15]) - ord('0')
          time.sleep(0.1)
          
      publish.sendMessage("PROJECTOR STATUS UPDATE")

# -----------------------------------------------------------------------------
# PageMaster()
# super class holding the more general methods like add buttons from xml file
# -----------------------------------------------------------------------------
class PageMaster(wx.Panel):
    def __init__(self, parent, type):

        wx.Panel.__init__(self, parent)

        # define sizer and layout
        self.sizer = wx.GridBagSizer(10, 10)
        self.SetSizerAndFit(self.sizer)
        self.SetAutoLayout(True)

        # now add all the buttons in lstNames
        posL = 1
        posR = 1
        pBreak = 4
        lstButtons = []

        for child in root.find(type):
            lstButtons.append(child.attrib['name'])

        for i in range(len(lstButtons)):

            # create and add the button
            self.btn = wx.Button(self, label=lstButtons[i])

            if lstButtons[i] in ['resync','framelock on','framelock off']:          #TODO enable these funcs
                self.btn.Disable()
            
            self.sizer.Add(self.btn, (posL, posR), flag=wx.EXPAND)

            # connect event to button
            func = functools.partial(self.on_button, button=(lstButtons[i]))
            self.btn.Bind(wx.EVT_BUTTON, func)

            if posL < pBreak:
                posL += 1
            else:
                posL = 1
                posR += 1

# -----------------------------------------------------------------------------
# PageFASI()
# add a panel for the driving simulator in parent notebook
# -----------------------------------------------------------------------------
class PageFASI(PageMaster):
    def __init__(self, parent):

        super(PageFASI, self).__init__(parent, 'FASI')

    def on_button(self, event, button):
        thread = threading.Thread(target=self.run, args=(button,))
        thread.setDaemon(True)
        thread.start()

    def run(self, button):
        global root
        global config

        for demo in root.getiterator('demo'):
            if (demo.get('name') == button):
                
                cmd = demo.find('command').text
                print "start: ", cmd

                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

                ssh.connect(config.target, username=config.targetaccount)
                stdin, stdout, stderr = ssh.exec_command(cmd)
                stdin.flush()
                data = stdout.read().splitlines()
                for line in data:
                    print line

                #proc = subprocess.Popen(cmd, shell=True, 
                #       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

# -----------------------------------------------------------------------------
# PageTIDY()
# add a panel for the tiled display in parent notebook
# -----------------------------------------------------------------------------
class PageTIDY(PageMaster):
    def __init__(self, parent):

        super(PageTIDY, self).__init__(parent, 'TIDY')

    def on_button(self, event, button):
        thread = threading.Thread(target=self.run, args=(button,))
        thread.setDaemon(True)
        thread.start()
        
    def run(self, button):
        global root
        global config

        for name in root.getiterator('demo'):
            if name.get('name') == button:
                cmd = name.find('command').text
                sasuctrl(cmd)

# -----------------------------------------------------------------------------
# PageCAVE()
# add a panel for the cave in parent notebook
# -----------------------------------------------------------------------------
class PageCAVE(PageMaster):
    def __init__(self, parent):

        super(PageCAVE, self).__init__(parent, 'CAVE')

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        publish.subscribe(self.Refresh, "PROJECTOR STATUS UPDATE")

    def drawStatusLED(self, dc, x, y, status):
        if status == 0:
            dc.SetPen(wx.Pen("red",style=wx.TRANSPARENT))
            dc.SetBrush(wx.Brush("red", wx.SOLID))
        elif status == 1:
            dc.SetPen(wx.Pen("magenta",style=wx.TRANSPARENT))
            dc.SetBrush(wx.Brush("magenta", wx.SOLID))
        elif status == 2:
            dc.SetPen(wx.Pen("green",style=wx.TRANSPARENT))
            dc.SetBrush(wx.Brush("green", wx.SOLID))
        elif status == 3:
            dc.SetPen(wx.Pen("black",style=wx.TRANSPARENT))
            dc.SetBrush(wx.Brush("black", wx.SOLID))
        elif status == 4:
            dc.SetPen(wx.Pen("blue",style=wx.TRANSPARENT))
            dc.SetBrush(wx.Brush("blue", wx.SOLID))
        elif status == 5:
            dc.SetPen(wx.Pen("white",style=wx.TRANSPARENT))
            dc.SetBrush(wx.Brush("white", wx.SOLID))
        else:
            dc.SetPen(wx.Pen("grey",style=wx.TRANSPARENT))
            dc.SetBrush(wx.Brush("grey", wx.SOLID))

        dc.DrawRectangle(x, y, 20, 20)

    def drawStatus(self):
        global status

        self.dc = wx.PaintDC(self)
        self.dc.Clear()
        self.dc.BeginDrawing()

        x = 330
        y = 75
        
        self.drawStatusLED(self.dc, x-70 , y, status.projectors[6])
        self.drawStatusLED(self.dc, x-40 , y, status.projectors[7])

        self.drawStatusLED(self.dc, x    , y, status.projectors[4])
        self.drawStatusLED(self.dc, x+30 , y, status.projectors[5])
        
        self.drawStatusLED(self.dc, x+70 , y, status.projectors[2])
        self.drawStatusLED(self.dc, x+100, y, status.projectors[3])

        self.drawStatusLED(self.dc, x    , y+35, status.projectors[0])
        self.drawStatusLED(self.dc, x+30 , y+35, status.projectors[1])

        self.drawStatusLED(self.dc, x    , y-35, status.projectors[8])
        self.drawStatusLED(self.dc, x+30 , y-35, status.projectors[9])

        self.dc.EndDrawing()

    def OnPaint(self, event=None):
        self.drawStatus()

    def on_button(self, event, button):
        thread = threading.Thread(target=self.run, args=(button,))
        thread.setDaemon(True)
        thread.start()
        
    def run(self, button):
        global root
        global config

        if button == "CAVE on":

            try:
                for i in range(0,5):
                    res = f35ctrl(i, ":ECOM0\n\r", 0.1)            
                    time.sleep(0.1)
                    res = f35ctrl(i, ":LMOD2\n\r", 0.1)            
                    time.sleep(0.1)
                    res = f35ctrl(i, ":POWR1\n\r", 0.5)            
                    if (res[0] > 0):
                        raise TimeLimitException(res[0])

                time.sleep(1)
                self.thread = InfoUpdate() 
                self.thread.start() 


            except TimeLimitException as e:
                self.message = "please wait " + str(int(e.value)) + " s!"
                publish.sendMessage("CHANGED", self.message)

        elif button == "CAVE-eco on":

            try:
                for i in range(0,5):
                    res = f35ctrl(i, ":ECOM1\n\r", 0.1)            
                    time.sleep(0.1)
                    res = f35ctrl(i, ":LMOD2\n\r", 0.1)            
                    time.sleep(0.1)
                    res = f35ctrl(i, ":POWR1\n\r", 0.5)            
                    if (res[0] > 0):
                        raise TimeLimitException(res[0])

                time.sleep(1)
                self.thread = InfoUpdate() 
                self.thread.start() 

            except TimeLimitException as e:
                self.message = "please wait " + str(int(e.value)) + " s!"
                publish.sendMessage("CHANGED", self.message)

        elif button == "CAVE-seco on":

            try:
                for i in range(0,5):
                    res = f35ctrl(i, ":ECOM1\n\r", 0.1)            
                    time.sleep(0.1)
                    res = f35ctrl(i, ":LMOD3\n\r", 0.1)            
                    time.sleep(0.1)
                    res = f35ctrl(i, ":POWR1\n\r", 0.5)            
                    if (res[0] > 0):
                        raise TimeLimitException(res[0])

                time.sleep(1)
                self.thread = InfoUpdate() 
                self.thread.start() 

            except TimeLimitException as e:
                self.message = "please wait " + str(int(e.value)) + " s!"
                publish.sendMessage("CHANGED", self.message)

        elif button == "CAVE off":

            try:
                for i in range(0,5):
                    res = f35ctrl(i, ":POWR0\n\r", 0.5)            
                    if (res[0] > 0):
                        raise TimeLimitException(res[0])

                time.sleep(1)
                self.thread = InfoUpdate() 
                self.thread.start() 

            except TimeLimitException as e:
                self.message = "please wait " + str(int(e.value)) + " s!"
                publish.sendMessage("CHANGED", self.message)

        elif button == "get info":
            
            self.thread = InfoUpdate() 
            self.thread.start() 
        
# -----------------------------------------------------------------------------
# add a notebook page to parent notebook
# -----------------------------------------------------------------------------
class PageApps(wx.Panel):
    def __init__(self, parent, lstNames):

        wx.Panel.__init__(self, parent)

        # define sizer and layout
        self.sizer = wx.GridBagSizer(10, 10)
        self.SetSizerAndFit(self.sizer)
        self.SetAutoLayout(True)

        # now add all the buttons in lstNames
        posL = 1
        posR = 1
        pBreak = 4

        for i in range(len(lstNames)):

            # create and add the button
            self.btn = wx.Button(self, label=lstNames[i])
            self.sizer.Add(self.btn, (posL, posR), flag=wx.EXPAND)

            # connect event to button
            func = functools.partial(self.on_button, button=(lstNames[i]))
            self.btn.Bind(wx.EVT_BUTTON, func)

            if posL < pBreak:
                posL += 1
            else:
                posL = 1
                posR += 1

    def on_button(self, event, button):

        thread = threading.Thread(target=self.run, args=(button,))
        thread.setDaemon(True)
        thread.start()
        
    def run(self, button):
        global root
        global config

        for demo in root.getiterator('demo'):
            if (demo.get('name') == button):
                
                cmd = demo.find('command').text
                print "start: ", cmd

                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

                ssh.connect(config.target, username=config.targetaccount)
                stdin, stdout, stderr = ssh.exec_command(cmd)
                stdin.flush()
                data = stdout.read().splitlines()
                for line in data:
                    print line

                #proc = subprocess.Popen(cmd, shell=True, 
                #       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

# -----------------------------------------------------------------------------
# add a notebook page to parent notebook
# -----------------------------------------------------------------------------
class NBPage(wx.Panel):

    def __init__(self, parent, tag):
        super(NBPage, self)
        global root

        self.panel = wx.Panel.__init__(self, parent)
        self.notebook = wx.Notebook(self, -1)

        self.bs = wx.BoxSizer(wx.VERTICAL)
        self.bs.Add(self.notebook, 1, wx.EXPAND)
        self.SetSizer(self.bs)

        lstApps = []
        lstlstNames = []

        # add all app-types to lstApps
        for child in root.find(tag):
            if child.attrib['type'] not in lstApps:
                lstApps.append(child.attrib['type'])
                lstNames = []
                lstlstNames.append(lstNames)
            
        # append app names to lstNames in lstlstNames according to
        # their type position in lstApps
        for child in root.find(tag):
            pos = lstApps.index(child.attrib['type'])
            lstlstNames[pos].append(child.attrib['name'])

        # now create tabs and buttons according to Apps & Names

        for i in range(len(lstApps)):
            aPage = PageApps(self.notebook, lstlstNames[i])
            self.notebook.AddPage(aPage, lstApps[i][0:3])
 
# -----------------------------------------------------------------------------
# the app class reads the xml config file and gets the basic configuration
# from that file. then it's creating the main pages as configured
# -----------------------------------------------------------------------------
class RunDemo(wx.Frame):

    def __init__(self, *args, **kwargs): 
        super(RunDemo, self).__init__(*args, **kwargs) 

        self.readData()
        self.getConfig()
        self.initUI()

        time.sleep(1)
        self.thread = InfoUpdate() 
        self.thread.start() 

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.timer.Start(30000)

    def update(self, event):
        self.thread = InfoUpdate() 
        self.thread.start()      

    def readData(self):
        global tree, root
        
        tree = ET.parse('demos.xml')
        root = tree.getroot()

    def getConfig(self):
        global config, root

        for child in root.find('config'):
            if child.tag == 'target':
                config.target = child.text
                #print config.target
            elif child.tag == 'targetaccount':
                config.targetaccount = child.text
                #print config.targetaccount
            elif child.tag == 'env':
                config.env = child.text
                #print config.env
            elif child.tag == 'tidy-ip':
                config.tidyip = child.text
                #print config.tidyip
            elif child.tag == 'tidy-port':
                config.tidyport = child.text
                #print config.tidyport
            elif child.tag == 'projectors':
                config.projectors = child.text.split()
                #print "list: ", config.projectors

    def initUI(self):
        global edtStdOut

        self.font = wx.SystemSettings_GetFont(wx.SYS_SYSTEM_FONT)
        self.font.SetPointSize(9)

        notebook = wx.Notebook(self, -1)

        nbpList = []

        for child in root:
            if child.tag == 'apps':
                nbp = NBPage(notebook, child.tag)
                notebook.AddPage(nbp, child.tag)
                nbpList.append(nbp)
            elif child.tag == 'CAVE':
                nbp = PageCAVE(notebook)
                notebook.AddPage(nbp, child.tag)
                nbpList.append(nbp)
            elif child.tag == 'TIDY':
                nbp = PageTIDY(notebook)
                notebook.AddPage(nbp, child.tag)
                nbpList.append(nbp)
            elif child.tag == 'FASI':
                nbp = PageFASI(notebook)
                notebook.AddPage(nbp, child.tag)
                nbpList.append(nbp)

        publish.subscribe(self.onMessage, "CHANGED")

        # size and position of window

        if config.env == 'embedded':
            self.SetSize((480, 272))
            self.Centre()
            self.Show(True)
            self.SetTitle('runDemo')
        elif config.env == 'desktop':
            self.SetSize((1275, 600))
            self.Centre()
            self.Show(True)
            self.SetTitle('runDemo')
        else:
            self.SetSize((800, 600))
            self.Centre()
            self.Show(True)
            self.SetTitle('runDemo')

        cursor = wx.StockCursor(wx.CURSOR_BLANK)
        self.SetCursor(cursor)

    def onMessage(self, message):

        print "debug: onMessage = ", message.data                                 #TODO

# -----------------------------------------------------------------------------
# check environment (Linux/X/Windows)
# call app main loop
# -----------------------------------------------------------------------------
def main():
    import os, sys, traceback, curses
    global config
    global status

    config = Config()
    status = Status()

    # check if we are under linux or windows
    if os.name == 'nt':
        
        # we are running windows, usally windows has windows
        config.os = e_os.windows
        config.xenv = e_xenv.yes

    elif os.name == 'posix':

        # we are running linux
        config.os = e_os.linux

        # check if there is a DISPLAY set
        try:
            if (os.environ['DISPLAY'] != ''):
                config.xenv = e_xenv.yes
            else:
                config.xenv = e_xenv.no
        except Exception:
            config.xenv = e_xenv.yes

    else:
        
        # we are running something we don't know yet
        print 'error: OS not supported yet'
        sys.exit(0)

    # now, start up window or console application
    try:
        if config.xenv == e_xenv.no:
            print "console application not done yet :("
        else:
            ex = wx.App()
            RunDemo(None, title='RunDemo')
            ex.MainLoop()

    except Exception:
        traceback.print_exc(file=sys.stdout)
    
    sys.exit(0)

# -----------------------------------------------------------------------------
# call main
# -----------------------------------------------------------------------------
if __name__ == '__main__':

    main()

# -----------------------------------------------------------------------------
