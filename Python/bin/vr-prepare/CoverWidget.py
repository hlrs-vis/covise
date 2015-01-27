
from PyQt5 import QtCore, QtGui, QtWidgets

import threading 
import socket
from queue import Queue

class ThreadedSender(threading.Thread): 
    def __init__(self, addr, msgQueue):
        threading.Thread.__init__(self, None)
        self._addr = addr
        self._socket_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)        
        self._msgQueue = msgQueue
        self._recvBufSize = 1024 

    def waitForCover(self):
        socket_receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        socket_receive.bind((self._addr[0], 0))
        (inetaddr, port) = socket_receive.getsockname()
        socket_receive.settimeout(.5)

        coverIsReady = False
        while (not coverIsReady): 
            msg = str("ping = [" + str(port) + "]")
            self._socket_send.sendto(msg, self._addr)
            try:             
                (retMsg, addr) = socket_receive.recvfrom(self._recvBufSize)
            except socket.timeout: 
                coverIsReady = False
            else: 
                while (retMsg != str("pong")):
                    (retMsg, addr) = socket_receive.recvfrom(self._recvBufSize)    
                coverIsReady = True
                

    def run(self):
        self.waitForCover()
        msg = self._msgQueue.get(1)
        while (msg):
            self._socket_send.sendto(msg, self._addr)
            msg = self._msgQueue.get(1)


class CoverMsgSender:
    def __init__(self, address, port):
        self._queue = Queue()
        self._addr  = (address, port)
        self._sender = ThreadedSender(self._addr, self._queue)
        self._sender.start()
        
    def sendMsg(self, msg):
        self._queue.put(msg)
        
class CoverWidget(QtWidgets.QWidget):

    def __init__(self, parent, mainWindow):
        QtWidgets.QWidget.__init__(self, parent)
        #size policy
        #sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        #sizePolicy.setHorizontalStretch(0)
        #sizePolicy.setVerticalStretch(0)
        #sizePolicy.setHeightForWidth(True)
        #self.setSizePolicy(sizePolicy)

        # need to set this true to get mouseMoveEvents if mousebutton isn't pressed
        self.setMouseTracking(True)

        # socket to write events in
        self.msgSender = CoverMsgSender('127.0.0.1', 7878)
        
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)
        self._firstEvent = True
        self.__mainWindow = mainWindow
        self.setAcceptDrops(True)
        self.blockSignals(True)
        
        self.__mainWindow.toggled.connect(self._pinButtonToggled)


        self.__buttonMapping = {QtCore.Qt.LeftButton: 1,
                              QtCore.Qt.MidButton: 2,
                              QtCore.Qt.RightButton: 3}
        self.__keyMapping = {QtCore.Qt.Key_Space:       0x20,
                             QtCore.Qt.Key_Backspace:   0xFF08,
                             QtCore.Qt.Key_Tab:         0xFF09,
                             QtCore.Qt.Key_Return:      0xFF0D,
                             QtCore.Qt.Key_Pause:       0xFF13,
                             QtCore.Qt.Key_ScrollLock:  0xFF14,
                             QtCore.Qt.Key_Escape:      0xFF1B,
                             QtCore.Qt.Key_Delete:      0xFFFF,
                             QtCore.Qt.Key_Home:        0xFF50,
                             QtCore.Qt.Key_Left:        0xFF51,
                             QtCore.Qt.Key_Up:          0xFF52,
                             QtCore.Qt.Key_Right:       0xFF53,
                             QtCore.Qt.Key_Down:        0xFF54,
                             QtCore.Qt.Key_PageUp:      0xFF55,
                             QtCore.Qt.Key_PageDown:    0xFF56,
                             QtCore.Qt.Key_End:         0xFF57,
                             QtCore.Qt.Key_Home:        0xFF58,
                             QtCore.Qt.Key_Print:       0xFF61,
                             QtCore.Qt.Key_Insert:      0xFF63,
                             QtCore.Qt.Key_NumLock:     0xFF7F,
                             # KeyPad not mapped
                             QtCore.Qt.Key_F1:          0xFFBE,
                             QtCore.Qt.Key_F2:          0xFFBF,
                             QtCore.Qt.Key_F3:          0xFFC0,
                             QtCore.Qt.Key_F4:          0xFFC1,
                             QtCore.Qt.Key_F5:          0xFFC2,
                             QtCore.Qt.Key_F6:          0xFFC3,
                             QtCore.Qt.Key_F7:          0xFFC4,
                             QtCore.Qt.Key_F8:          0xFFC5,
                             QtCore.Qt.Key_F9:          0xFFC6,
                             QtCore.Qt.Key_F10:         0xFFC7,
                             QtCore.Qt.Key_F11:         0xFFC8,
                             QtCore.Qt.Key_F12:         0xFFC9,
                             QtCore.Qt.Key_Shift:       0xFFE1, # Shift_L
                             QtCore.Qt.Key_Control:     0xFFE3, # Control_L
                             QtCore.Qt.Key_CapsLock:    0xFFE5,
                             QtCore.Qt.Key_Alt:         0xFFE9 # Alt_L
                             }
    # see http://developer.qt.nokia.com/doc/qt-4.8/qt.html#MouseButton-enum
    # and http://www.openscenegraph.org/documentation/OpenSceneGraphReferenceDocs/a00293.html

    def _pinButtonToggled(self, checked):
        self._resizeCover()

    def _resizeCover(self):
        offset = 0 # offset is used to simulate the COVER widget extending behind the rightFrame
        if self.__mainWindow:
            if self.__mainWindow.FrameRight.isVisible() and (not self.__mainWindow.FrameRightPinButton.isChecked()):
                offset = self.__mainWindow.FrameRight.frameGeometry().width() + self.__mainWindow.splitter1.handleWidth()
        self.__send("resizeEvent", self.width()+offset, self.height())

    def setEnabled(self, b):
        #if b and self.__mainWindow:
        #    self.__mainWindow.setUpdatesEnabled(False)
        QtWidgets.QWidget.setEnabled(self, b)

    # send message
    def __send(self, event, *values):
        msg = str(event)
        if (len(values) > 0):
            msg += " = ["
            if (len(values) == 1):
                msg += str(values[0])
            else:
                msg += reduce( lambda a,b: str(a)+","+str(b) , values)
            msg += "]"
        self.msgSender.sendMsg(msg)

    # returns tuple (key, modifier)
    def __keyToOsg(self, keyEvent):
        if (keyEvent.key() in self.__keyMapping):
            osgKey = self.__keyMapping[keyEvent.key()]
        else:
            osgKey = keyEvent.key() + 32
        osgMod = 0
        if (keyEvent.modifiers() & QtCore.Qt.ShiftModifier):
            osgMod |= 3 # MODKEY_LEFT_SHIFT|MODKEY_RIGHT_SHIFT
        if (keyEvent.modifiers() & QtCore.Qt.ControlModifier):
            osgMod |= 12 # MODKEY_LEFT_CTRL|MODKEY_RIGHT_CTRL
        if (keyEvent.modifiers() & QtCore.Qt.AltModifier):
            osgMod |= 48 # MODKEY_LEFT_ALT|MODKEY_RIGHT_ALT
        return (osgKey, osgMod)

    # mouse was moved -> take the appropriate actions 
    def __mouseMoved(self, pos):
        self.__send("mouseMoveEvent", pos.x(), pos.y())
        if not self.__mainWindow:
            return
        if self.__mainWindow.FrameRightPinButton.isChecked():
            return
        if (QtWidgets.QApplication.mouseButtons() != QtCore.Qt.NoButton):
            return
        #if mouse at right end of cover, show coxmlbrowser, hide otherwise
        if (self.width() - pos.x() < 15):
            #self.__mainWindow.widget.setUpdatesEnabled(True) # workaround for embedded cover with stylesheet
            self.__mainWindow.FrameRight.show()
        elif (not self.__mainWindow.FrameRightPinButton.isChecked()):
            self.__mainWindow.FrameRight.hide()
            #self.__mainWindow.widget.setUpdatesEnabled(False) # workaround for embedded cover with stylesheet


    def event(self, event):
        #print "event", event
        return QtWidgets.QWidget.event(self, event)

    def childEvent(self, event):
        pass  

    def closeEvent(self, closeEvent):
        self.__send("closeEvent")
        return QtWidgets.QWidget.closeEvent(self, closeEvent)

    def dragEnterEvent(self, dragEnterEvent):
        if isinstance(dragEnterEvent.source(), QtGui.QListView):
            dragEnterEvent.accept() # for now we accept all ListView Items
        return QtWidgets.QWidget.dragEnterEvent(self, dragEnterEvent)

    def dragLeaveEvent(self, dragLeaveEvent):
        return QtWidgets.QWidget.dragLeaveEvent(self, dragLeaveEvent)

    def dragMoveEvent(self, dragMoveEvent):
        self.__mouseMoved(dragMoveEvent.pos())
        return QtWidgets.QWidget.dragMoveEvent(self, dragMoveEvent)

    def dropEvent(self, dropEvent):
        if isinstance(dropEvent.source(), QtGui.QListView):
            # dropping of ListView Items signals a doubleClick on the item
            indexes = dropEvent.source().selectedIndexes()
            if len(indexes) == 1:
                dropEvent.source().doubleClicked.emit( indexes[0])
        self.__send("dropEvent")
        return QtWidgets.QWidget.dropEvent(self, dropEvent)

    def enterEvent(self, enterEvent):
        if (self._firstEvent) and (self.__mainWindow != None):
            # In some cases, the OpenCOVER doesn't display properly. A resize of the MainWindow helps.
            # Note: A resize event sent to the OpenCOVER isn't enough.
            self.__mainWindow.resize(self.__mainWindow.size().width()-1, self.__mainWindow.size().height()-1)
            self.__mainWindow.resize(self.__mainWindow.size().width()+1, self.__mainWindow.size().height()+1)
            self._firstEvent = False

        self.grabKeyboard()

        if self.__mainWindow and (not self.__mainWindow.FrameRightPinButton.isChecked()):
            self.__mainWindow.FrameRight.hide()

        self.__send("enterEvent")
        #return QtWidgets.QWidget.enterEvent(self, enterEvent)

    def focusInEvent(self, focusInEvent):
        self.__send("focusInEvent")
        #return QtWidgets.QWidget.focusInEvent(self, focusInEvent)

    def focusOutEvent(self, focusOutEvent):
        self.__send("focusOutEvent")
        return QtWidgets.QWidget.focusOutEvent(self, focusOutEvent)

    def hideEvent(self, hideEvent):
        self.__send("hideEvent")
        return QtWidgets.QWidget.hideEvent(self, hideEvent)

    def keyPressEvent(self, keyPressEvent):
        keyAndMod = self.__keyToOsg(keyPressEvent)
        self.__send("keyPressEvent", *keyAndMod)
        return QtWidgets.QWidget.keyPressEvent(self, keyPressEvent)

    def keyReleaseEvent(self, keyReleaseEvent):
        keyAndMod = self.__keyToOsg(keyReleaseEvent)
        self.__send("keyReleaseEvent", *keyAndMod)
        return QtWidgets.QWidget.keyReleaseEvent(self, keyReleaseEvent)

    def leaveEvent(self, leaveEvent):
        self.releaseKeyboard()
        self.__send("leaveEvent")
        return QtWidgets.QWidget.leaveEvent(self, leaveEvent)

    def mouseDoubleClickEvent(self, mouseDoubleClickEvent):
        if not mouseDoubleClickEvent.button() in self.__buttonMapping:
            return
        button = self.__buttonMapping[mouseDoubleClickEvent.button()]
        self.__send("mouseDoubleClickEvent", mouseDoubleClickEvent.x(), mouseDoubleClickEvent.y(), button)
        return QtWidgets.QWidget.mouseDoubleClickEvent(self, mouseDoubleClickEvent)

    def mouseMoveEvent(self, mouseMoveEvent):
        self.__mouseMoved(mouseMoveEvent.pos())
        return QtWidgets.QWidget.mouseMoveEvent(self, mouseMoveEvent)        

    def mousePressEvent(self, mousePressEvent):
        if not mousePressEvent.button() in self.__buttonMapping:
            return
        button = self.__buttonMapping[mousePressEvent.button()]
        self.__send("mousePressEvent", mousePressEvent.x(), mousePressEvent.y(), button)
        return QtWidgets.QWidget.mousePressEvent(self, mousePressEvent)

    def mouseReleaseEvent(self, mouseReleaseEvent):
        if not mouseReleaseEvent.button() in self.__buttonMapping:
            return
        button = self.__buttonMapping[mouseReleaseEvent.button()]
        self.__send("mouseReleaseEvent", mouseReleaseEvent.x(), mouseReleaseEvent.y(), button)
        return QtWidgets.QWidget.mouseReleaseEvent(self, mouseReleaseEvent)

    def moveEvent(self, moveEvent):
        self.__send("moveEvent", moveEvent.pos().x(), moveEvent.pos().y())
        return QtWidgets.QWidget.moveEvent(self, moveEvent)

    def paintEvent(self, paintEvent):
        pass

    def resizeEvent(self, resizeEvent):
        self._resizeCover()
        return QtWidgets.QWidget.resizeEvent(self, resizeEvent)

    def showEvent(self, showEvent):
        self.__send("showEvent")
        return QtWidgets.QWidget.showEvent(self, showEvent)

    def wheelEvent(self, wheelEvent):
        self.__send("wheelEvent", wheelEvent.delta() / 8)
        return QtWidgets.QWidget.wheelEvent(self, wheelEvent)

    def changeEvent(self, changeEvent):
        return QtWidgets.QWidget.changeEvent(self, changeEvent)

    def setStyleSheet(self, sheet):
        pass

    def setBackgroundRole(self, colorRole):
        pass

