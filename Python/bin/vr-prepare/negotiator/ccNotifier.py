# Part of the vr-prepare program for dc

# Copyright (c) 2006-2007 Visenso GmbH

import covise
import socket

def sendLoadingFinishedUdpMessage():
    '''Sends a message to Cyber-Classroom Ogre menu to stop loading-in-progress-screen.'''
    magicString = covise.getCoConfigEntry("vr-prepare.CCIntroUDP.MagicString")
    if not magicString:
        magicString = "CC_MOD_READY"
    
    magicPort = covise.getCoConfigEntry("vr-prepare.CCIntroUDP.MagicPort")
    if not magicPort:
        magicPort = "44449"
    
    destinationHost = covise.getCoConfigEntry("vr-prepare.CCIntroUDP.DestinationHost")
    if not destinationHost:
        destinationHost = "127.0.0.1"
    
    print("Sending", magicString, "to", destinationHost, magicPort)
    sock = socket.socket( socket.AF_INET, socket.SOCK_DGRAM )
    sock.sendto( bytes(magicString, 'utf-8'), (destinationHost, int(magicPort)) )
