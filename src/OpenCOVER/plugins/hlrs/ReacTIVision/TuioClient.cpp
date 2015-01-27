/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
	TUIO C++ Library - part of the reacTIVision project
	http://reactivision.sourceforge.net/

	Copyright (c) 2005-2008 Martin Kaltenbrunner <mkalten@iua.upf.edu>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include "TuioClient.h"
#include <exception>

#ifndef WIN32
static void *ThreadFunc(void *obj)
#else
static DWORD WINAPI ThreadFunc(LPVOID obj)
#endif
{
    static_cast<TuioClient *>(obj)->socket->Run();
    return 0;
};

TuioClient::TuioClient()
{
    TuioClient(3333);
}

TuioClient::TuioClient(int port)
{
    try
    {
        socket = new UdpListeningReceiveSocket(IpEndpointName(IpEndpointName::ANY_ADDRESS, port), this);
    }
    catch (...)
    {
        std::cout << "could not bind to UDP port " << port << std::endl;
        socket = NULL;
    }

    if (socket != NULL)
    {
        if (!socket->IsBound())
        {
            delete socket;
            socket = NULL;
        }
        else
            std::cout << "listening to TUIO messages on UDP port " << port << std::endl;
    }

    locked = false;
    running = false;
    currentFrame = lastFrame = maxFingerID = -1;
}

TuioClient::~TuioClient()
{
    delete socket;
}

void TuioClient::ProcessBundle(const ReceivedBundle &b, const IpEndpointName &remoteEndpoint)
{
    for (ReceivedBundle::const_iterator i = b.ElementsBegin(); i != b.ElementsEnd(); ++i)
    {
        if (i->IsBundle())
            ProcessBundle(ReceivedBundle(*i), remoteEndpoint);
        else
        // ProcessMessage( ReceivedMessage(*i), remoteEndpoint);
        {
            mutex.lock();
            ProcessMessage(ReceivedMessage(*i), remoteEndpoint);
            mutex.unlock();
        }
    }
}

void TuioClient::ProcessMessage(const ReceivedMessage &msg, const IpEndpointName &remoteEndpoint)
{
    try
    {
        ReceivedMessageArgumentStream args = msg.ArgumentStream();
        ReceivedMessage::const_iterator arg = msg.ArgumentsBegin();

        if (strcmp(msg.AddressPattern(), "/tuio/2Dobj") == 0)
        {

            const char *cmd;
            args >> cmd;

            if (strcmp(cmd, "set") == 0)
            {
                if ((currentFrame < lastFrame) && (currentFrame > 0))
                    return;

                int32 s_id, f_id;
                float xpos, ypos, angle, xspeed, yspeed, rspeed, maccel, raccel;

                args >> s_id >> f_id >> xpos >> ypos >> angle >> xspeed >> yspeed >> rspeed >> maccel >> raccel >> EndMessage;

                std::list<TuioObject *>::iterator tobj;
                for (tobj = objectList.begin(); tobj != objectList.end(); tobj++)
                    if ((*tobj)->getSessionID() == (long)s_id)
                        break;

                if (tobj == objectList.end())
                {

                    TuioObject *addObject = new TuioObject((long)s_id, (int)f_id, xpos, ypos, angle);
                    objectList.push_back(addObject);

                    for (std::list<TuioListener *>::iterator listener = listenerList.begin(); listener != listenerList.end(); listener++)
                        (*listener)->addTuioObject(addObject);
                }
                else if (((*tobj)->getX() != xpos) || ((*tobj)->getY() != ypos) || ((*tobj)->getAngle() != angle) || ((*tobj)->getXSpeed() != xspeed) || ((*tobj)->getYSpeed() != yspeed) || ((*tobj)->getRotationSpeed() != rspeed) || ((*tobj)->getMotionAccel() != maccel) || ((*tobj)->getRotationAccel() != raccel))
                {
                    (*tobj)->update(xpos, ypos, angle, xspeed, yspeed, rspeed, maccel, raccel);

                    for (std::list<TuioListener *>::iterator listener = listenerList.begin(); listener != listenerList.end(); listener++)
                        (*listener)->updateTuioObject((*tobj));
                }
            }
            else if (strcmp(cmd, "alive") == 0)
            {
                if ((currentFrame < lastFrame) && (currentFrame > 0))
                    return;

                int32 s_id;
                while (!args.Eos())
                {
                    args >> s_id;
                    objectBuffer.push_back((long)s_id);

                    std::list<long>::iterator iter;
                    iter = find(aliveObjectList.begin(), aliveObjectList.end(), (long)s_id);
                    if (iter != aliveObjectList.end())
                        aliveObjectList.erase(iter);
                }
                args >> EndMessage;

                std::list<long>::iterator alive_iter;
                for (alive_iter = aliveObjectList.begin(); alive_iter != aliveObjectList.end(); alive_iter++)
                {
                    std::list<TuioObject *>::iterator tobj;
                    for (tobj = objectList.begin(); tobj != objectList.end(); tobj++)
                    {
                        TuioObject *deleteObject = (*tobj);
                        if (deleteObject->getSessionID() == *alive_iter)
                        {
                            deleteObject->remove();
                            for (std::list<TuioListener *>::iterator listener = listenerList.begin(); listener != listenerList.end(); listener++)
                                (*listener)->removeTuioObject(deleteObject);
                            objectList.erase(tobj);
                            delete deleteObject;
                            break;
                        }
                    }
                }

                aliveObjectList = objectBuffer;
                objectBuffer.clear();
            }
            else if (strcmp(cmd, "fseq") == 0)
            {

                if (currentFrame > 0)
                    lastFrame = currentFrame;
                args >> currentFrame >> EndMessage;

                if ((currentFrame >= lastFrame) || (currentFrame < 0))
                {

                    long currentTime = lastTime;
                    if (currentFrame > lastFrame)
                    {
                        currentTime = getCurrentTime() - startTime;
                        lastTime = currentTime;
                    }

                    for (std::list<TuioObject *>::iterator refreshObject = objectList.begin(); refreshObject != objectList.end(); refreshObject++)
                        if ((*refreshObject)->getUpdateTime() == TUIO_UNDEFINED)
                            (*refreshObject)->setUpdateTime(currentTime);

                    for (std::list<TuioListener *>::iterator listener = listenerList.begin(); listener != listenerList.end(); listener++)
                        (*listener)->refresh(currentTime);
                }
            }
        }
        else if (strcmp(msg.AddressPattern(), "/tuio/2Dcur") == 0)
        {
            const char *cmd;
            args >> cmd;

            if (strcmp(cmd, "set") == 0)
            {
                if ((currentFrame < lastFrame) && (currentFrame > 0))
                    return;

                int32 s_id;
                float xpos, ypos, xspeed, yspeed, maccel;

                args >> s_id >> xpos >> ypos >> xspeed >> yspeed >> maccel >> EndMessage;

                std::list<TuioCursor *>::iterator tcur;
                for (tcur = cursorList.begin(); tcur != cursorList.end(); tcur++)
                    if ((*tcur)->getSessionID() == (long)s_id)
                        break;

                if (tcur == cursorList.end())
                {

                    int f_id = (int)cursorList.size();
                    if ((int)(cursorList.size()) <= maxFingerID)
                    {
                        std::list<TuioCursor *>::iterator closestCursor = freeCursorList.begin();

                        for (std::list<TuioCursor *>::iterator testCursor = freeCursorList.begin(); testCursor != freeCursorList.end(); testCursor++)
                        {
                            if ((*testCursor)->getDistance(xpos, ypos) < (*closestCursor)->getDistance(xpos, ypos))
                                closestCursor = testCursor;
                        }

                        f_id = (*closestCursor)->getFingerID();
                        freeCursorList.erase(closestCursor);
                        delete *closestCursor;
                    }
                    else
                        maxFingerID = f_id;

                    TuioCursor *addCursor = new TuioCursor((long)s_id, f_id, xpos, ypos);
                    cursorList.push_back(addCursor);

                    for (std::list<TuioListener *>::iterator listener = listenerList.begin(); listener != listenerList.end(); listener++)
                        (*listener)->addTuioCursor(addCursor);
                }
                else if (((*tcur)->getX() != xpos) || ((*tcur)->getY() != ypos) || ((*tcur)->getXSpeed() != xspeed) || ((*tcur)->getYSpeed() != yspeed) || ((*tcur)->getMotionAccel() != maccel))
                {
                    (*tcur)->update(xpos, ypos, xspeed, yspeed, maccel);
                    for (std::list<TuioListener *>::iterator listener = listenerList.begin(); listener != listenerList.end(); listener++)
                        (*listener)->updateTuioCursor((*tcur));
                }
            }
            else if (strcmp(cmd, "alive") == 0)
            {
                if ((currentFrame < lastFrame) && (currentFrame > 0))
                    return;

                int32 s_id;
                while (!args.Eos())
                {
                    args >> s_id;
                    cursorBuffer.push_back((long)s_id);

                    std::list<long>::iterator iter;
                    iter = find(aliveCursorList.begin(), aliveCursorList.end(), (long)s_id);
                    if (iter != aliveCursorList.end())
                        aliveCursorList.erase(iter);
                }
                args >> EndMessage;

                std::list<long>::iterator alive_iter;
                for (alive_iter = aliveCursorList.begin(); alive_iter != aliveCursorList.end(); alive_iter++)
                {
                    std::list<TuioCursor *>::iterator tcur;
                    for (tcur = cursorList.begin(); tcur != cursorList.end(); tcur++)
                    {
                        TuioCursor *deleteCursor = (*tcur);
                        if (deleteCursor->getSessionID() == *alive_iter)
                        {

                            cursorList.erase(tcur);
                            deleteCursor->remove();
                            for (std::list<TuioListener *>::iterator listener = listenerList.begin(); listener != listenerList.end(); listener++)
                                (*listener)->removeTuioCursor(deleteCursor);

                            if (deleteCursor->getFingerID() == maxFingerID)
                            {
                                maxFingerID = -1;
                                delete deleteCursor;

                                if (cursorList.size() > 0)
                                {
                                    std::list<TuioCursor *>::iterator clist;
                                    for (clist = cursorList.begin(); clist != cursorList.end(); clist++)
                                    {
                                        int f_id = (*clist)->getFingerID();
                                        if (f_id > maxFingerID)
                                            maxFingerID = f_id;
                                    }

                                    std::list<TuioCursor *>::iterator flist;
                                    for (flist = freeCursorList.begin(); flist != freeCursorList.end(); flist++)
                                    {
                                        TuioCursor *freeCursor = (*flist);
                                        if (freeCursor->getFingerID() > maxFingerID)
                                            delete freeCursor;
                                        else
                                            freeCursorBuffer.push_back(freeCursor);
                                    }

                                    freeCursorList = freeCursorBuffer;
                                    freeCursorBuffer.clear();
                                }
                            }
                            else if (deleteCursor->getFingerID() < maxFingerID)
                                freeCursorList.push_back(deleteCursor);

                            break;
                        }
                    }
                }

                aliveCursorList = cursorBuffer;
                cursorBuffer.clear();
            }
            else if (strcmp(cmd, "fseq") == 0)
            {

                if (currentFrame > 0)
                    lastFrame = currentFrame;
                args >> currentFrame >> EndMessage;

                if ((currentFrame >= lastFrame) || (currentFrame < 0))
                {
                    long currentTime = lastTime;
                    if (currentFrame > lastFrame)
                    {
                        currentTime = getCurrentTime() - startTime;
                        lastTime = currentTime;
                    }

                    for (std::list<TuioCursor *>::iterator refreshCursor = cursorList.begin(); refreshCursor != cursorList.end(); refreshCursor++)
                        if ((*refreshCursor)->getUpdateTime() == TUIO_UNDEFINED)
                            (*refreshCursor)->setUpdateTime(currentTime);

                    for (std::list<TuioListener *>::iterator listener = listenerList.begin(); listener != listenerList.end(); listener++)
                        (*listener)->refresh(currentTime);
                }
            }
        }
    }
    catch (Exception &e)
    {
        std::cout << "error while parsing message: " << msg.AddressPattern() << ": " << e.what() << "\n";
    }
}

void TuioClient::ProcessPacket(const char *data, int size, const IpEndpointName &remoteEndpoint)
{
    if (listenerList.size() == 0)
        return;
    ReceivedPacket p(data, size);
    if (p.IsBundle())
        ProcessBundle(ReceivedBundle(p), remoteEndpoint);
    else // OK ?
    // ProcessMessage( ReceivedMessage(*i), remoteEndpoint);
    {
        mutex.lock();
        ProcessMessage(ReceivedMessage(p), remoteEndpoint);
        mutex.unlock();
    }
}

/*
void TuioClient::start(bool lk) {

	if (socket==NULL) return;

	locked = lk;
	if (!locked) {
		#ifndef WIN32
		pthread_create(&thread , NULL, ThreadFunc, this);
		#else
		DWORD threadId;
//		thread = CreateThread( 0, 0, ThreadFunc, this, 0, &threadId );
		thread = CreateThread( 0, 0, ThreadFunc, this, 0, &threadId );
		#endif
	} else socket->Run();
	
	startTime = getCurrentTime();
	lastTime = 0;

	running = true;
}
*/

void TuioClient::run()
{

    if (socket == NULL)
        return;
    locked = false;
    startTime = getCurrentTime();
    lastTime = 0;
    running = true;

    socket->Run();
}

void TuioClient::stop()
{

    if (socket == NULL)
        return;
    socket->Break();

    if (!locked)
    {
#ifdef WIN32
        if (thread)
            CloseHandle(thread);
#endif
        thread = 0;
        locked = false;
    }
    running = false;
}

void TuioClient::addTuioListener(TuioListener *listener)
{
    listenerList.push_back(listener);
}

void TuioClient::removeTuioListener(TuioListener *listener)
{
    std::list<TuioListener *>::iterator result = find(listenerList.begin(), listenerList.end(), listener);
    if (result != listenerList.end())
        listenerList.remove(listener);
}

TuioObject *TuioClient::getTuioObject(long s_id)
{
    for (std::list<TuioObject *>::iterator iter = objectList.begin(); iter != objectList.end(); iter++)
        if ((*iter)->getSessionID() == s_id)
            return (*iter);

    return NULL;
}

TuioCursor *TuioClient::getTuioCursor(long s_id)
{
    for (std::list<TuioCursor *>::iterator iter = cursorList.begin(); iter != cursorList.end(); iter++)
        if ((*iter)->getSessionID() == s_id)
            return (*iter);

    return NULL;
}
//
//
//std::list<TuioObject*> TuioClient::getTuioObjects() {
//	return objectList;
//}

std::list<TuioObject *> *TuioClient::getTuioObjects()
{
    return &objectList;
}

//std::list<TuioCursor*> TuioClient::getTuioCursors() {
//	return cursorList;
//}

std::list<TuioCursor *> *TuioClient::getTuioCursors()
{
    return &cursorList;
}

long TuioClient::getCurrentTime()
{

#ifdef WIN32
    long timestamp = GetTickCount();
#else
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    long timestamp = (tv.tv_sec * 1000) + (tv.tv_usec / 1000);
#endif

    return timestamp;
}
