/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *									*
 *          								*
 *                            (C) 2001					*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 *	File			Tarsus.cpp 				*
 *									*
 *	Description		Tarsus optical tracking system interface class				*
 *									*
 *	Author			Uwe Woessner				*
 *									*
 *	Date			Jan 2004				*
 *									*
 *	Status			in dev					*
 *									*
 ************************************************************************/

#if !defined(_WIN32) && !defined(__APPLE__)
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>
#endif

#include <util/common.h>
#include <net/covise_socket.h>
#include <net/covise_host.h>
#include <cover/coVRPluginSupport.h>
#include "ClientCodes.h"
#include "Tarsus.h"

using namespace covise;
using namespace opencover;
//#define VERBOSE

bool Tarsus::receive(char *pBuffer, int BufferSize)
{
    char *p = pBuffer;
    char *e = pBuffer + BufferSize;

    int result;

    while (p != e)
    {
        result = sock->read(p, e - p);

        if (result <= 0)
        {
            // error or EOF (closed socket)
            return false;
        }

        p += result;
    }

    return true;
}

//	There are also some helpers to make the code a little less ugly.

bool Tarsus::receive(int32_t &Val)
{
    return receive((char *)&Val, sizeof(Val));
}

bool Tarsus::receive(uint32_t &Val)
{
    return receive((char *)&Val, sizeof(Val));
}

bool Tarsus::receive(double &Val)
{
    return receive((char *)&Val, sizeof(Val));
}

//-----------------------------------------------------------------------------

Tarsus::Tarsus(int portnumber, const char *host)
{
    sock = NULL;
#ifdef _WIN32
    //- Initialisation - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //  Windows-specific initialisation.

    WORD wVersionRequested;
    WSADATA wsaData;
    wVersionRequested = MAKEWORD(2, 0);
    if (WSAStartup(wVersionRequested, &wsaData) != 0)
    {
        std::cout << "Socket * Initialization Error" << std::endl;
    }
#endif

    port = portnumber;
    if (host == NULL)
        return;
    hostName = new char[strlen(host) + 1];
    strcpy(hostName, host);
    size = 0;

#ifdef HAVE_PTHREAD
    if (pthread_create(&trackerThread, NULL, startThread, this))
    {
        cerr << "failed to create trackerThread: " << strerror(errno) << endl;
    }
#endif
}

void Tarsus::setStationName(unsigned int station, const char *name)
{
    if (stationNames.size() < station + 1)
    {
        stationNames.resize(station + 1);
    }
    if (stationNames[station])
    {
        delete[] stationNames[station];
    }
    stationNames[station] = new char[strlen(name) + 1];
    strcpy(stationNames[station], name);

#ifndef VERBOSE
    if (cover->debugLevel(1))
    {
#endif
        cerr << "Tarsus: naming station " << station << " " << name << endl;
#ifndef VERBOSE
    }
#endif
}

void Tarsus::initialize()
{
    // clear out old data
    BodyChannels.clear();

    //- Get Info - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //	Request the channel information

    pBuff = buff;

    *((uint32_t *)pBuff) = ClientCodes::EInfo;
    pBuff += sizeof(uint32_t);
    *((uint32_t *)pBuff) = ClientCodes::ERequest;
    pBuff += sizeof(uint32_t);

    if (sock->write(buff, pBuff - buff) != (pBuff - buff))
        cerr << "Error Requesting" << endl;

    int32_t packet;
    int32_t type;

    if (!receive(packet))
        cerr << "Tarsus Init: Error Recieving" << endl;

    if (!receive(type))
        cerr << "Tarsus Init: Error Recieving" << endl;

    if (type != ClientCodes::EReply)
        cerr << "Tarsus Init: Bad Packet" << endl;

    if (packet != ClientCodes::EInfo)
        cerr << "Tarsus Init: Bad Reply Type" << endl;

    if (!receive(size))
        cerr << "Tarsus Init: Tarsus ERROR" << endl;

    info.resize(size);

    std::vector<std::string>::iterator iInfo;

    for (iInfo = info.begin(); iInfo != info.end(); iInfo++)
    {
        int32_t s;
        char c[255];
        char *p = c;

        if (!receive(s))
            cerr << "Tarsus Init: Tarsus ERROR1" << endl;

        if (!receive(c, s))
            cerr << "Tarsus Init: Tarsus ERROR2" << endl;

        p += s;

        *p = 0;

        *iInfo = std::string(c);

        //cerr << "Tarsus: Got string: " << c << endl;
    }

    //- Parse Info - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //	The info packets now contain the channel names.
    //	Identify the channels with the various dof's.

    for (iInfo = info.begin(); iInfo != info.end(); iInfo++)
    {
        //	Extract the channel type

        int openBrace = iInfo->find('<');

        if (openBrace == iInfo->npos)
            cerr << "Tarsus Init: Bad Channel Id" << endl;

        int closeBrace = iInfo->find('>');

        if (closeBrace == iInfo->npos)
            cerr << "Tarsus Init: Bad Channel Id" << endl;

        closeBrace++;

        std::string Type = iInfo->substr(openBrace, closeBrace - openBrace);

        //	Extract the Name

        std::string Name = iInfo->substr(0, openBrace);

        int space = Name.rfind(' ');

        if (space != Name.npos)
            Name.resize(space);

        std::vector<std::string>::const_iterator iTypes;

        std::vector<MarkerChannel>::iterator iMarker = std::find(MarkerChannels.begin(), MarkerChannels.end(), Name);

        std::vector<BodyChannel>::iterator iBody = std::find(BodyChannels.begin(), BodyChannels.end(), Name);

        if (iMarker != MarkerChannels.end())
        {
            //	The channel is for a marker we already have.
            iTypes = std::find(ClientCodes::MarkerTokens.begin(), ClientCodes::MarkerTokens.end(), Type);
            if (iTypes != ClientCodes::MarkerTokens.end())
                iMarker->operator[](iTypes - ClientCodes::MarkerTokens.begin()) = iInfo - info.begin();
        }
        else if (iBody != BodyChannels.end())
        {
            //	The channel is for a body we already have.
            iTypes = std::find(ClientCodes::BodyTokens.begin(), ClientCodes::BodyTokens.end(), Type);
            if (iTypes != ClientCodes::BodyTokens.end())
                iBody->operator[](iTypes - ClientCodes::BodyTokens.begin()) = iInfo - info.begin();
        }
        else if ((iTypes = std::find(ClientCodes::MarkerTokens.begin(), ClientCodes::MarkerTokens.end(), Type))
                 != ClientCodes::MarkerTokens.end())
        {
            //	Its a new marker.
            MarkerChannels.push_back(MarkerChannel(Name));
            MarkerChannels.back()[iTypes - ClientCodes::MarkerTokens.begin()] = iInfo - info.begin();
        }
        else if ((iTypes = std::find(ClientCodes::BodyTokens.begin(), ClientCodes::BodyTokens.end(), Type))
                 != ClientCodes::BodyTokens.end())
        {
            //	Its a new body.
            BodyChannels.push_back(BodyChannel(Name));
            BodyChannels.back()[iTypes - ClientCodes::BodyTokens.begin()] = iInfo - info.begin();
        }
        else if (Type == "<F>")
        {
            FrameChannel = iInfo - info.begin();
        }
        else
        {
            //	It could be a new channel type.
        }
    }

    //- Get Data - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //	Get the data using the request/reply protocol.

    markerPositions.resize(MarkerChannels.size());

    bodyPositions.resize(BodyChannels.size());

    data.resize(info.size());
}

#ifdef HAVE_PTHREAD
void *Tarsus::startThread(void *th)
{
    Tarsus *t = (Tarsus *)th;
    t->mainLoop();
    return NULL;
}
#endif

Tarsus::~Tarsus()
{
#ifdef HAVE_PTHREAD
    pthread_cancel(trackerThread);
    pthread_join(trackerThread, 0);
#endif

    delete sock;
}

bool Tarsus::openTCPPort()
{
    Host *h = new Host(hostName);
    sock = new Socket(h, port, 10);
    lastTime = cover->frameRealTime() + 1;
    if (sock->get_id() < 0)
    {
        sock = NULL;
        return false;
    }
    return true;
}

void Tarsus::reset()
{
    delete sock;
    sock = NULL;
    sleep(1);
    int i = 0;
    while (!openTCPPort())
    {
        if ((i % 10) == 0)
        {
            cerr << "trying to open TCP connection on port " << port << " to " << hostName << endl;
        }
        i++;
    }
    initialize();
}

void
Tarsus::mainLoop()
{
#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN);
#endif
    while (1)
    {
        if (!poll())
        {
            usleep(20000);
        }
    }
}

bool
Tarsus::poll()
{
    if (sock == NULL)
    {
        reset();
        return false;
    }

    if ((cover->frameRealTime() - lastTime) < 0.02)
    {
        return false;
    }
    lastTime = cover->frameRealTime();

    pBuff = buff;
    *((uint32_t *)pBuff) = ClientCodes::EData;
    pBuff += sizeof(uint32_t);
    *((uint32_t *)pBuff) = ClientCodes::ERequest;
    pBuff += sizeof(uint32_t);

    if (sock->write(buff, pBuff - buff) != pBuff - buff)
    {
        cerr << "Error Requesting" << endl;
        reset();
        return false;
    }

    //	Get and check the packet header.

    int32_t packet;
    if (!receive(packet))
        cerr << "Error Recieving" << endl;

    int32_t type;

    if (!receive(type))
        cerr << "Error Recieving" << endl;

    if (type != ClientCodes::EReply)
        cerr << "Bad Packet" << endl;

    if (packet != ClientCodes::EData)
        cerr << "Bad Reply Type" << endl;

    if (!receive(size))
        cerr << "Tarsus Error3" << endl;

    if (size != info.size())
    {
        cerr << "Bad Data Packet: size=" << size << ", info.size()=" << info.size() << endl;

        if (size > 0)
        {
            char *dummy = new char[size];
            receive(dummy, size);
            delete[] dummy;
        }
        reset();
        return false;
    }

    //	Get the data.

    std::vector<double>::iterator iData;

    for (iData = data.begin(); iData != data.end(); iData++)
    {
        if (!receive(*iData))
            cerr << "receive ERROR" << endl;
    }

    //- Look Up Channels - - - - - - - - - - - - - - - - - - - - - - -
    //  Get the TimeStamp

    //double timestamp = data[FrameChannel];

    //	Get the channels corresponding to the markers.
    //	Y is up
    //	The values are in millimeters

    std::vector<MarkerChannel>::iterator iMarker;
    std::vector<MarkerData>::iterator iMarkerData;

    int i = 0;
    for (iMarker = MarkerChannels.begin(),
        iMarkerData = markerPositions.begin();
         iMarker != MarkerChannels.end(); iMarker++, iMarkerData++)
    {
        iMarkerData->X = data[iMarker->X];
        iMarkerData->Y = data[iMarker->Y];
        iMarkerData->Z = data[iMarker->Z];
        if (data[iMarker->O] > 0.5)
            iMarkerData->Visible = false;
        else
            iMarkerData->Visible = true;
        i++;
    }

    //	Get the channels corresponding to the bodies.
    //=================================================================
    //	The bodies are in global space
    //	The world is Z-up
    //	The translational values are in millimeters
    //	The rotational values are in radians
    //=================================================================

    std::vector<BodyChannel>::iterator iBody;
    std::vector<BodyData>::iterator iBodyData;

    for (iBody = BodyChannels.begin(),
        iBodyData = bodyPositions.begin();
         iBody != BodyChannels.end(); iBody++, iBodyData++)
    {

        iBodyData->TX = data[iBody->TX];
        iBodyData->TY = data[iBody->TY];
        iBodyData->TZ = data[iBody->TZ];

        //	The channel data is in the angle-axis form.
        //	The following converts this to a quaternion.
        //=============================================================
        //	An angle-axis is vector, the direction of which is the axis
        //	of rotation and the length of which is the amount of
        //	rotation in radians.
        //=============================================================

        double len, tmp;

        len = sqrt(data[iBody->RX] * data[iBody->RX] + data[iBody->RY] * data[iBody->RY] + data[iBody->RZ] * data[iBody->RZ]);

        iBodyData->QW = cos(len / 2.0);
        tmp = sin(len / 2.0);
        if (len < 1e-10)
        {
            iBodyData->QX = data[iBody->RX];
            iBodyData->QY = data[iBody->RY];
            iBodyData->QZ = data[iBody->RZ];
        }
        else
        {
            iBodyData->QX = data[iBody->RX] * tmp / len;
            iBodyData->QY = data[iBody->RY] * tmp / len;
            iBodyData->QZ = data[iBody->RZ] * tmp / len;
        }

        //	The following converts angle-axis to a rotation matrix.

        double c, s, x, y, z;

        if (len < 1e-15)
        {
            iBodyData->GlobalRotation[0][0] = iBodyData->GlobalRotation[1][1] = iBodyData->GlobalRotation[2][2] = 1.0;
            iBodyData->GlobalRotation[0][1] = iBodyData->GlobalRotation[0][2] = iBodyData->GlobalRotation[1][0] = iBodyData->GlobalRotation[1][2] = iBodyData->GlobalRotation[2][0] = iBodyData->GlobalRotation[2][1] = 0.0;
        }
        else
        {
            x = data[iBody->RX] / len;
            y = data[iBody->RY] / len;
            z = data[iBody->RZ] / len;

            c = cos(len);
            s = sin(len);

            iBodyData->GlobalRotation[0][0] = c + (1 - c) * x * x;
            iBodyData->GlobalRotation[0][1] = (1 - c) * x * y + s * (-z);
            iBodyData->GlobalRotation[0][2] = (1 - c) * x * z + s * y;
            iBodyData->GlobalRotation[1][0] = (1 - c) * y * x + s * z;
            iBodyData->GlobalRotation[1][1] = c + (1 - c) * y * y;
            iBodyData->GlobalRotation[1][2] = (1 - c) * y * z + s * (-x);
            iBodyData->GlobalRotation[2][0] = (1 - c) * z * x + s * (-y);
            iBodyData->GlobalRotation[2][1] = (1 - c) * z * y + s * x;
            iBodyData->GlobalRotation[2][2] = c + (1 - c) * z * z;
        }

        // now convert rotation matrix to nasty Euler angles (yuk)
        // you could convert direct from angle-axis to Euler if you wish

        //	'Look out for angle-flips, Paul...'
        //  Algorithm: GraphicsGems II - Matrix Techniques VII.1 p 320
        assert(fabs(iBodyData->GlobalRotation[0][2]) <= 1);
        iBodyData->EulerY = asin(-iBodyData->GlobalRotation[2][0]);

        /*	if(fabs(cos(y)) >
            std::numeric_limits<double>::epsilon() ) 	// cos(y) != 0 Gimbal-Lock
         {
            iBodyData->EulerX = atan2(iBodyData->GlobalRotation[2][1], iBodyData->GlobalRotation[2][2]);
            iBodyData->EulerZ = atan2(iBodyData->GlobalRotation[1][0], iBodyData->GlobalRotation[0][0]);
         }
         else
         {
            iBodyData->EulerZ = 0;
            iBodyData->EulerX = atan2(iBodyData->GlobalRotation[0][1], iBodyData->GlobalRotation[1][1]);
         }*/
    }

    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //  The marker and body data now resides in the arrays
    //	markerPositions & bodyPositions.

    //cerr << "Frame: " << timestamp << std::endl;

    return true;
}

void Tarsus::getPositionMatrix(unsigned int station, float *x, float *y, float *z, float *m00, float *m01, float *m02, float *m10, float *m11, float *m12, float *m20, float *m21, float *m22)
{

#ifndef HAVE_PTHREAD
    poll();
#endif

    unsigned int chanNum = station;

    const char *stationName = NULL;
    if (stationNames.size() > station)
    {
        stationName = stationNames[station];
    }

    bool stationFound = false;
    if (stationName)
    {
        for (unsigned int i = 0; i < BodyChannels.size(); i++)
        {
            if (!strncmp(BodyChannels.at(i).Name.c_str(), stationName, strlen(stationName)))
            {
                chanNum = i;
#ifndef VERBOSE
                if (cover->debugLevel(5))
                {
#endif
                    cerr << "Tarsus: " << BodyChannels.at(i).Name << " -> " << i << endl;
#ifndef VERBOSE
                }
#endif
                stationFound = true;
                break;
            }
        }
        if (!stationFound)
        {
#ifndef VERBOSE
            if (cover->debugLevel(4))
            {
#endif
                cerr << "Tarsus: station name \"" << stationName << "\" not found, available stations are:";
                for (unsigned int i = 0; i < BodyChannels.size(); i++)
                {
                    cerr << " \"" << BodyChannels.at(i).Name.c_str() << "\"";
                }
                cerr << endl;
#ifndef VERBOSE
            }
#endif
        }
    }

    if (!stationFound)
    {
        if (station >= BodyChannels.size())
        {
#ifndef VERBOSE
            if (cover->debugLevel(4))
            {
#endif
                cerr << "Tarsus: no valid channel for station " << station << endl;
#ifndef VERBOSE
            }
#endif
            *x = *y = *z = 0.0;
            *m00 = 1.0;
            *m01 = 0.0;
            *m02 = 0.0;
            *m10 = 0.0;
            *m11 = 1.0;
            *m12 = 0.0;
            *m20 = 0.0;
            *m21 = 0.0;
            *m22 = 1.0;
            return;
        }

        BodyChannel chan = BodyChannels.at(station);
#ifdef VERBOSE
        cerr << station << " is " << chan.Name << endl;
#endif
        if (chan.Name[0] == 'w' && (chanNum == 0))
        {
            chanNum = 1;
        }
        if (chan.Name[0] == 'W' && (chanNum == 0))
        {
            chanNum = 1;
        }
        if (chan.Name[0] == 'g' && (chanNum == 1))
        {
            chanNum = 0;
        }
    }

    if (chanNum >= bodyPositions.size())
    {
        if (cover->debugLevel(4))
        {
            cerr << "Tarsus: invalid channel " << chanNum << "for station " << station << endl;
        }
        return;
    }

    BodyData body = bodyPositions.at(chanNum);
    *x = body.TX;
    *y = body.TY;
    *z = body.TZ;

    *m00 = body.GlobalRotation[0][0];
    *m01 = body.GlobalRotation[1][0];
    *m02 = body.GlobalRotation[2][0];

    *m10 = body.GlobalRotation[0][1];
    *m11 = body.GlobalRotation[1][1];
    *m12 = body.GlobalRotation[2][1];

    *m20 = body.GlobalRotation[0][2];
    *m21 = body.GlobalRotation[1][2];
    *m22 = body.GlobalRotation[2][2];
}

int Tarsus::getNumMarkers()
{
    return markerPositions.size();
}

bool Tarsus::getMarker(int index, float *pos)
{
    pos[0] = markerPositions[index].X;
    pos[1] = markerPositions[index].Y;
    pos[2] = markerPositions[index].Z;
    //fprintf(stderr, "i=%d: %d - (%f %f %f)\n", index, int(markerPositions[index].Visible), pos[0], pos[1], pos[2]);
    return markerPositions[index].Visible;
}
