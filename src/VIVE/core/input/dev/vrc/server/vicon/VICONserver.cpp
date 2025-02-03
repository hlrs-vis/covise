/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-----------------------------------------------------------------------------
//
//
//
//
//
//
//
//
//
//
//-----------------------------------------------------------------------------

#include <covise/covise.h>
#include <algorithm>

#include <math.h>
#ifdef __WINDOWS
#include <windows.h>
#else
#define SOCKET int
#define INVALID_SOCKET -1
#define SOCKET_ERROR -1
#include <sys/types.h>
#ifndef WIN32
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/poll.h>
#endif

#define closesocket close
#endif
#include "ClientCodes.h"

#ifndef WIN32
#include <sys/time.h>
#endif
#include <math.h>

#include <util/SerialCom.h>
#include <util/UDP_Sender.h>
#include <util/ArgsParser.h>
using namespace covise;
//static const char *VICONServerVersion="1.0";
/// ++++ Forward-declaration of functions defined later in the text ++++
// show help
void displayHelp(const char *progName);
void showbuffer(unsigned char *bytes, UDP_Sender &sender, int stationID);
void sigHandler(int signo)
{
    fprintf(stderr, "Signal %d caught by Handler\n", signo);
    exit(0);
}

void showbuffer(unsigned char *bytes, UDP_Sender &sender, int stationID)
{
    char sendbuffer[2048];
    bytes[3] = '\0';
    sprintf(sendbuffer, "VRC %d %3ld [0.0 0.0 0.0] - [0 0 0 0 0 0 0 0 0] - [0 0]",
            stationID, 0x7fL & ~strtol((const char *)bytes + 1, NULL, 16));
    cerr << sendbuffer << endl;
    //   fprintf(stderr,"%s\n",sendbuffer);
    sender.send(sendbuffer, strlen(sendbuffer) + 1);
}

//-----------------------------------------------------------------------------
//	The recv call may return with a half-full buffer.
//	revieve keeps going until the buffer is actually full.

bool receive(SOCKET Socket, char *pBuffer, int BufferSize)
{
    char *p = pBuffer;
    char *e = pBuffer + BufferSize;

    int result;

    while (p != e)
    {
        struct pollfd fd;
        fd.fd = Socket;
        fd.events = POLLIN;
        poll(&fd, 1, -1);
        result = read(Socket, p, e - p);
        if (result != -1)
        {
            p += result;
        }
    }

    return true;
}

//	There are also some helpers to make the code a little less ugly.

bool receive(SOCKET Socket, long int &Val)
{
    return receive(Socket, (char *)&Val, sizeof(Val));
}

bool receive(SOCKET Socket, unsigned long int &Val)
{
    return receive(Socket, (char *)&Val, sizeof(Val));
}

bool receive(SOCKET Socket, double &Val)
{
    return receive(Socket, (char *)&Val, sizeof(Val));
}

//-----------------------------------------------------------------------------

void initializeSockets()
{

#ifdef __WINDOWS
    //- Initialisation - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //  Windows-specific initialisation.
    WORD wVersionRequested;
    WSADATA wsaData;
    wVersionRequested = MAKEWORD(2, 0);
    if (WSAStartup(wVersionRequested, &wsaData) != 0)
    {
        cout << "Socket Initialization Error" << endl;
        return 1;
    }
#endif
}

SOCKET createSocket(const char *viconHost, int viconPort)
{

    SOCKET SocketHandle = INVALID_SOCKET;

    struct protoent *pProtocolInfoEntry;
    char *protocol;
    int type;

    protocol = (char *)"tcp";
    type = SOCK_STREAM;

    pProtocolInfoEntry = getprotobyname(protocol);
    assert(pProtocolInfoEntry);

    if (pProtocolInfoEntry)
        SocketHandle = socket(PF_INET, type, pProtocolInfoEntry->p_proto);

    if (SocketHandle == INVALID_SOCKET)
    {
        cout << "Socket Creation Error" << endl;
        return 1;
    }

    struct hostent *pHostInfoEntry;
    struct sockaddr_in Endpoint;

    memset(&Endpoint, 0, sizeof(Endpoint));
    Endpoint.sin_family = AF_INET;
    Endpoint.sin_port = htons(viconPort);

    pHostInfoEntry = gethostbyname(viconHost);

    if (pHostInfoEntry)
    {
        memcpy(&Endpoint.sin_addr, pHostInfoEntry->h_addr, pHostInfoEntry->h_length);
    }
    else
    {
        Endpoint.sin_addr.s_addr = inet_addr(viconHost);
    }

    if (Endpoint.sin_addr.s_addr == INADDR_NONE)
    {
        cout << "Bad Address" << endl;
        return 1;
    }

    //	Create Socket

    int result = connect(SocketHandle, (struct sockaddr *)&Endpoint, sizeof(Endpoint));
    fcntl(SocketHandle, F_SETFL, FNDELAY);
    if (result == SOCKET_ERROR)
    {
        cout << "Failed to create Socket" << endl;
#ifdef __WINDOWS
        int e = 0;
        e = WSAGetLastError();
#endif
        return -1;
    }
    return SocketHandle;
}

#define MAXBODIES 10

int main(int argc, char *argv[])
{
    int VICONSOCKET;

    initializeSockets();

    ArgsParser arg(argc, argv);

    //at least one stations has to be connected
    if (argc < 2
        || 0 == strcasecmp(argv[1], "-h")
        || 0 == strcasecmp(argv[1], "--help"))
    {
        displayHelp(argv[0]);
        exit(-1);
    }

    const char *target = arg.getOpt("-t", "--target", "localhost:7777");
    const char *source = arg.getOpt("-s", "--source", NULL);
    const char *portStr = arg.getOpt("-p", "--port", "800");

    if (NULL == source)
    {
        displayHelp(argv[0]);
        exit(-1);
    }
    int numArgs = arg.numArgs();
    if (numArgs < 1)
    {
        displayHelp(argv[0]);
        exit(-1);
    }

    VICONSOCKET = atoi(portStr);

    int i;
    //We get the data in the form Brille:root
    //We map them to id's, this is done by saying on the commandline
    // Brille:root=12 Flystick:
    char *mapStrings[MAXBODIES];
    int mapValues[MAXBODIES];
    char *mapString;
    int mapValue;
    int mapCounter = 0;
    for (i = 0; i < numArgs; i++)
    {
        char buf[4096];
        strcpy(buf, arg[i]);
        char *number = strstr(buf, "=");
        if (NULL != number)
        {
            *number++ = '\0';
            mapValue = atoi(number);
            mapString = new char[1 + strlen(buf)];
            strcpy(mapString, buf);
            mapStrings[mapCounter] = mapString;
            mapValues[mapCounter] = mapValue;
            mapCounter++;
        }
    }

    printf("\n");
    printf("  +-----------------------------------------------------+\n");
    printf("  + VRC VICONserver 1.0         (C) 2005 VISENSO GmbH   +\n");
    printf("  +-----------------------------------------------------+\n");
    printf("  + Settings:                                           +\n");
    printf("  +   UDP Target:        %-30s +\n", target);
    printf("  +   VICON server:      %-30s +\n", source);
    printf("  +   VICON port:        %-30d +\n", VICONSOCKET);
    printf("  +-----------------------------------------------------+\n");
    printf("  + Target Mapping:                                     +\n");
    for (i = 0; i < mapCounter; i++)
    {
        printf("  +  COVER #%-2d <-- %-30s       +\n", mapValues[i], mapStrings[i]);
    }
    printf("  +-----------------------------------------------------+\n\n");

    SOCKET SocketHandle = createSocket(source, VICONSOCKET);
    if (-1 == SocketHandle)
    {
        cout << "Failed to open socket" << endl;
        exit(-1);
    }

    UDP_Sender sender(target);
    if (sender.isBad())
    {
        cerr << "Could not start UDP server to "
             << target << endl;
        return -1;
    }

    //	A connection with the Vicon Realtime system is now open.
    //	The following section implements the new Computer Graphics Client interface.

    try
    {
        vector<string> info;
        const int bufferSize = 2040;
        char buff[bufferSize];
        char *pBuff;

        //- Get Info - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        //	Request the channel information

        pBuff = buff;

        *((long int *)pBuff) = ClientCodes::EInfo;
        pBuff += sizeof(long int);
        *((long int *)pBuff) = ClientCodes::ERequest;
        pBuff += sizeof(long int);

        if (send(SocketHandle, buff, pBuff - buff, 0) == SOCKET_ERROR)
            throw string("Error Requesting");

        long int packet;
        long int type;

        if (!receive(SocketHandle, packet))
            throw string("Error Recieving");

        if (!receive(SocketHandle, type))
            throw string("Error Recieving");

        if (type != ClientCodes::EReply)
            throw string("Bad Packet");

        if (packet != ClientCodes::EInfo)
            throw string("Bad Reply Type");

        long int size;

        if (!receive(SocketHandle, size))
            throw string();

        info.resize(size);

        vector<string>::iterator iInfo;

        for (iInfo = info.begin(); iInfo != info.end(); iInfo++)
        {
            long int s;
            char c[255];
            char *p = c;

            if (!receive(SocketHandle, s))
                throw string();

            if (!receive(SocketHandle, c, s))
                throw string();

            p += s;

            *p = 0;

            *iInfo = string(c);
        }

        //- Parse Info - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        //	The info packets now contain the channel names.
        //	Identify the channels with the various dof's.

        vector<MarkerChannel> MarkerChannels;
        vector<BodyChannel> BodyChannels;
        //int   FrameChannel=0;

        for (iInfo = info.begin(); iInfo != info.end(); iInfo++)
        {
            //	Extract the channel type

            int openBrace = iInfo->find('<');

            if (openBrace == iInfo->npos)
                throw string("Bad Channel Id");

            int closeBrace = iInfo->find('>');

            if (closeBrace == iInfo->npos)
                throw string("Bad Channel Id");

            closeBrace++;

            string Type = iInfo->substr(openBrace, closeBrace - openBrace);
            //cout << "My Type is" << Type << endl;
            //	Extract the Name

            string Name = iInfo->substr(0, openBrace);
            //cout << "My name is" << Name << endl;
            int space = Name.rfind(' ');

            if (space != Name.npos)
                Name.resize(space);

            vector<MarkerChannel>::iterator iMarker;
            vector<BodyChannel>::iterator iBody;
            vector<string>::const_iterator iTypes;

            iMarker = find(MarkerChannels.begin(),
                           MarkerChannels.end(), Name);

            iBody = find(BodyChannels.begin(), BodyChannels.end(), Name);

            if (iMarker != MarkerChannels.end())
            {
                //	The channel is for a marker we already have.
                //cout << "iMarkerName:" << iMarker->Name << endl;
                iTypes = find(ClientCodes::MarkerTokens.begin(), ClientCodes::MarkerTokens.end(), Type);
                if (iTypes != ClientCodes::MarkerTokens.end())
                    iMarker->operator[](iTypes - ClientCodes::MarkerTokens.begin()) = iInfo - info.begin();
            }
            else if (iBody != BodyChannels.end())
            {
                //	The channel is for a body we already have.
                //cout << "iBodyName:" << iBody->Name << endl;
                iTypes = find(ClientCodes::BodyTokens.begin(), ClientCodes::BodyTokens.end(), Type);
                if (iTypes != ClientCodes::BodyTokens.end())
                    iBody->operator[](iTypes - ClientCodes::BodyTokens.begin()) = iInfo - info.begin();
            }
            else if ((iTypes = find(ClientCodes::MarkerTokens.begin(), ClientCodes::MarkerTokens.end(), Type))
                     != ClientCodes::MarkerTokens.end())
            {
                //	Its a new marker.
                //cout << "iTypesNameA:" << Type << " " << Name <<" "  << endl;
                MarkerChannels.push_back(MarkerChannel(Name));
                MarkerChannels.back()[iTypes - ClientCodes::MarkerTokens.begin()] = iInfo - info.begin();
            }
            else if ((iTypes = find(ClientCodes::BodyTokens.begin(), ClientCodes::BodyTokens.end(), Type))
                     != ClientCodes::BodyTokens.end())
            {
                //	Its a new body.
                //cout << "iTypesNameB:" << Type << " " << Name << endl;
                cout << "Found Body: \"" << Name << "\"" << endl;
                BodyChannels.push_back(BodyChannel(Name));
                BodyChannels.back()[iTypes - ClientCodes::BodyTokens.begin()] = iInfo - info.begin();
            }
            else if (Type == "<F>")
            {
                //FrameChannel = iInfo - info.begin();
            }
            else
            {
                //	It could be a new channel type.
            }
        }

        //- Get Data - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        //	Get the data using the request/reply protocol.

        int i;

        vector<double> data;
        data.resize(info.size());

        //double timestamp;

        vector<MarkerData> markerPositions;
        markerPositions.resize(MarkerChannels.size());

        vector<BodyData> bodyPositions;
        bodyPositions.resize(BodyChannels.size());

        cout << endl;

        //// dump only once a secong
        long lastSec = 0;

        while (true)
        {
            struct timeval tv;
            gettimeofday(&tv, NULL);
            bool dumpPacket;
            if (tv.tv_sec > lastSec)
            {
                dumpPacket = true;
                lastSec = tv.tv_sec;
            }
            else
                dumpPacket = false;

            usleep(30000);
            pBuff = buff;
            *((long int *)pBuff) = ClientCodes::EData;
            pBuff += sizeof(long int);
            *((long int *)pBuff) = ClientCodes::ERequest;
            pBuff += sizeof(long int);

            if (send(SocketHandle, buff, pBuff - buff, 0) == SOCKET_ERROR)
                throw string("Error Requesting");

            long int packet;
            long int type;

            //	Get and check the packet header.

            if (!receive(SocketHandle, packet))
                throw string("Error Recieving");

            if (!receive(SocketHandle, type))
                throw string("Error Recieving");

            if (type != ClientCodes::EReply)
                throw string("Bad Packet");

            if (packet != ClientCodes::EData)
                throw string("Bad Reply Type");

            if (!receive(SocketHandle, size))
                throw string();

            if (size != info.size())
                throw string("Bad Data Packet");

            //	Get the data.

            vector<double>::iterator iData;

            for (iData = data.begin(); iData != data.end(); iData++)
            {
                if (!receive(SocketHandle, *iData))
                    throw string();
            }

            //- Look Up Channels - - - - - - - - - - - - - - - - - - - - - - -
            //  Get the TimeStamp

            //timestamp = data[FrameChannel];

            //	Get the channels corresponding to the markers.
            //	Y is up
            //	The values are in millimeters

            // vector< MarkerChannel >::iterator iMarker;
            // vector< MarkerData >::iterator iMarkerData;
            //          for(	iMarker = MarkerChannels.begin(),
            //                    iMarkerData = markerPositions.begin();
            //                 iMarker != MarkerChannels.end(); iMarker++, iMarkerData++)
            //          {
            //             iMarkerData->X = data[iMarker->X];
            //             iMarkerData->Y = data[iMarker->Y];
            // //FIXME: War vorher so				iMarkerData->Y = data[iMarker->Z];
            //             iMarkerData->Z = data[iMarker->Z];
            //             if(data[iMarker->O] > 0.5)
            //                iMarkerData->Visible = false;
            //             else
            //                iMarkerData->Visible = true;
            //             cout << "MX;MY;MZ:" << iMarker->Name << " " << iMarkerData->X<<" " <<iMarkerData->Y << " " <<
            //                iMarkerData->Z << endl;
            //          }

            //	Get the channels corresponding to the bodies.
            //=================================================================
            //	The bodies are in global space
            //	The world is Z-up
            //	The translational values are in millimeters
            //	The rotational values are in radians
            //=================================================================

            vector<BodyChannel>::iterator iBody;
            vector<BodyData>::iterator iBodyData;
            //int bodyCount=0;                         //in the house
            for (iBody = BodyChannels.begin(),
                iBodyData = bodyPositions.begin();
                 iBody != BodyChannels.end(); iBody++, iBodyData++)
            {
                //            int statID=stationID+bodyCount++;
                int statID = 0;
                //            cout << "bodyCount=" << bodyCount << endl;
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

                double c, s, x, y = 0.0, z;

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
                    //Original               s = sin(len);
                    s = -sin(len);

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
#define rot iBodyData->GlobalRotation
                char sendbuffer[2048];
                bool searching = true;
                //get the station ID from the body name
                for (i = 0; (i < mapCounter && searching); i++)
                {
                    //             if(0==strcmp(mapStrings[i], (char*)iBody->Name) {
                    if (iBody->Name == mapStrings[i])
                    {
                        statID = mapValues[i];
                        searching = false;
                    }
                }

                if (!searching)
                {
                    float x, y, z;
                    x = iBodyData->TX;
                    y = iBodyData->TY;
                    z = iBodyData->TZ;
                    if (x == 0.0 && y == 0.0 && z == 0.0)
                    {
                        x = 0.00001;
                        y = 0.00001;
                        z = 0.00001;
                    }
                    sprintf(sendbuffer, "VRC %d 0 [%6.1f %6.1f %6.1f] - [%6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f] - [0 0]", statID,
                            x, y, z,
                            rot[0][0], rot[0][1], rot[0][2],
                            rot[1][0], rot[1][1], rot[1][2],
                            rot[2][0], rot[2][1], rot[2][2]);
                    sender.send(sendbuffer, strlen(sendbuffer) + 1);
                    //cerr << iBody->Name  << endl;
                    if (dumpPacket)
                        cout << sendbuffer << " : " << iBody->Name << endl;
                }
#undef rot

                // now convert rotation matrix to nasty Euler angles (yuk)
                // you could convert direct from angle-axis to Euler if you wish

                //	'Look out for angle-flips, Paul...'
                //  Algorithm: GraphicsGems II - Matrix Techniques VII.1 p 320
                assert(fabs(iBodyData->GlobalRotation[0][2]) <= 1);
                iBodyData->EulerY = asin(-iBodyData->GlobalRotation[2][0]);

                if (fabs(cos(y)) > 0.00001) // cos(y) != 0 Gimbal-Lock
                {
                    iBodyData->EulerX = atan2(iBodyData->GlobalRotation[2][1], iBodyData->GlobalRotation[2][2]);
                    iBodyData->EulerZ = atan2(iBodyData->GlobalRotation[1][0], iBodyData->GlobalRotation[0][0]);
                }
                else
                {
                    iBodyData->EulerZ = 0;
                    iBodyData->EulerX = atan2(iBodyData->GlobalRotation[0][1], iBodyData->GlobalRotation[1][1]);
                }
            }

            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            //  The marker and body data now resides in the arrays
            //	markerPositions & bodyPositions.

            //         cout << "Frame: " << timestamp << endl;
        }
    }

    catch (const string &rMsg)
    {
        if (rMsg.empty())
            cout << "Error! Error! Error! Error! Error!" << endl;
        else
            cout << rMsg.c_str() << endl;
    }
    if (closesocket(SocketHandle) == SOCKET_ERROR)
    {
        cout << "Failed to close Socket" << endl;
        return 1;
    }

    return 0;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Help

// show help
void displayHelp(const char *progName)
{
    cout << progName << " [options]  targetName=coverID [ ... ]\n"
         << "\n"
         << "   coverID = Station ID for COVER's BUTTON_ADDR config\n"
         << "\n"
         << "Options:\n"
         << "\n"
         << "   -t <host:port>      set target to send tracking UDP packets\n"
         << "   --target=host:port  (default: localhost:7777)\n"
         << "\n"
         << "   -s <host>           set source to get TCP packets from\n"
         << "   --source=host       (no default value)"
         << "\n"
         << "   -p <portNo>         set port number the VICON server uses\n"
         << "   --port=portNo       (default: 800)\n"
         << "\n"
         << "Examples:\n"
         << "\n"
         << "   " << progName << " -s vicon3 Flystick:root=3\n\n"
         << "                       Read Vicon device \"Flystick:root\" from the host\n"
         << "                       vicon3 and send data to localhost:7777 with ID=3\n"
         << "\n"
         << "   " << progName << " -s gromit -t visenso:6666 Flystick:root=4 Brille:root=3\n\n"
         << "                       Read Vicon device from the host gromit and send data\n"
         << "                       of devices Flystick and Brille to Host \"visenso\"\n"
         << "                       Port 6666 with IDs 4 for Flystick and 3 for Brille\n"
         << endl;
}
