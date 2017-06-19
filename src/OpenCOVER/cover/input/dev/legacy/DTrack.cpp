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
 *	File			DTrack.cpp 				*
 *									*
 *	Description		DTrack optical tracking system interface class				*
 *									*
 *	Author			DUwe Woessner				*
 *									*
 *	Date			July 2001				*
 *									*
 *	Status			in dev					*
 *									*
 ************************************************************************/

#include "DTrack.h"
#ifndef STANDALONE
#endif
#include <util/common.h>
#include <util/unixcompat.h>

#include <stdio.h>
#include <string.h>

#ifndef WIN32
#include <sys/socket.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <netinet/in.h>
#endif

#include <iostream>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

using namespace covise;
using std::cout;
using std::cerr;
using std::endl;

DTrack::DTrack(int portnumber, const char *sendStr)
{
    port = portnumber;
    startStop = NULL;
    allocSharedMemoryData();
    running = false;

    if (openUDPPort())
    {
        mainLoop();
    }
    dataArrived = false;
    if (sendStr)
    {
        startStop = new covise::UDP_Sender(sendStr);
        sendStart();
    }
}

DTrack::~DTrack()
{
    running = false;
    fprintf(stderr, "~DTrack\n");
    // wait at most 3 seconds
    for (int i = 0; i < 300; ++i)
    {
        if (!isRunning())
            break;
        usleep(10000);
    }
    if (isRunning())
    {
        fprintf(stderr, "DTrack communication thread still running - exiting anyway\n");
    }
    if (startStop)
    {
        sendStop();
        delete startStop;
    }
    fprintf(stderr, "~DTrack done waiting for thread\n");
    if (sock > 0)
    {
#ifdef _WIN32
        closesocket(sock);
#else
        close(sock);
#endif
    }
}

void DTrack::sendStart()
{
    if (startStop)
    {
        dataArrived = false;
        startStop->send("dtrack 10 3"); // camera on
        //sleep(1);
        startStop->send("dtrack 31"); // start sending data
    }
}
void DTrack::sendStop()
{
    if (startStop)
    {
        startStop->send("dtrack 32"); // stop sending data
        //sleep(1);
        startStop->send("dtrack 10 0"); // camera off
    }
}

bool DTrack::openUDPPort()
{
    sockaddr_in any_adr;

    sock = -1;

    // CREATING UDP SOCKET
    sock = (int)socket(AF_INET, SOCK_DGRAM, 0);
    if (socket < 0)
    {
        fprintf(stderr, "socket creation failed\n");
        return false;
    }
    //fprintf(stderr, "socket created %d \n",sock);

    // FILL SOCKET ADRESS STRUCTURE
    memset((void *)&any_adr, 0, sizeof(any_adr));
    any_adr.sin_family = AF_INET;
    any_adr.sin_addr.s_addr = INADDR_ANY;
    any_adr.sin_port = htons(port);

    // BIND TO A LOCAL PROTOCOL PORT
    if (bind(sock, (sockaddr *)&any_adr, sizeof(any_adr)) < 0)
    {
        fprintf(stderr, "could not bind to port %d\n", port);
        return false;
    }
    //fprintf(stderr, "bind to port %d\n",port);
    return true;
}

void DTrack::receiveData()
{
    //sockaddr remote_adr;
    //socklen_t rlen;
    //rlen = sizeof(remote_adr);
    int n, recordid = -1, numBodys, numGloves, numFlysticks, i;
    numbytes = n = recvfrom(sock, rawdata, MAXBYTES, 0, 0, 0);
    //numbytes = n = recvfrom(sock, rawdata, 200, 0, NULL, NULL);

    if (n == MAXBYTES)
    {
        fprintf(stderr, "Message longer than %d bytes, increase MAXBYTES\n", MAXBYTES);
        return;
    }
    if (numbytes < 0)
    {
        fprintf(stderr, "socket %d MAXBYTES %d errno%d\n", sock, MAXBYTES, errno);
        perror("DTrack recvfrom failed");

        return;
    }
    else if (numbytes < 3)
    {
        fprintf(stderr, "short message \n");
        return;
    }
    c = rawdata;

    // ein neuer Farme faengt mit fr an
    if (strncmp(c, "fr", 2) != 0)
    {
        fprintf(stderr, "expected fieldrecord but got %s\n", rawdata);
        return;
    }
    numbytes -= 3;
    c += 3;
    int ret = sscanf(c, "%d", &recordid);
    if (ret != 1)
    {
        fprintf(stderr, "DTrack::receiveData: sscanf failed for recordid\n");
        return;
    }
    if (nextLine() < 0)
    {
        return;
    }

    // ab hier kommen alle Daten eines frames, das kann so aussehen:
    // 2 bodies
    // 6d <numBodies> [id_body0 qual] [pos angle] [matrix] [idbody_1 qual] [pos_angle] [matrix]
    //
    // 2 body 2 flysticks
    // 6d <numBodies> [id_body0 qual] [pos angle] [matrix] [idbody_1 qual] [pos_angle] [matrix]
    // 6df <numBodies> [id_body0 qual but] [pos angle] [matrix] [idbody_1 qual but] [pos_angle] [matrix]

    //fprintf(stderr,"DTrack::receiveData of Frame %d\n", recordid);
    //fprintf(stderr,"|--%s--|\n", c);

    dataArrived = true;
    char *bodyString = strstr(c, "6d ");
    char *gloveString = strstr(c, "gl ");
    char *flystick2String = strstr(c, "6df2");
    char *flystickString = strstr(c, "6df");

    if (bodyString)
    {
        c = bodyString;
        //fprintf(stderr,"found bodyString ---%s---\n", bodyString);
        //fprintf(stderr,"read bodyString\n");
        numBodys = 0;
        int ret = sscanf(c, "6d %d", &numBodys);
        if (ret != 1)
        {
            fprintf(stderr, "DTrack::receiveData: sscanf failed for numBodys\n");
        }
        for (i = 0; i < numBodys; i++)
        {
            int id;
            float quality;
            float x, y, z, h, p, r, m00, m01, m02, m10, m11, m12, m20, m21, m22;
            if (nextBlock() < 0)
            {
                fprintf(stderr, "unexpected end of data 1 %s\n", rawdata);
                return;
            }
            if (sscanf(c, "%d %f", &id, &quality) != 2)
            {
                fprintf(stderr, "unexpected end of data 2 %s\n", rawdata);
                fprintf(stderr, "i : %d    c:%s\n", i, c);
                return;
            }
            if (nextBlock() < 0)
            {
                fprintf(stderr, "unexpected end of data 3 %s\n", rawdata);
                return;
            }
            if (sscanf(c, "%f %f %f %f %f %f", &x, &y, &z, &h, &p, &r) != 6)
            {
                fprintf(stderr, "unexpected end of data 4 %s\n", rawdata);
                return;
            }
            if (nextBlock() < 0)
            {
                fprintf(stderr, "unexpected end of data 5 %s\n", rawdata);
                return;
            }
            if (sscanf(c, "%f %f %f %f %f %f %f %f %f", &m00, &m01, &m02, &m10, &m11, &m12, &m20, &m21, &m22) != 9)
            {
                fprintf(stderr, "unexpected end of data 6 %s\n", rawdata);
                return;
            }
            if (m00 != 0 || m01 != 0 || m02 != 0 || m10 != 0 || m11 != 0)
            {
                stationData[id].x = x;
                stationData[id].y = y;
                stationData[id].z = z;
                stationData[id].az = h;
                stationData[id].el = p;
                stationData[id].roll = r;
                stationData[id].quality = quality;
                stationData[id].matrix[0] = m00;
                stationData[id].matrix[1] = m01;
                stationData[id].matrix[2] = m02;
                stationData[id].matrix[3] = m10;
                stationData[id].matrix[4] = m11;
                stationData[id].matrix[5] = m12;
                stationData[id].matrix[6] = m20;
                stationData[id].matrix[7] = m21;
                stationData[id].matrix[8] = m22;
                stationData[id].button[0] = 0;
            }
            //fprintf(stderr,"body stationData[%d].x=%f\n", id, stationData[id].x);
        }
    }
    if (flystick2String)
    {
        c = flystick2String;
        //fprintf(stderr,"found flystickString ---%s---\n", flystickString);
        //fprintf(stderr,"read flystickString\n");

        numFlysticks = 0;
        int numFlysticksCalibrated;
        int ret = sscanf(c, "6df2 %d %d", &numFlysticksCalibrated, &numFlysticks);
        if (ret != 2)
        {
            fprintf(stderr, "DTrack::receiveData: sscanf failed for numFlysticks\n");
        }
        for (i = 0; i < numFlysticks; i++)
        {
            int id;
            float quality;
            float x, y, z, m00, m01, m02, m10, m11, m12, m20, m21, m22;
            int numButtons;
            int bt;
            int valCount = 0;
            if (nextBlock() < 0)
            {
                fprintf(stderr, "unexpected end of data 1 %s\n", rawdata);
                return;
            }
            if (sscanf(c, "%d %f %d %d", &id, &quality, &numButtons, &valCount) != 4)
            {
                fprintf(stderr, "unexpected end of data 2 %s\n", rawdata);
                return;
            }
            if (nextBlock() < 0)
            {
                fprintf(stderr, "unexpected end of data 3 %s\n", rawdata);
                return;
            }
            if (sscanf(c, "%f %f %f", &x, &y, &z) != 3)
            {
                fprintf(stderr, "unexpected end of data 4 %s\n", rawdata);
                return;
            }
            if (nextBlock() < 0)
            {
                fprintf(stderr, "unexpected end of data 5 %s\n", rawdata);
                return;
            }
            if (sscanf(c, "%f %f %f %f %f %f %f %f %f", &m00, &m01, &m02, &m10, &m11, &m12, &m20, &m21, &m22) != 9)
            {
                fprintf(stderr, "unexpected end of data 6 %s\n", rawdata);
                return;
            }
            if (nextBlock() < 0)
            {
                fprintf(stderr, "unexpected end of data 7 %s\n", rawdata);
                return;
            }
            int n = 0;
            int btNumbers = (numButtons / 32) + 1;
            for (n = 0; n < btNumbers; n++)
            {
                if (sscanf(c, "%d", &bt) != 1)
                {
                    fprintf(stderr, "unexpected end of data 8 %s\n", rawdata);
                    return;
                }
                if (n < DTRACK_MAX_BUTTONS)
                    stationData[id + MAXBODYS].button[n] = bt;
                while (*c != '\0' && *c != ']' && *c != ' ' && *c != '\t' && *c != '\r' && *c != '\n')
                    c++;
            }
            for (n = 0; n < valCount; n++)
            {
                float value;
                if (sscanf(c, "%f", &value) != 1)
                {
                    fprintf(stderr, "unexpected end of data 9 %s\n", rawdata);
                    return;
                }
                if (n < DTRACK_MAX_VALUATORS)
                    stationData[id + MAXBODYS].valuators[n] = value;
                while (*c != '\0' && *c != ']' && *c != ' ' && *c != '\t' && *c != '\r' && *c != '\n')
                    c++;
            }
            if (m00 != 0 || m01 != 0 || m02 != 0 || m10 != 0 || m11 != 0)
            {
                stationData[id + MAXBODYS].x = x;
                stationData[id + MAXBODYS].y = y;
                stationData[id + MAXBODYS].z = z;
                stationData[id + MAXBODYS].az = 0;
                stationData[id + MAXBODYS].el = 0;
                stationData[id + MAXBODYS].roll = 0;
                stationData[id + MAXBODYS].quality = quality;
                stationData[id + MAXBODYS].matrix[0] = m00;
                stationData[id + MAXBODYS].matrix[1] = m01;
                stationData[id + MAXBODYS].matrix[2] = m02;
                stationData[id + MAXBODYS].matrix[3] = m10;
                stationData[id + MAXBODYS].matrix[4] = m11;
                stationData[id + MAXBODYS].matrix[5] = m12;
                stationData[id + MAXBODYS].matrix[6] = m20;
                stationData[id + MAXBODYS].matrix[7] = m21;
                stationData[id + MAXBODYS].matrix[8] = m22;
            }
            //fprintf(stderr,"flystick stationData[%d].x=%f button=%d\n", id+MAXBODYS, stationData[id+MAXBODYS].x, stationData[id+MAXBODYS].button);
        }
    }
    else if (flystickString)
    {
        c = flystickString;
        //fprintf(stderr,"found flystickString ---%s---\n", flystickString);
        //fprintf(stderr,"read flystickString\n");

        numFlysticks = 0;
        int ret = sscanf(c, "6df %d", &numFlysticks);
        if (ret != 1)
        {
            fprintf(stderr, "DTrack::receiveData: sscanf failed for numFlysticks\n");
        }
        for (i = 0; i < numFlysticks; i++)
        {
            int id;
            float quality;
            float x, y, z, h, p, r, m00, m01, m02, m10, m11, m12, m20, m21, m22;
            int bt;
            if (nextBlock() < 0)
            {
                fprintf(stderr, "unexpected end of data 1 %s\n", rawdata);
                return;
            }
            if (sscanf(c, "%d %f %d", &id, &quality, &bt) != 3)
            {
                fprintf(stderr, "unexpected end of data 2 %s\n", rawdata);
                return;
            }
            if (nextBlock() < 0)
            {
                fprintf(stderr, "unexpected end of data 3 %s\n", rawdata);
                return;
            }
            if (sscanf(c, "%f %f %f %f %f %f", &x, &y, &z, &h, &p, &r) != 6)
            {
                fprintf(stderr, "unexpected end of data 4 %s\n", rawdata);
                return;
            }
            if (nextBlock() < 0)
            {
                fprintf(stderr, "unexpected end of data 5 %s\n", rawdata);
                return;
            }
            if (sscanf(c, "%f %f %f %f %f %f %f %f %f", &m00, &m01, &m02, &m10, &m11, &m12, &m20, &m21, &m22) != 9)
            {
                fprintf(stderr, "unexpected end of data 6 %s\n", rawdata);
                return;
            }
            if (m00 != 0 || m01 != 0 || m02 != 0 || m10 != 0 || m11 != 0)
            {
                stationData[id + MAXBODYS].x = x;
                stationData[id + MAXBODYS].y = y;
                stationData[id + MAXBODYS].z = z;
                stationData[id + MAXBODYS].az = h;
                stationData[id + MAXBODYS].el = p;
                stationData[id + MAXBODYS].roll = r;
                stationData[id + MAXBODYS].quality = quality;
                stationData[id + MAXBODYS].matrix[0] = m00;
                stationData[id + MAXBODYS].matrix[1] = m01;
                stationData[id + MAXBODYS].matrix[2] = m02;
                stationData[id + MAXBODYS].matrix[3] = m10;
                stationData[id + MAXBODYS].matrix[4] = m11;
                stationData[id + MAXBODYS].matrix[5] = m12;
                stationData[id + MAXBODYS].matrix[6] = m20;
                stationData[id + MAXBODYS].matrix[7] = m21;
                stationData[id + MAXBODYS].matrix[8] = m22;
                stationData[id + MAXBODYS].button[0] = bt;
            }
            //fprintf(stderr,"flystick stationData[%d].x=%f button=%d\n", id+MAXBODYS, stationData[id+MAXBODYS].x, stationData[id+MAXBODYS].button);
        }
    }

    if (gloveString)
    {
        c = gloveString;

        numGloves = 0;
        int ret = sscanf(c, "gl %d", &numGloves);
        if (ret != 1)
        {
            fprintf(stderr, "DTrack::receiveData: sscanf failed for numGloves. gloveString : %s\n", gloveString);
        }
        for (i = 0; i < numGloves; i++)
        {
            int lr = 0, nf = 0, id = 0;
            float quality = 0;
            float x = 0, y = 0, z = 0;
            float m00, m01, m02, m10, m11, m12, m20, m21, m22;
            if (nextBlock() < 0)
            {
                fprintf(stderr, "unexpected end of data 1 %s\n", rawdata);
                return;
            }
            if (sscanf(c, "%d %f %d %d", &id, &quality, &lr, &nf) != 4)
            {
                fprintf(stderr, "unexpected end of data 2 %s\n", rawdata);
                return;
            }
            if (nextBlock() < 0)
            {
                fprintf(stderr, "unexpected end of data 3 %s\n", rawdata);
                return;
            }
            if (sscanf(c, "%f %f %f", &x, &y, &z) != 3)
            {
                fprintf(stderr, "unexpected end of data 4 %s\n", rawdata);
                return;
            }
            if (nextBlock() < 0)
            {
                fprintf(stderr, "unexpected end of data 5 %s\n", rawdata);
                return;
            }
            if (sscanf(c, "%f %f %f %f %f %f %f %f %f", &m00, &m01, &m02, &m10, &m11, &m12, &m20, &m21, &m22) != 9)
            {
                fprintf(stderr, "unexpected end of data 6 %s\n", rawdata);
                return;
            }
            if (m00 != 0 || m01 != 0 || m02 != 0 || m10 != 0 || m11 != 0)
            {
                stationData[id + MAXBODYS + MAXFLYSTICKS].lr = lr;
                stationData[id + MAXBODYS + MAXFLYSTICKS].nf = nf;
                stationData[id + MAXBODYS + MAXFLYSTICKS].x = x;
                stationData[id + MAXBODYS + MAXFLYSTICKS].y = y;
                stationData[id + MAXBODYS + MAXFLYSTICKS].z = z;
                stationData[id + MAXBODYS + MAXFLYSTICKS].quality = (float)quality;
                stationData[id + MAXBODYS + MAXFLYSTICKS].matrix[0] = m00;
                stationData[id + MAXBODYS + MAXFLYSTICKS].matrix[1] = m01;
                stationData[id + MAXBODYS + MAXFLYSTICKS].matrix[2] = m02;
                stationData[id + MAXBODYS + MAXFLYSTICKS].matrix[3] = m10;
                stationData[id + MAXBODYS + MAXFLYSTICKS].matrix[4] = m11;
                stationData[id + MAXBODYS + MAXFLYSTICKS].matrix[5] = m12;
                stationData[id + MAXBODYS + MAXFLYSTICKS].matrix[6] = m20;
                stationData[id + MAXBODYS + MAXFLYSTICKS].matrix[7] = m21;
                stationData[id + MAXBODYS + MAXFLYSTICKS].matrix[8] = m22;
            }
            int i;
            for (i = 0; i < nf; i++)
            {
                float ro;
                float lo;
                float alphaom;
                float lm;
                float alphami;
                float li;
                if (nextBlock() < 0)
                {
                    fprintf(stderr, "unexpected end of data 3 %d %s\n", i, rawdata);
                    return;
                }
                if (sscanf(c, "%f %f %f", &x, &y, &z) != 3)
                {
                    fprintf(stderr, "unexpected end of data 4 %d %s\n", i, rawdata);
                    return;
                }
                if (nextBlock() < 0)
                {
                    fprintf(stderr, "unexpected end of data 5 %d %s\n", i, rawdata);
                    return;
                }
                if (sscanf(c, "%f %f %f %f %f %f %f %f %f", &m00, &m01, &m02, &m10, &m11, &m12, &m20, &m21, &m22) != 9)
                {
                    fprintf(stderr, "unexpected end of data 6 %d %s\n", i, rawdata);
                    return;
                }
                if (nextBlock() < 0)
                {
                    fprintf(stderr, "unexpected end of data 5 %d %s\n", i, rawdata);
                    return;
                }
                if (sscanf(c, "%f %f %f %f %f %f", &ro, &lo, &alphaom, &lm, &alphami, &li) != 6)
                {
                    fprintf(stderr, "unexpected end of data 6 %d %s\n", i, rawdata);
                    return;
                }
                stationData[id + MAXBODYS + MAXFLYSTICKS].finger[i].x = x;
                stationData[id + MAXBODYS + MAXFLYSTICKS].finger[i].y = y;
                stationData[id + MAXBODYS + MAXFLYSTICKS].finger[i].z = z;
                stationData[id + MAXBODYS + MAXFLYSTICKS].finger[i].matrix[0] = m00;
                stationData[id + MAXBODYS + MAXFLYSTICKS].finger[i].matrix[1] = m01;
                stationData[id + MAXBODYS + MAXFLYSTICKS].finger[i].matrix[2] = m02;
                stationData[id + MAXBODYS + MAXFLYSTICKS].finger[i].matrix[3] = m10;
                stationData[id + MAXBODYS + MAXFLYSTICKS].finger[i].matrix[4] = m11;
                stationData[id + MAXBODYS + MAXFLYSTICKS].finger[i].matrix[5] = m12;
                stationData[id + MAXBODYS + MAXFLYSTICKS].finger[i].matrix[6] = m20;
                stationData[id + MAXBODYS + MAXFLYSTICKS].finger[i].matrix[7] = m21;
                stationData[id + MAXBODYS + MAXFLYSTICKS].finger[i].matrix[8] = m22;
                stationData[id + MAXBODYS + MAXFLYSTICKS].finger[i].ro = ro;
                stationData[id + MAXBODYS + MAXFLYSTICKS].finger[i].lo = lo;
                stationData[id + MAXBODYS + MAXFLYSTICKS].finger[i].alphaom = alphaom;
                stationData[id + MAXBODYS + MAXFLYSTICKS].finger[i].lm = lm;
                stationData[id + MAXBODYS + MAXFLYSTICKS].finger[i].alphami = alphami;
                stationData[id + MAXBODYS + MAXFLYSTICKS].finger[i].li = li;
            }
        }
    }

    if (!bodyString && !flystickString && !flystick2String && !gloveString)
    {
        //fprintf(stderr, "expected 6d, 6df, 6df2 or gl data but got %s\n",rawdata);
    }
}

int DTrack::nextLine()
{
    //cerr << "*c = " << *c << endl;
    while ((numbytes > 0) && (*c != '\r') && (*c != '\n'))
    {
        //cerr << "*c = " << *c << endl;
        c++;
        numbytes--;
    }
    //cerr << "*c = " << *c << endl;
    while ((numbytes > 0) && ((*c == '\r') || (*c == '\n')))
    {
        //cerr << "*c = " << *c << endl;
        c++;
        numbytes--;
    }
    //cerr << "*c = " << *c << endl;
    return numbytes;
}

int DTrack::nextBlock()
{
    while ((numbytes > 0) && (*c != '['))
    {
        c++;
        numbytes--;
    }
    while ((numbytes > 0) && (*c == '['))
    {
        c++;
        numbytes--;
    }
    return numbytes;
}

void DTrack::run()
{
    while (running)
    {
        receiveData();
    }
}

void
DTrack::mainLoop()
{
    running = true;
    start();
}

void
DTrack::allocSharedMemoryData()
{
    stationData = new DTrackOutputData[MAXSENSORS];
    memset((void *)stationData, 0, MAXSENSORS * sizeof(DTrackOutputData));
    int i;
    for (i = 0; i < MAXSENSORS; i++)
    {
        stationData[i].x = 0;
        stationData[i].y = 0;
        stationData[i].z = 0;
        stationData[i].matrix[0] = 1;
        stationData[i].matrix[4] = 1;
        stationData[i].matrix[8] = 1;
    }
}

void DTrack::getPositionMatrix(int station, float *x, float *y, float *z, float *m00, float *m01, float *m02, float *m10, float *m11, float *m12, float *m20, float *m21, float *m22)
{
    *x = stationData[station].x;
    *y = stationData[station].y;
    *z = stationData[station].z;

    *m00 = stationData[station].matrix[0];
    *m01 = stationData[station].matrix[1];
    *m02 = stationData[station].matrix[2];

    *m10 = stationData[station].matrix[3];
    *m11 = stationData[station].matrix[4];
    *m12 = stationData[station].matrix[5];

    *m20 = stationData[station].matrix[6];
    *m21 = stationData[station].matrix[7];
    *m22 = stationData[station].matrix[8];
}

const DTrack::FingerData *DTrack::getFingerData(int station) const
{
    if (station < MAXSENSORS)
        return this->stationData[station].finger;
    else
        return 0;
}

void
DTrack::getButtons(int station, unsigned int *status)
{
    *status = stationData[station].button[0];
}

bool
DTrack::getButton(int station, int buttonNumber)
{
    if (buttonNumber < DTRACK_MAX_BUTTONS * 32)
        return (stationData[station].button[buttonNumber / 32] & (1 << (buttonNumber % 32))) != 0;
    else
        return false;
}

pid_t DTrack::getSlavePID()
{
    return slavePid_;
}
