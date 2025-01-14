/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TARSUS_H_
#define _CO_TARSUS_H_
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
#define MAXSENSORS 22
#define MAXBODYS 10
#define MAXFLYSTICKS 10
#define MAXBYTES 4000

#include <net/covise_socket.h>
#include <util/common.h>

#include <sys/types.h>
#define MAX_NUMBER_MICE 100
#ifdef WIN32
#else
#include <unistd.h>
#endif
#include <util/coTypes.h>
#include "ClientCodes.h"

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

class INPUT_LEGACY_EXPORT Tarsus
{
private:
    int port;
    int numMice;
    char *hostName;
    covise::Socket *sock;
    std::vector<std::string> info;
    char buff[2040];
    char *pBuff;
    uint32_t size;
    double lastTime;
    std::vector<MarkerChannel> MarkerChannels;
    std::vector<BodyChannel> BodyChannels;
    std::vector<BodyData> bodyPositions;
    std::vector<MarkerData> markerPositions;
    std::vector<double> data;
    std::vector<char *> stationNames;
    int FrameChannel;

    bool openTCPPort();
#ifdef HAVE_PTHREAD
    pthread_t trackerThread;
    static void *startThread(void *);
#endif
    bool poll();
    void mainLoop();
    void initialize();

    bool receive(char *pBuffer, int BufferSize);
    bool receive(int32_t &Val);
    bool receive(uint32_t &Val);
    bool receive(double &Val);
#ifdef WIN32
//TemporaryWindow window;
#endif

public:
    Tarsus(int portnumber, const char *host);
    ~Tarsus();
    void setStationName(unsigned int station, const char *beginningOfName);
    void getPositionMatrix(unsigned int station, float *x, float *y, float *z, float *m00, float *m01, float *m02, float *m10, float *m11, float *m12, float *m20, float *m21, float *m22);
    void reset();
    int getNumMarkers();
    bool getMarker(int index, float *pos);
};
#endif
