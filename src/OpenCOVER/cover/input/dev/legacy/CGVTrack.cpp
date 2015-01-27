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
*	File			CGVTrack.cpp 				*
*									*
*	Description		CGVTrack optical tracking system interface class				*
*									*
*	Author			DUwe Woessner				*
*									*
*	Date			July 2007				*
*									*
*	Status			in dev					*
*									*
************************************************************************/

#include "CGVTrack.h"
#include <config/CoviseConfig.h>
#include <osg/Matrix>

#define INCLUDE_PACKETNUMBER true

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

#include <cstring>
#include <iostream>
#include <errno.h>
using std::cout;
using std::cerr;
using std::endl;
using namespace covise;

CGVTrack::CGVTrack(const char *host, int portnumber)
{
    port = portnumber;
    char portStr[100];
    sprintf(portStr, "%d", port);

    stationData = new stationDataType[CGVMAXSENSORS];
    for (int i = 0; i < CGVMAXSENSORS; i++)
    {
        stationData[i].name = NULL;
        char key[1024];
        sprintf(key, "COVER.Input.CGV.StationName:%d", i);
        std::string name = coCoviseConfig::getEntry(key);
        if (!name.empty())
        {
            stationData[i].name = new char[name.length() + 1];
            strcpy(stationData[i].name, name.c_str());
        }
    }
    if (!client.setup(host, portStr))
    {
        cout << "Couldn't start the client with the given network address." << endl;
    }
}

CGVTrack::~CGVTrack()
{
    client.close();
    delete[] stationData;
}

void CGVTrack::receiveData()
{
    bool ret;
    bool oldDataAvailable;
    vMarker.clear();
    vTarget.clear();
    ret = client.requestData(All, INCLUDE_PACKETNUMBER);
    if (!ret)
    {
        cout << "request Data failed" << endl;
        return;
    }
    ret = client.receiveData(oldDataAvailable);
    if (!ret && !oldDataAvailable)
    {
        cout << "receive Data failed" << endl;
        return;
    }
    else if (oldDataAvailable)
    {
        cout << "Timeout getting new data, but old Data available" << endl;
    }
    std::string data = client.getData();
    std::string what;
    std::istringstream sstr(data);
    int len;
    sstr >> len;
    //std::cout << "Length: "<<len<<std::endl;

    while (sstr)
    {
        char c;
        sstr >> c;
        // std::cout << c;
    }
    //std::cout << std::endl;

    while (sstr)
    {
        sstr >> what;
        //std::cout << "What: "<<what<<std::endl;
        if (what == "MarkerPos")
        {
            //std::cout << "MarkerPos"<<std::endl;
            sstr >> vMarker;
        }
        else if (what == "Target")
        {
            //std::cout << "Target"<<std::endl;
            sstr >> vTarget;
        }
    }

    for (CGVOpticalTracking::RecognizedTargetList::iterator iter = vTarget.begin(); iter != vTarget.end(); ++iter)
    {
        std::cout << "read recognized target: " << iter->name << std::endl;

        for (int i = 0; i < CGVMAXSENSORS; i++) // search name
        {
            if (strcmp(stationData[i].name, iter->name.c_str()) == 0)
            {
                osg::Quat quat(iter->quatRotation.v_v.x, iter->quatRotation.v_v.y, iter->quatRotation.v_v.z, iter->quatRotation.v_s);

                stationData[i].mat.makeRotate(quat);
                stationData[i].mat(3, 0) = iter->position.x;
                stationData[i].mat(3, 1) = iter->position.y;
                stationData[i].mat(3, 2) = iter->position.z;

                break;
            }
        }
    }
}

void *CGVTrack::continuousThread(void *data)
{
    CGVTrack *dt = (CGVTrack *)data;

    while (1)
    {
        dt->receiveData();
    }

    return NULL;
}

void
CGVTrack::mainLoop()
{
#ifdef HAVE_PTHREAD
    pthread_t trackerThread;
    if (pthread_create(&trackerThread, NULL, continuousThread, this))
    {
        cerr << "failed to create trackerThread: " << strerror(errno) << endl;
    }
#else
    {
        cerr << "CGVTrack requires pthread support." << endl;
        exit(1);
    }
#endif
}

void CGVTrack::getPositionMatrix(int station, float *x, float *y, float *z, float *m00, float *m01, float *m02, float *m10, float *m11, float *m12, float *m20, float *m21, float *m22)
{
    *x = stationData[station].mat(3, 0);
    *y = stationData[station].mat(3, 1);
    *z = stationData[station].mat(3, 2);

    *m00 = stationData[station].mat(0, 0);
    *m01 = stationData[station].mat(0, 1);
    *m02 = stationData[station].mat(0, 2);

    *m10 = stationData[station].mat(1, 0);
    *m11 = stationData[station].mat(1, 1);
    *m12 = stationData[station].mat(1, 2);

    *m20 = stationData[station].mat(2, 0);
    *m21 = stationData[station].mat(2, 1);
    *m22 = stationData[station].mat(2, 2);
}

void
CGVTrack::getButtons(int, unsigned int *status)
{
    *status = 0; //stationData[station].button[0];
}

bool
CGVTrack::getButton(int /*station*/, int /*buttonNumber*/)
{
    /*if(buttonNumber < DTRACK_MAX_BUTTONS*32)
      return (stationData[station].button[buttonNumber/32]&(1<<(buttonNumber%32)))!=0;
   else
      return false;*/
    return false;
}
