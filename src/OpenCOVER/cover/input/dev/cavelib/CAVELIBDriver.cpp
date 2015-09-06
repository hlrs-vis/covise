/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

/*
* CAVELIBDriver.cpp
*
*  Created on: Feb 5, 2014
*      Author: hpcwoess
*/

#include "CAVELIBDriver.h"

#include <config/CoviseConfig.h>

#include <iostream>
#include <chrono>
#include <osg/Matrix>
#include <util/unixcompat.h>

#include <OpenVRUI/osg/mathUtils.h> //for MAKE_EULER_MAT
#include <stdio.h>
using namespace std;
using namespace covise;

CAVELIBDriver::CAVELIBDriver(const std::string &config)
    : InputDevice(config)
{       
    key = coCoviseConfig::getInt("SHMID", configPath(), 4126);
#ifdef WIN32
    //HANDLE handle;
    HANDLE filemap;
    char tmp_str[512];
    sprintf(tmp_str, "Global\\%d", key);
    /* while((handle = CreateFile(tmp_str, GENERIC_READ|GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
    NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL)) == INVALID_HANDLE_VALUE)
    {
    cerr << "CreateFile file " << key << " does not exist\n";
    cerr << "waiting\n";
    Sleep(1000);
    } 
    filemap = CreateFileMapping(handle, NULL, PAGE_READWRITE, 0, sizeof(struct TRACKD_TRACKING), NULL);*/

    filemap = OpenFileMapping(FILE_MAP_READ, FALSE, tmp_str);
    if (!(CaveLibTracker = (struct TRACKD_TRACKING *)MapViewOfFile(filemap, FILE_MAP_READ, 0, 0, sizeof(struct TRACKD_TRACKING))))
    {
        fprintf(stderr, "Could not attach shared memory key %x = %d for Cavelib TRACKER\n", key, key);
        CaveLibTracker=NULL;
    }
#else
    tracker_shmid = shmget(key, sizeof(struct TRACKD_TRACKING),
        PERMS | IPC_CREAT);
    if (tracker_shmid < 0)
    {
        fprintf(stderr, "Could get shared memory key %x = %d for Cavelib TRACKER\n", key, key);
        CaveLibTracker=NULL;
    }
    CaveLibTracker = (struct TRACKD_TRACKING *)shmat(tracker_shmid, (char *)0, 0);
    if (CaveLibTracker == (struct TRACKD_TRACKING *)-1)
    {
        fprintf(stderr, "Could attach shared memory key %x = %d for Cavelib TRACKER\n", key, key);
        CaveLibTracker=NULL;
    }
#endif

    std::string unit = coCoviseConfig::getEntry("unit", configPath(), "feet");
    if (unit == "feet")
        scaleFactor = 304.8;
    else if (unit == "mm")
        scaleFactor = 1.0;
    else if (unit == "m")
        scaleFactor = 1000.0;
    else if (unit == "cm")
        scaleFactor = 10.0;
    else if (unit == "dm")
        scaleFactor = 100.0;
    else if (unit == "inch")
        scaleFactor = 25.4;
    else
    {
        scaleFactor = 1.0;
        sscanf(unit.c_str(), "%f", &scaleFactor);
    }
    Yup = coCoviseConfig::isOn("Yup",configPath(), true);
    //fprintf(stderr, " \n\n\n\n Unit: %s factor: %f\n\n\n\n\n", unit.c_str(), scaleFactor);
    CaveLibWandController = coCoviseConfig::getInt("controller",configPath(), 0);

    key = coCoviseConfig::getInt("wandSHMID", configPath(), 4127);

#ifdef WIN32

    sprintf(tmp_str, "Global\\%d", key);

    filemap = OpenFileMapping(FILE_MAP_READ, FALSE, tmp_str);
    if (!(CaveLibWand = (TRACKD_WAND *)MapViewOfFile(filemap, FILE_MAP_READ, 0, 0, sizeof(struct TRACKD_WAND))))
    {
        fprintf(stderr, "Could not attach shared memory key %x = %d for Cavelib WAND\n", key, key);
        CaveLibWand=NULL;
    }
#else
    tracker_shmid = shmget(key, sizeof(struct TRACKD_WAND),
        PERMS | IPC_CREAT);
    if (tracker_shmid < 0)
    {
        fprintf(stderr, "Could access shared memory key %x = %d for Cavelib WAND\n", key, key);
        CaveLibWand=NULL;
    }
    CaveLibWand = (struct TRACKD_WAND *)shmat(tracker_shmid, (char *)0, 0);

#endif
    if (CaveLibWand == (struct TRACKD_WAND *)-1)
    {
        fprintf(stderr, "Could attach shared memory key %x = %d for Cavelib WAND\n", key, key);
        CaveLibWand=NULL;
    }
    long wandsize = (long)sizeof(struct TRACKD_WAND);


    m_bodyMatricesValid.resize(TRACKD_MAX_SENSORS);
    m_bodyMatrices.resize(TRACKD_MAX_SENSORS);
    //printf("wand %ld\n", wandsize);

}

//====================END of init section============================

CAVELIBDriver::~CAVELIBDriver()
{
}

//==========================main loop =================


bool CAVELIBDriver::needsThread() const
{

    return false;
}

void CAVELIBDriver::update()
{
    m_mutex.lock();
    

    if(CaveLibWand!=NULL)
    {
        int numValuators = CaveLibWand->controller[CaveLibWandController].num_valuators;
        int numCaveButtons = CaveLibWand->controller[CaveLibWandController].num_buttons;
        if(m_buttonStates.size() < numCaveButtons)
            m_buttonStates.resize(numCaveButtons);
        if(m_valuatorValues.size() < numValuators)
            m_valuatorValues.resize(numValuators);
        for (int i = 0; i < numCaveButtons; i++)
        {
            m_buttonStates[i]=CaveLibWand->controller[CaveLibWandController].button[i]!=0;
        }
        for (int i = 0; i < numValuators; i++)
        {
            m_valuatorValues[i]=CaveLibWand->controller[CaveLibWandController].valuator[i];
        }
    }
    


    if (CaveLibTracker)
    {
        for(int i=0;i<TRACKD_MAX_SENSORS;i++)
        {
            float x = CaveLibTracker->sensor[i].x;
            float y = CaveLibTracker->sensor[i].y;
            float z = CaveLibTracker->sensor[i].z;
            float h = CaveLibTracker->sensor[i].azim;
            float p = CaveLibTracker->sensor[i].elev;
            float r = CaveLibTracker->sensor[i].roll;
            //fprintf(stderr, "H: %f   ",h);
            //fprintf(stderr, "P: %f   ",p);
            //fprintf(stderr, "R: %f\n",r);
            // you can't use this because hpr means something
            // different int Performer than OpenGL mat.makeEuler(h,p,r);
            osg::Matrix H, P, R;
            if (Yup)
            {
                H.makeRotate((h / 180.0f) * M_PI, 0, 1, 0); // H + rot y
                P.makeRotate((p / 180.0f) * M_PI, 1, 0, 0); // P + rot x
                R.makeRotate((r / 180.0f) * M_PI, 0, 0, 1); // R + rot z
            }
            else //Z-Up
            {
                H.makeRotate((h / 180.0f) * M_PI, 0, 0, 1); // H + rot z
                P.makeRotate((p / 180.0f) * M_PI, 1, 0, 0); // P + rot x
                R.makeRotate((r / 180.0f) * M_PI, 0, 1, 0); // R + rot y
            }
            // MAT = R*P*H
            osg::Matrix mat = R;
            mat.postMult(P);
            mat.postMult(H);

            x *= scaleFactor; //  POSITION in some unit, default feet
            y *= scaleFactor;
            z *= scaleFactor; // now in mm

            mat(3,0)=x;
            mat(3,1)=y;
            mat(3,2)=z;
            m_bodyMatricesValid[i]=true;
            m_bodyMatrices[i]=mat;
        }
    }

    m_mutex.unlock();
    InputDevice::update();
}

INPUT_PLUGIN(CAVELIBDriver)
