/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          istracker.cpp  -  description
                             -------------------
    begin                : Sam Mär 8 2003
    copyright            : (C) 2003 by Thomas Reichl
    email                : reichl@ifb.tu-graz.ac.at
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "istracker.h"

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include "types.h"

#define ESC 0X1B
#define CAMERA_TRACKER 0

ISTracker::ISTracker(int TrackerPortNr)
    : m_bTrackerFound(false)
{
    //    WORD station = 1;
    Bool verbose = true;

    /* Detect first tracker. If you have more than one InterSense device and
      would like to have a specific tracker, connected to a known port,
      initialized first, then enter the port number instead of 0. Otherwise,
      tracker connected to the rs232 port with lower number is found first */

    handle = ISD_OpenTracker((Hwnd)NULL, TrackerPortNr, FALSE, verbose);

    /* Check value of handle to see if tracker was located */
    if (handle < 1)
    {
        printf("Failed to detect InterSense tracking device\n");
        return;
    }

    /* Get tracker configuration info */
    ISD_GetTrackerConfig(handle, &Tracker, verbose);

    /* Clear station configuration info to make sure GetAnalogData and other flags are FALSE */
    memset((void *)Station, 0, sizeof(Station));

    /* General procedure for changing any setting is to first retrieve current
      configuration, make the change, and then apply it. Calling
      ISD_GetStationConfig is important because you only want to change
      some of the settings, leaving the rest unchanged. */
    /*
       if( Tracker.TrackerType == ISD_PRECISION_SERIES )
       {
           for( station = 1; station <= 4; station++ )
           {
               // fill ISD_STATION_INFO_TYPE structure with current station configuration
               if( !ISD_GetStationConfig( handle,
                   &Station[station-1], station, verbose )) break;

               if( CAMERA_TRACKER && Tracker.TrackerModel == ISD_IS900 )
               {
   Station[station-1].GetCameraData = TRUE;

   // apply new configuration
   if( !ISD_SetStationConfig( handle,
   &Station[station-1], station, verbose )) break;
   }
   }
   }
   */
    // setting angleformat to Euler
    if (ISD_GetStationConfig(handle, &Station[0], 0, verbose))
    {
        Station[0].AngleFormat = ISD_EULER;
        ISD_SetStationConfig(handle, &Station[0], 0, verbose);
    }

    m_bTrackerFound = true;
}

const bool ISTracker::getData(float &rotX, float &rotY, float &rotZ)
{
    ISD_GetData(handle, &data);
    //  ISD_GetCameraData( handle, &cameraData );

    if (ISD_GetCommInfo(handle, &Tracker))
    {
        rotX = data.Station[0].Orientation[0];
        rotY = data.Station[0].Orientation[1];
        rotZ = data.Station[0].Orientation[2];
    }

    return true;
}

const bool ISTracker::DeviceFound() const
{
    return m_bTrackerFound;
}

ISTracker::~ISTracker()
{
    ISD_CloseTracker(handle);
}
