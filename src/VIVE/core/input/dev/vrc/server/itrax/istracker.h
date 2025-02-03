/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          istracker.h  -  description
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

#include <isense.h>

#ifndef ISTRACKER_H
#define ISTRACKER_H

/**
 *@author Thomas Reichl                  #include "isense.h"

 */

class ISTracker
{
public:
    ISTracker(int TrackerPortNr);
    ~ISTracker();

    const bool getData(float &rotX, float &rotY, float &rotZ);
    const bool DeviceFound() const;

private:
    ISD_TRACKER_HANDLE handle;
    ISD_TRACKER_DATA_TYPE data;
    ISD_STATION_INFO_TYPE Station[ISD_MAX_STATIONS];
    ISD_CAMERA_DATA_TYPE cameraData;
    ISD_TRACKER_INFO_TYPE Tracker;

    bool m_bTrackerFound;
};
#endif
