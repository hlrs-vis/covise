/* This file is part of COVISE.

  You can use it under the terms of the GNU Lesser General Public License
  version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef _Luft_DeviceSensor_H
#define _Luft_DeviceSensor_H
/****************************************************************************\
 **                                                          (C)2020 HLRS  **
 **                                                                        **
 ** Description: OpenCOVER Plug-In for reading Luftdaten sensor data       **
 **                                                                        **
 **                                                                        **
 ** Author: Leyla Kern                                                     **
 **                                                                        **
 ** History:                                                               **
 ** April 2020  v1                                                         **
 **                                                                        **
 **                                                                        **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
#include <PluginUtil/coSensor.h>
#include "Device.h"

using namespace covise;
using namespace opencover;


class DeviceSensor : public coPickSensor
{
private:
    Device *dev;

public:
    DeviceSensor(Device *h, osg::Node *n);
    ~DeviceSensor();

    void activate();
    void disactivate();
};

#endif

