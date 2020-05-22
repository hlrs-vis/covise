/* This file is part of COVISE.

  You can use it under the terms of the GNU Lesser General Public License
  version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

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
#include "DeviceSensor.h"

DeviceSensor::DeviceSensor(Device *d, osg::Node *n): coPickSensor(n)
{
    dev = d;
}
DeviceSensor::~DeviceSensor()
{
    if (active)
        disactivate();
}
void DeviceSensor::activate()
{
    dev->activate();
}
void DeviceSensor::disactivate()
{
    dev->disactivate();
}
