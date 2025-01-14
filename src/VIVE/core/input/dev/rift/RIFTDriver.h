/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * RIFTDriver.h
 *
 *  Created on: Feb 5, 2014
 *      Author: hpcwoess
 */

#ifndef RIFT_DRIVER_H
#define RIFT_DRIVER_H

#include <OpenThreads/Thread>
#include <osg/Matrix>
#include <string>

#include "OVR_CAPI.h"


#include <cover/input/inputdevice.h>
#include <util/coExport.h>

#if defined(input_rift_EXPORTS)
#define INPUT_RIFT_EXPORT COEXPORT
#else
#define INPUT_RIFT_EXPORT COIMPORT
#endif

/**
 * @brief The RIFTDriver class interacts with input hardware
 *
 * This class interacts with input hardware and stores the data
 * about all configured input hardware e.g. tracking systems,
 * button devices etc.
 *
 * Main interaction loop runs in its own thread
 */
class INPUT_RIFT_EXPORT RIFTDriver : public opencover::InputDevice
{
   

    bool init();
    virtual bool poll();

public:
    RIFTDriver(const std::string &name);
    virtual ~RIFTDriver();

    void setMatrix(osg::Matrix &m);

};
#endif
