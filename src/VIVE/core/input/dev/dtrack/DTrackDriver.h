/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * inputhdw.h
 *
 *  Created on: Dec 9, 2014
 *      Author: svnvlad
 */

#ifndef DTRACK_DRIVER_H
#define DTRACK_DRIVER_H


#include <vsg/maths/mat4.h>
#include <vsg/maths/vec3.h>
#include <string>

#include "DTrackSDK.hpp"

#include <core/input/inputdevice.h>

using namespace vive;

/**
 * @brief The DTrackDriver class interacts with input hardware
 *
 * This class interacts with input hardware and stores the data
 * about all configured input hardware e.g. tracking systems,
 * button devices etc.
 *
 * Main interaction loop runs in its own thread
 */
class DTrackDriver : public vive::InputDevice
{
    //-------------------DTrack related stuff
    DTrackSDK *dt; ///ART DTrack SDK class
    size_t m_numFlySticks; ///Number of DTrack flysticks
    size_t m_numBodies; ///Number of DTrack bodies
    size_t m_numHands;  ///Number of DTrack hands
    ssize_t m_flystickBase;
    ssize_t m_bodyBase;
    ssize_t m_handBase;
    int m_dtrack_port; 
    std::string m_dtrack_serverHost;
    int m_dtrack_serverPort;
    std::vector<size_t> m_buttonBase;   //Button base indices for flysticks
    std::vector<size_t> m_valuatorBase;

    std::vector<size_t> m_handButtonBase; //Temp button base indices for hands

    virtual bool poll();
    void initArrays();

    int timeoutcount = 0;

public:
    DTrackDriver(const std::string &name);
    virtual ~DTrackDriver();

    //==============Hardware related interface methods=====================
    bool updateBodyMatrix(size_t idx); ///get DTrack body matrix
    bool updateFlyStick(size_t idx); ///get flystick body matrix
    bool updateHand(size_t idx);	/// get Hand matrix and other data
};
#endif
