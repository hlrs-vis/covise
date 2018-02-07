/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#ifndef DTRACK_DRIVER_H
#define DTRACK_DRIVER_H

#include <string>

#include "DTrackSDK.hpp"

/*

class DTrackDriver : public opencover::InputDevice
{
    //-------------------DTrack related stuff
    size_t m_numFlySticks; ///Number of DTrack flysticks
    size_t m_numBodies; ///Number of DTrack bodies
    size_t m_numHands;  ///Number of DTrack hands
    ssize_t m_flystickBase;
    ssize_t m_bodyBase;
    ssize_t m_handBase;
    std::vector<size_t> m_buttonBase;   //Button base indices for flysticks
    std::vector<size_t> m_valuatorBase;

    std::vector<size_t> m_handButtonBase; //Temp button base indices for hands


    bool init();
    virtual bool poll();
    void initArrays();

public:
    DTrackDriver(const std::string &name);
    virtual ~DTrackDriver();

    //==============Hardware related interface methods=====================
    bool updateBodyMatrix(size_t idx); ///get DTrack body matrix
    bool updateFlyStick(size_t idx); ///get flystick body matrix
    bool updateHand(size_t idx);	/// get Hand matrix and other data
};
*/
#endif
