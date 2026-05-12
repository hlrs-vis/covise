/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EPHEMERAL_SKY_H
#define EPHEMERAL_SKY_H

#include <osg/Group>
#include <osgEphemeris/EphemerisModel.h>

#include <cover/ui/Menu.h>
#include <cover/ui/Action.h>
#include <cover/ui/Owner.h>
#include <cover/ui/Slider.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Button.h>
#include <cover/ui/SelectionList.h>

class EphemeralSky
{
public:
    enum TimeMode
    {
        CURRENT_TIME,
        TIMELAPSE,
        FIXED_TIME,
    };

    EphemeralSky(opencover::ui::Group *menuGroup, osg::Group *parent);
    ~EphemeralSky();

    void update();

    void setTimeMode(TimeMode timeMode);
    void setTime(double epoch);
    void setTimeLapseSpeed(double speed);

private:
    osg::ref_ptr<osgEphemeris::EphemerisModel> m_ephemerisModel;
    opencover::ui::SelectionList *m_timeModesList;

    TimeMode m_timeMode = CURRENT_TIME;
    double m_timeEpoch = 0.0;
    double m_oldTimeEpoch = -1.0;
    double m_timeLapseSpeed = 3600.f; // an hour every second
};

#endif
