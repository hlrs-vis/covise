/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "EphemeralSky.h"
#include "SkySphere.h"

#include <osg/Vec3>

#include <cover/coVRConfig.h>
#include <cover/coVRPluginSupport.h>
#include <geodata/GeoData.h>

using namespace opencover;

static osg::BoundingSphere BOUNDING_SPHERE(osg::Vec3(0, 0, 0), 0);

class myCB : public osg::Node::ComputeBoundingSphereCallback
{
public:
    osg::BoundingSphere computeBound(const osg::Node &) const
    {
        return BOUNDING_SPHERE;
    }
};

void EphemeralSky::setTimeMode(EphemeralSky::TimeMode timeMode)
{
    m_timeMode = timeMode;
    if (m_timeMode == CURRENT_TIME)
    {
        m_timeEpoch = cover->frameTime();
    }
}

void EphemeralSky::setTime(double epoch)
{
    m_timeEpoch = epoch;
    m_timeMode = FIXED_TIME;
}

void EphemeralSky::setTimeLapseSpeed(double speed)
{
    m_timeLapseSpeed = speed;
}

EphemeralSky::EphemeralSky(opencover::ui::Group *menuGroup, osg::Group *parent)
{
    m_ephemerisModel = new osgEphemeris::EphemerisModel;
    m_ephemerisModel->setSkyDomeMirrorSouthernHemisphere(false);
    m_ephemerisModel->setSkyDomeCenter(osg::Vec3(0, 0, 0));
    m_ephemerisModel->setSkyDomeRadius(1.0);
    m_ephemerisModel->setComputeBoundingSphereCallback(new myCB());
    m_ephemerisModel->setInitialBound(BOUNDING_SPHERE);
    m_ephemerisModel->setCullingActive(false);
    m_ephemerisModel->setTurbidity(10.f);

    setTimeMode(CURRENT_TIME);

    m_timeModesList = new ui::SelectionList(menuGroup, "TimeMode");
    m_timeModesList->setList({
        "Auto",
        "Time lapse",
        "Fixed time",
        "Fixed time: Now",
        "Fixed time: Noon (UTC)",
        "Fixed time: Midnight (UTC)",
    });
    m_timeModesList->setCallback([this](int selection)
        {
            auto item = m_timeModesList->items()[selection];

            if (item == "Auto")
            {
                setTimeMode(CURRENT_TIME);
            }
            else if (item == "Fixed time: Now")
            {
                setTime(cover->frameTime());
            }
            else if (item == "Fixed time: Noon (UTC)")
            {
                long t = cover->frameTime();
                setTime((t - t % 86400) + 43200);
            }
            else if (item == "Fixed time: Midnight (UTC)")
            {
                long t = cover->frameTime();
                setTime((t - t % 86400));
            }
            else if (item == "Time lapse")
            {
                setTimeMode(TIMELAPSE);
            }
            else if (item == "Fixed time")
            {
                setTimeMode(FIXED_TIME);
            } });

    parent->addChild(m_ephemerisModel.get());
}

EphemeralSky::~EphemeralSky()
{
    delete m_timeModesList;

    while (m_ephemerisModel->getNumParents())
        m_ephemerisModel->getParent(0)->removeChild(m_ephemerisModel);
}

void EphemeralSky::update()
{
    auto data = m_ephemerisModel->getEphemerisData();

    if (m_timeMode == TIMELAPSE)
    {
        m_timeEpoch += cover->frameDuration() * m_timeLapseSpeed;
    }
    else if (m_timeMode == CURRENT_TIME)
    {
        m_timeEpoch = cover->frameTime();
    }

    if (m_timeEpoch != m_oldTimeEpoch)
    {
        // TODO: debounce time update?
        m_oldTimeEpoch = m_timeEpoch;
        std::time_t t = m_timeEpoch;
        auto tm = std::gmtime(&t);
        if (tm)
        {
            auto &dt = data->dateTime;
            dt.setTimeZoneOffset(false, 0);
            dt.setYear(tm->tm_year + 1900);
            dt.setMonth(tm->tm_mon + 1);
            dt.setDayOfMonth(tm->tm_mday);
            dt.setHour(tm->tm_hour);
            dt.setMinute(tm->tm_min);
            dt.setSecond(tm->tm_sec);
        }
    }

    // Update location
    auto projectLocation = osg::Matrix::inverse(cover->getXformMat()).getTrans() / cover->getScale();
    auto global = GeoData::instance()->projectToGlobal(projectLocation);
    data->longitude = global.x();
    data->latitude = global.y();
    data->altitude = global.z();

    // TODO: what else to set?
    // data->turbidity = 1.0;
}
