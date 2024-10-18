/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#include "Sky.h"
SkyPlugin *SkyPlugin::plugin = NULL;

#include <cover/coVRTui.h>
#include <cover/coVRConfig.h>
#include <cover/VRSceneGraph.h>
#ifdef WIN32
#include <sys/timeb.h>
#else
#include <sys/time.h>
#endif

void VrmlNodeSky::initFields(VrmlNodeSky *node, vrml::VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
                     exposedField("enabled", node->d_enabled, [node](auto f){
                        SkyPlugin::plugin->setShowSky(node->d_enabled.get());
                     }),
                     exposedField("timeLapse", node->d_timeLapse, [node](auto f){
                        SkyPlugin::plugin->setTimeLapse(node->d_timeLapse.get());
                     }),
                     exposedField("currentTime", node->d_currentTime, [node](auto f){
                        SkyPlugin::plugin->setCurrentTime(node->d_currentTime.get());
                     }),
                     exposedField("year", node->d_year, [node](auto f){
                        SkyPlugin::plugin->setYear(node->d_year.get());
                     }),
                     exposedField("month", node->d_month, [node](auto f){
                        SkyPlugin::plugin->setMonth(node->d_month.get());
                     }),
                     exposedField("day", node->d_day, [node](auto f){
                        SkyPlugin::plugin->setDay(node->d_day.get());
                     }),
                     exposedField("hour", node->d_hour, [node](auto f){
                        SkyPlugin::plugin->setHour(node->d_hour.get());
                     }),
                     exposedField("minute", node->d_minute, [node](auto f){
                        SkyPlugin::plugin->setMinute(node->d_minute.get());
                     }),
                     exposedField("radius", node->d_radius, [node](auto f){
                        SkyPlugin::plugin->setRadius(node->d_radius.get() * 1000);
                     }),
                     exposedField("latitude", node->d_latitude, [node](auto f){
                        SkyPlugin::plugin->setLatitude(node->d_latitude.get());
                     }),
                     exposedField("longitude", node->d_longitude, [node](auto f){
                        SkyPlugin::plugin->setLongitude(node->d_longitude.get());
                     }),
                     exposedField("altitude", node->d_altitude, [node](auto f){
                        SkyPlugin::plugin->setAltitude(node->d_altitude.get());
                     }));
    if (t)
    {
        t->addEventIn("set_time", VrmlField::SFTIME);
    }
}

const char *VrmlNodeSky::name()
{
    return "Sky";
}

VrmlNodeSky::VrmlNodeSky(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , d_enabled(true)
    , d_timeLapse(false)
    , d_currentTime(true)
    , d_year(2006)
    , d_month(4)
    , d_day(7)
    , d_hour(13)
    , d_minute(12)
    , d_radius(coVRConfig::instance()->farClip() * 0.8)
    , d_latitude(48.6)
    , d_longitude(9.0008)
    , d_altitude(400)
{

    SkyPlugin::plugin->setShowSky(d_enabled.get());
    SkyPlugin::plugin->setTimeLapse(d_timeLapse.get());
    setModified();
}

VrmlNodeSky::VrmlNodeSky(const VrmlNodeSky &n)
    : VrmlNodeChild(n)
    , d_enabled(n.d_enabled)
    , d_timeLapse(n.d_timeLapse)
    , d_currentTime(n.d_currentTime)
    , d_year(n.d_year)
    , d_month(n.d_month)
    , d_day(n.d_day)
    , d_hour(n.d_hour)
    , d_minute(n.d_minute)
    , d_radius(n.d_radius)
    , d_latitude(n.d_latitude)
    , d_longitude(n.d_longitude)
    , d_altitude(n.d_altitude)
{

    setModified();
}

VrmlNodeSky *VrmlNodeSky::toSky() const
{
    return (VrmlNodeSky *)this;
}

void VrmlNodeSky::eventIn(double timeStamp,
                          const char *eventName,
                          const VrmlField *fieldValue)
{
    if (strcmp(eventName, "set_time") == 0)
    {
    }
    if (strcmp(eventName, "enabled") == 0)
    {
        d_enabled.set(fieldValue->toSFBool()->get());
        SkyPlugin::plugin->setShowSky(fieldValue->toSFBool()->get());
    }
    if (strcmp(eventName, "timeLapse") == 0)
    {
        d_timeLapse.set(fieldValue->toSFBool()->get());
        SkyPlugin::plugin->setTimeLapse(fieldValue->toSFBool()->get());
    }
    if (strcmp(eventName, "currentTime") == 0)
    {
        d_currentTime.set(fieldValue->toSFBool()->get());
        SkyPlugin::plugin->setCurrentTime(fieldValue->toSFBool()->get());
    }
    if (strcmp(eventName, "year") == 0)
    {
        d_year.set(fieldValue->toSFInt()->get());
        SkyPlugin::plugin->setYear(fieldValue->toSFInt()->get());
    }
    if (strcmp(eventName, "month") == 0)
    {
        d_month.set(fieldValue->toSFInt()->get());
        SkyPlugin::plugin->setMonth(fieldValue->toSFInt()->get());
    }
    if (strcmp(eventName, "day") == 0)
    {
        d_day.set(fieldValue->toSFInt()->get());
        SkyPlugin::plugin->setDay(fieldValue->toSFInt()->get());
    }
    if (strcmp(eventName, "hour") == 0)
    {
        d_hour.set(fieldValue->toSFInt()->get());
        SkyPlugin::plugin->setHour(fieldValue->toSFInt()->get());
    }
    if (strcmp(eventName, "minute") == 0)
    {
        d_minute.set(fieldValue->toSFInt()->get());
        SkyPlugin::plugin->setMinute(fieldValue->toSFInt()->get());
    }
    if (strcmp(eventName, "radius") == 0)
    {
        d_radius.set(fieldValue->toSFInt()->get());
        SkyPlugin::plugin->setRadius(fieldValue->toSFInt()->get() * 1000);
    }
    if (strcmp(eventName, "latitude") == 0)
    {
        d_latitude.set(fieldValue->toSFInt()->get());
        SkyPlugin::plugin->setLatitude(fieldValue->toSFInt()->get());
    }
    if (strcmp(eventName, "longitude") == 0)
    {
        d_longitude.set(fieldValue->toSFInt()->get());
        SkyPlugin::plugin->setLongitude(fieldValue->toSFInt()->get());
    }
    if (strcmp(eventName, "altitude") == 0)
    {
        d_altitude.set(fieldValue->toSFInt()->get());
        SkyPlugin::plugin->setAltitude(fieldValue->toSFInt()->get());
    }

    // Check exposedFields
    else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }

    setModified();
}

void VrmlNodeSky::render(Viewer *)
{
    if (!d_enabled.get())
        return;
}

osg::BoundingSphere bsphere;
class myCB : public osg::Node::ComputeBoundingSphereCallback
{
public:
    osg::BoundingSphere computeBound(const osg::Node &) const
    {
        return bsphere;
    }
};

SkyPlugin::SkyPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "SkyPlugin::SkyPlugin\n");
    plugin = this;
    ephemerisModel = new osgEphemeris::EphemerisModel;
    ephemerisModel->setSkyDomeRadius(coVRConfig::instance()->farClip() * 0.8 / cover->getScale());

    ephemerisModel->setComputeBoundingSphereCallback(new myCB());

    bsphere.set(osg::Vec3(0, 0, 0), 0);
    ephemerisModel->setInitialBound(bsphere);
    ephemerisModel->setCullingActive(false);

    skyTab = new coTUITab("Sky", coVRTui::instance()->mainFolder->getID());
    skyTab->setPos(0, 0);

    showSky = new coTUIToggleButton("show sky", skyTab->getID());
    showSky->setEventListener(this);
    showSky->setState(false);
    showSky->setPos(0, 0);
    radiusField = new coTUIEditFloatField("radius", skyTab->getID());
    radiusField->setEventListener(this);
    radiusField->setValue(coVRConfig::instance()->farClip() * 0.8);
    radiusField->setPos(1, 0);

    currentTime = new coTUIToggleButton("current time", skyTab->getID());
    currentTime->setEventListener(this);
    currentTime->setState(true);
    currentTime->setPos(0, 1);
    timeLapse = new coTUIToggleButton("time lapse", skyTab->getID());
    timeLapse->setEventListener(this);
    timeLapse->setState(false);
    timeLapse->setPos(1, 1);

    yearField = new coTUIEditIntField("year", skyTab->getID());
    yearField->setEventListener(this);
    yearField->setValue(2006);
    yearField->setPos(0, 2);
	yearLabel = new coTUILabel("Year", skyTab->getID());
    yearLabel->setPos(0, 2);
    yearLabel->setColor(Qt::black);

    monthField = new coTUIEditIntField("month", skyTab->getID());
    monthField->setEventListener(this);
    monthField->setValue(3);
    monthField->setPos(0, 3);
    dayField = new coTUIEditIntField("day", skyTab->getID());
    dayField->setEventListener(this);
    dayField->setValue(5);
    dayField->setPos(0, 4);
    hourField = new coTUIEditIntField("hour", skyTab->getID());
    hourField->setEventListener(this);
    hourField->setValue(12);
    hourField->setPos(0, 5);
    minuteField = new coTUIEditIntField("minute", skyTab->getID());
    minuteField->setEventListener(this);
    minuteField->setValue(12);
    minuteField->setPos(0, 6);
    latitudeField = new coTUIEditFloatField("latitude", skyTab->getID());
    latitudeField->setEventListener(this);
    latitudeField->setValue(48.6);
    latitudeField->setPos(1, 3);
    longitudeField = new coTUIEditFloatField("longitude", skyTab->getID());
    longitudeField->setEventListener(this);
    longitudeField->setValue(9.0008);
    longitudeField->setPos(1, 4);
    altitudeField = new coTUIEditFloatField("altitude", skyTab->getID());
    altitudeField->setEventListener(this);
    altitudeField->setValue(400);
    altitudeField->setPos(1, 5);

    cYear = 0, cMonth = 0, cDay = 0, cHour = 0, cMinute = 0;
    ephemerisModel->setSunLightNum(5);
    ephemerisModel->setLatitudeLongitudeAltitude(48.6, 9.0008, 400.0); // hoehe in m

    // default off cover->getScene()->addChild( ephemerisModel.get() );
}
void SkyPlugin::displaySky(bool visible)
{
    if (visible)
    {
        if (cover->getObjectsRoot()->getChildIndex(ephemerisModel.get()) == cover->getObjectsRoot()->getNumChildren())
        {
            cover->getObjectsRoot()->addChild(ephemerisModel.get());
            showSky->setState(true);
        }
    }
    else
    {
        if (cover->getObjectsRoot()->getChildIndex(ephemerisModel.get()) != cover->getObjectsRoot()->getNumChildren())
        {
            cover->getObjectsRoot()->removeChild(ephemerisModel.get());
            showSky->setState(false);
        }
    }
}

void SkyPlugin::setShowSky(bool v)
{
    showSky->setState(v);
    displaySky(showSky->getState());
}

void SkyPlugin::setTimeLapse(bool v)
{
    timeLapse->setState(v);
}

void SkyPlugin::setCurrentTime(bool v)
{
    currentTime->setState(v);
}
void SkyPlugin::setRadius(float v)
{
    radiusField->setValue(v);
    ephemerisModel->setSkyDomeRadius(radiusField->getValue() / cover->getScale());
}
void SkyPlugin::setYear(int v)
{
    yearField->setValue(v);
    cYear = yearField->getValue();
    ephemerisModel->getEphemerisData()->dateTime.setYear(cYear); // DateTime uses _actual_ year (not since 1900)
}
void SkyPlugin::setMonth(int v)
{
    monthField->setValue(v);
    cMonth = monthField->getValue();
    ephemerisModel->getEphemerisData()->dateTime.setMonth(cMonth); // DateTime uses _actual_ year (not since 1900)
}
void SkyPlugin::setDay(int v)
{
    dayField->setValue(v);
    cDay = dayField->getValue();
    ephemerisModel->getEphemerisData()->dateTime.setDayOfMonth(cDay); // DateTime uses _actual_ year (not since 1900)
}
void SkyPlugin::setHour(int v)
{
    hourField->setValue(v);
    cHour = hourField->getValue();
    ephemerisModel->getEphemerisData()->dateTime.setHour(cHour); // DateTime uses _actual_ year (not since 1900)
}
void SkyPlugin::setMinute(int v)
{
    minuteField->setValue(v);
    cMinute = minuteField->getValue();
    ephemerisModel->getEphemerisData()->dateTime.setMinute(cMinute); // DateTime uses _actual_ year (not since 1900)
}
void SkyPlugin::setLatitude(float v)
{
    latitudeField->setValue(v);
    ephemerisModel->setLatitudeLongitudeAltitude(latitudeField->getValue(), longitudeField->getValue(), altitudeField->getValue()); // hoehe in m
}
void SkyPlugin::setLongitude(float v)
{
    longitudeField->setValue(v);
    ephemerisModel->setLatitudeLongitudeAltitude(latitudeField->getValue(), longitudeField->getValue(), altitudeField->getValue()); // hoehe in m
}
void SkyPlugin::setAltitude(float v)
{
    altitudeField->setValue(v);
    ephemerisModel->setLatitudeLongitudeAltitude(latitudeField->getValue(), longitudeField->getValue(), altitudeField->getValue()); // hoehe in m
}
void SkyPlugin::tabletEvent(coTUIElement *tUIItem)
{
    osgEphemeris::EphemerisData *data = ephemerisModel->getEphemerisData();
    if (tUIItem == showSky)
    {
        displaySky(showSky->getState());
    }
    if (tUIItem == radiusField)
    {
        ephemerisModel->setSkyDomeRadius(radiusField->getValue() / cover->getScale());
    }
    if (tUIItem == yearField || tUIItem == monthField || tUIItem == dayField || tUIItem == hourField || tUIItem == minuteField)
    {
        cYear = yearField->getValue();
        cMonth = monthField->getValue();
        cDay = dayField->getValue();
        cHour = hourField->getValue();
        cMinute = minuteField->getValue();
        data->dateTime.setYear(cYear); // DateTime uses _actual_ year (not since 1900)
        data->dateTime.setMonth(cMonth); // DateTime numbers months from 1 to 12, not 0 to 11
        data->dateTime.setDayOfMonth(cDay); // DateTime numbers days from 1 to 31, not 0 to 30
        data->dateTime.setHour(cHour);
        data->dateTime.setMinute(cMinute);
    }
    if (tUIItem == latitudeField || tUIItem == longitudeField || tUIItem == altitudeField)
    {
        ephemerisModel->setLatitudeLongitudeAltitude(latitudeField->getValue(), longitudeField->getValue(), altitudeField->getValue()); // hoehe in m
    }
}

void SkyPlugin::tabletPressEvent(coTUIElement * /* tUIItem*/)
{
}
// this is called if the plugin is removed at runtime
SkyPlugin::~SkyPlugin()
{
    fprintf(stderr, "SkyPlugin::~SkyPlugin\n");
    if (cover->getObjectsRoot()->getChildIndex(ephemerisModel.get()) != cover->getObjectsRoot()->getNumChildren())
        cover->getObjectsRoot()->removeChild(ephemerisModel.get());

    delete latitudeField;
    delete longitudeField;
    delete altitudeField;
    delete yearField;
    delete monthField;
    delete dayField;
    delete hourField;
    delete minuteField;
    delete radiusField;
    delete currentTime;
    delete showSky;
    delete skyTab;
}

bool SkyPlugin::init()
{
    VrmlNamespace::addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeSky>());
    return true;
}

void
SkyPlugin::preFrame()
{

    int year, month, day, hour, minute;
    osg::Vec3 center;
    center = cover->getViewerMat().getTrans();
    center[2] = VRSceneGraph::instance()->floorHeight();
    center = cover->getInvBaseMat() * center;
    ephemerisModel->setSkyDomeCenter(center);
    ephemerisModel->setSkyDomeRadius(radiusField->getValue() / cover->getScale());
    osgEphemeris::EphemerisData *data = ephemerisModel->getEphemerisData();

    ephemerisModel->setInitialBound(bsphere);
    if (timeLapse->getState())
    {
        struct tm *_tm;
#ifdef _WIN32
        struct __timeb64 currentTime;
#if _MSC_VER < 1400
        _ftime64(&currentTime);
#else
        _ftime64_s(&currentTime);
#endif
        _tm = localtime(&currentTime.time);
        double ts = currentTime.millitm / 16.6;
#else
        timeval currentTime;
        gettimeofday(&currentTime, NULL);
        _tm = localtime(&currentTime.tv_sec);
        double ts = currentTime.tv_usec / 166666.6;
#endif
        year = cYear;
        month = cMonth;
        day = cDay;
        hour = cHour;
        minute = cMinute;
        static int oldSec = 0;
        if (_tm->tm_sec != oldSec)
        {
            oldSec = _tm->tm_sec;
            minute++;
        }
        if (minute > 60)
        {
            minute = 1;
            hour++;
        }
        if (hour > 24)
        {
            hour = 1;
            day++;
        }
        if (day > 30)
        {
            day = 1;
            month++;
        }
        if (month > 12)
        {
            month = 1;
            year++;
        }
        if (minute != cMinute)
        {
            cYear = year;
            cMonth = month;
            cDay = day;
            cHour = hour;
            cMinute = minute;
            yearField->setValue(year);
            monthField->setValue(month);
            dayField->setValue(day);
            hourField->setValue(hour);
            minuteField->setValue(minute);
            data->dateTime.setYear(year); // DateTime uses _actual_ year (not since 1900)
            data->dateTime.setMonth(month); // DateTime numbers months from 1 to 12, not 0 to 11
            data->dateTime.setDayOfMonth(day); // DateTime numbers days from 1 to 31, not 0 to 30
            data->dateTime.setHour(hour);
            data->dateTime.setMinute(minute);
        }
        //fprintf(stderr,"minute %d %lf\n",minute,ts);
        data->dateTime.setSecond((int)ts);
    }
    else if (currentTime->getState())
    {
        struct tm *_tm;
#ifdef _WIN32
        struct __timeb64 currentTime;
#if _MSC_VER < 1400
        _ftime64(&currentTime);
#else
        _ftime64_s(&currentTime);
#endif
        _tm = localtime(&currentTime.time);
#else
        timeval currentTime;
        gettimeofday(&currentTime, NULL);
        _tm = localtime(&currentTime.tv_sec);
#endif
        year = _tm->tm_year + 1900;
        month = _tm->tm_mon + 1;
        day = _tm->tm_mday + 1;
        hour = _tm->tm_hour;
        minute = _tm->tm_min;
        if (minute != cMinute)
        {
            cYear = year;
            cMonth = month;
            cDay = day;
            cHour = hour;
            cMinute = minute;
            yearField->setValue(year);
            monthField->setValue(month);
            dayField->setValue(day);
            hourField->setValue(hour);
            minuteField->setValue(minute);
            data->dateTime.setYear(year); // DateTime uses _actual_ year (not since 1900)
            data->dateTime.setMonth(month); // DateTime numbers months from 1 to 12, not 0 to 11
            data->dateTime.setDayOfMonth(day); // DateTime numbers days from 1 to 31, not 0 to 30
            data->dateTime.setHour(hour);
            data->dateTime.setMinute(minute);
        }
        data->dateTime.setSecond(_tm->tm_sec);
    }
}

// C plugin interface, don't do any coding down here, do it in the C++ Class!

int coVRInit(coVRPlugin *m)
{
    (void)m;
    SkyPlugin::plugin = new SkyPlugin();
    return SkyPlugin::plugin->init();
}

void coVRDelete(coVRPlugin *m)
{
    (void)m;
    delete SkyPlugin::plugin;
}

void coVRPreFrame()
{
    SkyPlugin::plugin->preFrame();
}

COVERPLUGIN(SkyPlugin);
