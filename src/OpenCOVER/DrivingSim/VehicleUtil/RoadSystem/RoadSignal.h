/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef RoadSignal_h
#define RoadSignal_h

#include "Element.h"
#include "Types.h"
#include <osgSim/MultiSwitch>
#include <osg/PositionAttitudeTransform>
#include <osg/Material>
#if __cplusplus >= 201103L || defined WIN32
#include <unordered_map>
using std::unordered_map;
#else
#include <tr1/unordered_map>
using std::tr1::unordered_map;
#endif

class SignalTurnCallback
{
public:
    virtual void on() = 0;
    virtual void off() = 0;
    virtual ~SignalTurnCallback(){};
};
class SignalPrototype
{
public:
    SignalPrototype(std::string name, std::string country, int type, int subtype, std::string subclass, bool isFlat);
    ~SignalPrototype();

    void createGeometry();
    osg::ref_ptr<osg::Node> signalNode;
    osg::ref_ptr<osg::Node> signalPost;
    std::string name;
    int type;
    int subtype;
    std::string subclass;
    std::string country;
    bool flat;
    osg::ref_ptr<osg::Material> streetmarkMaterial;
};

class VEHICLEUTILEXPORT TrafficLightSignalTurnCallback : public SignalTurnCallback
{
public:
    virtual ~TrafficLightSignalTurnCallback(){};
    TrafficLightSignalTurnCallback(osgSim::MultiSwitch *);

    void on();
    void off();

protected:
    osgSim::MultiSwitch *multiSwitch;
};

class VEHICLEUTILEXPORT RoadSignal : public Element
{
public:
    enum OrientationType
    {
        POSITIVE_TRACK_DIRECTION = 1,
        NEGATIVE_TRACK_DIRECTION = -1,
        BOTH_DIRECTIONS = 0
    };

    RoadSignal(const std::string &, const std::string &, const double &, const double &, const bool &, const OrientationType &,
               const double &, const std::string &, const int &, const int &, const std::string &, const double &, const double &);

    const std::string &getName()
    {
        return name;
    }
    const double &getS()
    {
        return s;
    }
    const double &getT()
    {
        return t;
    }
    const bool &isDynamic()
    {
        return dynamic;
    }
    const OrientationType &getOrientation()
    {
        return orientation;
    }
    const double &getZOffset()
    {
        return zOffset;
    }
    const std::string &getCountry()
    {
        return country;
    }
    const int &getType()
    {
        return type;
    }
    const int &getSubtype()
    {
        return subtype;
    }
    const double &getValue()
    {
        return value;
    }

    void setValue(const double &setVal)
    {
        value = setVal;
    }

    void setTransform(const Transform &_transform);
    const Transform &getTransform() const
    {
        return signalTransform;
    }
    //virtual void update(const double&) { }

    //virtual void signalGo() { }
    //virtual void signalStop() { }
    virtual osg::PositionAttitudeTransform *getRoadSignalNode();
    virtual osg::PositionAttitudeTransform *getRoadSignalPost();

protected:
    std::string name;
    double s;
    double t;
    double size;
    bool dynamic;
    OrientationType orientation;
    double zOffset;
    std::string country;
    int type;
    int subtype;
    std::string subclass;
    double value;

    Transform signalTransform;

    osg::PositionAttitudeTransform *roadSignalNode;
    osg::PositionAttitudeTransform *roadSignalPost;

    static unordered_map<std::string, SignalPrototype *> signalsMap;
};

class TrafficLightPrototype
{
public:
    TrafficLightPrototype(std::string name, std::string country, int type, int subtype, std::string subclass);
    ~TrafficLightPrototype();

    osg::ref_ptr<osg::Node> getTrafficLightNode()
    {
        return trafficLightNode;
    }

private:
    osg::Node * createGeometry();
    osg::ref_ptr<osg::Node> trafficLightNode;
    std::string name;
    int type;
    int subtype;
    std::string subclass;
    std::string country;
};

class VEHICLEUTILEXPORT TrafficLightSignal : public RoadSignal
{
public:
    enum SignalSwitchType
    {
        ON = 1,
        OFF = -1
    };
    enum SignalType
    {
        GREEN,
        YELLOW,
        RED
    };

    TrafficLightSignal(const std::string &, const std::string &, const double &, const double &, const bool &, const OrientationType &,
                       const double &, const std::string &, const int &, const int &, const std::string &setSubclass, const double &setSize, const double &);

    void setSignalGreenCallback(SignalTurnCallback *);
    void setSignalYellowCallback(SignalTurnCallback *);
    void setSignalRedCallback(SignalTurnCallback *);

    void switchGreenSignal(SignalSwitchType);
    void switchYellowSignal(SignalSwitchType);
    void switchRedSignal(SignalSwitchType);

    osg::PositionAttitudeTransform *getRoadSignalNode();

    //void signalGo();
    //void signalStop();

    //virtual void update(const double&);

protected:
    SignalTurnCallback *signalGreenCallback;
    SignalTurnCallback *signalYellowCallback;
    SignalTurnCallback *signalRedCallback;

    osg::PositionAttitudeTransform *trafficSignalNode;
    static osg::Node *trafficSignalNodeTemplate;
    //SignalSwitchType signalTurn;
    //bool signalTurnFinished;
    //double yellowPhaseTime;
    //double timer;

    static unordered_map<std::string, TrafficLightPrototype *> trafficLightsMap;
};

#endif
