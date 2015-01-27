/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef RoadObject_h
#define RoadObject_h

#include "Element.h"
class Road;
class LaneSection;

#include <string>
#include <osg/Node>
#include <osg/Geometry>
#include "Types.h"
class RoadCornerLocal
{
public:
    ~RoadCornerLocal(){};
    RoadCornerLocal(float h, float pu, float pv, float pz)
    {
        height = h;
        u = pu;
        v = pv;
        z = pz;
    };
    float height, u, v, z;
};

class RoadOutline : public std::vector<RoadCornerLocal>
{
};

class RoadObject : public Element
{
public:
    enum OrientationType
    {
        POSITIVE_TRACK_DIRECTION = 1,
        NEGATIVE_TRACK_DIRECTION = -1,
        BOTH_DIRECTIONS = 0
    };

    RoadObject(const std::string &, const std::string &, const std::string & = "noname", const std::string & = "notype", const double & = 0.0, const double & = 0.0,
               const double & = 0.0, const double & = 0.0, OrientationType = BOTH_DIRECTIONS, const double & = 0.0, const double & = 0.0, const double & = 0.0,
               const double & = 0.0, const double & = 0.0, const double & = 0.0, const double & = 0.0, Road *roadP = NULL);

    void setObjectRepeat(const double &, const double &, const double &);

    osg::Node *getObjectNode();

    void setOutline(RoadOutline *ro)
    {
        outline = ro;
    };

    bool isAbsolute()
    {
        return absolute;
    }

    double getS()
    {
        return s;
    }
    double getT()
    {
        return t;
    }
    double getZOffset()
    {
        return zOffset;
    }
    double getHdg()
    {
        return hdg;
    }
    double getPitch()
    {
        return pitch;
    }
    double getRoll()
    {
        return roll;
    }

    double getOrientation()
    {
        return orientation;
    }

    bool isObjectRepeating()
    {
        return objectRepeat;
    }
    double getRepeatLength()
    {
        return repeatLength;
    }
    double getRepeatDistance()
    {
        return repeatDistance;
    }

protected:
    bool fileExist(const char *);
    osg::Node *loadObjectGeometry(std::string);

    //std::string id;
    std::string name;
    std::string type;
    std::string fileName;

    bool absolute; // is object node in absolute coordinates, such as guardRails

    double s;
    double t;
    double zOffset;
    double validLength;

    OrientationType orientation;

    double length;
    double width;
    double radius;
    double height;
    double hdg;
    double pitch;
    double roll;

    osg::Node *objectNode;

    bool objectRepeat;
    double repeatLength;
    double repeatDistance;

    Road *road;

    osg::Geode *createGuardRailGeode();
    osg::Geode *createOutlineGeode();
    osg::Geode *createSTEPBarrierGeode();
    osg::Geode *createJerseyBarrierGeode();
    osg::Geode *createReflectorPostGeode(std::string &textureName);
    static osg::Geode *reflectorPostGeode;
    static osg::Geode *reflectorPostYellowGeode;

    RoadOutline *outline;
};

#endif
