/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _DNA_BASE_UNIT_CONNECTION_POINT_H
#define _DNA_BASE_UNIT_CONNECTION_POINT_H

class DNABaseUnit;
#include <osg/Vec3>
#include <string>

class DNABaseUnitConnectionPoint
{
private:
    //      /
    //     o -   myBaseUnit
    //     |this
    //      -o   connectedBaseUnit
    //      /
    std::string myBaseUnitName_; // name of my connectionPoint
    std::string connectableBaseUnitName_; // name of connectable point
    osg::Vec3 point_; // vector to connection point
    osg::Vec3 normal_; // normal of baseunit
    bool isConnected_; // flag if point has a connection
    bool isEnabled_; //flag if point is enabled
    bool rotation_;
    DNABaseUnit *myBaseUnit_;
    DNABaseUnitConnectionPoint *connectedPoint_;

public:
    // filename: name of geomtry file, radius for connections and intersection, list of connection names and vectors
    DNABaseUnitConnectionPoint(DNABaseUnit *myBase, std::string myName, osg::Vec3 point, osg::Vec3 normal, std::string myConnectionName);

    virtual ~DNABaseUnitConnectionPoint();

    std::string getMyBaseUnitName()
    {
        return myBaseUnitName_;
    };
    std::string getConnectableBaseUnitName()
    {
        return connectableBaseUnitName_;
    };
    osg::Vec3 getPoint()
    {
        return point_;
    };
    osg::Vec3 getNormal()
    {
        return normal_;
    };
    bool getRotation()
    {
        return rotation_;
    };
    void setRotation(bool b)
    {
        rotation_ = b;
    };
    bool isEnabled()
    {
        return isEnabled_;
    };
    void setEnabled(bool b);

    bool isConnected()
    {
        return isConnected_;
    };
    bool connectTo(DNABaseUnitConnectionPoint *connPoint);
    void disconnectBase();
    DNABaseUnit *getMyBaseUnit()
    {
        return myBaseUnit_;
    };
    DNABaseUnitConnectionPoint *getConnectedPoint()
    {
        return connectedPoint_;
    };
};
#endif
