/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __VehicleDynamics_H
#define __VehicleDynamics_H

#include "Vehicle.h"

#include <osg/LineSegment>
#include <osg/MatrixTransform>
#include <osgUtil/IntersectVisitor>

class PLUGINEXPORT VehicleDynamics
{
public:
    VehicleDynamics();
    virtual ~VehicleDynamics()
    {
    }

    virtual void update()
    {
    }

    virtual void move(VrmlNodeVehicle *vehicle) = 0;

    virtual void resetState() = 0;

    virtual double getVelocity()
    {
        return 0.0;
    }
    virtual double getAcceleration()
    {
        return 0.0;
    }
    virtual double getEngineSpeed()
    {
        return 0.0;
    }
    virtual double getEngineTorque()
    {
        return 0.0;
    }
    virtual double getTyreSlipFL()
    {
        return 0.0;
    }
    virtual double getTyreSlipFR()
    {
        return 0.0;
    }
    virtual double getTyreSlipRL()
    {
        return 0.0;
    }
    virtual double getTyreSlipRR()
    {
        return 0.0;
    }

    virtual double getSteeringWheelTorque()
    {
        return 0.0;
    }
    virtual void setVehicleTransformation(const osg::Matrix &) = 0;

    virtual const osg::Matrix &getVehicleTransformation() = 0;

    virtual int getRoadType();
    virtual void setRoadType(int);

protected:
    double testLengthUp, testLengthDown;
    int roadType;

    template <typename ElevType>
    void getWheelElevation(const osg::Vec2 &, const osg::Vec2 &, const osg::Vec2 &, const osg::Vec2 &, ElevType &, ElevType &, ElevType &, ElevType &);
};

inline int VehicleDynamics::getRoadType()
{
    return roadType;
}

inline void VehicleDynamics::setRoadType(int setRoadType)
{
    roadType = setRoadType;
}

template <typename ElevType>
inline void VehicleDynamics::getWheelElevation(const osg::Vec2 &wheelPosOne, const osg::Vec2 &wheelPosTwo, const osg::Vec2 &wheelPosThree, const osg::Vec2 &wheelPosFour, ElevType &wheelElevOne, ElevType &wheelElevTwo, ElevType &wheelElevThree, ElevType &wheelElevFour)
{

    osg::Matrix baseMat;
    baseMat = cover->getObjectsScale()->getMatrix();

    osg::Matrix transformMatrix = cover->getObjectsXform()->getMatrix();
    baseMat.postMult(transformMatrix);

    osg::Matrix baseMatInv = osg::Matrix::inverse(baseMat);

    osg::LineSegment *normalOne = new osg::LineSegment(baseMat.preMult(osg::Vec3d(wheelPosOne, wheelElevOne - testLengthDown)), baseMat.preMult(osg::Vec3d(wheelPosOne, wheelElevOne + testLengthUp)));
    osg::LineSegment *normalTwo = new osg::LineSegment(baseMat.preMult(osg::Vec3d(wheelPosTwo, wheelElevTwo - testLengthDown)), baseMat.preMult(osg::Vec3d(wheelPosTwo, wheelElevTwo + testLengthUp)));
    osg::LineSegment *normalThree = new osg::LineSegment(baseMat.preMult(osg::Vec3d(wheelPosThree, wheelElevThree - testLengthDown)), baseMat.preMult(osg::Vec3d(wheelPosThree, wheelElevThree + testLengthUp)));
    osg::LineSegment *normalFour = new osg::LineSegment(baseMat.preMult(osg::Vec3d(wheelPosFour, wheelElevFour - testLengthDown)), baseMat.preMult(osg::Vec3d(wheelPosFour, wheelElevFour + testLengthUp)));

    osgUtil::IntersectVisitor visitor;
    visitor.setTraversalMask(Isect::Collision);
    visitor.addLineSegment(normalOne);
    visitor.addLineSegment(normalTwo);
    visitor.addLineSegment(normalThree);
    visitor.addLineSegment(normalFour);
    cover->getObjectsXform()->accept(visitor);

    int num = visitor.getNumHits(normalOne);
    //std::cerr << "Hits wheel one: " << num << ", normalOne: down (" << normalOne->start()[0] << ", " << normalOne->start()[1] <<  ", " << normalOne->start()[2] << "), up (" << normalOne->end()[0] << ", " << normalOne->end()[1] <<  ", " << normalOne->end()[2] << ")" <<  std::endl;
    if (num)
    {
        osg::Vec3d intersectOne = baseMatInv.preMult(visitor.getHitList(normalOne).back().getWorldIntersectPoint());
        wheelElevOne = intersectOne.z();
    }
    num = visitor.getNumHits(normalTwo);
    if (num)
    {
        osg::Vec3d intersectTwo = baseMatInv.preMult(visitor.getHitList(normalTwo).back().getWorldIntersectPoint());
        wheelElevTwo = intersectTwo.z();
    }
    num = visitor.getNumHits(normalThree);
    if (num)
    {
        osg::Vec3d intersectThree = baseMatInv.preMult(visitor.getHitList(normalThree).back().getWorldIntersectPoint());
        wheelElevThree = intersectThree.z();
    }
    num = visitor.getNumHits(normalFour);
    if (num)
    {
        osg::Vec3d intersectFour = baseMatInv.preMult(visitor.getHitList(normalFour).back().getWorldIntersectPoint());
        wheelElevFour = intersectFour.z();
    }
}

#endif
