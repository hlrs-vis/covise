/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __VehicleDynamics_H
#define __VehicleDynamics_H

#include "Vehicle.h"

#include <osg/LineSegment>
#include <osg/MatrixTransform>
#include <cover/coIntersection.h>

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

    osg::ref_ptr<osgUtil::IntersectorGroup> igroup = new osgUtil::IntersectorGroup;
    osg::ref_ptr<osgUtil::LineSegmentIntersector> intersector[4];

    intersector[0] = coIntersection::instance()->newIntersector(baseMat.preMult(osg::Vec3d(wheelPosOne, wheelElevOne - testLengthDown)), baseMat.preMult(osg::Vec3d(wheelPosOne, wheelElevOne + testLengthUp)));
    igroup->addIntersector(intersector[0]);
    intersector[1] = coIntersection::instance()->newIntersector(baseMat.preMult(osg::Vec3d(wheelPosTwo, wheelElevTwo - testLengthDown)), baseMat.preMult(osg::Vec3d(wheelPosTwo, wheelElevTwo + testLengthUp)));
    igroup->addIntersector(intersector[1]);
    intersector[2] = coIntersection::instance()->newIntersector(baseMat.preMult(osg::Vec3d(wheelPosThree, wheelElevThree - testLengthDown)), baseMat.preMult(osg::Vec3d(wheelPosThree, wheelElevThree + testLengthUp)));
    igroup->addIntersector(intersector[2]);
    intersector[3] = coIntersection::instance()->newIntersector(baseMat.preMult(osg::Vec3d(wheelPosFour, wheelElevFour - testLengthDown)), baseMat.preMult(osg::Vec3d(wheelPosFour, wheelElevFour + testLengthUp)));
    igroup->addIntersector(intersector[3]);

    osgUtil::IntersectionVisitor visitor(igroup);
    visitor.setTraversalMask(Isect::Collision);
    cover->getObjectsXform()->accept(visitor);

    //std::cerr << "Hits wheel one: " << num << ", normalOne: down (" << normalOne->start()[0] << ", " << normalOne->start()[1] <<  ", " << normalOne->start()[2] << "), up (" << normalOne->end()[0] << ", " << normalOne->end()[1] <<  ", " << normalOne->end()[2] << ")" <<  std::endl;
    if (intersector[0]->containsIntersections())
    {
        osg::Vec3d intersectOne = baseMatInv.preMult(intersector[0]->getFirstIntersection().getWorldIntersectPoint());
        wheelElevOne = intersectOne.z();
    }
    if (intersector[1]->containsIntersections())
    {
        osg::Vec3d intersectTwo = baseMatInv.preMult(intersector[1]->getFirstIntersection().getWorldIntersectPoint());
        wheelElevTwo = intersectTwo.z();
    }
    if (intersector[2]->containsIntersections())
    {
        osg::Vec3d intersectThree = baseMatInv.preMult(intersector[2]->getFirstIntersection().getWorldIntersectPoint());
        wheelElevThree = intersectThree.z();
    }
    if (intersector[3]->containsIntersections())
    {
        osg::Vec3d intersectFour = baseMatInv.preMult(intersector[3]->getFirstIntersection().getWorldIntersectPoint());
        wheelElevFour = intersectFour.z();
    }
}

#endif
