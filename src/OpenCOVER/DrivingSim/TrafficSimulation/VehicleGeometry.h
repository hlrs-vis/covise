/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VehicleGeometry_h
#define VehicleGeometry_h

#include <VehicleUtil/RoadSystem/Types.h>

class TRAFFICSIMULATIONEXPORT VehicleGeometry
{
public:
    virtual ~VehicleGeometry()
    {
    }

    virtual void setTransform(Transform &, double) = 0;

    virtual double getBoundingCircleRadius() = 0;

    virtual const osg::Matrix &getVehicleTransformMatrix() = 0;

protected:
};

#endif
