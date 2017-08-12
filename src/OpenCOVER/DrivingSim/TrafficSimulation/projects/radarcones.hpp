/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** TrafficSimulation
**   Frank Naegele 2010
**   <mail@f-naegele.de> <frank.naegele@porsche.de>
**   2/11/2010
**
** This class can be used to visualize Porsche radars with cones.
**
**************************************************************************/

#ifndef RADARCONES_HPP
#define RADARCONES_HPP

#include "osg/Group"
#include "osg/StateSet"
#include "osg/Transform"

#include "../HumanVehicle.h"
#include "../PorscheFFZ.h"

class RadarCones
{
public:
    RadarCones(HumanVehicle *humanVehicle);

    //	void setTransform(osg::Transform&, double);
    //	const osg::Transform& getVehicleTransformation();

    void update(RadarConesData *data);

private:
    void init();
    void updateCone(int i);

private:
    HumanVehicle *humanVehicle_;

    RadarConesData *conesData_;

    // osg //
    //
    osg::Transform *humanVehicleTransform_;
    osg::Group *cones_;
    osg::StateSet *coneStateSet_;

    std::vector<osg::Geometry *> coneGeometries_;
};

#endif // RADARCONES_HPP
