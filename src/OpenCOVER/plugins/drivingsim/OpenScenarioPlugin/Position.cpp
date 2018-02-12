#include "Position.h"

Position::Position():oscPosition()
{}

Position::~Position()
{}


osg::Vec3 Position::getAbsolutePosition(osg::Vec3 referencePosition){
    if(Lane.exists())
    {
        // entityGeometry = AgentVehicle(name, entityGeometry->CarGeometry,0,r,inits,laneId,speed,1);
        // Road r; s inits;
        // auto vtrans = entityGeometry->getVehicleTransform();
        // osg::Vec3 pos(vtrans.v().x(), vtrans.v().y(), vtrans.v().z());
        // absPosition = pos;

        // return absPosition;
        osg::Vec3 absPosition (0.0,0.0,0.0);
        return absPosition;

    }
    else if(RelativeLane.exists())
    {
        osg::Vec3 absPosition (0.0,0.0,0.0);
        return absPosition;

    }
    else if(RelativeObject.exists())
    {
        dx = RelativeObject->dx.getValue();
        dy = RelativeObject->dy.getValue();
        dz = RelativeObject->dz.getValue();

        osg::Vec3 relPosition (dx,dy,dz);
        osg::Vec3 absPosition = relPosition+referencePosition;
        return absPosition;

    }
    else if(RelativeRoad.exists())
    {
        osg::Vec3 absPosition (0.0,0.0,0.0);
        return absPosition;

    }
    else if(RelativeWorld.exists())
    {
        dx = RelativeWorld->dx.getValue();
        dy = RelativeWorld->dy.getValue();
        dz = RelativeWorld->dz.getValue();

        osg::Vec3 relPosition (dx,dy,dz);
        osg::Vec3 absPosition = relPosition+referencePosition;
        return absPosition;


    }
    else if(Road.exists())
    {
        osg::Vec3 absPosition (0.0,0.0,0.0);
        return absPosition;

    }
    else if(Route.exists())
    {
        osg::Vec3 absPosition (0.0,0.0,0.0);
        return absPosition;

    }
    else if(World.exists())
    {
        x = World->x.getValue();
        y = World->y.getValue();
        z = World->z.getValue();
        osg::Vec3 absPosition (x,y,z);
        return absPosition;
    }

    return osg::Vec3(0.0, 0.0, 0.0);
}

double Position::getDx(){
    return dx;
}
