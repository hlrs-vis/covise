#include "Position.h"
#include "ReferencePosition.h"


Position::Position():oscPosition()
{}

Position::~Position()
{}


ReferencePosition* Position::getAbsolutePosition(Entity* currentEntity, RoadSystem* system, std::list<Entity*> entityList)
{
    referencePosition = currentEntity->refPos;
    newReferencePosition = new ReferencePosition(currentEntity->refPos);
    currentEntity->newRefPos = newReferencePosition;

    if(Lane.exists())
    {
        // read parameters from vertex
        roadId = Lane->roadId.getValue();
        laneId = Lane->laneId.getValue();
        offset = Lane->offset.getValue();
        s = Lane->s.getValue();

        // access road
        ::Road *road = system->getRoad(roadId);

        absPosition = getAbsoluteFromRoad(road, s, laneId);

        referencePosition->update(roadId, s, laneId);
        return referencePosition;

    }
    else if(RelativeLane.exists())
    {
        // relative lane coordinates
        int dlaneId = RelativeLane->dLane.getValue();
        double ds = RelativeLane->ds.getValue();

        referencePosition->update(dlaneId, ds);
        return referencePosition;
    }
    else if(RelativeObject.exists())
    {
        dx = RelativeObject->dx.getValue();
        dy = RelativeObject->dy.getValue();
        dz = RelativeObject->dz.getValue();
        relObject = RelativeObject->object.getValue();

        osg::Vec3 referenceObjectPosition = getRelObjectPos(relObject, currentEntity, system, entityList);

        return NULL;

    }
    else if(RelativeRoad.exists())
    {
        // relative road coordinates
        double ds = RelativeRoad->ds.getValue();
        double dt = RelativeRoad->dt.getValue();

        referencePosition->update(ds,dt);

        return referencePosition;

    }

    else if(Road.exists())
    {
        // read road coordinates
        roadId = Road->roadId.getValue();
        s = Road->s.getValue();
        t = Road->t.getValue();

        referencePosition->update(roadId,s,t);
        return referencePosition;

    }
    else if(Route.exists())
    {
        // Route is not implemented yet
        return NULL;

    }
    else if(World.exists())
    {
        x = World->x.getValue();
        y = World->y.getValue();
        z = World->z.getValue();

        referencePosition->update(x,y,z);
        return referencePosition;
    }

    else if(RelativeWorld.exists())
    {
        dx = RelativeWorld->dx.getValue();
        dy = RelativeWorld->dy.getValue();
        dz = RelativeWorld->dz.getValue();

        referencePosition->update(dx,dy,dz,true);
        return referencePosition;


    }

    return NULL;
}

osg::Vec3 Position::getAbsoluteFromRoad(::Road* road, double s, int laneId)
{
    // convert absolute lane coorinates to absolute world coorinates
    LaneSection* LS = road->getLaneSection(s);
    Vector2D laneCenter = LS->getLaneCenter(laneId, s);

    Transform vtrans = road->getRoadTransform(s, laneCenter[0]);
    osg::Vec3 pos(vtrans.v().x(), vtrans.v().y(), vtrans.v().z());

    return pos;
}

osg::Vec3 Position::getAbsoluteFromRoad(::Road* road, double s, double t)
{
    // convert absolute road coorinates to absolute world coorinates
    Transform vtrans = road->getRoadTransform(s, t);
    osg::Vec3 pos(vtrans.v().x(), vtrans.v().y(), vtrans.v().z());

    return pos;
}

osg::Vec3 Position::getRelObjectPos(std::string relObject, Entity* currentEntity, RoadSystem* system, std::list<Entity*> entityList)
{
    // check if relative Object is the Entity itself
    if(relObject == "self")
    {
        return currentEntity->referencePosition;
    }

    // check if relative Object is another Entity
    for(std::list<Entity*>::iterator refEntity = entityList.begin(); refEntity != entityList.end(); refEntity++)
    {
        if(relObject == (*refEntity)->name){
            return (*refEntity)->entityPosition;
        }
    }

    // check if road system object is relObject
    int numRoads = system->getNumRoads();
    for (int i = 0; i < numRoads; ++i)
    {
        ::Road *road = system->getRoad(i);
        int numSignals = road->getNumRoadSignals();
        for (int ii = 0; ii < numSignals; ++ii)
        {
            RoadSignal* refSignal = road->getRoadSignal(ii);
            if(relObject == refSignal->getName())
            {
                const Transform signalTransf =  refSignal->getTransform();
                osg::Vec3 refPos(signalTransf.v().x(), signalTransf.v().y(), signalTransf.v().z());
                return refPos;
            }
        }
    }
    return osg::Vec3(0.0, 0.0, 0.0);

}

double Position::getHdg()
{
    return World->h.getValue();
}

osg::Vec3 Position::getAbsoluteWorld()
{
    x = World->x.getValue();
    y = World->y.getValue();
    z = World->z.getValue();

    return osg::Vec3(x,y,z);
}

