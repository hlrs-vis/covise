#include "Position.h"
#include "ReferencePosition.h"


Position::Position():oscPosition()
{}

Position::~Position()
{}


ReferencePosition* Position::getAbsolutePosition(Entity* currentEntity, const std::list<Entity*> &entityList)
{

    if(Lane.exists())
    {
        // read parameters from vertex
        roadId = Lane->roadId.getValue();
        laneId = Lane->laneId.getValue();
        offset = Lane->offset.getValue();
        s = Lane->s.getValue();

        currentEntity->newRefPos->update(roadId, s, laneId);
        return currentEntity->newRefPos;

    }
    else if(RelativeLane.exists())
    {
        // relative lane coordinates
        int dlaneId = RelativeLane->dLane.getValue();
        double ds = RelativeLane->ds.getValue();
        std::string refObjectName = RelativeLane->object.getValue();

        ReferencePosition* refObject = getRelObjectPos(refObjectName, currentEntity, entityList);

        refObject->update();
        currentEntity->newRefPos->update(refObject->roadId, refObject->s+ds, refObject->laneId+dlaneId);

        return currentEntity->newRefPos;
    }
    else if(RelativeRoad.exists())
    {
        // relative road coordinates
        double ds = RelativeRoad->ds.getValue();
        double dt = RelativeRoad->dt.getValue();
        std::string relObject = RelativeRoad->object.getValue();

        ReferencePosition* refObject = getRelObjectPos(relObject, currentEntity, entityList);

        refObject->update();
        currentEntity->newRefPos->update(refObject->roadId,refObject->s+ds,refObject->t+dt);

        return currentEntity->newRefPos;

    }

    else if(Road.exists())
    {
        // read road coordinates
        roadId = Road->roadId.getValue();
        s = Road->s.getValue();
        t = Road->t.getValue();

        currentEntity->newRefPos->update(roadId,s,t);
        return currentEntity->newRefPos;

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
        hdg = World->h.getValue();

        currentEntity->newRefPos->update(x,y,z,hdg);
        return currentEntity->newRefPos;
    }

    else if(RelativeWorld.exists())
    {
        dx = RelativeWorld->dx.getValue();
        dy = RelativeWorld->dy.getValue();
        dz = RelativeWorld->dz.getValue();
        std::string relObject = RelativeWorld->object.getValue();

        ReferencePosition* refObject = getRelObjectPos(relObject, currentEntity, entityList);

        currentEntity->newRefPos->update(refObject->xyz[0]+dx,refObject->xyz[1]+dy,refObject->xyz[2]+dz,0.0);
        return currentEntity->newRefPos;


    }
    else if(RelativeObject.exists())
    {
        // is not implemented yet
        return NULL;

    }

    return NULL;
}

void Position::getAbsolutePosition(ReferencePosition *relativePos, ReferencePosition *position)
{
    if (Lane.exists())
    {
        // read parameters from vertex
        roadId = Lane->roadId.getValue();
        laneId = Lane->laneId.getValue();
        offset = Lane->offset.getValue();
        s = Lane->s.getValue();

        position->update(roadId, s, laneId);
    }
    else if (RelativeLane.exists())
    {
        // relative lane coordinates
        int dlaneId = RelativeLane->dLane.getValue();
        double ds = RelativeLane->ds.getValue();
        position->update(relativePos->roadId, relativePos->s + ds, relativePos->laneId + dlaneId);
    }
    else if (RelativeRoad.exists())
    {
        // relative road coordinates
        double ds = RelativeRoad->ds.getValue();
        double dt = RelativeRoad->dt.getValue();
        
        position->update(relativePos->roadId, relativePos->s + ds, relativePos->t + dt);
    }

    else if (Road.exists())
    {
        // read road coordinates
        roadId = Road->roadId.getValue();
        s = Road->s.getValue();
        t = Road->t.getValue();

        position->update(roadId, s, t);

    }
    else if (Route.exists())
    {
        // Route is not implemented yet

    }
    else if (World.exists())
    {
        x = World->x.getValue();
        y = World->y.getValue();
        z = World->z.getValue();
        hdg = World->h.getValue();

        position->update(x, y, z, hdg);
    }

    else if (RelativeWorld.exists())
    {
        dx = RelativeWorld->dx.getValue();
        dy = RelativeWorld->dy.getValue();
        dz = RelativeWorld->dz.getValue();

        position->update(relativePos->xyz[0] + dx, relativePos->xyz[1] + dy, relativePos->xyz[2] + dz, 0.0);
    }
    else if (RelativeObject.exists())
    {
        // is not implemented yet
        

    }
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

ReferencePosition* Position::getRelObjectPos(std::string refObjectName, Entity* currentEntity, const std::list<Entity*> &entityList)
{
    if(refObjectName == "")
    {
        currentEntity->refObject = currentEntity;
        return currentEntity->refPos;
    }
    if(currentEntity->refObject != NULL)
    {
        if(refObjectName == currentEntity->refObject->name)
        {
            return currentEntity->refObject->refPos;
        }
    }
    for(auto refEntity = entityList.begin(); refEntity != entityList.end(); refEntity++)
    {
        if(refObjectName == (*refEntity)->name)
        {
            currentEntity->refObject = (*refEntity);
            return (*refEntity)->refPos;
        }

    }
    return currentEntity->refPos;
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

