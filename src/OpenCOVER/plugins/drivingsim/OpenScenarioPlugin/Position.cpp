#include "Position.h"


Position::Position():oscPosition()
{}

Position::~Position()
{}


osg::Vec3 Position::getAbsolutePosition(Entity* currentEntity, RoadSystem* system, std::list<Entity*> entityList)
{
    referencePosition = currentEntity->referencePosition;

    if(Lane.exists())
    {
        // read parameters from vertex
        roadId = Lane->roadId.getValue();
        laneId = Lane->laneId.getValue();
        offset = Lane->offset.getValue();
        s = Lane->s.getValue();

        // access road
        // int roadId_int = atoi(roadId.c_str());
        ::Road *road = system->getRoad(roadId);

        absPosition = getAbsoluteFromRoad(road, s, laneId);

        currentEntity->setRefPos(absPosition);
        return absPosition;

    }
    else if(RelativeLane.exists())
    {
        // get current road of entity
        roadId = currentEntity->roadId.c_str();
        int roadId_int = atoi(currentEntity->roadId.c_str());
        ::Road *road = system->getRoad(roadId_int); // auch Road ID als String

        //convert reference position from absolute world coordinates to absolute lane coordinates
        const Vector3D vec3d = Vector3D(referencePosition[0], referencePosition[1], referencePosition[2]);
        Vector2D vec2D = road->searchPosition(vec3d, 0.0);
        s = vec2D[0];

        laneId =  road->searchLane(s, vec2D[1]);


        // relative lane coordinates
        int dlaneId = RelativeLane->dLane.getValue();
        offset = RelativeLane->offset.getValue();
        double ds = RelativeLane->ds.getValue();

        absPosition = getAbsoluteFromRoad(road, s+ds, laneId+dlaneId);

        currentEntity->setRefPos(absPosition);
        return absPosition;
    }
    else if(RelativeObject.exists())
    {
        dx = RelativeObject->dx.getValue();
        dy = RelativeObject->dy.getValue();
        dz = RelativeObject->dz.getValue();
        relObject = RelativeObject->object.getValue();

        referencePosition = getRelObjectPos(relObject, currentEntity, system, entityList);


        osg::Vec3 relPosition (dx,dy,dz);
        osg::Vec3 absPosition = relPosition+referencePosition;

        return absPosition;

    }
    else if(RelativeRoad.exists())
    {
        // get current road of entity
        int roadId_int = atoi(currentEntity->roadId.c_str());
        ::Road *road = system->getRoad(roadId_int);
        // relative road coordinates
        double ds = RelativeRoad->ds.getValue();
        double dt = RelativeRoad->dt.getValue();

        //convert reference position from absolute world coordinates to absolute lane coordinates
        const Vector3D vec3d = Vector3D(referencePosition[0], referencePosition[1], referencePosition[2]);
        Vector2D vec2D = road->searchPosition(vec3d, 0.0);
        double s = vec2D[0];
        double t = vec2D[1];

        absPosition = getAbsoluteFromRoad(road, s+ds, t+dt);

        currentEntity->setRefPos(absPosition);
        return absPosition;

    }
    else if(RelativeWorld.exists())
    {
        dx = RelativeWorld->dx.getValue();
        dy = RelativeWorld->dy.getValue();
        dz = RelativeWorld->dz.getValue();

        osg::Vec3 relPosition (dx,dy,dz);
        absPosition = relPosition+referencePosition;

        currentEntity->setRefPos(absPosition);
        return absPosition;


    }
    else if(Road.exists())
    {
        // read road coordinates
        roadId = Road->roadId.getValue();
        s = Road->s.getValue();
        double t = Road->t.getValue();

        int roadId_int = atoi(roadId.c_str());
        ::Road *road = system->getRoad(roadId_int);

        // find the lane which belongs to the coordinates in the vertex
        laneId =  road->searchLane(s,t);

        absPosition = getAbsoluteFromRoad(road, s, t);

        currentEntity->setRefPos(absPosition);
        return absPosition;

    }
    else if(Route.exists())
    {
        // Route is not implemented yet
        osg::Vec3 absPosition (0.0,0.0,0.0);
        return absPosition;

    }
    else if(World.exists())
    {
        x = World->x.getValue();
        y = World->y.getValue();
        z = World->z.getValue();

        osg::Vec3 absPosition (x,y,z);

        currentEntity->setRefPos(absPosition);
        return absPosition;
    }

    return osg::Vec3(0.0, 0.0, 0.0);
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

osg::Vec3 Position::hpr2directionVector()
{
    double h = World->h.getValue();
    double p = World->p.getValue();

    osg::Matrix rz;
    rz(0,0) = cos(h); rz(0,1)=sin(h);
    rz(1,0) = -sin(h); rz(1,1)=cos(h);
    osg::Matrix ry;
    ry(0,0) = cos(p); ry(0,2) = -sin(p);
    ry(2,0) = sin(p); ry(2,2) = cos(p);

    osg::Vec3 e0(1,0,0);


    osg::Vec3 dirVec = e0*ry*rz;
    return dirVec;
}

osg::Vec3 Position::getAbsoluteWorld()
{
    x = World->x.getValue();
    y = World->y.getValue();
    z = World->z.getValue();

    return osg::Vec3(x,y,z);
}

