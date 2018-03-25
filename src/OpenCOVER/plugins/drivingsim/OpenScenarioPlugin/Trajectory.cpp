#include "Trajectory.h"

using namespace std;

Trajectory::Trajectory():
oscTrajectory()
{}
Trajectory::~Trajectory(){}

void Trajectory::finishedParsing()
{
}
void Trajectory::initialize(int verticesCounter_temp)
{
    verticesCounter = verticesCounter_temp;

}

osg::Vec3 Trajectory::getAbsolute(Entity* currentEntity)
{
    auto vert = Vertex[currentEntity->visitedVertices];
    //MyPosition* myposition = ((MyPosition*)(*vert->Position));

    if(vert->Position->World.exists())
    {
        osg::Vec3 absCoordinates (vert->Position->World->x.getValue(),vert->Position->World->y.getValue(),vert->Position->World->z.getValue());
        currentEntity->referencePosition = absCoordinates;
        return absCoordinates;
    }
    else if(vert->Position->RelativeWorld.exists())
    {
        osg::Vec3 relCoordinates (vert->Position->RelativeWorld->dx.getValue(),vert->Position->RelativeWorld->dy.getValue(),vert->Position->RelativeWorld->dz.getValue());
        osg::Vec3 absCoordinates = relCoordinates + currentEntity->referencePosition;
        currentEntity->referencePosition = absCoordinates;
        cout << "Entity: " <<  currentEntity->name << ": " << absCoordinates[0] << " "<< absCoordinates[1] << endl;
        return absCoordinates;
    }
    return osg::Vec3(0.0, 0.0, 0.0);

}

float Trajectory::getReference(int visitedVertices){
    /*
    wie soll die Refernce in der Trajecotry definiert werden?
    - für jeden Agenten wieder bei 0 starten?
    - einfach immer nur das dt in die Reference schreiben?
    */


    if(visitedVertices==0)
    {
		if(Vertex.size()>1)
		{
        t1 = Vertex[1]->Shape->reference.getValue();
        t0 = Vertex[0]->Shape->reference.getValue();
		}
        dt = (float) t1-t0;
        return dt;
    }
    else if(visitedVertices==verticesCounter-1)
    {
        t1 = Vertex[visitedVertices]->Shape->reference.getValue();
        t0 = Vertex[visitedVertices-1]->Shape->reference.getValue();

        dt = t1-t0;

        dt = (float) t1-t0;
        return dt;
    }
    else
    {
        t0 = Vertex[visitedVertices]->Shape->reference.getValue();
        t1 = Vertex[visitedVertices+1]->Shape->reference.getValue();

        dt = t1-t0;

        dt = (float) t1-t0;
        return dt;
    }
    return 0.0;
}
