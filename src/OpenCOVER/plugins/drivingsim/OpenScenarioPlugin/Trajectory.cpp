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
        t1 = Vertex[1]->reference.getValue();
        t0 = Vertex[0]->reference.getValue();
		}
        dt = (float) t1-t0;
        return dt;
    }
    else if(visitedVertices==verticesCounter-1)
    {
        t1 = Vertex[visitedVertices]->reference.getValue();
        t0 = Vertex[visitedVertices-1]->reference.getValue();

        dt = t1-t0;

        dt = (float) t1-t0;
        return dt;
    }
    else
    {
        t0 = Vertex[visitedVertices]->reference.getValue();
        t1 = Vertex[visitedVertices+1]->reference.getValue();

        dt = t1-t0;

        dt = (float) t1-t0;
        return dt;
    }
    return 0.0;
}
