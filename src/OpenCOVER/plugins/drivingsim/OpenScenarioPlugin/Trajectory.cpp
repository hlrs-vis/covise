#include "Trajectory.h"

using namespace std;

Trajectory::Trajectory():
oscTrajectory()
{}
Trajectory::~Trajectory(){}

void Trajectory::finishedParsing()
{
}

float Trajectory::getReference(int vert){
    /*
    wie soll die Refernce in der Trajecotry definiert werden?
    - für jeden Agenten wieder bei 0 starten?
    - einfach immer nur das dt in die Reference schreiben?
    */


    if(vert ==0)
    {
		if(Vertex.size()>1)
		{
        t1 = Vertex[1]->reference.getValue();
        t0 = Vertex[0]->reference.getValue();
		}
        dt = (float) t1-t0;
        return dt;
    }
    else if(vert ==Vertex.size()-1)
    {
        t1 = Vertex[vert]->reference.getValue();
        t0 = Vertex[vert -1]->reference.getValue();

        dt = t1-t0;

        dt = (float) t1-t0;
        return dt;
    }
    else
    {
        t0 = Vertex[vert]->reference.getValue();
        t1 = Vertex[vert +1]->reference.getValue();

        dt = t1-t0;

        dt = (float) t1-t0;
        return dt;
    }
    return 0.0;
}
