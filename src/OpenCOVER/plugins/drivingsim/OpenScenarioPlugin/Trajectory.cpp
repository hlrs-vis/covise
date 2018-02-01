#include "Trajectory.h"

using namespace std;

Trajectory::Trajectory():
oscTrajectory()
{}
Trajectory::~Trajectory(){}

void Trajectory::finishedParsing()
{
}
void Trajectory::initialize(vector<osg::Vec3> vec_temp, vector<bool> isRelVertice_temp)
{
	polylineVertices = vec_temp;
    isRelVertice = isRelVertice_temp;

}

osg::Vec3 Trajectory::getAbsolute(int visitedVertices, Entity *currentEntity){
    auto vert = Vertex[visitedVertices];

    if(vert->Position->World.exists()){
        osg::Vec3 absCoordinates (vert->Position->World->x.getValue(),vert->Position->World->y.getValue(),vert->Position->World->z.getValue());
        return absCoordinates;
    }
    else if(vert->Position->RelativeWorld.exists()){
        osg::Vec3 relCoordinates (vert->Position->RelativeWorld->dx.getValue(),vert->Position->RelativeWorld->dy.getValue(),vert->Position->RelativeWorld->dz.getValue());
        osg::Vec3 absCoordinates = relCoordinates + currentEntity->entityPosition;

        return absCoordinates;
    }
	return osg::Vec3(0.0, 0.0, 0.0);
//    auto coords = vert->Position->World;
//    int a= 1;


}

double Trajectory::getRefernce(){
    // to be implemented
	return 0.0;
}
