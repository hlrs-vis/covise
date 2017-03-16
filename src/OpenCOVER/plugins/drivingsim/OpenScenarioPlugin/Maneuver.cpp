#include "Maneuver.h"
#include <cover/coVRPluginSupport.h>
#include <iterator>
#include <math.h>

Maneuver::Maneuver(string name):name(name),
	currentExecution(0)
{
	maneuverCondition=true,totalDistance=0;visitedVertices=0;verticesCounter=0;
}
Maneuver::~Maneuver()
{
}

osg::Vec3 &Maneuver::followTrajectory(osg::Vec3 currentPos, vector<float> targetPosition, float speed)
{
	//substract vectors
	osg::Vec3 targetPos(targetPosition[0],targetPosition[1],targetPosition[2]);
	norm_direction_vec = currentPos - targetPos;
	float distance = norm_direction_vec.length();
	norm_direction_vec.normalize();
	//calculate step distance
	float step_distance = 0.1*speed*opencover::cover->frameDuration();//speed
	//calculate remaining distance
	if(totalDistance==0){totalDistance=distance;}
	totalDistance=totalDistance-step_distance;
	//calculate new position
	//vector<float> newPosition;
	newPosition = currentPos+(norm_direction_vec*step_distance);
	if (totalDistance<-step_distance)
	{
		visitedVertices++;
		totalDistance=0;
		if (visitedVertices==verticesCounter){maneuverCondition=false;}
	}
	return newPosition;
}
string Maneuver::getName(){
return name;}

bool Maneuver::getManeuverCondition(){
return maneuverCondition;}
void Maneuver::setManeuverCondition(){
if ( maneuverCondition==true)
{ maneuverCondition=false;}
if ( maneuverCondition==false)
{ maneuverCondition=true;}}
/*void Maneuver::setTargetEntityPosition(vector<float> position){
targetEntityPosition=position;
}*/
void Maneuver::setPolylineVertices(float x, float y, float z){
vector<float> vertex;
vertex.push_back(x);
vertex.push_back(y);
vertex.push_back(z);
polylineVertices.push_back(vertex);
verticesCounter++;
vertex.clear();
}
/*vector<float> Maneuver::followTrajectory(float speed, vector<float> currentPosition){
for(list<vector<float>>::iterator vertex_iter = polylineVertices.begin(); vertex_iter != polylineVertices.end(); vertex_iter++)
{
	if(arriveAtVertex==false){
	calculateNewEntityPosition(currentPosition, (*vertex_iter), speed);
	return newPosition;
}
arriveAtVertex=false;
}
maneuverCondition=false;
}*/
