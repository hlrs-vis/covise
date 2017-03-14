#include "Maneuver.h"
#include <cover/coVRPluginSupport.h>
#include <iterator>
#include <math.h>

Maneuver::Maneuver(string name):name(name){currentExecution=0;maneuverCondition=true;totalDistance=0;}
Maneuver::~Maneuver(){}
vector<float> Maneuver::calculateNewEntityPosition(vector<float> currentPosition, float speed){
//substract vectors
vector<float> direction_vec;
transform(targetEntityPosition.begin(), targetEntityPosition.end(), currentPosition.begin(), std::back_inserter(direction_vec), [&](float t, float c)
{return abs(t - c);});
//calculate step distance
float step_distance = 0.1*speed*opencover::cover->frameDuration();//speed
//calculate remaining distance
float distance = sqrt(pow(direction_vec[0],2)+pow(direction_vec[1],2)+pow(direction_vec[2],2));
if(totalDistance==0){totalDistance=distance;}
totalDistance=totalDistance-step_distance;
//calculate new position
vector<float> newPosition;
newPosition.push_back(currentPosition[0]+(step_distance/distance)*direction_vec[0]);
newPosition.push_back(currentPosition[1]+(step_distance/distance)*direction_vec[1]);
newPosition.push_back(currentPosition[2]+(step_distance/distance)*direction_vec[2]);
newPosition.push_back(currentPosition[3]);
newPosition.push_back(currentPosition[4]);
newPosition.push_back(currentPosition[5]);
if (totalDistance<-step_distance){maneuverCondition=false;}
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
void Maneuver::setTargetEntityPosition(vector<float> position){
targetEntityPosition=position;
}