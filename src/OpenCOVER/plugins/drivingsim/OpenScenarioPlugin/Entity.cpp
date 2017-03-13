#include "Entity.h"


Entity::Entity(string entityName,  Vehicle* car):name(entityName), car(car){}

void Entity::move(){
//calculate step distance
float step_distance = 0.1*speed*opencover::cover->frameDuration();
entityPosition[0] = entityPosition[0]+step_distance;
}

vector<float> Entity::getPosition(){
return entityPosition;
}

void Entity::setPosition(vector<float> newPosition){
entityPosition = newPosition;
}  

string Entity::getName(){
return name;
}

void Entity::setSpeed(float speed_temp){
speed=speed_temp;}

float Entity::getSpeed(){
return speed;}