#include "MyPosition.h"

MyPosition::MyPosition():oscPosition()
{}

MyPosition::~MyPosition()
{}


std::string MyPosition::getPositionElement(){
    if(Lane.exists()){
        return Lane.getName();
    }
    else if(RelativeLane.exists()){
        return RelativeLane.getName();
    }
    else if(RelativeObject.exists()){
        return RelativeObject.getName();
    }
    else if(RelativeRoad.exists()){
        return RelativeRoad.getName();
    }
    else if(RelativeWorld.exists()){
        dx = RelativeWorld->dx.getValue();
        dy = RelativeWorld->dy.getValue();
        dz = RelativeWorld->dz.getValue();
        return RelativeWorld.getName();
    }
    else if(Road.exists()){
        return Road.getName();
    }
    else if(Route.exists()){
        return Route.getName();
    }
    else if(World.exists()){
        return World.getName();
    }

    return "0";
}
