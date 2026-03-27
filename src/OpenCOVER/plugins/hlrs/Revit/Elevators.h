/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#ifndef _Revit_PLUGIN_Elevators_H
#define _Revit_PLUGIN_Elevators_H

#include <cover/coVRPluginSupport.h>
#include <osg/MatrixTransform>
#include <net/tokenbuffer.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiHit.h>
#include <OpenVRUI/osg/OSGVruiNode.h>
#include <OpenVRUI/coCombinedButtonInteraction.h>

class Elevator;

class ElevatorPart: public vrui::coAction
{
public:
    enum PartType
    {
        PartUdefined = -1,
        Cabin = 0,
        Landing = 1,
        Door = 2,
        Button = 3
    };

    enum CabinState
    {
        MovementUndefined = -1,
        Idle = 0,
        Moving = 1,
        Opening = 2,
        Closing,
        DoorOpen,
        DoorClosed,
        MoveUp,
        MoveDown
    };
    ElevatorPart(Elevator *e) { elevator = e; };
    Elevator *elevator;
    std::string name;
    osg::MatrixTransform *transformNode;
    osg::Matrix initialTransform;
    int ID;
    double elevation;
    std::string levelName;
    osg::BoundingBox boundingBox;
    PartType type = PartUdefined;
    std::vector<ElevatorPart *> doors;
    int levelNumber;
    ElevatorPart *buttonLanding; // landing this button is for.
    int doorNumber;
    virtual int hit(vrui::vruiHit *hit);

    void goTo(int landing);
    std::set<int> stationList; // landings this cabin should go to
    double timeoutStart;
    CabinState state;
    CabinState travelDirection;
    CabinState oldTravelDirection;
    int currentLanding;
    float carPos = 0;
    float destinationY = 0;
    float startingY = 0;
    float v = 0;
    float a;
    double vMax = 5;
    double aMax = 1.2;
    float doorOpenTime = 6;
    float doorFraction = 0; // 0 = closed, 1 = open
    double openingDistance = 10;
    osg::Vec3 initialTranslation;
    
    void arrivedAtDestination();
    void openDoor();
    void closeDoor();
    bool update(osg::Vec3 &viewerPosition);
};


class Elevator
{
public:
    Elevator(int id, const char *Name, const std::string &elevatorName);
    std::string name;
    int ID;
    std::string elevatorName;
    ElevatorPart *cabin;
    std::vector<ElevatorPart *> landings;
    std::vector<ElevatorPart *> buttons;
    bool update(osg::Vec3 &viewerPosition); // returns false if updates are done and it can be removed from the list
    void addPart(const std::string &familyName, const std::string &subType, osg::MatrixTransform *mt, covise::TokenBuffer &tb);
};



#endif
