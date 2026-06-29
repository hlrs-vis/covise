/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /****************************************************************************\
 **                                                            (C)2009 HLRS  **
 **                                                                          **
 ** Description: Elevators for Revit Plugin                                      **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** Mar-09  v1	    				       		                             **
 **                                                                          **
 **                                                                          **
 \****************************************************************************/
#define QT_NO_EMIT

#include "Elevators.h"
#include "RevitPlugin.h"
#include <OpenVRUI/sginterface/vruiRendererInterface.h>


Elevator::Elevator(int id, const char *Name, const std::string &evn)
{
    ID = id;
    name = Name;
    elevatorName = evn;
}
bool Elevator::update(osg::Vec3 &viewerPosition)
{

    if ((cabin->isIdle()))
    {
        {
            // tell it to move to next stop
            cabin->moveToNext();
        }
        return false;
    }
    cabin->update(viewerPosition);
    for (auto &landing : landings)
    {
        if (landing != NULL)
        {
            landing->update(viewerPosition);
        }
    }
    return true;
}

int ElevatorPart::hit(vrui::vruiHit *hit)
{

    vruiRendererInterface *renderer = vruiRendererInterface::the();
    vruiButtons *buttons = hit->isMouseHit()
        ? renderer->getMouseButtons()
        : renderer->getButtons();

    if (buttons->wasPressed(vruiButtons::ACTION_BUTTON))
    {
        // left Button was pressed
    }
    else if (buttons->wasReleased(vruiButtons::ACTION_BUTTON))
    {
        // left Button was released
        elevator->cabin->goTo(levelNumber);
    }

    return vrui::coAction::Result::ACTION_DONE;
}
/// Miss is called once after a hit, if the button is not intersected anymore.
void ElevatorPart::miss()
{
    vruiRendererInterface::the()->miss(this);

}

bool ElevatorPart::isIdle()
{

    for (auto &door : doors)
    {
        if (door->state != DoorClosed && door->state != Idle)
        {
            return false;
        }
    }
    return state == Idle;
}

void ElevatorPart::openDoor()
    {
    for (auto &door : doors)
    {
        door->state = Opening;
        door->timeoutStart = cover->frameTime();
    }
}

void ElevatorPart::setDestination(int landing)
{
    if (landing != currentLanding)
    {
        state = Moving;

        startingY = elevator->landings[currentLanding]->elevation;

        destinationY = elevator->landings[landing]->elevation;
    }
}
void ElevatorPart::setTravelDirection(CabinState t)
{
    travelDirection = t;
}
void ElevatorPart::moveToNext()
{
    if (stationList.size() == 0)
        return; // nowhere to go
    int landing = *stationList.begin();
    if (landing > currentLanding)
    {
        setTravelDirection(MoveUp);
    }
    else if (landing < currentLanding)
    {
        setTravelDirection(MoveDown);
    }
    else
    {
        // we already are at this station, just remove it, this was a duplicate entry
        stationList.pop_front();
    }
    setDestination(landing);
}
bool ElevatorPart::update(osg::Vec3& viewerPosition) 
{
    if (state == Moving)
    {

        float dt = cover->frameDuration();
        if (dt > 1000) // first frameDuration is off because last FrameTime is 0
            dt = 0.00001;
        if (carPos != destinationY) // moving vertically
        {
            float direction;
            float diff = fabs(destinationY - carPos);
            float diffS = fabs(startingY - carPos);

            float v2 = v * v;
            float bakeDistance = (v2 / (2 * aMax)) * 1.5; // distance the XCar travels until it stops at max decelleration

            if (carPos < destinationY)
            {
                direction = 1;
            }
            else
            {
                direction = -1;
            }

            if (diff > (v2 / (2 * aMax)) * 1.5)
            { // beschleunigen
                a += 0.5 * dt;
                if (a > aMax)
                    a = aMax;
                v += a * dt;
                if (v > vMax)
                    v = vMax;
                carPos += direction * v * dt;
            }
            else
            { // verzögern
                if (diff > 0.0001)
                {
                    a = v2 / (2 * diff);
                    v -= a * dt;
                }
                else
                {
                    a = 0;
                    v = 0;
                }
                if (direction * (destinationY - carPos) < 0)
                {
                    carPos = destinationY;
                    v = 0;
                }
                if (v < 0)
                {
                    v = 0;
                }
                else
                {
                    carPos += direction * v * dt;
                }
            }

            if (direction > 0 && carPos > destinationY)
            {
                carPos = destinationY;
                a = 0;
                v = 0;
            }
            if (direction < 0 && carPos < destinationY)
            {
                carPos = destinationY;
                a = 0;
                v = 0;
            }
            osg::Matrix newTrans = transformNode->getMatrix();
            newTrans.setTrans(initialTranslation + osg::Vec3(0, 0, carPos));
            transformNode->setMatrix(newTrans);
        }
        else // we are there
        {
            v = 0;
            a = 0;
            if (stationList.size() > 0)
            {
                currentLanding = *stationList.begin();
            }
            arrivedAtDestination();
        }
    }
    else if (state == Opening)
    {
        if (abs(cover->frameTime() - timeoutStart) > 3)
        {
            timeoutStart = cover->frameTime();
            state = DoorOpen;
        }
        float dt = cover->frameDuration();
        if (dt > 1000) // first frameDuration is off because last FrameTime is 0
            dt = 0.00001;
        if (doorFraction != 1) // opening
        {
            doorFraction += 0.5 * dt;
            if (doorFraction > 1)
            {
                doorFraction = 1;
                state = DoorOpen;
            }
            float doorPos = doorFraction * openingDistance * (doorNumber+1);

            osg::Matrix newTrans = transformNode->getMatrix();
            newTrans.setTrans(initialTranslation + osg::Vec3(doorPos, 0, 0));
            transformNode->setMatrix(newTrans);
        }
        else // we are there
        {
            state = DoorOpen;
        }
    }
    else if (state == Closing)
    {
        if (abs(cover->frameTime() - timeoutStart) > 3)
        {
            timeoutStart = cover->frameTime();
            state = DoorClosed;
        }
        float dt = cover->frameDuration();
        if (dt > 1000) // first frameDuration is off because last FrameTime is 0
            dt = 0.00001;
        if (doorFraction != 0) // closing
        {
            doorFraction -= 0.5 * dt;
            if (doorFraction < 0)
            {
                doorFraction = 0;
                state = DoorClosed;
            }
            float doorPos = doorFraction * openingDistance * (doorNumber+1);

            osg::Matrix newTrans = transformNode->getMatrix();
            newTrans.setTrans(initialTranslation + osg::Vec3(doorPos, 0, 0));
            transformNode->setMatrix(newTrans);
        }
        else // we are there
        {
            state = DoorClosed;
        }
    }
    else if (state == DoorOpen)
    {
        if ((abs(cover->frameTime() - timeoutStart) > doorOpenTime) /* && openButton->getState() == false*/)
        {
            timeoutStart = cover->frameTime();
            if (currentLanding < elevator->landings.size() && elevator->landings[currentLanding] != NULL)
            {
                elevator->landings[currentLanding]->closeDoor();
            }
            state = Closing;
        }
    }
    for (auto &door : doors)
    {
        if (door != NULL)
        {
            door->update(viewerPosition);
        }
    }
    return false;
}
void ElevatorPart::closeDoor()
{
    state = Closing;
}
void Elevator::addPart(const std::string &familyName, const std::string &subType, osg::MatrixTransform* mt, covise::TokenBuffer& tb)
{

    ElevatorPart *ep = new ElevatorPart(this);
    ep->transformNode = mt;
    std::string levelName;
    tb >> ep->levelName;
    tb >> ep->elevation;
    std::string name;
    do
    {
        tb >> name;
        if (name != "Undefined" && name != "end")
        {
            int storageType;
            std::string typeName;
            tb >> storageType;
            tb >> typeName;
            double d;
            int i;
            int ElementReferenceID;
            std::string Value;
            switch (storageType)
            {
            case RevitPlugin::Double:
                tb >> d;
                break;
            case RevitPlugin::ElementId:
                tb >> ElementReferenceID;
                tb >> Value;
                break;
            case RevitPlugin::Integer:
                tb >> i;
                break;
            case RevitPlugin::String:
                tb >> Value;
                break;
            default:
                tb >> Value;
                break;
            }
            if (name == "elevatorSpeed")
            {
                ep->vMax = d*10;
            }
            else if (name.substr(0,20) == "elevatorAcceleration")
            {
                ep->aMax = d*10;
            }
            else if (name == "elevatorOpening")
            {
                ep->openingDistance = d;
            }
        }
    } while (name != "end");
    ep->initialTranslation = ep->transformNode->getMatrix().getTrans();
    if (levelName != "undefined")
    {
        // this object is attached to an actial level
    }

    if (subType == "elevatorCabin")
    {
        ep->type = ElevatorPart::Cabin;
        cabin = ep;
        cabin->destinationY = cabin->carPos = cabin->elevation; // car is on this elevation
    }
    else if (subType.substr(0,12) == "elevatorDoor")
    {
        ep->type = ElevatorPart::Door;
        std::string doorNumber = subType.substr(13, 1);
        ep->doorNumber = std::stoi(doorNumber);
        if (familyName.substr(0, 5) == "Cabin")
        {
            cabin->doors.push_back(ep);
            if (subType.substr(16, 4) == "ight")
            {

            }
            else
            {
                ep->doorNumber *= -1;
            }
        }
        else
        {
            ElevatorPart *landing=nullptr;
            for (const auto &landingPart : landings)
            {
                if (landingPart->elevation == ep->elevation)
                {
                    landing = landingPart;
                }
            }
            if (landing == nullptr)
            {
                landing = new ElevatorPart(this);
                landing->type = ElevatorPart::Landing;
                landing->elevation = ep->elevation;
                landing->levelName = ep->levelName;
                landings.push_back(landing);
                std::sort(landings.begin(), landings.end(),
                    [](ElevatorPart *const &a, ElevatorPart *const &b)
                    { return a->elevation < b->elevation; });
            }
            landing->doors.push_back(ep);
        }
    }
    else if (subType.substr(0, 14) == "elevatorButton")
    {
        ep->type = ElevatorPart::Button;
        buttons.push_back(ep);
        if (subType.length() > 14)
        {
            std::string levelNumber = subType.substr(15);
            ep->levelNumber = std::stoi(levelNumber);
        }
        else
        {
            int levelNumber = 0; // assign level numbers automatically
            for (const auto &landingPart : landings)
            {
                if (landingPart->elevation == ep->elevation)
                {
                    ep->buttonLanding = landingPart;
                    ep->levelNumber = levelNumber;
                    break;
                }
                else
                {
                    levelNumber++;
                }
            }
        }

        OSGVruiNode *vNode = new OSGVruiNode(mt);
        vruiIntersection::getIntersectorForAction("coAction")->add(vNode, ep);

    }
}


void ElevatorPart::goTo(int landing)
{
    if (landing >= 0 && landing < elevator->landings.size())
    {
        stationList.push_back(landing);
        if (currentLanding == landing && isIdle())
            arrivedAtDestination();
    }
}


void ElevatorPart::arrivedAtDestination() // the Cabin arrived at its destination
{
    timeoutStart = cover->frameTime();
    openDoor();

    if (stationList.size() > 0)
    {
        int landing = *stationList.begin();
        currentLanding = landing;
    }
    state = Idle;

    if (currentLanding < elevator->landings.size() && elevator->landings[currentLanding] != NULL)
    {
        elevator->landings[currentLanding]->openDoor();
    }

    if (stationList.size() > 0)
    {
        stationList.pop_front();
        if (stationList.size() > 0)
        {
            int Landing = *stationList.begin();

            if (Landing != currentLanding)
            {
                if (Landing > currentLanding)
                    travelDirection = MoveUp;
                else
                    travelDirection = MoveDown;
            }
            if (travelDirection != oldTravelDirection)
            {
                oldTravelDirection = travelDirection;
            }
        }
    }
}
