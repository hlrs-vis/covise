/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <coSensor.h>
#include <cassert>

// undef this to get no START/END messaged
#undef VERBOSE

#ifdef VERBOSE
#include <cstdio>

#define START(msg) fprintf(stderr, "START: %s\n", (msg))
#define END(msg) fprintf(stderr, "END:   %s\n", (msg))
#else
#define START(msg)
#define END(msg)
#endif

using namespace vrui;

coSensor::coSensor(osg::Node *n, vrui::coInteraction::InteractionType type, vrui::coInteraction::InteractionPriority priority)
{
    START("coSensor::coSensor");
    node = n;
    active = 0;
    osg::BoundingSphere bsphere;
    bsphere = node->getBound();
    threshold = bsphere.radius();
    threshold *= threshold;
    sqrDistance = 0;
    buttonSensitive = 0;
    enabled = 1;

    std::string name = "NoNodeSensor";
    if (node)
    {
        name = node->getName();
        if (name.empty())
        {
            name = "coSensor";
        }
    }

    interaction = new coCombinedButtonInteraction(type, name, priority);
}

void coSensor::setButtonSensitive(int s)
{
    START("coSensor::setButtonSensitive");
    buttonSensitive = s;
}

int coSensor::getType()
{
    START("coSensor::getType");
    return (NONE);
}

coPickSensor::coPickSensor(osg::Node *n, vrui::coInteraction::InteractionType type, vrui::coInteraction::InteractionPriority priority)
    : coSensor(n, type, priority)
{
    START("coPickSensor::coPickSensor");
    vNode = new OSGVruiNode(n);
    vruiIntersection::getIntersectorForAction("coAction")->add(vNode, this);
    hitPoint.set(0, 0, 0);
    hitActive = false;
}

coPickSensor::~coPickSensor()
{
    vruiIntersection::getIntersectorForAction("coAction")->remove(vNode);
    delete vNode;
}
/**
@param hitPoint,hit  Performer intersection information
@return ACTION_CALL_ON_MISS if you want miss to be called,
otherwise ACTION_DONE is returned
*/
int coPickSensor::hit(vruiHit *hit)
{
    hitActive = true;
    coVector v = hit->getWorldIntersectionPoint();
    hitPoint.set(v[0], v[1], v[2]);
    if (!interaction->isRegistered())
        vrui::coInteractionManager::the()->registerInteraction(interaction);
    interaction->setHitByMouse(hit->isMouseHit());
    return ACTION_CALL_ON_MISS;
}

/// Miss is called once after a hit, if the button is not intersected anymore.
void coPickSensor::miss()
{
    if (interaction->isRegistered())
        vrui::coInteractionManager::the()->unregisterInteraction(interaction);
    hitActive = false;
}

int coPickSensor::getType()
{
    START("coPickSensor::getType");
    return (coSensor::PICK);
}

void coPickSensor::update()
{
    //START("coPickSensor::update");
    if (interaction->wasStarted())
    {
        assert(enabled);
        active = 1;
        activate();
    }
    if (interaction->wasStopped())
    {
        active = 0;
        disactivate();
    }
}

void coSensor::update()
{
    //START("coSensor::update");
    calcDistance();
    bool newActiveFlag = sqrDistance <= threshold;
    if (newActiveFlag)
    {
        if (enabled && !interaction->isRegistered())
            vrui::coInteractionManager::the()->registerInteraction(interaction);
    }
    else
    {
        if (interaction->isRegistered())
            vrui::coInteractionManager::the()->unregisterInteraction(interaction);
    }

    if (interaction->wasStarted())
    {
        assert(enabled);
        active = 1;
        activate();
    }
    if (interaction->wasStopped())
    {
        active = 0;
        disactivate();
    }
}

void coSensor::activate()
{
    START("coSensor::activate");
    assert(active == 1);
}

void coSensor::disactivate()
{
    START("coSensor::disactivate")
    assert(active == 0);
}

void coSensor::enable()
{
    START("coSensor::enable");
    enabled = 1;
}

void coSensor::disable()
{
    START("coSensor::disable");
    enabled = 0;
    if (interaction->isRegistered())
        vrui::coInteractionManager::the()->unregisterInteraction(interaction);
}

coSensor::~coSensor()
{
    START("coSensor::~coSensor");
    if (active)
    {
        active = 0;
        disactivate();
    }
    if (interaction->isRegistered())
        vrui::coInteractionManager::the()->unregisterInteraction(interaction);
    delete interaction;
}

coSensorList::coSensorList()
{
    noDelete = 1;
}

void coSensorList::update()
{
    //START("");
    reset();
    while (current())
    {
        if (current()->getType() == coSensor::NONE)
        {
            coSensor *s = current();
            delete s;
            reset();
            while ((current()) && (current()->getType() == coSensor::NONE))
            {
                coSensor *s = current();
                delete s;
                reset();
            }
        }
        if (current())
        {
            current()->update();
            // hack for VRML sensors or others who want to be removed:
            // if they want to be removed, then they set their type to NONE
            // (you can't remove yourself)
            if (current()->getType() == coSensor::NONE)
            {
                coSensor *s = current();
                delete s;
                reset();
                while ((current()) && (current()->getType() == coSensor::NONE))
                {
                    coSensor *s = current();
                    delete s;
                    reset();
                }
            }
            else
                next();
        }
    }
}
