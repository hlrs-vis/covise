/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeViewpoint.cpp

#include "VrmlNodeViewpoint.h"
#include "MathUtils.h"
#include "VrmlNodeType.h"
#include "VrmlScene.h"
#include "Viewer.h"
#include "VrmlSFTime.h"
#include "System.h"

using std::cerr;
using std::endl;
using namespace vrml;

void VrmlNodeViewpoint::initFields(VrmlNodeViewpoint *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
                     exposedField("fieldOfView", node->d_fieldOfView),
                     exposedField("jump", node->d_jump),
                     exposedField("orientation", node->d_orientation, [node](auto fieldValue){
                            node->d_lastOrientation = *fieldValue;
                     }),
                     exposedField("centerOfRotation", node->d_centerOfRotation),
                     exposedField("position", node->d_position, [node](auto fieldValue){
                            node->d_lastPosition = *fieldValue;
                     }),
                     exposedField("type", node->d_type),
                     field("description", node->d_description));
    
    if (t)
    {
        t->addEventIn("set_bind", VrmlField::SFBOOL);
        t->addEventIn("set_bindLast", VrmlField::SFBOOL);
        t->addEventOut("bindTime", VrmlField::SFTIME);
        t->addEventOut("isBound", VrmlField::SFBOOL);
    }
}

const char *VrmlNodeViewpoint::name() { return "Viewpoint"; }

static const float DEFAULT_FIELD_OF_VIEW = 0.785398f;

VrmlNodeViewpoint::VrmlNodeViewpoint(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , lastBind(true)
    , d_fieldOfView(DEFAULT_FIELD_OF_VIEW)
    , d_jump(true)
    , d_orientation(0.0, 0.0, 1.0, 0.0)
    , d_position(0.0, 0.0, 10.0)
    , d_centerOfRotation(0.0, 0.0, 0.0)
    , d_parentTransform(0)
    , animFraction(1.0)
{
    std::string defaultType = System::the->getConfigEntry("COVER.Plugin.Vrml97.ViewpointType");
    // ViewpointType can be one of "horizontal" "standard" or "standardNoFov"
    if (defaultType.length() > 0)
    {
        d_type.set(defaultType.c_str());
    }
    if (d_scene)
        d_scene->addViewpoint(this);
}

// need copy constructor for d_parentTransform ...

VrmlNodeViewpoint::~VrmlNodeViewpoint()
{
    if (d_scene)
        d_scene->removeViewpoint(this);
}

void VrmlNodeViewpoint::addToScene(VrmlScene *s, const char *)
{
    if (d_scene != s && (d_scene = s) != 0)
        d_scene->addViewpoint(this);
}

// Cache a pointer to (one of the) parent transforms for proper
// rendering of bindables.

void VrmlNodeViewpoint::accumulateTransform(VrmlNode *parent)
{
    d_parentTransform = parent;
}

VrmlNode *VrmlNodeViewpoint::getParentTransform() { return d_parentTransform; }

void VrmlNodeViewpoint::eventIn(double timeStamp,
                                const char *eventName,
                                const VrmlField *fieldValue)
{
    if ((strcmp(eventName, "set_bind") == 0) || (strcmp(eventName, "set_bindLast") == 0))
    {
        if (strcmp(eventName, "set_bindLast") == 0)
            lastBind = true;
        else
            lastBind = false;
        VrmlNodeViewpoint *current = d_scene->bindableViewpointTop();
        const VrmlSFBool *b = fieldValue->toSFBool();

        if (!b)
        {
            cerr << "Error: invalid value for Viewpoint::set_bind eventIn "
                 << (*fieldValue) << endl;
            return;
        }

        if (b->get()) // set_bind TRUE
        {
            if (this != current)
            {
                if (current)
                    current->eventOut(timeStamp, "isBound", VrmlSFBool(false));
                d_scene->bindablePush(this);
                eventOut(timeStamp, "isBound", VrmlSFBool(true));
                eventOut(timeStamp, "bindTime", VrmlSFTime(timeStamp));
                const char *n = name();
                const char *d = d_description.get();
                if (*n && d && *d)
                    System::the->inform("%s: %s", n, d);
                else if (d && *d)
                    System::the->inform("%s", d);
                else if (*n)
                    System::the->inform("%s", n);
            }
            else
            {
                d_scene->resetViewpoint();
            }
        }
        else // set_bind FALSE
        {
            recalcLast();
            d_scene->bindableRemove(this);
            if (this == current)
            {
                eventOut(timeStamp, "isBound", VrmlSFBool(false));
                current = d_scene->bindableViewpointTop();
                if (current)
                {
                    current->eventOut(timeStamp, "isBound", VrmlSFBool(true));
                    current->eventOut(timeStamp, "bindTime", VrmlSFTime(timeStamp));
                }
            }
        }
    }

    else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }
}

void VrmlNodeViewpoint::recalcLast() // save current position as last pos
{
    double M[16];
    //double IBM[16];
    inverseTransform(M);

    float pos[3];
    float ori[4];
    //System::the->getInvBaseMat(IBM);
    //MM( M,IBM );
    System::the->getPositionAndOrientationOfOrigin(M, pos, ori);
    // we have to add InvBaseMat to M, otherwise this does not work...

    // bis jetzt einfach soll position
    /*pos[0]=d_position.get()[0];
   pos[1]=d_position.get()[1];
   pos[2]=d_position.get()[2];
   
   ori[0]=d_orientation.get()[0];
   ori[1]=d_orientation.get()[1];
   ori[2]=d_orientation.get()[2];
   ori[3]=d_orientation.get()[3];*/

    d_lastPosition.set(pos[0], pos[1], pos[2]);
    //cerr << "orientation: "<< orientation[0] << ";"<< orientation[1] << ";"<< orientation[2] << ";"<< orientation[3] << endl;
    d_lastOrientation.set(ori[0], ori[1], ori[2], ori[3]);
}

void VrmlNodeViewpoint::setLastViewpointPosition(float *p, float *o)
{
    if (!d_jump.get())
    {
        startTime = System::the->time();
        animFraction = 0.0;
    }
    d_lastVPPosition.set(p[0], p[1], p[2]);
    d_lastVPOrientation.set(o[0], o[1], o[2], o[3]);
}

void VrmlNodeViewpoint::getPosition(float *pos, float *ori)
{
    double M[16];
    inverseTransform(M);
    if (animFraction < 1.0 && !d_jump.get())
    {
        animFraction = (float)(System::the->time() - startTime) / 2.0f;
        if (animFraction > 1.0)
            animFraction = 1.0;
        if (animFraction < 0.0)
            animFraction = 0.0;
        pos[0] = d_lastVPPosition.x() + animFraction * (d_position.x() - d_lastVPPosition.x());
        pos[1] = d_lastVPPosition.y() + animFraction * (d_position.y() - d_lastVPPosition.y());
        pos[2] = d_lastVPPosition.z() + animFraction * (d_position.z() - d_lastVPPosition.z());

        VrmlSFRotation r(d_lastVPOrientation.x(), d_lastVPOrientation.y(), d_lastVPOrientation.z(), d_lastVPOrientation.r());
        d_orientation.slerp(&r, 1.0f - animFraction);
        ori[0] = r.x();
        ori[1] = r.y();
        ori[2] = r.z();
        ori[3] = r.r();
        System::the->transformByMatrix(M, pos, ori);
    }
    else
    {
        pos[0] = d_position.x();
        pos[1] = d_position.y();
        pos[2] = d_position.z();
        ori[0] = d_orientation.x();
        ori[1] = d_orientation.y();
        ori[2] = d_orientation.z();
        ori[3] = d_orientation.r();
        System::the->transformByMatrix(M, pos, ori);
    }
}

void VrmlNodeViewpoint::getLastPosition(float *pos, float *ori)
{
    if (animFraction < 1.0 && !d_jump.get())
    {
        animFraction = (float)(System::the->time() - startTime) / 2.0f;
        if (animFraction > 1.0)
            animFraction = 1.0;
        if (animFraction < 0.0)
            animFraction = 0.0;
        pos[0] = d_lastVPPosition.x() + animFraction * (d_lastPosition.x() - d_lastVPPosition.x());
        pos[1] = d_lastVPPosition.y() + animFraction * (d_lastPosition.y() - d_lastVPPosition.y());
        pos[2] = d_lastVPPosition.z() + animFraction * (d_lastPosition.z() - d_lastVPPosition.z());
        //cerr << animFraction<< " p "<<  pos[0] <<  endl;
        VrmlSFRotation r(d_lastVPOrientation.x(), d_lastVPOrientation.y(), d_lastVPOrientation.z(), d_lastVPOrientation.r());
        d_lastOrientation.slerp(&r, 1.0f - animFraction);
        ori[0] = r.x();
        ori[1] = r.y();
        ori[2] = r.z();
        ori[3] = r.r();
    }
    else
    {
        pos[0] = d_lastPosition.x();
        pos[1] = d_lastPosition.y();
        pos[2] = d_lastPosition.z();
        ori[0] = d_lastOrientation.x();
        ori[1] = d_lastOrientation.y();
        ori[2] = d_lastOrientation.z();
        ori[3] = d_lastOrientation.r();
    }
}
