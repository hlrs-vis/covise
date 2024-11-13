/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeSpaceSensor.cpp

#include "config.h"
#include "VrmlNodeSpaceSensor.h"
#include "VrmlNodeType.h"

#include "MathUtils.h"
#include "System.h"
#include "Viewer.h"
#include "VrmlScene.h"

using std::cerr;
using std::endl;
using namespace vrml;

// SpaceSensor factory.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeSpaceSensor(scene);
}

// Define the built in VrmlNodeType:: "SpaceSensor" fields


void VrmlNodeSpaceSensor::initFields(VrmlNodeSpaceSensor *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
                     exposedField("autoOffset", node->d_autoOffset),
                     exposedField("enabled", node->d_enabled),
                     exposedField("rotationEnabled", node->d_rotationEnabled),
                     exposedField("maxPosition", node->d_maxPosition),
                     exposedField("minPosition", node->d_minPosition),
                     exposedField("offset", node->d_offset));
    if(t)
    {
        t->addEventOut("isActive", VrmlField::SFBOOL);
        t->addEventOut("translation_changed", VrmlField::SFVEC3F);
        t->addEventOut("trackPoint_changed", VrmlField::SFVEC3F);
        t->addEventOut("rotation_changed", VrmlField::SFROTATION);
    }

}

const char *VrmlNodeSpaceSensor::name(){return "SpaceSensor";}

VrmlNodeSpaceSensor::VrmlNodeSpaceSensor(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , d_autoOffset(true)
    , d_enabled(true)
    , d_rotationEnabled(false)
    , d_maxPosition(-1.0, -1.0, -1.0)
    , d_minPosition(-1.0, -1.0, -1.0)
    , d_isActive(false)
    , d_rotation(0, 1, 0, 0)
    , d_parentTransform(0)
{
    Midentity(orientation);
    Midentity(startOrientation);
    setModified();
}

// Cache a pointer to (one of the) parent transforms for converting
// hits into local coords.

void VrmlNodeSpaceSensor::accumulateTransform(VrmlNode *parent)
{
    d_parentTransform = parent;
}

VrmlNode *VrmlNodeSpaceSensor::getParentTransform() { return d_parentTransform; }

// This is not correct. The local coords are computed for one instance,
// need to convert p to local coords for each instance (DEF/USE) of the
// sensor...

void VrmlNodeSpaceSensor::activate(double timeStamp,
                                   bool isActive,
                                   double *p, const double *N)
{
	static bool debug = true;
	if (N == 0)
    {
		if (debug)
		{
			cerr << "this has to be fixed, the matrix is not transmitted yet" << endl;
			debug = false;
		}
        return;
    }

    // Become active
    if (isActive && !d_isActive.get())
    {
        d_isActive.set(isActive);

        float V[3] = { (float)p[0], (float)p[1], (float)p[2] };
        double M[16];
        inverseTransform(M);
        VM(V, M, V);
        d_activationPoint.set(V[0], V[1], V[2]);

        Mcopy(SM, M);
        double pointerLocal[16];
        Mmult(pointerLocal, N, M);
        Minvert(invStartMove, pointerLocal);
        Mcopy(startOrientation, orientation);

#if 0
      System::the->warn(" SpaceSensor: activate at (%g %g %g)\n",
         p[0],p[1],p[2]);
      System::the->warn(" SpaceSensor: local coord (%g %g %g)\n",
         V[0],V[1],V[2]);
#endif
        eventOut(timeStamp, "isActive", d_isActive);
    }

    // Become inactive
    else if (!isActive && d_isActive.get())
    {
#if 0
      System::the->warn(" SpaceSensor: deactivate\n");
#endif
        d_isActive.set(isActive);
        eventOut(timeStamp, "isActive", d_isActive);

        // auto offset
        if (d_autoOffset.get())
        {
            d_offset = d_translation;
            eventOut(timeStamp, "offset_changed", d_offset);
        }
    }

    // Tracking
    else if (isActive)
    {
        float V[3] = { (float)p[0], (float)p[1], (float)p[2] };
        double M[16];
        if (d_rotationEnabled.get())
        {
            double pointerLocal[16];
            Mmult(pointerLocal, N, SM);
            double moveMat[16];
            Mmult(moveMat, invStartMove, pointerLocal);
            float startPos[3] = { d_activationPoint.x(), d_activationPoint.y(), d_activationPoint.z() };
            float currentPos[3];
            VM(currentPos, moveMat, startPos);
            float fromOriginNew[3];
            VM(fromOriginNew, moveMat, startPos);
            for (int i = 0; i < 3; i++)
                currentPos[i] = currentPos[i] - fromOriginNew[i] + startPos[i];

            Mmult(orientation, startOrientation, moveMat);

            MgetRot(d_rotation.get(), &(d_rotation.get()[3]), orientation);
            eventOut(timeStamp, "rotation_changed", d_rotation);

            V[0] = currentPos[0];
            V[1] = currentPos[1];
            V[2] = currentPos[2];
        }
        else
        {
            inverseTransform(M);
            VM(V, M, V);
        }
#if 0
      System::the->warn(" SpaceSensor: track at (%g %g %g)\n",
         p[0],p[1],p[2]);

      System::the->warn(" SpaceSensor: local cd (%g %g %g)\n",
         V[0],V[1],V[2]);
#endif
        d_trackPoint.set(V[0], V[1], V[2]);
        eventOut(timeStamp, "trackPoint_changed", d_trackPoint);

        float t[3];
        t[0] = V[0] - d_activationPoint.x() + d_offset.x();
        t[1] = V[1] - d_activationPoint.y() + d_offset.y();
        t[2] = V[2] - d_activationPoint.z() + d_offset.z();

        if (d_minPosition.x() == d_maxPosition.x())
            t[0] = d_minPosition.x();
        else if (d_minPosition.x() < d_maxPosition.x())
        {
            if (t[0] < d_minPosition.x())
                t[0] = d_minPosition.x();
            else if (t[0] > d_maxPosition.x())
                t[0] = d_maxPosition.x();
        }

        if (d_minPosition.y() == d_maxPosition.y())
            t[1] = d_minPosition.y();
        else if (d_minPosition.y() < d_maxPosition.y())
        {
            if (t[1] < d_minPosition.y())
                t[1] = d_minPosition.y();
            else if (t[1] > d_maxPosition.y())
                t[1] = d_maxPosition.y();
        }
        if (d_minPosition.z() == d_maxPosition.z())
            t[2] = d_minPosition.z();
        else if (d_minPosition.z() < d_maxPosition.z())
        {
            if (t[2] < d_minPosition.z())
                t[2] = d_minPosition.z();
            else if (t[2] > d_maxPosition.z())
                t[2] = d_maxPosition.z();
        }

        d_translation.set(t[0], t[1], t[2]);
        eventOut(timeStamp, "translation_changed", d_translation);
    }
}
