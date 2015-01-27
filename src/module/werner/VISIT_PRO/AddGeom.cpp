/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS AddGeom
//
// This class @@@
//
// Initial version: 2002-07-23 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include "AddGeom.h"
#include "Attachable.h"
#include <api/coModule.h>
#include <assert.h>
#include <api/coFeedback.h>

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Static Variable initializers
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Constructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

AddGeom::AddGeom(int id, coModule *mod)
{
    // save the ID
    d_id = id;

    // Case switching group start -
    char buf[64];

    sprintf(buf, "Vent_%d:Pos", id);
    p_pos = mod->addFloatVectorParam(buf, "Position");

    sprintf(buf, "Vent_%d:Euler", id);
    p_euler = mod->addFloatVectorParam(buf, "Euler Angles");

    sprintf(buf, "Vent_%d:Rot", id);
    p_rot = mod->addFloatVectorParam(buf, "Rotation Matrix");
    mod->paraEndCase();

    d_attachedPart = NULL; // clear() deletes it!
    clear(); // set everything zero
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Destructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

AddGeom::~AddGeom()
{
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Operations
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// cleanup everything
void AddGeom::clear()
{
    p_pos->setValue(0.0, 0.0, 0.0);
    d_pos[0] = d_pos[1] = d_pos[2] = 0.0;

    p_euler->setValue(0.0, 0.0, 0.0);
    d_euler[0] = d_euler[1] = d_euler[2] = 0.0;

    d_coverRot[0] = 1.0;
    d_coverRot[1] = 0.0;
    d_coverRot[2] = 0.0;
    d_coverRot[3] = 0.0;
    d_coverRot[4] = 1.0;
    d_coverRot[5] = 0.0;
    d_coverRot[6] = 0.0;
    d_coverRot[7] = 0.0;
    d_coverRot[8] = 1.0;
    p_rot->setValue(9, d_coverRot);

    delete d_attachedPart;
    d_attachedPart = NULL;
}

// assign this addGeometry to a specific type
void AddGeom::setAttachable(const Attachable *attachable)
{
    delete d_attachedPart;
    if (attachable)
        d_attachedPart = new Attachable(*attachable);
    else
        d_attachedPart = NULL;
}

void AddGeom::printStarObj(ostream &str) const
{
    if (d_attachedPart)
    {
        str << d_attachedPart->getObjPath()
            << " " << d_attachedPart->getChoiceLabel();
        int j;
        for (j = 0; j < 3; j++)
            str << " " << p_euler->getValue(j);
        for (j = 0; j < 3; j++)
            str << " " << p_pos->getValue(j);
        str << endl;
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Attribute request/set functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// get the gometry object, including all interactions
coDistributedObject *AddGeom::getCurrentObject(const char *baseName, const char *unit) const
{
    if (!d_attachedPart)
        return NULL;

    coDistributedObject *obj = d_attachedPart->getObjDO(baseName, d_id, unit);
    if (!obj)
        return NULL;

    // now add all interactors...
    coFeedback feedback("VisitProPlugin");
    char str[30];
    sprintf(str, "VENT_%d", d_id);
    feedback.addString(str);
    feedback.addPara(p_pos);
    feedback.addPara(p_rot);
    feedback.addPara(p_euler);
    feedback.apply(obj);

    // return the result
    return obj;
}

// check whether this one is attached
bool AddGeom::isAttached() const
{
    return (NULL != d_attachedPart);
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Internally used functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++ Prevent auto-generated functions by assert or implement
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// Copy-Constructor: NOT IMPLEMENTED
AddGeom::AddGeom(const AddGeom &)
{
    assert(0);
}

/// Assignment operator: NOT IMPLEMENTED
AddGeom &AddGeom::operator=(const AddGeom &)
{
    assert(0);
    return *this;
}

/// Default constructor: NOT IMPLEMENTED
AddGeom::AddGeom()
{
    assert(0);
}
