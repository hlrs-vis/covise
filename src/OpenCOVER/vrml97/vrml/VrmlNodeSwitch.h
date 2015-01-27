/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeSwitch.h

#ifndef _VRMLNODESWITCH_
#define _VRMLNODESWITCH_

#include "VrmlMFNode.h"
#include "VrmlSFInt.h"

#include "VrmlNode.h"

#include "VrmlNodeChild.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeSwitch : public VrmlNodeChild
{

public:
    // Define the fields of all built in switch nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeSwitch(VrmlScene *);
    virtual ~VrmlNodeSwitch();

    virtual VrmlNode *cloneMe() const;
    void cloneChildren(VrmlNamespace *);

    virtual VrmlNodeSwitch *toSwitch() const; //LarryD

    virtual bool isModified() const;

    virtual void clearFlags();

    virtual void addToScene(VrmlScene *s, const char *relUrl);

    virtual void copyRoutes(VrmlNamespace *ns);

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void render(Viewer *);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    virtual const VrmlField *getField(const char *fieldName) const;

    virtual void accumulateTransform(VrmlNode *parent);

    VrmlMFNode *getChoiceNodes()
    {
        return &d_choice;
    }
    virtual int getWhichChoice()
    {
        return d_whichChoice.get();
    }

protected:
    VrmlMFNode d_choice;
    VrmlSFInt d_whichChoice;
    bool firstTime;
};
}
#endif //_VRMLNODESWITCH_
