/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2007 ZAIK  **
 **                                                                          **
 ** Description:  BillboardSwitch Plugin                                     **
 **        implements a new VRML Node Type BillboardSwitch.                  **
 **        it is a mashup between / and based upon                           **
 **                the Billboard and Switch Nodes.                           **
 **                                                                          **
 **        Based on the angle under which the BillboardSwitch is watched,    **
 **        the node switches between its childs and billboards them.         **
 **                                                                          **
 **                         structure for BillboardSwitch node:              **
 **                 BillboardSwitch {                                        **
 **                     exposedField   SFVec3f axisOfRotation                **
 **                     field          MFFloat angle  []                     **
 **                     eventOut       MFInt   activeChildChanged            **
 **                     exposedFields  MFNode  choice  []                    **
 **                     exposedFields  MFNode  alternative  []               **
 **                 }                                                        **
 **                                                                          **
 **        with axisOfRotation like Axis in Billboard Node                   **
 **             angle are the angles under which the childs are switched     **
 **             activeChildChanged indicates if the active Child changed     **
 **             choice are the different childs which are switched           **
 **                                     and billboarded                      **
 **             alternative is a workaround for other VRML Browsers          **
 **                         with the PROTO definition, the BillboardSwitch   **
 **                         works like a normal Billboard with alternative   **
 **                         as its childs.                                   **
 **                                                                          **
 **                                                                          **
 **                                                                          **
 ** Author: Hauke Fuehres, based on the VRML Billboard and Switch Nodes      **
 **                                                                          **
 \****************************************************************************/

#ifndef _BILLBOARD_SWITCH_NODE_PLUGIN_H
#define _BILLBOARD_SWITCH_NODE_PLUGIN_H

#include <vrml97/vrml/VrmlNodeBillboard.h>
#include <vrml97/vrml/VrmlSFVec3f.h>
#include <vrml97/vrml/VrmlSFInt.h>
#include <vrml97/vrml/VrmlMFFloat.h>
#include <vrml97/vrml/VrmlSFTime.h>
#include <cover/coVRPlugin.h>
#include <vrml97/vrml/VrmlNamespace.h>
#include <iostream>

using namespace vrml;
using namespace opencover;

class PLUGINEXPORT VrmlNodeBillboardSwitch : public VrmlNodeBillboard
{

public:
    // Define the fields of BillboardSwitch nodes
    static void initFields(VrmlNodeBillboardSwitch *node, vrml::VrmlNodeType *t);
    static const char *name(); 

    VrmlNodeBillboardSwitch(VrmlScene *);

    virtual VrmlNodeBillboardSwitch *toBillboardSwitch() const;
    void cloneChildren(VrmlNamespace *);

    virtual bool isModified() const;

    virtual void copyRoutes(VrmlNamespace *ns);

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void render(Viewer *);

    virtual void accumulateTransform(VrmlNode *);
    virtual VrmlNode *getParentTransform();

    const VrmlField *getField(const char *fieldName) const;

    virtual void clearFlags();
    virtual void addToScene(VrmlScene *s, const char *relUrl);

private:
    VrmlSFVec3f d_axisOfRotation;
    VrmlSFInt d_activeChild;
    VrmlNode *d_parentTransform;
    VrmlMFNode d_choice;
    VrmlMFNode d_alternative;
    VrmlMFFloat d_angle;
    //int activeChoice;

protected:
    bool firstTime;
};

class BillboardSwitchPlugin : public coVRPlugin
{
public:
    BillboardSwitchPlugin();
    ~BillboardSwitchPlugin();
    bool init();
};
#endif
