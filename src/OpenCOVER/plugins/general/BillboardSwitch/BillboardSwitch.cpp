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

#include "BillboardSwitch.h"
#include <vrml97/vrml/MathUtils.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <osg/Matrix>
#include "../Vrml97/ViewerOsg.h"
#include <math.h>

static VrmlNode *creator(VrmlScene *s)
{
    return new VrmlNodeBillboardSwitch(s);
}

// Define the built in VrmlNodeType:: "BillboardSwitch" fields

VrmlNodeType *VrmlNodeBillboardSwitch::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("BillboardSwitch", creator);
    }

    VrmlNodeBillboard::defineType(t); // Parent class
    t->addExposedField("choice", VrmlField::MFNODE);
    t->addExposedField("alternative", VrmlField::MFNODE);
    t->addExposedField("axisOfRotation", VrmlField::SFVEC3F);
    t->addEventOut("activeChildChanged", VrmlField::MFINT32);
    t->addField("angle", VrmlField::MFFLOAT);
    return t;
}

VrmlNodeType *VrmlNodeBillboardSwitch::nodeType() const
{
    return defineType(0);
}

VrmlNodeBillboardSwitch::VrmlNodeBillboardSwitch(VrmlScene *scene)
    : VrmlNodeBillboard(scene)
    , d_axisOfRotation(0.0, 1.0, 0.0)
    , d_activeChild(0)
{
    firstTime = true;
    setModified();
}

VrmlNodeBillboardSwitch::~VrmlNodeBillboardSwitch()
{
}

VrmlNode *VrmlNodeBillboardSwitch::cloneMe() const
{
    return new VrmlNodeBillboardSwitch(*this);
}

VrmlNodeBillboardSwitch *VrmlNodeBillboardSwitch::toBillboardSwitch() const
{
    return (VrmlNodeBillboardSwitch *)this;
}

void VrmlNodeBillboardSwitch::cloneChildren(VrmlNamespace *ns)
{
    int n = d_choice.size();
    VrmlNode **kids = d_choice.get();
    for (int i = 0; i < n; ++i)
    {
        if (!kids[i])
            continue;
        VrmlNode *newKid = kids[i]->clone(ns)->reference();
        kids[i]->dereference();
        kids[i] = newKid;
        kids[i]->parentList.push_back(this);
    }
}

bool VrmlNodeBillboardSwitch::isModified() const
{
    if (d_modified)
        return true;

    int w = d_activeChild.get(); ////d_whichChoice.get();

    return (w >= 0 && w < d_choice.size() && d_choice[w]->isModified());
}

void VrmlNodeBillboardSwitch::copyRoutes(VrmlNamespace *ns)
{
    nodeStack.push_front(this);
    VrmlNode::copyRoutes(ns);

    int n = d_choice.size();
    for (int i = 0; i < n; ++i)
        d_choice[i]->copyRoutes(ns);
    nodeStack.pop_front();
}

ostream &VrmlNodeBillboardSwitch::printFields(ostream &os, int indent)
{
    if (!FPZERO(d_axisOfRotation.x()) || !FPZERO(d_axisOfRotation.y()) || !FPZERO(d_axisOfRotation.z()))
        PRINT_FIELD(axisOfRotation);
    if (d_choice.size() > 0)
        PRINT_FIELD(choice);
    if (d_alternative.size() > 0)
        PRINT_FIELD(alternative);
    PRINT_FIELD(activeChild);
    if (d_angle.size() > 0)
        PRINT_FIELD(angle);
    VrmlNodeGroup::printFields(os, indent);
    return os;
}

void VrmlNodeBillboardSwitch::render(Viewer *viewer)
{
    ////DEBUG: fprintf(stderr, "VrmlNodeBillboardSwitch::render\n");
    if (!haveToRender())
        return;

    ViewerOsg *myOsgViewer;
    myOsgViewer = dynamic_cast<ViewerOsg *>(viewer);
    if (myOsgViewer == NULL)
    {
        fprintf(stderr, "ERROR: VrmlNodeBillboardSwitch::render viewer is not of type ViewerOsg\n");
        return;
    }

    myOsgViewer->beginObject(name(), 0, this);
    myOsgViewer->setBillboardTransform(d_axisOfRotation.get());
    osg::Matrix billMat;
    billMat.makeIdentity();
    myOsgViewer->d_currentObject->billBoard->computeLocalToWorldMatrix(billMat, NULL);
    /*fprintf(stderr, "BillMat:\n    %f  %f  %f  %f\n    %f  %f  %f  %f\n    %f  %f  %f  %f\n    %f  %f  %f  %f\n",
               billMat(0,0),billMat(0,1),billMat(0,2),billMat(0,3),
               billMat(1,0),billMat(1,1),billMat(1,2),billMat(1,3),
               billMat(2,0),billMat(2,1),billMat(2,2),billMat(2,3),
               billMat(3,0),billMat(3,1),billMat(3,2),billMat(3,3));
*/
    //calculate angle between 'billboard forward vec and local origin vec'
    //Billobard forward is taken from the third column in the billboard Matrix
    double beta = (acos(billMat(2, 2) / sqrt(billMat(0, 2) * billMat(0, 2) + billMat(1, 2) * billMat(1, 2) + billMat(2, 2) * billMat(2, 2))));
    //check which side of the billboard object is visible:
    //cross product of the vectors to get a normal vector on them:
    double crossX = -billMat(1, 2);
    double crossY = billMat(0, 2);
    double crossZ = 0;
    //check how the cross vector is alligned with the axisOfRotation
    // dot product of cross and axisOfRotation:
    double dotProd = d_axisOfRotation.x() * crossX + d_axisOfRotation.y() * crossY + d_axisOfRotation.z() * crossZ;
    //adjust angle for back-views
    if (dotProd > 0)
        beta = 2 * M_PI - beta;

    //w: which child to render
    int w = 0;
    ////Debug: fprintf(stderr," the W = %d\n",w);
    int angleCount = d_angle.size();
    // No angles given or rotation axis = 0, use proportional partition
    if ((angleCount == 0) || (FPZERO(d_axisOfRotation.x()) && FPZERO(d_axisOfRotation.y()) && FPZERO(d_axisOfRotation.z())))
    {
        w = (int)((beta / M_PI) * (d_choice.size()));
        ////Debug: fprintf(stderr, "VrmlNodeBillboardSwitch::render using angleCount = 0\n");
    }
    else
    {
        for (w = 0; w < angleCount; ++w)
            if (beta < d_angle[w])
                break;
        // Not enough children to chose, take the last one
        if (w >= d_choice.size())
            w = d_choice.size() - 1;
        ////Debug: fprintf(stderr, "VrmlNodeBillboardSwitch::render set choice to %d with angle %f\n",w,beta);
    }
    //send event if another child gets active
    if (d_activeChild.get() != w)
    {
        VrmlSFTime timeNow(System::the->time());
        d_activeChild = w;
        eventOut(timeNow.get(), "activeChildChanged", d_activeChild);
        ////Debug: fprintf(stderr, "VrmlNodeBillboardSwitch::render active Child changed to #%d\n",w);
    }
    //finally draw the right child in billboard mode
    if (w < d_choice.size())
    {
        myOsgViewer->setChoice(w);
        if (w >= 0)
        {
            d_choice[w]->render(viewer);
        }
    }
    myOsgViewer->unsetBillboardTransform(d_axisOfRotation.get());
    myOsgViewer->endObject();
}

// Cache a pointer to (one of the) parent transforms for proper
// rendering of bindables.

void VrmlNodeBillboardSwitch::accumulateTransform(VrmlNode *parent)
{
    d_parentTransform = parent;

    int i, n = d_choice.size();

    for (i = 0; i < n; ++i)
    {
        VrmlNode *kid = d_choice[i];
        kid->accumulateTransform(this);
    }
}

VrmlNode *VrmlNodeBillboardSwitch::getParentTransform()
{
    return d_parentTransform;
}

// Set the value of one of the node fields.
void VrmlNodeBillboardSwitch::setField(const char *fieldName,
                                       const VrmlField &fieldValue)
{
    if
        TRY_FIELD(axisOfRotation, SFVec3f)
    else if
        TRY_FIELD(angle, MFFloat)
    else if
        TRY_FIELD(choice, MFNode)
    else if
        TRY_FIELD(alternative, MFNode)
    else
        VrmlNodeGroup::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeBillboardSwitch::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "axisOfRotation") == 0)
        return &d_axisOfRotation;
    else if (strcmp(fieldName, "activeChildChanged") == 0)
        return &d_activeChild;
    else if (strcmp(fieldName, "choice") == 0)
        return &d_choice;
    else if (strcmp(fieldName, "alternative") == 0)
        return &d_alternative;
    return VrmlNodeGroup::getField(fieldName);
}

void VrmlNodeBillboardSwitch::clearFlags()
{
    VrmlNode::clearFlags();

    int n = d_choice.size();
    for (int i = 0; i < n; ++i)
        d_choice[i]->clearFlags();
}

void VrmlNodeBillboardSwitch::addToScene(VrmlScene *s, const char *rel)
{
    nodeStack.push_front(this);
    d_scene = s;

    int n = d_choice.size();

    for (int i = 0; i < n; ++i)
        d_choice[i]->addToScene(s, rel);
    nodeStack.pop_front();
}

BillboardSwitchPlugin::BillboardSwitchPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "BillboardSwitchPlugin::BillboardSwitchPlugin\n");
    //plugin=this;
}

BillboardSwitchPlugin::~BillboardSwitchPlugin()
{
}

bool
BillboardSwitchPlugin::init()
{
    VrmlNamespace::addBuiltIn(VrmlNodeBillboardSwitch::defineType());

    return true;
}

COVERPLUGIN(BillboardSwitchPlugin)
