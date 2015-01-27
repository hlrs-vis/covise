/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeTUIElement.cpp

#ifdef _WIN32
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <winsock2.h>
#include <windows.h>
#endif

#include <util/unixcompat.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/VrmlScene.h>
#include <vrml97/vrml/Viewer.h>
#include <cover/coVRTui.h>
#include <cover/coVRShader.h>
#include <string>

#include "VrmlNodeTUI.h"

using std::cerr;
using std::endl;

covise::coDLPtrList<coTUIElement *> VrmlTUIElements;
list<coTUITab *> VrmlNodeTUITab::VrmlTUITabs;
list<coTUITabFolder *> VrmlNodeTUITabFolder::VrmlTUITabFolders;
list<coTUIFrame *> VrmlNodeTUIFrame::VrmlTUIFrames;

// Return a new VrmlNodeGroup
static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeTUIElement(s); }

// Define the built in VrmlNodeType:: "TUI" fields

VrmlNodeType *VrmlNodeTUIElement::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("TUIElement", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addExposedField("elementName", VrmlField::SFSTRING);
    t->addExposedField("parent", VrmlField::SFSTRING);
    t->addExposedField("shaderParam", VrmlField::SFSTRING);
    t->addExposedField("pos", VrmlField::SFVEC2F);

    return t;
}

VrmlNodeType *VrmlNodeTUIElement::nodeType() const { return defineType(0); }

VrmlNodeTUIElement::VrmlNodeTUIElement(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_elementName("")
    , d_parent("")
    , d_shaderParam("")
    , d_pos(0, 0)
    , d_TUIElement(NULL)
{
}

VrmlNodeTUIElement::~VrmlNodeTUIElement()
{
    if (VrmlTUIElements.find(d_TUIElement))
        VrmlTUIElements.remove();
    delete d_TUIElement;
}

VrmlNode *VrmlNodeTUIElement::cloneMe() const
{
    return new VrmlNodeTUIElement(*this);
}

void VrmlNodeTUIElement::render(Viewer *)
{
    if (isModified())
    {
        if (d_TUIElement != NULL)
        {
            d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
        }
        clearModified();
    }
    // the elements are created in subclasses
}

VrmlNodeTUIElement *VrmlNodeTUIElement::toTUIElement() const
{
    return (VrmlNodeTUIElement *)this;
}

std::ostream &VrmlNodeTUIElement::printFields(std::ostream &os, int indent)
{
    if (strcmp(d_elementName.get(), "") != 0)
        PRINT_FIELD(elementName);
    if (strcmp(d_parent.get(), "") != 0)
        PRINT_FIELD(parent);
    if (strcmp(d_shaderParam.get(), "") != 0)
        PRINT_FIELD(shaderParam);
    PRINT_FIELD(pos);

    return os;
}

void VrmlNodeTUIElement::eventIn(double timeStamp,
                                 const char *eventName,
                                 const VrmlField *fieldValue)
{
    setModified();
    VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    if ((d_TUIElement != NULL) && strcmp(eventName, "elementName") == 0)
    {
        d_TUIElement->setLabel(d_elementName.get());
    }
}

// Set the value of one of the node fields.

void VrmlNodeTUIElement::setField(const char *fieldName,
                                  const VrmlField &fieldValue)
{
    setModified();
    if
        TRY_FIELD(elementName, SFString)
    else if
        TRY_FIELD(parent, SFString)
    else if
        TRY_FIELD(shaderParam, SFString)
    else if
        TRY_FIELD(pos, SFVec2f)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);

    if ((d_TUIElement != NULL) && strcmp(fieldName, "elementName") == 0)
    {
        d_TUIElement->setLabel(d_elementName.get());
    }
}

const VrmlField *VrmlNodeTUIElement::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "elementName") == 0)
        return &d_elementName;
    else if (strcmp(fieldName, "parent") == 0)
        return &d_parent;
    else if (strcmp(fieldName, "shaderParam") == 0)
        return &d_shaderParam;
    else if (strcmp(fieldName, "pos") == 0)
        return &d_pos;
    else
        cerr << "Node does not have this eventOunt or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}

int VrmlNodeTUIElement::getID(const char *name)
{
    covise::coDLListIter<coTUIElement *> iter;
    iter = VrmlTUIElements.first();
    while (iter)
    {
        if (iter->getName() == name)
        {
            return iter->getID();
        }
        iter++;
    }
    return coVRTui::instance()->mainFolder->getID();
}

//
//
//
// TAB
//
//
//

static VrmlNode *creatorTab(VrmlScene *s) { return new VrmlNodeTUITab(s); }

// Define the built in VrmlNodeType:: "TUI" fields

VrmlNodeType *VrmlNodeTUITab::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("TUITab", creatorTab);
    }

    t->addEventOut("touchTime", VrmlField::SFTIME);
    t->addEventOut("deactivateTime", VrmlField::SFTIME);
    t->addEventIn("activate", VrmlField::SFTIME);
    VrmlNodeTUIElement::defineType(t); // Parent class

    return t;
}

VrmlNodeType *VrmlNodeTUITab::nodeType() const { return defineType(0); }

VrmlNodeTUITab::VrmlNodeTUITab(VrmlScene *scene)
    : VrmlNodeTUIElement(scene)
{
}

VrmlNodeTUITab::~VrmlNodeTUITab()
{

    VrmlTUITabs.remove(static_cast<coTUITab *>(d_TUIElement));
}

void VrmlNodeTUITab::tabletPressEvent(coTUIElement *)
{
    double timeStamp = System::the->time();
    d_touchTime.set(timeStamp);
    eventOut(timeStamp, "touchTime", d_touchTime);
}

void VrmlNodeTUITab::tabletReleaseEvent(coTUIElement *)
{
    double timeStamp = System::the->time();
    d_deactivateTime.set(timeStamp);
    eventOut(timeStamp, "deactivateTime", d_deactivateTime);
}

void VrmlNodeTUITab::render(Viewer *viewer)
{
    if (isModified())
    {
        list<coTUITab *>::iterator tuiListIter;
        int found = -1;
        for (tuiListIter = VrmlTUITabs.begin(); tuiListIter != VrmlTUITabs.end(); tuiListIter++)
            if ((found = strcmp((*tuiListIter)->getName().c_str(), d_elementName.get())) >= 0)
                break;

        if (found != 0)
        {
            if (d_TUIElement == NULL)
            {
                coTUITab *d_TUITab = new coTUITab(d_elementName.get(), getID(d_parent.get()));
                if ((found > 0) || (tuiListIter == VrmlTUITabs.end()))
                    VrmlTUITabs.insert(tuiListIter, d_TUITab);
                else
                    VrmlTUITabs.insert(++tuiListIter, d_TUITab);
                d_TUIElement = (coTUIElement *)d_TUITab;
                d_TUIElement->setEventListener(this);
                VrmlTUIElements.append(d_TUIElement);
                VrmlTUIElements.setNoDelete();
            }
            d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());

            VrmlNodeTUIElement::render(viewer);
        }
        clearModified();
    }
    // the elements are created in subclasses
}

// Set the value of one of the node fields.

void VrmlNodeTUITab::setField(const char *fieldName,
                              const VrmlField &fieldValue)
{

    if (strcmp(fieldName, "activate") == 0 || strcmp(fieldName, "set_activate") == 0)
        ((coTUITab *)d_TUIElement)->setVal(true);
    VrmlNodeTUIElement::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeTUITab::getField(const char *fieldName) const
{

    return VrmlNodeTUIElement::getField(fieldName);
}

//
//
//
// ProgressBar
//
//
//

static VrmlNode *creatorProgressBar(VrmlScene *s) { return new VrmlNodeTUIProgressBar(s); }

// Define the built in VrmlNodeType:: "TUI" fields

VrmlNodeType *VrmlNodeTUIProgressBar::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("TUIProgressBar", creatorProgressBar);
    }

    t->addExposedField("max", VrmlField::SFINT32);
    t->addExposedField("value", VrmlField::SFINT32);
    VrmlNodeTUIElement::defineType(t); // Parent class

    return t;
}

VrmlNodeType *VrmlNodeTUIProgressBar::nodeType() const { return defineType(0); }

VrmlNodeTUIProgressBar::VrmlNodeTUIProgressBar(VrmlScene *scene)
    : VrmlNodeTUIElement(scene)
    , d_max(100)
    , d_value(0)
{
}

VrmlNodeTUIProgressBar::~VrmlNodeTUIProgressBar()
{
}

void VrmlNodeTUIProgressBar::tabletPressEvent(coTUIElement *)
{
}

void VrmlNodeTUIProgressBar::tabletReleaseEvent(coTUIElement *)
{
}

void VrmlNodeTUIProgressBar::render(Viewer *viewer)
{
    if (isModified())
    {
        if (d_TUIElement == NULL)
        {
            d_TUIElement = (coTUIElement *)new coTUIProgressBar(d_elementName.get(), getID(d_parent.get()));
            d_TUIElement->setEventListener(this);
            VrmlTUIElements.append(d_TUIElement);
            VrmlTUIElements.setNoDelete();
        }
        coTUIProgressBar *pb = (coTUIProgressBar *)d_TUIElement;
        pb->setMax(d_max.get());
        pb->setValue(d_value.get());
        d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
        VrmlNodeTUIElement::render(viewer);
        clearModified();
    }
    // the elements are created in subclasses
}

void VrmlNodeTUIProgressBar::setField(const char *fieldName,
                                      const VrmlField &fieldValue)
{

    setModified();
    if
        TRY_FIELD(max, SFInt)
    else if
        TRY_FIELD(value, SFInt)
    else
        VrmlNodeTUIElement::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeTUIProgressBar::getField(const char *fieldName) const
{

    return VrmlNodeTUIElement::getField(fieldName);
}

//
//
//
// TABFolder
//
//
//

static VrmlNode *creatorTabFolder(VrmlScene *s) { return new VrmlNodeTUITabFolder(s); }

// Define the built in VrmlNodeType:: "TUI" fields

VrmlNodeType *VrmlNodeTUITabFolder::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("TUITabFolder", creatorTabFolder);
    }

    VrmlNodeTUIElement::defineType(t); // Parent class

    return t;
}

VrmlNodeType *VrmlNodeTUITabFolder::nodeType() const { return defineType(0); }

VrmlNodeTUITabFolder::VrmlNodeTUITabFolder(VrmlScene *scene)
    : VrmlNodeTUIElement(scene)
{
}

VrmlNodeTUITabFolder::~VrmlNodeTUITabFolder()
{
    VrmlTUITabFolders.remove(static_cast<coTUITabFolder *>(d_TUIElement));
}

void VrmlNodeTUITabFolder::tabletPressEvent(coTUIElement *)
{
}

void VrmlNodeTUITabFolder::tabletReleaseEvent(coTUIElement *)
{
}

void VrmlNodeTUITabFolder::render(Viewer *viewer)
{
    if (isModified())
    {
        list<coTUITabFolder *>::iterator tuiListIter;
        int found = -1;
        for (tuiListIter = VrmlTUITabFolders.begin(); tuiListIter != VrmlTUITabFolders.end(); tuiListIter++)
            if ((found = strcmp((*tuiListIter)->getName().c_str(), d_elementName.get())) >= 0)
                break;

        if (found != 0)
        {
            if (d_TUIElement == NULL)
            {
                coTUITabFolder *d_TUITabFolder = new coTUITabFolder(d_elementName.get(), getID(d_parent.get()));
                if ((found > 0) || (tuiListIter == VrmlTUITabFolders.end()))
                    VrmlTUITabFolders.insert(tuiListIter, d_TUITabFolder);
                else
                    VrmlTUITabFolders.insert(++tuiListIter, d_TUITabFolder);
                d_TUIElement = (coTUIElement *)d_TUITabFolder;

                d_TUIElement->setEventListener(this);
                VrmlTUIElements.append(d_TUIElement);
                VrmlTUIElements.setNoDelete();
            }
            d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
            VrmlNodeTUIElement::render(viewer);
        }
        clearModified();
    }
    // the elements are created in subclasses
}

// Set the value of one of the node fields.

void VrmlNodeTUITabFolder::setField(const char *fieldName,
                                    const VrmlField &fieldValue)
{

    VrmlNodeTUIElement::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeTUITabFolder::getField(const char *fieldName) const
{

    return VrmlNodeTUIElement::getField(fieldName);
}

//
//
//
// BUTTON
//
//
//

// Return a new VrmlNodeGroup
static VrmlNode *creatorButton(VrmlScene *s) { return new VrmlNodeTUIButton(s); }

// Define the built in VrmlNodeType:: "TUI" fields

VrmlNodeType *VrmlNodeTUIButton::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("TUIButton", creatorButton);
    }

    t->addEventOut("touchTime", VrmlField::SFTIME);
    t->addEventOut("releaseTime", VrmlField::SFTIME);
    VrmlNodeTUIElement::defineType(t); // Parent class

    return t;
}

VrmlNodeType *VrmlNodeTUIButton::nodeType() const { return defineType(0); }

VrmlNodeTUIButton::VrmlNodeTUIButton(VrmlScene *scene)
    : VrmlNodeTUIElement(scene)
    , d_touchTime(0.0)
{
}

VrmlNodeTUIButton::~VrmlNodeTUIButton()
{
}

void VrmlNodeTUIButton::tabletPressEvent(coTUIElement *)
{
    double timeStamp = System::the->time();
    d_touchTime.set(timeStamp);
    eventOut(timeStamp, "touchTime", d_touchTime);
}

void VrmlNodeTUIButton::tabletReleaseEvent(coTUIElement *)
{
    double timeStamp = System::the->time();
    d_releaseTime.set(timeStamp);
    eventOut(timeStamp, "releaseTime", d_releaseTime);
}

void VrmlNodeTUIButton::render(Viewer *viewer)
{
    if (isModified())
    {
        if (d_TUIElement == NULL)
        {
            d_TUIElement = (coTUIElement *)new coTUIButton(d_elementName.get(), getID(d_parent.get()));
            d_TUIElement->setEventListener(this);
            VrmlTUIElements.append(d_TUIElement);
            VrmlTUIElements.setNoDelete();
        }
        d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
        VrmlNodeTUIElement::render(viewer);
        clearModified();
    }
    // the elements are created in subclasses
}

//
//
//
// TOGGLE_BUTTON
//
//
//

// Return a new VrmlNodeGroup
static VrmlNode *creatorToggleButton(VrmlScene *s) { return new VrmlNodeTUIToggleButton(s); }

// Define the built in VrmlNodeType:: "TUI" fields

VrmlNodeType *VrmlNodeTUIToggleButton::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("TUIToggleButton", creatorToggleButton);
    }

    t->addExposedField("choice", VrmlField::SFINT32);
    t->addExposedField("state", VrmlField::SFBOOL);
    VrmlNodeTUIElement::defineType(t); // Parent class

    return t;
}

// Set the value of one of the node fields.

void VrmlNodeTUIToggleButton::setField(const char *fieldName,
                                       const VrmlField &fieldValue)
{
    setModified();
    if
        TRY_FIELD(state, SFBool)
    else if
        TRY_FIELD(choice, SFInt)
    else
        VrmlNodeTUIElement::setField(fieldName, fieldValue);

    if (d_TUIElement)
    {
        coTUIToggleButton *tb = (coTUIToggleButton *)d_TUIElement;
        if (strcmp(fieldName, "state") == 0 && fieldValue.toSFBool() != NULL)
            tb->setState(fieldValue.toSFBool()->get());
        else if (strcmp(fieldName, "choice") == 0 && fieldValue.toSFInt() != NULL)
        {
            d_state.set(fieldValue.toSFInt()->get() >= 0);
            tb->setState(fieldValue.toSFInt()->get() >= 0);
        }
    }
}

const VrmlField *VrmlNodeTUIToggleButton::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "state") == 0)
        return &d_state;
    return VrmlNodeTUIElement::getField(fieldName);
    return 0;
}

VrmlNodeType *VrmlNodeTUIToggleButton::nodeType() const { return defineType(0); }

VrmlNodeTUIToggleButton::VrmlNodeTUIToggleButton(VrmlScene *scene)
    : VrmlNodeTUIElement(scene)
    , d_state(false)
    , d_choice(-1)
{
}

VrmlNodeTUIToggleButton::~VrmlNodeTUIToggleButton()
{
}

void VrmlNodeTUIToggleButton::tabletEvent(coTUIElement *)
{
    double timeStamp = System::the->time();
    coTUIToggleButton *tb = (coTUIToggleButton *)d_TUIElement;
    d_state.set(tb->getState());
    eventOut(timeStamp, "state", d_state);
    if (tb->getState())
        d_choice.set(0);
    else
        d_choice.set(-1);
    eventOut(timeStamp, "choice", d_choice);
}

void VrmlNodeTUIToggleButton::render(Viewer *viewer)
{
    if (isModified())
    {
        if (d_TUIElement == NULL)
        {
            d_TUIElement = (coTUIElement *)new coTUIToggleButton(d_elementName.get(), getID(d_parent.get()));
            d_TUIElement->setEventListener(this);
            VrmlTUIElements.append(d_TUIElement);
            VrmlTUIElements.setNoDelete();
        }
        d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
        VrmlNodeTUIElement::render(viewer);
        coTUIToggleButton *tb = (coTUIToggleButton *)d_TUIElement;
        tb->setState(d_state.get());

        double timeStamp = System::the->time();
        // create eventOuts for default values
        d_state.set(tb->getState());
        eventOut(timeStamp, "state", d_state);
        if (tb->getState())
            d_choice.set(0);
        else
            d_choice.set(-1);
        eventOut(timeStamp, "choice", d_choice);
        clearModified();
    }
    // the elements are created in subclasses
}

//
//
//
// FRAME
//
//
//

static VrmlNode *creatorFrame(VrmlScene *s) { return new VrmlNodeTUIFrame(s); }

// Define the built in VrmlNodeType:: "TUI" fields

VrmlNodeType *VrmlNodeTUIFrame::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("TUIFrame", creatorFrame);
    }

    t->addExposedField("shape", VrmlField::SFINT32);
    t->addExposedField("style", VrmlField::SFINT32);
    VrmlNodeTUIElement::defineType(t); // Parent class

    return t;
}

VrmlNodeType *VrmlNodeTUIFrame::nodeType() const { return defineType(0); }

VrmlNodeTUIFrame::VrmlNodeTUIFrame(VrmlScene *scene)
    : VrmlNodeTUIElement(scene)
    , d_shape(0)
    , d_style(0)
{
}

VrmlNodeTUIFrame::~VrmlNodeTUIFrame()
{
    VrmlTUIFrames.remove(static_cast<coTUIFrame *>(d_TUIElement));
}

void VrmlNodeTUIFrame::render(Viewer *viewer)
{
    if (isModified())
    {
        list<coTUIFrame *>::iterator tuiListIter;
        int found = -1;
        for (tuiListIter = VrmlTUIFrames.begin(); tuiListIter != VrmlTUIFrames.end(); tuiListIter++)
            if ((found = strcmp((*tuiListIter)->getName().c_str(), d_elementName.get())) >= 0)
                break;

        if (found != 0)
        {
            if (d_TUIElement == NULL)
            {
                coTUIFrame *coFrame = new coTUIFrame(d_elementName.get(), getID(d_parent.get()));
                coFrame->setShape(d_shape.get());
                coFrame->setStyle(d_style.get());
                if ((found > 0) || (tuiListIter == VrmlTUIFrames.end()))
                    VrmlTUIFrames.insert(tuiListIter, coFrame);
                else
                    VrmlTUIFrames.insert(++tuiListIter, coFrame);
                d_TUIElement = (coTUIElement *)coFrame;
                VrmlTUIElements.append(d_TUIElement);
                VrmlTUIElements.setNoDelete();
            }
            d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
            VrmlNodeTUIElement::render(viewer);
        }
        clearModified();
    }
    // the elements are created in subclasses
}

// Set the value of one of the node fields.

void VrmlNodeTUIFrame::setField(const char *fieldName,
                                const VrmlField &fieldValue)
{
    setModified();
    if
        TRY_FIELD(shape, SFInt)
    else if
        TRY_FIELD(style, SFInt)
    else
        VrmlNodeTUIElement::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeTUIFrame::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "shape") == 0)
        return &d_shape;
    else if (strcmp(fieldName, "style") == 0)
        return &d_style;
    else
        return VrmlNodeTUIElement::getField(fieldName);
    return 0;
}

//
//
//
// Splitter
//
//
//

static VrmlNode *creatorSplitter(VrmlScene *s) { return new VrmlNodeTUISplitter(s); }

// Define the built in VrmlNodeType:: "TUI" fields

VrmlNodeType *VrmlNodeTUISplitter::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("TUISplitter", creatorSplitter);
    }

    t->addExposedField("shape", VrmlField::SFINT32);
    t->addExposedField("style", VrmlField::SFINT32);
    t->addExposedField("orientation", VrmlField::SFINT32);
    VrmlNodeTUIElement::defineType(t); // Parent class

    return t;
}

VrmlNodeType *VrmlNodeTUISplitter::nodeType() const { return defineType(0); }

VrmlNodeTUISplitter::VrmlNodeTUISplitter(VrmlScene *scene)
    : VrmlNodeTUIElement(scene)
    , d_shape(0)
    , d_style(0)
    , d_orientation(0x1)
{
}

VrmlNodeTUISplitter::~VrmlNodeTUISplitter()
{
}

void VrmlNodeTUISplitter::render(Viewer *viewer)
{
    if (isModified())
    {
        if (d_TUIElement == NULL)
        {
            coTUISplitter *coSplit = new coTUISplitter(d_elementName.get(), getID(d_parent.get()));
            coSplit->setShape(d_shape.get());
            coSplit->setStyle(d_style.get());
            coSplit->setOrientation(d_orientation.get());
            d_TUIElement = (coTUIElement *)coSplit;
            VrmlTUIElements.append(d_TUIElement);
            VrmlTUIElements.setNoDelete();
        }
        d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
        VrmlNodeTUIElement::render(viewer);
        clearModified();
    }
    // the elements are created in subclasses
}

// Set the value of one of the node fields.

void VrmlNodeTUISplitter::setField(const char *fieldName,
                                   const VrmlField &fieldValue)
{
    setModified();
    if
        TRY_FIELD(shape, SFInt)
    else if
        TRY_FIELD(style, SFInt)
    else if
        TRY_FIELD(orientation, SFInt)
    else
        VrmlNodeTUIElement::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeTUISplitter::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "shape") == 0)
        return &d_shape;
    else if (strcmp(fieldName, "style") == 0)
        return &d_style;
    else if (strcmp(fieldName, "orientation") == 0)
        return &d_orientation;
    else
        return VrmlNodeTUIElement::getField(fieldName);
    return 0;
}

//
//
//
// LABEL
//
//
//

static VrmlNode *creatorLabel(VrmlScene *s) { return new VrmlNodeTUILabel(s); }

// Define the built in VrmlNodeType:: "TUI" fields

VrmlNodeType *VrmlNodeTUILabel::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("TUILabel", creatorLabel);
    }

    VrmlNodeTUIElement::defineType(t); // Parent class

    return t;
}

VrmlNodeType *VrmlNodeTUILabel::nodeType() const { return defineType(0); }

VrmlNodeTUILabel::VrmlNodeTUILabel(VrmlScene *scene)
    : VrmlNodeTUIElement(scene)
{
}

VrmlNodeTUILabel::~VrmlNodeTUILabel()
{
}

void VrmlNodeTUILabel::render(Viewer *viewer)
{
    if (isModified())
    {
        if (d_TUIElement == NULL)
        {
            d_TUIElement = (coTUIElement *)new coTUILabel(d_elementName.get(), getID(d_parent.get()));
            VrmlTUIElements.append(d_TUIElement);
            VrmlTUIElements.setNoDelete();
        }
        d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
        VrmlNodeTUIElement::render(viewer);
        clearModified();
    }
    // the elements are created in subclasses
}

//
//
//
// FloatSlider
//
//
//

// Return a new VrmlNodeGroup
static VrmlNode *creatorFloatSlider(VrmlScene *s) { return new VrmlNodeTUIFloatSlider(s); }

// Define the built in VrmlNodeType:: "TUI" fields

VrmlNodeType *VrmlNodeTUIFloatSlider::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("TUIFloatSlider", creatorFloatSlider);
    }

    t->addExposedField("min", VrmlField::SFFLOAT);
    t->addExposedField("max", VrmlField::SFFLOAT);
    t->addExposedField("value", VrmlField::SFFLOAT);
    t->addExposedField("orientation", VrmlField::SFSTRING);
    VrmlNodeTUIElement::defineType(t); // Parent class

    return t;
}

VrmlNodeType *VrmlNodeTUIFloatSlider::nodeType() const { return defineType(0); }

VrmlNodeTUIFloatSlider::VrmlNodeTUIFloatSlider(VrmlScene *scene)
    : VrmlNodeTUIElement(scene)
    , d_min(0.0)
    , d_max(100.0)
    , d_value(50.0)
    , d_orientation("horizontal")
{
}

VrmlNodeTUIFloatSlider::~VrmlNodeTUIFloatSlider()
{
}

void VrmlNodeTUIFloatSlider::tabletEvent(coTUIElement *)
{
    double timeStamp = System::the->time();
    coTUIFloatSlider *sl = (coTUIFloatSlider *)d_TUIElement;
    d_value.set(sl->getValue());
    eventOut(timeStamp, "value", d_value);
    if (strlen(d_shaderParam.get()) > 0)
    {
        std::string shaderName;
        std::string param;
        shaderName = d_shaderParam.get();
        size_t pos = shaderName.find_last_of('.');
        param = shaderName.substr(pos + 1, std::string::npos);
        shaderName = shaderName.substr(0, pos);
        coVRShader *shader = coVRShaderList::instance()->get(shaderName);
        if (shader != NULL)
        {
            osg::Uniform *uniform = shader->getUniform(param);
            if (uniform != NULL)
            {
                uniform->set(sl->getValue());
            }
        }
    }
}

void VrmlNodeTUIFloatSlider::render(Viewer *viewer)
{
    if (isModified())
    {
        coTUIFloatSlider *fs;
        if (d_TUIElement == NULL)
        {
            fs = new coTUIFloatSlider(d_elementName.get(), getID(d_parent.get()));
            d_TUIElement = (coTUIElement *)fs;

            d_TUIElement->setEventListener(this);
            VrmlTUIElements.append(d_TUIElement);
            VrmlTUIElements.setNoDelete();
        }
        fs = (coTUIFloatSlider *)d_TUIElement;
        fs->setMin(d_min.get());
        fs->setMax(d_max.get());
        fs->setValue(d_value.get());

        bool ori = false;
        if (strcasecmp(d_orientation.get(), "horizontal") == 0)
            ori = true;
        fs->setOrientation(ori);
        d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
        VrmlNodeTUIElement::render(viewer);
        clearModified();
    }
    // the elements are created in subclasses
}

// Set the value of one of the node fields.

void VrmlNodeTUIFloatSlider::setField(const char *fieldName,
                                      const VrmlField &fieldValue)
{
    setModified();
    if
        TRY_FIELD(min, SFFloat)
    else if
        TRY_FIELD(max, SFFloat)
    else if
        TRY_FIELD(value, SFFloat)
    else if
        TRY_FIELD(orientation, SFString)
    else
        VrmlNodeTUIElement::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeTUIFloatSlider::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "min") == 0)
        return &d_min;
    else if (strcmp(fieldName, "max") == 0)
        return &d_max;
    else if (strcmp(fieldName, "value") == 0)
        return &d_value;
    else if (strcmp(fieldName, "orientation") == 0)
        return &d_orientation;
    else
        return VrmlNodeTUIElement::getField(fieldName);
    return 0;
}

//
//
//
// Slider
//
//
//

// Return a new VrmlNodeGroup
static VrmlNode *creatorSlider(VrmlScene *s) { return new VrmlNodeTUISlider(s); }

// Define the built in VrmlNodeType:: "TUI" fields

VrmlNodeType *VrmlNodeTUISlider::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("TUISlider", creatorSlider);
    }

    t->addExposedField("min", VrmlField::SFINT32);
    t->addExposedField("max", VrmlField::SFINT32);
    t->addExposedField("value", VrmlField::SFINT32);
    t->addExposedField("orientation", VrmlField::SFSTRING);
    VrmlNodeTUIElement::defineType(t); // Parent class

    return t;
}

VrmlNodeType *VrmlNodeTUISlider::nodeType() const { return defineType(0); }

VrmlNodeTUISlider::VrmlNodeTUISlider(VrmlScene *scene)
    : VrmlNodeTUIElement(scene)
    , d_min(0)
    , d_max(100)
    , d_value(50)
    , d_orientation("horizontal")
{
}

VrmlNodeTUISlider::~VrmlNodeTUISlider()
{
}

void VrmlNodeTUISlider::tabletEvent(coTUIElement *)
{
    double timeStamp = System::the->time();
    coTUISlider *sl = (coTUISlider *)d_TUIElement;
    d_value.set(sl->getValue());
    eventOut(timeStamp, "value", d_value);
    if (strlen(d_shaderParam.get()) > 0)
    {
        std::string shaderName;
        std::string param;
        shaderName = d_shaderParam.get();
        size_t pos = shaderName.find_last_of('.');
        param = shaderName.substr(pos, std::string::npos);
        shaderName = shaderName.substr(0, pos);
        coVRShader *shader = coVRShaderList::instance()->get(shaderName);
        if (shader != NULL)
        {
            osg::Uniform *uniform = shader->getUniform(param);
            if (uniform != NULL)
            {
                uniform->set(sl->getValue());
            }
        }
    }
}

void VrmlNodeTUISlider::render(Viewer *viewer)
{
    if (isModified())
    {
        if (d_TUIElement == NULL)
        {
            d_TUIElement = (coTUIElement *)new coTUISlider(d_elementName.get(), getID(d_parent.get()));
            d_TUIElement->setEventListener(this);
            VrmlTUIElements.append(d_TUIElement);
            VrmlTUIElements.setNoDelete();
        }
        coTUISlider *ts = (coTUISlider *)d_TUIElement;
        ts->setMin(d_min.get());
        ts->setMax(d_max.get());
        ts->setValue(d_value.get());
        d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());

        bool ori = false;
        if (strcasecmp(d_orientation.get(), "horizontal") == 0)
            ori = true;
        ts->setOrientation(ori);
        VrmlNodeTUIElement::render(viewer);
        clearModified();
    }
    // the elements are created in subclasses
}

// Set the value of one of the node fields.

void VrmlNodeTUISlider::setField(const char *fieldName,
                                 const VrmlField &fieldValue)
{
    setModified();
    if
        TRY_FIELD(min, SFInt)
    else if
        TRY_FIELD(max, SFInt)
    else if
        TRY_FIELD(value, SFInt)
    else if
        TRY_FIELD(orientation, SFString)
    else
        VrmlNodeTUIElement::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeTUISlider::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "min") == 0)
        return &d_min;
    else if (strcmp(fieldName, "max") == 0)
        return &d_max;
    else if (strcmp(fieldName, "value") == 0)
        return &d_value;
    else if (strcmp(fieldName, "orientation") == 0)
        return &d_orientation;
    else
        return VrmlNodeTUIElement::getField(fieldName);
    return 0;
}

//
//
//
// ComboBox
//
//
//

// Return a new VrmlNodeGroup
static VrmlNode *creatorComboBox(VrmlScene *s) { return new VrmlNodeTUIComboBox(s); }

// Define the built in VrmlNodeType:: "TUI" fields

VrmlNodeType *VrmlNodeTUIComboBox::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("TUIComboBox", creatorComboBox);
    }

    t->addExposedField("items", VrmlField::MFSTRING);
    t->addExposedField("withNone", VrmlField::SFBOOL);
    t->addExposedField("defaultChoice", VrmlField::SFINT32);
    t->addExposedField("choice", VrmlField::SFINT32);
    VrmlNodeTUIElement::defineType(t); // Parent class

    return t;
}

VrmlNodeType *VrmlNodeTUIComboBox::nodeType() const { return defineType(0); }

VrmlNodeTUIComboBox::VrmlNodeTUIComboBox(VrmlScene *scene)
    : VrmlNodeTUIElement(scene)
    , d_withNone(true)
    , d_defaultChoice(0)
{
}

VrmlNodeTUIComboBox::~VrmlNodeTUIComboBox()
{
}

void VrmlNodeTUIComboBox::tabletEvent(coTUIElement *)
{
    double timeStamp = System::the->time();
    coTUIComboBox *cb = (coTUIComboBox *)d_TUIElement;
    if (d_withNone.get())
        d_choice.set(cb->getSelectedEntry() - 1);
    else
        d_choice.set(cb->getSelectedEntry());
    eventOut(timeStamp, "choice", d_choice);
}

void VrmlNodeTUIComboBox::render(Viewer *viewer)
{
    if (isModified())
    {
        if (d_TUIElement == NULL)
        {
            coTUIComboBox *cb = new coTUIComboBox(d_elementName.get(), getID(d_parent.get()));
            int i;
            for (i = 0; i < d_items.size(); i++)
                cb->addEntry(d_items.get(i));
            if (d_withNone.get())
                cb->setSelectedEntry(d_defaultChoice.get() + 1);
            else
                cb->setSelectedEntry(d_defaultChoice.get());
            d_TUIElement = (coTUIElement *)cb;
            d_TUIElement->setEventListener(this);
            VrmlTUIElements.append(d_TUIElement);
            VrmlTUIElements.setNoDelete();
        }
        d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
        VrmlNodeTUIElement::render(viewer);
        clearModified();
    }
    // the elements are created in subclasses
}

// Set the value of one of the node fields.

void VrmlNodeTUIComboBox::setField(const char *fieldName,
                                   const VrmlField &fieldValue)
{
    setModified();
    if
        TRY_FIELD(items, MFString)
    else if
        TRY_FIELD(withNone, SFBool)
    else if
        TRY_FIELD(defaultChoice, SFInt)
    else if
        TRY_FIELD(choice, SFInt)
    else
        VrmlNodeTUIElement::setField(fieldName, fieldValue);

    coTUIComboBox *cb = (coTUIComboBox *)d_TUIElement;
    if (d_TUIElement)
    {
        if (strcmp(fieldName, "choice") == 0 && fieldValue.toSFInt() != NULL)
        {
            if (d_withNone.get())
                cb->setSelectedEntry(fieldValue.toSFInt()->get() + 1);
            else
                cb->setSelectedEntry(fieldValue.toSFInt()->get());
        }
    }
}

const VrmlField *VrmlNodeTUIComboBox::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "items") == 0)
        return &d_items;
    else if (strcmp(fieldName, "withNone") == 0)
        return &d_withNone;
    else if (strcmp(fieldName, "defaultChoice") == 0)
        return &d_defaultChoice;
    else
        return VrmlNodeTUIElement::getField(fieldName);
    return 0;
}

//
//
//
// ListBox
//
//
//

// Return a new VrmlNodeGroup
static VrmlNode *creatorListBox(VrmlScene *s) { return new VrmlNodeTUIListBox(s); }

// Define the built in VrmlNodeType:: "TUI" fields

VrmlNodeType *VrmlNodeTUIListBox::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("TUIListBox", creatorListBox);
    }

    t->addExposedField("items", VrmlField::MFSTRING);
    t->addExposedField("withNone", VrmlField::SFBOOL);
    t->addExposedField("defaultChoice", VrmlField::SFINT32);
    t->addEventOut("choice", VrmlField::SFINT32);
    VrmlNodeTUIElement::defineType(t); // Parent class

    return t;
}

VrmlNodeType *VrmlNodeTUIListBox::nodeType() const { return defineType(0); }

VrmlNodeTUIListBox::VrmlNodeTUIListBox(VrmlScene *scene)
    : VrmlNodeTUIElement(scene)
    , d_withNone(true)
    , d_defaultChoice(0)
{
}

VrmlNodeTUIListBox::~VrmlNodeTUIListBox()
{
}

void VrmlNodeTUIListBox::tabletEvent(coTUIElement *)
{
    double timeStamp = System::the->time();
    coTUIListBox *cb = (coTUIListBox *)d_TUIElement;
    if (d_withNone.get())
        d_choice.set(cb->getSelectedEntry() - 1);
    else
        d_choice.set(cb->getSelectedEntry());
    eventOut(timeStamp, "choice", d_choice);
}

void VrmlNodeTUIListBox::render(Viewer *viewer)
{
    if (isModified())
    {
        if (d_TUIElement == NULL)
        {
            coTUIListBox *cb = new coTUIListBox(d_elementName.get(), getID(d_parent.get()));
            int i;
            for (i = 0; i < d_items.size(); i++)
                cb->addEntry(d_items.get(i));
            if (d_withNone.get())
                cb->setSelectedEntry(d_defaultChoice.get() + 1);
            else
                cb->setSelectedEntry(d_defaultChoice.get());
            d_TUIElement = (coTUIElement *)cb;
            d_TUIElement->setEventListener(this);
            VrmlTUIElements.append(d_TUIElement);
            VrmlTUIElements.setNoDelete();
        }
        d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
        VrmlNodeTUIElement::render(viewer);
        clearModified();
    }
    // the elements are created in subclasses
}

// Set the value of one of the node fields.

void VrmlNodeTUIListBox::setField(const char *fieldName,
                                  const VrmlField &fieldValue)
{
    setModified();
    if
        TRY_FIELD(items, MFString)
    else if
        TRY_FIELD(withNone, SFBool)
    else if
        TRY_FIELD(defaultChoice, SFInt)
    else
        VrmlNodeTUIElement::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeTUIListBox::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "items") == 0)
        return &d_items;
    else if (strcmp(fieldName, "withNone") == 0)
        return &d_withNone;
    else if (strcmp(fieldName, "defaultChoice") == 0)
        return &d_defaultChoice;
    else
        return VrmlNodeTUIElement::getField(fieldName);
    return 0;
}

//
//
//
// Map
//
//
//

// Return a new VrmlNodeGroup
static VrmlNode *creatorMap(VrmlScene *s) { return new VrmlNodeTUIMap(s); }

// Define the built in VrmlNodeType:: "TUI" fields

VrmlNodeType *VrmlNodeTUIMap::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("TUIMap", creatorMap);
    }

    t->addExposedField("ox", VrmlField::MFSTRING);
    t->addExposedField("oy", VrmlField::MFFLOAT);
    t->addExposedField("xSize", VrmlField::MFFLOAT);
    t->addExposedField("ySize", VrmlField::MFFLOAT);
    t->addExposedField("height", VrmlField::MFFLOAT);
    t->addExposedField("maps", VrmlField::MFSTRING);
    t->addEventOut("currentMap", VrmlField::SFINT32);
    t->addEventOut("currentPos", VrmlField::SFVEC3F);
    ;
    t->addEventOut("currentRot", VrmlField::SFROTATION);
    ;
    VrmlNodeTUIElement::defineType(t); // Parent class

    return t;
}

VrmlNodeType *VrmlNodeTUIMap::nodeType() const { return defineType(0); }

VrmlNodeTUIMap::VrmlNodeTUIMap(VrmlScene *scene)
    : VrmlNodeTUIElement(scene)
    , d_currentMap(-1)
    , d_currentPos(0, 0, 0)
{
}

VrmlNodeTUIMap::~VrmlNodeTUIMap()
{
}

void VrmlNodeTUIMap::tabletEvent(coTUIElement *)
{
    double timeStamp = System::the->time();
    coTUIMap *cb = (coTUIMap *)d_TUIElement;
    d_currentMap.set(cb->mapNum);
    d_currentPos.set(cb->xPos, cb->yPos, cb->height);
    d_currentRot.set(0, 1, 0, cb->angle);
    eventOut(timeStamp, "currentMap", d_currentMap);
    eventOut(timeStamp, "currentPos", d_currentPos);
    eventOut(timeStamp, "currentRot", d_currentRot);
}

void VrmlNodeTUIMap::render(Viewer *viewer)
{
    if (isModified())
    {
        if (d_TUIElement == NULL)
        {
            coTUIMap *cb = new coTUIMap(d_elementName.get(), getID(d_parent.get()));
            int i;
            for (i = 0; i < d_maps.size(); i++)
                cb->addMap(d_maps.get(i), d_ox[i], d_oy[i], d_xSize[i], d_ySize[i], d_height[i]);
            d_TUIElement = (coTUIElement *)cb;
            d_TUIElement->setEventListener(this);
            VrmlTUIElements.append(d_TUIElement);
            VrmlTUIElements.setNoDelete();
        }
        d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
        VrmlNodeTUIElement::render(viewer);
        clearModified();
    }
    // the elements are created in subclasses
}

// Set the value of one of the node fields.

void VrmlNodeTUIMap::setField(const char *fieldName,
                              const VrmlField &fieldValue)
{
    setModified();
    if
        TRY_FIELD(maps, MFString)
    else if
        TRY_FIELD(ox, MFFloat)
    else if
        TRY_FIELD(oy, MFFloat)
    else if
        TRY_FIELD(xSize, MFFloat)
    else if
        TRY_FIELD(ySize, MFFloat)
    else if
        TRY_FIELD(height, MFFloat)
    else
        VrmlNodeTUIElement::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeTUIMap::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "maps") == 0)
        return &d_maps;
    else if (strcmp(fieldName, "ox") == 0)
        return &d_ox;
    else if (strcmp(fieldName, "xSize") == 0)
        return &d_xSize;
    else if (strcmp(fieldName, "ySize") == 0)
        return &d_ySize;
    else if (strcmp(fieldName, "height") == 0)
        return &d_height;
    else
        return VrmlNodeTUIElement::getField(fieldName);
    return 0;
}
