/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeTUIElement.h

#ifndef _VrmlNodeTUIElement_
#define _VrmlNodeTUIElement_

#include <vrml97/vrml/VrmlNode.h>

#include <vrml97/vrml/VrmlSFString.h>
#include <vrml97/vrml/VrmlSFVec2f.h>
#include <vrml97/vrml/VrmlSFTime.h>
#include <vrml97/vrml/VrmlSFBool.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlMFFloat.h>
#include <vrml97/vrml/VrmlMFString.h>
#include <vrml97/vrml/VrmlSFInt.h>
#include <vrml97/vrml/VrmlSFVec3f.h>
#include <vrml97/vrml/VrmlSFRotation.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <cover/coTabletUI.h>

namespace vrml
{
class VrmlScene;
}

using namespace vrml;
using namespace opencover;

class VRML97COVEREXPORT VrmlNodeTUIElement : public VrmlNodeChild, public coTUIListener
{

public:
    // Define the fields of TUI nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTUIElement(VrmlScene *);
    virtual ~VrmlNodeTUIElement();

    virtual std::ostream &printFields(std::ostream &os, int indent);

    VrmlNode *cloneMe() const;
    VrmlNodeTUIElement *toTUIElement() const;
    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);
    virtual void render(Viewer *);
    int getID(const char *name);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    virtual const VrmlField *getField(const char *fieldName) const;

protected:
    VrmlSFString d_elementName;
    VrmlSFString d_parent;
    VrmlSFString d_shaderParam;
    VrmlSFVec2f d_pos;
    coTUIElement *d_TUIElement;
};

class VRML97COVEREXPORT VrmlNodeTUIProgressBar : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTUIProgressBar(VrmlScene *);
    virtual ~VrmlNodeTUIProgressBar();

    virtual void render(Viewer *);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    virtual const VrmlField *getField(const char *fieldName) const;

    virtual void tabletPressEvent(coTUIElement *);
    virtual void tabletReleaseEvent(coTUIElement *);

private:
    VrmlSFInt d_max;
    VrmlSFInt d_value;
};
class VRML97COVEREXPORT VrmlNodeTUITabFolder : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTUITabFolder(VrmlScene *);
    virtual ~VrmlNodeTUITabFolder();

    virtual void render(Viewer *);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    virtual const VrmlField *getField(const char *fieldName) const;

    virtual void tabletPressEvent(coTUIElement *);
    virtual void tabletReleaseEvent(coTUIElement *);

private:
    static list<coTUITabFolder *> VrmlTUITabFolders;
};
class VRML97COVEREXPORT VrmlNodeTUITab : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTUITab(VrmlScene *);
    virtual ~VrmlNodeTUITab();

    virtual void render(Viewer *);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    virtual const VrmlField *getField(const char *fieldName) const;

    virtual void tabletPressEvent(coTUIElement *);
    virtual void tabletReleaseEvent(coTUIElement *);

private:
    VrmlSFTime d_touchTime;
    VrmlSFTime d_deactivateTime;
    static list<coTUITab *> VrmlTUITabs;
};

class VRML97COVEREXPORT VrmlNodeTUIButton : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTUIButton(VrmlScene *);
    virtual ~VrmlNodeTUIButton();
    virtual void tabletPressEvent(coTUIElement *);
    virtual void tabletReleaseEvent(coTUIElement *);

    virtual void render(Viewer *);

private:
    VrmlSFTime d_touchTime;
    VrmlSFTime d_releaseTime;
};

class VRML97COVEREXPORT VrmlNodeTUIToggleButton : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTUIToggleButton(VrmlScene *);
    virtual ~VrmlNodeTUIToggleButton();
    virtual void tabletEvent(coTUIElement *);

    virtual void render(Viewer *);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    virtual const VrmlField *getField(const char *fieldName) const;

private:
    VrmlSFBool d_state;
    VrmlSFInt d_choice;
};

class VRML97COVEREXPORT VrmlNodeTUIFrame : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTUIFrame(VrmlScene *);
    virtual ~VrmlNodeTUIFrame();

    virtual void render(Viewer *);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    virtual const VrmlField *getField(const char *fieldName) const;

private:
    VrmlSFInt d_shape;
    VrmlSFInt d_style;
    static list<coTUIFrame *> VrmlTUIFrames;
};

class VRML97COVEREXPORT VrmlNodeTUISplitter : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTUISplitter(VrmlScene *);
    virtual ~VrmlNodeTUISplitter();

    virtual void render(Viewer *);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    virtual const VrmlField *getField(const char *fieldName) const;

private:
    VrmlSFInt d_shape;
    VrmlSFInt d_style;
    VrmlSFInt d_orientation;
};

class VRML97COVEREXPORT VrmlNodeTUISlider : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTUISlider(VrmlScene *);
    virtual ~VrmlNodeTUISlider();
    virtual void tabletEvent(coTUIElement *);

    virtual void render(Viewer *);
    
    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);
    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    virtual const VrmlField *getField(const char *fieldName) const;

private:
    VrmlSFInt d_min;
    VrmlSFInt d_max;
    VrmlSFInt d_value;
    VrmlSFString d_orientation;
};

class VRML97COVEREXPORT VrmlNodeTUIFloatSlider : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTUIFloatSlider(VrmlScene *);
    virtual ~VrmlNodeTUIFloatSlider();
    virtual void tabletEvent(coTUIElement *);

    virtual void render(Viewer *);
    
    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);
    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    virtual const VrmlField *getField(const char *fieldName) const;

private:
    VrmlSFFloat d_min;
    VrmlSFFloat d_max;
    VrmlSFFloat d_value;
    VrmlSFString d_orientation;
};

class VRML97COVEREXPORT VrmlNodeTUIComboBox : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTUIComboBox(VrmlScene *);
    virtual ~VrmlNodeTUIComboBox();
    virtual void tabletEvent(coTUIElement *);

    virtual void render(Viewer *);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    virtual const VrmlField *getField(const char *fieldName) const;

private:
    VrmlMFString d_items;
    VrmlSFBool d_withNone;
    VrmlSFInt d_defaultChoice;
    VrmlSFInt d_choice;
};

class VRML97COVEREXPORT VrmlNodeTUIListBox : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTUIListBox(VrmlScene *);
    virtual ~VrmlNodeTUIListBox();
    virtual void tabletEvent(coTUIElement *);

    virtual void render(Viewer *);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    virtual const VrmlField *getField(const char *fieldName) const;

private:
    VrmlMFString d_items;
    VrmlSFBool d_withNone;
    VrmlSFInt d_defaultChoice;
    VrmlSFInt d_choice;
};

class VRML97COVEREXPORT VrmlNodeTUIMap : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTUIMap(VrmlScene *);
    virtual ~VrmlNodeTUIMap();
    virtual void tabletEvent(coTUIElement *);

    virtual void render(Viewer *);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    virtual const VrmlField *getField(const char *fieldName) const;

private:
    VrmlMFFloat d_ox;
    VrmlMFFloat d_oy;
    VrmlMFFloat d_xSize;
    VrmlMFFloat d_ySize;
    VrmlMFFloat d_height;
    VrmlMFString d_maps;
    VrmlSFInt d_currentMap;
    VrmlSFVec3f d_currentPos;
    VrmlSFRotation d_currentRot;
};

class VRML97COVEREXPORT VrmlNodeTUILabel : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTUILabel(VrmlScene *);
    virtual ~VrmlNodeTUILabel();

    virtual void render(Viewer *);

private:
};
#endif //_VrmlNodeTUIElement_
