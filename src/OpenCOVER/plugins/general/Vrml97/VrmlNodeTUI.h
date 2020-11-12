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
#include <vrb/client/SharedState.h>

#include <memory>

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
    VrmlNodeTUIElement(const VrmlNodeTUIElement&);
    virtual ~VrmlNodeTUIElement();

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual VrmlNode *cloneMe() const;
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
	VrmlSFBool d_shared;
    coTUIElement *d_TUIElement;
};

class VRML97COVEREXPORT VrmlNodeTUIProgressBar : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTUIProgressBar(VrmlScene *);
    VrmlNodeTUIProgressBar(const VrmlNodeTUIProgressBar&);
    virtual ~VrmlNodeTUIProgressBar();

    virtual VrmlNode* cloneMe() const;

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
    VrmlNodeTUITabFolder(const VrmlNodeTUITabFolder&);
    virtual ~VrmlNodeTUITabFolder();
    virtual VrmlNode* cloneMe() const;

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
    VrmlNodeTUITab(const VrmlNodeTUITab&);
    virtual ~VrmlNodeTUITab();
    virtual VrmlNode* cloneMe() const;

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
    VrmlNodeTUIButton(const VrmlNodeTUIButton&);
    virtual ~VrmlNodeTUIButton();
    virtual VrmlNode* cloneMe() const;
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
	virtual void eventIn(double timeStamp, const char* eventName, const VrmlField* fieldValue);
    virtual VrmlNodeType *nodeType() const;
    VrmlNodeTUIToggleButton(VrmlScene *);
    VrmlNodeTUIToggleButton(const VrmlNodeTUIToggleButton&);
    virtual ~VrmlNodeTUIToggleButton();
    virtual VrmlNode* cloneMe() const;
    virtual void tabletEvent(coTUIElement *);

    virtual void render(Viewer *);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    virtual const VrmlField *getField(const char *fieldName) const;

private:
    VrmlSFBool d_state;
    VrmlSFInt d_choice;
    std::unique_ptr<vrb::SharedState<bool>> sharedState;
};

class VRML97COVEREXPORT VrmlNodeTUIFrame : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTUIFrame(VrmlScene *);
    VrmlNodeTUIFrame(const VrmlNodeTUIFrame&);
    virtual ~VrmlNodeTUIFrame();
    virtual VrmlNode* cloneMe() const;

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
    VrmlNodeTUISplitter(const VrmlNodeTUISplitter&);
    virtual ~VrmlNodeTUISplitter();
    virtual VrmlNode* cloneMe() const;

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
    VrmlNodeTUISlider(const VrmlNodeTUISlider&);
    virtual ~VrmlNodeTUISlider();
    virtual VrmlNode* cloneMe() const;
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
    VrmlNodeTUIFloatSlider(const VrmlNodeTUIFloatSlider& n);
    virtual ~VrmlNodeTUIFloatSlider();
    virtual VrmlNode* cloneMe() const;
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
    std::unique_ptr<vrb::SharedState<float>> sharedState;
};


class VRML97COVEREXPORT VrmlNodeTUIComboBox : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTUIComboBox(VrmlScene *);
    VrmlNodeTUIComboBox(const VrmlNodeTUIComboBox& n);
    virtual ~VrmlNodeTUIComboBox();
    virtual VrmlNode* cloneMe() const;
    virtual void tabletEvent(coTUIElement *);

    virtual void render(Viewer *);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    virtual const VrmlField *getField(const char *fieldName) const;

private:
    VrmlMFString d_items;
    VrmlSFBool d_withNone;
    VrmlSFInt d_defaultChoice;
    VrmlSFInt d_choice;
    std::unique_ptr<vrb::SharedState<int>> sharedState;
};

class VRML97COVEREXPORT VrmlNodeTUIListBox : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTUIListBox(VrmlScene *);
    VrmlNodeTUIListBox(const VrmlNodeTUIListBox& n);
    virtual ~VrmlNodeTUIListBox();
    virtual VrmlNode* cloneMe() const;
    virtual void tabletEvent(coTUIElement *);

    virtual void render(Viewer *);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    virtual const VrmlField *getField(const char *fieldName) const;

private:
    VrmlMFString d_items;
    VrmlSFBool d_withNone;
    VrmlSFInt d_defaultChoice;
    VrmlSFInt d_choice;
    std::unique_ptr<vrb::SharedState<int>> sharedState;
};

class VRML97COVEREXPORT VrmlNodeTUIMap : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTUIMap(VrmlScene *);
    VrmlNodeTUIMap(const VrmlNodeTUIMap& n);
    virtual ~VrmlNodeTUIMap();
    virtual VrmlNode* cloneMe() const;
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
    VrmlNodeTUILabel(const VrmlNodeTUILabel& n);
    virtual ~VrmlNodeTUILabel();
    virtual VrmlNode* cloneMe() const;

    virtual void render(Viewer *);

private:
};
#endif //_VrmlNodeTUIElement_
