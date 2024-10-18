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
    static void initFields(VrmlNodeTUIElement *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeTUIElement(VrmlScene *, const std::string &name);
    VrmlNodeTUIElement(const VrmlNodeTUIElement&);
    virtual ~VrmlNodeTUIElement();

    VrmlNodeTUIElement *toTUIElement() const;
    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);
    void render(Viewer *) override;
    int getID(const char *name);

protected:
    VrmlSFString d_elementName;
    VrmlSFString d_parent;
    VrmlSFString d_shaderParam;
    VrmlSFVec2f d_pos;
	VrmlSFBool d_shared;
    coTUIElement *d_TUIElement;
};

class VRML97COVEREXPORT VrmlNodeTUITab : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static void initFields(VrmlNodeTUITab *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeTUITab(VrmlScene *);
    VrmlNodeTUITab(const VrmlNodeTUITab&);
    virtual ~VrmlNodeTUITab();

    void render(Viewer *) override;

    virtual void tabletPressEvent(coTUIElement *);
    virtual void tabletReleaseEvent(coTUIElement *);

private:
    VrmlSFTime d_touchTime;
    VrmlSFTime d_deactivateTime;
    static list<coTUITab *> VrmlTUITabs;
};

class VRML97COVEREXPORT VrmlNodeTUIProgressBar : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static void initFields(VrmlNodeTUIProgressBar *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeTUIProgressBar(VrmlScene *);
    VrmlNodeTUIProgressBar(const VrmlNodeTUIProgressBar&);

    void render(Viewer *) override;

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
    static void initFields(VrmlNodeTUITabFolder *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeTUITabFolder(VrmlScene *);
    VrmlNodeTUITabFolder(const VrmlNodeTUITabFolder&);
    ~VrmlNodeTUITabFolder();
    
    void render(Viewer *) override;

    virtual void tabletPressEvent(coTUIElement *);
    virtual void tabletReleaseEvent(coTUIElement *);

private:
    static list<coTUITabFolder *> VrmlTUITabFolders;
};


class VRML97COVEREXPORT VrmlNodeTUIButton : public VrmlNodeTUIElement
{

public:
    static void initFields(VrmlNodeTUIButton *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeTUIButton(VrmlScene *);
    VrmlNodeTUIButton(const VrmlNodeTUIButton&);

    virtual void tabletPressEvent(coTUIElement *);
    virtual void tabletReleaseEvent(coTUIElement *);

    void render(Viewer *) override;

private:
    VrmlSFTime d_touchTime;
    VrmlSFTime d_releaseTime;
};

class VRML97COVEREXPORT VrmlNodeTUIToggleButton : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static void initFields(VrmlNodeTUIToggleButton *node, vrml::VrmlNodeType *t);
    static const char *name();

	virtual void eventIn(double timeStamp, const char* eventName, const VrmlField* fieldValue);
    VrmlNodeTUIToggleButton(VrmlScene *);
    VrmlNodeTUIToggleButton(const VrmlNodeTUIToggleButton&);
    virtual void tabletEvent(coTUIElement *);

    void render(Viewer *) override;

private:
    VrmlSFBool d_state;
    VrmlSFInt d_choice;
    std::unique_ptr<vrb::SharedState<bool>> sharedState;
};

class VRML97COVEREXPORT VrmlNodeTUIFrame : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static void initFields(VrmlNodeTUIFrame *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeTUIFrame(VrmlScene *);
    VrmlNodeTUIFrame(const VrmlNodeTUIFrame&);
    virtual ~VrmlNodeTUIFrame();

    void render(Viewer *) override;
private:
    VrmlSFInt d_shape;
    VrmlSFInt d_style;
    static list<coTUIFrame *> VrmlTUIFrames;
};

class VRML97COVEREXPORT VrmlNodeTUISplitter : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static void initFields(VrmlNodeTUISplitter *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeTUISplitter(VrmlScene *);
    VrmlNodeTUISplitter(const VrmlNodeTUISplitter&);

    void render(Viewer *) override;
private:
    VrmlSFInt d_shape;
    VrmlSFInt d_style;
    VrmlSFInt d_orientation;
};

class VRML97COVEREXPORT VrmlNodeTUISlider : public VrmlNodeTUIElement
{

public:
    // Define the fields of TUI nodes
    static void initFields(VrmlNodeTUISlider *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeTUISlider(VrmlScene *);
    VrmlNodeTUISlider(const VrmlNodeTUISlider&);

    virtual void tabletEvent(coTUIElement *);

    void render(Viewer *) override;
    
    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);

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
    static void initFields(VrmlNodeTUIFloatSlider *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeTUIFloatSlider(VrmlScene *);
    VrmlNodeTUIFloatSlider(const VrmlNodeTUIFloatSlider& n);
    virtual void tabletEvent(coTUIElement *);

    void render(Viewer *) override;
    
    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);

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
    static void initFields(VrmlNodeTUIComboBox *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeTUIComboBox(VrmlScene *);
    VrmlNodeTUIComboBox(const VrmlNodeTUIComboBox& n);

    virtual void tabletEvent(coTUIElement *);

    void render(Viewer *) override;
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
    static void initFields(VrmlNodeTUIListBox *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeTUIListBox(VrmlScene *);
    VrmlNodeTUIListBox(const VrmlNodeTUIListBox& n);

    virtual void tabletEvent(coTUIElement *);
    void render(Viewer *) override;

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
    static void initFields(VrmlNodeTUIMap *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeTUIMap(VrmlScene *);
    VrmlNodeTUIMap(const VrmlNodeTUIMap& n);

    virtual void tabletEvent(coTUIElement *);
    void render(Viewer *) override;

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
    static void initFields(VrmlNodeTUILabel *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeTUILabel(VrmlScene *);
    VrmlNodeTUILabel(const VrmlNodeTUILabel& n);

    void render(Viewer *) override;

private:
};
#endif //_VrmlNodeTUIElement_
