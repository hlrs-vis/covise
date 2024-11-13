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
#include <vrml97/vrml/System.h>
#include <cover/coVRTui.h>
#include <cover/coVRShader.h>
#include <string>

#include "VrmlNodeTUI.h"

using std::cerr;
using std::endl;

std::list<coTUIElement *> VrmlTUIElements;
list<coTUITab *> VrmlNodeTUITab::VrmlTUITabs;
list<coTUITabFolder *> VrmlNodeTUITabFolder::VrmlTUITabFolders;
list<coTUIFrame *> VrmlNodeTUIFrame::VrmlTUIFrames;

void VrmlNodeTUIElement::initFields(VrmlNodeTUIElement *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t);
    initFieldsHelper(node, t, 
        exposedField("elementName", node->d_elementName, [node](auto f){
            if(node->d_TUIElement != NULL)
            {
                node->d_TUIElement->setLabel(node->d_elementName.get());
            }
        }),
        exposedField("parent", node->d_parent),
        exposedField("shaderParam", node->d_shaderParam),
        exposedField("pos", node->d_pos, [node](auto f){
            if(node->d_TUIElement != NULL)
            {
                node->d_TUIElement->setPos((int)node->d_pos.x(), (int)node->d_pos.y());
            }
        }),
        exposedField("shared", node->d_shared));
}

const char *VrmlNodeTUIElement::name()
{
    return "TUIElement";
}

VrmlNodeTUIElement::VrmlNodeTUIElement(VrmlScene *scene, const std::string &name)
: VrmlNodeChild(scene, name == "" ? this->name() : name)
, d_elementName("")
, d_parent("")
, d_shaderParam("")
, d_pos(0, 0)
, d_shared(false)
, d_TUIElement(NULL)
{
    setModified();
}

VrmlNodeTUIElement::VrmlNodeTUIElement(const VrmlNodeTUIElement& n)
: VrmlNodeChild(n)
, d_elementName(n.d_elementName)
, d_parent(n.d_parent)
, d_shaderParam(n.d_shaderParam)
, d_pos(n.d_pos)
, d_shared(n.d_shared)
, d_TUIElement(NULL)
{
}

VrmlNodeTUIElement::~VrmlNodeTUIElement()
{
	VrmlTUIElements.remove(d_TUIElement);
    delete d_TUIElement;
}

void VrmlNodeTUIElement::render(Viewer *)
{
    if (isModified())
    {
      /*  if (d_TUIElement != NULL)
        {
            d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
        }*/
        clearModified();
    }
    // the elements are created in subclasses
}

VrmlNodeTUIElement *VrmlNodeTUIElement::toTUIElement() const
{
    return (VrmlNodeTUIElement *)this;
}

void VrmlNodeTUIElement::eventIn(double timeStamp,
                                 const char *eventName,
                                 const VrmlField *fieldValue)
{
    setModified();
    VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    if(d_TUIElement != NULL)
    {
        if(strcmp(eventName, "elementName") == 0)
        {
            d_TUIElement->setLabel(d_elementName.get());
        }
        if (strcmp(eventName, "pos") == 0)
        {
            d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
        }
    }
}

int VrmlNodeTUIElement::getID(const char *name)
{
	for(const auto &it: VrmlTUIElements)
    {
        if (it->getName() == name)
        {
            return it->getID();
        }
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

void VrmlNodeTUITab::initFields(VrmlNodeTUITab *node, vrml::VrmlNodeType *t)
{
    VrmlNodeTUIElement::initFields(node, t);
    initFieldsHelper(node, t, 
        eventInCallBack<VrmlSFTime>("activate", [node](auto f){
            ((coTUITab *)node->d_TUIElement)->setVal(true);
        }),
        eventInCallBack<VrmlSFTime>("set_activate", [node](auto f){
            ((coTUITab *)node->d_TUIElement)->setVal(true);
        }));

    if(t)
    {
        t->addEventOut("touchTime", VrmlField::SFTIME);
        t->addEventOut("deactivateTime", VrmlField::SFTIME);
    }
}

const char *VrmlNodeTUITab::name()
{
    return "TUITab";
}


VrmlNodeTUITab::VrmlNodeTUITab(VrmlScene *scene)
    : VrmlNodeTUIElement(scene, name())
{
}

VrmlNodeTUITab::VrmlNodeTUITab(const VrmlNodeTUITab& n)
    : VrmlNodeTUIElement(n)
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
                VrmlTUIElements.push_back(d_TUIElement);
                d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
            }

            VrmlNodeTUIElement::render(viewer);
        }
        clearModified();
    }
    // the elements are created in subclasses
}

//
//
//
// ProgressBar
//
//
//

void VrmlNodeTUIProgressBar::initFields(VrmlNodeTUIProgressBar *node, vrml::VrmlNodeType *t)
{
    VrmlNodeTUIElement::initFields(node, t);
    initFieldsHelper(node, t, 
        exposedField("max", node->d_max),
        exposedField("value", node->d_value));
}

const char *VrmlNodeTUIProgressBar::name()
{
    return "TUIProgressBar";
}

VrmlNodeTUIProgressBar::VrmlNodeTUIProgressBar(VrmlScene *scene)
    : VrmlNodeTUIElement(scene, name())
    , d_max(100)
    , d_value(0)
{
}
VrmlNodeTUIProgressBar::VrmlNodeTUIProgressBar(const VrmlNodeTUIProgressBar& n)
    : VrmlNodeTUIElement(n)
    , d_max(n.d_max)
    , d_value(n.d_value)
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
            VrmlTUIElements.push_back(d_TUIElement);
            d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
            coTUIProgressBar *pb = (coTUIProgressBar *)d_TUIElement;
            pb->setMax(d_max.get());
            pb->setValue(d_value.get());
        }
        VrmlNodeTUIElement::render(viewer);
        clearModified();
    }
    // the elements are created in subclasses
}

//
//
//
// TABFolder
//
//
//

void VrmlNodeTUITabFolder::initFields(VrmlNodeTUITabFolder *node, vrml::VrmlNodeType *t)
{
    VrmlNodeTUIElement::initFields(node, t);
}

const char *VrmlNodeTUITabFolder::name()
{
    return "TUITabFolder";
}


VrmlNodeTUITabFolder::VrmlNodeTUITabFolder(VrmlScene *scene)
    : VrmlNodeTUIElement(scene, name())
{
}
VrmlNodeTUITabFolder::VrmlNodeTUITabFolder(const VrmlNodeTUITabFolder& n)
    : VrmlNodeTUIElement(n)
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
                VrmlTUIElements.push_back(d_TUIElement);
            }
            d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
            VrmlNodeTUIElement::render(viewer);
        }
        clearModified();
    }
    // the elements are created in subclasses
}

//
//
//
// BUTTON
//
//
//

void VrmlNodeTUIButton::initFields(VrmlNodeTUIButton *node, vrml::VrmlNodeType *t)
{
    VrmlNodeTUIElement::initFields(node, t);
    if(t)
    {
        t->addEventOut("touchTime", VrmlField::SFTIME);
        t->addEventOut("releaseTime", VrmlField::SFTIME); 
    }
}

const char *VrmlNodeTUIButton::name()
{
    return "TUIButton";
}

VrmlNodeTUIButton::VrmlNodeTUIButton(VrmlScene *scene)
    : VrmlNodeTUIElement(scene, name())
    , d_touchTime(0.0)
{
}
VrmlNodeTUIButton::VrmlNodeTUIButton(const VrmlNodeTUIButton& n)
    : VrmlNodeTUIElement(n)
    , d_touchTime(n.d_touchTime)
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
            VrmlTUIElements.push_back(d_TUIElement);
            d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
        }
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

void VrmlNodeTUIToggleButton::initFields(VrmlNodeTUIToggleButton *node, vrml::VrmlNodeType *t)
{
    VrmlNodeTUIElement::initFields(node, t);
    initFieldsHelper(node, t, 
        exposedField("choice", node->d_choice, [node](auto f){
            if(node->d_TUIElement != NULL)
            {
                coTUIToggleButton *tb = (coTUIToggleButton *)node->d_TUIElement;
                node->d_state.set(node->d_choice.get() >= 0);
                tb->setState(node->d_choice.get() >= 0);
            }
        }),
        exposedField("state", node->d_state, [node](auto f){
            if(node->d_TUIElement != NULL)
            {
                coTUIToggleButton *tb = (coTUIToggleButton *)node->d_TUIElement;
                tb->setState(node->d_state.get());
                if (node->sharedState && node->d_shared.get())
				    *node->sharedState = node->d_state.get();
            }
        }));
}

const char *VrmlNodeTUIToggleButton::name()
{
    return "TUIToggleButton";
}

void VrmlNodeTUIToggleButton::eventIn(double timeStamp,
	const char* eventName,
	const VrmlField* fieldValue)
{
	setModified();
	VrmlNodeTUIElement::eventIn(timeStamp, eventName, fieldValue);
	if (d_TUIElement != NULL)
	{
		if (strcmp(eventName, "elementName") == 0)
		{
		}
	}
}

VrmlNodeTUIToggleButton::VrmlNodeTUIToggleButton(VrmlScene *scene)
    : VrmlNodeTUIElement(scene, name())
    , d_state(false)
    , d_choice(-1)
{
}

VrmlNodeTUIToggleButton::VrmlNodeTUIToggleButton(const VrmlNodeTUIToggleButton& n)
: VrmlNodeTUIElement(n), d_state(n.d_state), d_choice(n.d_choice)
{
}

void VrmlNodeTUIToggleButton::tabletEvent(coTUIElement *)
{
    double timeStamp = System::the->time();
    coTUIToggleButton *tb = (coTUIToggleButton *)d_TUIElement;
    d_state.set(tb->getState());
    eventOut(timeStamp, "state", d_state);

	
	if (sharedState)
	{
		*sharedState = tb->getState();
	}
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
            VrmlTUIElements.push_back(d_TUIElement);
        d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
		if (strcmp(d_elementName.get(), "") != 0 && strcmp(d_parent.get(), "") != 0 && sharedState == nullptr)
		{
			std::string sName = d_parent.get();
			sName = sName + ".";
			sName = sName + d_elementName.get();
			sharedState.reset(new vrb::SharedState<bool>(sName, d_state.get()));
			sharedState->setUpdateFunction([this]() {
				if (d_shared.get())
				{
					coTUIToggleButton* tb = (coTUIToggleButton*)d_TUIElement;
					tb->setState(*sharedState);
					d_state.set(tb->getState());
					double timeStamp = System::the->time();
					eventOut(timeStamp, "state", d_state);
					if (tb->getState())
						d_choice.set(0);
					else
						d_choice.set(-1);
					eventOut(timeStamp, "choice", d_choice);
				}
				});
		}
        }
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



void VrmlNodeTUIFrame::initFields(VrmlNodeTUIFrame *node, vrml::VrmlNodeType *t)
{
    VrmlNodeTUIElement::initFields(node, t);
    initFieldsHelper(node, t, 
        exposedField("shape", node->d_shape),
        exposedField("style", node->d_style));
}

const char *VrmlNodeTUIFrame::name()
{
    return "TUIFrame";
}

VrmlNodeTUIFrame::VrmlNodeTUIFrame(VrmlScene *scene)
    : VrmlNodeTUIElement(scene, name())
    , d_shape(0)
    , d_style(0)
{
}
VrmlNodeTUIFrame::VrmlNodeTUIFrame(const VrmlNodeTUIFrame& n)
    : VrmlNodeTUIElement(n)
    , d_shape(n.d_shape)
    , d_style(n.d_style)
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
                VrmlTUIElements.push_back(d_TUIElement);
                d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
            }
            VrmlNodeTUIElement::render(viewer);
        }
        clearModified();
    }
    // the elements are created in subclasses
}

//
//
//
// Splitter
//
//
//

void VrmlNodeTUISplitter::initFields(VrmlNodeTUISplitter *node, vrml::VrmlNodeType *t)
{
    VrmlNodeTUIElement::initFields(node, t);
    initFieldsHelper(node, t, 
        exposedField("shape", node->d_shape),
        exposedField("style", node->d_style),
        exposedField("orientation", node->d_orientation));
}

const char *VrmlNodeTUISplitter::name()
{
    return "TUISplitter";
}   

VrmlNodeTUISplitter::VrmlNodeTUISplitter(VrmlScene *scene)
    : VrmlNodeTUIElement(scene, name())
    , d_shape(0)
    , d_style(0)
    , d_orientation(0x1)
{
}
VrmlNodeTUISplitter::VrmlNodeTUISplitter(const VrmlNodeTUISplitter& n)
    : VrmlNodeTUIElement(n)
    , d_shape(n.d_shape)
    , d_style(n.d_style)
    , d_orientation(n.d_orientation)
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
            VrmlTUIElements.push_back(d_TUIElement);
            d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
        }
        VrmlNodeTUIElement::render(viewer);
        clearModified();
    }
    // the elements are created in subclasses
}

//
//
//
// LABEL
//
//
//


void VrmlNodeTUILabel::initFields(VrmlNodeTUILabel *node, vrml::VrmlNodeType *t)
{
    VrmlNodeTUIElement::initFields(node, t);
}

const char *VrmlNodeTUILabel::name()
{
    return "TUILabel";
}

VrmlNodeTUILabel::VrmlNodeTUILabel(VrmlScene *scene)
    : VrmlNodeTUIElement(scene, name())
{
}
VrmlNodeTUILabel::VrmlNodeTUILabel(const VrmlNodeTUILabel& n)
    : VrmlNodeTUIElement(n)
{
}

void VrmlNodeTUILabel::render(Viewer *viewer)
{
    if (isModified())
    {
        if (d_TUIElement == NULL)
        {
            d_TUIElement = (coTUIElement *)new coTUILabel(d_elementName.get(), getID(d_parent.get()));
            VrmlTUIElements.push_back(d_TUIElement);
            d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
        }
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

void VrmlNodeTUIFloatSlider::initFields(VrmlNodeTUIFloatSlider *node, vrml::VrmlNodeType *t)
{
    VrmlNodeTUIElement::initFields(node, t);
    initFieldsHelper(node, t, 
        exposedField("min", node->d_min, [node](auto f){
            coTUISlider *ts = dynamic_cast<coTUISlider *>(node->d_TUIElement);
            if(ts)
                ts->setMin(node->d_min.get());
        }),
        exposedField("max", node->d_max, [node](auto f){
            coTUISlider *ts = dynamic_cast<coTUISlider *>(node->d_TUIElement);
            if(ts)
                ts->setMax(node->d_max.get());
        }),
        exposedField("value", node->d_value, [node](auto f){
            coTUISlider *ts = dynamic_cast<coTUISlider *>(node->d_TUIElement);
            if(ts)
            {
                ts->setValue(node->d_value.get());
                if(node->sharedState)
                    *node->sharedState = node->d_value.get();
            }
        }),
        exposedField("orientation", node->d_orientation, [node](auto f){
            coTUISlider *ts = dynamic_cast<coTUISlider *>(node->d_TUIElement);
            if(ts)
            {
                bool ori = false;
                if (strcasecmp(node->d_orientation.get(), "horizontal") == 0)
                    ori = true;
                ts->setOrientation(ori);
            }
        }));
}

const char *VrmlNodeTUIFloatSlider::name()
{
    return "TUIFloatSlider";
}

VrmlNodeTUIFloatSlider::VrmlNodeTUIFloatSlider(VrmlScene *scene)
    : VrmlNodeTUIElement(scene, name())
    , d_min(0.0)
    , d_max(0.0)
    , d_value(50.0)
    , d_orientation("horizontal")
{
}

VrmlNodeTUIFloatSlider::VrmlNodeTUIFloatSlider(const VrmlNodeTUIFloatSlider& n)
    : VrmlNodeTUIElement(n)
    , d_min(n.d_min)
    , d_max(n.d_max)
    , d_value(n.d_value)
    , d_orientation(n.d_orientation)
{
}

void VrmlNodeTUIFloatSlider::tabletEvent(coTUIElement *)
{
    double timeStamp = System::the->time();
    coTUIFloatSlider *sl = (coTUIFloatSlider *)d_TUIElement;
    d_value.set(sl->getValue());
	if (sharedState)
	{
		*sharedState = sl->getValue();
	}
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
            VrmlTUIElements.push_back(d_TUIElement);
            fs = (coTUIFloatSlider *)d_TUIElement;
            fs->setMin(d_min.get());
            fs->setMax(d_max.get());
            fs->setValue(d_value.get());
	    //fprintf(stderr,"%s: min %f max %f value %f\n",d_elementName.get(),d_min.get(),d_max.get(),d_value.get());

            bool ori = false;
            if (strcasecmp(d_orientation.get(), "horizontal") == 0)
                ori = true;
            fs->setOrientation(ori);
            d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
			if (strcmp(d_elementName.get(), "") != 0 && strcmp(d_parent.get(), "") != 0 && sharedState == nullptr)
			{
				std::string sName = d_parent.get();
				sName = sName + ".";
				sName = sName + d_elementName.get();
				sharedState.reset(new vrb::SharedState<float>(sName, d_value.get()));
				sharedState->setUpdateFunction([this]() {
					if (d_shared.get())
					{
						coTUIFloatSlider* sl = (coTUIFloatSlider*)d_TUIElement;
						double timeStamp = System::the->time();
						sl->setValue(*sharedState);
						d_value.set(sl->getValue());
						eventOut(timeStamp, "value", d_value);
					}
					});
			}
        }
        VrmlNodeTUIElement::render(viewer);
        clearModified();
    }
    // the elements are created in subclasses
}



void VrmlNodeTUIFloatSlider::eventIn(double timeStamp,
                                 const char *fieldName,
                                 const VrmlField *fieldValue)
{
    setModified();
    VrmlNodeTUIElement::eventIn(timeStamp, fieldName, fieldValue);     
    coTUIFloatSlider *ts = dynamic_cast<coTUIFloatSlider *>(d_TUIElement);
    if(ts!=NULL)
    {
        if(strcmp(fieldName,"min")==0)
        {
            ts->setMin(d_min.get());
        }
        else if(strcmp(fieldName,"max")==0)
        {
            ts->setMax(d_max.get());
        }
        else if(strcmp(fieldName,"value")==0)
        {
            ts->setValue(d_value.get());
			if (sharedState)
			{
				*sharedState = d_value.get();
			}
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
                uniform->set(d_value.get());
            }
        }
    }
        }
        else if(strcmp(fieldName,"orientation")==0)
        {
            bool ori = false;
            if (strcasecmp(d_orientation.get(), "horizontal") == 0)
                ori = true;
            ts->setOrientation(ori);
        }
    }
}

//
//
//
// Slider
//
//
//

void VrmlNodeTUISlider::initFields(VrmlNodeTUISlider *node, vrml::VrmlNodeType *t)
{
    VrmlNodeTUIElement::initFields(node, t);
    initFieldsHelper(node, t, 
        exposedField("min", node->d_min, [node](auto f){
            coTUISlider *ts = dynamic_cast<coTUISlider *>(node->d_TUIElement);
            if(ts)
                ts->setMin(node->d_min.get());
        }),
        exposedField("max", node->d_max, [node](auto f){
            coTUISlider *ts = dynamic_cast<coTUISlider *>(node->d_TUIElement);
            if(ts)
                ts->setMax(node->d_max.get());
        }),
        exposedField("value", node->d_value, [node](auto f){
            coTUISlider *ts = dynamic_cast<coTUISlider *>(node->d_TUIElement);
            if(ts)
                ts->setValue(node->d_value.get());
        }),
        exposedField("orientation", node->d_orientation, [node](auto f){
            coTUISlider *ts = dynamic_cast<coTUISlider *>(node->d_TUIElement);
            if(ts)
            {
                bool ori = false;
                if (strcasecmp(node->d_orientation.get(), "horizontal") == 0)
                    ori = true;
                ts->setOrientation(ori);
            }
        }));
}

const char *VrmlNodeTUISlider::name()
{
    return "TUISlider";
}

VrmlNodeTUISlider::VrmlNodeTUISlider(VrmlScene *scene)
    : VrmlNodeTUIElement(scene, name())
    , d_min(0)
    , d_max(0)
    , d_value(50)
    , d_orientation("horizontal")
{
}
VrmlNodeTUISlider::VrmlNodeTUISlider(const VrmlNodeTUISlider& n)
    : VrmlNodeTUIElement(n)
    , d_min(n.d_min)
    , d_max(n.d_min)
    , d_value(n.d_value)
    , d_orientation(n.d_orientation)
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
            VrmlTUIElements.push_back(d_TUIElement);
            coTUISlider *ts = (coTUISlider *)d_TUIElement;
            ts->setMin(d_min.get());
            ts->setMax(d_max.get());
            ts->setValue(d_value.get());
            d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
            bool ori = false;
            if (strcasecmp(d_orientation.get(), "horizontal") == 0)
                ori = true;
            ts->setOrientation(ori);
        }

        VrmlNodeTUIElement::render(viewer);
        clearModified();
    }
    // the elements are created in subclasses
}

void VrmlNodeTUISlider::eventIn(double timeStamp,
                                 const char *fieldName,
                                 const VrmlField *fieldValue)
{
    setModified();
    VrmlNodeTUIElement::eventIn(timeStamp, fieldName, fieldValue);     
    coTUISlider *ts = dynamic_cast<coTUISlider *>(d_TUIElement);
    if(ts!=NULL)
    {
        if(strcmp(fieldName,"min")==0)
        {
            ts->setMin(d_min.get());
        }
        else if(strcmp(fieldName,"max")==0)
        {
            ts->setMax(d_max.get());
        }
        else if(strcmp(fieldName,"value")==0)
        {
            ts->setValue(d_value.get());
        }
        else if(strcmp(fieldName,"orientation")==0)
        {
            bool ori = false;
            if (strcasecmp(d_orientation.get(), "horizontal") == 0)
                ori = true;
            ts->setOrientation(ori);
        }
        
    }
}

//
//
//
// ComboBox
//
//
//

void VrmlNodeTUIComboBox::initFields(VrmlNodeTUIComboBox *node, vrml::VrmlNodeType *t)
{
    VrmlNodeTUIElement::initFields(node, t);
    initFieldsHelper(node, t, 
        exposedField("items", node->d_items),
        exposedField("withNone", node->d_withNone),
        exposedField("defaultChoice", node->d_defaultChoice),
        exposedField("choice", node->d_choice, [node](auto f){
            coTUIComboBox *cb = dynamic_cast<coTUIComboBox *>(node->d_TUIElement);
            if(cb)
            {
                if (node->sharedState)
                {
                    *node->sharedState = node->d_choice.get();
                }
                if (node->d_withNone.get())
                    cb->setSelectedEntry(node->d_choice.get() + 1);
                else
                    cb->setSelectedEntry(node->d_choice.get());
            }
        }));
}

const char *VrmlNodeTUIComboBox::name()
{
    return "TUIComboBox";
}

VrmlNodeTUIComboBox::VrmlNodeTUIComboBox(VrmlScene *scene)
    : VrmlNodeTUIElement(scene, name())
    , d_withNone(true)
    , d_defaultChoice(0)
{
}
VrmlNodeTUIComboBox::VrmlNodeTUIComboBox(const VrmlNodeTUIComboBox& n)
    : VrmlNodeTUIElement(n)
    , d_withNone(n.d_withNone)
    , d_defaultChoice(n.d_defaultChoice)
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
	if (sharedState)
	{
		*sharedState = d_choice.get();
	}
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
            VrmlTUIElements.push_back(d_TUIElement);
            d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());

			if (strcmp(d_elementName.get(), "") != 0 && strcmp(d_parent.get(), "") != 0 && sharedState == nullptr)
			{
				std::string sName = d_parent.get();
				sName = sName + ".";
				sName = sName + d_elementName.get();
				sharedState.reset(new vrb::SharedState<int>(sName, d_choice.get()));
				sharedState->setUpdateFunction([this]() {
					if (d_shared.get())
					{
						coTUIComboBox* cb = (coTUIComboBox*)d_TUIElement;
						double timeStamp = System::the->time();
						cb->setSelectedEntry(*sharedState);
						d_choice.set(*sharedState);
						eventOut(timeStamp, "choice", d_choice);
					}
					});
			}
        }
        VrmlNodeTUIElement::render(viewer);
        clearModified();
    }
    // the elements are created in subclasses
}

//
//
//
// ListBox
//
//
//

void VrmlNodeTUIListBox::initFields(VrmlNodeTUIListBox *node, vrml::VrmlNodeType *t)
{
    VrmlNodeTUIElement::initFields(node, t);
    initFieldsHelper(node, t, 
        exposedField("items", node->d_items),
        exposedField("withNone", node->d_withNone),
        exposedField("defaultChoice", node->d_defaultChoice));

    if(t)
        t->addEventOut("choice", VrmlField::SFINT32);     
}

const char *VrmlNodeTUIListBox::name()
{
    return "TUIListBox";
}

VrmlNodeTUIListBox::VrmlNodeTUIListBox(VrmlScene *scene)
    : VrmlNodeTUIElement(scene, name())
    , d_withNone(true)
    , d_defaultChoice(0)
{
}
VrmlNodeTUIListBox::VrmlNodeTUIListBox(const VrmlNodeTUIListBox& n)
    : VrmlNodeTUIElement(n)
    , d_withNone(n.d_withNone)
    , d_defaultChoice(n.d_defaultChoice)
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
	if (sharedState)
	{
		*sharedState = d_choice.get();
	}
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
            VrmlTUIElements.push_back(d_TUIElement);
            d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
			if (strcmp(d_elementName.get(), "") != 0 && strcmp(d_parent.get(), "") != 0 && sharedState == nullptr)
			{
				std::string sName = d_parent.get();
				sName = sName + ".";
				sName = sName + d_elementName.get();
				sharedState.reset(new vrb::SharedState<int>(sName, d_choice.get()));
				sharedState->setUpdateFunction([this]() {
					if (d_shared.get())
					{
						coTUIListBox* cb = (coTUIListBox*)d_TUIElement;
						double timeStamp = System::the->time();
						cb->setSelectedEntry(*sharedState);
						d_choice.set(*sharedState);
						eventOut(timeStamp, "choice", d_choice);
					}
					});
			}
        }
        VrmlNodeTUIElement::render(viewer);
        clearModified();
    }
    // the elements are created in subclasses
}

//
//
//
// Map
//
//
//

void VrmlNodeTUIMap::initFields(VrmlNodeTUIMap *node, vrml::VrmlNodeType *t)
{
    VrmlNodeTUIElement::initFields(node, t);
    initFieldsHelper(node, t, 
        exposedField("ox", node->d_ox),
        exposedField("oy", node->d_oy),
        exposedField("xSize", node->d_xSize),
        exposedField("ySize", node->d_ySize),
        exposedField("height", node->d_height),
        exposedField("maps", node->d_maps));

    if(t)
    {
        t->addEventOut("currentMap", VrmlField::SFINT32);
        t->addEventOut("currentPos", VrmlField::SFVEC3F);
        t->addEventOut("currentRot", VrmlField::SFROTATION);
    }        

}

const char *VrmlNodeTUIMap::name()
{
    return "TUIMap";
}

VrmlNodeTUIMap::VrmlNodeTUIMap(VrmlScene *scene)
    : VrmlNodeTUIElement(scene, name())
    , d_currentMap(-1)
    , d_currentPos(0, 0, 0)
{
}

VrmlNodeTUIMap::VrmlNodeTUIMap(const VrmlNodeTUIMap& n)
    : VrmlNodeTUIElement(n)
    , d_currentMap(n.d_currentMap)
    , d_currentPos(n.d_currentPos)
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
            VrmlTUIElements.push_back(d_TUIElement);
            d_TUIElement->setPos((int)d_pos.x(), (int)d_pos.y());
        }
        VrmlNodeTUIElement::render(viewer);
        clearModified();
    }
    // the elements are created in subclasses
}
