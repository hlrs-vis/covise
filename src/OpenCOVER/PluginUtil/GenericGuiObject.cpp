/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "GenericGuiObject.h"

#include <cover/coVRPluginSupport.h>
#include <net/message.h>

#include <grmsg/coGRGenericParamRegisterMsg.h>
#include <grmsg/coGRGenericParamChangedMsg.h>
#include <cover/coVRTui.h>

using namespace covise;
using namespace grmsg;

namespace opencover
{
GuiTuiMapper *GuiTuiMapper::inst = NULL;

GuiTuiMapper *GuiTuiMapper::instance()
{
    if (inst == NULL)
    {
        inst = new GuiTuiMapper();
    }
    return inst;
}
int GuiTuiMapper::getParentID(std::string &name)
{
    GenericGuiObject *e = parents[name];
    if (e)
        return e->tab->getID();
    else
        return -1;
}
GuiParamBool::GuiParamBool(std::string parentName, std::string name, bool defaultValue)
    : GuiParam(parentName, name)
    , value_(defaultValue)
    , defaultValue_(defaultValue)
{
    registerAtGui();

    if (e)
    {
        int parent = e->tab->getID();
        tuiToggleButton = new coTUIToggleButton(name, parent);
        tuiToggleButton->setPos(0, e->getNextPos());
        tuiToggleButton->setEventListener(this);
        tuiToggleButton->setState(defaultValue);
    }
}

void GuiParamBool::tabletEvent(coTUIElement *tUIItem)
{
    setValue(tuiToggleButton->getState());
    if (e)
    {
        e->guiParamChanged(this);
    }
}

GuiParamBool::~GuiParamBool()
{
    delete tuiToggleButton;
}

GuiParamInt::GuiParamInt(std::string parentName, std::string name, int defaultValue)
    : GuiParam(parentName, name)
    , value_(defaultValue)
    , defaultValue_(defaultValue)
{
    registerAtGui();
    if (e)
    {
        int parent = e->tab->getID();
        tuiEdit = new coTUIEditIntField(name, parent);
        int pos = e->getNextPos();
        tuiEdit->setPos(1, pos);
        tuiEdit->setEventListener(this);
        tuiEdit->setValue(defaultValue);
        tuiLabel = new coTUILabel(name, parent);
        tuiLabel->setPos(0, pos);
    }
}

GuiParamInt::~GuiParamInt()
{
    delete tuiEdit;
    delete tuiLabel;
}

void GuiParamInt::tabletEvent(coTUIElement *tUIItem)
{
    setValue(tuiEdit->getValue());
    if (e)
    {
        e->guiParamChanged(this);
    }
}

GuiParamFloat::GuiParamFloat(std::string parentName, std::string name, float defaultValue)
    : GuiParam(parentName, name)
    , value_(defaultValue)
    , defaultValue_(defaultValue)
{
    registerAtGui();
    if (e)
    {
        int parent = e->tab->getID();
        tuiEdit = new coTUIEditFloatField(name, parent);
        int pos = e->getNextPos();
        tuiEdit->setPos(1, pos);
        tuiEdit->setEventListener(this);
        tuiEdit->setValue(defaultValue);
        tuiLabel = new coTUILabel(name, parent);
        tuiLabel->setPos(0, pos);
    }
}

GuiParamFloat::~GuiParamFloat()
{
    delete tuiEdit;
    delete tuiLabel;
}

void GuiParamFloat::tabletEvent(coTUIElement *tUIItem)
{
    setValue(tuiEdit->getValue());
    if (e)
    {
        e->guiParamChanged(this);
    }
}

GuiParamString::GuiParamString(std::string parentName, std::string name, std::string defaultValue)
    : GuiParam(parentName, name)
    , value_(defaultValue)
    , defaultValue_(defaultValue)
{
    registerAtGui();
    if (e)
    {
        int parent = e->tab->getID();
        tuiEdit = new coTUIEditField(name, parent);
        int pos = e->getNextPos();
        tuiEdit->setPos(1, pos);
        tuiEdit->setEventListener(this);
        tuiEdit->setText(defaultValue);
        tuiLabel = new coTUILabel(name, parent);
        tuiLabel->setPos(0, pos);
    }
}

GuiParamString::~GuiParamString()
{
    delete tuiEdit;
    delete tuiLabel;
}

void GuiParamString::tabletEvent(coTUIElement *tUIItem)
{
    setValue(tuiEdit->getText());
    if (e)
    {
        e->guiParamChanged(this);
    }
}

void GuiParam::registerAtGui()
{
    char *parentNameBuffer = new char[strlen(parentName_.c_str()) + 1];
    strcpy(parentNameBuffer, parentName_.c_str());
    char *nameBuffer = new char[strlen(name_.c_str()) + 1];
    strcpy(nameBuffer, name_.c_str());
    std::string tmp_value = this->getDefaultValueAsString();
    char *defaultValueBuffer = new char[strlen(tmp_value.c_str()) + 1];
    strcpy(defaultValueBuffer, tmp_value.c_str());

    coGRGenericParamRegisterMsg msg(parentNameBuffer, nameBuffer, this->getType(), defaultValueBuffer);
    Message grmsg{ Message::UI, DataHandle{(char*)msg.c_str(), strlen(msg.c_str()) + 1, false} };
    cover->sendVrbMessage(&grmsg);

    delete[] parentNameBuffer;
    delete[] nameBuffer;
    delete[] defaultValueBuffer;
}

void GuiParam::sendChangeToGui()
{
    char *parentNameBuffer = new char[strlen(parentName_.c_str()) + 1];
    strcpy(parentNameBuffer, parentName_.c_str());
    char *nameBuffer = new char[strlen(name_.c_str()) + 1];
    strcpy(nameBuffer, name_.c_str());
    std::string tmp_value = this->getValueAsString();
    char *valueBuffer = new char[strlen(tmp_value.c_str()) + 1];
    strcpy(valueBuffer, tmp_value.c_str());

    coGRGenericParamChangedMsg msg(parentNameBuffer, nameBuffer, valueBuffer);
    Message grmsg{ Message::UI, DataHandle{(char*)msg.c_str(), strlen(msg.c_str()) + 1, false} };
    cover->sendVrbMessage(&grmsg);

    delete[] parentNameBuffer;
    delete[] nameBuffer;
    delete[] valueBuffer;
}

GenericGuiObject::GenericGuiObject(std::string genericObjectName)
{
    genericObjectName_ = genericObjectName;
    tab = new coTUITab(genericObjectName, coVRTui::instance()->mainFolder->getID());
    tab->setPos(0, 0);
    ParamPos = 0;
    GuiTuiMapper::instance()->parents[genericObjectName] = this;
}

GenericGuiObject::~GenericGuiObject()
{
    GuiTuiMapper::instance()->parents.erase(genericObjectName_);
    delete tab;
}

GuiParamBool *GenericGuiObject::addNextPresStepAllowed(bool defaultValue)
{
    return addGuiParamBool("NextPresStepAllowed", defaultValue);
}

GuiParamBool *GenericGuiObject::addGuiParamBool(std::string paramName, bool defaultValue)
{
    GuiParamBool *param = new GuiParamBool(genericObjectName_, paramName, defaultValue);
    guiParams_[paramName] = param;
    return param;
}

GuiParamInt *GenericGuiObject::addGuiParamInt(std::string paramName, int defaultValue)
{
    GuiParamInt *param = new GuiParamInt(genericObjectName_, paramName, defaultValue);
    guiParams_[paramName] = param;
    return param;
}

GuiParamFloat *GenericGuiObject::addGuiParamFloat(std::string paramName, float defaultValue)
{
    GuiParamFloat *param = new GuiParamFloat(genericObjectName_, paramName, defaultValue);
    guiParams_[paramName] = param;
    return param;
}

GuiParamString *GenericGuiObject::addGuiParamString(std::string paramName, std::string defaultValue)
{
    GuiParamString *param = new GuiParamString(genericObjectName_, paramName, defaultValue);
    guiParams_[paramName] = param;
    return param;
}

GuiParamVec3 *GenericGuiObject::addGuiParamVec3(std::string paramName, osg::Vec3 defaultValue)
{
    GuiParamVec3 *param = new GuiParamVec3(genericObjectName_, paramName, defaultValue);
    guiParams_[paramName] = param;
    return param;
}

GuiParamMatrix *GenericGuiObject::addGuiParamMatrix(std::string paramName, osg::Matrix defaultValue)
{
    GuiParamMatrix *param = new GuiParamMatrix(genericObjectName_, paramName, defaultValue);
    guiParams_[paramName] = param;
    return param;
}

void GenericGuiObject::guiToRenderMsg(const char *msg)
{
    string fullMsg(string("GRMSG\n") + msg);
    coGRMsg grMsg(fullMsg.c_str());
    if (grMsg.isValid())
    {
        if (grMsg.getType() == coGRMsg::GENERIC_PARAM_CHANGED)
        {
            coGRGenericParamChangedMsg paramChangedMsg(fullMsg.c_str());
            if (genericObjectName_ == std::string(paramChangedMsg.getObjectName()))
            {
                GuiParam *param = guiParams_[std::string(paramChangedMsg.getParamName())];
                if (param != NULL)
                {
                    param->setValueFromGui(paramChangedMsg.getValue());
                    guiParamChanged(param);
                }
            }
        }
    }
}
}
