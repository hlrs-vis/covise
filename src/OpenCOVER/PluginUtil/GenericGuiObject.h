/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2010 Visenso  **
 **                                                                        **
 ** Description: GenericGuiObject                                          **
 **              Provides a generic object with parameters displayed in    **
 **              vr-prepare.                                               **
 **                                                                        **
 ** Author: C. Spenrath                                                    **
 **                                                                        **
 **                                                                        **
 ** Use:                                                                   **
 **      Let your objects inherit from GenericGuiObject.                   **
 **      Call guiToRenderMsg(msg) in your plugins guiToRenderMsg function. **
 **                                                                        **
 **      You add params by calling addGuiParam* in the objects             **
 **      constructor. Use setValue and getValue to access the value.       **
 **      setValue forwards to change to vr-prepare. You can override       **
 **      guiParamChanged to act on param changes from vr-prepare.          **
 **                                                                        **
 **      Use addNextPresStepAllowed to add a param which controls whether  **
 **      the next PresentationStep is allowed. You have to make sure the   **
 **      value is correct all the time.                                    **
 **                                                                        **
\****************************************************************************/

#ifndef _GENERIC_GUI_OBJECT_H
#define _GENERIC_GUI_OBJECT_H

#include <osg/Matrix>
#include <osg/Vec3>

#include <util/coExport.h>
#include <cover/coTabletUI.h>

#include <string>
#include <sstream>
#include <map>
#include <cstring>
#include <cstdlib>

namespace opencover
{

class GenericGuiObject;
class PLUGIN_UTILEXPORT GuiTuiMapper
{
public:
    GuiTuiMapper(){};
    ~GuiTuiMapper(){};
    static GuiTuiMapper *instance();
    int getParentID(std::string &name);
    std::map<std::string, GenericGuiObject *> parents;

private:
    static GuiTuiMapper *inst;
};

class PLUGIN_UTILEXPORT GuiParam
{
    friend class GenericGuiObject; // GenericGuiObject needs access to setValueFromGui

public:
    GuiParam(std::string parentName, std::string name)
        : parentName_(parentName)
        , name_(name)
    {
        e = GuiTuiMapper::instance()->parents[parentName];
    };
    virtual ~GuiParam(){};

    std::string getName()
    {
        return name_;
    };
    virtual int getType() = 0;

protected:
    virtual void setValueFromGui(const char *value) = 0;
    virtual std::string getValueAsString() = 0;
    virtual std::string getDefaultValueAsString() = 0;
    void registerAtGui();
    void sendChangeToGui();
    GenericGuiObject *e;

private:
    std::string parentName_;
    std::string name_;
};

class PLUGIN_UTILEXPORT GuiParamBool : public GuiParam, public coTUIListener
{
public:
    GuiParamBool(std::string parentName, std::string name, bool defaultValue);
    ~GuiParamBool();

    int getType()
    {
        return 0;
    };
    bool getValue()
    {
        return value_;
    };
    void setValue(bool value)
    {
        bool c = (value != value_);
        value_ = value;
        if (c)
            sendChangeToGui();
        tuiToggleButton->setState(value);
    };

private:
    void setValueFromGui(const char *value)
    {
        value_ = (strcmp(value, "0") != 0);
    };
    std::string getValueAsString()
    {
        return value_ ? "1" : "0";
    };
    std::string getDefaultValueAsString()
    {
        return defaultValue_ ? "1" : "0";
    };

    bool value_, defaultValue_;
    virtual void tabletEvent(coTUIElement *tUIItem);
    coTUIToggleButton *tuiToggleButton;
};

class PLUGIN_UTILEXPORT GuiParamInt : public GuiParam, public coTUIListener
{
public:
    GuiParamInt(std::string parentName, std::string name, int defaultValue);
    ~GuiParamInt();

    int getType()
    {
        return 1;
    };
    int getValue()
    {
        return value_;
    };
    void setValue(int value)
    {
        bool c = (value != value_);
        value_ = value;
        if (c)
            sendChangeToGui();
        tuiEdit->setValue(value);
    };

private:
    void setValueFromGui(const char *value)
    {
        value_ = atoi(value);
    };
    std::string getValueAsString()
    {
        std::stringstream sstr;
        sstr << value_;
        return sstr.str();
    };
    std::string getDefaultValueAsString()
    {
        std::stringstream sstr;
        sstr << defaultValue_;
        return sstr.str();
    };

    int value_, defaultValue_;
    virtual void tabletEvent(coTUIElement *tUIItem);
    coTUIEditIntField *tuiEdit;
    coTUILabel *tuiLabel;
};

class PLUGIN_UTILEXPORT GuiParamFloat : public GuiParam, public coTUIListener
{
public:
    GuiParamFloat(std::string parentName, std::string name, float defaultValue);
    ~GuiParamFloat();

    int getType()
    {
        return 2;
    };
    float getValue()
    {
        return value_;
    };
    void setValue(float value)
    {
        bool c = (value != value_);
        value_ = value;
        if (c)
            sendChangeToGui();
    };

private:
    void setValueFromGui(const char *value)
    {
        value_ = atof(value);
    };
    std::string getValueAsString()
    {
        std::stringstream sstr;
        sstr << value_;
        return sstr.str();
    };
    std::string getDefaultValueAsString()
    {
        std::stringstream sstr;
        sstr << defaultValue_;
        return sstr.str();
    };

    float value_, defaultValue_;
    virtual void tabletEvent(coTUIElement *tUIItem);
    coTUIEditFloatField *tuiEdit;
    coTUILabel *tuiLabel;
};

class PLUGIN_UTILEXPORT GuiParamString : public GuiParam, public coTUIListener
{
public:
    GuiParamString(std::string parentName, std::string name, std::string defaultValue);
    ~GuiParamString();

    int getType()
    {
        return 3;
    };
    std::string getValue()
    {
        return value_;
    };
    void setValue(std::string value)
    {
        bool c = (value != value_);
        value_ = value;
        if (c)
            sendChangeToGui();
    };

private:
    void setValueFromGui(const char *value)
    {
        value_ = std::string(value);
    };
    std::string getValueAsString()
    {
        return value_;
    };
    std::string getDefaultValueAsString()
    {
        return defaultValue_;
    };

    std::string value_, defaultValue_;
    virtual void tabletEvent(coTUIElement *tUIItem);
    coTUIEditField *tuiEdit;
    coTUILabel *tuiLabel;
};

class PLUGIN_UTILEXPORT GuiParamVec3 : public GuiParam
{
public:
    GuiParamVec3(std::string parentName, std::string name, osg::Vec3 defaultValue)
        : GuiParam(parentName, name)
        , value_(defaultValue)
        , defaultValue_(defaultValue)
    {
        registerAtGui();
    };
    ~GuiParamVec3(){};

    int getType()
    {
        return 4;
    };
    osg::Vec3 getValue()
    {
        return value_;
    };
    void setValue(osg::Vec3 value)
    {
        bool c = (value != value_);
        value_ = value;
        if (c)
            sendChangeToGui();
    };

private:
    void setValueFromGui(const char *value)
    {
        std::stringstream sstr(value);
        std::string s;
        for (int i = 0; i < 3; ++i)
        {
            getline(sstr, s, '/');
            value_[i] = atof(s.c_str());
        }
    };
    std::string getValueAsString()
    {
        std::stringstream sstr;
        sstr << value_[0] << "/" << value_[1] << "/" << value_[2];
        return sstr.str();
    };
    std::string getDefaultValueAsString()
    {
        std::stringstream sstr;
        sstr << defaultValue_[0] << "/" << defaultValue_[1] << "/" << defaultValue_[2];
        return sstr.str();
    };

    osg::Vec3 value_, defaultValue_;
};

class PLUGIN_UTILEXPORT GuiParamMatrix : public GuiParam
{
public:
    GuiParamMatrix(std::string parentName, std::string name, osg::Matrix defaultValue)
        : GuiParam(parentName, name)
        , value_(defaultValue)
        , defaultValue_(defaultValue)
    {
        registerAtGui();
    };
    ~GuiParamMatrix(){};

    int getType()
    {
        return 5;
    };
    osg::Matrix getValue()
    {
        return value_;
    };
    void setValue(osg::Matrix value)
    {
        bool c = (value != value_);
        value_ = value;
        if (c)
            sendChangeToGui();
    };

private:
    void setValueFromGui(const char *value)
    {
        std::stringstream sstr(value);
        std::string s;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
            {
                getline(sstr, s, '/');
                value_(i, j) = atof(s.c_str());
            }
    };
    std::string getValueAsString()
    {
        std::stringstream sstr;
        sstr << value_(0, 0) << "/" << value_(0, 1) << "/" << value_(0, 2) << "/" << value_(0, 3) << "/" << value_(1, 0) << "/" << value_(1, 1) << "/" << value_(1, 2) << "/" << value_(1, 3) << "/" << value_(2, 0) << "/" << value_(2, 1) << "/" << value_(2, 2) << "/" << value_(2, 3) << "/" << value_(3, 0) << "/" << value_(3, 1) << "/" << value_(3, 2) << "/" << value_(3, 3);
        return sstr.str();
    };
    std::string getDefaultValueAsString()
    {
        std::stringstream sstr;
        sstr << defaultValue_(0, 0) << "/" << defaultValue_(0, 1) << "/" << defaultValue_(0, 2) << "/" << defaultValue_(0, 3) << "/" << defaultValue_(1, 0) << "/" << defaultValue_(1, 1) << "/" << defaultValue_(1, 2) << "/" << defaultValue_(1, 3) << "/" << defaultValue_(2, 0) << "/" << defaultValue_(2, 1) << "/" << defaultValue_(2, 2) << "/" << defaultValue_(2, 3) << "/" << defaultValue_(3, 0) << "/" << defaultValue_(3, 1) << "/" << defaultValue_(3, 2) << "/" << defaultValue_(3, 3);
        return sstr.str();
    };

    osg::Matrix value_, defaultValue_;
};

class PLUGIN_UTILEXPORT GenericGuiObject
{
public:
    GenericGuiObject(std::string genericObjectName);
    virtual ~GenericGuiObject();

    GuiParamBool *addNextPresStepAllowed(bool defaultValue);

    GuiParamBool *addGuiParamBool(std::string paramName, bool defaultValue);
    GuiParamInt *addGuiParamInt(std::string paramName, int defaultValue);
    GuiParamFloat *addGuiParamFloat(std::string paramName, float defaultValue);
    GuiParamString *addGuiParamString(std::string paramName, std::string defaultValue);
    GuiParamVec3 *addGuiParamVec3(std::string paramName, osg::Vec3 defaultValue);
    GuiParamMatrix *addGuiParamMatrix(std::string paramName, osg::Matrix defaultValue);

    void guiToRenderMsg(const char *msg);
    coTUITab *tab;
    int getNextPos()
    {
        ParamPos++;
        return (ParamPos - 1);
    };
    virtual void guiParamChanged(GuiParam *guiParam)
    {
        (void)guiParam;
    };

protected:
private:
    std::string genericObjectName_;
    std::map<std::string, GuiParam *> guiParams_;
    int ParamPos;
};
}
#endif
