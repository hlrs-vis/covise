/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ScriptVrmlNode.h"
#include "ScriptPlugin.h"

#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/VrmlNamespace.h>

#include <vrml97/vrml/VrmlMFString.h>
#include <vrml97/vrml/VrmlSFString.h>

#include <cover/coVRPluginSupport.h>

#include "ScriptWsCovise.h"

ScriptPlugin *ScriptVrmlNode::scriptPlugin = 0;

using namespace vrml;

static vrml::VrmlNode *creator(vrml::VrmlScene *scene)
{
    return new ScriptVrmlNode(scene);
}

vrml::VrmlNodeType *ScriptVrmlNode::defineType(vrml::VrmlNodeType *t)
{
    static vrml::VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new vrml::VrmlNodeType("COVERScript", creator);
    }

    vrml::VrmlNodeChild::defineType(t); // Parent class
    t->addExposedField("command", vrml::VrmlField::SFSTRING);
    t->addEventOut("coviseEventLink", vrml::VrmlField::MFSTRING);
    t->addEventOut("coviseEventModuleAdd", vrml::VrmlField::SFSTRING);
    t->addEventOut("coviseEventModuleDel", vrml::VrmlField::SFSTRING);
    t->addEventOut("coviseEventModuleDied", vrml::VrmlField::SFSTRING);
    t->addEventOut("coviseEventParameterChanged", vrml::VrmlField::MFSTRING);
    t->addEventOut("coviseEventOpenNet", vrml::VrmlField::SFSTRING);
    t->addEventOut("coviseEventOpenNetDone", vrml::VrmlField::SFSTRING);
    t->addEventOut("coviseEventQuit", vrml::VrmlField::SFSTRING); // TODO How to define an event without parameter?

    return t;
}

vrml::VrmlNodeType *ScriptVrmlNode::nodeType() const
{
    return defineType(0);
}

ScriptVrmlNode::ScriptVrmlNode(vrml::VrmlScene *scene)
    : QObject(0)
    , vrml::VrmlNodeChild(scene)
{

    connect(ScriptVrmlNode::scriptPlugin->covise(), SIGNAL(eventLink(QString, QString)),
            this, SLOT(coviseEventLink(QString, QString)));

    connect(ScriptVrmlNode::scriptPlugin->covise(), SIGNAL(eventModuleAdd(QString)),
            this, SLOT(coviseEventModuleAdd(QString)));

    connect(ScriptVrmlNode::scriptPlugin->covise(), SIGNAL(eventModuleDel(QString)),
            this, SLOT(coviseEventModuleDel(QString)));

    connect(ScriptVrmlNode::scriptPlugin->covise(), SIGNAL(eventModuleDied(QString)),
            this, SLOT(coviseEventModuleDied(QString)));

    connect(ScriptVrmlNode::scriptPlugin->covise(), SIGNAL(eventOpenNet(QString)),
            this, SLOT(coviseEventOpenNet(QString)));

    connect(ScriptVrmlNode::scriptPlugin->covise(), SIGNAL(eventOpenNetDone(QString)),
            this, SLOT(coviseEventOpenNetDone(QString)));

    connect(ScriptVrmlNode::scriptPlugin->covise(), SIGNAL(eventQuit()),
            this, SLOT(coviseEventQuit()));

    connect(ScriptVrmlNode::scriptPlugin->covise(), SIGNAL(eventParameterChanged(QString, QString, QString)),
            this, SLOT(coviseEventParameterChanged(QString, QString, QString)));
}

ScriptVrmlNode::~ScriptVrmlNode()
{
}

vrml::VrmlNode *ScriptVrmlNode::cloneMe() const
{
    ScriptVrmlNode *node = new ScriptVrmlNode(this->scene());
    node->d_command = this->d_command;
    return node;
}

std::ostream &ScriptVrmlNode::printFields(std::ostream &os, int indent)
{
    if (!this->d_command.get())
        PRINT_FIELD(command);
    return os;
}

void ScriptVrmlNode::setField(const char *fieldName,
                              const vrml::VrmlField &fieldValue)
{
    if
        TRY_FIELD(command, SFString)
    else
        vrml::VrmlNodeChild::setField(fieldName, fieldValue);

    if (strcmp(fieldName, "command") == 0)
    {
        executeCommand(d_command.get());
    }
}

const VrmlField *ScriptVrmlNode::getField(const char *fieldName)
{
    if (strcmp(fieldName, "command") == 0)
        return &d_command;
    else
        std::cout << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << VrmlNode::name() << "." << fieldName << std::endl;
    return 0;
}

void ScriptVrmlNode::eventIn(double timeStamp,
                             const char *eventName,
                             const vrml::VrmlField *fieldValue)
{
    if (strcmp(eventName, "command") == 0)
    {
        setField(eventName, *fieldValue);
    }
    // Check exposedFields
    else
    {
        vrml::VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }
}

void ScriptVrmlNode::executeCommand(const char *command)
{
    ScriptVrmlNode::scriptPlugin->evaluate(command);
}

void ScriptVrmlNode::init(ScriptPlugin *plugin)
{
    if (ScriptVrmlNode::scriptPlugin == 0)
    {
        ScriptVrmlNode::scriptPlugin = plugin;
        vrml::VrmlNamespace::addBuiltIn(ScriptVrmlNode::defineType());
    }
}

void ScriptVrmlNode::coviseEventLink(const QString &fromModuleID, const QString &toModuleID)
{
    const char *out[2];
    QByteArray fm = fromModuleID.toLocal8Bit();
    QByteArray tm = toModuleID.toLocal8Bit();
    out[0] = fm.data();
    out[1] = tm.data();
    vrml::VrmlMFString d_out(2, out);
    eventOut(cover->frameTime(), "coviseEventLink", d_out);
}

void ScriptVrmlNode::coviseEventModuleAdd(const QString &moduleID)
{
    vrml::VrmlSFString d_out(moduleID.toLocal8Bit().data());
    eventOut(cover->frameTime(), "coviseEventModuleAdd", d_out);
}

void ScriptVrmlNode::coviseEventModuleDel(const QString &moduleID)
{
    vrml::VrmlSFString d_out(moduleID.toLocal8Bit().data());
    eventOut(cover->frameTime(), "coviseEventModuleDel", d_out);
}

void ScriptVrmlNode::coviseEventModuleDied(const QString &moduleID)
{
    vrml::VrmlSFString d_out(moduleID.toLocal8Bit().data());
    eventOut(cover->frameTime(), "coviseEventModuleDied", d_out);
}

void ScriptVrmlNode::coviseEventParameterChanged(const QString &moduleID,
                                                 const QString &name, const QString &value)
{
    const char *out[3];
    QByteArray mid = moduleID.toLocal8Bit();
    QByteArray nm = name.toLocal8Bit();
    QByteArray val = value.toLocal8Bit();
    out[0] = mid.data();
    out[1] = nm.data();
    out[2] = val.data();
    vrml::VrmlMFString d_out(3, out);
    eventOut(cover->frameTime(), "coviseEventParameterChanged", d_out);
}

void ScriptVrmlNode::coviseEventOpenNet(const QString &mapname)
{
    vrml::VrmlSFString d_out(mapname.toLocal8Bit().data());
    eventOut(cover->frameTime(), "coviseEventOpenNet", d_out);
}

void ScriptVrmlNode::coviseEventOpenNetDone(const QString &mapname)
{
    vrml::VrmlSFString d_out(mapname.toLocal8Bit().data());
    eventOut(cover->frameTime(), "coviseEventOpenNetDone", d_out);
}

void ScriptVrmlNode::coviseEventQuit()
{
    vrml::VrmlSFString d_out;
    eventOut(cover->frameTime(), "coviseEventQuit", d_out);
}
