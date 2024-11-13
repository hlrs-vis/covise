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

void ScriptVrmlNode::initFields(vrml::VrmlNodeChild *node, vrml::VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t);
    initFieldsHelper(node, t,
                     exposedField("command", node->d_command, [node](auto t){
                        executeCommand(t);
                     }));
    if(t)
    {
        t->addExposedField("command", vrml::VrmlField::SFSTRING);
        t->addEventOut("coviseEventLink", vrml::VrmlField::MFSTRING);
        t->addEventOut("coviseEventModuleAdd", vrml::VrmlField::SFSTRING);
        t->addEventOut("coviseEventModuleDel", vrml::VrmlField::SFSTRING);
        t->addEventOut("coviseEventModuleDied", vrml::VrmlField::SFSTRING);
        t->addEventOut("coviseEventParameterChanged", vrml::VrmlField::MFSTRING);
        t->addEventOut("coviseEventOpenNet", vrml::VrmlField::SFSTRING);
        t->addEventOut("coviseEventOpenNetDone", vrml::VrmlField::SFSTRING);
        t->addEventOut("coviseEventQuit", vrml::VrmlField::SFSTRING); // TODO How to define an event without parameter?

    }
}

ScriptVrmlNode::ScriptVrmlNode(vrml::VrmlScene *scene)
    : QObject(0)
    , vrml::VrmlNodeChild(scene, name())
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

ScriptVrmlNode::ScriptVrmlNode(const ScriptVrmlNode &other)
:ScriptVrmlNode(other.scene())
{
    d_command = other.d_command;
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
