/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SCRIPTVRMLNODE_H
#define SCRIPTVRMLNODE_H

#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlSFString.h>

#include <QObject>

class ScriptPlugin;

class ScriptVrmlNode : public QObject, public vrml::VrmlNodeChild
{
    Q_OBJECT

public:
    ScriptVrmlNode(vrml::VrmlScene *scene = 0);
    ScriptVrmlNode(const ScriptVrmlNode &other);

    static void initFields(vrml::VrmlNodeChild *node, vrml::VrmlNodeType *t);
    static const char *name();

    void eventIn(double timeStamp, const char *eventName,
                 const vrml::VrmlField *fieldValue);

    //virtual void render(Viewer *);

    static void init(ScriptPlugin *plugin);

public slots:
    void coviseEventLink(const QString &fromModuleID, const QString &toModuleID);
    void coviseEventModuleAdd(const QString &moduleID);
    void coviseEventModuleDel(const QString &moduleID);
    void coviseEventModuleDied(const QString &moduleID);
    void coviseEventParameterChanged(const QString &moduleID, const QString &name, const QString &value);
    void coviseEventOpenNet(const QString &mapname);
    void coviseEventOpenNetDone(const QString &mapname);
    void coviseEventQuit();

private:
    vrml::VrmlSFString d_command;

    static ScriptPlugin *scriptPlugin;

    void executeCommand(const char *command);
};

#endif // SCRIPTVRMLNODE_H
