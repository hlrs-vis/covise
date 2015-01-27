/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SCENE_OBJECT_MANAGER_H
#define SCENE_OBJECT_MANAGER_H

#include <osg/Node>
#include <map>
#include <QDomDocument>
#include <QDir>

#include "SceneObjectCreator.h"
#include "SceneObject.h"
#include "Room.h"
#include "Events/Event.h"

class SceneObjectManager
{
public:
    static SceneObjectManager *instance();
    virtual ~SceneObjectManager();

    SceneObject *createSceneObjectFromCoxmlFile(std::string filename);

    SceneObject *getLatestSceneObject();
    SceneObject *findSceneObject(osg::Node *n);
    SceneObject *findSceneObject(const char *covise_key);
    std::vector<SceneObject *> getSceneObjectsOfType(SceneObjectTypes::Type t);
    std::vector<SceneObject *> getSceneObjects();
    Room *getRoom();

    int requestDeleteSceneObject(SceneObject *so);
    int deletingSceneObject(SceneObject *so);

    int broadcastEvent(Event *e);

    void setResourceDirectory(std::string dir);
    std::string getResourceDirectory();

private:
    SceneObjectManager();

    SceneObject *createSceneObjectFromCoxmlElement(QDomElement *coxmlRoot);

    bool prepareDomTree(QDomElement *element, QDomElement *includeParams, QString currentDir);
    bool prepareQString(QString &qs, QDomElement *includeParams, QString currentDir);

    void combineBehaviors(QDomElement *coxmlRoot);

    std::vector<SceneObject *> _sceneObjects;
    std::map<std::string, SceneObjectCreator *> _creators;

    SceneObject *_latestSceneObject;

    std::string _resourceDir;
};

#endif
