/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SceneObjectManager.h"

#include <iostream>
#include <QFile>

#include "SceneEditor.h"
#include "Asset.h"
#include "AssetCreator.h"
#include "RoomCreator.h"
#include "ShapeCreator.h"
#include "GroundCreator.h"
#include "LightCreator.h"
#include "WindowCreator.h"

#include <cover/coVRPluginSupport.h>

#include <appl/RenderInterface.h>
#include <cover/coVRMSController.h>
#include <covise/covise_appproc.h>

#include <grmsg/coGRObjDelMsg.h>

SceneObjectManager *SceneObjectManager::instance()
{
    static SceneObjectManager *singleton = NULL;
    if (!singleton)
        singleton = new SceneObjectManager;
    return singleton;
}

SceneObjectManager::SceneObjectManager()
{
    _creators.insert(std::pair<std::string, SceneObjectCreator *>("sceneobject.asset", new AssetCreator()));
    _creators.insert(std::pair<std::string, SceneObjectCreator *>("sceneobject.room", new RoomCreator()));
    _creators.insert(std::pair<std::string, SceneObjectCreator *>("sceneobject.shape", new ShapeCreator()));
    _creators.insert(std::pair<std::string, SceneObjectCreator *>("sceneobject.ground", new GroundCreator()));
    _creators.insert(std::pair<std::string, SceneObjectCreator *>("sceneobject.light", new LightCreator()));
    _creators.insert(std::pair<std::string, SceneObjectCreator *>("sceneobject.window", new WindowCreator()));

    _resourceDir = ""; //QDir::currentPath().toStdString();
}

SceneObjectManager::~SceneObjectManager()
{
}

// Note: This function can be called with relative or absolute filenames.
//       If a relative filename cannot be found it will be searched for in the resource directory (if given).
//       The working directory is updated to the filenames directory before processing the Dom tree and reset afterwards.
//       (If an coxml is included, this function is called again and therefore the working directory will be updated again.)
//
//       During the process of including coinc files, the working directory remains unchanged (so %ROOT% can be used).
SceneObject *SceneObjectManager::createSceneObjectFromCoxmlFile(std::string filename)
{
    // normalized filename is potentially created from resource directory or current path
    std::string normalizedFilename = filename;

    if (!QFile::exists(normalizedFilename.c_str()))
    {
        // if filename is relative, search in resource directory if given
        QDir d(normalizedFilename.c_str());
        if (d.isRelative() && _resourceDir != "")
        {
            normalizedFilename = _resourceDir + "/" + filename;
            if (!QFile::exists(normalizedFilename.c_str()))
            {
                // relative file not found in resource directory
                std::cerr << "Error: Cannot find file " << normalizedFilename << std::endl;

                return NULL;
            }
        }
        else
        {
            // absolute file not found
            std::cerr << "Error: Cannot find file " << normalizedFilename << std::endl;

            return NULL;
        }
    }

    QFile file(normalizedFilename.c_str());
    if (!file.open(QFile::ReadOnly | QFile::Text))
    {
        std::cerr << "Error: Cannot read file " << normalizedFilename
                  << ": " << file.errorString().toStdString()
                  << std::endl;

        return NULL;
    }

    QString errorStr;
    int errorColumn;
    int errorLine;

    QDomDocument doc;
    if (!doc.setContent(&file, false, &errorStr, &errorLine, &errorColumn))
    {
        std::cerr << "Error: Parse error at line " << errorLine << ", "
                  << "column " << errorColumn << ": "
                  << errorStr.toStdString() << std::endl;
        return NULL;
    }

    QDomElement root = doc.documentElement();

    // is coxml?
    if (root.tagName() != "coxml")
    {
        std::cerr << "Error: Not a valid coxml file (coxml missing)" << std::endl;
        return NULL;
    }

    // include all files and resolve relative paths (%ROOT%, %LOCAL%)
    if (!prepareDomTree(&root, NULL, QFileInfo(normalizedFilename.c_str()).dir().absolutePath()))
    {
        return NULL;
    }

    SceneObject *so = createSceneObjectFromCoxmlElement(&root);
    if (!so)
    {
        return NULL;
    }

    _latestSceneObject = so;
    return so;
}

bool SceneObjectManager::prepareDomTree(QDomElement *element, QDomElement *includeParams, QString currentDir)
{
    // check element for variables in attributes
    QDomNamedNodeMap attributes = element->attributes();
    for (int i = 0; i < attributes.count(); ++i)
    {
        QString value = attributes.item(i).toAttr().value();
        if (prepareQString(value, includeParams, currentDir))
        {
            attributes.item(i).toAttr().setValue(value);
        }
    }

    // check element for variables in text
    if ((!element->firstChild().isNull()) && (element->firstChild().isText()))
    {
        QString data = element->firstChild().toText().data();
        if (prepareQString(data, includeParams, currentDir))
        {
            element->firstChild().toText().setData(data);
        }
    }

    // prepare children
    QDomElement child = element->firstChildElement();
    while (!child.isNull())
    {

        // if include, add elements
        if (child.tagName() == "include")
        {

            QString incFilename = child.text();
            prepareQString(incFilename, includeParams, currentDir);

            QFile incFile(incFilename);
            if (!incFile.open(QFile::ReadOnly | QFile::Text))
            {
                std::cerr << "Error: Cannot read include file " << incFilename.toStdString()
                          << ": " << incFile.errorString().toStdString()
                          << std::endl;
                return false;
            }

            QString errorStr;
            int errorColumn;
            int errorLine;

            QDomDocument incDoc;
            if (!incDoc.setContent(&incFile, false, &errorStr, &errorLine, &errorColumn))
            {
                std::cerr << "Error: Parse error at line " << errorLine << ", "
                          << "column " << errorColumn << ": "
                          << errorStr.toStdString() << std::endl;
                return false;
            }
            QDomElement incRootElem = incDoc.documentElement();

            // If we already are inside an include (and have includeParams):
            // Add attributes from existing include to the new include QDomElement.
            if (includeParams != NULL)
            {
                QDomNamedNodeMap attributes = includeParams->attributes();
                for (int i = 0; i < attributes.length(); ++i)
                {
                    if (!attributes.item(i).isAttr())
                        continue;
                    if (!child.hasAttribute(attributes.item(i).toAttr().name()))
                        child.setAttributeNode(attributes.item(i).toAttr());
                }
            }

            prepareDomTree(&incRootElem, &child, QFileInfo(incFilename).dir().absolutePath());

            QDomNode incRootNode = incRootElem;
            if (incRootElem.tagName() == "coinc")
            {
                // move all children to a DocumentFragment -> if child is replaced with DocumentFragment, the children will be used instead
                QDomDocumentFragment newRoot = incDoc.createDocumentFragment();
                while (!incRootNode.firstChild().isNull())
                {
                    newRoot.appendChild(incRootNode.firstChild());
                }
                incRootNode = newRoot;
            }

            element->ownerDocument().importNode(incRootNode, true);
            QDomNode incPos = child;
            child = child.nextSiblingElement();

            element->replaceChild(incRootNode, incPos);
        }
        else
        {

            prepareDomTree(&child, includeParams, currentDir);
            child = child.nextSiblingElement();
        }
    }

    return true;
}

bool SceneObjectManager::prepareQString(QString &qs, QDomElement *includeParams, QString currentDir)
{
    QString old = qs;

    qs.replace("%ROOT%", _resourceDir.c_str());
    if ((_resourceDir == "") && (qs != old))
    {
        std::cout << "WARNING: Encountered %ROOT%-Tag but no ResourceDirectory given." << std::endl;
    }

    qs.replace("%LOCAL%", currentDir);

    if (includeParams)
    {
        int startIndex = qs.indexOf("%");
        if (startIndex > -1)
        {
            int endIndex = qs.indexOf("%", startIndex + 1);
            if (endIndex > startIndex)
            {
                QString paramName = qs.mid(startIndex + 1, endIndex - startIndex - 1);
                QString paramValue = includeParams->attribute(paramName, "");
                qs.replace("%" + paramName + "%", paramValue);
                prepareQString(qs, includeParams, currentDir);
            }
        }
    }

    return (old != qs);
}

void SceneObjectManager::setResourceDirectory(std::string dir)
{
    _resourceDir = dir;
}

std::string SceneObjectManager::getResourceDirectory()
{
    return _resourceDir;
}

SceneObject *SceneObjectManager::createSceneObjectFromCoxmlElement(QDomElement *coxmlRoot)
{
    // class element found?
    QDomElement classElem = coxmlRoot->firstChildElement("class");
    if (classElem.isNull())
    {
        std::cerr << "Error: Class missing in coxml section" << std::endl;
        return NULL;
    }

    std::string classStr = classElem.attribute("value").toStdString();

    // search creator
    std::map<std::string, SceneObjectCreator *>::iterator iter = _creators.find(classStr);
    if (iter == _creators.end())
    {
        std::cerr << "Error: No creator found for: " << classStr << std::endl;
        return NULL;
    }
    SceneObjectCreator *soc = iter->second;

    combineBehaviors(coxmlRoot);

    // create scene object
    SceneObject *so = soc->createFromXML(coxmlRoot);

    if (so)
    {
        _sceneObjects.push_back(so);

        // parse for more coxml entries (child SceneObjects within the dom)
        QDomElement childCoxml = coxmlRoot->firstChildElement("coxml");
        while (!childCoxml.isNull())
        {
            SceneObject *childSo = createSceneObjectFromCoxmlElement(&childCoxml);
            if (childSo)
            {
                so->addChild(childSo);
            }
            childCoxml = childCoxml.nextSiblingElement("coxml");
        }
    }

    return so;
}

void SceneObjectManager::combineBehaviors(QDomElement *coxmlRoot)
{
    QDomElement firstBehaviorRoot = coxmlRoot->firstChildElement("behavior");
    if (firstBehaviorRoot.isNull())
    {
        return;
    }

    QDomElement behaveRoot = firstBehaviorRoot.nextSiblingElement("behavior");
    while (!behaveRoot.isNull())
    {

        // get the next sibling first because the behaveRoot will be deleted/moved
        QDomElement nextBehaveRoot = behaveRoot.nextSiblingElement("behavior");

        // process the current behavior section
        QDomElement b = behaveRoot.firstChildElement();
        while (!b.isNull())
        {

            // search same behavior in firstBehaviorRoot
            QDomElement firstB = firstBehaviorRoot.firstChildElement(b.tagName());

            if (firstB.isNull())
            {
                // if not in firstBehaviorRoot, we move the entire behavior
                firstBehaviorRoot.appendChild(b);
            }
            else
            {
                // move elements of behavior to firstBehaviorRoot
                while (!b.firstChildElement().isNull())
                {
                    firstB.appendChild(b.firstChildElement());
                }
                // delete behavior
                behaveRoot.removeChild(b);
                b.clear();
            }

            b = behaveRoot.firstChildElement();
        }

        coxmlRoot->removeChild(behaveRoot);
        behaveRoot.clear();
        behaveRoot = nextBehaveRoot;
    }
}

int SceneObjectManager::requestDeleteSceneObject(SceneObject *so)
{
    // send to GUI
    if (opencover::coVRMSController::instance()->isMaster())
    {
        grmsg::coGRObjDelMsg selectMsg(so->getCoviseKey().c_str(), 1);
        covise::Message grmsg{ covise::COVISE_MESSAGE_UI, covise::DataHandle{(char*)selectMsg.c_str(),strlen(selectMsg.c_str()) + 1 , false} };
        opencover::cover->sendVrbMessage(&grmsg);
    }
    return 1;
}

// received delete from vr-prepare
int SceneObjectManager::deletingSceneObject(SceneObject *so)
{
    // deselect in plugin
    SceneEditor::plugin->deselect(so);

    for (std::vector<SceneObject *>::iterator it = _sceneObjects.begin(); it < _sceneObjects.end(); it++)
    {
        if ((*it) == so)
        {
            // manually detach/remove/delete all behaviors here, because SceneObjectManager handles behaviors
            (*it)->deleteAllBehaviors();

            SceneObject *parent;
            parent = so->getParent();
            if (parent)
            {
                parent->removeChild(so);
            }

            delete (*it);
            _sceneObjects.erase(it);
            _latestSceneObject = NULL;

            return 1;
        }
    }

    return -1;
}

SceneObject *SceneObjectManager::getLatestSceneObject()
{
    return _latestSceneObject;
}

SceneObject *SceneObjectManager::findSceneObject(osg::Node *n)
{
    if (!n)
    {
        return NULL;
    }

    for (int i = 0; i < _sceneObjects.size(); i++)
    {
        SceneObject *so = _sceneObjects[i];
        osg::Node *soNode = so->getRootNode();

        osg::NodePathList npl = n->getParentalNodePaths(soNode);
        for (osg::NodePathList::iterator npl_iter = npl.begin(); npl_iter != npl.end(); npl_iter++)
        {
            osg::NodePath np = (*npl_iter);
            for (osg::NodePath::iterator np_iter = np.begin(); np_iter != np.end(); np_iter++)
            {
                osg::Node *node = (*np_iter);
                if (node == soNode)
                {
                    return _sceneObjects[i];
                }
            }
        }
    }

    return NULL;
}

SceneObject *SceneObjectManager::findSceneObject(const char *covise_key)
{
    // find scene object with covise key
    for (int i = 0; i < _sceneObjects.size(); i++)
    {
        if (_sceneObjects[i]->getCoviseKey().compare(covise_key) == 0)
            return _sceneObjects[i];
    }

    return NULL;
}

std::vector<SceneObject *> SceneObjectManager::getSceneObjectsOfType(SceneObjectTypes::Type t)
{
    std::vector<SceneObject *> so;
    std::vector<SceneObject *>::iterator it;
    for (it = _sceneObjects.begin(); it != _sceneObjects.end(); it++)
    {
        if ((*it)->getType() == t)
        {
            so.push_back(*it);
        }
    }
    return so;
}

std::vector<SceneObject *> SceneObjectManager::getSceneObjects()
{
    return _sceneObjects;
}

Room *SceneObjectManager::getRoom()
{
    std::vector<SceneObject *>::iterator it;
    for (it = _sceneObjects.begin(); it != _sceneObjects.end(); it++)
    {
        if ((*it)->getType() == SceneObjectTypes::ROOM)
        {
            return (Room *)(*it);
        }
    }
    return NULL;
}

int SceneObjectManager::broadcastEvent(Event *e)
{
    for (int i = 0; i < _sceneObjects.size(); i++)
    {
        _sceneObjects[i]->receiveEvent(e);
    }

    return 1;
}
