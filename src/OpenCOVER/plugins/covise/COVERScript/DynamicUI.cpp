/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                            (C)2011 HLRS  **
 **                                                                          **
 ** Description: Dynamic UI plugin                                           **
 **                                                                          **
 **                                                                          **
 ** Author: Andreas Kopecki                                                  **
 **                                                                          **
 ** History:  								     **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "DynamicUI.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRTui.h>
#include <cover/coVRMSController.h>

#include "CoviseEventMessageSerialiser.h"

#include <PluginUtil/PluginMessageTypes.h>

#include <wslib/WSCoviseStub.h>
#include <wslib/WSMap.h>
#include <wslib/WSModule.h>

DynamicUI::DynamicUI(ScriptEngineProvider *plugin, ScriptWsCovise *covise)
    : QObject(0)
    , loading(false)
    , client(covise)
    , provider(plugin)
{
    this->supportedReadFileExtensions.push_back("ui");
}

DynamicUI::~DynamicUI()
{
}

osg::Node *DynamicUI::load(const std::string &location, osg::Group *group)
{
    this->loading = true;

    QString modelName = QString::fromStdString(location).section("/", -1);
    coTUIUITab *tab = new coTUIUITab(modelName.toStdString(), coVRTui::instance()->mainFolder->getID());
    tab->setPos(0, 0);

    if (!tab->loadUIFile(location))
    {
        delete tab;
        this->loading = false;
        return 0;
    }

    connect(tab, SIGNAL(tabletUICommand(QString, QString)), this, SLOT(tabletUICommand(QString, QString)));

    QList<covise::WSModule *> modules = client->getClient()->getMap()->getModules();
    foreach (covise::WSModule *module, modules)
    {
        covise::covise__ModuleAddEvent event(module->getSerialisable());
        QDomDocument serialisedEvent;
        serialisedEvent.appendChild(CoviseEventMessageSerialiser::serialise(&event, serialisedEvent));
        if (serialisedEvent.hasChildNodes())
        {
            //std::cerr << "DynamicUI info: " << qPrintable(serialisedEvent.toString()) << std::endl;
            tab->sendEvent("de.hlrs.covise", serialisedEvent.toString());
        }
    }

    osg::Group *dummyNode = new osg::Group();
    this->uiTabs[dummyNode] = tab;
    this->loading = false;
    return dummyNode;
}

osg::Node *DynamicUI::getLoaded()
{
    assert(0);
    return 0;
}

bool DynamicUI::unload(osg::Node *node)
{
    osg::Group *group = dynamic_cast<osg::Group *>(node);
    if (group && uiTabs.contains(group))
    {
        delete uiTabs.take(group);
        return true;
    }
    else
    {
        return false;
    }
}

void DynamicUI::tabletUICommand(const QString &target, const QString &c)
{
    //std::cerr << "DynamicUI::tabletUICommand info: received command for " << qPrintable(target) << ": " << qPrintable(c) << std::endl;

    QDomDocument commandDoc;
    if (!commandDoc.setContent(c, true))
    {
        std::cerr << "DynamicUI::tabletUICommand err: received invalid command: " << qPrintable(c) << std::endl;
        return;
    }

    QDomElement commandElement = commandDoc.documentElement();
    QString command = commandElement.tagName();

    if (target == "de.hlrs.covise")
    {
        if (command == "setParameter")
        {
            QString moduleID = commandElement.elementsByTagName("moduleID").item(0).toElement().text();
            QString parameter = commandElement.elementsByTagName("name").item(0).toElement().text();
            QString value = commandElement.elementsByTagName("value").item(0).toElement().text();
            //std::cerr << "DynamicUI::tabletUICommand info: setParameter(" << qPrintable(moduleID)
            //          << ", " << qPrintable(parameter) << ", " << qPrintable(value) << ")" << std::endl;
            if (opencover::coVRMSController::instance()->isMaster())
                client->getClient()->setParameterFromString(moduleID, parameter, value);
        }
        else if (command == "execute")
        {
            QString moduleID = commandElement.elementsByTagName("moduleID").item(0).toElement().text();
            //std::cerr << "DynamicUI::tabletUICommand info: execute(" << qPrintable(moduleID) << ")" << std::endl;
            if (opencover::coVRMSController::instance()->isMaster())
                client->getClient()->executeModule(moduleID);
        }
        else if (command == "executeNet")
        {
            //std::cerr << "DynamicUI::tabletUICommand info: executeNet()" << std::endl;
            if (opencover::coVRMSController::instance()->isMaster())
                client->getClient()->executeNet();
        }
    }
    else if (target == "de.hlrs.opencover")
    {
        if (command == "evaluateScript")
        {
            QString script = commandElement.text();
            this->provider->engine().evaluate(script);
        }
    }
}

void DynamicUI::preFrame()
{
    if (!opencover::coVRMSController::instance()->isMaster())
        return;

    for (covise::covise__Event *event = client->getClient()->takeEvent(); event != 0; event = client->getClient()->takeEvent())
    {
        QDomDocument serialisedEvent;
        serialisedEvent.appendChild(CoviseEventMessageSerialiser::serialise(event, serialisedEvent));
        if (serialisedEvent.hasChildNodes())
        {
            //std::cerr << "DynamicUI info: " << qPrintable(serialisedEvent.toString()) << std::endl;
            foreach (coTUIUITab *tab, this->uiTabs)
            {
                tab->sendEvent("de.hlrs.covise", serialisedEvent.toString());
            }
        }
        delete event;
    }
}
