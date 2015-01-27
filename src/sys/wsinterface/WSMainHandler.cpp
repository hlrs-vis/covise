/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "WSMainHandler.h"

#include <QDebug>
#include "WSMessageHandler.h"
#include "WSEventManager.h"
#include "WSLink.h"
#include "WSTools.h"

#include <covise/covise_msg.h>
#include <covise/covise_appproc.h>

using namespace covise;

covise::WSMainHandler *covise::WSMainHandler::singleton = 0;

covise::WSMainHandler *covise::WSMainHandler::instance()
{
    if (!singleton)
        singleton = new WSMainHandler();
    return singleton;
}

covise::WSMainHandler::WSMainHandler()
{
    this->map = new WSMap();
    this->eventManager = new WSEventManager();
}

covise::WSMainHandler::~WSMainHandler()
{
    delete this->map;
    delete this->eventManager;
}

QUuid covise::WSMainHandler::addEventListener()
{
    QUuid uuid = QUuid::createUuid();
    this->eventManager->addEventListener(uuid);
    //cerr << "WSMainHandler::addEventListener info: added listener " << qPrintable(uuid.toString()) << endl;
    return uuid;
}

void covise::WSMainHandler::removeEventListener(const QString &uuidString)
{
    //cerr << "WSMessageHandler::removeEventListener info: removing endpoint " << qPrintable(uuidString) << endl;
    QUuid uuid(uuidString);
    this->eventManager->removeEventListener(uuid);
}

covise::covise__Event *covise::WSMainHandler::consumeEvent(const QString &uuid, int timeout)
{
    return this->eventManager->consumeEvent(uuid, timeout);
}

void covise::WSMainHandler::postEvent(const covise::covise__Event *event)
{
    this->eventManager->postEvent(event);
}

void covise::WSMainHandler::setParameterFromString(const QString &moduleID, const QString &parameter, const QString &value)
{

    QStringList buffer;
    WSModule *m = getMap()->getModule(moduleID);
    if (m != 0)
    {
        WSParameter *param = m->getParameter(parameter);
        std::cerr << qPrintable(parameter) << std::endl;

        if (param != 0)
        {
            param->blockSignals(true);
            bool changed = WSTools::setParameterFromString(param, value);
            param->blockSignals(false);
            if (changed)
            {
                QString pType = param->getType();
                if (pType == "FileBrowser")
                    pType = "Browser";
                std::cerr << qPrintable(buffer.join(" ")) << std::endl;
                buffer << "PARAM" << m->getName() << m->getInstance() << m->getHost()
                       << param->getName() << pType << param->toCoviseString();
                WSMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, buffer.join("\n"));
            }
        }
    }
}

void covise::WSMainHandler::setParameter(const QString &moduleID, covise__Parameter *parameter)
{
    if (parameter == 0)
        return;

    QStringList buffer;
    QString pType = QString::fromStdString(parameter->type);

    if (parameter->type == "FileBrowser")
        pType = "Browser";

    WSModule *m = getMap()->getModule(moduleID);
    if (m != 0)
    {
        WSParameter *param = m->getParameter(QString::fromStdString(parameter->name));
        if (param != 0)
        {
            param->blockSignals(true);
            bool changed = param->setValueFromSerialisable(parameter);
            param->blockSignals(false);

            if (changed)
            {
                buffer << "PARAM" << m->getName() << m->getInstance() << m->getHost()
                       << param->getName() << pType << param->toCoviseString();
                WSMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, buffer.join("\n"));
            }
        }
    }
}

covise::WSModule *covise::WSMainHandler::getModule(const QString &name, const QString &host) const
{
    if (!this->availableModules.contains(host))
    {
        std::cerr << "WSMainHandler::getModule err: host " << qPrintable(host) << " is not in the host list" << std::endl;
        return 0;
    }

    foreach (WSModule *module, this->availableModules[host])
        if (module->getName() == name)
            return module;

    std::cerr << "WSMainHandler::getModule err: module " << qPrintable(name) << " not available on " << qPrintable(host) << std::endl;
    return 0;
}

QString covise::WSMainHandler::removeHost(const QString &inName)
{
    if (!this->availableModules.empty())
    {
        QLinkedList<WSModule *> modules = this->availableModules.take(inName);
        foreach (WSModule *module, modules)
            delete module;
        qDebug() << "WSMainHandler::addModule info: host" << inName << "removed.";
    }
    else
        qDebug() << "WSMainHandler::addModule info: host" << inName << "not in list.";

    return inName;
}

covise::WSModule *covise::WSMainHandler::addModule(const QString &inName, const QString &inCategory, const QString &inHost)
{
    //qDebug() << "WSMainHandler::addModule info: adding module" << inCategory << "/" << inName << " on " << inHost;
    WSModule *module = new WSModule(inName, inCategory, inHost);
    this->availableModules[inHost].push_back(module);

    //createHostList(availableModules);

    //qDebug () << " Module " << inName << " added.";
    //printList(this->availableModules);
    return module;
}

void covise::WSMainHandler::deleteModule(const QString &moduleID)
{
    WSModule *module = getMap()->takeModule(moduleID);
    if (module != 0)
        WSMessageHandler::instance()->deleteModule(module);
    delete module;
}

void covise::WSMainHandler::executeModule(const QString &moduleID)
{
    QStringList buffer;
    WSModule *module = getMap()->getModule(moduleID);
    if (module != 0)
    {
        buffer << "EXEC" << module->getName() << module->getInstance() << module->getHost();
        WSMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, buffer.join("\n"));
    }
}

void covise::WSMainHandler::instantiateModule(const QString &module, const QString &host, int x, int y)
{
    QStringList buffer;
    buffer << "INIT" << module << "-1" << host << QString::number(x) << QString::number(y);
    WSMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, buffer.join("\n"));
}

void covise::WSMainHandler::link(const QString &fromModule, const QString &fromPort, const QString &toModule, const QString &toPort)
{
    const WSModule *from = getMap()->getModule(fromModule);
    const WSModule *to = getMap()->getModule(toModule);
    QStringList buffer;
    buffer << "OBJCONN";
    buffer << from->getName() << from->getInstance() << from->getHost() << fromPort;
    buffer << to->getName() << to->getInstance() << to->getHost() << toPort;
    WSMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, buffer.join("\n"));
}

void covise::WSMainHandler::unlink(const QString &linkID)
{
    //std::cerr << "WSMainHandler::unlink info: unlinking " << qPrintable(linkID) << std::endl;
    covise::WSLink *link = this->map->getLink(linkID);
    if (link)
        WSMessageHandler::instance()->deleteLink(link);
    else
        std::cerr << "WSMainHandler::unlink warn: link " << qPrintable(linkID) << " not found" << std::endl;
}

QList<covise::WSLink *> covise::WSMainHandler::getLinks() const
{
    return getMap()->getLinks();
}

WSMap *covise::WSMainHandler::setMap(WSMap *inMap)
{

    if (inMap != this->map)
    {
        delete this->map;
        this->map = inMap;
    }

    return this->map;
}

void covise::WSMainHandler::printList(QMap<QString, QLinkedList<WSModule *> > inMap)
{
    qDebug() << "---Available modules-----------------";
    QMapIterator<QString, QLinkedList<WSModule *> > iterM(inMap);
    QLinkedListIterator<WSModule *> iterLL(availableModules[iterM.key()]);
    while (iterM.hasNext())
    {
        iterM.next();
        qDebug() << " Available module on host: " << iterM.key();
        while (iterLL.hasNext())
        {
            iterLL.next();
            qDebug() << " Module: "
                     << iterLL.next()->getName() << " "
                     << iterLL.next()->getHost() << " "
                     << iterLL.next()->getCategory() << " "
                     << iterLL.next()->getDescription();
        }
    }
}

// EOF
