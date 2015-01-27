/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "WSMap.h"
#include "WSLink.h"

#include <QDebug>
#include <QMutex>

Q_DECLARE_METATYPE(covise::WSMap *)

covise::WSMap::WSMap()
{
}

covise::WSMap::~WSMap()
{
    foreach (WSModule *module, this->runningModules)
        delete module;
}

covise::WSModule *covise::WSMap::addModule(const covise::WSModule *inModule, const QString &instance, const QString &host)
{
    covise::WSModule *module = new covise::WSModule(inModule->getSerialisable());

    module->instantiate(host, instance);

    QString moduleKeyName = makeKeyName(module->getName(), module->getInstance(), module->getHost());
    module->setID(moduleKeyName);
    this->runningModules.insert(moduleKeyName, module);

    emit moduleAdded(module);

    return module;
}

covise::WSModule *covise::WSMap::addModule(const covise::WSModule *inModule)
{
    covise::WSModule *module = new covise::WSModule(inModule->getSerialisable());

    QString moduleKeyName = makeKeyName(module->getName(), module->getInstance(), module->getHost());
    module->setID(moduleKeyName);
    this->runningModules.insert(moduleKeyName, module);

    emit moduleAdded(module);

    return module;
}

covise::WSModule *covise::WSMap::addModule(const covise::covise__Module &inModule)
{
    covise::WSModule *module = new covise::WSModule(inModule);

    QString moduleKeyName = makeKeyName(module->getName(), module->getInstance(), module->getHost());
    module->setID(moduleKeyName);
    this->runningModules.insert(moduleKeyName, module);

    emit moduleAdded(module);

    return module;
}

const QString &covise::WSMap::removeModule(const QString &moduleID)
{
    static QString nullString = QString::null;
    covise::WSModule *module = takeModule(moduleID);
    if (module != 0)
    {
#ifdef DEBUG
        std::cerr << "WSMap::removeModule info: module " << qPrintable(moduleID) << " removed" << std::endl;
#endif
        delete module;
        return moduleID;
    }
    else
    {
#ifdef DEBUG
        std::cerr << "WSMap::removeModule err: module " << qPrintable(moduleID) << " not found" << std::endl;
#endif
        return nullString;
    }
}

covise::WSModule *covise::WSMap::takeModule(const QString &moduleID)
{
    covise::WSModule *module = runningModules.take(moduleID);
    if (module != 0)
        emit moduleRemoved(moduleID);
    return module;
}

covise::WSModule *covise::WSMap::getModule(const QString &inName, const QString &inInstance, const QString &inHost) const
{
    return getModule(makeKeyName(inName, inInstance, inHost));
}

covise::WSModule *covise::WSMap::getModule(const QString &moduleKeyName) const
{
    if (this->runningModules.contains(moduleKeyName))
    {
        //std::cerr << "WSMap::getModule info: found module " << qPrintable(moduleKeyName) << std::endl;
        return this->runningModules[moduleKeyName];
    }
    else
    {
        //std::cerr << "WSMap::getModule info: module " << qPrintable(moduleKeyName) << " not found" << std::endl;
        return 0;
    }
}

covise::WSModule *covise::WSMap::getModuleByTitle(const QString &title) const
{
    foreach (covise::WSModule *module, this->runningModules)
    {
        if (module->getTitle() == title)
        {
            return module;
        }
    }

    return 0;
}

QList<covise::WSModule *> covise::WSMap::getModules() const
{
    return this->runningModules.values();
}

covise::WSLink *covise::WSMap::link(const QString &fromModule, const QString &fromPort, const QString &toModule, const QString &toPort)
{

    static QMutex lock;
    QMutexLocker locker(&lock);

    covise::WSModule *from = getModule(fromModule);
    covise::WSModule *to = getModule(toModule);
    if (from == 0 || to == 0)
    {
#ifdef DEBUG
        std::cerr << "WSMap::link err: unable to find module " << (from ? "" : qPrintable(fromModule)) << (to ? "" : qPrintable((from ? "" : " / ") + toModule)) << std::endl;
#endif
        return 0;
    }

    covise::WSPort *outPort = from->getOutputPort(fromPort);
    covise::WSPort *inPort = to->getInputPort(toPort);

    if (outPort == 0 || inPort == 0)
    {
#ifdef DEBUG
        std::cerr << "WSMap::link err: unable to find port " << (outPort ? "" : qPrintable(fromPort + " (" + from->getID() + ")")) << (inPort ? "" : qPrintable((outPort ? "" : " / ") + toPort + " (" + to->getID() + ")")) << std::endl;
        std::cerr << qPrintable(fromModule) << ":" << qPrintable(QStringList(from->getOutputPorts().keys()).join("|")) << std::endl;
        std::cerr << qPrintable(toModule) << ":" << qPrintable(QStringList(to->getInputPorts().keys()).join("|")) << std::endl;
#endif
        return 0;
    }

    // Prevent looping messages
    if (this->links.contains(covise::WSLink::makeID(outPort, inPort)))
        return 0;

    covise::WSLink *link = new covise::WSLink(outPort, inPort);

    connect(link, SIGNAL(deleted(QString)), this, SLOT(linkDestroyed(QString)));
    this->links.insert(link->getLinkID(), link);

#ifdef DEBUG
    std::cerr << "WSMap::link info: created link " << qPrintable(link->getLinkID()) << std::endl;
#endif

    return link;
}

bool covise::WSMap::unlink(const QString &fromModule, const QString &fromPort,
                           const QString &toModule, const QString &toPort)
{
    return unlink(covise::WSLink::makeID(fromModule, fromPort, toModule, toPort));
}

bool covise::WSMap::unlink(covise::WSLink *link)
{
    return unlink(link->getLinkID());
}

bool covise::WSMap::unlink(const QString &linkID)
{
    if (this->links.contains(linkID))
    {
        std::cerr << "WSMap::unlink info: removing link " << qPrintable(linkID) << std::endl;
        delete this->links.take(linkID);
        return true;
    }
    else
    {
        return false;
    }
}

covise::WSLink *covise::WSMap::getLink(const QString &linkID) const
{
    if (this->links.contains(linkID))
    {
        return this->links[linkID];
    }
    else
    {
        return 0;
    }
}

QList<covise::WSLink *> covise::WSMap::getLinks() const
{
    return this->links.values();
}

void covise::WSMap::linkDestroyed(const QString &linkID)
{
    if (this->links.contains(linkID))
        this->links.remove(linkID);
}

QString covise::WSMap::makeKeyName(const QString &inName, const QString &inInstance, const QString &inHost)
{
    QString moduleKeyName = inName + "_" + inInstance + "_" + inHost;

    //qDebug() << " New name of module: " << moduleKeyName;

    return moduleKeyName;
}

void covise::WSMap::printList(const QMap<QString, covise::WSModule *> &inMap)
{
    qDebug() << "--- Loaded modules -----------------";
    QMapIterator<QString, WSModule *> iterModule(inMap);
    while (iterModule.hasNext())
    {
        iterModule.next();
        qDebug() << " Loaded module on host: " << iterModule.key();
        qDebug() << "  " << iterModule.value()->getName()
                 << "  " << iterModule.value()->getHost()
                 << "  " << iterModule.value()->getCategory()
                 << "  " << iterModule.value()->getDescription();
        QMap<QString, WSParameter *> paraMap = iterModule.value()->getParameters();
        QMapIterator<QString, WSParameter *> iterParameter(paraMap);
        while (iterParameter.hasNext())
        {
            iterParameter.next();
            qDebug() << " Parameter name: " << iterParameter.key();
            qDebug() << "   " << iterParameter.value()->getName()
                     << "   " << iterParameter.value()->getType()
                     << "   " << iterParameter.value()->getDescription();
        }
        QMap<QString, WSPort *> portMapI = iterModule.value()->getInputPorts();
        QMapIterator<QString, WSPort *> iterInputPorts(portMapI);
        while (iterInputPorts.hasNext())
        {
            iterInputPorts.next();
            qDebug() << " Input port name:  " << iterInputPorts.key();
            qDebug() << "   " << iterInputPorts.value()->getName();
            QStringList dataTypesList = iterInputPorts.value()->getTypes();
            QStringListIterator iterDataTypes(dataTypesList);
            while (iterDataTypes.hasNext())
            {
                qDebug() << "  " << dataTypesList;
            }
        }
        QMap<QString, WSPort *> portMapO = iterModule.value()->getOutputPorts();
        QMapIterator<QString, WSPort *> iterOutputPorts(portMapO);
        while (iterOutputPorts.hasNext())
        {
            iterOutputPorts.next();
            qDebug() << " Output port name: " << iterOutputPorts.key();
            qDebug() << "   " << iterOutputPorts.value()->getName();
            QStringList dataTypesList = iterOutputPorts.value()->getTypes();
            QStringListIterator iterDataTypes(dataTypesList);
            while (iterDataTypes.hasNext())
            {
                qDebug() << "   " << dataTypesList;
            }
        }
    }
}

void covise::WSMap::setMapName(const QString &name)
{
    this->mapName = name;
    setObjectName(this->mapName);
}

// EOF
