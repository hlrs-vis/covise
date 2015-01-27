/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "WSCOVISEClient.h"

#include "WSCoviseStub.h"
#include "coviseCOVISEProxy.h"

#include "WSModule.h"
#include "WSMap.h"

#include "WSTools.h"

covise::WSCOVISEClient::WSCOVISEClient()
    : map(0)
    , attached(false)
    , keepRunning(true)
    , eventUUID("")
    , eventsAsSignal(false)
    , alsoQueueRaw(true)
    , readOnly(false)
    , inExecute(true)
{
    start();
}

covise::WSCOVISEClient::~WSCOVISEClient()
{
    this->keepRunning = false;
    msleep(50);
    terminate();
    if (this->attached)
        detach();
}

QList<covise::WSModule *> covise::WSCOVISEClient::getModules(const QString &host) const
{
    if (this->availableModules.contains(host))
        return this->availableModules[host];
    else
        return QList<covise::WSModule *>();
}

QStringList covise::WSCOVISEClient::getHosts() const
{
    return this->availableModules.keys();
}

covise::WSMap *covise::WSCOVISEClient::getMap() const
{
    return this->map;
}

bool covise::WSCOVISEClient::attach(const QString &endpoint)
{
    if (this->attached)
        return false;

    this->endpoint = endpoint;

    covise::COVISEProxy client;
    client.soap_endpoint = soap_strdup(&client, this->endpoint.toLocal8Bit().data());

    covise::_covise__listHosts listHostsRequest;
    covise::_covise__listHostsResponse listHostsResponse;

    if (client.listHosts(&listHostsRequest, &listHostsResponse) != SOAP_OK)
    {
        std::cerr << "WSCOVISEClient::attach err: cannot attach to COVISE" << std::endl;
        detach();
        return false;
    }

    for (std::vector<std::string>::iterator host = listHostsResponse.hosts.begin(); host != listHostsResponse.hosts.end(); ++host)
    {

        client.soap_endpoint = soap_strdup(&client, this->endpoint.toLocal8Bit().data());

        covise::_covise__listModules listModulesRequest;
        covise::_covise__listModulesResponse listModulesResponse;

        listModulesRequest.ipaddr = *host;

        if (client.listModules(&listModulesRequest, &listModulesResponse) != SOAP_OK)
        {
            std::cerr << "WSCOVISEClient::attach err: cannot access COVISE module list" << std::endl;
            detach();
            return false;
        }

        QList<WSModule *> modules;

        for (unsigned int ctr = 0; ctr < listModulesResponse.modules.size(); ++ctr)
        {
            covise::WSModule *module = new covise::WSModule(QString::fromStdString(listModulesResponse.modules[ctr].second),
                                                            QString::fromStdString(listModulesResponse.modules[ctr].first),
                                                            QString::fromStdString(*host));

            modules.append(module);
            connect(module, SIGNAL(parameterChanged(covise::WSParameter *)), this, SLOT(parameterChangeCB(covise::WSParameter *)));
        }

        this->availableModules[QString::fromStdString(*host)] = modules;
    }

    covise::_covise__getRunningModules rmRequest;
    covise::_covise__getRunningModulesResponse rmResponse;

    if (client.getRunningModules(&rmRequest, &rmResponse) != SOAP_OK)
    {
        std::cerr << "WSCOVISEClient::attach err: cannot access COVISE map" << std::endl;
        detach();
        return false;
    }

    delete this->map;
    this->map = new WSMap();

    for (std::vector<covise__Module>::iterator module = rmResponse.modules.begin(); module != rmResponse.modules.end(); ++module)
    {
        covise::WSModule *m = map->addModule(*module);
        connect(m, SIGNAL(parameterChanged(covise::WSParameter *)), this, SLOT(parameterChangeCB(covise::WSParameter *)));
    }

    covise::_covise__addEventListener aelRequest;
    covise::_covise__addEventListenerResponse aelResponse;

    if (client.addEventListener(&aelRequest, &aelResponse) != SOAP_OK)
    {
        std::cerr << "WSCOVISEClient::attach err: unable to register event listener" << std::endl;
        detach();
        return false;
    }

    this->eventUUID = QString::fromStdString(aelResponse.uuid);

    this->attached = true;
    return true;
}

bool covise::WSCOVISEClient::detach()
{
    clearAvailableModules();

    if (this->attached && this->eventUUID != "")
    {

        covise::COVISEProxy client;
        client.soap_endpoint = soap_strdup(&client, this->endpoint.toLocal8Bit().data());

        covise::_covise__removeEventListener request;
        covise::_covise__removeEventListenerResponse response;

        client.removeEventListener(&request, &response);
    }

    this->attached = false;
    this->eventUUID = "";
    this->endpoint = "";

    delete this->map;
    this->map = 0;

    return true;
}

covise::WSModule *covise::WSCOVISEClient::getModule(const QString &name, const QString &host) const
{
    if (!this->availableModules.contains(host))
    {
        std::cerr << "WSCOVISEClient::getModule err: host " << qPrintable(host) << " is not in the host list" << std::endl;
        return 0;
    }

    foreach (WSModule *module, this->availableModules[host])
        if (module->getName() == name)
            return module;

    std::cerr << "WSCOVISEClient::getModule err: module " << qPrintable(name) << " not available on " << qPrintable(host) << std::endl;
    return 0;
}

void covise::WSCOVISEClient::executeNet()
{
    if (!this->attached || this->readOnly)
        return;

    covise::COVISEProxy client;
    client.soap_endpoint = soap_strdup(&client, this->endpoint.toLocal8Bit().data());

    covise::_covise__executeNet request;
    covise::_covise__executeNetResponse response;

    client.executeNet(&request, &response);
}

void covise::WSCOVISEClient::executeModule(const QString &moduleID)
{
    if (!this->attached || this->readOnly)
        return;

    covise::COVISEProxy client;
    client.soap_endpoint = soap_strdup(&client, this->endpoint.toLocal8Bit().data());

    covise::_covise__executeModule request;
    covise::_covise__executeModuleResponse response;

    request.moduleID = moduleID.toStdString();

    client.executeModule(&request, &response);
}

void covise::WSCOVISEClient::setParameterFromString(const QString &moduleID, const QString &parameter, const QString &value)
{
    if (!this->attached || this->readOnly)
        return;

    covise::COVISEProxy client;
    client.soap_endpoint = soap_strdup(&client, this->endpoint.toLocal8Bit().data());

    covise::_covise__setParameterFromString request;
    covise::_covise__setParameterFromStringResponse response;

    request.moduleID = moduleID.toStdString();
    request.parameter = parameter.toStdString();
    request.value = value.toStdString();

    client.setParameterFromString(&request, &response);
}

QString covise::WSCOVISEClient::getParameterAsString(const QString &moduleID, const QString &parameter)
{
    if (!this->attached || this->readOnly)
        return "";

    covise::COVISEProxy client;
    client.soap_endpoint = soap_strdup(&client, this->endpoint.toLocal8Bit().data());

    covise::_covise__getParameterAsString request;
    covise::_covise__getParameterAsStringResponse response;

    request.moduleID = moduleID.toStdString();
    request.parameter = parameter.toStdString();

    client.getParameterAsString(&request, &response);

    return QString::fromStdString(response.value);
}

void covise::WSCOVISEClient::openNet(const QString &filename)
{
    if (!this->attached || this->readOnly)
        return;

    covise::COVISEProxy client;
    client.soap_endpoint = soap_strdup(&client, this->endpoint.toLocal8Bit().data());

    covise::_covise__openNet request;
    covise::_covise__openNetResponse response;

    request.filename = filename.toStdString();

    client.openNet(&request, &response);
}

void covise::WSCOVISEClient::quit()
{
    if (!this->attached || this->readOnly)
        return;

    covise::COVISEProxy client;
    client.soap_endpoint = soap_strdup(&client, this->endpoint.toLocal8Bit().data());

    covise::_covise__quit request;
    covise::_covise__quitResponse response;

    client.quit(&request, &response);

    this->attached = false;
}

void covise::WSCOVISEClient::instantiateModule(const QString &module, const QString &host)
{
    if (!this->attached || this->readOnly)
        return;

    covise::COVISEProxy client;
    client.soap_endpoint = soap_strdup(&client, this->endpoint.toLocal8Bit().data());

    covise::_covise__instantiateModule request;
    covise::_covise__instantiateModuleResponse response;

    request.module = module.toStdString();
    request.host = host.toStdString();

    client.instantiateModule(&request, &response);
}

void covise::WSCOVISEClient::deleteModule(const QString &moduleID)
{
    if (!this->attached || this->readOnly)
        return;

    covise::COVISEProxy client;
    client.soap_endpoint = soap_strdup(&client, this->endpoint.toLocal8Bit().data());

    covise::_covise__deleteModule request;
    covise::_covise__deleteModuleResponse response;

    request.moduleID = moduleID.toStdString();

    client.deleteModule(&request, &response);
}

void covise::WSCOVISEClient::link(const QString &fromModuleID, const QString &fromPort,
                                  const QString &toModuleID, const QString &toPort)
{
    if (!this->attached || this->readOnly)
        return;

    covise::COVISEProxy client;
    client.soap_endpoint = soap_strdup(&client, this->endpoint.toLocal8Bit().data());

    covise::_covise__link request;
    covise::_covise__linkResponse response;

    request.fromModule = fromModuleID.toStdString();
    request.fromPort = fromPort.toStdString();
    request.toModule = toModuleID.toStdString();
    request.toPort = toPort.toStdString();

    client.link(&request, &response);
}

void covise::WSCOVISEClient::unlink(const QString &linkID)
{
    if (!this->attached || this->readOnly)
        return;

    covise::COVISEProxy client;
    client.soap_endpoint = soap_strdup(&client, this->endpoint.toLocal8Bit().data());

    covise::_covise__unlink request;
    covise::_covise__unlinkResponse response;

    request.linkID = linkID.toStdString();

    //std::cerr << "WSCOVISEClient::unlink info: unlinking " << request.linkID << std::endl;

    client.unlink(&request, &response);
}

void covise::WSCOVISEClient::clearAvailableModules()
{
    foreach (QList<WSModule *> modules, this->availableModules)
    {
        foreach (WSModule *module, modules)
        {
            delete module;
        }
    }
    this->availableModules.clear();
}

covise::covise__Event *covise::WSCOVISEClient::takeEvent()
{
    QMutexLocker(&(this->eventQueueLock));
    if (this->eventQueue.empty())
        return 0;
    else
        return this->eventQueue.dequeue();
}

void covise::WSCOVISEClient::run()
{
    while (this->keepRunning)
    {
        if (this->attached && this->eventUUID != "")
        {
            covise::COVISEProxy client;
            client.soap_endpoint = soap_strdup(&client, this->endpoint.toLocal8Bit().data());

            covise::_covise__getEvent request;
            covise::_covise__getEventResponse response;

            request.uuid = this->eventUUID.toStdString();

            if (client.getEvent(&request, &response) == SOAP_OK)
            {
                if (response.event != 0)
                {
                    QMutexLocker(&(this->eventQueueLock));

                    if (!this->eventsAsSignal || this->alsoQueueRaw)
                        this->eventQueue.enqueue(response.event->clone());

                    if (response.event->type == "ParameterChange")
                    {
                        covise::covise__ParameterChangeEvent *e = dynamic_cast<covise::covise__ParameterChangeEvent *>(response.event);
                        covise::WSModule *module = map->getModule(QString::fromStdString(e->moduleID));
                        covise::WSParameter *parameter = module->getParameter(QString::fromStdString(e->parameter->name));
                        parameter->blockSignals(true);
                        parameter->setValueFromSerialisable(e->parameter);
                        parameter->blockSignals(false);
                        if (this->eventsAsSignal)
                            emit eventParameterChanged(module->getID(),
                                                       parameter->getName(),
                                                       parameter->toCoviseString());
                    }
                    else if (response.event->type == "ModuleAdd")
                    {
                        covise::covise__ModuleAddEvent *e = dynamic_cast<covise::covise__ModuleAddEvent *>(response.event);
                        WSModule *m = this->map->addModule(e->module);
                        connect(m, SIGNAL(parameterChanged(covise::WSParameter *)), this, SLOT(parameterChangeCB(covise::WSParameter *)));
                        if (this->eventsAsSignal)
                            emit eventModuleAdd(QString::fromStdString(e->module.id));
                    }
                    else if (response.event->type == "ModuleDel")
                    {
                        covise::covise__ModuleDelEvent *e = dynamic_cast<covise::covise__ModuleDelEvent *>(response.event);
                        this->map->removeModule(QString::fromStdString(e->moduleID));
                        if (this->eventsAsSignal)
                            emit eventModuleDel(QString::fromStdString(e->moduleID));
                    }
                    else if (response.event->type == "ModuleChange")
                    {
                        covise::covise__ModuleChangeEvent *e = dynamic_cast<covise::covise__ModuleChangeEvent *>(response.event);
                        this->map->removeModule(QString::fromStdString(e->module.id));
                        WSModule *m = this->map->addModule(e->module);
                        connect(m, SIGNAL(parameterChanged(covise::WSParameter *)), this, SLOT(parameterChangeCB(covise::WSParameter *)));
                        if (this->eventsAsSignal)
                            emit eventModuleChanged(QString::fromStdString(e->module.id));
                    }
                    else if (response.event->type == "LinkAdd")
                    {
                        covise::covise__LinkAddEvent *e = dynamic_cast<covise::covise__LinkAddEvent *>(response.event);

                        QString fromModule = QString::fromStdString(e->link.from.moduleID);
                        QString fromPort = QString::fromStdString(e->link.from.name);
                        QString toModule = QString::fromStdString(e->link.to.moduleID);
                        QString toPort = QString::fromStdString(e->link.to.name);

                        this->map->link(fromModule, fromPort, toModule, toPort);
                        if (this->eventsAsSignal)
                            emit eventLink(QString::fromStdString(e->link.from.id),
                                           QString::fromStdString(e->link.to.id));
                    }
                    else if (response.event->type == "LinkDel")
                    {
                        covise::covise__LinkDelEvent *e = dynamic_cast<covise::covise__LinkDelEvent *>(response.event);

                        QString id = QString::fromStdString(e->linkID);
                        this->map->unlink(id);
                        if (this->eventsAsSignal)
                            emit eventUnlink(id);
                    }
                    else if (response.event->type == "ModuleDied")
                    {
                        covise::covise__ModuleDiedEvent *e = dynamic_cast<covise::covise__ModuleDiedEvent *>(response.event);
                        map->getModule(QString::fromStdString(e->moduleID))->setDead(true);
                        if (this->eventsAsSignal)
                            emit eventModuleDied(QString::fromStdString(e->moduleID));
                    }
                    else if (response.event->type == "ModuleExecuteStart")
                    {
                        covise::covise__ModuleExecuteStartEvent *e = dynamic_cast<covise::covise__ModuleExecuteStartEvent *>(response.event);
                        if (this->eventsAsSignal)
                            emit eventModuleExecuteStart(QString::fromStdString(e->moduleID));
                    }
                    else if (response.event->type == "ModuleExecuteFinish")
                    {
                        covise::covise__ModuleExecuteFinishEvent *e = dynamic_cast<covise::covise__ModuleExecuteFinishEvent *>(response.event);
                        if (this->eventsAsSignal)
                            emit eventModuleExecuteFinish(QString::fromStdString(e->moduleID));
                    }
                    else if (response.event->type == "ExecuteStart")
                    {
                        this->inExecute = true;
                        if (this->eventsAsSignal)
                            emit eventExecuteStart();
                    }
                    else if (response.event->type == "ExecuteFinish")
                    {
                        this->inExecute = false;
                        if (this->eventsAsSignal)
                            emit eventExecuteFinish();
                    }
                    else if (response.event->type == "OpenNet")
                    {
                        this->inExecute = false;
                        covise::covise__OpenNetEvent *e = dynamic_cast<covise::covise__OpenNetEvent *>(response.event);
                        delete this->map;
                        this->map = new covise::WSMap();
                        this->map->setMapName(QString::fromStdString(e->mapname));
                        if (this->eventsAsSignal)
                            emit eventOpenNet(QString::fromStdString(e->mapname));
                    }
                    else if (response.event->type == "OpenNetDone")
                    {
                        this->inExecute = false;
                        covise::covise__OpenNetDoneEvent *e = dynamic_cast<covise::covise__OpenNetDoneEvent *>(response.event);
                        if (this->eventsAsSignal)
                            emit eventOpenNetDone(QString::fromStdString(e->mapname));
                    }
                    else if (response.event->type == "Quit")
                    {
                        this->inExecute = false;
                        if (this->eventsAsSignal)
                            emit eventQuit();
                        detach();
                    }
                }
            }
        }
        else
        {
            msleep(10);
        }
    }
}

void covise::WSCOVISEClient::parameterChangeCB(covise::WSParameter *p)
{

    if (!this->attached || this->readOnly)
        return;

    covise::COVISEProxy client;
    client.soap_endpoint = soap_strdup(&client, this->endpoint.toLocal8Bit().data());

    covise::_covise__setParameter request;
    covise::_covise__setParameterResponse response;

    request.moduleID = qobject_cast<covise::WSModule *>(sender())->getID().toStdString();
    request.parameter = const_cast<covise::covise__Parameter *>(p->getSerialisable());

    client.setParameter(&request, &response);
}

void covise::WSCOVISEClient::setEventsAsSignal(bool on, bool alsoQueueRaw)
{
    this->eventsAsSignal = on;
    this->alsoQueueRaw = alsoQueueRaw;
}

void covise::WSCOVISEClient::setReadOnly(bool ro)
{
    this->readOnly = ro;
}

bool covise::WSCOVISEClient::isReadOnly() const
{
    return this->readOnly;
}

bool covise::WSCOVISEClient::isInExecute() const
{
    return this->inExecute;
}
