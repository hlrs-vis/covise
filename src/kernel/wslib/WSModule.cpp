/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "WSModule.h"
#include "WSPort.h"

#include <QDebug>
#include <QCoreApplication>

#include "WSIntVectorParameter.h"
#include "WSIntSliderParameter.h"
#include "WSIntScalarParameter.h"
#include "WSFloatVectorParameter.h"
#include "WSFloatSliderParameter.h"
#include "WSFloatScalarParameter.h"
#include "WSFileBrowserParameter.h"
#include "WSChoiceParameter.h"
#include "WSColormapChoiceParameter.h"
#include "WSBooleanParameter.h"
#include "WSStringParameter.h"

covise::WSModule::WSModule(const QString &name, const QString &category, const QString &host)
    : QObject(0)
    , name(name)
    , id(name)
    , host(host)
    , category(category)
{
    setObjectName(this->id);
}

covise::WSModule::WSModule(const covise::covise__Module &module)
    : QObject(0)
{
    this->setFromSerialisable(module);
}

covise::WSModule::~WSModule()
{
    emit deleted(this->id);
    foreach (covise::WSParameter *parameter, this->parameters)
        delete parameter;
    foreach (covise::WSPort *port, this->inputPorts)
        delete port;
    foreach (covise::WSPort *port, this->outputPorts)
        delete port;
}

void covise::WSModule::instantiate(const QString &host, const QString &instance)
{
    this->host = host;
    this->instance = instance;
    emit changed();
}

covise::WSPort *covise::WSModule::addInputPort(const QString &inName, const QStringList &inTypes, covise::WSPort::PortType inPortType)
{
    covise::WSPort *port = new covise::WSPort(this, inName, inTypes, inPortType);

    inputPorts.insert(inName, port);

    //std::cerr << "WSModule::addInputPort info: port " << inName << " added." << std::endl;
    emit changed();

    return port;
}

covise::WSPort *covise::WSModule::addOutputPort(const QString &inName, const QStringList &inTypes, covise::WSPort::PortType inPortType)
{
    covise::WSPort *port = new covise::WSPort(this, inName, inTypes, inPortType);

    outputPorts.insert(inName, port);

    //std::cerr << "WSModule::addOutputPort info: port " << inName << " added." << std::endl;
    emit changed();

    return port;
}

covise::WSPort *covise::WSModule::getOutputPort(const QString &name) const
{
    if (this->outputPorts.contains(name))
        return this->outputPorts[name];
    else
        return 0;
}

covise::WSPort *covise::WSModule::getInputPort(const QString &name) const
{
    if (this->inputPorts.contains(name))
        return this->inputPorts[name];
    else
        return 0;
}

covise::WSParameter *covise::WSModule::addParameter(const QString &inName, const QString &inType, const QString &inDescription)
{
    WSParameter *parameter = 0;

    if (inType == "Boolean")
    {
        parameter = new covise::WSBooleanParameter(inName, inDescription);
    }
    else if (inType == "Choice")
    {
        parameter = new covise::WSChoiceParameter(inName, inDescription);
    }
    else if (inType == "ColormapChoice")
    {
        parameter = new covise::WSColormapChoiceParameter(inName, inDescription);
    }
    else if (inType == "FileBrowser")
    {
        parameter = new covise::WSFileBrowserParameter(inName, inDescription);
    }
    else if (inType == "FloatScalar")
    {
        parameter = new covise::WSFloatScalarParameter(inName, inDescription);
    }
    else if (inType == "FloatSlider")
    {
        parameter = new covise::WSFloatSliderParameter(inName, inDescription);
    }
    else if (inType == "FloatVector")
    {
        parameter = new covise::WSFloatVectorParameter(inName, inDescription);
    }
    else if (inType == "IntScalar")
    {
        parameter = new covise::WSIntScalarParameter(inName, inDescription);
    }
    else if (inType == "IntSlider")
    {
        parameter = new covise::WSIntSliderParameter(inName, inDescription);
    }
    else if (inType == "IntVector")
    {
        parameter = new covise::WSIntVectorParameter(inName, inDescription);
    }
    else if (inType == "String")
    {
        parameter = new covise::WSStringParameter(inName, inDescription);
    }
    else
    {
#ifdef DEBUG
        std::cerr << "WSModule::addParameter err: unknown parameter type " << qPrintable(inType) << " for parameter " << qPrintable(inName) << std::endl;
#endif
        return 0;
    }

    parameters.insert(inName, parameter);
    connect(parameter, SIGNAL(parameterChanged(covise::WSParameter *)), this, SLOT(parameterChangedCB(covise::WSParameter *)));

    //std::cerr << "WSModule::addParameter info: added parameter " << qPrintable(inName) << " (" << qPrintable(inType) << ")" << std::endl;
    emit changed();
    emit parameterChanged(parameter);

    return parameter;
}

covise::WSParameter *covise::WSModule::getParameter(const QString &name) const
{
    if (this->parameters.contains(name))
        return this->parameters[name];
    else
        return 0;
}

void covise::WSModule::setTitle(const QString &title)
{
    this->title = title;
    emit changed();
}

QStringList covise::WSModule::getParameterNames() const
{
    std::cerr << "WSModule::getParameterNames info: " << QStringList(this->parameters.keys()).join(" ").toStdString() << std::endl;
    return this->parameters.keys();
}

void covise::WSModule::setDead(bool dead)
{
    this->dead = dead;
    emit died();
}

void covise::WSModule::setFromSerialisable(const covise::covise__Module &m)
{
    this->name = QString::fromStdString(m.name);
    this->category = QString::fromStdString(m.category);
    this->host = QString::fromStdString(m.host);
    this->description = QString::fromStdString(m.description);
    this->id = QString::fromStdString(m.id);
    this->position.setX(m.position.x);
    this->position.setY(m.position.y);
    this->instance = QString::fromStdString(m.instance);
    this->title = QString::fromStdString(m.title);

    foreach (covise::WSParameter *parameter, this->parameters.values())
    {
        delete parameter;
    }
    this->parameters.clear();
    for (std::vector<covise::covise__Parameter *>::const_iterator cp = m.parameters.begin(); cp != m.parameters.end(); ++cp)
    {
        covise::WSParameter *p = covise::WSParameter::create(*cp);
        if (p)
        {
            this->parameters[p->getName()] = p;
            connect(p, SIGNAL(parameterChanged(covise::WSParameter *)), this, SLOT(parameterChangedCB(covise::WSParameter *)));
        }
    }

    for (std::vector<covise::covise__Port>::const_iterator ip = m.inputPorts.begin(); ip != m.inputPorts.end(); ++ip)
    {
        this->inputPorts.insert(QString::fromStdString(ip->name), new WSPort(this, *ip));
    }

    for (std::vector<covise::covise__Port>::const_iterator op = m.outputPorts.begin(); op != m.outputPorts.end(); ++op)
    {
        this->outputPorts.insert(QString::fromStdString(op->name), new WSPort(this, *op));
    }

    setObjectName(this->id);

    emit changed();
}

covise::covise__Module covise::WSModule::getSerialisable() const
{
    covise::covise__Module m;

    m.name = getName().toStdString();
    m.category = getCategory().toStdString();
    m.host = getHost().toStdString();
    m.description = getDescription().toStdString();
    m.id = getID().toStdString();
    m.position.x = getPosition().x();
    m.position.y = getPosition().y();
    m.instance = getInstance().toStdString();
    m.title = getTitle().toStdString();

    foreach (covise::WSParameter *parameter, getParameters().values())
    {
        m.parameters.push_back(parameter->getSerialisable()->clone());
    }

    foreach (covise::WSPort *port, getInputPorts().values())
    {
        m.inputPorts.push_back(port->getSerialisable());
    }

    foreach (covise::WSPort *port, getOutputPorts().values())
    {
        m.outputPorts.push_back(port->getSerialisable());
    }

    return m;
}

void covise::WSModule::parameterChangedCB(covise::WSParameter *p)
{
    emit parameterChanged(p);
    //dumpObjectInfo();
}

// EOF
