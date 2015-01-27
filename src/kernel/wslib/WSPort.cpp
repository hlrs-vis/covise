/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "WSPort.h"

#include "WSLink.h"
#include "WSModule.h"

#include <cassert>
#include <vector>

covise::WSPort::WSPort(const covise::WSModule *module, const QString &name, const QStringList &acceptedTypes, PortType portType)
    : portName(name)
    , dataTypes(acceptedTypes)
    , portType(portType)
    , module(module)
{
    this->id = this->module->getID() + "/" + this->portName;
}

covise::WSPort::WSPort(const covise::WSModule *module, const covise::covise__Port &port)
{
    setFromSerialisable(module, port);
}

void covise::WSPort::setFromSerialisable(const covise::WSModule *module, const covise::covise__Port &port)
{
    this->portName = QString::fromStdString(port.name);
    this->dataTypes.clear();
    for (std::vector<std::string>::const_iterator ptypes = port.types.begin(); ptypes != port.types.end(); ++ptypes)
        this->dataTypes << QString::fromStdString(*ptypes);

    if (port.portType == "Optional")
        this->portType = Optional;
    else if (port.portType == "Dependent")
        this->portType = Dependent;
    else
        this->portType = Default;

    this->module = module;

    this->id = this->module->getID() + "/" + this->portName;
}

covise::WSPort::~WSPort()
{
    //    while(!this->links.empty())
    //       delete this->links.takeFirst();
}

covise::covise__Port covise::WSPort::getSerialisable() const
{
    covise::covise__Port p;

    p.name = getName().toStdString();
    foreach (QString type, getTypes())
    {
        p.types.push_back(type.toStdString());
    }

    switch (getPortType())
    {
    case Default:
        p.portType = "Default";
        break;
    case Optional:
        p.portType = "Optional";
        break;
    case Dependent:
        p.portType = "Dependent";
        break;
    }

    p.id = this->id.toStdString();
    p.moduleID = this->module->getID().toStdString();

    return p;
}

// void covise::WSPort::addLink(covise::WSLink * link)
// {
//    assert (link->from() == this || link->to() == this);
//    covise::WSPort * otherPort;
//    if (link->from()->getID() == this->getID())
//       otherPort = link->to();
//    else
//       otherPort = link->from();

//    foreach(covise::WSLink * l, this->links)
//    {
//       if (otherPort->getID() == l->from()->getID() || otherPort->getID() == l->to()->getID())
//          return;
//    }

// #ifdef DEBUG
//    std::cerr << "WSPort::addLink info: adding link " << qPrintable(link->getLinkID()) << std::endl;
// #endif
//    this->links.append(link);
//    emit linkAdded(link);

// }

// void covise::WSPort::removeLink(covise::WSLink * link)
// {

//    foreach(covise::WSLink * l, this->links)
//    {
//       if (link->getLinkID() == l->getLinkID())
//       {
//          this->links.removeOne(l);
// #ifdef DEBUG
//          std::cerr << "WSPort::addLink info: removing link " << qPrintable(link->getLinkID()) << std::endl;
// #endif
//          emit linkRemoved(l);
//          return;
//       }
//    }

// }

// EOF
