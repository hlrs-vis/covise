/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "WSLink.h"

#include "WSPort.h"
#include "WSModule.h"

#include <cassert>

covise::WSLink::WSLink(covise::WSPort *from, covise::WSPort *to)
    : QObject(0)
    , fromPort(from)
    , toPort(to)
{
    this->id = makeID(from, to);
    connect(from, SIGNAL(destroyed()), this, SLOT(portDeleted()));
    connect(to, SIGNAL(destroyed()), this, SLOT(portDeleted()));
}

covise::WSLink::~WSLink()
{
    //    if (to())
    //       to()->removeLink(this);
    //    if (from())
    //       from()->removeLink(this);
    emit deleted(this->getLinkID());
}

covise::WSPort *covise::WSLink::from() const
{
    return this->fromPort;
}

covise::WSPort *covise::WSLink::to() const
{
    return this->toPort;
}

bool covise::WSLink::isLinkTo(const covise::WSPort *port) const
{
    return (port->getID() == this->fromPort->getID() || port->getID() == this->toPort->getID());
}

const QString &covise::WSLink::getLinkID() const
{
    return this->id;
}

QString covise::WSLink::makeID(covise::WSPort *from, covise::WSPort *to)
{
    QString id = from->getModule()->getID() + "_" + from->getName() + "_" + to->getModule()->getID() + "_" + to->getName();
    std::cerr << qPrintable(id) << std::endl;
    return from->getModule()->getID() + "_" + from->getName() + "_" + to->getModule()->getID() + "_" + to->getName();
}

QString covise::WSLink::makeID(const QString &fromModule, const QString &fromPort,
                               const QString &toModule, const QString &toPort)
{
    return fromModule + "_" + fromPort + "_" + toModule + "_" + toPort;
}

covise::covise__Link covise::WSLink::getSerialisable() const
{
    covise::covise__Link p;

    p.id = getLinkID().toStdString();
    p.from = from()->getSerialisable();
    p.to = to()->getSerialisable();

    return p;
}

void covise::WSLink::portDeleted()
{
    if (sender() == this->fromPort)
        this->fromPort = 0;
    else if (sender() == this->toPort)
        this->toPort = 0;
    else
        assert(0);
}
