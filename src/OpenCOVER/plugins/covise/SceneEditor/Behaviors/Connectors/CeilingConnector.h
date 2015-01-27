/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CEILING_CONNECTOR_H
#define CEILING_CONNECTOR_H

#include "Connector.h"

class CeilingConnector : public Connector
{
public:
    CeilingConnector(MountBehavior *mb);
    virtual ~CeilingConnector();

    virtual bool buildFromXML(QDomElement *connectorElement);

    virtual void applyRestriction(Connector *slaveConnector);

    virtual void prepareTransform(Connector *slaveConnector);

private:
};

#endif
