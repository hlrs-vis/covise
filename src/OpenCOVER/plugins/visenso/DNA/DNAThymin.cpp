/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "DNAThymin.h"
#include "DNAAdenin.h"
#include "DNACytosin.h"
#include "DNAGuanin.h"
#include "DNABaseUnitConnectionPoint.h"

#include <cover/coVRPluginSupport.h>

using namespace opencover;

DNAThymin::DNAThymin(osg::Matrix m, int num)
    : DNABase(m, 100.0, "THYMIN", "dna/Thymin", 55.0, false, num)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nDNAThymin::DNAThymin\n");
    // add connections to list
    a1 = new DNABaseUnitConnectionPoint(this, "thymin1", osg::Vec3(-0.75, 0, 0), osg::Vec3(0, -1, 0), "adenin1");
    c1 = new DNABaseUnitConnectionPoint(this, "thymin2", osg::Vec3(0.5, 0, 0), osg::Vec3(0, -1, 0), "cytosin2");
    g1 = new DNABaseUnitConnectionPoint(this, "thymin3", osg::Vec3(-0.8, 0, 0), osg::Vec3(0, -1, 0), "guanin2");
    addConnectionPoint(a1);
    addConnectionPoint(c1);
    addConnectionPoint(g1);
    this->setConnection("thymin2", "cytosin2", false, false, NULL);
    this->setConnection("thymin3", "guanin2", false, false, NULL);
    enabledCP.clear();
    enabledCP.empty();
}

DNAThymin::~DNAThymin()
{
    delete a1;
    delete c1;
    delete g1;
}

void DNAThymin::setConnection(std::string nameConnPoint, std::string nameConnPoint2, bool connected, bool enabled, DNABaseUnit *connObj, bool sendBack)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "DNAThymin::setConnection %s %s %d %d\n", nameConnPoint.c_str(), nameConnPoint2.c_str(), connected, enabled);

    // check if other connection is already set
    if (nameConnPoint2.compare("thymin1") == 0 && (c1->isConnected() || g1->isConnected()))
    {
        a1->setEnabled(enabled);
        return;
    }
    else if (nameConnPoint2.compare("thymin2") == 0 && (a1->isConnected() || g1->isConnected()))
    {
        c1->setEnabled(enabled);
        return;
    }
    else if (nameConnPoint2.compare("thymin3") == 0 && (a1->isConnected() || c1->isConnected()))
    {
        g1->setEnabled(enabled);
        return;
    }

    DNABaseUnit::setConnection(nameConnPoint, nameConnPoint2, connected, enabled, connObj, sendBack);

    DNABaseUnitConnectionPoint *mycp = getConnectionPoint(nameConnPoint);
    if (connObj != NULL)
    {
        DNABaseUnitConnectionPoint *ocp = connObj->getConnectionPoint(nameConnPoint2);
        enableOtherConnPoints(mycp, ocp, connected);
    }
    else
        enableOtherConnPoints(mycp, NULL, connected);
}

bool DNAThymin::connectTo(DNABaseUnit *otherUnit, DNABaseUnitConnectionPoint *myConnectionPoint, DNABaseUnitConnectionPoint *otherConnectionPoint)
{
    if (cover->debugLevel(3))
    {
        fprintf(stderr, "DNAThymin::connectTo %s %s\n", myConnectionPoint->getConnectableBaseUnitName().c_str(), otherConnectionPoint->getConnectableBaseUnitName().c_str());
        fprintf(stderr, "    a1=%d c1=%d g1=%d\n", a1->isConnected(), c1->isConnected(), g1->isConnected());
    }

    // check if other connection is already set
    if (!(isConnectionPossible(otherConnectionPoint->getConnectableBaseUnitName())) || !(myConnectionPoint->getMyBaseUnit()->isConnectionPossible(otherConnectionPoint->getConnectableBaseUnitName())))
        return false;

    if (DNABaseUnit::connectTo(otherUnit, myConnectionPoint, otherConnectionPoint))
    {
        if (myConnectionPoint && otherConnectionPoint)
            enableOtherConnPoints(myConnectionPoint, otherConnectionPoint, true);
        else
            fprintf(stderr, "ERROR connectin base\n");
        return true;
    }
    return false;
}

bool DNAThymin::isConnectionPossible(std::string connPoint)
{
    if (connPoint.compare("thymin1") == 0 && (c1->isConnected() || g1->isConnected()))
        return false;
    else if (connPoint.compare("thymin2") == 0 && (a1->isConnected() || g1->isConnected()))
        return false;
    else if (connPoint.compare("thymin3") == 0 && (a1->isConnected() || c1->isConnected()))
        return false;
    return true;
}

void DNAThymin::enableOtherConnPoints(DNABaseUnitConnectionPoint *mycp, DNABaseUnitConnectionPoint *ocp, bool connected, bool callConnectedPoint)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "DNAThymin::enableOtherConnPoints, %d\n", callConnectedPoint);
    // clear list of enabled connectionpoints
    if (connected)
    {
        enabledCP.clear();
        enabledCP.empty();
        // disable all other connections
        if (a1->getMyBaseUnitName().compare(mycp->getMyBaseUnitName()) == 0)
        {
            if (callConnectedPoint)
                dynamic_cast<DNAAdenin *>(ocp->getMyBaseUnit())->enableOtherConnPoints(ocp, mycp, connected, false);
            if (c1->isEnabled())
            {
                enabledCP.push_back(c1);
                c1->setEnabled(false);
            }
            if (g1->isEnabled())
            {
                enabledCP.push_back(g1);
                g1->setEnabled(false);
            }
        }
        else if (c1->getMyBaseUnitName().compare(mycp->getMyBaseUnitName()) == 0)
        {
            if (callConnectedPoint)
                dynamic_cast<DNACytosin *>(ocp->getMyBaseUnit())->enableOtherConnPoints(ocp, mycp, connected, false);
            if (a1->isEnabled())
            {
                enabledCP.push_back(a1);
                a1->setEnabled(false);
            }
            if (g1->isEnabled())
            {
                enabledCP.push_back(g1);
                g1->setEnabled(false);
            }
        }
        else if (g1->getMyBaseUnitName().compare(mycp->getMyBaseUnitName()) == 0)
        {
            if (callConnectedPoint)
                dynamic_cast<DNAGuanin *>(ocp->getMyBaseUnit())->enableOtherConnPoints(ocp, mycp, connected, false);
            if (c1->isEnabled())
            {
                enabledCP.push_back(c1);
                c1->setEnabled(false);
            }
            if (a1->isEnabled())
            {
                enabledCP.push_back(a1);
                a1->setEnabled(false);
            }
        }
    }
    else
    {
        for (std::list<DNABaseUnitConnectionPoint *>::iterator it = enabledCP.begin(); it != enabledCP.end(); it++)
            (*it)->setEnabled(true);
        enabledCP.clear();
        enabledCP.empty();
    }
}
