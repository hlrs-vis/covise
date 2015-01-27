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

DNACytosin::DNACytosin(osg::Matrix m, int num)
    : DNABase(m, 100.0, "CYTOSIN", "dna/Cytosin", 55.0, false, num)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nDNACytosin::DNACytosin\n");
    // add connections to list
    g1 = new DNABaseUnitConnectionPoint(this, "cytosin1", osg::Vec3(-0.7, 0, 0), osg::Vec3(0, -1, 0), "guanin1");
    a1 = new DNABaseUnitConnectionPoint(this, "cytosin3", osg::Vec3(-0.65, 0, 0), osg::Vec3(0, -1, 0), "adenin2");
    t1 = new DNABaseUnitConnectionPoint(this, "cytosin2", osg::Vec3(0.4, 0, 0), osg::Vec3(0, 1, 0), "thymin2");
    t1->setRotation(false);
    addConnectionPoint(g1);
    addConnectionPoint(a1);
    addConnectionPoint(t1);
    this->setConnection("cytosin3", "adenin2", false, false, NULL);
    this->setConnection("cytosin2", "thymin2", false, false, NULL);
    enabledCP.clear();
    enabledCP.empty();
}

DNACytosin::~DNACytosin()
{
    delete g1;
    delete a1;
    delete t1;
}

void DNACytosin::setConnection(std::string nameConnPoint, std::string nameConnPoint2, bool connected, bool enabled, DNABaseUnit *connObj, bool sendBack)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "DNACytosin::setConnection %s %s %d %d\n", nameConnPoint.c_str(), nameConnPoint2.c_str(), connected, enabled);

    // check if other connection is already set
    if (nameConnPoint2.compare("cytosin1") == 0 && (a1->isConnected() || t1->isConnected()))
    {
        g1->setEnabled(enabled);
        return;
    }
    else if (nameConnPoint2.compare("cytosin3") == 0 && (g1->isConnected() || t1->isConnected()))
    {
        a1->setEnabled(enabled);
        return;
    }
    else if (nameConnPoint2.compare("cytosin2") == 0 && (a1->isConnected() || g1->isConnected()))
    {
        t1->setEnabled(enabled);
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

bool DNACytosin::connectTo(DNABaseUnit *otherUnit, DNABaseUnitConnectionPoint *myConnectionPoint, DNABaseUnitConnectionPoint *otherConnectionPoint)
{
    if (cover->debugLevel(3))
    {
        fprintf(stderr, "DNACytosin::connectTo %s %s\n", myConnectionPoint->getConnectableBaseUnitName().c_str(), otherConnectionPoint->getConnectableBaseUnitName().c_str());
        fprintf(stderr, "    a1=%d g1=%d t1=%d\n", a1->isConnected(), g1->isConnected(), t1->isConnected());
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

bool DNACytosin::isConnectionPossible(std::string connPoint)
{
    if (connPoint.compare("cytosin1") == 0 && (a1->isConnected() || t1->isConnected()))
        return false;
    else if (connPoint.compare("cytosin3") == 0 && (g1->isConnected() || t1->isConnected()))
        return false;
    else if (connPoint.compare("cytosin2") == 0 && (a1->isConnected() || g1->isConnected()))
        return false;
    return true;
}

void DNACytosin::enableOtherConnPoints(DNABaseUnitConnectionPoint *mycp, DNABaseUnitConnectionPoint *ocp, bool connected, bool callConnectedPoint)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "DNACytosin::enableOtherConnPoints, %d\n", callConnectedPoint);
    if (connected)
    {
        enabledCP.clear();
        enabledCP.empty();
        // disable all other connections
        if (a1->getMyBaseUnitName().compare(mycp->getMyBaseUnitName()) == 0)
        {
            if (callConnectedPoint)
                dynamic_cast<DNAAdenin *>(ocp->getMyBaseUnit())->enableOtherConnPoints(ocp, mycp, connected, false);
            if (t1->isEnabled())
            {
                enabledCP.push_back(t1);
                t1->setEnabled(false);
            }
            if (g1->isEnabled())
            {
                enabledCP.push_back(g1);
                g1->setEnabled(false);
            }
        }
        else if (t1->getMyBaseUnitName().compare(mycp->getMyBaseUnitName()) == 0)
        {
            if (callConnectedPoint)
                dynamic_cast<DNAThymin *>(ocp->getMyBaseUnit())->enableOtherConnPoints(ocp, mycp, connected, false);
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
            if (t1->isEnabled())
            {
                enabledCP.push_back(t1);
                t1->setEnabled(false);
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
