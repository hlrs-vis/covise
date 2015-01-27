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

DNAAdenin::DNAAdenin(osg::Matrix m, int num)
    : DNABase(m, 100.0, "ADENIN", "dna/Adenin", 55.0, true, num)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nDNAAdenin::DNAAdenin\n");

    // add connections to list
    t1 = new DNABaseUnitConnectionPoint(this, "adenin1", osg::Vec3(0.4, 0, 0), osg::Vec3(0, -1, 0), "thymin1");
    c1 = new DNABaseUnitConnectionPoint(this, "adenin2", osg::Vec3(0.55, 0, 0), osg::Vec3(0, -1, 0), "cytosin3");
    g1 = new DNABaseUnitConnectionPoint(this, "adenin3", osg::Vec3(0.65, 0, 0), osg::Vec3(0, -1, 0), "guanin3");
    addConnectionPoint(t1);
    addConnectionPoint(c1);
    addConnectionPoint(g1);
    this->setConnection("adenin2", "cytosin3", false, false, NULL);
    this->setConnection("adenin3", "guanin3", false, false, NULL);
    enabledCP.clear();
    enabledCP.empty();
}

DNAAdenin::~DNAAdenin()
{
    delete t1;
    delete g1;
    delete c1;
}

void DNAAdenin::setConnection(std::string nameConnPoint, std::string nameConnPoint2, bool connected, bool enabled, DNABaseUnit *connObj, bool sendBack)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "DNAAdenin::setConnection %s %s %d %d\n", nameConnPoint.c_str(), nameConnPoint2.c_str(), connected, enabled);

    // check if other connection is already set
    if (nameConnPoint2.compare("adenin1") == 0 && (c1->isConnected() || g1->isConnected()))
    {
        t1->setEnabled(enabled);
        return;
    }
    else if (nameConnPoint2.compare("adenin2") == 0 && (t1->isConnected() || g1->isConnected()))
    {
        c1->setEnabled(enabled);
        return;
    }
    else if (nameConnPoint2.compare("adenin3") == 0 && (c1->isConnected() || t1->isConnected()))
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

bool DNAAdenin::connectTo(DNABaseUnit *otherUnit, DNABaseUnitConnectionPoint *myConnectionPoint, DNABaseUnitConnectionPoint *otherConnectionPoint)
{
    if (cover->debugLevel(3))
    {
        fprintf(stderr, "DNAAdenin::connectTo %s %s\n", myConnectionPoint->getConnectableBaseUnitName().c_str(), otherConnectionPoint->getConnectableBaseUnitName().c_str());
        fprintf(stderr, "    c1=%d g1=%d t1=%d\n", c1->isConnected(), g1->isConnected(), t1->isConnected());
    }

    // check if other connection is already set
    if (!(isConnectionPossible(otherConnectionPoint->getConnectableBaseUnitName())) || !(otherConnectionPoint->getMyBaseUnit()->isConnectionPossible(myConnectionPoint->getConnectableBaseUnitName())))
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

bool DNAAdenin::isConnectionPossible(std::string connPoint)
{
    if (connPoint.compare("adenin1") == 0 && (c1->isConnected() || g1->isConnected()))
        return false;
    else if (connPoint.compare("adenin2") == 0 && (t1->isConnected() || g1->isConnected()))
        return false;
    else if (connPoint.compare("adenin3") == 0 && (c1->isConnected() || t1->isConnected()))
        return false;
    return true;
}

void DNAAdenin::enableOtherConnPoints(DNABaseUnitConnectionPoint *mycp, DNABaseUnitConnectionPoint *ocp, bool connected, bool callConnectedPoint)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "DNAAdenin::enableOtherConnPoints, %d\n", callConnectedPoint);
    // clear list of enabled connectionpoints
    if (connected)
    {
        enabledCP.clear();
        enabledCP.empty();
        if (cover->debugLevel(3))
            fprintf(stderr, "    connect %s\n", mycp->getMyBaseUnitName().c_str());
        // disable all other connections
        if (mycp->getMyBaseUnitName().compare("adenin1") == 0)
        {
            if (callConnectedPoint)
                dynamic_cast<DNAThymin *>(ocp->getMyBaseUnit())->enableOtherConnPoints(ocp, mycp, connected, false);
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
        else if (mycp->getMyBaseUnitName().compare("adenin2") == 0)
        {
            if (callConnectedPoint)
                dynamic_cast<DNACytosin *>(ocp->getMyBaseUnit())->enableOtherConnPoints(ocp, mycp, connected, false);
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
        else if (mycp->getMyBaseUnitName().compare("adenin3") == 0)
        {
            if (cover->debugLevel(3))
                fprintf(stderr, "         adenin3 \n");
            if (callConnectedPoint)
                dynamic_cast<DNAGuanin *>(ocp->getMyBaseUnit())->enableOtherConnPoints(ocp, mycp, connected, false);
            if (c1->isEnabled())
            {
                enabledCP.push_back(c1);
                c1->setEnabled(false);
            }
            if (t1->isEnabled())
            {
                enabledCP.push_back(t1);
                t1->setEnabled(false);
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
