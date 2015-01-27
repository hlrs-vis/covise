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

DNAGuanin::DNAGuanin(osg::Matrix m, int num)
    : DNABase(m, 100.0, "GUANIN", "dna/Guanin", 55.0, true, num)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nDNAGuanin::DNAGuanin\n");

    // add connections to list
    c1 = new DNABaseUnitConnectionPoint(this, "guanin1", osg::Vec3(0.4, 0, 0), osg::Vec3(0, -1, 0), "cytosin1");
    a1 = new DNABaseUnitConnectionPoint(this, "guanin3", osg::Vec3(0.65, 0, 0), osg::Vec3(0, -1, 0), "adenin3");
    t1 = new DNABaseUnitConnectionPoint(this, "guanin2", osg::Vec3(0.6, 0, 0), osg::Vec3(0, -1, 0), "thymin3");
    a1->setRotation(true);
    addConnectionPoint(c1);
    addConnectionPoint(a1);
    addConnectionPoint(t1);
    this->setConnection("guanin3", "adenin3", false, false, NULL);
    this->setConnection("guanin2", "thymin3", false, false, NULL);
    enabledCP.clear();
    enabledCP.empty();
}

DNAGuanin::~DNAGuanin()
{
    delete c1;
    delete a1;
    delete t1;
}

void DNAGuanin::setConnection(std::string nameConnPoint, std::string nameConnPoint2, bool connected, bool enabled, DNABaseUnit *connObj, bool sendBack)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "DNAGuanin::setConnection %s %s %d %d\n", nameConnPoint.c_str(), nameConnPoint2.c_str(), connected, enabled);

    // check if other connection is already set
    if (nameConnPoint2.compare("guanin1") == 0 && (a1->isConnected() || t1->isConnected()))
    {
        c1->setEnabled(enabled);
        return;
    }
    else if (nameConnPoint2.compare("guanin2") == 0 && (a1->isConnected() || c1->isConnected()))
    {
        t1->setEnabled(enabled);
        return;
    }
    else if (nameConnPoint2.compare("guanin3") == 0 && (c1->isConnected() || t1->isConnected()))
    {
        a1->setEnabled(enabled);
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

bool DNAGuanin::connectTo(DNABaseUnit *otherUnit, DNABaseUnitConnectionPoint *myConnectionPoint, DNABaseUnitConnectionPoint *otherConnectionPoint)
{
    if (cover->debugLevel(3))
    {
        fprintf(stderr, "DNAGuanin::connectTo %s %s\n", myConnectionPoint->getConnectableBaseUnitName().c_str(), otherConnectionPoint->getConnectableBaseUnitName().c_str());
        fprintf(stderr, "    a1=%d c1=%d t1=%d\n", a1->isConnected(), c1->isConnected(), t1->isConnected());
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

bool DNAGuanin::isConnectionPossible(std::string connPoint)
{
    if (connPoint.compare("guanin1") == 0 && (a1->isConnected() || t1->isConnected()))
        return false;
    else if (connPoint.compare("guanin2") == 0 && (a1->isConnected() || c1->isConnected()))
        return false;
    else if (connPoint.compare("guanin3") == 0 && (c1->isConnected() || t1->isConnected()))
        return false;
    return true;
}

void DNAGuanin::enableOtherConnPoints(DNABaseUnitConnectionPoint *mycp, DNABaseUnitConnectionPoint *ocp, bool connected, bool callConnectedPoint)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "DNAGuanin::enableOtherConnPoints, %d\n", callConnectedPoint);
    if (connected)
    {
        enabledCP.clear();
        enabledCP.empty();
        // disable all other connections
        if (mycp->getMyBaseUnitName().compare("guanin3") == 0)
        {
            if (callConnectedPoint)
                dynamic_cast<DNAAdenin *>(ocp->getMyBaseUnit())->enableOtherConnPoints(ocp, mycp, connected, false);
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
        else if (mycp->getMyBaseUnitName().compare("guanin1") == 0)
        {
            if (callConnectedPoint)
                dynamic_cast<DNACytosin *>(ocp->getMyBaseUnit())->enableOtherConnPoints(ocp, mycp, connected, false);
            if (a1->isEnabled())
            {
                enabledCP.push_back(a1);
                a1->setEnabled(false);
            }
            if (t1->isEnabled())
            {
                enabledCP.push_back(t1);
                t1->setEnabled(false);
            }
        }
        else if (mycp->getMyBaseUnitName().compare("guanin2") == 0)
        {
            if (callConnectedPoint)
                dynamic_cast<DNAThymin *>(ocp->getMyBaseUnit())->enableOtherConnPoints(ocp, mycp, connected, false);
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
