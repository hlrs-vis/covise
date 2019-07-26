/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "DNABaseUnit.h"
#include "DNAPlugin.h"
#include "DNABaseUnitConnectionPoint.h"
#include <cover/coVRFileManager.h>

#include <osg/Vec3>
#include <osg/Geode>
#include <osg/MatrixTransform>
#include <OpenVRUI/osg/mathUtils.h>

#include <cover/coVRMSController.h>
#include <cover/OpenCOVER.h>
#include <grmsg/coGRObjRegisterMsg.h>
#include <grmsg/coGRObjMovedMsg.h>
#include <grmsg/coGRObjSetConnectionMsg.h>
#include <grmsg/coGRObjTransformMsg.h>
#include <cover/coVRNavigationManager.h>
#include <cover/coVRPluginList.h>
#include <cover/coVRPluginSupport.h>
#include <config/CoviseConfig.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <net/message_types.h>
#include <net/message.h>

#include <algorithm>

using namespace covise;
using namespace opencover;
using namespace grmsg;

DNABaseUnit::DNABaseUnit(osg::Matrix m, float size, const char *interactorName, string geofilename, float boundingRadius, int num)
    : coVR3DTransRotInteractor(m, size, coInteraction::ButtonA, "Hand", interactorName, coInteraction::Medium)
    , radius_(boundingRadius)
    , size_(size)
    , geofilename_(geofilename)
    , num_(num)
    , dontMove_(false)
    , movedByInteraction_(false)
    , isMoving_(false)
    , visible_(true)
    , registered_(false)
    , name_(NULL)
    , presentationMode_(false)
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\nDNABaseUnit::DNABaseUnit\n");

    stringstream idString;
    idString << _interactorName;
    idString << "_"
             << "DNA"
             << "_";
    idString << num_;
    name_ = new char[idString.str().size() + 1];
    strcpy(name_, idString.str().c_str());

    registerAtGui();

    rot180z.makeRotate(osg::inDegrees(180.0), 0, 0, 1);
    rot180y.makeRotate(osg::inDegrees(180.0), 0, 1, 0);
    osg::Quat rot10z, rotm10z;
    rot10z.makeRotate(osg::inDegrees(12.0), 0, 0, 1);
    rotm10z.makeRotate(osg::inDegrees(-12.0), 0, 0, 1);
    rotMat10.makeRotate(rot10z);
    rotMatm10.makeRotate(rotm10z);

    presentationMode_ = coCoviseConfig::isOn("COVER.Plugin.DNA.PresentationMode", false);

    // enable interactors for intersection
    enableIntersection();

    //sendMatrix to GUI
    sendMatrixToGUI();
}

DNABaseUnit::~DNABaseUnit()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\nDNABaseUnit::~DNABaseUnit\n");

    if (name_ != NULL)
        delete[] name_;
}

/**
  * registers the dnaBaseUnit at the GUI
  */
void DNABaseUnit::registerAtGui()
{
    //register geometry at gui
    if (coVRMSController::instance()->isMaster())
    {
        if (cover->debugLevel(3))
            fprintf(stderr, "DNABaseUnit::registerObjAtUi %s\n", _interactorName);

        coGRObjRegisterMsg regMsg(getNameforMessage(), NULL);
        Message grmsg{ Message::UI, covise::DataHandle{(char*)(regMsg.c_str()), strlen(regMsg.c_str()) + 1 , false} }; +1;
        coVRPluginList::instance()->sendVisMessage(&grmsg);
    }

    registered_ = true;
}

/**
  * adds a connection point
  * push back the connectionPoint in connections_-list
  * sends the connection to the GUI
  */
void DNABaseUnit::addConnectionPoint(DNABaseUnitConnectionPoint *connection)
{
    connections_.push_back(connection);
    sendConnectionToGUI(connection->getMyBaseUnitName(), connection->getConnectableBaseUnitName(), -1, -1, " ");
}

/**
  * returns the connectionpoint to the name
  * returns null if no connection point of that name is in the list
  */
DNABaseUnitConnectionPoint *DNABaseUnit::getConnectionPoint(string name)
{
    for (list<DNABaseUnitConnectionPoint *>::iterator it = connections_.begin(); it != connections_.end(); it++)
    {
        if (((*it)->getMyBaseUnitName()).compare(name) == 0)
            return (*it);
    }
    return NULL;
}

/**
  * checks if the baseUnit is connected to the other baseUnit
  */
bool DNABaseUnit::isConnectedTo(DNABaseUnit *otherUnit)
{
    std::list<DNABaseUnitConnectionPoint *>::iterator it;
    for (it = connections_.begin(); it != connections_.end(); it++)
    {
        if ((*it)->isConnected() && (*it)->getConnectedPoint()->getMyBaseUnit() == otherUnit)
            return true;
    }
    return false;
}

/**
  * connects to baseUnits
  * sends connection to GUI
  */
bool DNABaseUnit::connectTo(DNABaseUnit *otherUnit, DNABaseUnitConnectionPoint *myConnectionPoint, DNABaseUnitConnectionPoint *otherConnectionPoint)
{
    if (cover->debugLevel(3))
    {
        fprintf(stderr, "connect To %s %s\n", myConnectionPoint->getMyBaseUnitName().c_str(), otherConnectionPoint->getMyBaseUnitName().c_str());
        fprintf(stderr, "      my-enabled=%d other-enabled=%d my-connected=%d other-connected=%d\n", myConnectionPoint->isEnabled(), otherConnectionPoint->isEnabled(), myConnectionPoint->isConnected(), otherConnectionPoint->isConnected());
    }

    // check if connectionPoints are free
    if (!myConnectionPoint->isEnabled() || !otherConnectionPoint->isEnabled() || myConnectionPoint->isConnected() || otherConnectionPoint->isConnected())
    {
        return false;
    }

    // check if units are already connected
    if (isConnectedTo(otherUnit))
    {
        return false;
    }

    bool myConnection = myConnectionPoint->connectTo(otherConnectionPoint);
    bool otherConnection = otherConnectionPoint->connectTo(myConnectionPoint);
    if (cover->debugLevel(3))
        fprintf(stderr, "      myConn=%d otherConn=%d\n", myConnection, otherConnection);
    if (myConnection && otherConnection)
    {
        if (cover->debugLevel(3))
            fprintf(stderr, "      connected succesfull\n");
        sendConnectionToGUI(myConnectionPoint->getMyBaseUnitName(), myConnectionPoint->getConnectableBaseUnitName(), myConnectionPoint->isConnected(), true, otherConnectionPoint->getMyBaseUnit()->getObjectName());
        otherConnectionPoint->getMyBaseUnit()->sendConnectionToGUI(otherConnectionPoint->getMyBaseUnitName(), otherConnectionPoint->getConnectableBaseUnitName(), otherConnectionPoint->isConnected(), true, getObjectName());
        // show connection Geometry
        showConnectionGeom(true, myConnectionPoint->getMyBaseUnitName());
        otherConnectionPoint->getMyBaseUnit()->showConnectionGeom(true, otherConnectionPoint->getMyBaseUnitName());
    }
    else
    {
        return false;
    }

    return true;
}

/**
  * is called from to GUI to connect or disconnect baseUnits
  */
void DNABaseUnit::setConnection(std::string nameConnPoint, std::string nameConnPoint2, bool connected, bool enabled, DNABaseUnit *connObj, bool sendBack)
{
    if (cover->debugLevel(3))
    {
        if (connObj)
            fprintf(stderr, "DNABaseUnit::setConnection %s %s connected=%d enabled=%d to %s\n", nameConnPoint.c_str(), nameConnPoint2.c_str(), connected, enabled, connObj->getNameforMessage());
        else
            fprintf(stderr, "DNABaseUnit::setConnection %s %s connected=%d enabled=%d to NONE\n", nameConnPoint.c_str(), nameConnPoint2.c_str(), connected, enabled);
    }

    // find the correct connection point
    std::list<DNABaseUnitConnectionPoint *>::iterator it;
    for (it = connections_.begin(); it != connections_.end(); it++)
    {
        if ((*it)->getMyBaseUnitName().compare(nameConnPoint) == 0 && (*it)->getConnectableBaseUnitName().compare(nameConnPoint2) == 0)
        {
            (*it)->setEnabled(enabled);
            // disconnect
            // check if connectionpoint is connected
            if (!connected && (*it)->isConnected())
            {
                // send disconnect of other connectionpoint to gui
                DNABaseUnitConnectionPoint *op = (*it)->getConnectedPoint();
                // hide connection Geometry
                showConnectionGeom(false, nameConnPoint);
                if (op->isConnected())
                {
                    op->getMyBaseUnit()->showConnectionGeom(false, nameConnPoint2);
                    // disconnect both connectionpoints
                    op->disconnectBase();
                }
                (*it)->disconnectBase();
                // reposition
                osg::Vec3 pos = getPosition();
                osg::Vec3 connPos = op->getMyBaseUnit()->getPosition();
                osg::Vec3 rePos = (connPos - pos) * 0.5;
                osg::Matrix m = getMatrix();
                m.setTrans(pos - rePos);
                updateTransform(m);
                m = op->getMyBaseUnit()->getMatrix();
                m.setTrans(connPos + rePos);
                op->getMyBaseUnit()->updateTransform(m);
                if (sendBack)
                {
                    // send disconnect
                    op->getMyBaseUnit()->sendConnectionToGUI(op->getMyBaseUnitName(), op->getConnectableBaseUnitName(), op->isConnected(), op->isEnabled(), " ");
                    sendConnectionToGUI((*it)->getMyBaseUnitName(), (*it)->getConnectableBaseUnitName(), connected, enabled, " ");
                }
            }
            // connect
            else if (connected && !(*it)->isConnected())
            {
                //fprintf(stderr, "connect isn't connected yet %s\n", nameConnPoint.c_str());
                if (connObj == NULL)
                {
                    fprintf(stderr, "%s wrong second obj name for connection\n", nameConnPoint.c_str());
                    return;
                }
                DNABaseUnitConnectionPoint *op = connObj->getConnectionPoint(nameConnPoint2);
                connectTo(connObj, (*it), op);
                if (sendBack)
                    sendConnectionToGUI((*it)->getMyBaseUnitName(), (*it)->getConnectableBaseUnitName(), connected, enabled, connObj->getObjectName());
            }
            break;
        }
    }
}

/**
  * protected function to declare geometry
  */
void DNABaseUnit::createGeometry()
{
    // read geometry from file and scale it
    geometryNode = coVRFileManager::instance()->loadIcon(geofilename_.c_str());
    //fprintf(stderr, "%s = %f\n", geofilename_.c_str(), geometryNode->getBound().radius());

    // remove old geometry
    scaleTransform->removeChild(0, scaleTransform->getNumChildren());
    // add geometry to node
    scaleTransform->addChild(geometryNode.get());
    scaleTransform->setName(name_);
}

void DNABaseUnit::startInteraction()
{
    // tell all connected baseUnits that they are moving
    std::list<DNABaseUnitConnectionPoint *>::iterator it;
    for (it = connections_.begin(); it != connections_.end(); it++)
    {
        if ((*it)->isConnected())
            (*it)->getConnectedPoint()->getMyBaseUnit()->setMoving(true);
    }

    coVR3DTransRotInteractor::startInteraction();
}

void DNABaseUnit::stopInteraction()
{
    // tell all connected baseUnits that they are moving
    std::list<DNABaseUnitConnectionPoint *>::iterator it;
    for (it = connections_.begin(); it != connections_.end(); it++)
    {
        if ((*it)->isConnected())
            (*it)->getConnectedPoint()->getMyBaseUnit()->setMoving(false);
    }
    unsetMoveByInteraction();

    coVR3DTransRotInteractor::stopInteraction();

    // send Matrix to Gui
    sendUpdateTransform(moveTransform->getMatrix(), true, true);
}

void DNABaseUnit::doInteraction()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\nDNABaseUnit::doInteraction\n");
    unsetMoveByInteraction();

    coVR3DTransRotInteractor::doInteraction();
}

/**
  * set transformation matrix
  * calls all connected BaseUnits
  * and sends transformationmatrix to GUI
  */
void DNABaseUnit::sendUpdateTransform(osg::Matrix m, bool sendBack, bool recursive = true)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "sendUpdateTransform %s \n", geofilename_.c_str());

    coVR3DTransRotInteractor::updateTransform(m);

    movedByInteraction_ = true;

    if (recursive)
    {
        // call transform of all connected units
        std::list<DNABaseUnitConnectionPoint *>::iterator it;
        for (it = connections_.begin(); it != connections_.end(); it++)
        {
            //fprintf(stderr, "check conn %s\n", (*it)->getConnectableBaseUnitName());
            if ((*it)->isConnected())
            {
                (*it)->getConnectedPoint()->getMyBaseUnit()->sendTransform(m, (*it)->getConnectedPoint());
                //fprintf(stderr, "   call transform %s\n", (*it)->getConnectableBaseUnitName());
            }
        }
    }
    if (sendBack)
        sendMatrixToGUI();
}

/**
  * set transformation matrix
  * calls all connected BaseUnits
  */
void DNABaseUnit::updateTransform(osg::Matrix m)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "updateTransform %s \n", geofilename_.c_str());

    coVR3DTransRotInteractor::updateTransform(m);

    movedByInteraction_ = true;

    // call transform of all connected units
    std::list<DNABaseUnitConnectionPoint *>::iterator it;
    for (it = connections_.begin(); it != connections_.end(); it++)
    {
        //fprintf(stderr, "check conn %s\n", (*it)->getConnectableBaseUnitName());
        if ((*it)->isConnected())
        {
            (*it)->getConnectedPoint()->getMyBaseUnit()->transform(m, (*it)->getConnectedPoint());
            //fprintf(stderr, "   call transform %s\n", (*it)->getConnectableBaseUnitName());
        }
    }
}

void DNABaseUnit::unsetMoveByInteraction()
{
    // stop recursion if baseUnit was allready moved
    if (!movedByInteraction_)
        return;

    // call unsetMoveByInteraction of all connected units
    movedByInteraction_ = false;
    std::list<DNABaseUnitConnectionPoint *>::iterator it;
    for (it = connections_.begin(); it != connections_.end(); it++)
    {
        if ((*it)->isConnected())
        {
            (*it)->getConnectedPoint()->getMyBaseUnit()->unsetMoveByInteraction();
        }
    }
}

/**
  * calculates transformation depending from which connectionPoint transform is called
  */
osg::Matrix DNABaseUnit::calcTransform(osg::Matrix m, DNABaseUnitConnectionPoint *cp)
{
    // rotation from connection
    osg::Matrix rot;
    rot.makeRotate(cp->getNormal(), cp->getConnectedPoint()->getNormal());

    // translation from connection
    osg::Matrix trans;
    osg::Vec3 rotPoint = rot * cp->getPoint();
    osg::Vec3 transVec;

    //rotate for bases on the left side
    if (cp->getRotation() /*|| cp->getConnectedPoint()->getRotation()*/)
    {
        //rotiere um 180°
        transVec = cp->getConnectedPoint()->getPoint() * cp->getConnectedPoint()->getMyBaseUnit()->getRadius() + rotPoint * getRadius();
        trans.makeTranslate(transVec);
        trans.setRotate(rot180z);
    }
    // rotate desoxyribose if connected to base on the left side
    else if (cp->getConnectedPoint()->getRotation())
    {
        //rotiere um 180°
        transVec = cp->getConnectedPoint()->getPoint() * cp->getConnectedPoint()->getMyBaseUnit()->getRadius() + rotPoint * getRadius();
        trans.makeTranslate(transVec);
        trans.setRotate(rot180y);
    }
    else
    {
        transVec = cp->getConnectedPoint()->getPoint() * cp->getConnectedPoint()->getMyBaseUnit()->getRadius() - rotPoint * getRadius();
        trans.makeTranslate(transVec);
    }

    trans.postMult(rot);

    // rotate desoxyribose if connected to phosphat for inner rotatio of DNA
    if (cp->getMyBaseUnitName().compare("desoxybirose1") == 0)
    {
        //rotiere um 10°
        trans.postMult(rotMat10);
    }
    else if (cp->getMyBaseUnitName().compare("desoxybirose2") == 0)
    {
        //rotiere um 10°
        trans.postMult(rotMatm10);
    }

    trans.postMult(m);

    return trans;
}

void DNABaseUnit::sendTransform(osg::Matrix m, DNABaseUnitConnectionPoint *cp)
{
    // do not call transform if allready transformed
    if (movedByInteraction_)
        return;

    osg::Matrix trans = calcTransform(m, cp);

    sendUpdateTransform(trans, true);
}

void DNABaseUnit::transform(osg::Matrix m, DNABaseUnitConnectionPoint *cp)
{
    // do not call transform if allready transformed
    if (movedByInteraction_)
        return;

    osg::Matrix trans = calcTransform(m, cp);

    updateTransform(trans);
}

std::string DNABaseUnit::getObjectName()
{
    return scaleTransform->getName();
}

/**
  * creates the name which is needed for all messages to the GUI
  */
const char *DNABaseUnit::getNameforMessage()
{
    if (name_ == NULL)
    {
        stringstream idString;
        idString << _interactorName;
        idString << "_"
                 << "DNA"
                 << "_";
        idString << num_;
        name_ = new char[idString.str().size() + 1];
        strcpy(name_, idString.str().c_str());
    }

    string message = "DNA: \t";
    message.append(name_);
    message.append(";;");
    message.append("NULL");
    message.append("\t");

    char *msg = new char[message.size() + 1];
    strcpy(msg, message.c_str());
    return msg;
}

/**
  * sends transform matrix of baseUnti to GUI
  */
void DNABaseUnit::sendMatrixToGUI()
{
    if (presentationMode_)
        return;
    //send matrix to GUI
    if (coVRMSController::instance()->isMaster())
    {
        if (cover->debugLevel(3))
            fprintf(stderr, "DNABaseUnit::sendMatrixToGUI %s\n", scaleTransform->getName().c_str());

        osg::Matrix m = getMatrix();
        osg::Vec3 trans = m.getTrans();
        osg::Quat rot = m.getRotate();

        if (!registered_)
            registerAtGui();

        coGRObjMovedMsg moveMsg(getNameforMessage(), trans.x(), trans.y(), trans.z(), rot.x(), rot.y(), rot.z(), rot.w());
        coGRObjTransformMsg moveMsg2(getNameforMessage(), m(0, 0), m(1, 0), m(2, 0), m(3, 0), m(0, 1), m(1, 1), m(2, 1), m(3, 1), m(0, 2), m(1, 2), m(2, 2), m(3, 2), m(0, 3), m(1, 3), m(2, 3), m(3, 3));
        Message grmsg{ Message::UI, covise::DataHandle{(char*)(moveMsg.c_str()), strlen(moveMsg.c_str()) + 1 , false} }; +1;
        Message grmsg2{ Message::UI, covise::DataHandle{(char*)(moveMsg2.c_str()), strlen(moveMsg2.c_str()) + 1 , false} }; +1;
        coVRPluginList::instance()->sendVisMessage(&grmsg);
        coVRPluginList::instance()->sendVisMessage(&grmsg2);
    }
}

void DNABaseUnit::sendConnectionToGUI(std::string conn1, std::string conn2, int connected, int enabled, std::string connObj)
{
    //send connection to GUI
    if (coVRMSController::instance()->isMaster())
    {
        if (cover->debugLevel(3))
            fprintf(stderr, "DNABaseUnit::sendConnectionToGUI %s %s %s\n", conn1.c_str(), conn2.c_str(), connObj.c_str());

        if (!registered_)
            registerAtGui();

        coGRObjSetConnectionMsg setConnMsg(getNameforMessage(), conn1.c_str(), conn2.c_str(), connected, enabled, connObj.c_str());
        Message grmsg{ Message::UI, covise::DataHandle{(char*)(setConnMsg.c_str()), strlen(setConnMsg.c_str()) + 1 , false} }; +1;
        coVRPluginList::instance()->sendVisMessage(&grmsg);
    }
}

bool DNABaseUnit::disconnectAll()
{
    std::list<DNABaseUnitConnectionPoint *>::iterator it;
    for (it = connections_.begin(); it != connections_.end(); it++)
    {
        if ((*it)->isConnected())
        {
            (*it)->getConnectedPoint()->getMyBaseUnit()->setConnection((*it)->getConnectableBaseUnitName(), (*it)->getMyBaseUnitName(), false, true, NULL, false);
            setConnection((*it)->getMyBaseUnitName(), (*it)->getConnectableBaseUnitName(), false, true, NULL, false);
        }
    }
    return true;
}

bool DNABaseUnit::isConnected()
{
    std::list<DNABaseUnitConnectionPoint *>::iterator it;
    for (it = connections_.begin(); it != connections_.end(); it++)
    {
        if ((*it)->isConnected())
            return true;
    }
    return false;
}

void DNABaseUnit::setVisible(bool b)
{
    visible_ = b;
    if (b)
    {
        show();
        enableIntersection();
    }
    else
    {
        hide();
        disableIntersection();
    }
}

void DNABaseUnit::setMoving(bool b)
{
    if (isMoving_ == b)
        return;
    isMoving_ = b;
    // tell all connected baseUnits that they are moving
    std::list<DNABaseUnitConnectionPoint *>::iterator it;
    for (it = connections_.begin(); it != connections_.end(); it++)
    {
        if ((*it)->isConnected())
            (*it)->getConnectedPoint()->getMyBaseUnit()->setMoving(b);
    }
}
