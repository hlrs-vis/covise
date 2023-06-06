/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//--------------------------------------------------------------------------//
//--------------------------------------------------------------------------//
//                       Cyber Classrom                                     //
//                       Visenso GmbH                                       //
//                       2012                                               //
//                                                                          //
//$Id$
//--------------------------------------------------------------------------//
//--------------------------------------------------------------------------//

#include <osg/Vec3>
#include <osg/Geometry>
#include <osgText/Text>

#include <cover/VRSceneGraph.h>
#include <cover/coVRConfig.h>
#include <cover/coVRNavigationManager.h>
#include <grmsg/coGRObjVisMsg.h>
#include <grmsg/coGRObjTransformMsg.h>
#include <grmsg/coGRObjSetConnectionMsg.h>
#include <grmsg/coGRObjMovedMsg.h>
#include <grmsg/coGRKeyWordMsg.h>
#include <OpenVRUI/osg/OSGVruiMatrix.h>
#include <OpenVRUI/coRowMenu.h>
#include <config/CoviseConfig.h>
#include <cover/coVRPluginSupport.h>

#include <cover/coTranslator.h>

#include "DNAPlugin.h"
#include "DNABaseUnit.h"
#include "DNABaseUnitConnectionPoint.h"
#include "DNADesoxyribose.h"
#include "DNAPhosphat.h"
#include "DNAAdenin.h"
#include "DNACytosin.h"
#include "DNAGuanin.h"
#include "DNAThymin.h"

using namespace covise;
using namespace grmsg;
using namespace vrui;
using namespace opencover;

DNAPlugin::DNAPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    if (cover->debugLevel(2))
    {
        fprintf(stderr, "\nDNAPlugin::DNAPlugin\n");
    }
}

DNAPlugin::~DNAPlugin()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\nDNAPlugin::~DNAPlugin\n");

    delete menu_;
    delete disconnMenuItem_;
    for (std::list<DNABaseUnit *>::iterator it = dnaBaseUnits.begin(); it != dnaBaseUnits.end(); it++)
    {
        delete *it;
    }
}

bool DNAPlugin::init()
{

    if (cover->debugLevel(2))
        fprintf(stderr, "\nDNAPlugin::init\n");

    // load 1 adenin
    // load 1 thymin
    // load 1 guanin
    // load 1 cytosin
    // load 6 phosphat
    // load 4 desoxyribose
    osg::Matrix m;
    m.makeIdentity();
    for (int i = 0; i < 6; i++)
    {
        dnaBaseUnits.push_back(new DNAPhosphat(m, i));
        if (i < 4)
            dnaBaseUnits.push_back(new DNADesoxyribose(m, i));
        if (i < 1)
        {
            dnaBaseUnits.push_back(new DNAAdenin(m, i));
            dnaBaseUnits.push_back(new DNACytosin(m, i));
            dnaBaseUnits.push_back(new DNAGuanin(m, i));
            dnaBaseUnits.push_back(new DNAThymin(m, i));
        }
    }

    //create menu
    if (cover->debugLevel(3))
        fprintf(stderr, "DNAPlugin::makeMenu\n");

    OSGVruiMatrix matrix, transMatrix, rotateMatrix, scaleMatrix;
    menu_ = new coRowMenu("DNA");

    double px = (double)coCoviseConfig::getFloat("x", "COVER.Menu.Position", -1000);
    double py = (double)coCoviseConfig::getFloat("y", "COVER.Menu.Position", 0);
    double pz = (double)coCoviseConfig::getFloat("z", "COVER.Menu.Position", 600);

    px = (double)coCoviseConfig::getFloat("x", "COVER.Plugin.DNA.MenuPosition", px);
    py = (double)coCoviseConfig::getFloat("y", "COVER.Plugin.DNA.MenuPosition", py);
    pz = (double)coCoviseConfig::getFloat("z", "COVER.Plugin.DNA.MenuPosition", pz);

    barrierAngle = (float)coCoviseConfig::getFloat("value", "COVER.Plugin.DNA.BarrierAngle", 0.1);

    // default is DNA.MenuSize then COVER.Menu.Size then 1.0
    float s = coCoviseConfig::getFloat("value", "COVER.Menu.Size", 1.0);
    s = coCoviseConfig::getFloat("value", "COVER.Plugin.DNA.MenuSize", s);

    transMatrix.makeTranslate(px, py, pz);
    rotateMatrix.makeEuler(0, 90, 0);
    scaleMatrix.makeScale(s, s, s);

    matrix.makeIdentity();
    matrix.mult(&scaleMatrix);
    matrix.mult(&rotateMatrix);
    matrix.mult(&transMatrix);

    menu_->setTransformMatrix(&matrix);
    menu_->setScale(cover->getSceneSize() / 2500);
    menu_->setAttachment(coUIElement::RIGHT);

    string languageEntry = coCoviseConfig::getEntry("COVER.Plugin.DNA.Language");
    disconnMenuItem_ = new coButtonMenuItem(coTranslator::coTranslate("disconnect all"));
    // show menu item
    disconnMenuItem_->setMenuListener(this);
    menu_->add(disconnMenuItem_);

    menu_->setVisible(false);

    // for message in renderer
    hud_ = opencover::coHud::instance();
    justConnected_ = 0;
    showHud_ = 0.0;
    hudNotForward_ = false;
    return true;
}

void DNAPlugin::menuEvent(coMenuItem *menuItem)
{

    if (cover->debugLevel(3))
        fprintf(stderr, "DNAPlugin::menuEvent for %s\n", menuItem->getName());

    if (menuItem == disconnMenuItem_)
    {
        for (std::list<DNABaseUnit *>::iterator it = dnaBaseUnits.begin(); it != dnaBaseUnits.end(); it++)
        {
            if ((*it)->isVisible() && (*it)->isConnected())
                (*it)->disconnectAll();
        }
    }
}

void DNAPlugin::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- DNAPlugin::guiToRenderMsg\n %s\n\n", msg.getString().c_str());


    if (msg.isValid())
    {
        switch (msg.getType())
        {
        case coGRMsg::GEO_VISIBLE: // visible message from gui
        {
            auto &geometryVisibleMsg = msg.as<coGRObjVisMsg>();
            const char *objectName = geometryVisibleMsg.getObjName();

            // check if message is for the plugin
            if (strstr(objectName, "_DNA_") == NULL)
                return;

            if (cover->debugLevel(3))
                fprintf(stderr, "in DNA coGRMsg::GEO_VISIBLE object=%s visible=%d\n", objectName, geometryVisibleMsg.isVisible());

            std::list<DNABaseUnit *>::iterator it;
            bool visible = false; // flag if menu is visible
            bool setVisible = geometryVisibleMsg.isVisible();
            for (it = dnaBaseUnits.begin(); it != dnaBaseUnits.end(); it++)
            {
                if ((*it)->getObjectName().compare(objectName) == 0)
                {
                    (*it)->setVisible(setVisible);
                    std::list<DNABaseUnitConnectionPoint *> connPoints = (*it)->getAllConnectionPoints();
                    if (setVisible)
                    {
                        for (std::list<DNABaseUnitConnectionPoint *>::iterator connIt = connPoints.begin(); connIt != connPoints.end(); connIt++)
                        {
                            if ((*connIt)->isEnabled() && !(*connIt)->isConnected())
                                availableConnectionPoints.push_back(*connIt);
                        }
                        visible = true;
                    }
                    else
                    {
                        for (std::list<DNABaseUnitConnectionPoint *>::iterator connIt = connPoints.begin(); connIt != connPoints.end(); connIt++)
                        {
                            availableConnectionPoints.remove(*connIt);
                        }
                    }
                    break;
                }
                if ((*it)->isVisible())
                    visible = true;
            }
            menu_->setVisible(visible);
            VRSceneGraph::instance()->applyMenuModeToMenus(); // apply menuMode state to menus just made visible
        }
        break;
        case coGRMsg::KEYWORD: // new presentation step message -> disconnect all
        {
            auto &keywordmsg = msg.as<coGRKeyWordMsg>();
            const char *keyword = keywordmsg.getKeyWord();
            if ((strcmp(keyword, "presForward") == 0) || (strcmp(keyword, "presBackward") == 0) || (strcmp(keyword, "goToStep") == 0))
            {
                //fprintf(stderr, "------------------------\n");
                justConnected_ = 0;
                for (std::list<DNABaseUnit *>::iterator it = dnaBaseUnits.begin(); it != dnaBaseUnits.end(); it++)
                {
                    (*it)->disconnectAll();
                    (*it)->unsetMoveByInteraction();
                }
                availableConnectionPoints.clear();
            }
            else if (strcmp(keyword, "showNotReady") == 0)
            {
                //fprintf(stderr, "showNotReady\n");
                showHud_ += cover->frameDuration();
                hud_->setText1(coTranslator::coTranslate("Die Aufgabe ist falsch geloest! \nVersuchen Sie es noch einmal."));
                hud_->show();
                hud_->redraw();
                hudNotForward_ = true;
            }
        }
        break;
        case coGRMsg::TRANSFORM_OBJECT: // transform message from gui
        {
            auto &geometryTransformMsg = msg.as<coGRObjTransformMsg>();
            const char *objectName = geometryTransformMsg.getObjName();

            if (cover->debugLevel(3))
                fprintf(stderr, "in DNA coGRMsg::TRANSFORM_OBJECT object=%s \n", objectName);

            osg::Matrix m = osg::Matrix(geometryTransformMsg.getMatrix(0, 0), geometryTransformMsg.getMatrix(1, 0), geometryTransformMsg.getMatrix(2, 0), geometryTransformMsg.getMatrix(3, 0), geometryTransformMsg.getMatrix(0, 1), geometryTransformMsg.getMatrix(1, 1), geometryTransformMsg.getMatrix(2, 1), geometryTransformMsg.getMatrix(3, 1), geometryTransformMsg.getMatrix(0, 2), geometryTransformMsg.getMatrix(1, 2), geometryTransformMsg.getMatrix(2, 2), geometryTransformMsg.getMatrix(3, 2), geometryTransformMsg.getMatrix(0, 3), geometryTransformMsg.getMatrix(1, 3), geometryTransformMsg.getMatrix(2, 3), geometryTransformMsg.getMatrix(3, 3));

            std::list<DNABaseUnit *>::iterator it;
            for (it = dnaBaseUnits.begin(); it != dnaBaseUnits.end(); it++)
            {
                if ((*it)->getObjectName().compare(objectName) == 0)
                {
                    (*it)->unsetMoveByInteraction();
                    (*it)->sendUpdateTransform(m, true, false);
                    break;
                }
            }
        }
        break;
        case coGRMsg::SET_CONNECTIONPOINT: // got connection form gui
        {
            auto &connMsg = msg.as<coGRObjSetConnectionMsg>();
            std::string objectName = connMsg.getObjName();
            std::string secondObj = connMsg.getSecondObjName();

            if (cover->debugLevel(3))
            {
                if (connMsg.isConnected())
                    fprintf(stderr, "in DNA coGRMsg::SET_CONNECTIONPOINT object=%s with %s \n", objectName.c_str(), secondObj.c_str());
            }

            std::list<DNABaseUnit *>::iterator it, it2;
            for (it = dnaBaseUnits.begin(); it != dnaBaseUnits.end(); it++)
            {
                if ((*it)->getObjectName().compare(objectName) == 0)
                {
                    break;
                }
            }
            if (secondObj.size() != 1)
            {
                for (it2 = dnaBaseUnits.begin(); it2 != dnaBaseUnits.end(); it2++)
                {
                    if ((*it2)->getObjectName().compare(secondObj) == 0)
                    {
                        (*it)->setConnection(connMsg.getConnPoint1(), connMsg.getConnPoint2(), connMsg.isConnected()!=0, connMsg.isEnabled(), (*it2), false);
                        // remove bothe connPoints from available list
                        if (connMsg.isConnected())
                        {
                            availableConnectionPoints.remove((*it)->getConnectionPoint(connMsg.getConnPoint1()));
                            availableConnectionPoints.remove((*it2)->getConnectionPoint(connMsg.getConnPoint2()));
                        }
                        // if disconnected add to available list
                        else
                        {
                            if (connMsg.isEnabled())
                            {
                                availableConnectionPoints.push_back((*it)->getConnectionPoint(connMsg.getConnPoint1()));
                                availableConnectionPoints.push_back((*it2)->getConnectionPoint(connMsg.getConnPoint2()));
                            }
                            else
                            {
                                availableConnectionPoints.remove((*it)->getConnectionPoint(connMsg.getConnPoint1()));
                                availableConnectionPoints.remove((*it2)->getConnectionPoint(connMsg.getConnPoint2()));
                            }
                        }
                        break;
                    }
                }
            }
            else
            {
                // first connectionPoint is enabled/disabled
                (*it)->setConnection(connMsg.getConnPoint1(), connMsg.getConnPoint2(), connMsg.isConnected(), connMsg.isEnabled()!=0, NULL, false);
                if (connMsg.isEnabled())
                    availableConnectionPoints.push_back((*it)->getConnectionPoint(connMsg.getConnPoint1()));
                else
                    availableConnectionPoints.remove((*it)->getConnectionPoint(connMsg.getConnPoint1()));
            }
        }
        break;
        default:
            break;
        }
    }
}

void DNAPlugin::preFrame()
{
    // go through alle connectionPoints and check if there are any possible Connections left
    int possibleConnections = 0;
    for (std::list<DNABaseUnitConnectionPoint *>::iterator connPointIt = availableConnectionPoints.begin(); connPointIt != availableConnectionPoints.end(); connPointIt++)
    {
        for (std::list<DNABaseUnitConnectionPoint *>::iterator connPointIt2 = connPointIt; connPointIt2 != availableConnectionPoints.end(); connPointIt2++)
        {
            if ((*connPointIt)->getConnectableBaseUnitName().compare((*connPointIt2)->getMyBaseUnitName()) == 0)
                possibleConnections++;
            if (possibleConnections > 0)
                break;
        }
        if (possibleConnections > 0)
            break;
    }
    //fprintf(stderr, "possConn=%d justConn=%d showHud=%d\n", possibleConnections, justConnected_, showHud_);

    // check if we have to show a hud
    if (showHud_ > 2.5f)
    {
        justConnected_ = 0;
        hudNotForward_ = false;
        showHud_ = 0.0;
        hud_->hide();
    }
    else if (showHud_ > 0.0)
        showHud_ += cover->frameDuration();

    std::list<DNABaseUnit *>::iterator it, it2;
    for (it = dnaBaseUnits.begin(); it != dnaBaseUnits.end(); it++)
    {
        // check if baseUnit is moving
        if ((*it)->isRunning() || ((*it)->isMoving() && (*it)->isVisible()))
        {
            //fprintf(stderr, "possConn=%d justConn=%d showHud=%d\n", possibleConnections, justConnected_, showHud_);
            // if no possible connections are left, show text
            if (possibleConnections == 0 && (justConnected_ > 20 || justConnected_ == 0) && !hudNotForward_)
            {
                justConnected_ = 0;
                if (showHud_ == 0.0)
                    showHud_ += cover->frameDuration();
                hud_->setText1(coTranslator::coTranslate("Keine moegliche Kombination mehr vorhanden!"));
                hud_->show();
                hud_->redraw();
                break;
            }
            // wait 20 frames from last connection to show the hud-message
            else if (possibleConnections == 0 && (justConnected_ <= 20 && justConnected_ != 0) && !hudNotForward_)
            {
                justConnected_++;
                break;
            }

            // check collision with all baseUnits which are not moving
            osg::Vec3 p1 = (*it)->getPosition();
            for (it2 = dnaBaseUnits.begin(); it2 != dnaBaseUnits.end(); it2++)
            {
                if ((*it2) != (*it) && !(*it2)->isRunning() /*&& !(*it2)->isMoving()*/ && (*it2)->isVisible())
                {
                    osg::Vec3 p2 = (*it2)->getPosition();

                    // check all distances
                    float minDistance = (*it2)->getRadius() + (*it)->getRadius();
                    osg::Vec3 dir = (p1 - p2);
                    float dist = dir.length();
                    dir.normalize();
                    float collisionDist = (*it2)->getRadius() + (*it)->getRadius();

                    if (dist < collisionDist)
                    {
                        // get connectionlists of both base
                        list<DNABaseUnitConnectionPoint *> conn = (*it)->getAllConnectionPoints();
                        list<DNABaseUnitConnectionPoint *> conn2 = (*it2)->getAllConnectionPoints();

                        // check if the two connectionPoints of these two base are connactable
                        list<DNABaseUnitConnectionPoint *>::iterator itConn, itConn2;
                        for (itConn = conn.begin(); itConn != conn.end(); itConn++)
                        {
                            // do nothing if connectionPoint already has a connection
                            if (!(*itConn)->isConnected())
                            {
                                for (itConn2 = conn2.begin(); itConn2 != conn2.end(); itConn2++)
                                {
                                    if (!(*itConn2)->isConnected() && ((*itConn)->getMyBaseUnitName().compare((*itConn2)->getConnectableBaseUnitName()) == 0))
                                    {
                                        // we found a connection
                                        // check if connectionPoints are close enough and the angle
                                        osg::Vec3 cp1 = (*it)->getPosition() + (*itConn)->getPoint() * (*it)->getRadius();
                                        osg::Vec3 cp2 = (*it2)->getPosition() + (*itConn2)->getPoint() * (*it2)->getRadius();
                                        float distanceCPoints = (cp1 - cp2).length();
                                        float angleShould = (*itConn)->getNormal() * (*itConn2)->getNormal();
                                        float angleReal = ((*it)->getMatrix().getRotate() * (*itConn)->getNormal()) * ((*it2)->getMatrix().getRotate() * (*itConn2)->getNormal());
                                        //fprintf(stderr, "angleShould=%f realAngle=%f\n", angleShould, angleReal);
                                        if (distanceCPoints < minDistance && (angleShould - angleReal < barrierAngle || angleShould - angleReal > -1 * barrierAngle))
                                        {
                                            //fprintf(stderr, "minDistance %f / distance %f\n", minDistance, distanceCPoints);

                                            (*it)->connectTo(*it2, *itConn, *itConn2);
                                            justConnected_ = 1;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        // collision detection if parts are not connected
                        if (!(*it)->isConnectedTo(*it2))
                        {
                            // stopp moving of base if it's too close to another base
                            osg::Matrix m = (*it)->getMatrix();
                            m.setTrans((*it2)->getPosition() + dir * collisionDist * 1.05);
                            (*it)->unsetMoveByInteraction();
                            (*it)->sendUpdateTransform(m, true, true);
                        }
                    }
                }
            }
        }
    }
}

COVERPLUGIN(DNAPlugin)
