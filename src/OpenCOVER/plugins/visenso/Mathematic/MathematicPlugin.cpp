/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2009 Visenso  **
 **                                                                        **
 ** Description: MathematicPlugin                                          **
 **              for VR4Schule mathematics                                 **
 **                                                                        **
 ** Author: M. Theilacker                                                  **
 **                                                                        **
 ** History:                                                               **
 **     4.2009 initial version                                             **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include "MathematicPlugin.h"

#include <osg/Vec3>
#include <osg/LineWidth>

#include <cover/VRSceneGraph.h>
#include <cover/coTranslator.h>

const int NUM_GEOMETRY = 2;
const double MATH_PI = 3.14159265358979323846264338327950;

using namespace osg;
using namespace covise;

MathematicPlugin *MathematicPlugin::plugin = NULL;

//
// Constructor
//
MathematicPlugin::MathematicPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, boundary_(10.0)
, showAxisMenuItem_(NULL)
, hideLabelsMenuItem_(NULL)
, addPointMenuItem_(NULL)
, addLineMenuItem_(NULL)
, addPlaneMenuItem_(NULL)
, deleteButtonNum_(-1)
, stateLabel_(NULL)
, sepMainMenu_(NULL)
, mainMenuSepPos_(0)
, axis_(NULL)
, boundingBox_(NULL)
, geometryCount_(0)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nMathematicPlugin::MathematicPlugin\n");
}

//
// Destructor
//
MathematicPlugin::~MathematicPlugin()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nMathematicPlugin::~MathematicPlugin\n");

    delete axis_;
    delete mathematicsMenu_;
    delete showAxisMenuItem_;
    delete hideLabelsMenuItem_;
    delete addPointMenuItem_;
    delete addLineMenuItem_;
    delete addPlaneMenuItem_;
    delete sepMainMenu_;

    coVRLine::deleteBoundingBox();
    delete boundingBox_;

    delete stateLabel_;

    for (size_t i = 0; i < pointsVec_.size(); i++)
        delete pointsVec_.at(i);

    for (size_t i = 0; i < linesVec_.size(); i++)
        delete linesVec_.at(i);

    for (size_t i = 0; i < planesVec_.size(); i++)
        delete planesVec_.at(i);

    for (size_t i = 0; i < distancesVec_.size(); i++)
        delete distancesVec_.at(i);

    for (size_t i = 0; i < isectPoints_.size(); i++)
        delete isectPoints_.at(i);

    for (size_t i = 0; i < isectLines_.size(); i++)
        delete isectLines_.at(i);
}

//
// INIT
//
bool MathematicPlugin::init()
{
    if (plugin)
        return false;

    if (cover->debugLevel(3))
        fprintf(stderr, "\nMathematicPlugin::MathematicPlugin\n");

    // set plugin
    MathematicPlugin::plugin = this;

    // rotate scene so that x axis is in front
    MatrixTransform *trafo = VRSceneGraph::instance()->getTransform();
    Matrix m;
    m.makeRotate(inDegrees(-90.0), 0.0, 0.0, 1.0);
    trafo->setMatrix(m);

    // bounding box
    boundary_ = (double)coCoviseConfig::getFloat("max", "COVER.Plugin.Mathematic.Boundary", boundary_);
    boundingBox_ = new BoundingBox(-boundary_, -boundary_, -boundary_,
                                   boundary_, boundary_, boundary_);
    coVRPoint::setBoundingBox(boundingBox_);
    coVRLine::setBoundingBox(boundingBox_);
    coVRPlane::setBoundingBox(boundingBox_);
    coVRDirection::setBoundingBox(boundingBox_);
    coVRDistance::setBoundingBox(boundingBox_);
    drawBoundingBox(boundingBox_);

    // mathematics menu
    makeMathematicsMenu();

    // make the menus state label, it shows which objects intersects, etc
    stateLabel_ = new coLabelMenuItem(" ");
    mathematicsMenu_->insert(stateLabel_, mainMenuSepPos_ - 1);

    // colored axis
    axis_ = new coVRCoordinateAxis(0.07, boundary_, true);

    // zoom scene
    VRSceneGraph::instance()->viewAll();

    return true;
}

//----------------------------------------------------------------------
void MathematicPlugin::makeMathematicsMenu()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::makeMathematicsMenu\n");

    OSGVruiMatrix matrix, transMatrix, rotateMatrix, scaleMatrix;
    string text(coTranslator::coTranslate("Mathematik"));
    mathematicsMenu_ = new coRowMenu(text.c_str());
    mathematicsMenu_->setVisible(true);
    mathematicsMenu_->setAttachment(coUIElement::RIGHT);

    //position the menu
    double px = (double)coCoviseConfig::getFloat("x", "COVER.Menu.Position", -1000);
    double py = (double)coCoviseConfig::getFloat("y", "COVER.Menu.Position", 0);
    double pz = (double)coCoviseConfig::getFloat("z", "COVER.Menu.Position", 600);

    px = (double)coCoviseConfig::getFloat("x", "COVER.Plugin.Mathematic.MenuPosition", px);
    py = (double)coCoviseConfig::getFloat("y", "COVER.Plugin.Mathematic.MenuPosition", py);
    pz = (double)coCoviseConfig::getFloat("z", "COVER.Plugin.Mathematic.MenuPosition", pz);

    // default is Mathematic.MenuSize then COVER.Menu.Size then 1.0
    float s = coCoviseConfig::getFloat("value", "COVER.Menu.Size", 1.0);
    s = coCoviseConfig::getFloat("value", "COVER.Plugin.Mathematic.MenuSize", s);

    transMatrix.makeTranslate(px, py, pz);
    rotateMatrix.makeEuler(0, 90, 0);
    scaleMatrix.makeScale(s, s, s);

    matrix.makeIdentity();
    matrix.mult(&scaleMatrix);
    matrix.mult(&rotateMatrix);
    matrix.mult(&transMatrix);

    mathematicsMenu_->setTransformMatrix(&matrix);
    mathematicsMenu_->setScale(cover->getSceneSize() / 2500);

    // show axis menu item
    text = coTranslator::coTranslate("zeige Achsen");
    showAxisMenuItem_ = new coCheckboxMenuItem(text.c_str(), true);
    showAxisMenuItem_->setMenuListener(this);
    mathematicsMenu_->add(showAxisMenuItem_);

    // hide labels
    text = coTranslator::coTranslate("zeige Labels");
    hideLabelsMenuItem_ = new coCheckboxMenuItem(text.c_str(), true);
    hideLabelsMenuItem_->setMenuListener(this);
    mathematicsMenu_->add(hideLabelsMenuItem_);

    // add point menu item
    text = coTranslator::coTranslate("Punkt hinzufuegen");
    addPointMenuItem_ = new coButtonMenuItem(text.c_str());
    addPointMenuItem_->setMenuListener(this);
    mathematicsMenu_->add(addPointMenuItem_);

    // add line menu item
    text = coTranslator::coTranslate("Gerade hinzufuegen");
    addLineMenuItem_ = new coButtonMenuItem(text.c_str());
    addLineMenuItem_->setMenuListener(this);
    mathematicsMenu_->add(addLineMenuItem_);

    // add plane menu item
    text = coTranslator::coTranslate("Ebene hinzufuegen");
    addPlaneMenuItem_ = new coButtonMenuItem(text.c_str());
    addPlaneMenuItem_->setMenuListener(this);
    mathematicsMenu_->add(addPlaneMenuItem_);

    // seperator to rest menu
    sepMainMenu_ = new coLabelMenuItem("______________");
    mathematicsMenu_->add(sepMainMenu_);

    // keep end of main menu position
    mainMenuSepPos_ = mathematicsMenu_->getItemCount();

    mathematicsMenu_->setVisible(true);
    mathematicsMenu_->show();
}

//----------------------------------------------------------------------
void MathematicPlugin::preFrame()
{
    //fprintf(stderr,"MathematicPlugin::preFrame\n");

    // delete menu button in next frame (so that the menu is not inconsistent)
    if (deleteButtonNum_ >= 0)
    {
        //fprintf(stderr,"MathematicPlugin::preFrame deleteButton %d size=%d\n", deleteButtonNum_, (int)deleteButtonsVec_.size());

        delete deleteButtonsVec_.at(deleteButtonNum_);
        deleteButtonsVec_.erase(deleteButtonsVec_.begin() + deleteButtonNum_);
        deleteButtonNum_ = -1;
    }

    // preframe of all points
    for (size_t i = 0; i < pointsVec_.size(); i++)
        pointsVec_.at(i)->preFrame();

    // preframe of all lines
    for (size_t i = 0; i < linesVec_.size(); i++)
        linesVec_.at(i)->preFrame();

    // preframe of all planes
    for (size_t i = 0; i < planesVec_.size(); i++)
        planesVec_.at(i)->preFrame();

    // preframe of all isect points
    for (size_t i = 0; i < isectPoints_.size(); i++)
        isectPoints_.at(i)->preFrame();

    // preframe of all isect lines
    for (size_t i = 0; i < isectLines_.size(); i++)
        isectLines_.at(i)->preFrame();

    // preframe of all distances
    for (size_t i = 0; i < distancesVec_.size(); i++)
        distancesVec_.at(i)->preFrame();

    // distance between point and point
    if (pointsVec_.size() == 2)
        distancesVec_.at(0)->update(pointsVec_.at(0)->getPosition(), pointsVec_.at(1)->getPosition());

    // distance between line and point
    if (pointsVec_.size() == 1 && linesVec_.size() == 1)
    {
        Vec3 perpendicular = Vec3(0.0, 0.0, 0.0);
        linesVec_.at(0)->distance(pointsVec_.at(0), &perpendicular);
        distancesVec_.at(0)->update(perpendicular, pointsVec_.at(0)->getPosition());
    }

    bool line0 = false;
    bool line1 = false;
    bool line2 = false;

    // test for line intersections, parallel or skew
    // distance line line
    if (linesVec_.size() == 2)
    {
        line0 = linesVec_.at(0)->isChanged();
        line1 = linesVec_.at(1)->isChanged();

        // g0 and g1
        if (line0 || line1)
        {
            testLines(0, 1);
            if (statesVec_.at(0) == coVRLine::INTERSECT)
            {
                // distance goes from isectpoint to isectpoint
                Vec3 p = isectPoints_.at(0)->getPosition();
                distancesVec_.at(0)->update(p, p);
                distancesVec_.at(0)->setVisible(false);
            }
            else
            {
                Vec3 point1, point2;
                linesVec_.at(0)->distance(linesVec_.at(1), &point1, &point2);
                distancesVec_.at(0)->update(point1, point2);
                //fprintf(stderr,"MathematicPlugin::preFrame p1(%f %f %f)\n", point1.x(), point1.y(), point1.z());
                //fprintf(stderr,"MathematicPlugin::preFrame p2(%f %f %f)\n", point2.x(), point2.y(), point2.z());
            }
        }
    }

    // more than 2 lines (TODO: distance)
    if (linesVec_.size() == 3)
    {
        line2 = linesVec_.at(2)->isChanged();
        // g1 and g2
        if (line1 || line2)
            testLines(1, 2);

        // g2 and g0
        if (line2 || line0)
            testLines(2, 0);
    }

    // distance between plane and point
    if (planesVec_.size() == 1 && pointsVec_.size() == 1)
    {
        Vec3 perpendicular = Vec3(0.0, 0.0, 0.0);
        planesVec_.at(0)->distance(pointsVec_.at(0), &perpendicular);
        distancesVec_.at(0)->update(perpendicular, pointsVec_.at(0)->getPosition());
    }

    // test for plane and line intersections or parallel
    // distance plane line
    if (planesVec_.size() == 1 && linesVec_.size() == 1)
    {
        bool line = linesVec_.at(0)->isChanged();
        bool plane = planesVec_.at(0)->isChanged();

        // E0 and g0
        if (plane || line)
        {
            testPlaneLine(0, 0);

            if (statesVec_.at(0) == coVRPlane::INTERSECT)
            {
                // distance goes from isectpoint to isectpoint
                Vec3 p = isectPoints_.at(0)->getPosition();
                distancesVec_.at(0)->update(p, p);
                distancesVec_.at(0)->setVisible(false);
            }
            else if (statesVec_.at(0) == coVRPlane::PARALLEL)
            {
                Vec3 point1, point2;
                if (planesVec_.at(0)->distance(linesVec_.at(0), &point1, &point2))
                {
                    distancesVec_.at(0)->update(point1, point2);
                    //fprintf(stderr,"MathematicPlugin::preFrame p1(%f %f %f)\n", point1.x(), point1.y(), point1.z());
                    //fprintf(stderr,"MathematicPlugin::preFrame p2(%f %f %f)\n", point2.x(), point2.y(), point2.z());
                }
            }
            else if (statesVec_.at(0) == coVRPlane::LIESIN)
            {
                // distance is 0
                Vec3 p = linesVec_.at(0)->getBasePoint();
                distancesVec_.at(0)->update(p, p);
                distancesVec_.at(0)->setVisible(false);
            }
        }
    }

    // test for plane intersections or parallel
    // distance plane plane
    if (planesVec_.size() == 2)
    {
        bool plane0 = planesVec_.at(0)->isChanged();
        bool plane1 = planesVec_.at(1)->isChanged();

        // E0 and E1
        if (plane0 || plane1)
        {
            //fprintf(stderr,"MathematicPlugin::preFrame planesVec_.size()==2 plane0 %d plane1 %d \n", plane0, plane1 );
            testPlanes(0, 1);
            if (statesVec_.at(0) == coVRPlane::INTERSECT)
            {
                // distance goes from isectpoint to isectpoint
                Vec3 p = isectLines_.at(0)->getBasePoint();
                distancesVec_.at(0)->update(p, p);
                distancesVec_.at(0)->setVisible(false);
            }
            else
            {
                Vec3 point1, point2;
                planesVec_.at(0)->distance(planesVec_.at(1), &point1, &point2);

                if (planesVec_.at(0)->distance(planesVec_.at(1), &point1, &point2))
                {
                    distancesVec_.at(0)->update(point1, point2);
                    //fprintf(stderr,"MathematicPlugin::preFrame p1(%f %f %f)\n", point1.x(), point1.y(), point1.z());
                    //fprintf(stderr,"MathematicPlugin::preFrame p2(%f %f %f)\n", point2.x(), point2.y(), point2.z());
                }
            }
        }
    }

    updateStateLabel();
}

//----------------------------------------------------------------------
void MathematicPlugin::testLines(int line1, int line2)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::testLines %d %d\n", line1, line2);

    Vec3 isectpoint = Vec3(0.0, 0.0, 0.0);
    double angle;
    statesVec_.at(line1) = linesVec_.at(line1)->test(linesVec_.at(line2), &isectpoint, &angle);

    if (statesVec_.at(line1) == coVRLine::INTERSECT)
    {
        isectPoints_.at(line1)->setPosition(isectpoint);
        isectPoints_.at(line1)->setVisible(true);
        anglesVec_.at(line1) = angle;
    }
    else
        isectPoints_.at(line1)->setVisible(false);
}

//----------------------------------------------------------------------
void MathematicPlugin::testPlaneLine(int plane, int line)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::testPlaneLine %d %d\n", plane, line);

    Vec3 isectpoint = Vec3(0.0, 0.0, 0.0);
    double angle;
    statesVec_.at(plane) = planesVec_.at(plane)->test(linesVec_.at(line), &isectpoint, &angle);

    if (statesVec_.at(plane) == coVRLine::INTERSECT)
    {
        isectPoints_.at(plane)->setPosition(isectpoint);
        isectPoints_.at(plane)->setVisible(true);
        anglesVec_.at(plane) = angle;
    }
    else
        isectPoints_.at(plane)->setVisible(false);
}

//----------------------------------------------------------------------
void MathematicPlugin::testPlanes(int plane1, int plane2)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::testPlanes %d %d\n", plane1, plane2);

    Vec3 isectLinePoint1 = Vec3(0.0, 0.0, 0.0);
    Vec3 isectLinePoint2 = Vec3(0.0, 0.0, 0.0);
    double angle = 0.0;

    // test intersection
    statesVec_.at(plane1) = planesVec_.at(plane1)->test(planesVec_.at(plane2), &isectLinePoint1, &isectLinePoint2, &angle);

    if (statesVec_.at(plane1) == coVRPlane::INTERSECT)
    {
        isectLines_.at(plane1)->setPoints(isectLinePoint1, isectLinePoint2);
        isectLines_.at(plane1)->setVisible(true);
        anglesVec_.at(plane1) = angle;
    }
    else
    {
        isectLines_.at(plane1)->setVisible(false);
    }
}

//----------------------------------------------------------------------
void MathematicPlugin::makeLineIntersections(int line1, int line2)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::makeLineIntersections %d %d\n", line1, line2);

    Vec3 isectpoint = Vec3(0.0, 0.0, 0.0);
    double angle = 0.0;

    // test intersection
    int state = linesVec_.at(line1)->test(linesVec_.at(line2), &isectpoint, &angle);
    isectPoints_.push_back(new coVRPoint(isectpoint, "S", false));
    statesVec_.push_back(state);
    anglesVec_.push_back(angle);

    // show isect point if intersection
    if (statesVec_.back() == coVRLine::INTERSECT)
        isectPoints_.back()->setVisible(true);
}

//----------------------------------------------------------------------
void MathematicPlugin::makePlaneLineIntersections(int plane, int line)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::makePlaneLineIntersections %d %d\n", plane, line);

    Vec3 isectpoint = Vec3(0.0, 0.0, 0.0);
    double angle = 0.0;

    // test intersection
    int state = planesVec_.at(plane)->test(linesVec_.at(line), &isectpoint, &angle);
    isectPoints_.push_back(new coVRPoint(isectpoint, "S", false));

    //fprintf(stderr,"MathematicPlugin::makePlaneLineIntersections state %d\n", state );

    statesVec_.push_back(state);
    anglesVec_.push_back(angle);

    // show isect point if intersection
    if (statesVec_.back() == coVRLine::INTERSECT)
        isectPoints_.back()->setVisible(true);
}

//----------------------------------------------------------------------
void MathematicPlugin::makePlaneIntersections(int plane1, int plane2)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::makePlaneIntersections %d %d\n", plane1, plane2);

    Vec3 isectLinePoint1 = Vec3(0.0, 0.0, 0.0);
    Vec3 isectLinePoint2 = Vec3(1.0, 0.0, 0.0);
    double angle = 0.0;

    // test intersection
    int state = planesVec_.at(plane1)->test(planesVec_.at(plane2), &isectLinePoint1, &isectLinePoint2, &angle);
    isectLines_.push_back(new coVRLine(isectLinePoint1, isectLinePoint2, coVRLine::POINT_POINT, "s", false));
    statesVec_.push_back(state);
    anglesVec_.push_back(angle);

    // show isect line if intersection
    if (statesVec_.back() == coVRPlane::INTERSECT)
        isectLines_.back()->setVisible(true);
}

//----------------------------------------------------------------------
void MathematicPlugin::makePointDistance2Point(int point1, int point2)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::makePointDistance2Point point1=%d point2=%d\n", point1, point2);

    distancesVec_.push_back(new coVRDistance(pointsVec_.at(point1)->getPosition(), pointsVec_.at(point2)->getPosition(), coVRDistance::ONLY_LINE));
    distancesVec_.back()->addToMenu(mathematicsMenu_, mainMenuSepPos_ - 1);
}

//----------------------------------------------------------------------
void MathematicPlugin::makeLineDistance2Point(int line, int point)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::makeLineDistance2Point line=%d point=%d\n", line, point);

    Vec3 perpendicular = Vec3(0.0, 0.0, 0.0);
    linesVec_.at(line)->distance(pointsVec_.at(point), &perpendicular);
    distancesVec_.push_back(new coVRDistance(perpendicular, pointsVec_.at(point)->getPosition(), coVRDistance::POINT_LINE));
    distancesVec_.back()->addToMenu(mathematicsMenu_, mainMenuSepPos_ - 1);
}

//----------------------------------------------------------------------
void MathematicPlugin::makeLineDistance2Line(int line1, int line2)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::makeLineDistance2Line line1=%d line2=%d\n", line1, line2);

    Vec3 perpendicular1 = Vec3(0.0, 0.0, 0.0);
    Vec3 perpendicular2 = Vec3(0.0, 0.0, 0.0);

    linesVec_.at(line1)->distance(linesVec_.at(line2), &perpendicular1, &perpendicular2);
    distancesVec_.push_back(new coVRDistance(perpendicular1, perpendicular2, coVRDistance::POINT_LINE_POINT));
    distancesVec_.back()->addToMenu(mathematicsMenu_, mainMenuSepPos_ - 1);
}

//----------------------------------------------------------------------
void MathematicPlugin::makePlaneDistance2Point(int plane, int point)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::makePlaneDistance2Point plane=%d point=%d\n", plane, point);

    Vec3 perpendicular = Vec3(0.0, 0.0, 0.0);

    planesVec_.at(plane)->distance(pointsVec_.at(point), &perpendicular);
    distancesVec_.push_back(new coVRDistance(perpendicular, pointsVec_.at(point)->getPosition(), coVRDistance::POINT_LINE));
    distancesVec_.back()->addToMenu(mathematicsMenu_, mainMenuSepPos_ - 1);
}

//----------------------------------------------------------------------
void MathematicPlugin::makePlaneDistance2Line(int plane, int line)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::makePlaneDistance2Line plane=%d line=%d\n", plane, line);

    Vec3 perpendicular1 = Vec3(0.0, 0.0, 0.0);
    Vec3 perpendicular2 = Vec3(0.0, 0.0, 0.0);

    planesVec_.at(plane)->distance(linesVec_.at(line), &perpendicular1, &perpendicular2);
    distancesVec_.push_back(new coVRDistance(perpendicular1, perpendicular2, coVRDistance::POINT_LINE_POINT));
    distancesVec_.back()->addToMenu(mathematicsMenu_, mainMenuSepPos_ - 1);
}

//----------------------------------------------------------------------
void MathematicPlugin::makePlaneDistance2Plane(int plane1, int plane2)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::makePlaneDistance2Plane plane1=%d plane2=%d\n", plane1, plane2);

    Vec3 perpendicular1 = Vec3(0.0, 0.0, 0.0);
    Vec3 perpendicular2 = Vec3(0.0, 0.0, 0.0);

    planesVec_.at(plane1)->distance(planesVec_.at(plane2), &perpendicular1, &perpendicular2);
    distancesVec_.push_back(new coVRDistance(perpendicular1, perpendicular2, coVRDistance::POINT_LINE));
    distancesVec_.back()->addToMenu(mathematicsMenu_, mainMenuSepPos_ - 1);
}

//----------------------------------------------------------------------
void MathematicPlugin::menuEvent(coMenuItem *menuItem)
{

    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::menuEvent for %s\n", menuItem->getName());

    if (menuItem == showAxisMenuItem_)
    {
        axis_->setVisible(showAxisMenuItem_->getState());
    }
    else if (menuItem == hideLabelsMenuItem_)
        hideAllLabels(!hideLabelsMenuItem_->getState());
    else if (menuItem == addPointMenuItem_)
    {
        addPoint();
    }
    else if (menuItem == addLineMenuItem_)
    {
        addLine();
    }
    else if (menuItem == addPlaneMenuItem_)
    {
        addPlane();
    }
    else
    {
        // send menu event to points
        for (size_t i = 0; i < pointsVec_.size(); i++)
            pointsVec_.at(i)->menuEvent(menuItem);

        // send menu event to lines
        for (size_t i = 0; i < linesVec_.size(); i++)
            linesVec_.at(i)->menuEvent(menuItem);

        // send menu event to planes
        for (size_t i = 0; i < planesVec_.size(); i++)
            planesVec_.at(i)->menuEvent(menuItem);

        // send menu event to distance
        for (size_t i = 0; i < distancesVec_.size(); i++)
            distancesVec_.at(i)->menuEvent(menuItem);

        // deleting geometry objects
        for (size_t i = 0; i < deleteButtonsVec_.size(); i++)
        {
            if (deleteButtonsVec_.at(i) && menuItem == deleteButtonsVec_.at(i))
            {
                // remove this button from menu
                mathematicsMenu_->remove(deleteButtonsVec_.at(i));
                // delete button in next frame
                // right now entry is still selected
                deleteButtonNum_ = i;

                //for(int j=0;j<geometryInfoVec_.size();j++)
                //fprintf(stderr,"MathematicPlugin::menuEvent geometryInfoVec_.at(%d) %s\n",j, geometryInfoVec_.at(j).c_str());

                string geoObj = geometryInfoVec_.at(2 * i);
                string geoName = geometryInfoVec_.at(2 * i + 1);
                // remove strings from vector
                geometryInfoVec_.erase(geometryInfoVec_.begin() + 2 * i + 1);
                geometryInfoVec_.erase(geometryInfoVec_.begin() + 2 * i);

                if (geoObj == "POINT")
                {
                    // delete point
                    int num = -1;
                    for (size_t j = 0; j < pointsVec_.size(); j++)
                    {
                        if (pointsVec_.at(j)->getName() == geoName)
                        {
                            num = j;
                            break;
                        }
                    }
                    if (num == -1)
                        return;

                    pointsVec_.at(num)->removeFromMenu();
                    delete pointsVec_.at(num);
                    pointsVec_.erase(pointsVec_.begin() + num);
                    geometryCount_--;
                }
                else if (geoObj == "LINE")
                {
                    // delete line
                    int num = -1;
                    for (size_t j = 0; j < linesVec_.size(); j++)
                    {
                        if (linesVec_.at(j)->getName() == geoName)
                        {
                            num = j;
                            break;
                        }
                    }
                    if (num == -1)
                        return;

                    linesVec_.at(num)->removeFromMenu();
                    delete linesVec_.at(num);
                    linesVec_.erase(linesVec_.begin() + num);
                    geometryCount_--;
                }
                else if (geoObj == "PLANE")
                {
                    // delete plane
                    int num = -1;
                    for (size_t j = 0; j < planesVec_.size(); j++)
                    {
                        if (planesVec_.at(j)->getName() == geoName)
                        {
                            num = j;
                            break;
                        }
                    }
                    if (num == -1)
                        return;

                    planesVec_.at(num)->removeFromMenu();
                    delete planesVec_.at(num);
                    planesVec_.erase(planesVec_.begin() + num);
                    geometryCount_--;
                }

                // remove distance
                if (distancesVec_.size())
                {
                    // TODO for NUM_GEOMETRY>2 remove all
                    distancesVec_.back()->removeFromMenu();
                    delete distancesVec_.back();
                    distancesVec_.pop_back();
                }

                // remove isectPoints
                if (isectPoints_.size())
                {
                    // TODO for NUM_GEOMETRY>2
                    delete isectPoints_.back();
                    isectPoints_.pop_back();
                }

                // remove isectLines
                if (isectLines_.size())
                {
                    // TODO for NUM_GEOMETRY>2
                    delete isectLines_.back();
                    isectLines_.pop_back();
                }

                // remove states
                if (statesVec_.size())
                {
                    // TODO for NUM_GEOMETRY>2
                    statesVec_.pop_back();
                }

                // remove angle
                if (anglesVec_.size())
                {
                    // TODO for NUM_GEOMETRY>2
                    anglesVec_.pop_back();
                }
            }
        }
    }
}

//----------------------------------------------------------------------
void MathematicPlugin::addPoint()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::addPoint\n");

    if (geometryCount_ < NUM_GEOMETRY)
    {
        geometryCount_++;

        int menuPos = mathematicsMenu_->getItemCount();
        double vecPos = (double)pointsVec_.size() + 1;
        pointsVec_.push_back(new coVRPoint(Vec3(0.0, vecPos, 0.0))); // should lie in BB
        menuPos = pointsVec_.back()->addToMenu(mathematicsMenu_, ++menuPos);

        // info what geometry object is added
        geometryInfoVec_.push_back("POINT");
        geometryInfoVec_.push_back(pointsVec_.back()->getName());

        //menu for delete
        string text(coTranslator::coTranslate(": loeschen"));
        string deleteText = pointsVec_.back()->getName() + text.c_str();
        deleteButtonsVec_.push_back(new coButtonMenuItem(deleteText.c_str()));
        deleteButtonsVec_.back()->setMenuListener(this);
        mathematicsMenu_->insert(deleteButtonsVec_.back(), menuPos - 2);

        // distance point to point
        if (pointsVec_.size() == 2)
            makePointDistance2Point(0, 1);

        // distance point to line
        if (linesVec_.size() == 1 && pointsVec_.size() == 1)
            makeLineDistance2Point(0, 0);

        // distance point to plane
        if (planesVec_.size() == 1 && pointsVec_.size() == 1)
            makePlaneDistance2Point(0, 0);

        if (!hideLabelsMenuItem_->getState())
            pointsVec_.back()->hideLabel(true);
    }

    updateStateLabel();
}

//----------------------------------------------------------------------
void MathematicPlugin::addLine()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::addLine\n");

    if (geometryCount_ < NUM_GEOMETRY)
    {
        geometryCount_++;

        int menuPos = mathematicsMenu_->getItemCount();
        double vecPos = 1;
        // dont place line over other line) TODO NUM_GEOMETRY>2
        if (linesVec_.size()
            && linesVec_.back()->getBasePoint() == Vec3(0.0, -vecPos, vecPos))
            vecPos++;

        linesVec_.push_back(new coVRLine(Vec3(0.0, -vecPos, vecPos), Vec3(0.0, -(vecPos + 5), vecPos), coVRLine::POINT_POINT)); // should lie in BB

        // setting other color
        if ((int)linesVec_.size() > 0)
        {
            //if color of plane is violet set this to orange
            if (linesVec_.at(0)->color() == Vec4(1.0, 1.0, 0.0, 1.0))
                linesVec_.back()->setColor(Vec4(0.1961, 0.7765, 0.6510, 1));
        }

        menuPos = linesVec_.back()->addToMenu(mathematicsMenu_, menuPos);

        // setting mode according to config
        if (coCoviseConfig::isOn("COVER.Plugin.Mathematic.LineVectorMode", false))
            linesVec_.back()->setMode(coVRLine::POINT_DIR);

        // info what geometry object is added
        geometryInfoVec_.push_back("LINE");
        geometryInfoVec_.push_back(linesVec_.back()->getName());

        //menu for delete
        string text(coTranslator::coTranslate(": loeschen"));
        string deleteText = linesVec_.back()->getName() + text.c_str();
        deleteButtonsVec_.push_back(new coButtonMenuItem(deleteText.c_str()));
        deleteButtonsVec_.back()->setMenuListener(this);
        mathematicsMenu_->insert(deleteButtonsVec_.back(), menuPos - 3);

        // distance line to point
        if (linesVec_.size() == 1 && pointsVec_.size() == 1)
            makeLineDistance2Point(0, 0);

        // two lines 1 intersection possible
        if (linesVec_.size() == 2)
        {
            // g0 and g1
            makeLineIntersections(0, 1);
            makeLineDistance2Line(0, 1);
        }

        // three lines 3 intersections possible
        if (linesVec_.size() == 3)
        {
            // g1 and g2
            makeLineIntersections(1, 2);

            // g2 and g0
            makeLineIntersections(2, 0);
        }

        // testing line and plane
        if (linesVec_.size() == 1 && planesVec_.size() == 1)
        {
            makePlaneLineIntersections(0, 0);
            makePlaneDistance2Line(0, 0);
        }
        if (!hideLabelsMenuItem_->getState())
            linesVec_.back()->hideLabel(true);
    }

    updateStateLabel();
}

//----------------------------------------------------------------------
void MathematicPlugin::addPlane()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::addPlane\n");

    if (geometryCount_ < NUM_GEOMETRY)
    {
        geometryCount_++;

        int menuPos = mathematicsMenu_->getItemCount();
        double vecPos = 5;
        // dont place plane over other plane) TODO NUM_GEOMETRY>2
        if (planesVec_.size()
            && planesVec_.back()->getBasePoint() == Vec3(5.0, 0.0, -vecPos))
            vecPos++;
        planesVec_.push_back(new coVRPlane(Vec3(5.0, 0.0, -vecPos), Vec3(0.0, 5.0, -vecPos), Vec3(-2.7, -2.7, -vecPos))); // should lie in BB
        menuPos = planesVec_.back()->addToMenu(mathematicsMenu_, menuPos);

        // setting other color
        if ((int)planesVec_.size() > 0)
        {
            //if color of plane is violet set this to orange
            if (planesVec_.at(0)->color() == Vec4(0.545, 0.0, 0.545, 0.5))
                planesVec_.back()->setColor(Vec4(1, 0.4980, 0, 0.5));
        }

        // setting mode according to config
        if (coCoviseConfig::isOn("COVER.Plugin.Mathematic.PlanePointsMode", false))
            planesVec_.back()->setMode(coVRPlane::POINT_POINT_POINT);
        else if (coCoviseConfig::isOn("COVER.Plugin.Mathematic.PlaneNormalMode", false))
            planesVec_.back()->setMode(coVRPlane::POINT_DIR);
        else if (coCoviseConfig::isOn("COVER.Plugin.Mathematic.PlaneDirectionsMode", false))
            planesVec_.back()->setMode(coVRPlane::POINT_DIR_DIR);

        // setting equation according to config
        bool paramEqu = false;
        bool coordEqu = false;
        bool normEqu = false;
        if (coCoviseConfig::isOn("COVER.Plugin.Mathematic.PlaneParameterEquation", false))
            paramEqu = true;
        if (coCoviseConfig::isOn("COVER.Plugin.Mathematic.PlaneCoordinateEquation", false))
            coordEqu = true;
        if (coCoviseConfig::isOn("COVER.Plugin.Mathematic.PlaneNormalEquation", false))
            normEqu = true;
        if (paramEqu || coordEqu || normEqu)
            planesVec_.back()->showEquations(paramEqu, coordEqu, normEqu);

        // info what geometry object is added
        geometryInfoVec_.push_back("PLANE");
        geometryInfoVec_.push_back(planesVec_.back()->getName());

        //menu for delete
        string text(coTranslator::coTranslate(": loeschen"));
        string deleteText = planesVec_.back()->getName() + text.c_str();
        deleteButtonsVec_.push_back(new coButtonMenuItem(deleteText.c_str()));
        deleteButtonsVec_.back()->setMenuListener(this);
        mathematicsMenu_->insert(deleteButtonsVec_.back(), menuPos - 4);

        // distance point to plane
        if (planesVec_.size() == 1 && pointsVec_.size() == 1)
            makePlaneDistance2Point(0, 0);

        // testing line and plane
        if (planesVec_.size() == 1 && linesVec_.size() == 1)
        {
            makePlaneLineIntersections(0, 0);
            makePlaneDistance2Line(0, 0);
        }

        // testing plane and plane
        if (planesVec_.size() == 2)
        {
            makePlaneIntersections(0, 1);
            makePlaneDistance2Plane(0, 1);
        }
        if (!hideLabelsMenuItem_->getState())
            planesVec_.back()->hideLabel(true);
    }

    updateStateLabel();
}

//----------------------------------------------------------------------
void MathematicPlugin::updateStateLabel()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::updateStateLabel\n");

    string text;

    // state of line and point
    if (linesVec_.size() == 1 && pointsVec_.size() == 1
        && distancesVec_.size())
    {
        char sentense[1024];
        if (distancesVec_.at(0)->getDistance() != 0)
        {
            sprintf(sentense, coTranslator::coTranslate("%s liegt nicht auf %s").c_str(), pointsVec_.at(0)->getName().c_str(), linesVec_.at(0)->getName().c_str());
        }
        else
        {
            sprintf(sentense, coTranslator::coTranslate("%s liegt auf %s").c_str(), pointsVec_.at(0)->getName().c_str(), linesVec_.at(0)->getName().c_str());
        }

        text = std::string(sentense);
    }

    // state of lines
    if (linesVec_.size() == 2)
    {
        // g0 and g1
        text.append(computeLineText(0, 1, 0));
    }
    if (linesVec_.size() == 3)
    {
        // g1 and g2
        text.append("\n");
        text.append(computeLineText(1, 2, 1));

        // g2 and g0
        text.append("\n");
        text.append(computeLineText(2, 0, 2));
    }

    // state of plane and point
    if (planesVec_.size() == 1 && pointsVec_.size() == 1
        && distancesVec_.size())
    {
        char sentense[1024];
        if (distancesVec_.at(0)->getDistance() != 0)
        {
            sprintf(sentense, coTranslator::coTranslate("%s liegt nicht auf %s").c_str(), pointsVec_.at(0)->getName().c_str(), planesVec_.at(0)->getName().c_str());
        }
        else
        {
            sprintf(sentense, coTranslator::coTranslate("%s liegt auf %s").c_str(), pointsVec_.at(0)->getName().c_str(), planesVec_.at(0)->getName().c_str());
        }

        text = std::string(sentense);
    }

    // state of plane and line
    if (planesVec_.size() == 1 && linesVec_.size() == 1)
        text = computePlaneLineText(0, 0);

    // state of planes
    if (planesVec_.size() == 2)
        text.append(computePlaneText(0, 1, 0));

    // make the label
    if (text.empty())
        text = " ";
    stateLabel_->setLabel(text);

    if (!hideLabelsMenuItem_->getState())
    {
        for (vector<coVRDistance *>::iterator it = distancesVec_.begin(); it != distancesVec_.end(); it++)
            (*it)->hideLabels(true);
    }
}

//----------------------------------------------------------------------
string MathematicPlugin::computeLineText(int line1, int line2, int lineState)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::computeLineText line1 %d line2 %d lineState %d\n", line1, line2, lineState);

    string text = linesVec_.at(line1)->getName();
    text.append(linesVec_.at(line2)->getName());
    text.append(": ");
    if (statesVec_.at(lineState) == coVRLine::INTERSECT)
        text.append(computeLineIsectText(line1));
    else if (statesVec_.at(lineState) == coVRLine::PARALLEL)
    {
        text.append(coTranslator::coTranslate("PARALLEL"));
    }
    else if (statesVec_.at(lineState) == coVRLine::SKEW)
    {
        text.append(coTranslator::coTranslate("WINDSCHIEF"));
    }
    else
        // wrong line state type
        return "";

    return text;
}

//----------------------------------------------------------------------
string MathematicPlugin::computePlaneText(int plane1, int plane2, int planeState)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::computePlaneText plane1 %d plane2 %d\n", plane1, plane2);

    string text = planesVec_.at(plane1)->getName();
    text.append(planesVec_.at(plane2)->getName());
    text.append(": ");
    if (statesVec_.at(planeState) == coVRPlane::INTERSECT)
        text.append(computePlaneIsectText(plane1));
    else if (statesVec_.at(planeState) == coVRPlane::PARALLEL)
        text.append(coTranslator::coTranslate("PARALLEL"));
    else
        // wrong plane state type
        return "";

    return text;
}

//----------------------------------------------------------------------
string MathematicPlugin::computePlaneLineText(int plane, int state)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::computePlaneLineText plane %d stateNum %d state %d\n", plane, state, statesVec_.at(plane));

    string text = linesVec_.at(0)->getName();

    if (statesVec_.at(0) == coVRPlane::LIESIN)
    {
        text.append(coTranslator::coTranslate(" liegt in "));
        text.append(planesVec_.at(plane)->getName());
    }
    else if (statesVec_.at(plane) == coVRPlane::PARALLEL)
    {
        text.append(planesVec_.at(plane)->getName());
        text.append(": ");
        text.append(coTranslator::coTranslate("PARALLEL"));
    }
    else if (statesVec_.at(plane) == coVRPlane::INTERSECT)
    {
        text.append(planesVec_.at(plane)->getName());
        text.append(": ");
        text.append(computePlaneLineIsectText(plane));
    }

    return text;
}

//----------------------------------------------------------------------
string MathematicPlugin::computeLineIsectText(int line)
{
    string text(coTranslator::coTranslate("SCHNITT"));
    ostringstream txtStream;
    /*
   txtStream << "(";
   txtStream << round10(isectPoints_.at(line)->x());
   txtStream << ", ";
   txtStream << round10(isectPoints_.at(line)->y());
   txtStream << ", ";
   txtStream << round10(isectPoints_.at(line)->z());
   txtStream << ") ";
   */
    txtStream << round10(anglesVec_.at(line));
    txtStream << coTranslator::coTranslate("GRAD");
    text.append(txtStream.str());

    return text;
}

//----------------------------------------------------------------------
string MathematicPlugin::computePlaneLineIsectText(int plane)
{
    string text(coTranslator::coTranslate("SCHNITT "));
    ostringstream txtStream;
    /*
   txtStream << "(";
   txtStream << round10(isectPoints_.at(plane)->x());
   txtStream << ", ";
   txtStream << round10(isectPoints_.at(plane)->y());
   txtStream << ", ";
   txtStream << round10(isectPoints_.at(plane)->z());
   txtStream << ") ";
   */
    txtStream << round10(anglesVec_.at(plane));
    txtStream << coTranslator::coTranslate("GRAD");
    text.append(txtStream.str());

    return text;
}

//----------------------------------------------------------------------
string MathematicPlugin::computePlaneIsectText(int plane)
{
    string text(coTranslator::coTranslate("SCHNITT "));
    ostringstream txtStream;

    txtStream << round10(anglesVec_.at(plane));
    txtStream << coTranslator::coTranslate("GRAD");
    text.append(txtStream.str());

    return text;
}

//----------------------------------------------------------------------
Vec3 MathematicPlugin::roundVec10(Vec3 vec)
{
    return Vec3(round10(vec.x()), round10(vec.y()), round10(vec.z()));
}

//----------------------------------------------------------------------
double MathematicPlugin::round10(double num)
{
    int factor = 1;

    if (num < 0)
        factor = -1;

    int integer = factor * (int)num;
    double rest = factor * num - integer;

    // rounding one decimal
    if (rest >= 0.0 && rest < 0.05)
        return (double)(factor * integer);
    else if (rest >= 0.05 && rest < 0.15)
        return factor * ((double)integer + 0.1);
    else if (rest >= 0.15 && rest < 0.25)
        return factor * ((double)integer + 0.2);
    else if (rest >= 0.25 && rest < 0.35)
        return factor * ((double)integer + 0.3);
    else if (rest >= 0.35 && rest < 0.45)
        return factor * ((double)integer + 0.4);
    else if (rest >= 0.45 && rest < 0.55)
        return factor * ((double)integer + 0.5);
    else if (rest >= 0.55 && rest < 0.65)
        return factor * ((double)integer + 0.6);
    else if (rest >= 0.65 && rest < 0.75)
        return factor * ((double)integer + 0.7);
    else if (rest >= 0.75 && rest < 0.85)
        return factor * ((double)integer + 0.8);
    else if (rest >= 0.85 && rest < 0.95)
        return factor * ((double)integer + 0.9);
    else if (rest >= 0.95)
        return (double)factor * (integer + 1);

    return 0.0;
}

//----------------------------------------------------------------------
Vec3 MathematicPlugin::roundVec2(Vec3 vec)
{
    return Vec3(round2(vec.x()), round2(vec.y()), round2(vec.z()));
}

//----------------------------------------------------------------------
double MathematicPlugin::round2(double num)
{
    int factor = 1;

    if (num < 0)
        factor = -1;

    int integer = factor * (int)num;
    double rest = factor * num - integer;

    if (rest >= 0.0 && rest < 0.25)
        return (double)(factor * integer);
    else if (rest >= 0.25 && rest <= 0.75)
        return factor * ((double)integer + 0.5);
    else if (rest > 0.75)
        return (double)factor * (integer + 1);
    return 0.0;
}

//----------------------------------------------------------------------
double MathematicPlugin::computeAngle(Vec3 direction1, Vec3 direction2)
{
    ///if (cover->debugLevel(3))
    fprintf(stderr, "MathematicPlugin::computeAngle dir1(%f %f %f) dir2(%f %f %f)\n", direction1.x(), direction1.y(), direction1.z(), direction2.x(), direction2.y(), direction2.z());

    //            dir1*dir2
    // cos phi = -----------
    //           |dir1| * |dir2|
    double phi = acos((direction1 * direction2) / (direction1.length() * direction2.length()));

    // choose the bigger angle -> phi+phi2 = 180 DEG
    double phi2 = MATH_PI - phi;
    fprintf(stderr, "MathematicPlugin::computeAngle phi %f phi2 %f\n", (180. / MATH_PI) * phi, (180. / MATH_PI) * phi2);
    if (phi2 < phi)
        phi = phi2;

    // in degrees = 180°/phi * MATH_PI
    return (180. / MATH_PI) * phi;
}

//----------------------------------------------------------------------
void MathematicPlugin::drawBoundingBox(BoundingBox *box)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::drawBoundingBox\n");

    boxGeode_ = new Geode();

    Vec3 bpoints[8];
    bpoints[0].set(box->xMin(), box->yMin(), box->zMin());
    bpoints[1].set(box->xMax(), box->yMin(), box->zMin());
    bpoints[2].set(box->xMax(), box->yMax(), box->zMin());
    bpoints[3].set(box->xMin(), box->yMax(), box->zMin());
    bpoints[4].set(box->xMin(), box->yMin(), box->zMax());
    bpoints[5].set(box->xMax(), box->yMin(), box->zMax());
    bpoints[6].set(box->xMax(), box->yMax(), box->zMax());
    bpoints[7].set(box->xMin(), box->yMax(), box->zMax());

    Geometry *lineGeometry[12];
    Vec3Array *vArray[12];
    DrawArrays *drawable[12];

    for (int i = 0; i < 12; i++)
    {
        lineGeometry[i] = new Geometry();
        vArray[i] = new Vec3Array();
        //fprintf(stderr,"MathematicPlugin::drawBoundingBox bpoints[0] %f %f %f\n", bpoints[0].x(), bpoints[0].y(), bpoints[0].z());
        //fprintf(stderr,"MathematicPlugin::drawBoundingBox bpoints[1] %f %f %f\n", bpoints[1].x(), bpoints[1].y(), bpoints[1].z());
        lineGeometry[i]->setVertexArray(vArray[i]);
        drawable[i] = new DrawArrays(PrimitiveSet::LINES, 0, 2);
        lineGeometry[i]->addPrimitiveSet(drawable[i]);
        LineWidth *linewidth = new LineWidth();
        linewidth->setWidth(1.0);
        boxGeode_->addDrawable(lineGeometry[i]);
    }

    // lines
    vArray[0]->push_back(bpoints[0]);
    vArray[0]->push_back(bpoints[1]);
    vArray[1]->push_back(bpoints[1]);
    vArray[1]->push_back(bpoints[2]);
    vArray[2]->push_back(bpoints[2]);
    vArray[2]->push_back(bpoints[3]);
    vArray[3]->push_back(bpoints[3]);
    vArray[3]->push_back(bpoints[0]);
    vArray[4]->push_back(bpoints[4]);
    vArray[4]->push_back(bpoints[5]);
    vArray[5]->push_back(bpoints[5]);
    vArray[5]->push_back(bpoints[6]);
    vArray[6]->push_back(bpoints[6]);
    vArray[6]->push_back(bpoints[7]);
    vArray[7]->push_back(bpoints[7]);
    vArray[7]->push_back(bpoints[4]);
    vArray[8]->push_back(bpoints[0]);
    vArray[8]->push_back(bpoints[4]);
    vArray[9]->push_back(bpoints[3]);
    vArray[9]->push_back(bpoints[7]);
    vArray[10]->push_back(bpoints[2]);
    vArray[10]->push_back(bpoints[6]);
    vArray[11]->push_back(bpoints[1]);
    vArray[11]->push_back(bpoints[5]);

    Material *material = new Material();
    material->setDiffuse(Material::FRONT_AND_BACK, Vec4(1.0, 1.0, 1.0, 1.0));
    material->setAmbient(Material::FRONT_AND_BACK, Vec4(1.0, 1.0, 1.0, 1.0));
    material->setSpecular(Material::FRONT_AND_BACK, Vec4(1.0, 1.0, 1.0, 1.0));
    StateSet *stateSet = VRSceneGraph::instance()->loadDefaultGeostate();
    stateSet->setMode(GL_BLEND, StateAttribute::ON);
    stateSet->setAttributeAndModes(material);
    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    boxGeode_->setStateSet(stateSet);

    boxGeode_->setNodeMask(boxGeode_->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));

    cover->getObjectsRoot()->addChild(boxGeode_);
}

//----------------------------------------------------------------------
void MathematicPlugin::showBoundingBox(bool show)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::showBoundingBox %d\n", show);

    if (show)
        boxGeode_->setNodeMask(boxGeode_->getNodeMask() | (Isect::Visible));
    else
        boxGeode_->setNodeMask(boxGeode_->getNodeMask() & (~Isect::Visible));
}

//----------------------------------------------------------------------
void MathematicPlugin::hideAllLabels(bool hide)
{
    for (vector<coVRPoint *>::iterator it = pointsVec_.begin(); it != pointsVec_.end(); it++)
        (*it)->hideLabel(hide);
    for (vector<coVRLine *>::iterator it = linesVec_.begin(); it != linesVec_.end(); it++)
        (*it)->hideLabel(hide);
    for (vector<coVRPlane *>::iterator it = planesVec_.begin(); it != planesVec_.end(); it++)
        (*it)->hideLabel(hide);
    for (vector<coVRDistance *>::iterator it = distancesVec_.begin(); it != distancesVec_.end(); it++)
        (*it)->hideLabels(hide);
}

COVERPLUGIN(MathematicPlugin)
