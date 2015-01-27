/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2009 Visenso  **
 **                                                                        **
 ** Description: coVRPlane                                                 **
 **              Draws a plane according to mode                           **
 **               either as a plane through three points                   **
 **               or a plane with a base point and a (normal)direction     **
 **               only within the bounding box (needs to be set)           **
 **               with a nameTag and equations                             **
 **                                                                        **
 ** Author: M. Theilacker                                                  **
 **                                                                        **
 ** History:                                                               **
 **     4.2009 initial version                                             **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>
#include <cover/VRViewer.h>
#include <cover/coTranslator.h>

#include "MathematicPlugin.h"

#include "coVRPlane.h"

const double MATH_PI = 3.14159265358979323846264338327950;

using namespace osg;

int coVRPlane::_planeID_ = 0;
BoundingBox *coVRPlane::_boundingBox_ = NULL;

//
// Constructor
//
coVRPlane::coVRPlane(Vec3 point, Vec3 normal, string name)
    : coPlane(point, normal)
    , name_(name)
    , node_(NULL)
    , mode_(POINT_DIR)
    , point1_(NULL)
    , point2_(NULL)
    , point3_(NULL)
    , direction1_(NULL)
    , direction2_(NULL)
    , normalDirection_(NULL)
    , parentMenu_(NULL)
    , pointsModeCheckbox_(NULL)
    , normalModeCheckbox_(NULL)
    , directionsModeCheckbox_(NULL)
    , sepLabel_(NULL)
    , plane_(NULL)
    , planeGeode_(NULL)
    , planeDraw_(NULL)
    , stateSet_(NULL)
    , material_(NULL)
    , isVisible_(true)
    , isChanged_(false)
    , nameTag_(NULL)
    , isBBSet_(false)
    , showParamEqu_(true)
    , showCoordEqu_(true)
    , showNormEqu_(true)
    , labelsShown_(false)
    , scale_(5)
    , numDrawPoints_(0)
{
    if (cover->debugLevel(1))
        fprintf(stderr, "coVRPlane::coVRPlane POINT_DIR\n");

    if (normal.length() == 0)
    {
        fprintf(stderr, "ERROR: Can not create a plane with no normal!\n   Changing normal to z axis");
        normal.set(0.0, 0.0, 1.0);
    }
    normal.normalize();
    normal *= scale_;

    //   normal_
    //   ^
    //   |
    //   point1         point2
    //   o----------->o
    //   |      ^direction1
    //   |
    //   | < direction2
    //   |
    //   v
    //   o
    //  point3

    // point1 (base point)
    point1_ = new coVRPoint(point);
    point1_->setVisible(true);

    // normal of the plane
    normalDirection_ = new coVRDirection(normal, point);
    normalDirection_->setVisible(true);
    oldNormalDirection_ = normalDirection_->getDirection();

    // computing point2 and getting direction from point1 to point2
    point2_ = new coVRPoint(computePoint2(normal));
    point2_->setVisible(false);
    direction1_ = new coVRDirection(point2_->getPosition() - point, point);
    direction1_->setVisible(false);

    // computing point3 and getting direction from point1 to point3
    point3_ = new coVRPoint(computePoint3(normal));
    point3_->setVisible(false);
    direction2_ = new coVRDirection(point3_->getPosition() - point, point);
    direction2_->setVisible(false);

    // initialise plane
    int error = init();
    if (error)
        return;
}

//
// Constructor
//
coVRPlane::coVRPlane(Vec3 point1, Vec3 point2, Vec3 point3, string name)
    : coPlane(point1, point2)
    , name_(name)
    , node_(NULL)
    , mode_(POINT_POINT_POINT)
    , point1_(NULL)
    , point2_(NULL)
    , point3_(NULL)
    , direction1_(NULL)
    , direction2_(NULL)
    , normalDirection_(NULL)
    , parentMenu_(NULL)
    , pointsModeCheckbox_(NULL)
    , normalModeCheckbox_(NULL)
    , directionsModeCheckbox_(NULL)
    , sepLabel_(NULL)
    , plane_(NULL)
    , planeGeode_(NULL)
    , planeDraw_(NULL)
    , stateSet_(NULL)
    , material_(NULL)
    , isVisible_(true)
    , isChanged_(false)
    , nameTag_(NULL)
    , isBBSet_(false)
    , showParamEqu_(true)
    , showCoordEqu_(true)
    , showNormEqu_(true)
    , labelsShown_(false)
    , scale_(5)
{
    if (cover->debugLevel(1))
        fprintf(stderr, "coVRPlane::coVRPlane POINT_POINT_POINT\n");

    Vec3 direction1 = point1 - point2;
    direction1.normalize();
    direction1 *= scale_;
    Vec3 direction2 = point1 - point3;
    direction2.normalize();
    direction2 *= scale_;
    if (direction1 == direction2)
    {
        fprintf(stderr, "ERROR: Can not create a plane of colinear points!\n   Changing points a little");
        point2.set(point2.x(), point2.y(), point2.z() + 1.0);
    }

    //   normal_
    //   ^
    //   |
    //   point1         point2
    //   o----------->o
    //   |      ^direction1
    //   |
    //   | < direction2
    //   |
    //   v
    //   o
    //  point3

    // point1 (base point), point2 and point3
    point1_ = new coVRPoint(point1);
    point1_->setVisible(true);
    point2_ = new coVRPoint(point2);
    point2_->setVisible(true);
    point3_ = new coVRPoint(point3);
    point3_->setVisible(true);

    // directions from point1 to point2 and from point1 to point3
    direction1_ = new coVRDirection(point2 - point1, point1);
    direction1_->setVisible(false);
    direction2_ = new coVRDirection(point3 - point1, point1);
    direction2_->setVisible(false);

    // normal of the plane is the crossproduct of the 2 directions
    Vec3 normalVec = direction1_->getDirection() ^ direction2_->getDirection();
    normalVec.normalize();
    normalVec *= scale_;
    normalDirection_ = new coVRDirection(normalVec, point1);
    normalDirection_->setVisible(false);
    oldNormalDirection_ = normalDirection_->getDirection();

    // initialise plane
    int error = init();
    if (error)
        return;
}

//
// Destructor
//
coVRPlane::~coVRPlane()
{
    if (cover->debugLevel(1))
        fprintf(stderr, "coVRPlane::~coVRPlane\n");

    if (isBBSet_)
        delete _boundingBox_;

    cover->getObjectsRoot()->removeChild(node_.get());
    node_->unref();

    delete point1_;
    delete point2_;
    delete point3_;
    delete direction1_;
    delete direction2_;
    delete normalDirection_;

    delete pointsModeCheckbox_;
    delete normalModeCheckbox_;
    delete directionsModeCheckbox_;
    delete modeGroup_;

    delete sepLabel_;

    delete nameTag_;
}

//----------------------------------------------------------------------
int coVRPlane::init()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "coVRPlane::init\n");

    _planeID_++;

    // settings for coPlane
    coPlane::update(normalDirection_->getDirection(), point1_->getPosition());

    // set boundingbox if not already set
    if (!_boundingBox_)
    {
        fprintf(stderr, "WARNING: coVRPlane has no bounding box, will be set to 10\n");
        double boundary = 10.;
        BoundingBox *boundingBox = new BoundingBox(-boundary, -boundary, -boundary,
                                                   boundary, boundary, boundary);
        setBoundingBox(boundingBox);
        isBBSet_ = true;
    }

    // make unique name
    ostringstream numStream;
    numStream << _planeID_;
    name_.append(numStream.str());

    // root node
    node_ = new MatrixTransform();
    node_->ref();
    node_->setName(name_);

    // compute plane
    plane_ = new Geometry();
    planeGeode_ = new Geode();
    planeGeode_->addDrawable(plane_);
    planeGeode_->setNodeMask(planeGeode_->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));
    node_->addChild(planeGeode_);
    // normal
    plane_->setUseDisplayList(false);

    int error = updatePlane();
    if (error)
        return error;

    // make color violet
    makeColor();
    setColor(Vec4(0.545, 0.0, 0.545, 0.5));
    stateSet_->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

    // name tag as pinboard
    makeNameTag();
    updateNameTag(true);

    // add root node to cover scenegraph
    cover->getObjectsRoot()->addChild(node_.get());

    return 0;
}

//----------------------------------------------------------------------
void coVRPlane::preFrame()
{
    //fprintf(stderr,"coVRPlane::preFrame\n");

    if (isVisible_)
    {
        point1_->preFrame();

        if (point2_->isVisible())
            point2_->preFrame();

        if (point3_->isVisible())
            point3_->preFrame();

        if (direction1_->isVisible())
            direction1_->preFrame();

        if (direction2_->isVisible())
            direction2_->preFrame();

        if (normalDirection_->isVisible())
            normalDirection_->preFrame();

        bool p1 = point1_->isChanged();
        bool p2 = point2_->isChanged();
        bool p3 = point3_->isChanged();
        bool dir1 = direction1_->isChanged();
        bool dir2 = direction2_->isChanged();
        bool norm = normalDirection_->isChanged();

        int value = -1;
        if (p1)
            value = P1;
        if (p2)
            value = P2;
        if (p3)
            value = P3;
        if (dir1)
            value = DIR1;
        if (dir2)
            value = DIR2;
        if (norm)
            value = NORM;

        if (p1 || p2 || p3 || dir1 || dir2 || norm)
        {
            isChanged_ = true;
            update(value);
            updateNameTag(true);
        }
        else
        {
            // cover BaseMat changes per frame
            updateNameTag(false);
        }
    }
}

//----------------------------------------------------------------------
void coVRPlane::update(int value)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::update\n");

    if (value == P1)
    {
        //fprintf(stderr,"coVRPlane::update point1\n");

        // adjusting all values to the changed base point position
        Vec3 posP1 = point1_->getPosition();
        direction1_->setPosition(posP1);
        direction2_->setPosition(posP1);
        normalDirection_->setPosition(posP1);
        point2_->setPosition(posP1 + direction1_->getDirection());
        point3_->setPosition(posP1 + direction2_->getDirection());
    }
    else if (value == P2)
    {
        //fprintf(stderr,"coVRPlane::update point2\n");

        // adjusting direction1 to the changed point position
        Vec3 dir1 = point2_->getPosition() - point1_->getPosition();
        Vec3 dir2 = direction2_->getDirection();

        direction1_->setDirection(dir1);

        // adjusting normal to changed point2
        updateNormal();
    }
    else if (value == P3)
    {
        //fprintf(stderr,"coVRPlane::update point3\n");

        // adjusting direction2 to the changed point position
        Vec3 dir1 = direction1_->getDirection();
        Vec3 dir2 = point3_->getPosition() - point1_->getPosition();

        direction2_->setDirection(dir2);

        // adjusting normal to changed point3
        updateNormal();
    }
    else if (value == DIR1)
    {
        //fprintf(stderr,"coVRPlane::update direction1\n");

        Vec3 dir1 = direction1_->getDirection();
        Vec3 dir2 = direction2_->getDirection();

        // adjusting point2 to the changed direction
        point2_->setPosition(point1_->getPosition() + dir1);

        // adjusting normal to the changed direction1
        updateNormal();
    }
    else if (value == DIR2)
    {
        //fprintf(stderr,"coVRPlane::update direction2\n");
        Vec3 dir1 = direction1_->getDirection();
        Vec3 dir2 = direction2_->getDirection();

        // adjusting point2 to the changed direction
        point3_->setPosition(point1_->getPosition() + dir2);

        // adjusting normal to the changed direction2
        updateNormal();
    }
    else if (value == NORM && oldNormalDirection_ != normalDirection_->getDirection())
    {
        //fprintf(stderr,"coVRPlane::update normalDirection\n");

        // adjusting all values to the changed normal direction (via rotation matrix)
        Vec3 norDir = normalDirection_->getDirection();
        Vec3 baseP = point1_->getPosition();

        // get rotation matrix
        Matrix rotationMatrix;
        rotationMatrix.makeRotate(oldNormalDirection_, norDir);

        // adjusting direction1 to the changed normal
        Vec3 dir = direction1_->getDirection();
        dir = rotationMatrix.preMult(dir);
        direction1_->setDirection(dir);
        // adjusting point2 to the changed direction1
        point2_->setPosition(baseP + dir);

        // adjusting direction2 to the changed normal
        dir = direction2_->getDirection();
        dir = rotationMatrix.preMult(dir);
        direction2_->setDirection(dir);
        // adjusting point3 to the changed direction2
        point3_->setPosition(baseP + dir);

        // remember value
        oldNormalDirection_ = norDir;
    }

    coPlane::update(normalDirection_->getDirection(), point1_->getPosition());

    int error = updatePlane();
    if (error) //keep old plane
        return;

    //renew node
    node_->dirtyBound();
}

//----------------------------------------------------------------------
int coVRPlane::updatePlane()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::updatePlane\n");

    numDrawPoints_ = coPlane::getBoxIntersectionPoints(*(_boundingBox_), drawPoints_);

    //for (int i=0; i<numIsectPoints; i++)
    //fprintf(stderr,"coVRPlane::computePlane drawPoints_[%d]=(%f %f %f) \n", i, drawPoints_[i].x(), drawPoints_[i].y(), drawPoints_[i].z());

    // array of vertices
    Vec3Array *vecArray = new Vec3Array();

    if (numDrawPoints_ > 0)
    {
        for (int i = 0; i < numDrawPoints_; i++)
            vecArray->push_back(drawPoints_[i]);
    }
    else // no numIsectPoints: keep old plane
    {
        fprintf(stderr, "Warning: Points of plane where colinear\n");
        return -1;
    }

    plane_->setVertexArray(vecArray);

    // array for normal
    Vec3Array *normalArray = new Vec3Array();
    normalArray->push_back(normalDirection_->getDirection());
    plane_->setNormalArray(normalArray);
    plane_->setNormalBinding(Geometry::BIND_OVERALL);

    // primitive set
    DrawArrays *drawArray = new DrawArrays(PrimitiveSet::POLYGON, 0, numDrawPoints_);
    if (plane_->getNumPrimitiveSets() > 0)
    {
        //fprintf(stderr,"coVRPlane::computePlane rem num %d\n", plane_->getNumPrimitiveSets());
        plane_->removePrimitiveSet(0);
    }
    plane_->addPrimitiveSet(drawArray);

    return 0;
}

//----------------------------------------------------------------------
void coVRPlane::updateNameTag(bool planeChanged)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::updateNameTag %d\n", planeChanged);

    // only change position if something changed
    // otherwise only adjust nametagPosition to maybe changed BaseMat
    // its anoying if nametag jumps around while the plane is moving
    if (planeChanged)
    {
        Matrix w_to_o = cover->getInvBaseMat();
        Vec3 viewerPos_o = VRViewer::instance()->getViewerPos() * w_to_o;

        // get closest and farthest point of plane drawPoints_[i]
        // (intersection points of plane with bounding box) to
        // viewerPos (users eye point)
        int min = 0, max = 0;
        double dist = (viewerPos_o - drawPoints_[0]).length();
        double dist_min = dist, dist_max = dist;

        for (int i = 1; i < numDrawPoints_; i++)
        {
            dist = (viewerPos_o - drawPoints_[i]).length();
            //fprintf(stderr,"             drawPoints[%d] (%f %f %f)\n", i, drawPoints_[i].x(), drawPoints_[i].y(), drawPoints_[i].z());
            if (dist < dist_min)
            {
                dist_min = dist;
                min = i;
            }
            if (dist > dist_max)
            {
                dist_max = dist;
                max = i;
            }
        }

        // place nametag 3/4 on the line between dist_min and dist_max
        nameTagPosition_ = drawPoints_[min] / 4.0 + (drawPoints_[max] * 3.0) / 4.0;
        Matrix o_to_w = cover->getBaseMat();
        //Vec3 p = nameTagPosition_* o_to_w; fprintf(stderr,"coVRPlane::updateNameTag pos (%f %f %f)\n",p.x(),p.y(),p.z());
        nameTag_->setPosition(nameTagPosition_ * o_to_w);

        // make text
        string text = computeNameTagText();
        nameTag_->setString(text.c_str());
    }
    else
    {
        // adjust nametag position to maybe changed BaseMat
        Matrix o_to_w = cover->getBaseMat();
        //Vec3 p = nameTagPosition_* o_to_w; fprintf(stderr,"coVRPlane::updateNameTag pos (%f %f %f)\n",p.x(),p.y(),p.z());
        nameTag_->setPosition(nameTagPosition_ * o_to_w);
    }
}

//----------------------------------------------------------------------
void coVRPlane::menuEvent(coMenuItem *menuItem)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::menuEvent for %s\n", menuItem->getName());

    point1_->menuEvent(menuItem);
    point2_->menuEvent(menuItem);
    point3_->menuEvent(menuItem);

    direction1_->menuEvent(menuItem);
    direction2_->menuEvent(menuItem);
    normalDirection_->menuEvent(menuItem);

    if (pointsModeCheckbox_
        && menuItem == pointsModeCheckbox_
        && pointsModeCheckbox_->getState()
        && mode_ != POINT_POINT_POINT)
    {
        setMode(POINT_POINT_POINT);
    }
    else if (normalModeCheckbox_
             && menuItem == normalModeCheckbox_
             && normalModeCheckbox_->getState()
             && mode_ != POINT_DIR)
    {
        setMode(POINT_DIR);
    }
    else if (directionsModeCheckbox_
             && menuItem == directionsModeCheckbox_
             && directionsModeCheckbox_->getState()
             && mode_ != POINT_DIR_DIR)
    {
        setMode(POINT_DIR_DIR);
    }
}

//----------------------------------------------------------------------
int coVRPlane::addToMenu(coRowMenu *parentMenu, int position)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::addToMenu\n");

    parentMenu_ = parentMenu;

    // making mode checkboxes
    modeGroup_ = new coCheckboxGroup(false);

    //  points mode checkboxes
    if (!pointsModeCheckbox_)
    {
        string modeText = name_;
        modeText.append(coTranslator::coTranslate(": Punktemodus"));
        pointsModeCheckbox_ = new coCheckboxMenuItem(modeText.c_str(), false, modeGroup_);
        pointsModeCheckbox_->setMenuListener(MathematicPlugin::plugin);
    }
    parentMenu_->insert(pointsModeCheckbox_, ++position);

    //  normal mode checkboxes
    if (!normalModeCheckbox_)
    {
        string modeText = name_;
        modeText.append(coTranslator::coTranslate(": Normalenmodus"));
        normalModeCheckbox_ = new coCheckboxMenuItem(modeText.c_str(), false, modeGroup_);
        normalModeCheckbox_->setMenuListener(MathematicPlugin::plugin);
    }
    parentMenu_->insert(normalModeCheckbox_, ++position);

    //  directions mode checkboxes
    if (!directionsModeCheckbox_)
    {
        string modeText = name_;
        modeText.append(coTranslator::coTranslate(": Richtungsmodus"));
        directionsModeCheckbox_ = new coCheckboxMenuItem(modeText.c_str(), false, modeGroup_);
        directionsModeCheckbox_->setMenuListener(MathematicPlugin::plugin);
    }
    parentMenu_->insert(directionsModeCheckbox_, ++position);

    if (mode_ == POINT_POINT_POINT)
        pointsModeCheckbox_->setState(true);
    else if (mode_ == POINT_DIR)
        normalModeCheckbox_->setState(true);
    else if (mode_ == POINT_DIR_DIR)
        directionsModeCheckbox_->setState(true);

    // pointMenus
    position = point1_->addToMenu(parentMenu_, ++position, false);
    position = point2_->addToMenu(parentMenu_, ++position, false);
    position = point3_->addToMenu(parentMenu_, ++position, false);

    // separator
    if (!sepLabel_)
        sepLabel_ = new coLabelMenuItem("______________");
    parentMenu_->insert(sepLabel_, ++position);

    return position;
}

//----------------------------------------------------------------------
void coVRPlane::removeFromMenu()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::removeFromMenu\n");

    //  points mode checkboxes
    if (pointsModeCheckbox_)
        parentMenu_->remove(pointsModeCheckbox_);

    //  normal mode checkboxes
    if (normalModeCheckbox_)
        parentMenu_->remove(normalModeCheckbox_);

    //  directions mode checkboxes
    if (directionsModeCheckbox_)
        parentMenu_->remove(directionsModeCheckbox_);

    // pointMenus
    point1_->removeFromMenu();
    point2_->removeFromMenu();
    point3_->removeFromMenu();

    // separator
    if (sepLabel_)
        parentMenu_->remove(sepLabel_);
}

//----------------------------------------------------------------------
int coVRPlane::test(coVRLine *line, Vec3 *isectPoint, double *angle)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::test Line\n");

    Vec3 p1 = line->getBasePoint();
    Vec3 p2 = line->getBasePoint() + line->getDirection();

    if (isParallel(line))
    {
        //fprintf(stderr,"coVRPlane::test PARALLEL\n");

        // test if basepoint of line lies in plane
        if (getPointDistance(p1) == 0)
            return LIESIN;
        else
            return PARALLEL;
    }
    else if (coPlane::getLineIntersectionPoint(p1, p2, *isectPoint))
    {
        //fprintf(stderr,"coVRPlane::test INTERSECT (%f %f %f) \n",isectPoint->x(),isectPoint->y(),isectPoint->z());
        *angle = computeAngle(line->getDirection(), normalDirection_->getDirection());
        return INTERSECT;
    }
    else
        return -1;
}

//----------------------------------------------------------------------
int coVRPlane::test(coVRPlane *otherPlane, Vec3 *isectLinePoint1,
                    Vec3 *isectLinePoint2, double *angle)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::test Plane\n");

    if (isParallel(otherPlane))
    {
        //fprintf(stderr,"coVRPlane::test PARALLEL\n");
        return PARALLEL;
    }
    else if (isIntersection(otherPlane, isectLinePoint1, isectLinePoint2, angle))
    {
        //fprintf(stderr,"coVRPlane::test PARALLEL\n");
        return INTERSECT;
    }

    return -1;
}

//----------------------------------------------------------------------
bool coVRPlane::isParallel(coVRLine *line)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::isParallel Line\n");

    // if line direction and normal direction are orthogonal
    // line and plane are parallel
    if (line->getDirection() * normalDirection_->getDirection() == 0)
        return true;

    return false;
}

//----------------------------------------------------------------------
bool coVRPlane::isParallel(coVRPlane *otherPlane)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::isParallel otherPlane\n");

    Vec3 thisNormal = normalDirection_->getDirection();
    thisNormal.normalize();
    Vec3 otherNormal = otherPlane->getNormalDirection();
    otherNormal.normalize();

    // parallel if normal directions are the same
    if (thisNormal == otherNormal || thisNormal == -otherNormal)
        return true;

    return false;
}

//----------------------------------------------------------------------
bool coVRPlane::intersect(coVRLine *line, osg::Vec3 *isectPoint, double *angle)
{
    Vec3 p1 = line->getBasePoint();
    Vec3 p2 = line->getBasePoint() + line->getDirection();

    *angle = computeAngle(line->getDirection(), normalDirection_->getDirection());

    return coPlane::getLineIntersectionPoint(p1, p2, *isectPoint);
}

//----------------------------------------------------------------------
bool coVRPlane::intersect(coVRPlane *otherPlane, osg::Vec3 *isectLinePoint1, osg::Vec3 *isectLinePoint2, double *angle)
{

    if (isParallel(otherPlane))
        return false;

    isIntersection(otherPlane, isectLinePoint1, isectLinePoint2, angle);

    return true;
}

//----------------------------------------------------------------------
bool coVRPlane::isIntersection(coVRPlane *otherPlane, osg::Vec3 *isectLinePoint1, osg::Vec3 *isectLinePoint2, double *angle)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::isIntersection\n");

    Vec3 p1 = point1_->getPosition();
    Vec3 p2 = otherPlane->getBasePoint();
    Vec3 n1 = normalDirection_->getDirection();
    Vec3 n2 = otherPlane->getNormalDirection();

    // plane : n_*X = d_
    double d1 = p1.x() * n1.x() + p1.y() * n1.y() + p1.z() * n1.z();
    double d2 = p2.x() * n2.x() + p2.y() * n2.y() + p2.z() * n2.z();

    // intersection line : p = c1 n1 + c2 n2 + u n1^n2
    double det = ((n1 * n1) * (n2 * n2)) - ((n1 * n2) * (n1 * n2));
    if (det == 0.0)
        return false;

    // taking the dot product with each normal
    // gives two equations with unknowns c1 and c2
    // n_*X = d_ = c1 n1*n_ + c2 n_*n2
    double c1 = ((n2 * n2) * d1) - ((n1 * n2) * d2);
    c1 /= det;
    double c2 = ((n1 * n1) * d2) - ((n1 * n2) * d1);
    c2 /= det;

    Vec3 lineDirection = n1 ^ n2;
    lineDirection.normalize();

    isectLinePoint1->set(n1 * c1 + n2 * c2);
    isectLinePoint2->set(*isectLinePoint1 + lineDirection);

    //fprintf(stderr,"coVRPlane::isIntersection p2 (%f %f %f)\n", isectLinePoint2->x(),isectLinePoint2->y(),isectLinePoint2->z());

    // returns angle
    *angle = computeAngle(n1, n2);

    return true;
}

//----------------------------------------------------------------------
double coVRPlane::distance(coVRPlane *otherPlane, Vec3 *perpendicularP1, Vec3 *perpendicularP2)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::distance Plane\n");

    Vec3 oPoint1 = otherPlane->getBasePoint();
    perpendicularP2->set(oPoint1);

    if (isParallel(otherPlane))
    {
        // are planes the same, distance is null
        if (getPointDistance(oPoint1) == 0)
        {
            perpendicularP1->set(oPoint1);
            return 0;
        }

        // intersection with line (other plane base point and normal direction)
        Vec3 isectPoint = Vec3(0.0, 0.0, 0.0);
        Vec3 linePoint = otherPlane->getBasePoint() + otherPlane->getNormalDirection();
        coPlane::getLineIntersectionPoint(oPoint1, linePoint, *perpendicularP1);

        return (*perpendicularP2 - *perpendicularP1).length();
    }

    // not parallel, therefore intersection, therefore distance null
    perpendicularP1->set(oPoint1);
    return 0;
}

//----------------------------------------------------------------------
double coVRPlane::distance(coVRLine *line, Vec3 *perpendicularP, Vec3 *perpendicularL)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::distance Line\n");

    if (isParallel(line))
    {
        // perpendicular on plane  is intersection point of a line
        // (with base point of line and normalDirection) and this plane
        // perpendicular of line is then its base point
        Vec3 b = line->getBasePoint();
        Vec3 lP = b + normalDirection_->getDirection();
        coPlane::getLineIntersectionPoint(b, lP, *perpendicularP);
        perpendicularL->set(b);

        // distance is the distance between plane and the base point of line
        return coPlane::getPointDistance(b);
    }

    // if not parallel then they have to intersect
    // which means distance is null
    return 0;
}

//----------------------------------------------------------------------
double coVRPlane::distance(coVRPoint *point, Vec3 *perpendicular)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::distance Point\n");

    Vec3 p = point->getPosition();
    Vec3 lP = p + normalDirection_->getDirection(); // point on line

    // distance
    double dist = coPlane::getPointDistance(p);

    if (dist)
    {
        // perpendicular is intersection point of line (with point and normalDirection)
        // and this plane
        coPlane::getLineIntersectionPoint(p, lP, *perpendicular);
    }
    else
    {
        // distance is null, so perpendicular is point
        perpendicular->set(p);
    }

    return dist;
}

//----------------------------------------------------------------------
double coVRPlane::computeAngle(Vec3 direction1, Vec3 direction2)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::computeAngle\n");

    //            dir1*dir2
    // cos phi = -----------
    //           |dir1| * |dir2|
    double phi = acos((direction1 * direction2) / (direction1.length() * direction2.length()));

    // choose the bigger angle -> phi+phi2 = 180 DEG
    double phi2 = MATH_PI - phi;
    //fprintf(stderr,"coVRPlane::computeAngle phi %f phi2 %f\n", (180./MATH_PI) * phi, (180./MATH_PI) * phi2 );
    if (phi2 < phi)
        phi = phi2;

    // in degrees = 180°/phi * MATH_PI
    return (180. / MATH_PI) * phi;
}

//----------------------------------------------------------------------
void coVRPlane::makeNameTag()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::makeNameTag\n");

    Vec4 fgColor(0.5451, 0.7020, 0.2431, 1.0);
    Vec4 bgColor(0.0, 0.0, 0.0, 0.8);
    double linelen = 0.04 * cover->getSceneSize();
    double fontsize = 20;

    string text = computeNameTagText();
    nameTag_ = new coVRLabel(text.c_str(), fontsize, linelen, fgColor, bgColor);

    // set to default position
    nameTagPosition_ = Vec3(0, 0, 0);
    nameTag_->setPosition(nameTagPosition_);
}

//----------------------------------------------------------------------
void coVRPlane::setMode(int mode)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::setMode %d\n", mode);

    mode_ = mode;
    if (mode_ == POINT_POINT_POINT)
    {
        point2_->setVisible(true);
        point2_->addToMenu(parentMenu_, point1_->getMenuPosition(), false);
        point3_->setVisible(true);
        point3_->addToMenu(parentMenu_, point2_->getMenuPosition(), false);

        direction1_->setVisible(false);
        direction1_->removeFromMenu();
        direction2_->setVisible(false);
        direction2_->removeFromMenu();
        normalDirection_->setVisible(false);
        normalDirection_->removeFromMenu();
    }
    else if (mode_ == POINT_DIR)
    {
        point2_->setVisible(false);
        point2_->removeFromMenu();
        point3_->setVisible(false);
        point3_->removeFromMenu();

        direction1_->setVisible(false);
        direction1_->removeFromMenu();
        direction2_->setVisible(false);
        direction2_->removeFromMenu();
        normalDirection_->setVisible(true);
        normalDirection_->addToMenu(parentMenu_, point1_->getMenuPosition());
    }
    else if (mode_ == POINT_DIR_DIR)
    {
        point2_->setVisible(false);
        point2_->removeFromMenu();
        point3_->setVisible(false);
        point3_->removeFromMenu();

        direction1_->setVisible(true);
        direction1_->addToMenu(parentMenu_, point1_->getMenuPosition());
        direction2_->setVisible(true);
        direction2_->addToMenu(parentMenu_, direction1_->getMenuPosition());
        normalDirection_->setVisible(false);
        normalDirection_->removeFromMenu();
    }
    else
    {
        fprintf(stderr, "ERROR: wrong mode for coVRPlane\n");
        return;
    }

    if (pointsModeCheckbox_ && normalModeCheckbox_ && directionsModeCheckbox_)
    {
        if (mode_ == POINT_POINT_POINT)
        {
            pointsModeCheckbox_->setState(true);
            normalModeCheckbox_->setState(false);
            directionsModeCheckbox_->setState(false);
        }
        else if (mode_ == POINT_DIR)
        {
            pointsModeCheckbox_->setState(false);
            normalModeCheckbox_->setState(true);
            directionsModeCheckbox_->setState(false);
        }
        else if (mode_ == POINT_DIR_DIR)
        {
            pointsModeCheckbox_->setState(false);
            normalModeCheckbox_->setState(false);
            directionsModeCheckbox_->setState(true);
        }
    }
    hideLabel(labelsShown_);
}

//----------------------------------------------------------------------
void coVRPlane::setVisible(bool visible)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::setVisible %d\n", visible);

    isVisible_ = visible;

    if (isVisible_)
    {
        // root node
        node_->setNodeMask(node_->getNodeMask() | (Isect::Visible));

        // points and directions according to mode
        if (mode_ == POINT_POINT_POINT)
        {
            point1_->setVisible(true);
            point2_->setVisible(true);
            point3_->setVisible(true);
        }
        else if (mode_ == POINT_DIR)
        {
            point1_->setVisible(true);
            normalDirection_->setVisible(true);
        }
        else if (mode_ == POINT_DIR_DIR)
        {
            point1_->setVisible(true);
            direction1_->setVisible(true);
            direction2_->setVisible(true);
        }

        // nametag
        nameTag_->show();
        updateNameTag(true);

        hideLabel(labelsShown_);
    }
    else
    {
        // root node
        node_->setNodeMask(node_->getNodeMask() & (~Isect::Visible));

        point1_->setVisible(false);
        point2_->setVisible(false);
        point3_->setVisible(false);
        direction1_->setVisible(false);
        direction2_->setVisible(false);
        normalDirection_->setVisible(false);

        // nameTag
        nameTag_->hide();
    }
}

//----------------------------------------------------------------------
void coVRPlane::setColor(Vec4 color)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::setColor (%f %f %f   %f)\n", color.x(), color.y(), color.z(), color.w());

    color_ = color;
    material_->setDiffuse(Material::FRONT_AND_BACK, color_);
    material_->setAmbient(Material::FRONT_AND_BACK, Vec4(color_.x() * 0.3, color_.y() * 0.3, color_.z() * 0.3, color_.w()));
}

//----------------------------------------------------------------------
void coVRPlane::setBoundingBox(BoundingBox *boundingBox)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::setBoundingBox\n");

    _boundingBox_ = boundingBox;
}

//----------------------------------------------------------------------
// called from extern
bool coVRPlane::isChanged()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::isChanged\n");

    if (isChanged_)
    {
        isChanged_ = false;
        return true;
    }

    return isChanged_;
}

//----------------------------------------------------------------------
bool coVRPlane::isVisible()
{
    return isVisible_;
}

//----------------------------------------------------------------------
Vec3 coVRPlane::getBasePoint()
{
    return point1_->getPosition();
}

//----------------------------------------------------------------------
string coVRPlane::getName()
{
    return name_;
}

//----------------------------------------------------------------------
Vec3 coVRPlane::getNormalDirection()
{
    return normalDirection_->getDirection();
}

//----------------------------------------------------------------------
void coVRPlane::makeColor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::makeColor\n");

    material_ = new Material();
    stateSet_ = VRSceneGraph::instance()->loadDefaultGeostate();
    stateSet_->setMode(GL_BLEND, StateAttribute::ON);
    stateSet_->setAttributeAndModes(material_);
    planeGeode_->setStateSet(stateSet_);
}

//----------------------------------------------------------------------
Vec3 coVRPlane::computePoint2(Vec3 normalVec)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::computePoint2\n");

    // to find the second point on the plane we find
    // the projection of the x axis on the plane
    // therefore rotate the x axis so, that the normal
    // is the z axis and translate this coordinate system
    // into point1
    Matrix m2;
    m2.makeRotate(Vec3(0.0, 0.0, 1.0), normalVec);
    m2.setTrans(point1_->getPosition());
    Vec3 point2 = Vec3(1.0, 0.0, 0.0);
    point2 = point2 * m2;

    // if point doesn't lie in Boundingbox move closer to (base)point1
    if (!_boundingBox_->contains(point2))
    {
        //fprintf(stderr, "point2 (%f %f %f)\n",point2.x(),point2.y(),point2.z());
        Vec3 dist = point2 - point1_->getPosition();
        dist.normalize();
        point2 = point1_->getPosition() + dist * 3;
    }
    return point2;
}

//----------------------------------------------------------------------
Vec3 coVRPlane::computePoint3(Vec3 normalVec)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::computePoint3\n");

    // the third point is orthogonal to the other two
    // therefore find vector which is orthogonal
    // (crossproduct) to normal and direction1
    // then translate along point1
    Vec3 point3 = direction1_->getDirection() ^ normalVec;
    Matrix m3;
    m3.makeTranslate(point1_->getPosition());
    point3 = point3 * m3;

    // if point doesn't lie in Boundingbox move closer to (base)point1
    if (!_boundingBox_->contains(point3))
    {
        //fprintf(stderr, "point3 (%f %f %f)\n",point3.x(),point3.y(),point3.z());
        Vec3 dist = point3 - point1_->getPosition();
        dist.normalize();
        point3 = point1_->getPosition() + dist * 3;
    }

    return point3;
}

//----------------------------------------------------------------------
void coVRPlane::updateNormal()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::updateNormal\n");

    // computes and sets normal from dir1 and dir2
    // make sure it has the same length and direction as the old one

    // normal of the plane is the cross product of the two directions
    Vec3 norDir = direction1_->getDirection() ^ direction2_->getDirection();

    // dot product of normal
    //float dotProduct = norDir*oldNormalDirection_;
    //fprintf(stderr,"coVRPlane::update dotpro %f \n", dotProduct);

    // adjusting length of normal
    double length = oldNormalDirection_.length();
    //fprintf(stderr,"coVRPlane::update norDir.length %f\n",length);
    norDir.normalize();
    norDir = norDir * length;

    // adjusting side of normal
    //if( dotProduct < 0 )
    //norDir = norDir*-1;

    // remember old value and set new one
    oldNormalDirection_ = normalDirection_->getDirection();
    normalDirection_->setDirection(norDir);
}

//----------------------------------------------------------------------
void coVRPlane::showEquations(bool showParamEqu, bool showCoordEqu, bool showNormEqu)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPlane::showEquations\n");

    showParamEqu_ = showParamEqu;
    showCoordEqu_ = showCoordEqu;
    showNormEqu_ = showNormEqu;

    updateNameTag(true);
}

//----------------------------------------------------------------------
string coVRPlane::computeNameTagText()
{
    ostringstream textStream;
    string text = name_;

    // parametric form
    if (showParamEqu_)
    {
        textStream << " x = (";
        textStream << MathematicPlugin::round10(point1_->x());
        textStream << ", ";
        textStream << MathematicPlugin::round10(point1_->y());
        textStream << ", ";
        textStream << MathematicPlugin::round10(point1_->z());
        textStream << ") + s (";
        textStream << MathematicPlugin::round10(direction1_->x());
        textStream << ", ";
        textStream << MathematicPlugin::round10(direction1_->y());
        textStream << ", ";
        textStream << MathematicPlugin::round10(direction1_->z());
        textStream << ") + t (";
        textStream << MathematicPlugin::round10(direction2_->x());
        textStream << ", ";
        textStream << MathematicPlugin::round10(direction2_->y());
        textStream << ", ";
        textStream << MathematicPlugin::round10(direction2_->z());
        textStream << ")\n";
    }

    // coordinate form
    if (showCoordEqu_)
    {
        textStream << name_;
        textStream << "   ";
        textStream << MathematicPlugin::round10(normalDirection_->x());
        textStream << "x ";
        double y = MathematicPlugin::round10(normalDirection_->y());
        if (y >= 0)
            textStream << "+ ";
        else
        {
            textStream << "- ";
            y *= (-1);
        }
        textStream << y;
        textStream << "y";
        double z = MathematicPlugin::round10(normalDirection_->z());
        if (z >= 0)
            textStream << "+ ";
        else
        {
            textStream << "- ";
            z *= (-1);
        }
        textStream << z;
        textStream << "z ";
        double cons = MathematicPlugin::round10(
            point1_->x() * normalDirection_->x() + point1_->y() * normalDirection_->y() + point1_->z() * normalDirection_->z());
        if (cons >= 0)
            textStream << "+ ";
        else
        {
            textStream << "- ";
            cons *= (-1);
        }
        textStream << cons;
        textStream << " = 0\n";
    }

    // normal form
    if (showNormEqu_)
    {
        textStream << name_;
        textStream << "   [ X - (";
        textStream << MathematicPlugin::round10(point1_->x());
        textStream << ", ";
        textStream << MathematicPlugin::round10(point1_->y());
        textStream << ", ";
        textStream << MathematicPlugin::round10(point1_->z());
        textStream << ") ] * (";
        textStream << MathematicPlugin::round10(normalDirection_->x());
        textStream << ", ";
        textStream << MathematicPlugin::round10(normalDirection_->y());
        textStream << ", ";
        textStream << MathematicPlugin::round10(normalDirection_->z());
        textStream << ")";
    }

    text.append(textStream.str());

    return text;
}

//----------------------------------------------------------------------
void coVRPlane::hideLabel(bool hide)
{
    labelsShown_ = hide;
    if (isVisible())
    {
        if (hide)
            nameTag_->hide();
        else
            nameTag_->show();
        if (point1_->isVisible())
            point1_->hideLabel(hide);
        if (normalDirection_->isVisible())
            normalDirection_->hideLabel(hide);
        if (point2_->isVisible())
            point2_->hideLabel(hide);
        if (direction1_->isVisible())
            direction1_->hideLabel(hide);
        if (point3_->isVisible())
            point3_->hideLabel(hide);
        if (direction2_->isVisible())
            direction2_->hideLabel(hide);
    }
}
