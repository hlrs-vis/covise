/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2009 Visenso  **
 **                                                                        **
 ** Description: coVRLine                                                  **
 **              Draws a line according to mode                            **
 **               either as a line through two points                      **
 **               or a line with a base point and a direction              **
 **               with a nameTag and equation                              **
 **               only within the bounding box (needs to be set)           **
 **                                                                        **
 ** Author: M. Theilacker                                                  **
 **                                                                        **
 ** History:                                                               **
 **     4.2009 initial version                                             **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include <math.h> //acos

#include <cover/coVRPluginSupport.h>

#include <cover/coTranslator.h>

#include "MathematicPlugin.h"

#include "coVRLine.h"

const double EPSILON = 0.3;
const double MATH_PI = 3.14159265358979323846264338327950;

using namespace osg;

// variables of the class
int coVRLine::_lineID_ = 0;
BoundingBox *coVRLine::_boundingBox_ = NULL;
vector<coPlane *> coVRLine::_boundingPlanes_;

//
// Constructor
//
coVRLine::coVRLine(Vec3 vec1, Vec3 vec2, int mode, string name, bool normal, double radius)
    : name_(name)
    , node_(NULL)
    , mode_(mode)
    , point1_(NULL)
    , point2_(NULL)
    , direction_(NULL)
    , parentMenu_(NULL)
    , modeCheckbox_(NULL)
    , sepLabel_(NULL)
    , line_(NULL)
    , lineGeode_(NULL)
    , lineDraw_(NULL)
    , lineRadius_(radius)
    , isVisible_(true)
    , isChanged_(false)
    , nameTag_(NULL)
    , normal_(normal)
    , inBoundingBox_(true)
    , isBBSet_(false)
    , labelsShown_(false)
{
    if (cover->debugLevel(1))
        fprintf(stderr, "coVRLine::coVRLine %s\n", name.c_str());

    _lineID_++;

    // set boundingbox if not already set
    if (!_boundingBox_)
    {
        fprintf(stderr, "WARNING: coVRLine has no bounding box, will be set to 10\n");
        double boundary = 10.;
        BoundingBox *boundingBox = new BoundingBox(-boundary, -boundary, -boundary,
                                                   boundary, boundary, boundary);
        setBoundingBox(boundingBox);
        isBBSet_ = true;
    }

    // make unique name
    ostringstream numStream;
    numStream << _lineID_;
    name_.append(numStream.str());

    // root node
    node_ = new MatrixTransform();
    node_->ref();
    node_->setName(name_);

    // settings for different modes
    if (mode_ == POINT_POINT)
    {
        if (vec1 == vec2)
        {
            fprintf(stderr, "ERROR: Can not create a line of equal points!\n   Changing point a little");
            vec2.set(vec2.x(), vec2.y(), vec2.z() + 1.0);
        }

        point1_ = new coVRPoint(vec1);
        point2_ = new coVRPoint(vec2);
        direction_ = new coVRDirection(vec2 - vec1, point1_->getPosition());

        direction_->setVisible(false);
    }
    else
    {
        if (mode_ != POINT_DIR)
            fprintf(stderr, "ERROR: wrong mode for coVRLine assuming mode POINT_DIR \n");

        point1_ = new coVRPoint(vec1);
        direction_ = new coVRDirection(vec2, point1_->getPosition());
        point2_ = new coVRPoint(vec1 + direction_->getDirection());

        point1_->setVisible(true);
        direction_->setVisible(true);
        point2_->setVisible(false);
    }

    // compute line
    line_ = new Cylinder();
    lineDraw_ = new ShapeDrawable();
    lineGeode_ = new Geode();
    lineDraw_->setShape(line_);
    lineGeode_->addDrawable(lineDraw_);
    lineGeode_->setNodeMask(lineGeode_->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));
    lineDraw_->setUseDisplayList(false);
    node_->addChild(lineGeode_);
    int error = updateLine();
    if (!error)
    {
        inBoundingBox_ = _boundingBox_->contains(drawMin_) && _boundingBox_->contains(drawMax_);
    }

    // make color yellow
    makeColor();
    setColor(Vec4(1.0, 1.0, 0.0, 1.0));

    // name tag as pinboard
    makeNameTag();
    updateNameTag();

    // line lies not within bounding box
    if (error || !inBoundingBox_)
        hide();

    if (!normal)
    {
        point1_->setVisible(false);
        point2_->setVisible(false);
        direction_->setVisible(false);
        setVisible(false);
    }

    // add root node to cover scenegraph
    cover->getObjectsRoot()->addChild(node_.get());
}

//
// Destructor
//
coVRLine::~coVRLine()
{
    if (cover->debugLevel(1))
        fprintf(stderr, "coVRLine::~coVRLine\n");

    if (isBBSet_)
        delete _boundingBox_;

    cover->getObjectsRoot()->removeChild(node_.get());
    node_->unref();

    delete point1_;
    delete point2_;
    delete direction_;

    delete modeCheckbox_;
    delete sepLabel_;

    delete nameTag_;
}

//----------------------------------------------------------------------
// setting the line visible only if inBoundingBox_
//
void coVRLine::setVisible(bool visible)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRLine::setVisible %d\n", visible);

    isVisible_ = visible;

    if (isVisible_ && inBoundingBox_)
    {
        show();
        hideLabel(labelsShown_);
    }
    else
        hide();
}

//----------------------------------------------------------------------
void coVRLine::hide()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRLine::hide\n");

    // root node
    node_->setNodeMask(node_->getNodeMask() & (~Isect::Visible));

    // point and direction
    if (mode_ == POINT_POINT && normal_)
    {
        point1_->setVisible(false);
        point2_->setVisible(false);
    }
    else if (mode_ == POINT_DIR && normal_)
    {
        point1_->setVisible(false);
        direction_->setVisible(false);
    }

    // nameTag
    nameTag_->hide();
}

//----------------------------------------------------------------------
void coVRLine::show()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRLine::show\n");

    // root node
    node_->setNodeMask(node_->getNodeMask() | (Isect::Visible));

    // point and direction dependend on mode
    if (mode_ == POINT_POINT && normal_)
    {
        point1_->setVisible(true);
        point2_->setVisible(true);
    }
    else if (mode_ == POINT_DIR && normal_)
    {
        point1_->setVisible(true);
        direction_->setVisible(true);
    }

    // nametag
    nameTag_->show();
    updateNameTag();
}

//----------------------------------------------------------------------
void coVRLine::setBoundingBox(BoundingBox *boundingBox)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRLine::setBoundingBox\n");

    _boundingBox_ = boundingBox;
    _boundingPlanes_.push_back(new coPlane(Vec3(1.0, 0.0, 0.0), _boundingBox_->_min));
    _boundingPlanes_.push_back(new coPlane(Vec3(0.0, 1.0, 0.0), _boundingBox_->_min));
    _boundingPlanes_.push_back(new coPlane(Vec3(0.0, 0.0, 1.0), _boundingBox_->_min));
    _boundingPlanes_.push_back(new coPlane(Vec3(1.0, 0.0, 0.0), _boundingBox_->_max));
    _boundingPlanes_.push_back(new coPlane(Vec3(0.0, 1.0, 0.0), _boundingBox_->_max));
    _boundingPlanes_.push_back(new coPlane(Vec3(0.0, 0.0, 1.0), _boundingBox_->_max));
}

//----------------------------------------------------------------------
void coVRLine::deleteBoundingBox()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRLine::deleteBoundingBox\n");

    // only delete boundingPlanes at last line
    for (int i = 0; i < _boundingPlanes_.size(); i++)
        delete _boundingPlanes_.at(i);
}

//----------------------------------------------------------------------
// setting drawMin_ + drawMax_
// setting inBoundingBox_ = false, if not in bounding box, else true
//
int coVRLine::findEndPoints()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRLine::findEndPoints()\n");

    // find the intersections with the bounding planes
    // only use these who lie in the bounding box
    // there should be only two
    vector<Vec3> isectPoints;
    bool found = false;

    for (int i = 0; i < _boundingPlanes_.size(); i++)
    {
        Vec3 isectPoint;
        Vec3 pos1 = point1_->getPosition();
        Vec3 pos2 = point2_->getPosition();
        _boundingPlanes_.at(i)->getLineIntersectionPoint(pos1, pos2, isectPoint);
        //fprintf(stderr,"                  coVRLine::findEndPoints isectPoint (%f %f %f)\n\n", isectPoint.x(), isectPoint.y(),isectPoint.z());

        // isect points of planes should be in bounding box and not be the origin
        if (_boundingBox_->contains(isectPoint * 0.9999f + _boundingBox_->center() * 0.0001f) && isectPoint.length() != 0.0)
            isectPoints.push_back(isectPoint);
    }

    // there should be an intersection with the bounding box
    if (isectPoints.size() < 2)
    {
        // line lies not within bounding box
        inBoundingBox_ = false;

        // drawMin_ and drawMax_ need to be defined
        drawMin_ = Vec3(-1.0, 0.0, 0.0);
        drawMax_ = Vec3(1.0, 0.0, 0.0);

        return -1;
    }

    drawMin_ = isectPoints.at(0);

    for (int i = 1; i < isectPoints.size(); i++)
    {
        // distance between the two points smaller than epsilon
        // according to rounding errors
        if ((isectPoints.at(i) - drawMin_).length() > EPSILON)
        {
            drawMax_ = isectPoints.at(i);
            found = true;
            break;
        }
    }
    //fprintf(stderr,"coVRLine::findEndPoints %s   drawMin_ (%f %f %f) drawMax_ (%f %f %f)\n", name_.c_str(), drawMin_.x(), drawMin_.y(),drawMin_.z(), drawMax_.x(), drawMax_.y(),drawMax_.z());

    if (!found)
    {
        // line lies not within bounding box
        inBoundingBox_ = false;

        // drawMin_ and drawMax_ need to be defined
        //drawMin_ = Vec3(-1.0,0.0,0.0);
        drawMax_ = Vec3(1.0, 0.0, 0.0);
        return -1;
    }

    inBoundingBox_ = true;

    return 0;
}

//----------------------------------------------------------------------
int coVRLine::updateLine()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRLine::updateLine\n");

    int error = findEndPoints();
    // do not adjust line if no end points are found
    // (drawMin_, drawMax_ are not set)
    if (error)
        return -1;

    Vec3 center = (drawMin_ + drawMax_) * 0.5;
    double length = (drawMin_ - drawMax_).length();
    //fprintf(stderr,"coVRLine::updateLine %d center (%f %f %f) l = %f\n", _lineID_, center.x(), center.y(),center.z(), length);

    line_->set(center, lineRadius_, length);
    Matrix rotation;
    rotation.makeRotate(Vec3(0.0, 0.0, 1.0), direction_->getDirection());
    line_->setRotation(rotation.getRotate());

    return 0;
}

//----------------------------------------------------------------------
int coVRLine::test(coVRLine *otherLine, Vec3 *isectPoint, double *angle)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRLine::test\n");

    // check distance
    Vec3 p1 = Vec3();
    Vec3 p2 = Vec3();
    double EPS = 0.1;
    double dist = distance(otherLine, &p1, &p2);
    if (dist < EPS)
    {
        if (isParallel(otherLine))
        {
            //fprintf(stderr,"coVRLine::test PARALLEL\n");
            return PARALLEL;
        }
        else if (isIntersection(otherLine, isectPoint, angle))
        {
            //fprintf(stderr,"coVRLine::test INTERSECT (%f %f %f) \n",isectPoint->x(),isectPoint->y(),isectPoint->z());
            return INTERSECT;
        }
    }
    else if (isParallel(otherLine))
    {
        //fprintf(stderr,"coVRLine::test PARALLEL\n");
        return PARALLEL;
    }
    else if (isSkew(otherLine))
    {
        //fprintf(stderr,"coVRLine::test SKEW\n");
        return SKEW;
    }

    return -1;
}

//----------------------------------------------------------------------
bool coVRLine::isSkew(coVRLine *otherLine)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRLine::isSkew\n");

    // only test if the direction vector of this line, the direction
    // vector of the other line and a vector from one point of this line to
    // one point of the other line are linearly independent
    // therefore the determinate of the matix shouldn't be 0
    Vec3 dir = direction_->getDirection();
    Vec3 oDir = otherLine->getDirection();
    Vec3 vec = otherLine->getBasePoint() - point1_->getPosition();

    double det = dir.x() * oDir.y() * vec.z()
                 + oDir.x() * vec.y() * dir.z()
                 + vec.x() * dir.y() * oDir.z()
                 - vec.x() * oDir.y() * dir.z()
                 - oDir.x() * dir.y() * vec.z()
                 - dir.x() * vec.y() * oDir.z();

    //fprintf(stderr,"coVRLine::isSkew det %f dir (%f %f %f) oDir (%f %f %f) vec (%f %f %f)\n", det, dir.x(), dir.y(), dir.z(), oDir.x(), oDir.y(), oDir.z(), vec.x(), vec.y(), vec.z());

    if (det < 0)
        det *= (-1);

    return det > 0.002;
}

//----------------------------------------------------------------------
bool coVRLine::intersect(coVRLine *otherLine, Vec3 *isectPoint, double *angle)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRLine::intersect\n");

    if (isSkew(otherLine))
        return false;

    if (isParallel(otherLine))
        return false;

    isIntersection(otherLine, isectPoint, angle);

    return true;
}

//----------------------------------------------------------------------
bool coVRLine::isIntersection(coVRLine *otherLine, Vec3 *isectPoint, double *angle)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRLine::isIntersection\n");

    Vec3 otherDirection = otherLine->getDirection();
    Vec3 perpendicularL1 = Vec3();
    Vec3 perpendicularL2 = Vec3();
    double EPS = 0.1;

    // check distance
    double dist = distance(otherLine, &perpendicularL1, &perpendicularL2);
    if (dist < EPS)
    {
        isectPoint->set(perpendicularL1);

        // returns angle
        *angle = MathematicPlugin::computeAngle(direction_->getDirection(), otherDirection);
        return true;
    }

    return false;
}

//----------------------------------------------------------------------
bool coVRLine::isParallel(coVRLine *otherLine)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRLine::isParallel\n");

    Vec3 otherDirection = otherLine->getDirection();
    Vec3 d = direction_->getDirection();
    d.normalize();
    Vec3 oD = otherDirection;
    oD.normalize();

    //fprintf(stderr,"coVRLine::isParallel d(%f %f %f) oD(%f %f %f)\n", d.x(), d.y(),d.z(), oD.x(), oD.y(),oD.z());

    if (d == oD || d == -oD)
        return true;

    return false;
}

//----------------------------------------------------------------------
double coVRLine::distance(coVRLine *otherLine, Vec3 *perpendicularL1, Vec3 *perpendicularL2)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRLine::distance\n");

    // compute the line segment PaPb that is the shortest route between two
    //  lines P1P2 and P3P4. Calculate also the values of mua and mub where
    //   Pa = P1 + mua (P2 - P1)
    //   Pb = P3 + mub (P4 - P3)
    Vec3 p3 = otherLine->getBasePoint();
    Vec3 p4 = otherLine->getBasePoint() + otherLine->getDirection();
    Vec3 p13 = Vec3(point1_->x() - p3.x(), point1_->y() - p3.y(), point1_->z() - p3.z());
    Vec3 p43 = otherLine->getDirection();
    Vec3 p21 = direction_->getDirection();
    double EPS = 1.0e-5;

    double d1343 = p13.x() * p43.x() + p13.y() * p43.y() + p13.z() * p43.z();
    double d4321 = p43.x() * p21.x() + p43.y() * p21.y() + p43.z() * p21.z();
    double d1321 = p13.x() * p21.x() + p13.y() * p21.y() + p13.z() * p21.z();
    double d4343 = p43.x() * p43.x() + p43.y() * p43.y() + p43.z() * p43.z();
    double d2121 = p21.x() * p21.x() + p21.y() * p21.y() + p21.z() * p21.z();

    double denom = d2121 * d4343 - d4321 * d4321;

    if (fabs(denom) < EPS)
    {
        ///return false;
        if (cover->debugLevel(3))
            fprintf(stderr, "INFO: value denom in coVRLine::distance is to small: %f set to .00001\n", denom);
        denom = .00001;
    }

    double numer = d1343 * d4321 - d1321 * d4343;

    double mua = numer / denom;
    double mub = (d1343 + d4321 * mua) / d4343;

    perpendicularL1->set(point1_->x() + mua * p21.x(), point1_->y() + mua * p21.y(), point1_->z() + mua * p21.z());
    perpendicularL2->set(p3.x() + mub * p43.x(), p3.y() + mub * p43.y(), p3.z() + mub * p43.z());

    //fprintf(stderr,"coVRLine::distance perpendicularL1 %f %f %f perpendicularL2 %f %f %f\n", perpendicularL1->x(), perpendicularL1->y(), perpendicularL1->z(), perpendicularL2->x(), perpendicularL2->y(), perpendicularL2->z());

    // distance is length of vector between the two perpendiculars
    return (*perpendicularL2 - *perpendicularL1).length();
}

//----------------------------------------------------------------------
double coVRLine::distance(coVRPoint *point, Vec3 *perpendicular)
{
    Vec3 vL = direction_->getDirection();
    Vec3 w = point->getPosition() - point1_->getPosition();
    Vec3 uL = direction_->getDirection();
    uL.normalize();

    double b = (w * vL) / (vL * vL);

    perpendicular->set(point1_->getPosition() + (point2_->getPosition() - point1_->getPosition()) * b);

    double dist = (w - uL * (w * uL)).length();

    return dist;
}

//----------------------------------------------------------------------
void coVRLine::preFrame()
{
    //fprintf(stderr,"coVRLine::preFrame\n");

    if (isVisible_ && inBoundingBox_)
    {
        if (point1_->isVisible())
            point1_->preFrame();

        if (point2_->isVisible())
            point2_->preFrame();

        if (direction_->isVisible())
            direction_->preFrame();

        if (point1_->isChanged() || point2_->isChanged() || direction_->isChanged())
        {
            isChanged_ = true;
            update();
            //fprintf(stderr,"coVRLine::preFrame isChanged_ %d %d %d\n", point1_->isChanged() , point2_->isChanged() , direction_->isChanged());
        }
        else
        {
            // cover BaseMat changes per frame
            updateNameTag();
        }
    }
}

//----------------------------------------------------------------------
void coVRLine::update()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRLine::update\n");

    if (mode_ == POINT_POINT)
        direction_->setDirection(point2_->getPosition() - point1_->getPosition());
    else if (mode_ == POINT_DIR)
        point2_->setPosition(point1_->getPosition() + direction_->getDirection());

    direction_->setPosition(point1_->getPosition());
    updateLine(); //if line is not updated, inBoundingBox_ is false

    if (isVisible_ && inBoundingBox_)
        show();
    else
        hide();

    // cover BaseMat changes per frame
    updateNameTag();
}

//----------------------------------------------------------------------
void coVRLine::updateNameTag()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRLine::updateNameTag\n");

    // position: set to one quater between 1st and 2nd
    //  intersection with boundingBox
    Matrix o_to_w = cover->getBaseMat();
    Vec3 pos_w;
    pos_w = (drawMin_) / 4.0 + (drawMax_ * 3.0) / 4.0;
    pos_w = pos_w * o_to_w;
    nameTag_->setPosition(pos_w);

    // text
    string text = computeNameTagText();
    nameTag_->setString(text.c_str());
}

//----------------------------------------------------------------------
void coVRLine::menuEvent(coMenuItem *menuItem)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRLine::menuEvent for %s\n", menuItem->getName());

    point1_->menuEvent(menuItem);
    point2_->menuEvent(menuItem);
    direction_->menuEvent(menuItem);

    if (modeCheckbox_
        && menuItem == modeCheckbox_
        && mode_ != modeCheckbox_->getState())
        setMode(modeCheckbox_->getState());
}

//----------------------------------------------------------------------
int coVRLine::addToMenu(coRowMenu *parentMenu, int position)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRLine::addToMenu\n");

    parentMenu_ = parentMenu;

    // making mode checkbox
    if (!modeCheckbox_)
    {
        string modeText = name_;
        modeText.append(coTranslator::coTranslate(": Vektormodus"));
        modeCheckbox_ = new coCheckboxMenuItem(modeText.c_str(), mode_);
        modeCheckbox_->setMenuListener(MathematicPlugin::plugin);
    }
    parentMenu_->insert(modeCheckbox_, ++position);

    // pointMenus
    position = point1_->addToMenu(parentMenu_, ++position, false);
    position = point2_->addToMenu(parentMenu_, ++position, false);

    // separator
    if (!sepLabel_)
        sepLabel_ = new coLabelMenuItem("______________");
    parentMenu_->insert(sepLabel_, ++position);

    return position;
}

//----------------------------------------------------------------------
void coVRLine::removeFromMenu()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRLine:::removeFromMenu\n");

    if (modeCheckbox_)
        parentMenu_->remove(modeCheckbox_);

    // pointMenus
    point1_->removeFromMenu();
    point2_->removeFromMenu();

    // separator
    if (sepLabel_)
        parentMenu_->remove(sepLabel_);
}

//----------------------------------------------------------------------
void coVRLine::makeNameTag()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRLine::makeNameTag\n");

    Vec4 fgColor(0.5451, 0.7020, 0.2431, 1.0);
    Vec4 bgColor(0.0, 0.0, 0.0, 0.8);
    double linelen = 0.04 * cover->getSceneSize();
    double fontsize = 20;

    string text = computeNameTagText();
    nameTag_ = new coVRLabel(text.c_str(), fontsize, linelen, fgColor, bgColor);

    // set to default position
    nameTag_->setPosition(drawMax_);

    if (!inBoundingBox_)
        nameTag_->hide();
    else
        nameTag_->show();
}

//----------------------------------------------------------------------
void coVRLine::setMode(int mode)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRLine::setMode\n");

    mode_ = mode;
    if (mode_ == POINT_POINT)
    {
        point2_->setVisible(true);
        point2_->addToMenu(parentMenu_, point1_->getMenuPosition(), false);

        direction_->setVisible(false);
        direction_->removeFromMenu();
    }
    else if (mode_ == POINT_DIR)
    {
        point2_->setVisible(false);
        point2_->removeFromMenu();

        direction_->setVisible(true);
        direction_->addToMenu(parentMenu_, point1_->getMenuPosition());
    }
    else
    {
        fprintf(stderr, "ERROR: wrong mode for coVRLine\n");
        return;
    }
    hideLabel(labelsShown_);

    if (modeCheckbox_)
        modeCheckbox_->setState(mode_);
}

//----------------------------------------------------------------------
void coVRLine::setColor(Vec4 color)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRLine::setColor\n");

    color_ = color;
    material_->setDiffuse(Material::FRONT_AND_BACK, color_);
    material_->setAmbient(Material::FRONT_AND_BACK, Vec4(color_.x() * 0.3, color_.y() * 0.3, color_.z() * 0.3, color_.w()));
}

//----------------------------------------------------------------------
void coVRLine::setPoints(Vec3 point1, Vec3 point2)
{
    point1_->setPosition(point1);
    point2_->setPosition(point2);
    update();
}

//----------------------------------------------------------------------
Vec3 coVRLine::getBasePoint()
{
    return point1_->getPosition();
}

//----------------------------------------------------------------------
Vec3 coVRLine::getDirection()
{
    return direction_->getDirection();
}

//----------------------------------------------------------------------
bool coVRLine::isVisible()
{
    return isVisible_;
}

//----------------------------------------------------------------------
string coVRLine::getName()
{
    return name_;
}

//----------------------------------------------------------------------
bool coVRLine::isChanged()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRLine::isChanged\n");

    if (isChanged_)
    {
        isChanged_ = false;
        return true;
    }

    return isChanged_;
}

//----------------------------------------------------------------------
void coVRLine::makeColor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRLine::makeColor\n");

    material_ = new Material();
    stateSet_ = lineDraw_->getOrCreateStateSet();
    stateSet_->setAttributeAndModes(material_);
    lineDraw_->setStateSet(stateSet_);
}

//----------------------------------------------------------------------
string coVRLine::computeNameTagText()
{
    ostringstream textStream;
    string text = name_;
    textStream << " x = (";
    textStream << MathematicPlugin::round10(point1_->x());
    textStream << ", ";
    textStream << MathematicPlugin::round10(point1_->y());
    textStream << ", ";
    textStream << MathematicPlugin::round10(point1_->z());
    textStream << ") + r (";
    textStream << MathematicPlugin::round10(direction_->x());
    textStream << ", ";
    textStream << MathematicPlugin::round10(direction_->y());
    textStream << ", ";
    textStream << MathematicPlugin::round10(direction_->z());
    textStream << ")";
    text.append(textStream.str());

    return text;
}

//----------------------------------------------------------------------
void coVRLine::hideLabel(bool hide)
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
        if (point2_->isVisible())
            point2_->hideLabel(hide);
        if (direction_->isVisible())
            direction_->hideLabel(hide);
    }
}
