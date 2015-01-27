/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2009 Visenso  **
 **                                                                        **
 ** Description: coVRDistance                                                 **
 **              Draws a triangle and the angle into it                    **
 **                                                                        **
 ** Author: M. Theilacker                                                  **
 **                                                                        **
 ** History:                                                               **
 **     4.2009 initial version                                             **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include <config/CoviseConfig.h>
#include <cover/coTranslator.h>

#include "MathematicPlugin.h"

#include "coVRDistance.h"

const double EPSILON = 1.0e-2;

using namespace osg;

BoundingBox *coVRDistance::_boundingBox_ = NULL;

//
// Constructor
//
coVRDistance::coVRDistance(Vec3 point1, Vec3 point2, int mode, double radius)
    : node_(NULL)
    , mode_(mode)
    , point1_(NULL)
    , point2_(NULL)
    , nameTag_(NULL)
    , line_(NULL)
    , lineGeode_(NULL)
    , lineDraw_(NULL)
    , lineRadius_(radius)
    , parentMenu_(NULL)
    , visibleCheckbox_(NULL)
    , isVisible_(true)
    , isBBSet_(false)
    , labelsShown_(false)
{
    if (cover->debugLevel(1))
        fprintf(stderr, "coVRDistance::coVRDistance\n");

    // set boundingbox if not already set
    if (!_boundingBox_)
    {
        fprintf(stderr, "WARNING: coVRDistance no bounding box, will be set to 10\n");
        double boundary = 10.;
        BoundingBox *boundingBox = new BoundingBox(-boundary, -boundary, -boundary,
                                                   boundary, boundary, boundary);
        setBoundingBox(boundingBox);
        isBBSet_ = true;
    }

    // root node
    node_ = new MatrixTransform();
    node_->ref();
    //node_->setName("Distance");

    // points
    point1_ = new coVRPoint(point1, "Z", false);
    point2_ = new coVRPoint(point2, "Z", false);

    // if wrong mode, set to POINT_LINE_POINT
    if (mode_ != ONLY_LINE
        && mode_ != POINT_LINE
        && mode_ != POINT_LINE_POINT)
    {
        fprintf(stderr, "ERROR: wrong mode for coVRDistance assuming mode POINT_LINE_POINT \n");
        mode_ = POINT_LINE_POINT;
    }

    // compute line
    line_ = new Cylinder();
    lineDraw_ = new ShapeDrawable();
    lineGeode_ = new Geode();
    lineDraw_->setShape(line_);
    lineGeode_->addDrawable(lineDraw_);
    lineDraw_->setUseDisplayList(false);
    node_->addChild(lineGeode_);
    int error = updateLine();
    if (error)
        return;

    // name tag as pinboard
    makeNameTag();
    updateNameTag();

    // add root node to cover scenegraph
    cover->getObjectsRoot()->addChild(node_.get());

    // dont show at first
    setVisible(false);
}

//
// Destructor
//
coVRDistance::~coVRDistance()
{
    if (cover->debugLevel(1))
        fprintf(stderr, "coVRDistance::~coVRDistance\n");

    if (isBBSet_)
        delete _boundingBox_;

    cover->getObjectsRoot()->removeChild(node_.get());
    node_->unref();

    delete point1_;
    delete point2_;
    delete nameTag_;

    delete visibleCheckbox_;
}

//----------------------------------------------------------------------
void coVRDistance::setVisible(bool visible)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDistance::setVisible %d\n", visible);

    isVisible_ = visible;

    if (isVisible_)
    {
        // root node
        node_->setNodeMask(node_->getNodeMask() | (Isect::Visible));

        // settings for different modes
        if (mode_ == POINT_LINE)
            point1_->setVisible(true);
        else if (mode_ == POINT_LINE_POINT)
        {
            point1_->setVisible(true);
            point2_->setVisible(true);
        }

        // nametag
        nameTag_->show();
        updateNameTag();

        hideLabels(labelsShown_);
    }
    else
    {
        // root node
        node_->setNodeMask(node_->getNodeMask() & (~Isect::Visible));

        // settings for different modes
        if (mode_ == POINT_LINE)
            point1_->setVisible(false);
        else if (mode_ == POINT_LINE_POINT)
        {
            point1_->setVisible(false);
            point2_->setVisible(false);
        }

        // nameTag
        nameTag_->hide();
    }
}

//----------------------------------------------------------------------
void coVRDistance::setBoundingBox(BoundingBox *boundingBox)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDistance::setBoundingBox\n");

    _boundingBox_ = boundingBox;
}

//----------------------------------------------------------------------
void coVRDistance::update(Vec3 point1, Vec3 point2)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDistance::update p1(%f %f %f) p2(%f %f %f)\n", point1.x(), point1.y(), point1.z(), point2.x(), point2.y(), point2.z());

    if (visibleCheckbox_)
    {
        if (visibleCheckbox_->getState())
        {
            // dont show out of bounding box or distance = 0
            if (_boundingBox_->contains(point1_->getPosition())
                && _boundingBox_->contains(point2_->getPosition())
                && (point2 - point1).length() > EPSILON)
                setVisible(true);
            else
            {
                // should not be shown because out of bounding box
                // but state is still visible
                setVisible(false);
                isVisible_ = true;
            }
        }
    }
    visibleCheckbox_->setLabel(computeCheckboxText());

    point1_->setPosition(point1);
    point2_->setPosition(point2);

    updateLine();

    // cover BaseMat changes per frame
    updateNameTag();
}

//----------------------------------------------------------------------
int coVRDistance::updateLine()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDistance::updateLine\n");

    Vec3 center = (point1_->getPosition() + point2_->getPosition()) * 0.5;
    Vec3 direction = point1_->getPosition() - point2_->getPosition();
    double length = direction.length();

    line_->set(center, lineRadius_, length);
    Matrix rotation;
    rotation.makeRotate(Vec3(0.0, 0.0, 1.0), direction);
    line_->setRotation(rotation.getRotate());

    return 0;
}

//----------------------------------------------------------------------
void coVRDistance::updateNameTag()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDistance::updateNameTag\n");

    // position: set to between 1st and 2nd perpendicular
    Matrix o_to_w = cover->getBaseMat();
    Vec3 pos_w;
    pos_w = point1_->getPosition() + (point2_->getPosition() - point1_->getPosition()) / 2.0;
    pos_w = pos_w * o_to_w;
    nameTag_->setPosition(pos_w);

    // text
    string text = computeNameTagText();
    nameTag_->setString(text.c_str());
}

//----------------------------------------------------------------------
void coVRDistance::preFrame()
{
    //fprintf(stderr,"coVRDistance::preFrame\n");

    if (isVisible_ && visibleCheckbox_ && visibleCheckbox_->getState())
    {
        // dont show out of bounding box
        if (_boundingBox_->contains(point1_->getPosition())
            && _boundingBox_->contains(point2_->getPosition()))
        {
            point1_->preFrame();
            point2_->preFrame();

            setVisible(true);
        }
        else
        {
            setVisible(false);
            isVisible_ = true;
        }

        // cover BaseMat changes per frame
        updateNameTag();
    }
}

//----------------------------------------------------------------------
void coVRDistance::makeNameTag()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDistance::makeNameTag\n");

    Vec4 fgColor(0.5451, 0.7020, 0.2431, 1.0);
    Vec4 bgColor(0.0, 0.0, 0.0, 0.8);
    double linelen = 0.04 * cover->getSceneSize();
    double fontsize = 20;

    string text = computeNameTagText();
    nameTag_ = new coVRLabel(text.c_str(), fontsize, linelen, fgColor, bgColor);

    // set to default position
    nameTag_->setPosition(point1_->getPosition());
}

//----------------------------------------------------------------------
int coVRDistance::addToMenu(coRowMenu *parentMenu, int posInMenu)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDistance::addToMenu\n");

    parentMenu_ = parentMenu;

    // making visibleCheckbox
    if (!visibleCheckbox_)
    {
        visibleCheckbox_ = new coCheckboxMenuItem(computeCheckboxText(), false);
        visibleCheckbox_->setMenuListener(MathematicPlugin::plugin);
    }
    parentMenu_->insert(visibleCheckbox_, ++posInMenu);

    return posInMenu;
}

//----------------------------------------------------------------------
void coVRDistance::removeFromMenu()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDistance::removeFromMenu\n");

    if (visibleCheckbox_)
        parentMenu_->remove(visibleCheckbox_);
}

//----------------------------------------------------------------------
void coVRDistance::menuEvent(coMenuItem *menuItem)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDistance::menuEvent for %s\n", menuItem->getName());

    if (visibleCheckbox_ && menuItem == visibleCheckbox_)
    {
        if (_boundingBox_->contains(point1_->getPosition())
            && _boundingBox_->contains(point2_->getPosition())
            && (point1_->getPosition() - point2_->getPosition()).length() > EPSILON)
            setVisible(visibleCheckbox_->getState());
    }
}

//----------------------------------------------------------------------
double coVRDistance::getDistance()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDistance::getDistance %f\n", (point1_->getPosition() - point2_->getPosition()).length());

    return (point1_->getPosition() - point2_->getPosition()).length();
}

//----------------------------------------------------------------------
bool coVRDistance::isVisible()
{
    return isVisible_;
}

//----------------------------------------------------------------------
string coVRDistance::computeNameTagText()
{
    ostringstream textStream;
    string text(coTranslator::coTranslate("Abstand: "));

    textStream << MathematicPlugin::round10(getDistance());
    text.append(textStream.str());

    return text;
}

//----------------------------------------------------------------------
string coVRDistance::computeCheckboxText()
{
    string text = coTranslator::coTranslate("zeige Abstand : ");

    ostringstream txtStream;
    txtStream << MathematicPlugin::round10(getDistance());

    text.append(txtStream.str());

    return text;
}

void coVRDistance::hideLabels(bool hide)
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
    }
}
