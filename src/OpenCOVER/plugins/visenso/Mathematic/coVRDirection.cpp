/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2009 Visenso  **
 **                                                                        **
 ** Description: coVRDirection                                             **
 **              Draws a direction vector with rotInteractor and a nameTag **
 **               only within the bounding box (needs to be set)           **
 **                                                                        **
 ** Author: M. Theilacker                                                  **
 **                                                                        **
 ** History:                                                               **
 **     4.2009 initial version                                             **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include <osg/Matrix>

#include <cover/coVRPluginSupport.h>

#include <cover/coTranslator.h>

#include "MathematicPlugin.h"
#include "coVRDirection.h"

static double interSize_ = 90.0;

using namespace osg;
using covise::coCoviseConfig;

int coVRDirection::_directionID_ = 0;
BoundingBox *coVRDirection::_boundingBox_ = NULL;

//
// Constructor
//
coVRDirection::coVRDirection(Vec3 direction, Vec3 position, string name, double radius)
    : name_(name)
    , direction_(direction)
    , position_(position)
    , nameTag_(NULL)
    , parentMenu_(NULL)
    , visibleCheckbox_(NULL)
    , isChanged_(false)
    , isRunning_(false)
    , isVisible_(true)
    , isBBSet_(false)
    , labelsShown_(false)
{
    if (cover->debugLevel(1))
        fprintf(stderr, "coVRDirection::coVRDirection direction (%f %f %f) position (%f %f %f)\n", direction_.x(), direction_.y(), direction_.z(), position_.x(), position_.y(), position_.z());

    _directionID_++;

    // set boundingbox if not already set
    if (!_boundingBox_)
    {
        fprintf(stderr, "WARNING: coVRDirection no bounding box, will be set to 10\n");
        double boundary = 10.;
        BoundingBox *boundingBox = new BoundingBox(-boundary, -boundary, -boundary,
                                                   boundary, boundary, boundary);
        setBoundingBox(boundingBox);
        isBBSet_ = true;
    }

    // make unique name
    ostringstream numStream;
    numStream << _directionID_;
    name_.append(numStream.str());

    // root node
    node_ = new MatrixTransform();
    node_->ref();
    node_->setName(name_);

    // direction vector line
    ///direction_.normalize();
    dirLineRadius_ = radius;
    dirLine_ = new Cylinder();
    dirLineDraw_ = new ShapeDrawable();
    dirLineGeode_ = new Geode();
    dirLineDraw_->setShape(dirLine_);
    dirLineGeode_->addDrawable(dirLineDraw_);
    dirLineGeode_->setNodeMask(dirLineGeode_->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));
    dirLineDraw_->setUseDisplayList(false);
    node_->addChild(dirLineGeode_);
    updateDirectionVector();

    // direction interactor
    makeDirectionInteractor();

    // name tag as pinboard
    makeNameTag();
    updateNameTag();

    // add root node to cover scenegraph
    cover->getObjectsRoot()->addChild(node_.get());
}

//
// Destructor
//
coVRDirection::~coVRDirection()
{
    if (cover->debugLevel(1))
        fprintf(stderr, "coVRDirection::~coVRDirection\n");

    cover->getObjectsRoot()->removeChild(node_.get());
    node_->unref();

    delete directionInteractor_;
    delete nameTag_;
    delete visibleCheckbox_;

    if (isBBSet_)
        delete _boundingBox_;
}

//----------------------------------------------------------------------
void coVRDirection::updateDirectionVector()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDirection::updateDirectionVector\n");

    Matrix m;
    double length = direction_.length(); /// MINUS SPHERE RADIUS

    Vec3 lineCenter = (position_ + (position_ + direction_)) * 0.5;
    double lineLength = (position_ - (position_ + direction_)).length();

    dirLine_->set(lineCenter, dirLineRadius_, lineLength);
    m.makeRotate(Vec3(0.0, 0.0, length), direction_);
    dirLine_->setRotation(m.getRotate());

    // renew node
    node_->dirtyBound();
}

//----------------------------------------------------------------------
void coVRDirection::makeDirectionInteractor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDirection::makeDirectionInteractor\n");

    //default size for all interactors
    interSize_ = cover->getSceneSize() / 50;
    //if defined, COVER.IconSize overrides the default
    interSize_ = (double)coCoviseConfig::getFloat("COVER.IconSize", interSize_);
    //if defined, COVERConfig Mathematic IconSize overrides both
    interSize_ = (double)coCoviseConfig::getFloat("COVER.Plugin.Mathematic.IconSize", interSize_);

    directionInteractor_ = new coVR3DRotInteractor(position_, position_ + direction_, interSize_, coInteraction::ButtonA, "hand", "directionVector", coInteraction::Medium);
    directionInteractor_->show();
    directionInteractor_->enableIntersection();
}

//----------------------------------------------------------------------
void coVRDirection::setVisible(bool visible)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDirection::setVisible %d\n", visible);

    isVisible_ = visible;

    if (isVisible_)
    {
        // node
        node_->setNodeMask(node_->getNodeMask() | (Isect::Visible));

        // interactor
        directionInteractor_->show();
        directionInteractor_->enableIntersection();

        // nametag
        nameTag_->show();
        updateNameTag();
        hideLabel(labelsShown_);
    }
    else
    {
        // node
        node_->setNodeMask(node_->getNodeMask() & (~Isect::Visible));

        // interactor
        directionInteractor_->hide();
        directionInteractor_->disableIntersection();

        // nameTag
        nameTag_->hide();
    }
}

//----------------------------------------------------------------------
void coVRDirection::setBoundingBox(BoundingBox *boundingBox)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDirection::setBoundingBox\n");

    _boundingBox_ = boundingBox;
}

//----------------------------------------------------------------------
void coVRDirection::update()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDirection::update\n");

    updateDirectionVector();
    directionInteractor_->updateTransform(position_, position_ + direction_);

    updateNameTag();
}

//----------------------------------------------------------------------
void coVRDirection::updateNameTag()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDirection::updateNameTag\n");

    // position
    Matrix o_to_w = cover->getBaseMat();
    Vec3 pos_w;
    pos_w = (position_ + direction_) * o_to_w;
    nameTag_->setPosition(pos_w);

    // text
    string text = computeNameTagText();
    nameTag_->setString(text.c_str());
}

//----------------------------------------------------------------------
void coVRDirection::preFrame()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDirection::preFrame\n");

    if (isVisible_)
    {
        directionInteractor_->preFrame();

        if (directionInteractor_->isRunning())
        {
            // update position only if in boundingBox, not null vector and new position
            Vec3 interPos = MathematicPlugin::roundVec2(directionInteractor_->getPos());
            Vec3 dir = interPos - position_;
            if (_boundingBox_->contains(interPos)
                && dir.length() > 0
                && dir != direction_)
            {
                directionInteractor_->updateTransform(position_, interPos);
                oldDirection_ = direction_;
                direction_ = dir;
                isChanged_ = true;
                isRunning_ = true;
                update();
            } // set interactor back
            else
                directionInteractor_->updateTransform(position_, position_ + direction_);

            ///direction_.normalize();
            //fprintf(stderr,"coVRDirection::preFrame direction (%f %f %f) position (%f %f %f)\n", direction_.x(), direction_.y(), direction_.z(),position_.x(), position_.y(),position_.z());
        }
        else
        {
            isRunning_ = false;
            // cover BaseMat changes per frame
            // take care of nameTag changes
            updateNameTag();
        }
    }
}

//----------------------------------------------------------------------
void coVRDirection::makeNameTag()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDirection::makeNameTag\n");

    Vec4 fgColor(0.5451, 0.7020, 0.2431, 1.0);
    Vec4 bgColor(0.0, 0.0, 0.0, 0.8);
    double linelen = 0.04 * cover->getSceneSize();
    double fontsize = 20;

    string text = computeNameTagText();
    nameTag_ = new coVRLabel(text.c_str(), fontsize, linelen, fgColor, bgColor);

    // set to default position
    nameTag_->setPosition(position_ + direction_);
}

//----------------------------------------------------------------------
int coVRDirection::addToMenu(coRowMenu *parentMenu, int position)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDirection::addToMenu %d\n", position);

    parentMenu_ = parentMenu;

    // making visibleCheckbox
    if (!visibleCheckbox_)
    {
        string visibleText = coTranslator::coTranslate(name_);
        visibleText.append(coTranslator::coTranslate(": zeigen"));
        visibleCheckbox_ = new coCheckboxMenuItem(visibleText.c_str(), true);
        visibleCheckbox_->setMenuListener(MathematicPlugin::plugin);
    }
    parentMenu_->insert(visibleCheckbox_, ++position);

    return position;
}

//----------------------------------------------------------------------
void coVRDirection::removeFromMenu()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDirection::removeFromMenu\n");

    if (visibleCheckbox_)
        parentMenu_->remove(visibleCheckbox_);
}

//----------------------------------------------------------------------
int coVRDirection::getMenuPosition()
{
    if (visibleCheckbox_)
    {
        coMenuItemVector items = parentMenu_->getAllItems();

        for (int i = 0; i < parentMenu_->getItemCount(); i++)
        {
            // check for last menuitem
            if (items[i] == visibleCheckbox_)
                return i;
        }
    }

    return 0;
}

//----------------------------------------------------------------------
void coVRDirection::menuEvent(coMenuItem *menuItem)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDirection::menuEvent for %s\n", menuItem->getName());

    if (visibleCheckbox_ && menuItem == visibleCheckbox_)
        setVisible(visibleCheckbox_->getState());
}

//----------------------------------------------------------------------
void coVRDirection::setDirection(Vec3 direction)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDirection::setPosition\n");

    direction_ = direction;
    ///direction_.normalize();
    update();
}

//----------------------------------------------------------------------
void coVRDirection::setPosition(Vec3 position)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDirection::setPosition\n");

    position_ = position;
    update();
}

//----------------------------------------------------------------------
Vec3 coVRDirection::getPosition()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDirection::getPosition\n");

    return position_;
}

//----------------------------------------------------------------------
Vec3 coVRDirection::getDirection()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDirection::getDirection\n");

    return direction_;
}

//----------------------------------------------------------------------
string coVRDirection::getName()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDirection::getName\n");

    return name_;
}

//----------------------------------------------------------------------
bool coVRDirection::isChanged()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRDirection::isChanged\n");

    if (isChanged_)
    {
        isChanged_ = false;
        return true;
    }

    return isChanged_;
}

//----------------------------------------------------------------------
bool coVRDirection::isVisible()
{
    return isVisible_;
}

//----------------------------------------------------------------------
double coVRDirection::x()
{
    return direction_.x();
}

//----------------------------------------------------------------------
double coVRDirection::y()
{
    return direction_.y();
}

//----------------------------------------------------------------------
double coVRDirection::z()
{
    return direction_.z();
}

//----------------------------------------------------------------------
string coVRDirection::computeNameTagText()
{
    ostringstream textStream;
    string text = name_;
    if (isRunning_)
    {
        textStream << " (";
        textStream << MathematicPlugin::round10(direction_.x());
        textStream << ", ";
        textStream << MathematicPlugin::round10(direction_.y());
        textStream << ", ";
        textStream << MathematicPlugin::round10(direction_.z());
        textStream << ")";
        text.append(textStream.str());
    }

    return text;
}

void coVRDirection::hideLabel(bool hide)
{
    labelsShown_ = hide;
    if (isVisible())
    {
        if (hide)
            nameTag_->hide();
        else
            nameTag_->show();
    }
}
