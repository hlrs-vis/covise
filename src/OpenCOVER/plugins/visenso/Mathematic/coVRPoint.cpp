/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2009 Visenso  **
 **                                                                        **
 ** Description: coVRPoint                                                 **
 **              Draws a point with a sphere interactor, a nameTag,        **
 **               an base vector and an axis projection                    **
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
#include <osg/Vec4>

#include <cover/coVRPluginSupport.h>

#include <cover/coTranslator.h>

#include "MathematicPlugin.h"
#include "coVRPoint.h"

static double interSize_ = 90.0;

using namespace osg;
using covise::coCoviseConfig;

int coVRPoint::_pointID_ = 0;
BoundingBox *coVRPoint::_boundingBox_ = NULL;

//
// Constructor
//
coVRPoint::coVRPoint(Vec3 position, string name, bool normal)
    : name_(name)
    , node_(NULL)
    , baseVectorNode_(NULL)
    , position_(position)
    , oldPosition_(position)
    , basePointInteractor_(NULL)
    , nameTag_(NULL)
    , baseVector_(NULL)
    , projection_(NULL)
    , parentMenu_(NULL)
    , visibleCheckbox_(NULL)
    , sepLabel_(NULL)
    , showVec_(false)
    , showProj_(false)
    , isVisible_(true)
    , isChanged_(false)
    , isRunning_(false)
    , enableInteractor_(true)
    , normal_(normal)
    , inBoundingBox_(true)
    , isBBSet_(false)
    , labelsShown_(false)
{
    if (cover->debugLevel(1))
        fprintf(stderr, "coVRPoint::coVRPoint\n");

    _pointID_++;

    // set boundingbox if not already set
    if (!_boundingBox_)
    {
        fprintf(stderr, "WARNING: coVRPoint no bounding box, will be set to 10\n");
        double boundary = 10.;
        BoundingBox *boundingBox = new BoundingBox(-boundary, -boundary, -boundary,
                                                   boundary, boundary, boundary);
        setBoundingBox(boundingBox);
        isBBSet_ = true;
    }

    inBoundingBox_ = _boundingBox_->contains(position_);

    // make unique name
    ostringstream numStream;
    numStream << _pointID_;
    name_.append(numStream.str());

    // root node
    node_ = new MatrixTransform();
    node_->ref();
    node_->setName(name_);

    // base vector
    makeBaseVector();

    // interactor
    makeInteractor();

    // projection
    projection_ = new coVRAxisProjection(position_);
    projection_->setVisible(false);

    // name tag as pinboard
    makeNameTag();

    // not a normal Point
    // no interactor, no projection, no base vector, not visible
    if (!normal_)
    {
        enableInteractor(false);
        showProjection(false);
        showVector(false);
        setVisible(false);
        // make color orange for not normal points
        //       makeColor();
        //       setColor(Vec4(1.0,0.65,0.0,1.0));
    }

    // add root node to cover scenegraph
    cover->getObjectsRoot()->addChild(node_.get());
}

//
// Destructor
//
coVRPoint::~coVRPoint()
{
    if (cover->debugLevel(1))
        fprintf(stderr, "coVRPoint::~coVRPoint %s\n", getName().c_str());

    cover->getObjectsRoot()->removeChild(node_.get());
    node_->unref();

    baseVectorNode_->unref();
    baseVector_->unref();
    delete basePointInteractor_;
    delete projection_;

    delete nameTag_;
    delete visibleCheckbox_;
    delete sepLabel_;

    if (isBBSet_)
        delete _boundingBox_;
}

//----------------------------------------------------------------------
void coVRPoint::setVisible(bool visible)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPoint::setVisible %d %s\n", visible, name_.c_str());

    isVisible_ = visible;

    if (isVisible_ /*&& inBoundingBox_*/)
    {
        show();
        hideLabel(labelsShown_);
    }
    else
        hide();
}

//----------------------------------------------------------------------
void coVRPoint::setBoundingBox(BoundingBox *boundingBox)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPoint::setBoundingBox\n");

    _boundingBox_ = boundingBox;
}

//----------------------------------------------------------------------
void coVRPoint::update()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPoint::update\n");

    projection_->update(position_);
    basePointInteractor_->updateTransform(position_);
    updateBaseVector();
    updateNameTag();

    //renew node
    node_->dirtyBound();
}

//----------------------------------------------------------------------
void coVRPoint::updateNameTag()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPoint::updateNameTag\n");

    // position
    Matrix o_to_w = cover->getBaseMat();
    Vec3 pos_w;
    pos_w = position_ * o_to_w;
    nameTag_->setPosition(pos_w);

    // text
    string text = computeNameTagText();
    nameTag_->setString(text.c_str());
}

//----------------------------------------------------------------------
void coVRPoint::updateBaseVector()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPoint::updateBaseVector\n");

    Matrix m;
    MatrixTransform *transform = new MatrixTransform();
    double length = position_.length(); /// MINUS SPHERE RADIUS

    baseVector_->drawArrow(Vec3(0.0, 0.0, 0.0), 0.05, length);

    m.makeRotate(Vec3(0.0, 0.0, length), position_);
    transform->setMatrix(m);
    transform->addChild(baseVector_.get());
    baseVectorNode_->addChild(transform);

    // cant remove first child of MatrixTransform
    //  so it has to be removed after second is added
    if (baseVectorNode_->getNumChildren() > 1)
        baseVectorNode_->removeChild(baseVectorNode_->getChild(0));

    if (node_->getNumChildren() > 0)
        node_->removeChild(baseVectorNode_.get());
    node_->addChild(baseVectorNode_.get());
}

//----------------------------------------------------------------------
int coVRPoint::addToMenu(coRowMenu *parentMenu, int position, bool sep)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPoint::addToMenu %d sep=%d\n", position, sep);

    parentMenu_ = parentMenu;

    // making visibleCheckbox
    if (!visibleCheckbox_)
    {
        string visibleText = name_;
        visibleText.append(coTranslator::coTranslate(": zeigen"));
        visibleCheckbox_ = new coCheckboxMenuItem(visibleText.c_str(), true);
        visibleCheckbox_->setMenuListener(MathematicPlugin::plugin);
    }
    parentMenu_->insert(visibleCheckbox_, ++position);

    // separator
    if (sep)
    {
        if (!sepLabel_)
            sepLabel_ = new coLabelMenuItem("______________");
        parentMenu_->insert(sepLabel_, ++position);
    }

    return position;
}

//----------------------------------------------------------------------
void coVRPoint::removeFromMenu()
{
    if (visibleCheckbox_)
        parentMenu_->remove(visibleCheckbox_);

    // separator
    if (sepLabel_)
        parentMenu_->remove(sepLabel_);
}

//----------------------------------------------------------------------
void coVRPoint::makeNameTag()
{
    Vec4 fgColor(0.5451, 0.7020, 0.2431, 1.0);
    Vec4 bgColor(0.0, 0.0, 0.0, 0.8);
    double linelen = 0.04 * cover->getSceneSize();
    double fontsize = 20;

    string text = computeNameTagText();
    nameTag_ = new coVRLabel(text.c_str(), fontsize, linelen, fgColor, bgColor);

    // set to default position
    nameTag_->setPosition(position_);

    if (!inBoundingBox_)
        nameTag_->hide();
    else
        nameTag_->show();

    updateNameTag();
}

//----------------------------------------------------------------------
void coVRPoint::makeBaseVector()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPoint::makeBaseVector\n");

    baseVectorNode_ = new MatrixTransform();
    baseVectorNode_->ref();
    baseVector_ = new coArrow(0.0, 0.0, false, false);
    baseVector_->ref();
    updateBaseVector();
    baseVector_->setVisible(false);
}

//----------------------------------------------------------------------
void coVRPoint::makeInteractor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPoint::makeInteractor\n");

    // default size for all interactors
    interSize_ = cover->getSceneSize() / 50;
    // if defined, COVER.IconSize overrides the default
    interSize_ = (double)coCoviseConfig::getFloat("COVER.IconSize", interSize_);
    // if defined, COVERConfig Mathematic IconSize overrides both
    interSize_ = (double)coCoviseConfig::getFloat("COVER.Plugin.Mathematic.IconSize", interSize_);

    if (!normal_)
    {
        // if not normal make interactor size smaller
        interSize_ *= 0.7;
    }

    basePointInteractor_ = new coVR3DTransInteractor(position_, interSize_, coInteraction::ButtonA, "hand", "Point", coInteraction::Medium);

    if (inBoundingBox_)
    {
        basePointInteractor_->show();
        basePointInteractor_->enableIntersection();
    }
    else
    {
        basePointInteractor_->hide();
        basePointInteractor_->disableIntersection();
    }
}

//----------------------------------------------------------------------
void coVRPoint::preFrame()
{
    //fprintf(stderr,"coVRPoint::preFrame\n");

    inBoundingBox_ = _boundingBox_->contains(position_);

    if (isVisible_ /*&& inBoundingBox_*/)
    {
        basePointInteractor_->preFrame();

        if (basePointInteractor_->isRunning())
        {
            // update position only if in boundingBox
            Vec3 interPos = MathematicPlugin::roundVec2(basePointInteractor_->getPos());
            if (_boundingBox_->contains(interPos) && interPos != position_)
            {
                oldPosition_ = position_;

                //TODO FEHLER im Trunk: nicht in 0.0.0 legen
                if (interPos != Vec3(0.0, 0.0, 0.0))
                    position_ = interPos;
                else
                    position_.set(0.0, 0.0, 0.00001);
                basePointInteractor_->updateTransform(position_);

                projection_->setVisible(true);
                baseVector_->setVisible(true);
                //fprintf(stderr,"coVRPoint::preFrame position (%f %f %f)\n",position_.x(), position_.y(),position_.z());
                isChanged_ = true;
                isRunning_ = true;
                update();
            } // set interactor back
            else
                basePointInteractor_->updateTransform(position_);
        }
        else
        {
            isRunning_ = false;
            projection_->setVisible(false);
            baseVector_->setVisible(false);
            // cover BaseMat changes per frame
            // take care of nameTag changes
            updateNameTag();
        }
    }
}

//----------------------------------------------------------------------
void coVRPoint::setPosition(Vec3 position)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPoint::setPosition (%f %f %f)\n", position.x(), position.y(), position.z());

    position_ = position;

    //inBoundingBox_ = _boundingBox_->contains(position_);

    //if( isVisible_ && inBoundingBox_ )
    //   show();
    //else
    //   hide();

    update();
}

//----------------------------------------------------------------------
void coVRPoint::show()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPoint::show %s\n", name_.c_str());

    // root node
    node_->setNodeMask(node_->getNodeMask() | (Isect::Visible));

    // interactor
    basePointInteractor_->show();
    if (enableInteractor_)
        basePointInteractor_->enableIntersection();

    // nametag
    nameTag_->show();
    updateNameTag();

    // show vector only if wanted
    if (showVec_)
        baseVector_->setVisible(true);

    // show projection only if wanted
    if (showProj_)
        projection_->setVisible(true);
}

//----------------------------------------------------------------------
void coVRPoint::hide()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPoint::hide\n");

    // root node
    node_->setNodeMask(node_->getNodeMask() & (~Isect::Visible));

    // interactor
    basePointInteractor_->hide();
    if (enableInteractor_)
        basePointInteractor_->disableIntersection();

    // nameTag
    nameTag_->hide();

    baseVector_->setVisible(false);
    projection_->setVisible(false);
}

//----------------------------------------------------------------------
Vec3 coVRPoint::getPosition()
{
    return position_;
}

//----------------------------------------------------------------------
double coVRPoint::x()
{
    return position_.x();
}

//----------------------------------------------------------------------
double coVRPoint::y()
{
    return position_.y();
}

//----------------------------------------------------------------------
double coVRPoint::z()
{
    return position_.z();
}

//----------------------------------------------------------------------
void coVRPoint::showVector(bool show)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPoint::showVector\n");

    showVec_ = show;

    if (isVisible_)
        baseVector_->setVisible(show);
}

//----------------------------------------------------------------------
void coVRPoint::showProjection(bool show)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPoint::showProjection %d %s\n", show, name_.c_str());

    showProj_ = show;

    if (isVisible_)
        projection_->setVisible(show);
}

//----------------------------------------------------------------------
bool coVRPoint::isChanged()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPoint::isChanged\n");

    if (isChanged_)
    {
        isChanged_ = false;
        return true;
    }

    return isChanged_;
}

//----------------------------------------------------------------------
bool coVRPoint::isVisible()
{
    return isVisible_;
}

//----------------------------------------------------------------------
string coVRPoint::getName()
{
    return name_;
}

//----------------------------------------------------------------------
int coVRPoint::getMenuPosition()
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
void coVRPoint::enableInteractor(bool enable)
{
    enableInteractor_ = enable;

    if (enableInteractor_)
        basePointInteractor_->enableIntersection();
    else
        basePointInteractor_->disableIntersection();
}

//----------------------------------------------------------------------
void coVRPoint::menuEvent(coMenuItem *menuItem)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRPoint::menuEvent for %s\n", menuItem->getName());

    if (visibleCheckbox_ && menuItem == visibleCheckbox_)
        setVisible(visibleCheckbox_->getState());
}

//----------------------------------------------------------------------
string coVRPoint::computeNameTagText()
{
    ostringstream textStream;
    string text = name_;
    if (isRunning_ || !normal_)
    {
        textStream << " (";
        textStream << MathematicPlugin::round10(position_.x());
        textStream << ", ";
        textStream << MathematicPlugin::round10(position_.y());
        textStream << ", ";
        textStream << MathematicPlugin::round10(position_.z());
        textStream << ")";
        text.append(textStream.str());
    }

    return text;
}

//----------------------------------------------------------------------
void coVRPoint::hideLabel(bool hide)
{
    labelsShown_ = hide;
    if (this->isVisible())
    {
        if (hide)
            nameTag_->hide();
        else
            nameTag_->show();
    }
}

// //----------------------------------------------------------------------
// void coVRPoint::setColor(Vec4 color)
// {
//    //fprintf(stderr,"coVRPoint::setColor\n");
//
//    color_ = color;
//    material_->setDiffuse(Material::FRONT_AND_BACK, color_);
//    material_->setAmbient(Material::FRONT_AND_BACK, Vec4(color_.x()*0.3, color_.y()*0.3, color_.z()*0.3, color_.w()));
// }
//
// //----------------------------------------------------------------------
// void coVRPoint::makeColor()
// {
//    //fprintf(stderr,"ccoVRPoint::makeColor\n");
//
//    material_ = new Material();
//    stateSet_ = node_->getOrCreateStateSet();
//    stateSet_->setAttributeAndModes(material_);
//    node_->setStateSet(stateSet_);
// }
