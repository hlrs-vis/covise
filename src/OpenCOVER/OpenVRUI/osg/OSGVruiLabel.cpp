/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/osg/OSGVruiLabel.h>

#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coUIContainer.h>

#include <OpenVRUI/osg/OSGVruiTransformNode.h>
#include <OpenVRUI/osg/OSGVruiPresets.h>

#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osg/Version>

#include <config/CoviseConfig.h>

#include <string>

using namespace std;
using namespace osg;
using namespace osgText;

namespace vrui
{

using covise::coCoviseConfig;

#define BORDERWIDTH 5.0

/// Constructor.
OSGVruiLabel::OSGVruiLabel(coLabel *label)
    : OSGVruiUIElement(label)
{
    this->label = label;

    textColor = osg::Vec4(coCoviseConfig::getFloat("r", "COVER.VRUI.TextColorDefault", 0.9f),
                          coCoviseConfig::getFloat("g", "COVER.VRUI.TextColorDefault", 0.9f),
                          coCoviseConfig::getFloat("b", "COVER.VRUI.TextColorDefault", 0.9f), 1.0f);
    textColorHL = osg::Vec4(coCoviseConfig::getFloat("r", "COVER.VRUI.TextColorHighlighted", 0.0f),
                            coCoviseConfig::getFloat("g", "COVER.VRUI.TextColorHighlighted", 1.0f),
                            coCoviseConfig::getFloat("b", "COVER.VRUI.TextColorHighlighted", 0.0f), 1.0f);

    labelText = 0;
    backgroundTextureState = 0;
}

/// Destructor.
OSGVruiLabel::~OSGVruiLabel()
{
}

float OSGVruiLabel::getWidth() const
{

    float rv = 0.0f;

    if (labelText.valid())
    {
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
        BoundingBox bound = labelText->getBoundingBox();
#else
        BoundingBox bound = labelText->getBound();
#endif
        rv = bound.xMax() - bound.xMin();
    }

    //VRUILOG("OSGVruiLabel::getWidth info: width is " << rv)
    return rv;
}

float OSGVruiLabel::getHeight() const
{

    float rv = 0.0f;

    if (labelText.valid())
    {
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
        BoundingBox bound = labelText->getBoundingBox();
#else
        BoundingBox bound = labelText->getBound();
#endif
        rv = bound.yMax() - bound.yMin();
    }

    //VRUILOG("OSGVruiLabel::getHeight info: height is " << rv)
    return rv;
}

float OSGVruiLabel::getDepth() const
{

    float rv = 0.0f;

    if (labelText.valid())
    {
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
        BoundingBox bound = labelText->getBoundingBox();
#else
        BoundingBox bound = labelText->getBound();
#endif
        rv = bound.zMax() - bound.zMin();
    }

    //VRUILOG("OSGVruiLabel::getDepth info: depth is " << rv)
    return rv;
}

void OSGVruiLabel::createGeometry()
{

    if (myDCS)
        return;

    ref_ptr<MatrixTransform> transform = new MatrixTransform();

    myDCS = new OSGVruiTransformNode(transform.get());

    labelText = new Text();
    labelText->setFont(OSGVruiPresets::getFontFile());
    labelText->setDrawMode(Text::TEXT);
    labelText->setColor(textColor);

    Quat rot;
    rot.makeRotate(-90.0f, 1.0f, 0.0f, 0.0f);
    //labelText->setRotation(rot);

    ref_ptr<Geode> textNode = new Geode();

    makeText();

    textNode->setStateSet(OSGVruiPresets::getStateSetCulled(coUIElement::WHITE_NL));
    textNode->addDrawable(labelText.get());

    transform->addChild(textNode.get());
}

/// Private method to generate text string and attach it to a node.
void OSGVruiLabel::makeText()
{

    if (label->getString() == 0)
        return;

    Text::AlignmentType align = Text::LEFT_BOTTOM_BASE_LINE;
    switch (label->getJustify())
    {
    case coLabel::LEFT:
        align = Text::LEFT_BOTTOM_BASE_LINE;
        break;
    case coLabel::CENTER:
        align = Text::CENTER_BOTTOM_BASE_LINE;
        break;
    case coLabel::RIGHT:
        align = Text::RIGHT_BOTTOM_BASE_LINE;
        break;
    }

    Text::Layout direction = Text::LEFT_TO_RIGHT;
    switch (label->getDirection())
    {
    case coLabel::HORIZONTAL:
        direction = Text::LEFT_TO_RIGHT;
        break;
    case coLabel::VERTICAL:
        direction = Text::VERTICAL;
        break;
    }

    labelText->setAlignment(align);
    labelText->setCharacterSize(label->getFontSize());
    labelText->setText(label->getString(), String::ENCODING_UTF8);
    labelText->setLayout(direction);
    labelText->setAxisAlignment(Text::XY_PLANE);

    labelText->setSupportsDisplayList(false);
    labelText->setUseVertexBufferObjects(true);
    //labelText->setUseVertexArrayObject(true);
    

    labelText->dirtyDisplayList();
}

void OSGVruiLabel::setHighlighted(bool hl)
{
    if (hl)
    {
        labelText->setColor(textColorHL);
    }
    else
    {
        labelText->setColor(textColor);
    }
}

void OSGVruiLabel::resizeGeometry()
{
    createGeometry();
    makeText();
}

void OSGVruiLabel::update()
{
    createGeometry();
    makeText();
}
}
