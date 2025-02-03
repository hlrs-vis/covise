/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiLabel.h>

#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coUIContainer.h>

#include <OpenVRUI/vsg/VSGVruiTransformNode.h>
#include <OpenVRUI/vsg/VSGVruiPresets.h>

#include <vsg/nodes/MatrixTransform.h>
#include <vsg/text/Text.h>
#include <vsg/text/StandardLayout.h>


#include <config/CoviseConfig.h>

#include <string>

using namespace std;
using namespace vsg;

namespace vrui
{

using covise::coCoviseConfig;

#define BORDERWIDTH 5.0

/// Constructor.
VSGVruiLabel::VSGVruiLabel(coLabel *label)
    : VSGVruiUIElement(label)
{
    this->label = label;

    textColor = vsg::vec4(coCoviseConfig::getFloat("r", "COVER.VRUI.TextColorDefault", 0.9f),
                          coCoviseConfig::getFloat("g", "COVER.VRUI.TextColorDefault", 0.9f),
                          coCoviseConfig::getFloat("b", "COVER.VRUI.TextColorDefault", 0.9f), 1.0f);
    textColorHL = vsg::vec4(coCoviseConfig::getFloat("r", "COVER.VRUI.TextColorHighlighted", 0.0f),
                            coCoviseConfig::getFloat("g", "COVER.VRUI.TextColorHighlighted", 1.0f),
                            coCoviseConfig::getFloat("b", "COVER.VRUI.TextColorHighlighted", 0.0f), 1.0f);

}

/// Destructor.
VSGVruiLabel::~VSGVruiLabel()
{
}

float VSGVruiLabel::getWidth() const
{

    double rv = 0.0f;

    if (labelText.valid())
    {
        dbox bound = labelText->layout->extents(labelText->text, *(VSGVruiPresets::instance()->font.get()));
        rv = bound.max[0] - bound.min[0];
    }

    //VRUILOG("VSGVruiLabel::getWidth info: width is " << rv)
    return (float)rv;
}

float VSGVruiLabel::getHeight() const
{

    double rv = 0.0f;

    if (labelText.valid())
    {
        dbox bound = labelText->layout->extents(labelText->text, *(VSGVruiPresets::instance()->font.get()));
        rv = bound.max[1] - bound.min[1];
    }

    //VRUILOG("VSGVruiLabel::getHeight info: height is " << rv)
    return (float)rv;
}

float VSGVruiLabel::getDepth() const
{

    double rv = 0.0f;

    if (labelText.valid())
    {
        dbox bound = labelText->layout->extents(labelText->text, *(VSGVruiPresets::instance()->font.get()));
        rv = bound.max[2] - bound.min[2];
    }

    //VRUILOG("VSGVruiLabel::getDepth info: depth is " << rv)
    return (float)rv;
}

void VSGVruiLabel::createGeometry()
{

    if (myDCS)
        return;

    ref_ptr<MatrixTransform> transform = MatrixTransform::create();

    myDCS = new VSGVruiTransformNode(transform);

    labelText = Text::create();
    labelText->font = VSGVruiPresets::instance()->font;


    makeText();

    transform->addChild(labelText);
}

/// Private method to generate text string and attach it to a node.
void VSGVruiLabel::makeText()
{

    if (label->getString() == 0)
        return;

    vsg::StandardLayout::Alignment align = StandardLayout::LEFT_ALIGNMENT;
    switch (label->getJustify())
    {
    case coLabel::LEFT:
        align = StandardLayout::LEFT_ALIGNMENT;
        break;
    case coLabel::CENTER:
        align = StandardLayout::CENTER_ALIGNMENT;
        break;
    case coLabel::RIGHT:
        align = StandardLayout::RIGHT_ALIGNMENT;
        break;
    }

    StandardLayout::GlyphLayout direction = StandardLayout::VERTICAL_LAYOUT;
    switch (label->getDirection())
    {
    case coLabel::HORIZONTAL:
        direction = StandardLayout::LEFT_TO_RIGHT_LAYOUT;
        break;
    case coLabel::VERTICAL:
        direction = StandardLayout::VERTICAL_LAYOUT;
        break;
    }



    auto layout = StandardLayout::create();
    layout->horizontalAlignment = align;
    layout->verticalAlignment = StandardLayout::BASELINE_ALIGNMENT;
    layout->glyphLayout = direction;
    labelText->layout = layout;
    labelText->text = vsg::stringValue::create(label->getString());

    

}

void VSGVruiLabel::setHighlighted(bool hl)
{
   /* if (hl)
    {
        labelText->setColor(textColorHL);
    }
    else
    {
        labelText->setColor(textColor);
    }*/
}

void VSGVruiLabel::resizeGeometry()
{
    createGeometry();
    makeText();
}

void VSGVruiLabel::update()
{
    createGeometry();
    makeText();
}
}
