/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiDefaultButtonGeometry.h>
#include <OpenVRUI/vsg/VSGVruiTransformNode.h>
#include <OpenVRUI/vsg/VSGVruiPresets.h>

#include <vsg/nodes/MatrixTransform.h>
#include <vsg/nodes/Switch.h>
#include <vsg/utils/Builder.h>


#define STYLE_IN 1
#define STYLE_OUT 2
#define DETAIL_LEVEL 40 // number of triangles

using namespace std;
using namespace vsg;

namespace vrui
{

VSGVruiDefaultButtonGeometry::VSGVruiDefaultButtonGeometry(coDefaultButtonGeometry *geometry)
    : vruiButtonProvider(geometry)
{
    textString = geometry->getTextureName();
}

VSGVruiDefaultButtonGeometry::~VSGVruiDefaultButtonGeometry()
{
}

void VSGVruiDefaultButtonGeometry::createGeometry()
{
    
    if (!normalNode.get())
    {

        normalNode = createNode(false, false);
        pressedNode = createNode(true, false);
        highlightNode = createNode(false, true);
        pressedHighlightNode = createNode(true, true);
        disabledNode = createNode(false, false, true);

        vsg::ref_ptr<MatrixTransform> transformNode = vsg::MatrixTransform::create();
        switchNode = new vsg::Switch();

        switchNode->addChild(true,normalNode);
        switchNode->addChild(false,pressedNode);
        switchNode->addChild(false, highlightNode);
        switchNode->addChild(false, pressedHighlightNode);
        switchNode->addChild(false, disabledNode);

        transformNode->addChild(switchNode);

        myDCS = new VSGVruiTransformNode(transformNode);
    }
}


ref_ptr<Text> VSGVruiDefaultButtonGeometry::createText(const string &textString,
                                                        vsg::StandardLayout::Alignment align,
                                                       float size)
{

    vsg::ref_ptr<vsg::Text> text = vsg::Text::create();

    text->text = vsg::stringValue::create(textString);
    text->font = VSGVruiPresets::instance()->font;

    auto layout = vsg::StandardLayout::create();
    layout->horizontalAlignment = align;
    layout->position = vsg::vec3(0.0, 0.0, 0.0);
    layout->horizontal = vsg::vec3(1.0, 0.0, 0.0);
    layout->vertical = vsg::vec3(0.0, 0.0, 1.0);
    layout->color = vsg::vec4(1.0, 1.0, 1.0, 1.0);
    text->layout = layout;
    //text->setCharacterSize(size);
    text->setup(0, VSGVruiPresets::instance()->options);

    return text;
}

ref_ptr<Node> VSGVruiDefaultButtonGeometry::createNode(bool pressed, bool highlighted, bool disabled)
{
    auto builder = vsg::Builder::create();
    //builder->options = VSGVruiPresets::instance()->options;

    vsg::GeometryInfo geomInfo;
    geomInfo.dx.set(1.0f, 0.0f, 0.0f);
    geomInfo.dy.set(0.0f, 1.0f, 0.0f);
    geomInfo.dz.set(0.0f, 0.0f, 1.0f);

    vsg::StateInfo stateInfo;

    ref_ptr<Text> textNode;
    ref_ptr<MatrixTransform> transform;
    ref_ptr<MatrixTransform> buttonTransform;
    ref_ptr<MatrixTransform> textTransform;




    transform = MatrixTransform::create();
    buttonTransform = MatrixTransform::create();
    buttonTransform->addChild(builder->createCylinder(geomInfo, stateInfo));
    dmat4 matrix;
    dmat4 scaleMatrix;
    if (pressed)
    {
        matrix = translate(0.0, 0.0, 2.0);
        scaleMatrix = scale(10.0, 5.0, 2.0);
        matrix = scaleMatrix * matrix;
    }
    else
    {
        matrix = translate(0.0, 0.0, 5.0);
        scaleMatrix = scale(10.0, 5.0, 2.0);
        matrix = scaleMatrix * matrix;
    }
    buttonTransform->matrix = matrix;


    transform->addChild(buttonTransform);

    textTransform = new MatrixTransform();
    matrix = translate(0.0f, 0.0f, 3.0f);
    matrix = rotate(vsg::radians(270.0f), vec3(1.0f, 0.0f, 0.0f));
    textTransform->matrix = matrix;
    transform->addChild(textTransform);

    textNode = createText(textString, StandardLayout::CENTER_ALIGNMENT, 8.0f);
    textTransform->addChild(textNode);

    return transform;
}

void VSGVruiDefaultButtonGeometry::switchGeometry(coButtonGeometry::ActiveGeometry active)
{

    createGeometry();
    switchNode->setSingleChildOn(active);
}

void VSGVruiDefaultButtonGeometry::resizeGeometry()
{
}

vruiTransformNode *VSGVruiDefaultButtonGeometry::getDCS()
{
    createGeometry();
    return myDCS;
}

float VSGVruiDefaultButtonGeometry::getWidth() const
{
    return 10.0f;
}

float VSGVruiDefaultButtonGeometry::getHeight() const
{
    return 5.0f;
}
}
