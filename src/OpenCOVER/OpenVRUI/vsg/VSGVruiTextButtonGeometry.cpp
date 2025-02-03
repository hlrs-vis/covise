/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiTextButtonGeometry.h>
#include <OpenVRUI/vsg/VSGVruiTransformNode.h>
#include <OpenVRUI/vsg/VSGVruiPresets.h>

#include <vsg/all.h>

#define STYLE_IN 1
#define STYLE_OUT 2
#define DETAIL_LEVEL 40 // number of triangles

using namespace std;
using namespace vsg;

namespace vrui
{

VSGVruiTextButtonGeometry::VSGVruiTextButtonGeometry(coTextButtonGeometry *geometry)
    : vruiButtonProvider(geometry)
    , myDCS(0)
{
    textString = geometry->getTextureName();
}

VSGVruiTextButtonGeometry::~VSGVruiTextButtonGeometry()
{
}

void VSGVruiTextButtonGeometry::createGeometry()
{

    if (normalNode.get() == nullptr)
    {

        color1 = vsg::vec4(((coTextButtonGeometry *)element)->c1r, ((coTextButtonGeometry *)element)->c1g, ((coTextButtonGeometry *)element)->c1b, ((coTextButtonGeometry *)element)->c1a);
        color2 = vsg::vec4(((coTextButtonGeometry *)element)->c2r, ((coTextButtonGeometry *)element)->c2g, ((coTextButtonGeometry *)element)->c2b, ((coTextButtonGeometry *)element)->c2a);

        normalNode = createNode(false, false);
        pressedNode = createNode(true, false);
        highlightNode = normalNode.get();
        pressedHighlightNode = pressedNode.get();

        ref_ptr<MatrixTransform> transformNode =  MatrixTransform::create();
        switchNode = Switch::create();

        switchNode->addChild(true,normalNode);
        switchNode->addChild(false,pressedNode);
        switchNode->addChild(false, highlightNode);
        switchNode->addChild(false, pressedHighlightNode);

        transformNode->addChild(switchNode);

        myDCS = new VSGVruiTransformNode(transformNode);
    }
}

ref_ptr<Node> VSGVruiTextButtonGeometry::createNode(bool pressed, bool)
{
    /*
    ref_ptr<vsg::MatrixTransform> nodemt = new vsg::MatrixTransform();

    ref_ptr<vsg::Geometry> textnode = new vsg::Geometry();
    ref_ptr<vsg::Geometry> bgnode = new vsg::Geometry();
    ref_ptr<vsg::MatrixTransform> textmt = new vsg::MatrixTransform();

    ref_ptr<vsg::Geometry> bggeo = new vsg::Geometry();

    float bwidth = this->getWidth();
    float bheight = this->getHeight();

    ref_ptr<Text> text = new Text();

    text->setText(textString, String::ENCODING_UTF8);
    text->setFont(VSGVruiPresets::getFontFile());
    text->setAlignment(vsg::Text::CENTER_CENTER);
    text->setCharacterSize(20.0);
    text->setDrawMode(Text::TEXT);
    text->setFontResolution(32, 32);
    text->setPosition(vsg::vec3(0, 0, 0));
    text->setAxisAlignment(vsg::Text::XY_PLANE);
    if (pressed)
    {
        text->setColor(color1);
        //text->setColor(vsg::vec4(0.9, 0.9, 0.9, 1.0));
    }
    else
    {
        text->setColor(color2);
        //text->setColor(vsg::vec4(0.1, 0.1, 0.1, 1.0));
    }

    textnode->addDrawable(text.get());

    ref_ptr<vsg::vec3Array> bgVerts = new vsg::vec3Array();

    bgVerts->push_back(vsg::vec3(0, 0, 0));
    bgVerts->push_back(vsg::vec3(bwidth, 0, 0));
    bgVerts->push_back(vsg::vec3(bwidth, bheight, 0));
    bgVerts->push_back(vsg::vec3(0, bheight, 0));

    bggeo->setVertexArray(bgVerts.get());

    ref_ptr<vsg::DrawElementsUInt> bge = new vsg::DrawElementsUInt(vsg::PrimitiveSet::QUADS, 0);
    bge->push_back(0);
    bge->push_back(1);
    bge->push_back(2);
    bge->push_back(3);

    bggeo->addPrimitiveSet(bge.get());

    ref_ptr<vsg::vec4Array> colors = new vsg::vec4Array;
    for (int i = 0; i < 4; ++i)
    {
        if (pressed)
        {
            //colors->push_back( vsg::vec4(0.1, 0.1, 0.1, 1.0) );
            colors->push_back(color2);
        }
        else
        {
            colors->push_back(color1);
            //colors->push_back( vsg::vec4(0.9, 0.9, 0.9, 1.0) );
        }
    }

    bggeo->setColorArray(colors.get());
    bggeo->setColorBinding(vsg::Geometry::BIND_PER_VERTEX);

    bgnode->addDrawable(bggeo.get());

    vsg::StateGroup *stateset = textnode->getOrCreateStateSet();

    stateset->setMode(GL_BLEND, vsg::StateAttribute::ON);
    VSGVruiPresets::makeTransparent(stateset);
    stateset->setMode(GL_LIGHTING, vsg::StateAttribute::OFF);
    stateset->setMode(GL_DEPTH_TEST, vsg::StateAttribute::OFF);

    stateset = bgnode->getOrCreateStateSet();

    stateset->setMode(GL_BLEND, vsg::StateAttribute::ON);
    VSGVruiPresets::makeTransparent(stateset);
    stateset->setMode(GL_LIGHTING, vsg::StateAttribute::OFF);

    nodemt->addChild(bgnode.get());
    nodemt->addChild(textmt.get());
    textmt->addChild(textnode.get());

    Matrix mtrans, mscale;
    mtrans.makeTranslate(Vec3(bwidth / 2.0f, bheight / 2.0, 1.0));

    float scale;
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
    float twidth = fabs(text->computeBoundingBox().xMax() - text->computeBoundingBox().xMin());
    float theight = fabs(text->computeBoundingBox().yMax() - text->computeBoundingBox().yMin());
#else
    float twidth = fabs(text->computeBound().xMax() - text->computeBound().xMin());
    float theight = fabs(text->computeBound().yMax() - text->computeBound().yMin());
#endif

    if (theight / bheight > twidth / bwidth)
    {
        scale = (bheight * 0.9) / theight;
    }
    else
    {
        scale = (bwidth * 0.9) / twidth;
    }

    mscale.makeScale(Vec3(scale, scale, 1.0));

    textmt->setMatrix(mscale * mtrans);

    mtrans.makeTranslate(Vec3(0.0, 0.0, 1.0));

    nodemt->setMatrix(mtrans);

    return nodemt.get();*/
vsg::ref_ptr<vsg::MatrixTransform> node = MatrixTransform::create();
return node;
}

void VSGVruiTextButtonGeometry::switchGeometry(coButtonGeometry::ActiveGeometry active)
{

    createGeometry();
    switchNode->setSingleChildOn(active);
}

void VSGVruiTextButtonGeometry::resizeGeometry()
{
}

vruiTransformNode *VSGVruiTextButtonGeometry::getDCS()
{
    createGeometry();
    return myDCS;
}

float VSGVruiTextButtonGeometry::getWidth() const
{
    return ((coTextButtonGeometry *)element)->getInnerWidth();
}

float VSGVruiTextButtonGeometry::getHeight() const
{
    return ((coTextButtonGeometry *)element)->getInnerHeight();
}
}
