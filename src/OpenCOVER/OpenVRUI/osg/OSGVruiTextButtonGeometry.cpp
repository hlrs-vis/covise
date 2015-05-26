/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/osg/OSGVruiTextButtonGeometry.h>
#include <OpenVRUI/osg/OSGVruiTransformNode.h>
#include <OpenVRUI/osg/OSGVruiPresets.h>

#include <osg/Geode>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osgText/Font>
#include <osg/StateSet>
#include <osg/Geometry>
#include <osg/Version>

#define STYLE_IN 1
#define STYLE_OUT 2
#define DETAIL_LEVEL 40 // number of triangles

using namespace std;
using namespace osg;
using namespace osgText;

namespace vrui
{

OSGVruiTextButtonGeometry::OSGVruiTextButtonGeometry(coTextButtonGeometry *geometry)
    : vruiButtonProvider(geometry)
    , normalNode(0)
    , pressedNode(0)
    , highlightNode(0)
    , pressedHighlightNode(0)
    , myDCS(0)
{
    textString = geometry->getTextureName();
}

OSGVruiTextButtonGeometry::~OSGVruiTextButtonGeometry()
{
}

void OSGVruiTextButtonGeometry::createGeometry()
{

    if (normalNode == 0)
    {

        color1 = osg::Vec4(((coTextButtonGeometry *)element)->c1r, ((coTextButtonGeometry *)element)->c1g, ((coTextButtonGeometry *)element)->c1b, ((coTextButtonGeometry *)element)->c1a);
        color2 = osg::Vec4(((coTextButtonGeometry *)element)->c2r, ((coTextButtonGeometry *)element)->c2g, ((coTextButtonGeometry *)element)->c2b, ((coTextButtonGeometry *)element)->c2a);

        normalNode = createNode(false, false);
        pressedNode = createNode(true, false);
        highlightNode = normalNode.get();
        pressedHighlightNode = pressedNode.get();

        ref_ptr<MatrixTransform> transformNode = new MatrixTransform();
        switchNode = new Switch();

        switchNode->addChild(normalNode.get());
        switchNode->addChild(pressedNode.get());
        switchNode->addChild(highlightNode.get());
        switchNode->addChild(pressedHighlightNode.get());

        transformNode->addChild(switchNode.get());

        myDCS = new OSGVruiTransformNode(transformNode.get());
    }
}

ref_ptr<Node> OSGVruiTextButtonGeometry::createNode(bool pressed, bool)
{

    ref_ptr<osg::MatrixTransform> nodemt = new osg::MatrixTransform();

    ref_ptr<osg::Geode> textnode = new osg::Geode();
    ref_ptr<osg::Geode> bgnode = new osg::Geode();
    ref_ptr<osg::MatrixTransform> textmt = new osg::MatrixTransform();

    ref_ptr<osg::Geometry> bggeo = new osg::Geometry();

    float bwidth = this->getWidth();
    float bheight = this->getHeight();

    ref_ptr<Text> text = new Text();

    text->setDataVariance(Object::DYNAMIC);
    text->setText(textString, String::ENCODING_UTF8);
    text->setFont(OSGVruiPresets::getFontFile());
    text->setAlignment(osgText::Text::CENTER_CENTER);
    text->setCharacterSize(20.0);
    text->setDrawMode(Text::TEXT);
    text->setFontResolution(32, 32);
    text->setPosition(osg::Vec3(0, 0, 0));
    text->setAxisAlignment(osgText::Text::XY_PLANE);
    if (pressed)
    {
        text->setColor(color1);
        //text->setColor(Vec4(0.9, 0.9, 0.9, 1.0));
    }
    else
    {
        text->setColor(color2);
        //text->setColor(Vec4(0.1, 0.1, 0.1, 1.0));
    }

    textnode->addDrawable(text.get());

    ref_ptr<osg::Vec3Array> bgVerts = new osg::Vec3Array();

    bgVerts->push_back(osg::Vec3(0, 0, 0));
    bgVerts->push_back(osg::Vec3(bwidth, 0, 0));
    bgVerts->push_back(osg::Vec3(bwidth, bheight, 0));
    bgVerts->push_back(osg::Vec3(0, bheight, 0));

    bggeo->setVertexArray(bgVerts.get());

    ref_ptr<osg::DrawElementsUInt> bge = new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS, 0);
    bge->push_back(0);
    bge->push_back(1);
    bge->push_back(2);
    bge->push_back(3);

    bggeo->addPrimitiveSet(bge.get());

    ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array;
    for (int i = 0; i < 4; ++i)
    {
        if (pressed)
        {
            //colors->push_back( Vec4(0.1, 0.1, 0.1, 1.0) );
            colors->push_back(color2);
        }
        else
        {
            colors->push_back(color1);
            //colors->push_back( Vec4(0.9, 0.9, 0.9, 1.0) );
        }
    }

    bggeo->setColorArray(colors.get());
    bggeo->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    bgnode->addDrawable(bggeo.get());

    osg::StateSet *stateset = textnode->getOrCreateStateSet();

    stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
    OSGVruiPresets::makeTransparent(stateset);
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    stateset->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);

    stateset = bgnode->getOrCreateStateSet();

    stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
    OSGVruiPresets::makeTransparent(stateset);
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

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

    return nodemt.get();
}

void OSGVruiTextButtonGeometry::switchGeometry(coButtonGeometry::ActiveGeometry active)
{

    createGeometry();
    switchNode->setSingleChildOn(active);
}

void OSGVruiTextButtonGeometry::resizeGeometry()
{
}

vruiTransformNode *OSGVruiTextButtonGeometry::getDCS()
{
    createGeometry();
    return myDCS;
}

float OSGVruiTextButtonGeometry::getWidth() const
{
    return ((coTextButtonGeometry *)element)->getInnerWidth();
}

float OSGVruiTextButtonGeometry::getHeight() const
{
    return ((coTextButtonGeometry *)element)->getInnerHeight();
}
}
