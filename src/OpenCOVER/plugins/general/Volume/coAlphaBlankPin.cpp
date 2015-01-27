/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coAlphaBlankPin.h"
#include <virvo/vvtoolshed.h>
#include <virvo/vvtransfunc.h>

#include <osg/Geode>

#include <OpenVRUI/osg/OSGVruiTransformNode.h>

#define ZOFFSET 0.4f

using namespace osg;

#define jPin static_cast<vvTFSkip *>(jPin)

coAlphaBlankPin::coAlphaBlankPin(Group *root, float Height, float Width,
                                 vvTFSkip *myPin)
    : coPin(root, Height, Width, myPin)
{
    numAlphaPins++;
    createGraphLists();
    root->addChild(createGraphGeode().get());
    adjustGraph();
    w1 = w2 = w3 = 0.0;
}

coAlphaBlankPin::~coAlphaBlankPin()
{
    numAlphaPins--;
    Node::ParentList parents = graphGeode->getParents();
    for (Node::ParentList::iterator i = parents.begin(); i != parents.end(); ++i)
        (*i)->removeChild(graphGeode.get());
}

void coAlphaBlankPin::setPos(float x)
{
    jPin->_pos[0] = x;
    myX = x;
    myDCS->setTranslation(x * W, 0.0, 0.0);
    adjustGraph();
}

void coAlphaBlankPin::setWidth(float w)
{
    jPin->_size[0] = w;
    jPin->_size[1] = jPin->_size[2] = 1.;
    adjustGraph();
}

void coAlphaBlankPin::adjustGraph()
{

    float x1, x2;

    if (jPin->_size[0] < 0.0)
    {
        jPin->_size[0] = 0.0;
    }
    x1 = myX - jPin->_size[0] / 2.0;
    x2 = myX + jPin->_size[0] / 2.0;
    if (x1 < 0.0)
    {
        x1 = 0.0;
    }
    if (x2 > 1.0)
    {
        x2 = 1.0;
    }

    (*graphCoord)[3].set(W * x1, 0.0, ZOFFSET);
    (*graphCoord)[2].set(W * x2, 0.0, ZOFFSET);
    (*graphCoord)[1].set(W * x2, -H, ZOFFSET);
    (*graphCoord)[0].set(W * x1, -H, ZOFFSET);

    graphGeode->dirtyBound();
    geometry->dirtyDisplayList();

    // adjust the selectionBar
    w1 = x1;
    if (w1 < 0.0)
        w1 = 0.0;
    w2 = (x2 - w1);
    if (w2 < 0.1f)
        w2 = 0.1f;
    w1 = myX - (w2 / 2.0);
    if (w1 < 0.0)
        w1 = 0.0;
    w2 = (x2 - w1);
    if (w2 < 0.1f)
        w2 = 0.1f;
    w3 = (1.0 - (w1 + w2));
    if (w3 < 0.0)
    {
        w3 = 0.0;
        if (w2 > w1)
            w2 = (1.0 - (w1 + w3));
        else if (w1 >= w2)
            w1 = (1.0 - (w2 + w3));
    }
}

void coAlphaBlankPin::createGraphLists()
{

    graphColor = new Vec4Array(1);
    graphCoord = new Vec3Array(4);
    graphNormal = new Vec3Array(1);

    (*graphCoord)[3].set(0.0, 0.0, ZOFFSET);
    (*graphCoord)[2].set(W * 1.0, 0.0, ZOFFSET);
    (*graphCoord)[1].set(W * 1.0, -H, ZOFFSET);
    (*graphCoord)[0].set(0.0, -H, ZOFFSET);

    (*graphColor)[0].set(1.0f, 1.0f, 1.0f, 0.9f);

    (*graphNormal)[0].set(0.0, 0.0, 1.0);
}

ref_ptr<Geode> coAlphaBlankPin::createGraphGeode()
{
    geometry = new Geometry();
    geometry->setColorArray(graphColor.get());
    geometry->setColorBinding(Geometry::BIND_OVERALL);
    geometry->setVertexArray(graphCoord.get());
    geometry->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));
    geometry->setNormalArray(graphNormal.get());
    geometry->setNormalBinding(Geometry::BIND_OVERALL);

    graphGeode = new Geode();
    graphGeode->setStateSet(normalGeostate.get());
    graphGeode->addDrawable(geometry.get());
    return graphGeode.get();
}
