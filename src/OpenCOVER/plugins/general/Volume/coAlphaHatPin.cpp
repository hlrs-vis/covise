/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coAlphaHatPin.h"
#include <virvo/vvtoolshed.h>
#include <virvo/vvtransfunc.h>

#include <OpenVRUI/osg/OSGVruiTransformNode.h>

#define ZOFFSET 0.2f

using namespace osg;

#define jPin static_cast<vvTFPyramid *>(jPin)

coAlphaHatPin::coAlphaHatPin(Group *root, float Height, float Width, vvTFPyramid *myPin)
    : coPin(root, Height, Width, myPin)
{
    numAlphaPins++;
    createGraphLists();
    root->addChild(createGraphGeode().get());
    adjustGraph();
    w1 = w2 = w3 = 0.0;
}

coAlphaHatPin::~coAlphaHatPin()
{
    numAlphaPins--;
    Node::ParentList parents = graphGeode->getParents();
    for (Node::ParentList::iterator i = parents.begin(); i != parents.end(); ++i)
        (*i)->removeChild(graphGeode.get());
}

void coAlphaHatPin::setPos(float x, float minv, float maxv)
{
    coPin::setPos(x, minv, maxv);
    adjustGraph(minv, maxv);
}

void coAlphaHatPin::setTopWidth(float s, float minv, float maxv)
{
    myTopWidth = (s - minv) / (maxv - minv);
    //cerr << "setTopWidth=" << s << endl;
    if (s > jPin->_bottom[0])
        jPin->_top[0] = jPin->_bottom[0];
    else
        jPin->_top[0] = s;
    /*
   float slope = (jPin->_bottom[0]-jPin->_top[0]) * .5 / jPin->_opacity;
   float nbottom = jPin->_opacity / s + jPin->_top[0];
   bool canHandle = false;
   float hoff = nbottom/4. + jPin->_top[0]/2.;
   if( 0. <= jPin->_pos[0] && jPin->_pos[0] <= 1. )
      canHandle = true;
   if( 0. <= jPin->_pos[0]+hoff && jPin->_pos[0]+hoff <= 1. )
      canHandle = true;
   if( 0. <= jPin->_pos[0]-hoff && jPin->_pos[0]-hoff <= 1. )
      canHandle = true;

   if(canHandle)
   {
      jPin->_bottom[0] = nbottom;
      slope = s;
   }

   return slope;
*/
    adjustGraph(minv, maxv);
}

float coAlphaHatPin::getTopWidthValue() const
{
    return jPin->top()[0];
}

float coAlphaHatPin::getTopWidth01() const
{
    return myTopWidth;
}

void coAlphaHatPin::setMax(float m, float minv, float maxv)
{
    //cerr << "setMax=" << m << endl;
    jPin->_opacity = m;
    /*
   float slope = jPin->_opacity / (jPin->_bottom[0]-jPin->_top[0]) * 2.;
   float nbottom = m / slope * 2. + jPin->_top[0];
   float hoff = nbottom/4. + jPin->_top[0]/2.;

   bool canHandle = false;
   if( 0. <= jPin->_pos[0] && jPin->_pos[0] <= 1. )
      canHandle = true;
   if( 0. <= jPin->_pos[0]+hoff && jPin->_pos[0]+hoff <= 1. )
      canHandle = true;
   if( 0. <= jPin->_pos[0]-hoff && jPin->_pos[0]-hoff <= 1. )
      canHandle = true;

   if(canHandle)
   {
      jPin->_opacity = m;
      jPin->_bottom[0] = nbottom;
   }
*/

    adjustGraph(minv, maxv);
}

void coAlphaHatPin::setBotWidth(float m, float minv, float maxv)
{
    myBotWidth = (m - minv) / (maxv - minv);
    //std::cerr << "setBotWidth=" << m << std::endl;
    if (m < jPin->_top[0])
        jPin->_bottom[0] = jPin->_top[0];
    else
        jPin->_bottom[0] = m;
    /*
   float diff = m - jPin->_top[0];
   float nbottom = jPin->_bottom[0] + diff;  // new bottom width
   float hoff = nbottom/4. + m/2.;

   bool canHandle = false;
   if( 0. <= jPin->_pos[0] && jPin->_pos[0] <= 1. )
      canHandle = true;
   if( 0. <= jPin->_pos[0]+hoff && jPin->_pos[0]+hoff <= 1. )
      canHandle = true;
   if( 0. <= jPin->_pos[0]-hoff && jPin->_pos[0]-hoff <= 1. )
      canHandle = true;

   if(canHandle)
   {
      jPin->_top[0] = m;
      jPin->_bottom[0] += diff;
   }
*/
    adjustGraph(minv, maxv);
}

float coAlphaHatPin::getBotWidthValue() const
{
    return jPin->bottom()[0];
}

float coAlphaHatPin::getBotWidth01() const
{
    return myBotWidth;
}

void coAlphaHatPin::adjustGraph(float minv, float maxv)
{
    // x-coords, [0..1]
    float x2 = myX - ((jPin->_top[0] - minv) / (maxv - minv)) / 2.0;
    float x3 = myX + ((jPin->_top[0] - minv) / (maxv - minv)) / 2.0;
    float x1 = myX - ((jPin->_bottom[0] - minv) / (maxv - minv)) / 2.;
    float x4 = myX + ((jPin->_bottom[0] - minv) / (maxv - minv)) / 2.;
    x1 = ts_clamp(x1, 0.0f, 1.0f);
    x2 = ts_clamp(x2, 0.0f, 1.0f);
    x3 = ts_clamp(x3, 0.0f, 1.0f);
    x4 = ts_clamp(x4, 0.0f, 1.0f);

    // x-coords, [minv..maxv]
    float vx2 = (myX + minv) * (maxv - minv) - jPin->_top[0] / 2.0;
    float vx3 = (myX + minv) * (maxv - minv) + jPin->_top[0] / 2.0;
    float vx1 = (myX + minv) * (maxv - minv) - jPin->_bottom[0] / 2.;
    float vx4 = (myX + minv) * (maxv - minv) + jPin->_bottom[0] / 2.;
    vx1 = ts_clamp(vx1, minv, maxv);
    vx2 = ts_clamp(vx2, minv, maxv);
    vx3 = ts_clamp(vx3, minv, maxv);
    vx4 = ts_clamp(vx4, minv, maxv);

    float y1 = jPin->getOpacity(vx1);
    float y2 = jPin->getOpacity(vx2);
    float y3 = jPin->getOpacity(vx3);
    float y4 = jPin->getOpacity(vx4);

//    std::cout << vx1 << ": " << y1 << ' '
//              << vx2 << ": " << y2 << ' '
//              << vx3 << ": " << y3 << ' '
//              << vx4 << ": " << y4 << '\n';

    (*graphCoord)[0].set(W * x1, -((1.0 - y1) * H), ZOFFSET);
    (*graphCoord)[1].set(W * x2, -((1.0 - y2) * H), ZOFFSET);
    (*graphCoord)[2].set(W * x3, -((1.0 - y3) * H), ZOFFSET);
    (*graphCoord)[3].set(W * x4, -((1.0 - y4) * H), ZOFFSET);
    (*graphCoord)[4].set(W * x4, -H, ZOFFSET);
    (*graphCoord)[5].set(W * x3, -H, ZOFFSET);
    (*graphCoord)[6].set(W * x2, -H, ZOFFSET);
    (*graphCoord)[7].set(W * x1, -H, ZOFFSET);

    graphGeode->dirtyBound();
	graphCoord->dirty();
    geometry->dirtyDisplayList();

    // FIXME: wofuer ist das?
    w2 = (x3 - x2);
    if (w2 < 0.2f)
        w2 = 0.2f;
    if (w2 > 0.6f)
        w2 = 0.6f;
    w1 = myX - (w2 / 2.0);
    if (w1 < 0.01f)
        w1 = 0.01f;
    if (w1 + w2 > 0.99f)
        w1 = 0.99f - w2;
    w3 = (1.0 - (w1 + w2));
}

void coAlphaHatPin::createGraphLists()
{

    graphColor = new Vec4Array(1);
    graphCoord = new Vec3Array(8);
    graphNormal = new Vec3Array(1);

    ushort *graphVertices = new ushort[3 * 4];

    (*graphCoord)[0].set(0.0f, -(0.2f * H), ZOFFSET);
    (*graphCoord)[1].set(W * 0.4f, -(0.8f * H), ZOFFSET);
    (*graphCoord)[2].set(W * 0.6f, -(0.8f * H), ZOFFSET);
    (*graphCoord)[3].set(W, -(0.2f * H), ZOFFSET);
    (*graphCoord)[4].set(W, -H, ZOFFSET);
    (*graphCoord)[5].set(W * 0.6f, -H, ZOFFSET);
    (*graphCoord)[6].set(W * 0.4f, -H, ZOFFSET);
    (*graphCoord)[7].set(0.0f, -H, ZOFFSET);

    (*graphColor)[0].set(0.5f, 0.5f, 0.5f, 0.9f);

    (*graphNormal)[0].set(0.0, 0.0, 1.0);

    graphVertices[0] = 1;
    graphVertices[1] = 0;
    graphVertices[2] = 7;
    graphVertices[3] = 6;

    graphVertices[4] = 2;
    graphVertices[5] = 1;
    graphVertices[6] = 6;
    graphVertices[7] = 5;

    graphVertices[8] = 3;
    graphVertices[9] = 2;
    graphVertices[10] = 5;
    graphVertices[11] = 4;

    coordIndex = new DrawElementsUShort(PrimitiveSet::QUADS, 12, graphVertices);

    delete[] graphVertices;
}

ref_ptr<Geode> coAlphaHatPin::createGraphGeode()
{

    geometry = new Geometry();

    geometry->setColorArray(graphColor.get());
    geometry->setColorBinding(Geometry::BIND_OVERALL);
    geometry->setVertexArray(graphCoord.get());
    geometry->addPrimitiveSet(coordIndex.get());
    geometry->setNormalArray(graphNormal.get());
    geometry->setNormalBinding(Geometry::BIND_OVERALL);

    graphGeode = new Geode();
    graphGeode->setStateSet(normalGeostate.get());
    graphGeode->addDrawable(geometry.get());
    return graphGeode.get();
}
