/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <OpenVRUI/osg/OSGVruiTransformNode.h>
#include <OpenVRUI/osg/OSGVruiHit.h>
#include <OpenVRUI/osg/OSGVruiPresets.h>
#include <OpenVRUI/osg/OSGVruiTexture.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/util/vruiLog.h>

#include <osg/CullFace>
#include <osg/Material>
#include <osg/LineWidth>
#include <osg/PolygonMode>
#include <osgUtil/IntersectVisitor>

#include <config/CoviseConfig.h>
#include "coPlotItem.h"

//--------------------------------------------------------------------
int *lenList_plotterline = NULL;

//--------------------------------------------------------------------
coPlotItem::coPlotItem(int plotter_no)
{
    myDCS = new OSGVruiTransformNode(new osg::MatrixTransform());
    myDCS->getNodePtr()->asGroup()->addChild(createPlotter(plotter_no).get());
}

//--------------------------------------------------------------------
coPlotItem::~coPlotItem()
{
    myDCS->removeAllChildren();
    myDCS->removeAllParents();
    delete myDCS;
}

//--------------------------------------------------------------------
vruiTransformNode *coPlotItem::getDCS(void)
{
    return (myDCS);
}

//--------------------------------------------------------------------
void coPlotItem::setPos(float x, float y, float)
{
    myX = x;
    myY = y;
    myDCS->setTranslation(x, y + getHeight(), 0.0);
}

//--------------------------------------------------------------------
int coPlotItem::hit(vruiHit *)
{
    return (0);
}

//--------------------------------------------------------------------
void coPlotItem::miss(void)
{
}

//--------------------------------------------------------------------
void coPlotItem::update(void)
{
}

//--------------------------------------------------------------------
osg::ref_ptr<osg::Geode> coPlotItem::createPlotter(int plotterid)
{

    osg::ref_ptr<osg::Vec3Array> coordList_face;
    osg::ref_ptr<osg::Vec3Array> coordList_axisbar;
    osg::ref_ptr<osg::Vec3Array> coordList_arrows;
    osg::ref_ptr<osg::Vec3Array> coordList_lines;

    osg::ref_ptr<osg::Vec4Array> color_face;
    osg::ref_ptr<osg::Vec4Array> color_axes;
    osg::ref_ptr<osg::Vec4Array> color_lines;

    osg::ref_ptr<osg::Geometry> geoset_face;
    osg::ref_ptr<osg::Geometry> geoset_axisbar;
    osg::ref_ptr<osg::Geometry> geoset_arrows;
    osg::ref_ptr<osg::Geometry> geoset_lines;

    osg::ref_ptr<osg::Geode> geode;

    osg::ref_ptr<osg::Material> mtl;
    mtl = new osg::Material;
    mtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    mtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.1, 0.1, 0.1, 1.0));
    mtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0.6, 0.6, 0.6, 1.0));
    mtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0, 1.0, 1.0, 1.0));
    mtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0, 0.0, 0.0, 1.0));
    mtl->setShininess(osg::Material::FRONT_AND_BACK, 80.0f);

    osg::ref_ptr<osg::StateSet> normalGeostate;
    normalGeostate = new osg::StateSet();
    normalGeostate->setGlobalDefaults();

    osg::ref_ptr<osg::CullFace> cullFace = new osg::CullFace();
    cullFace->setMode(osg::CullFace::BACK);

    normalGeostate->setAttributeAndModes(cullFace.get(), osg::StateAttribute::ON);
    normalGeostate->setAttributeAndModes(mtl.get(), osg::StateAttribute::ON);
    normalGeostate->setMode(GL_BLEND, osg::StateAttribute::ON);
    normalGeostate->setMode(GL_LIGHTING, osg::StateAttribute::ON);

    float x, y;

    plo.h = 150;
    plo.b = 250;
    plo.t = plo.h / 100;

    if ((dat.maxy[plotterid] >= 0) && (dat.miny[plotterid] <= 0))
    {
        plo.ky[plotterid] = (-dat.maxy[plotterid] / (dat.maxy[plotterid] - dat.miny[plotterid])) * plo.h;
    }
    if ((dat.maxy[plotterid] > 0) && (dat.miny[plotterid] > 0))
    {
        plo.ky[plotterid] = -plo.h;
    }
    if ((dat.maxy[plotterid] < 0) && (dat.miny[plotterid] < 0))
    {
        plo.ky[plotterid] = 0;
    }

    geode = new osg::Geode();
    geode->setName("plotter");

    // draw face
    coordList_face = new osg::Vec3Array(4);
    (*coordList_face)[0].set(-plo.b * 0.05, -plo.h * 1.05, -plo.t);
    (*coordList_face)[1].set(plo.b * 1.05, -plo.h * 1.05, -plo.t);
    (*coordList_face)[2].set(plo.b * 1.05, plo.h * 0.05, -plo.t);
    (*coordList_face)[3].set(-plo.b * 0.05, plo.h * 0.05, -plo.t);

    color_face = new osg::Vec4Array(1);
    (*color_face)[0].set(colorindex[47][0],
                         colorindex[47][1],
                         colorindex[47][2],
                         colorindex[47][3]);

    geoset_face = new osg::Geometry();
    geoset_face->setVertexArray(coordList_face.get());
    geoset_face->setColorArray(color_face.get());
    geoset_face->setColorBinding(osg::Geometry::BIND_OVERALL);
    geoset_face->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 4));
    geoset_face->setStateSet(normalGeostate.get());

    geode->addDrawable(geoset_face.get());

    // draw axes
    coordList_axisbar = new osg::Vec3Array(8);
    (*coordList_axisbar)[0].set(0, -plo.h, 0);
    (*coordList_axisbar)[1].set(plo.b * 0.02, -plo.h, 0);
    (*coordList_axisbar)[2].set(plo.b * 0.02, -plo.h * 0.05, 0);
    (*coordList_axisbar)[3].set(0, -plo.h * 0.05, 0);

    (*coordList_axisbar)[4].set(0, plo.ky[plotterid] - plo.h * 0.02, 0);
    (*coordList_axisbar)[5].set(plo.b * 0.95, plo.ky[plotterid] - plo.h * 0.02, 0);
    (*coordList_axisbar)[6].set(plo.b * 0.95, plo.ky[plotterid], 0);
    (*coordList_axisbar)[7].set(0, plo.ky[plotterid], 0);

    color_axes = new osg::Vec4Array(1);
    (*color_axes)[0].set(colorindex[48][0],
                         colorindex[48][1],
                         colorindex[48][2],
                         colorindex[48][3]);
    geoset_axisbar = new osg::Geometry();
    geoset_axisbar->setVertexArray(coordList_axisbar.get());
    geoset_axisbar->setColorArray(color_axes.get());
    geoset_axisbar->setColorBinding(osg::Geometry::BIND_OVERALL);
    geoset_axisbar->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 8));
    geoset_axisbar->setStateSet(normalGeostate.get());

    geode->addDrawable(geoset_axisbar.get());

    coordList_arrows = new osg::Vec3Array(6);
    (*coordList_arrows)[0].set(-plo.b * 0.02, -plo.h * 0.05, 0);
    (*coordList_arrows)[1].set(-plo.b * 0.04, -plo.h * 0.05, 0);
    (*coordList_arrows)[2].set(-plo.b * 0.01, 0, 0);

    (*coordList_arrows)[3].set(plo.b * 0.95, plo.ky[plotterid] - plo.h * 0.04, 0);
    (*coordList_arrows)[4].set(plo.b, plo.ky[plotterid] - plo.h * 0.01, 0);
    (*coordList_arrows)[5].set(plo.b * 0.95, plo.ky[plotterid] + plo.h * 0.02, 0);

    geoset_arrows = new osg::Geometry();
    geoset_arrows->setVertexArray(coordList_arrows.get());
    geoset_arrows->setColorArray(color_axes.get());
    geoset_arrows->setColorBinding(osg::Geometry::BIND_OVERALL);
    geoset_arrows->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES, 0, 6));
    geoset_arrows->setStateSet(normalGeostate.get());
    geode->addDrawable(geoset_arrows.get());
    // draw lines
    coordList_lines = new osg::Vec3Array(str.timesteps);
    for (int j = 0; j < str.timesteps; j++)
    {
        float width;
        if ((dat.maxy[plotterid] >= 0) && (dat.miny[plotterid] <= 0))
        {
            width = dat.maxy[plotterid] - dat.miny[plotterid];
        }
        if ((dat.maxy[plotterid] > 0) && (dat.miny[plotterid] > 0))
        {
            width = dat.maxy[plotterid];
        }
        if ((dat.maxy[plotterid] < 0) && (dat.miny[plotterid] < 0))
        {
            width = -dat.miny[plotterid];
        }
        x = (float)j * plo.b / (float)str.timesteps;
        y = plo.ky[plotterid] + plo.h * dat.data[plotterid][j] / width;
        (*coordList_lines)[j].set(x, y, 0);
    }

    color_lines = new osg::Vec4Array(1);
    (*color_lines)[0].set(colorindex[49][0],
                          colorindex[49][1],
                          colorindex[49][2],
                          colorindex[49][3]);
    geoset_lines = new osg::Geometry();
    geoset_lines->setVertexArray(coordList_lines.get());
    geoset_lines->setColorArray(color_lines.get());
    geoset_lines->setColorBinding(osg::Geometry::BIND_OVERALL);
    lineDrawArray = new osg::DrawArrays(osg::PrimitiveSet::LINE_STRIP, 0, str.timesteps);
    geoset_lines->addPrimitiveSet(lineDrawArray.get());
    geoset_lines->setStateSet(normalGeostate.get());
    geoset_lines->setUseDisplayList(false);

    osg::LineWidth *lineWidth = new osg::LineWidth(0.2);
    normalGeostate->setAttributeAndModes(lineWidth, osg::StateAttribute::ON);
    geode->addDrawable(geoset_lines.get());
    /*
  if(lenList_plotterline == NULL){
    lenList_plotterline = (int*)pfCalloc(1, sizeof(int), pfGetSharedArena());
  }
  lenList_plotterline[0] =str.timesteps-1;
  geoset_lines->setPrimLengths(lenList_plotterline);
  geoset_lines->setLineWidth(0.2f);
  geoset_lines->setGState(geostate);
  geode->addGSet(geoset_lines);
*/
    return (geode);
}

//----------------------------------------------------------------------------
