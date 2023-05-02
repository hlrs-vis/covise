/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <GL/glew.h>

#include <cassert>
#include <stdexcept>
#include <iostream>

#include <osg/Depth>
#include <osg/Geometry>
#include <osg/TexEnv>
#include <osg/MatrixTransform>
#include <osg/Drawable>

#include <cover/coVRConfig.h>
#include <cover/coVRPluginSupport.h>

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_gl_interop.h>
#include "CudaTextureRectangle.h"
#include "CudaGraphicsResource.h"
#endif

#include "MultiChannelDrawer.h"

//#define INSTANCED // use instanced points instead of one vertex/pixel

// requires GL 3.2
#ifndef GL_PROGRAM_POINT_SIZE
#define GL_PROGRAM_POINT_SIZE             0x8642
#endif

namespace opencover
{

struct EmptyBounding: public osg::Drawable::ComputeBoundingBoxCallback {
    osg::BoundingBox computeBound(const osg::Drawable&) const {
        return osg::BoundingBox();
    }
};

//! osg::Drawable::DrawCallback for rendering selected geometry on one channel only
/*! decision is made based on cameras currently on osg's stack */
struct SingleScreenCB: public osg::Drawable::DrawCallback {

   osg::ref_ptr<MultiChannelDrawer> m_drawer;
   std::shared_ptr<ViewChannelData> m_viewChan;
   osg::ref_ptr<osg::Camera> m_cam;
   int m_channel = 0;
   bool m_second = false;
   mutable int m_renderCount = 0;

   SingleScreenCB(MultiChannelDrawer *drawer, osg::ref_ptr<osg::Camera> cam, int channel, bool second=false)
      : m_drawer(drawer)
      , m_cam(cam)
      , m_channel(channel)
      , m_second(second)
      {
          assert(m_channel >= 0);
      }

   void setViewChan(std::shared_ptr<ViewChannelData> vcd) {
       m_viewChan = vcd;
   }

   void  drawImplementation(osg::RenderInfo &ri, const osg::Drawable *d) const {

      bool render = false;
      if (!render) {
          auto stm = coVRConfig::instance()->channels[m_channel].stereoMode;
          const bool twoEyes = coVRConfig::requiresTwoViewpoints(stm);
          bool quadbuf = stm == osg::DisplaySettings::QUAD_BUFFER;
          ++m_renderCount;

          bool right = true;
          if (quadbuf) {
              GLint db=0;
              glGetIntegerv(GL_DRAW_BUFFER, &db);
              if (db != GL_BACK_RIGHT && db != GL_FRONT_RIGHT && db != GL_RIGHT)
                  right = false;
          } else if (twoEyes) {
              right = !(m_renderCount%2);
          }

          std::vector<osg::ref_ptr<osg::Camera> > cameraStack;
          while (ri.getCurrentCamera()) {
              if (ri.getCurrentCamera() == m_cam) {
                  render = true;
                  break;
              }
              cameraStack.push_back(ri.getCurrentCamera());
              ri.popCamera();
          }
          while (!cameraStack.empty()) {
              ri.pushCamera(cameraStack.back());
              cameraStack.pop_back();
          }

          if (twoEyes) {
              if (m_second && !right)
                  render = false;
              if (!m_second && right)
                  render = false;
          }
          //std::cerr << "investigated " << cameraStack.size() << " cameras for channel " << m_channel << " (2nd: " << m_second << "): render=" << render << ", right=" << right << ", count=" << m_renderCount << std::endl;
      }

      if (render)
         d->drawImplementation(ri);
   }
};


ViewChannelData::ViewChannelData(std::shared_ptr<ViewData> view, ChannelData *chan)
: chan(chan)
, view(view)
{
    std::string name = "channel"+std::to_string(chan->channelNum);
    name += "_view"+std::to_string(view->viewNum);
    drawCallback = new SingleScreenCB(chan->drawer, chan->camera, chan->channelNum, chan->second);
    drawCallback->setName(name+"_singlescreen");

    geode = new osg::Geode();
    geode->setName(name+"_geode");
    state = geode->getOrCreateStateSet();

    fixedGeo = new osg::Geometry(*view->fixedGeo);
    fixedGeo->setName(name+"_fixed");
    fixedGeo->setDrawCallback(drawCallback);
    fixedGeo->setComputeBoundingBoxCallback(new EmptyBounding);

    reprojGeo = new osg::Geometry(*view->reprojGeo);
    reprojGeo->setName(name+"_reprojected");
    reprojGeo->setDrawCallback(drawCallback);
    reprojGeo->setComputeBoundingBoxCallback(new EmptyBounding);

    reprojMat = new osg::Uniform(osg::Uniform::FLOAT_MAT4, "ReprojectionMatrix");
    reprojMat->set(osg::Matrix::identity());
    state->addUniform(reprojMat);

    state->setTextureAttributeAndModes(0, view->colorTex, osg::StateAttribute::ON);
    state->setTextureAttributeAndModes(1, view->depthTex, osg::StateAttribute::ON);
    osg::Uniform* colSampler = new osg::Uniform("col", 0);
    osg::Uniform* depSampler = new osg::Uniform("dep", 1);
    state->addUniform(colSampler);
    state->addUniform(depSampler);
}

ViewChannelData::~ViewChannelData()
{
    std::cerr << "delete ViewChannelData, view=" << view->viewNum << std::endl;
    while (geode->getNumParents() > 0)
        geode->getParent(0)->removeChild(geode);
}

void ViewChannelData::setThis(std::shared_ptr<ViewChannelData> vcd) {
    static_cast<SingleScreenCB *>(drawCallback.get())->setViewChan(vcd);
}

void ViewChannelData::update() {

    osg::Matrix cur = chan->curModel * chan->curView * chan->curProj;
    osg::Matrix old = view->imgModel * view->imgView * view->imgProj;
    osg::Matrix oldInv = osg::Matrix::inverse(old);
    osg::Matrix reproj = oldInv * cur;
    reprojMat->set(reproj);
}

ChannelData::~ChannelData() {
    std::cerr << "delete ChannelData" << std::endl;

    clearViews();

    while (scene->getNumParents() > 0)
        scene->getParent(0)->removeChild(scene);
}

void ChannelData::addView(std::shared_ptr<ViewData> vd) {
    auto vcd = std::make_shared<ViewChannelData>(vd, this);
    vcd->setThis(vcd);
    viewChan.emplace_back(vcd);
    vd->viewChan.emplace_back(vcd.get());
    scene->addChild(vcd->geode);
}

void ChannelData::enableView(std::shared_ptr<ViewData> vd, bool enable) {
    for (auto vcd: viewChan) {
        if (vcd->view == vd) {
            unsigned idx = scene->getChildIndex(vcd->geode);
            if (enable) {
                if (idx == scene->getNumChildren()) {
                    scene->addChild(vcd->geode);
                }
            } else {
                if (idx != scene->getNumChildren()) {
                    scene->removeChild(idx);
                }
            }
            return;
        }
    }
}

void ChannelData::clearViews() {

    for (auto &vcd: viewChan) {
        scene->removeChild(vcd->geode);
    }
    assert(scene->getNumChildren() == 0);

    viewChan.clear();
}

void ChannelData::updateViews() {
    for (auto &vcd: viewChan)
        vcd->update();
}

const char reprojVert[] =

#ifdef INSTANCED
      // for % operator on integers
      "#extension GL_EXT_gpu_shader4: enable\n"
      "#extension GL_ARB_draw_instanced: enable\n"
#endif
      "#extension GL_ARB_texture_rectangle: enable\n"
      "\n"
      "uniform sampler2DRect col;\n"
      "uniform sampler2DRect dep;\n"
      "uniform vec2 size;\n"
      "uniform mat4 ReprojectionMatrix;\n"
      "\n"

      "float depth(vec2 xy) {\n"
      "   return texture2DRect(dep, xy).r;\n"
      "}\n"

      "vec4 pos(vec2 xy, float d) {\n"
      "   vec4 p = vec4(xy.x/size.x-0.5, 0.5-xy.y/size.y, d-0.5, 0.5)*2.;\n"
      "   return ReprojectionMatrix * p;\n"
      "}\n"

      "void main(void) {\n"
#ifdef INSTANCED
      "   vec2 xy = vec2(float(gl_InstanceIDARB%int(size.x)), float(gl_InstanceIDARB/int(size.x)))+vec2(0.5,0.5);\n"
#else
      "   vec2 xy = gl_Vertex.xy;\n"
#endif
      "   vec4 color = texture2DRect(col, xy);\n"
      "   gl_FrontColor = color;\n"

      "   gl_Position = pos(xy, depth(xy));\n"

      "}\n";

const char reprojAdaptVert[] =

#ifdef INSTANCED
      // for % operator on integers
      "#extension GL_EXT_gpu_shader4: enable\n"
      "#extension GL_ARB_draw_instanced: enable\n"
#else
      // for round
      "#extension GL_EXT_gpu_shader4: enable\n"
#endif
      "#extension GL_ARB_texture_rectangle: enable\n"
      "\n"
      "uniform sampler2DRect col;\n"
      "uniform sampler2DRect dep;\n"
      "uniform vec2 size;\n"
      "uniform mat4 ReprojectionMatrix;\n"
      "uniform vec2 offset;\n"
      "uniform bool withNeighbors;\n"
      "\n"

      "bool is_far(float d) {\n"
      "   return d == 1.;\n"
      "}\n"

      "float depth(vec2 xy) {\n"
      "   return texture2DRect(dep, xy).r;\n"
      "}\n"

      "vec4 pos(vec2 xy, float d) {\n"
      "   vec4 p = vec4(xy.x/size.x-0.5, 0.5-xy.y/size.y, d-0.5, 0.5)*2.;\n"
      "   return ReprojectionMatrix * p;\n"
      "}\n"

      "vec2 screenpos(vec4 p) {\n"
      "   return round(p.xy/p.w*size.xy*0.5+offset);\n"
      "}\n"

      "vec2 sdiff(float dd, vec2 xy, vec2 ref) {\n"
      "   float d = withNeighbors ? depth(xy) : dd;\n"
      "   if (is_far(d)) return vec2(1.,1.);\n"
      "   return abs(screenpos(pos(xy, d))-ref);\n"
      "}\n"

      "void main(void) {\n"
#ifdef INSTANCED
      "   vec2 xy = vec2(float(gl_InstanceIDARB%int(size.x)), float(gl_InstanceIDARB/int(size.x)))+vec2(0.5,0.5);\n"
#else
      "   vec2 xy = gl_Vertex.xy;\n"
#endif
      "   const vec4 Clip = vec4(2.,2.,2.,1.);\n"

      "   float d = depth(xy);\n"
      "   if (is_far(d)) { gl_Position = Clip; return; }\n"
      "   gl_Position = pos(xy, d);\n"
      //"   if (is_far(d)) { gl_PointSize=1.; gl_FrontColor = vec4(1,0,0,1); return; }\n"

      "   vec4 color = texture2DRect(col, xy);\n"
      "   gl_FrontColor = color;\n"

      "   vec2 spos = screenpos(gl_Position);\n"
      "   vec2 dxp = sdiff(d, xy+vec2(1.,0.), spos);\n"
      "   vec2 dyp = sdiff(d, xy+vec2(0.,1.), spos);\n"
      "   vec2 dxm = sdiff(d, xy+vec2(-1.,0.), spos);\n"
      "   vec2 dym = sdiff(d, xy+vec2(0.,-1.), spos);\n"
      "   vec2 dmax = max(max(dxp,dxm),max(dyp,dym));\n"
      //"   vec2 dmax = max(dxp,dym);\n"
      "   float ps = max(dmax.x, dmax.y);\n"
      //"   if (ps > 1.00008 && ps < 2.) { ps=2.; }\n"
      //"   if (ps > 1. && ps <= 2.) { ps=2.; gl_FrontColor=vec4(1,0,0,1); }\n"
      //"   if (ps >= 4.) ps = 1.;\n"

      "   gl_PointSize = clamp(ps, 1., 3.);\n"
      "}\n";

const char reprojFrag[] =
        "void main(void) {\n"
        "   gl_FragColor = gl_Color;\n"
        "}\n";

const char reprojMeshVert[] =
      "void main(void) {\n"
      "   gl_Position = vec4(gl_Vertex.xy, 0., 1.);\n"
      "}\n";

const char reprojMeshGeo[] =
      "#version 120\n"
      "#extension GL_EXT_geometry_shader4 : enable\n"
      "#extension GL_ARB_texture_rectangle: enable\n"
      "\n"
      "uniform sampler2DRect col;\n"
      "uniform sampler2DRect dep;\n"
      "uniform vec2 size;\n"
      "uniform mat4 ReprojectionMatrix;\n"
      "uniform bool withHoles;\n"
      "uniform vec2 offset;\n"
      "uniform vec2 off[] = vec2[4]( vec2(0,0), vec2(0,1), vec2(1,0), vec2(1,1) );\n"
      "const float tolerance = 10.f;\n"
      "\n"

      "float depth(vec2 xy) {\n"
      "   return texture2DRect(dep, xy).r;\n"
      "}\n"

      "vec4 pos(vec2 xy, float d) {\n"
      "   vec4 p = vec4(xy.x/size.x-0.5, 0.5-xy.y/size.y, d-0.5, 0.5)*2.;\n"
      "   return ReprojectionMatrix * p;\n"
      "}\n"

      "vec2 screenpos(vec4 p) {\n"
      "   return p.xy/p.w*size.xy*0.5+offset;\n"
      "}\n"

      "vec2 pos2d(vec2 xy) {\n"
      "   float d = depth(xy);\n"
      "   return screenpos(pos(xy, d));\n"
      "}\n"

      "bool is_far(float d) {\n"
      "   return d == 1.;\n"
      "}\n"

      "void createVertex(vec2 xy) {\n"
      "   float d = depth(xy);\n"
      "   if (is_far(d)) return;\n"

      "   vec4 color = texture2DRect(col, xy);\n"
      "   gl_FrontColor = color;\n"
      "   gl_Position = pos(xy, d);\n"
      "   EmitVertex();\n"
      "}\n"

      "void main() {\n"
      "   vec2 xy = gl_PositionIn[0].xy;\n"
      ""
      "   bool render = true;\n"
      "   if (withHoles) {\n"
      "      vec2 p[4];\n"
      "      for (int i=0; i<4; ++i)\n"
      "         p[i] = pos2d(xy+off[i]);\n"
      "      float mindist = distance(p[0],p[1]), maxdist=mindist;\n"
      "      for (int i=1; i<4; ++i) {\n"
      "          for (int j=0; j<i; ++j) {\n"
      "              float dist = distance(p[i],p[j]);\n"
      "              mindist = min(mindist, dist);\n"
      "              maxdist = max(maxdist, dist);\n"
      "          }\n"
      "      }\n"
      "      if (maxdist < 0.001 || mindist*tolerance < maxdist) {\n"
      "         render = false;\n"
      "      }\n"
      "   }\n"
      "   if (render) {\n"
      "      for (int i=0; i<4; ++i)\n"
      "         createVertex(xy+off[i]);\n"
      "   }\n"
      "   EndPrimitive();\n"
      "}\n";

MultiChannelDrawer::MultiChannelDrawer(bool flipped, bool useCuda)
: m_flipped(flipped)
, m_mode(MultiChannelDrawer::ReprojectMesh)
, m_useCuda(useCuda)
{
   setAllowEventFocus(false);
   setProjectionMatrix(osg::Matrix::identity());
   setComputeNearFarMode(osg::CullSettings::DO_NOT_COMPUTE_NEAR_FAR);
   setReferenceFrame(osg::Transform::ABSOLUTE_RF);
   setViewMatrix(osg::Matrix::identity());
   setName("MultiChannelDrawer");

   //setRenderTargetImplementation( osg::Camera::FRAME_BUFFER, osg::Camera::FRAME_BUFFER );


   //setClearDepth(0.9999);
   //setClearColor(osg::Vec4f(1., 0., 0., 1.));
   //setClearMask(GL_COLOR_BUFFER_BIT);
   //setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   setClearMask(0);

   //int win = coVRConfig::instance()->channels[0].window;
   //setGraphicsContext(coVRConfig::instance()->windows[win].window);
   setRenderOrder(osg::Camera::NESTED_RENDER);
   //setRenderer(new osgViewer::Renderer(m_remoteCam.get()));

   int numChannels = coVRConfig::instance()->numChannels();
   for (int i=0; i<numChannels; ++i) {
       bool stereo = coVRConfig::instance()->channels[i].stereo;
       int stereomode = coVRConfig::instance()->channels[i].stereoMode;
       bool left = stereomode != osg::DisplaySettings::RIGHT_EYE;
       m_channelData.emplace_back(std::make_shared<ChannelData>(this, i));
       m_channelData.back()->eye = Middle;
       if (stereo) {
           m_channelData.back()->eye = left ? Left : Right;
       }
       initChannelData(*m_channelData.back());
       if (coVRConfig::requiresTwoViewpoints(stereomode)) {
           m_channelData.emplace_back(std::make_shared<ChannelData>(this, i));
           m_channelData.back()->eye = Right;
           m_channelData.back()->second = true;
           initChannelData(*m_channelData.back());
       }
   }

   setViewsToRender(Same);
   setNumViews(-1);
}

MultiChannelDrawer::~MultiChannelDrawer() {

   for (size_t i=0; i<m_channelData.size(); ++i)
       removeChild(m_channelData[i]->scene);
   m_viewData.clear();
   m_channelData.clear();
}

int MultiChannelDrawer::numViews() const {

    return m_viewData.size();
}

const osg::Matrix &MultiChannelDrawer::modelMatrix(int idx) const {

    return m_channelData[idx]->curModel;
}

const osg::Matrix &MultiChannelDrawer::viewMatrix(int idx) const {

    return m_channelData[idx]->curView;
}

const osg::Matrix &MultiChannelDrawer::projectionMatrix(int idx) const {

    return m_channelData[idx]->curProj;
}

void MultiChannelDrawer::update() {

   const osg::Matrix &transform = cover->getXformMat();
   const osg::Matrix &scale = cover->getObjectsScale()->getMatrix();
   const osg::Matrix model = scale * transform;

   auto updateChannel = [&model](ChannelData &cd, int i, bool second) {

       const channelStruct &chan = coVRConfig::instance()->channels[i];
       const bool left = chan.stereoMode == osg::DisplaySettings::LEFT_EYE
           || (!second && coVRConfig::requiresTwoViewpoints(chan.stereoMode));
       const osg::Matrix &view = left ? chan.leftView : chan.rightView;
       const osg::Matrix &proj = left ? chan.leftProj : chan.rightProj;
       cd.curModel = model;
       cd.curView = view;
       cd.curProj = proj;

       cd.updateViews();
   };

   int numChannels = coVRConfig::instance()->numChannels();
   int view = 0;
   for (int i=0; i<numChannels; ++i) {
       ChannelData &cd = *m_channelData[view];
       updateChannel(cd, i, false);
       auto stereomode = coVRConfig::instance()->channels[i].stereoMode;
       if (coVRConfig::requiresTwoViewpoints(stereomode)) {
           ++view;
           ChannelData &cd = *m_channelData[view];
           updateChannel(cd, i, true);
       }
       ++view;
   }
}
//! create geometry for mapping remote image
void MultiChannelDrawer::createGeometry(ChannelData &cd)
{
   cd.scene = new osg::Group;
   cd.scene->setName("channel_"+std::to_string(cd.channelNum)+(cd.second?"A":"B"));
}

//! create geometry for mapping remote image
void MultiChannelDrawer::createGeometry(ViewData &vd)
{

   vd.texcoord  = new osg::Vec2Array(4);
   (*vd.texcoord)[0].set(0.0,480.0);
   (*vd.texcoord)[1].set(640.0,480.0);
   (*vd.texcoord)[2].set(640.0,0.0);
   (*vd.texcoord)[3].set(0.0,0.0);

   osg::Vec4Array *color = new osg::Vec4Array(1);
   osg::Vec3Array *normal = new osg::Vec3Array(1);
   (*color)    [0].set(1, 1, 0, 1.0f);
   (*normal)   [0].set(0.0f, -1.0f, 0.0f);

   osg::TexEnv * texEnv = new osg::TexEnv();
   texEnv->setMode(osg::TexEnv::REPLACE);

   vd.fixedGeo = new osg::Geometry();
   vd.fixedGeo->setName("fixed_geo");
   ushort vertices[4] = { 0, 1, 2, 3 };
   osg::DrawElementsUShort *plane = new osg::DrawElementsUShort(osg::PrimitiveSet::QUADS, 4, vertices);

   vd.fixedGeo->addPrimitiveSet(plane);
   vd.fixedGeo->setColorArray(color);
   vd.fixedGeo->setColorBinding(osg::Geometry::BIND_OVERALL);
   vd.fixedGeo->setNormalArray(normal);
   vd.fixedGeo->setNormalBinding(osg::Geometry::BIND_OVERALL);
   vd.fixedGeo->setTexCoordArray(0, vd.texcoord);
   vd.fixedGeo->setComputeBoundingBoxCallback(new EmptyBounding);
   {
      osg::Vec3Array *coord  = new osg::Vec3Array(4);
      (*coord)[0 ].set(-1., -1., 0.);
      (*coord)[1 ].set( 1., -1., 0.);
      (*coord)[2 ].set( 1.,  1., 0.);
      (*coord)[3 ].set(-1.,  1., 0.);
      vd.fixedGeo->setVertexArray(coord);
      vd.fixedGeo->setUseVertexBufferObjects( true );

      vd.fixedGeo->setUseDisplayList( false ); // required for DrawCallback
      osg::StateSet *stateSet = vd.fixedGeo->getOrCreateStateSet();
      //stateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
      //stateSet->setRenderBinDetails(-20,"RenderBin");
      //stateSet->setNestRenderBins(false);
      osg::Depth* depth = new osg::Depth;
      depth->setFunction(osg::Depth::LEQUAL);
      depth->setRange(0.0,1.0);
      stateSet->setAttribute(depth);
      stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
      stateSet->setTextureAttribute(0, texEnv);
      stateSet->setTextureAttribute(1, texEnv);

      osg::Program *depthProgramObj = new osg::Program;
      osg::Shader *depthFragmentObj = new osg::Shader( osg::Shader::FRAGMENT );
      depthProgramObj->addShader(depthFragmentObj);
      depthFragmentObj->setShaderSource(
            "#version 120\n"
            "#extension GL_ARB_texture_rectangle : enable\n"
            "\n"
            "uniform sampler2DRect col;"
            "uniform sampler2DRect dep;"
            "void main(void) {"
            "   vec4 color = texture2DRect(col, gl_TexCoord[0].xy);"
            "   gl_FragColor = color;"
            "   gl_FragDepth = texture2DRect(dep, gl_TexCoord[0].xy).x;"
            "}"
            );
      stateSet->setAttributeAndModes(depthProgramObj, osg::StateAttribute::ON);
      vd.fixedGeo->setStateSet(stateSet);
   }

   vd.reprojGeo = new osg::Geometry();
   vd.reprojGeo->setName("reprojection_geo");
   {
      vd.reprojGeo->setUseDisplayList( false );
      vd.reprojGeo->setComputeBoundingBoxCallback(new EmptyBounding);
#ifndef INSTANCED
      vd.pointCoord = new osg::Vec2Array(1);
      (*vd.pointCoord)[0].set(0., 0.);
      vd.reprojGeo->setVertexArray(vd.pointCoord);
#endif
      vd.quadCoord = new osg::Vec2Array(0);
      vd.reprojGeo->setColorArray(color);
      vd.reprojGeo->setColorBinding(osg::Geometry::BIND_OVERALL);
      vd.reprojGeo->setNormalArray(normal);
      vd.reprojGeo->setNormalBinding(osg::Geometry::BIND_OVERALL);
      // required for instanced rendering and also for SingleScreenCB
      vd.reprojGeo->setSupportsDisplayList( false );
      vd.reprojGeo->setUseVertexBufferObjects( true );

      osg::StateSet *stateSet = vd.reprojGeo->getOrCreateStateSet();
      osg::Depth* depth = new osg::Depth;
      depth->setFunction(osg::Depth::LEQUAL);
      depth->setRange(0.0,1.0);
      stateSet->setAttribute(depth);
      stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
      stateSet->setTextureAttribute(0, texEnv);
      stateSet->setTextureAttribute(1, texEnv);

      vd.size = new osg::Uniform(osg::Uniform::FLOAT_VEC2, "size");
      vd.pixelOffset = new osg::Uniform(osg::Uniform::FLOAT_VEC2, "offset");
      vd.withNeighbors = new osg::Uniform(osg::Uniform::BOOL, "withNeighbors");
      vd.withHoles = new osg::Uniform(osg::Uniform::BOOL, "withHoles");
      stateSet->addUniform(vd.size);
      stateSet->addUniform(vd.pixelOffset);
      stateSet->addUniform(vd.withNeighbors);
      stateSet->addUniform(vd.withHoles);

      {
         vd.reprojConstProgram = new osg::Program;
         osg::Shader *reprojVertexObj = new osg::Shader( osg::Shader::VERTEX );
         reprojVertexObj->setShaderSource(reprojVert);
         vd.reprojConstProgram->addShader(reprojVertexObj);

         osg::Shader *reprojFragmentObj = new osg::Shader( osg::Shader::FRAGMENT );
         reprojFragmentObj->setShaderSource(reprojFrag);
         vd.reprojConstProgram->addShader(reprojFragmentObj);
      }

      {
         vd.reprojAdaptProgram = new osg::Program;
         osg::Shader *reprojVertexObj = new osg::Shader( osg::Shader::VERTEX );
         reprojVertexObj->setShaderSource(reprojAdaptVert);
         vd.reprojAdaptProgram->addShader(reprojVertexObj);

         osg::Shader *reprojFragmentObj = new osg::Shader( osg::Shader::FRAGMENT );
         reprojFragmentObj->setShaderSource(reprojFrag);
         vd.reprojAdaptProgram->addShader(reprojFragmentObj);
      }

      {
          vd.reprojMeshProgram = new osg::Program;
          osg::Shader *reprojVertexObj = new osg::Shader( osg::Shader::VERTEX );
          reprojVertexObj->setShaderSource(reprojMeshVert);
          vd.reprojMeshProgram->addShader(reprojVertexObj);

          osg::Shader *reprojGeoObj = new osg::Shader( osg::Shader::GEOMETRY );
          reprojGeoObj->setShaderSource(reprojMeshGeo);
          vd.reprojMeshProgram->addShader(reprojGeoObj);
          vd.reprojMeshProgram->setParameter( GL_GEOMETRY_VERTICES_OUT_EXT, 4 );
          vd.reprojMeshProgram->setParameter( GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS );
          vd.reprojMeshProgram->setParameter( GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP );

          osg::Shader *reprojFragmentObj = new osg::Shader( osg::Shader::FRAGMENT );
          reprojFragmentObj->setShaderSource(reprojFrag);
          vd.reprojMeshProgram->addShader(reprojFragmentObj);
      }

      vd.reprojGeo->setStateSet(stateSet);
   }
}

void MultiChannelDrawer::initChannelData(ChannelData &cd) {

   cd.camera = coVRConfig::instance()->channels[cd.channelNum].camera;
   createGeometry(cd);
   cd.scene->setNodeMask(cd.scene->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));
   addChild(cd.scene);
}


void MultiChannelDrawer::initViewData(ViewData &vd) {

#ifdef HAVE_CUDA
   if (m_useCuda)
   {
       vd.colorTex = new CudaTextureRectangle;
       vd.depthTex = new CudaTextureRectangle;
   }
   else
#endif
   {
       for (int i=0; i<ViewData::NumImages; ++i) {
           vd.colorImg[i] = new osg::Image;
           auto pboc = new osg::PixelBufferObject(vd.colorImg[i]);
           pboc->setUsage(GL_STREAM_DRAW);
           vd.colorImg[i]->setPixelBufferObject(pboc);
           vd.depthImg[i] = new osg::Image;
           auto pbod = new osg::PixelBufferObject(vd.depthImg[i]);
           pbod->setUsage(GL_STREAM_DRAW);
           vd.depthImg[i]->setPixelBufferObject(pbod);
       }

       vd.colorTex = new osg::TextureRectangle;
       vd.depthTex = new osg::TextureRectangle;
       for (auto tex: {vd.colorTex, vd.depthTex}) {
           tex->setResizeNonPowerOfTwoHint(false);
           tex->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
           tex->setWrap(osg::Texture::WRAP_T, osg::Texture::CLAMP_TO_EDGE);
       }
       vd.colorTex->setImage(vd.colorImg[writeTex]);
       vd.depthTex->setImage(vd.depthImg[writeTex]);
   }

   for (auto tex: {vd.colorTex, vd.depthTex}) {
       tex->setBorderWidth( 0 );
       tex->setFilter( osg::Texture::MIN_FILTER, osg::Texture::NEAREST );
       tex->setFilter( osg::Texture::MAG_FILTER, osg::Texture::NEAREST );
   }


   vd.colorTex->setInternalFormat( GL_RGBA );
   vd.depthTex->setInternalFormat( GL_DEPTH_COMPONENT32F );

   createGeometry(vd);

   //std::cout << "vp: " << vp->width() << "," << vp->height() << std::endl;
}

void MultiChannelDrawer::clearViewData() {

    for (size_t view=0; view<m_channelData.size(); ++view) {
       clearColor(view);
       clearDepth(view);
    }
}

bool MultiChannelDrawer::haveEye(MultiChannelDrawer::ViewEye eye) {
    if (m_viewsToRender != MatchingEye)
        return true;

    if (eye == Middle)
        return m_availableEyes.middle;
    if (eye == Left)
        return m_availableEyes.left;
    if (eye == Right)
        return m_availableEyes.right;

    return false;
}

void MultiChannelDrawer::swapFrame() {

   //std::cerr << "MultiChannelDrawer::swapFrame" << std::endl;

   for (size_t s=0; s<m_viewData.size(); ++s) {
      ViewData &vd = *m_viewData[s];

      updateGeoForView(vd);

      vd.imgView = vd.newView;
      vd.imgProj = vd.newProj;
      vd.imgModel = vd.newModel;

      if (!haveEye(vd.eye))
          continue;

#ifdef HAVE_CUDA
      if (m_useCuda)
      {
          vd.colorTex->dirtyTextureObject();
          vd.depthTex->dirtyTextureObject();
      }
      else
#endif
      {
          vd.colorImg[renderTex]->dirty();
          vd.depthImg[renderTex]->dirty();
      }
   }
}

void MultiChannelDrawer::updateMatrices(int idx, const osg::Matrix &model, const osg::Matrix &view, const osg::Matrix &proj) {

   ViewData &vd = *m_viewData[idx];
   vd.newModel = model;
   vd.newView = view;
   vd.newProj = proj;
}

void MultiChannelDrawer::resizeView(int idx, int w, int h, GLenum depthFormat, GLenum colorFormat) {

    ViewData &vd = *m_viewData[idx];
    ChannelData *cd = nullptr;
    if (m_useCuda) {
        if (size_t(idx) < m_channelData.size())
            cd = m_channelData[idx].get();
    }

    if (colorFormat && (vd.width[writeTex] != w || vd.height[writeTex] != h || vd.colorFormat[writeTex] != colorFormat))
    {
        GLenum colorInternalFormat = 0;
        int colorTypeSize = 0;
        switch (colorFormat)
        {
        case GL_FLOAT:
            colorInternalFormat = GL_RGBA32F;
            colorTypeSize = 16;
            break;
        case GL_UNSIGNED_BYTE:
            colorInternalFormat = GL_RGBA8;
            colorTypeSize = 4;
            break;
        default:
            throw std::runtime_error("Color pixel type not supported!");
        }

#ifdef HAVE_CUDA
        if (m_useCuda && cd)
        {
            vd.colorTex->setTextureSize(w, h);
            vd.colorTex->setSourceFormat(GL_RGBA);
            vd.colorTex->setSourceType(colorFormat);
            vd.colorTex->setInternalFormat(colorInternalFormat);

            osg::State* state = cd->camera->getGraphicsContext()->getState();

            static_cast<CudaTextureRectangle*>(vd.colorTex.get())->resize(state, w, h, colorTypeSize);
        }
        else
#endif
        {
            auto cimg = vd.colorImg[writeTex];
            cimg->setInternalTextureFormat(colorFormat);
            cimg->allocateImage(w, h, 1, GL_RGBA, colorFormat);
        }

        vd.width[writeTex] = w;
        vd.height[writeTex] = h;
        vd.colorFormat[writeTex] = colorFormat;
    }

    if (depthFormat != 0 && (vd.depthWidth[writeTex] != w || vd.depthHeight[writeTex] != h || vd.depthFormat[writeTex] != depthFormat))
    {
        //std::cerr << "MultiChannelDrawer: need to update geo, format=" << depthFormat << ", w=" << w << ", h=" << h << std::endl;
        GLenum depthInternalFormat = 0;
        int depthTypeSize = 0;
        switch (depthFormat)
        {
        case GL_FLOAT:
            depthInternalFormat = GL_DEPTH_COMPONENT32F;
            depthTypeSize = 4;
            break;
        case GL_UNSIGNED_INT_24_8:
            depthInternalFormat = GL_DEPTH_COMPONENT24;
            depthTypeSize = 4;
            break;
        default:
            throw std::runtime_error("Depth pixel type not supported!");
        }

#ifdef HAVE_CUDA
        if (m_useCuda && cd)
        {
            vd.depthTex->setTextureSize(w, h);
            vd.depthTex->setSourceFormat(GL_DEPTH_COMPONENT);
            vd.depthTex->setSourceType(depthFormat == GL_UNSIGNED_INT_24_8 ? GL_UNSIGNED_INT : depthFormat);
            vd.depthTex->setInternalFormat(depthInternalFormat);

            osg::State* state = cd->camera->getGraphicsContext()->getState();

            static_cast<CudaTextureRectangle*>(vd.depthTex.get())->resize(state, w, h, depthTypeSize);
        }
        else
#endif
        {
            auto dimg = vd.depthImg[writeTex];
            dimg->setInternalTextureFormat(depthInternalFormat);
            dimg->allocateImage(w, h, 1, GL_DEPTH_COMPONENT, depthFormat == GL_UNSIGNED_INT_24_8 ? GL_UNSIGNED_INT : depthFormat);
        }

        vd.depthWidth[writeTex] = w;
        vd.depthHeight[writeTex] = h;
        vd.depthFormat[writeTex] = depthFormat;
    }
}

void MultiChannelDrawer::updateGeoForView(ViewData &vd) {

    if (vd.width[renderTex] != vd.depthWidth[renderTex]) {
        std::cerr << "MultiChannelDrawer::updateGeoForView(" << vd.viewNum << "), renderTex=" << renderTex
                  << ": width mismatch: " << vd.width[renderTex] << " != " << vd.depthWidth[renderTex] << std::endl;
    }
    if (vd.height[renderTex] != vd.depthHeight[renderTex]) {
        std::cerr << "MultiChannelDrawer::updateGeoForView(" << vd.viewNum << "), renderTex=" << renderTex
                  << ": height mismatch: " << vd.height[renderTex] << " != " << vd.depthHeight[renderTex] << std::endl;
    }

    int w = vd.width[renderTex], h = vd.height[renderTex];

    if (w == vd.geoWidth && h == vd.geoHeight)
        return;

    vd.geoWidth = w;
    vd.geoHeight = h;

    w = std::max(w, 1);
    h = std::max(h, 1);

    if (m_flipped) {
        (*vd.texcoord)[0].set(0., h);
        (*vd.texcoord)[1].set(w, h);
        (*vd.texcoord)[2].set(w, 0.);
        (*vd.texcoord)[3].set(0., 0.);
    }
    else {
        (*vd.texcoord)[0].set(0., 0.);
        (*vd.texcoord)[1].set(w, 0.);
        (*vd.texcoord)[2].set(w, h);
        (*vd.texcoord)[3].set(0., h);
    }
    vd.fixedGeo->setTexCoordArray(0, vd.texcoord);
    for (auto &vcd: vd.viewChan) {
        vcd->fixedGeo->setTexCoordArray(0, vd.texcoord);
        vcd->fixedGeo->getTexCoordArray(0)->dirty();
    }
    vd.texcoord->dirty();

#ifndef INSTANCED
    vd.pointCoord->resizeArray(w*h);
    for (int y = 0; y<h; ++y) {
        for (int x = 0; x<w; ++x) {
            (*vd.pointCoord)[y*w + x].set(x + 0.5f, m_flipped ? y + 0.5f : h - y + 0.5f);
        }
    }
    vd.pointCoord->dirty();
    vd.pointArr = new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, w*h);
#endif

    vd.quadCoord->resizeArray((w - 1)*(h - 1));
    size_t idx = 0;
    for (int y = 0; y<h - 1; ++y) {
        for (int x = 0; x<w - 1; ++x) {
            (*vd.quadCoord)[idx++].set(x + 0.5f, m_flipped ? y + 0.5f : h - y + 0.5f);
        }
    }
    vd.quadCoord->dirty();
    vd.quadArr = new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, (w - 1)*(h - 1));

    osg::ref_ptr<osg::DrawArrays> arr = (m_mode == ReprojectMesh || m_mode == ReprojectMeshWithHoles) ? vd.quadArr : vd.pointArr;

    auto updateGeo = [this, arr, w, h](osg::Geometry *geo){
#ifdef INSTANCED
        if (geo->getNumPrimitiveSets() > 0) {
            geo->setPrimitiveSet(0, new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, 1, w*h));
        }
        else {
            geo->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, 1, w*h));
        }
#else

        if (geo->getNumPrimitiveSets() > 0) {
            geo->setPrimitiveSet(0, arr);
        } else {
            geo->addPrimitiveSet(arr);
        }
        geo->dirtyDisplayList();
#endif
    };
    updateGeo(vd.reprojGeo);
    for (auto &vcd: vd.viewChan) {
        updateGeo(vcd->reprojGeo);
    }

    vd.size->set(osg::Vec2(w, h));
    vd.pixelOffset->set(osg::Vec2((w + 1) % 2 * 0.5f, (h + 1) % 2 * 0.5f));
}

std::shared_ptr<MultiChannelDrawer::ViewData> MultiChannelDrawer::getViewData(int idx) {

    return m_viewData[idx];
}

unsigned char *MultiChannelDrawer::rgba(int idx) const {
    if (idx < 0 || idx >= m_viewData.size()) {
        std::cerr << "MultiChannelDrawer::rgba: index=" << idx << " out of range: #channels=" << m_channelData.size() << std::endl;
        return nullptr;
    }

    const ViewData &vd = *m_viewData[idx];

#ifdef HAVE_CUDA
    if (m_useCuda)
    {
        return static_cast<unsigned char*>(static_cast<CudaTextureRectangle*>(vd.colorTex.get())->resourceData());
    }
    else
#endif
    {
        return vd.colorImg[writeTex]->data();
    }
}

unsigned char *MultiChannelDrawer::depth(int idx) const {
    if (idx < 0 || idx >= m_viewData.size()) {
        std::cerr << "MultiChannelDrawer::depth: index=" << idx << " out of range: #channels=" << m_channelData.size() << std::endl;
        return nullptr;
    }

    const ViewData &vd = *m_viewData[idx];

#ifdef HAVE_CUDA
    if (m_useCuda)
    {
        return static_cast<unsigned char*>(static_cast<CudaTextureRectangle*>(vd.depthTex.get())->resourceData());
    }
    else
#endif
    {
        return vd.depthImg[writeTex]->data();
    }
}

void MultiChannelDrawer::clearColor(int idx) {
    ViewData &vd = *m_viewData[idx];

#ifdef HAVE_CUDA
    if (m_useCuda)
    {
        static_cast<CudaTextureRectangle*>(vd.colorTex.get())->clear();
    }
    else
#endif
    {
        osg::Image *color = vd.colorImg[writeTex];
        memset(color->data(), 0, color->getTotalSizeInBytes());
        color->dirty();
    }
}

void MultiChannelDrawer::clearDepth(int idx) {
    ViewData &vd = *m_viewData[idx];

#ifdef HAVE_CUDA
    if (m_useCuda)
    {
        static_cast<CudaTextureRectangle*>(vd.depthTex.get())->clear();
    }
    else
#endif
    {
        osg::Image *depth = vd.depthImg[writeTex];
        memset(depth->data(), 0, depth->getTotalSizeInBytes());
        depth->dirty();
    }
}

MultiChannelDrawer::Mode MultiChannelDrawer::mode() const {

    return m_mode;
}

void MultiChannelDrawer::setMode(MultiChannelDrawer::Mode mode) {

    m_mode = mode;

    for (auto &cd: m_channelData) {

        for (auto &vcd: cd->viewChan) {
            vcd->geode->removeDrawable(vcd->fixedGeo);
            vcd->geode->removeDrawable(vcd->reprojGeo);
            assert(vcd->geode->getNumDrawables() == 0);
        }

        for (auto &vcd: cd->viewChan) {

            const auto &vd = *vcd->view;

            auto &geo = mode==AsIs ? vcd->fixedGeo : vcd->reprojGeo;
            vcd->geode->addDrawable(geo);

            switch(mode) {
            case AsIs:
                vcd->fixedGeo->setStateSet(vd.fixedGeo->getStateSet());
                break;
            case Reproject:
            case ReprojectAdaptive:
            case ReprojectAdaptiveWithNeighbors: {
#ifdef INSTANCED
                int numpix = vd.width*vd.height;
                if (geo->getNumPrimitiveSets() > 0) {
                    geo->setPrimitiveSet(0, new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, 1, numpix));
                }
                else {
                    geo->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, 1, numpix));
                }
#else
                geo->setVertexArray(vd.pointCoord);
                if (vd.pointArr) {
                    if (geo->getNumPrimitiveSets() > 0)
                        geo->setPrimitiveSet(0, vd.pointArr);
                    else
                        geo->addPrimitiveSet(vd.pointArr);
                }
#endif
                break;
            }
            case ReprojectMesh:
            case ReprojectMeshWithHoles:
                geo->setVertexArray(vd.quadCoord);
                if (vd.quadArr) {
                    if (geo->getNumPrimitiveSets() > 0)
                        geo->setPrimitiveSet(0, vd.quadArr);
                    else
                        geo->addPrimitiveSet(vd.quadArr);
                }
                break;
            }
        }
    }

    for (size_t i=0; i<m_viewData.size(); ++i) {
        ViewData &vd = *m_viewData[i];
        osg::StateSet *state = vd.reprojGeo->getStateSet();
        assert(state);

        switch(mode) {
        case AsIs:
            break;
        case Reproject:
            state->setMode(GL_PROGRAM_POINT_SIZE, osg::StateAttribute::OFF);
            state->setAttributeAndModes(vd.reprojAdaptProgram, osg::StateAttribute::OFF);
            state->setAttributeAndModes(vd.reprojMeshProgram, osg::StateAttribute::OFF);
            state->setAttributeAndModes(vd.reprojConstProgram, osg::StateAttribute::ON);
            break;
        case ReprojectAdaptive:
        case ReprojectAdaptiveWithNeighbors:
            state->setMode(GL_PROGRAM_POINT_SIZE, osg::StateAttribute::ON);
            state->setAttributeAndModes(vd.reprojConstProgram, osg::StateAttribute::OFF);
            state->setAttributeAndModes(vd.reprojMeshProgram, osg::StateAttribute::OFF);
            state->setAttributeAndModes(vd.reprojAdaptProgram, osg::StateAttribute::ON);
            vd.withNeighbors->set(mode == ReprojectAdaptiveWithNeighbors ? true : false);
            break;
        case ReprojectMesh:
        case ReprojectMeshWithHoles:
            state->setMode(GL_PROGRAM_POINT_SIZE, osg::StateAttribute::OFF);
            state->setAttributeAndModes(vd.reprojConstProgram, osg::StateAttribute::OFF);
            state->setAttributeAndModes(vd.reprojAdaptProgram, osg::StateAttribute::OFF);
            state->setAttributeAndModes(vd.reprojMeshProgram, osg::StateAttribute::ON);
            vd.withHoles->set(mode == ReprojectMeshWithHoles ? true : false);
            break;
        }
    }
}

MultiChannelDrawer::ViewSelection MultiChannelDrawer::viewsToRender() const {
    return m_viewsToRender;
}

void MultiChannelDrawer::setViewsToRender(ViewSelection views) {
    m_viewsToRender = views;

    for (auto &cd: m_channelData) {
        for (auto &vd: m_viewData) {
            cd->enableView(vd, m_viewsToRender==Same || m_viewsToRender==All || (m_viewsToRender==MatchingEye && cd->eye==vd->eye));
        }
    }
}

void MultiChannelDrawer::setNumViews(int nv) {

    for (auto &cd: m_channelData) {
        cd->clearViews();
    }
    for (auto &vd: m_viewData) {
        vd->viewChan.clear();
    }

    int n = nv==-1 ? m_channelData.size() : nv;
    if (int(m_viewData.size()) > n) {
        m_viewData.resize(n);
    } else {
        for (int i=m_viewData.size(); i<n; ++i) {
            m_viewData.emplace_back(std::make_shared<ViewData>(i));
            initViewData(*m_viewData.back());
        }
    }

    bool matchEyes = m_viewsToRender == MatchingEye;

    int idx = 0;
    for (auto &cd: m_channelData) {
        if (m_viewsToRender == Same) {
            assert(m_viewData.size() > idx);
            cd->addView(m_viewData[idx]);
        } else {
            for (auto &vd: m_viewData) {
                cd->addView(vd);
                cd->enableView(vd, cd->eye==vd->eye || !matchEyes || m_numViews!=nv);
            }
        }
        ++idx;
    }

    m_numViews = nv;

    setMode(m_mode);

    std::cerr << "setNumViews(nv=" << nv << "): to render=" << m_viewsToRender << ", #chan=" << m_channelData.size() << ", #views=" << m_viewData.size() << ", channels:";
    for (auto &cd: m_channelData) {
        std::cerr << " " << cd->viewChan.size();
    }

    std::cerr << ", eyes: ";

    for (auto &cd: m_channelData) {
        if (cd->eye == Middle) {
            m_availableEyes.middle = true;
            std::cerr << "M";
        }
        if (cd->eye == Left) {
            m_availableEyes.left = true;
            std::cerr << "L";
        }
        if (cd->eye == Right) {
            m_availableEyes.right = true;
            std::cerr << "R";
        }
    }
    std::cerr << std::endl;
}

void MultiChannelDrawer::setViewEye(int view, ViewEye eye) {

    if (m_viewData[view]->eye == eye)
        return;

    m_viewData[view]->eye = eye;

    bool matchEyes = m_viewsToRender==MatchingEye;

    auto vd = m_viewData[view];
    for (auto &cd: m_channelData) {
        cd->enableView(vd, cd->eye==eye || !matchEyes);
    }
}

ViewData::~ViewData()
{
    std::cerr << "delete ViewData: view=" << viewNum << ", #chan=" << viewChan.size() << std::endl;
}

}
