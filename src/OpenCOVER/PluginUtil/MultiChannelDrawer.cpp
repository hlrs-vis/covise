/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "MultiChannelDrawer.h"

#include <cassert>

#include <osg/Depth>
#include <osg/Geometry>
#include <osg/TexEnv>
#include <osg/MatrixTransform>
#include <osg/Drawable>

#include <cover/coVRConfig.h>
#include <cover/coVRPluginSupport.h>

//#define INSTANCED

// requires GL 3.2
#ifndef GL_PROGRAM_POINT_SIZE
#define GL_PROGRAM_POINT_SIZE             0x8642
#endif

namespace opencover
{

//! osg::Drawable::DrawCallback for rendering selected geometry on one channel only
/*! decision is made based on cameras currently on osg's stack */
struct SingleScreenCB: public osg::Drawable::DrawCallback {

   osg::ref_ptr<osg::Camera> m_cam;
   int m_channel;
   bool m_second;

   SingleScreenCB(osg::ref_ptr<osg::Camera> cam, int channel, bool second=false)
      : m_cam(cam)
      , m_channel(channel)
      , m_second(second)
      {
      }

   void  drawImplementation(osg::RenderInfo &ri, const osg::Drawable *d) const {

      bool render = false;
      const bool stereo = coVRConfig::instance()->channels[m_channel].stereoMode == osg::DisplaySettings::QUAD_BUFFER;

      bool right = true;
      if (stereo) {
         GLint db=0;
         glGetIntegerv(GL_DRAW_BUFFER, &db);
         if (db != GL_BACK_RIGHT && db != GL_FRONT_RIGHT && db != GL_RIGHT)
            right = false;
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

      if (stereo) {
         if (m_second && right)
            render = false;
         if (!m_second && !right)
            render = false;
      }
      //std::cerr << "investigated " << cameraStack.size() << " cameras for channel " << m_channel << " (2nd: " << m_second << "): render=" << render << ", right=" << right << std::endl;

      if (render)
         d->drawImplementation(ri);
   }
};

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

MultiChannelDrawer::MultiChannelDrawer(int numChannels, bool flipped)
: m_flipped(flipped)
, m_mode(MultiChannelDrawer::ReprojectMesh)
{
   setAllowEventFocus(false);
   setProjectionMatrix(osg::Matrix::identity());
   setComputeNearFarMode(osg::CullSettings::DO_NOT_COMPUTE_NEAR_FAR);
   setReferenceFrame(osg::Transform::ABSOLUTE_RF);
   setViewMatrix(osg::Matrix::identity());

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

   for (int i=0; i<numChannels; ++i) {
      m_channelData.push_back(ChannelData(i));
      initChannelData(m_channelData.back());
      addChild(m_channelData.back().scene);
      if (coVRConfig::instance()->channels[i].stereoMode == osg::DisplaySettings::QUAD_BUFFER) {
         m_channelData.push_back(ChannelData(i));
         m_channelData.back().second = true;
         initChannelData(m_channelData.back());
         addChild(m_channelData.back().scene);
      }
   }

   setMode(m_mode);
}

MultiChannelDrawer::~MultiChannelDrawer() {

   for (size_t i=0; i<m_channelData.size(); ++i)
       removeChild(m_channelData[i].scene);
   m_channelData.clear();
}


//! create geometry for mapping remote image
void MultiChannelDrawer::createGeometry(ChannelData &cd)
{

   cd.texcoord  = new osg::Vec2Array(4);
   (*cd.texcoord)[0].set(0.0,480.0);
   (*cd.texcoord)[1].set(640.0,480.0);
   (*cd.texcoord)[2].set(640.0,0.0);
   (*cd.texcoord)[3].set(0.0,0.0);

   osg::Vec4Array *color = new osg::Vec4Array(1);
   osg::Vec3Array *normal = new osg::Vec3Array(1);
   (*color)    [0].set(1, 1, 0, 1.0f);
   (*normal)   [0].set(0.0f, -1.0f, 0.0f);

   osg::TexEnv * texEnv = new osg::TexEnv();
   texEnv->setMode(osg::TexEnv::REPLACE);

   osg::Geode *geometryNode = new osg::Geode();

   osg::ref_ptr<osg::Drawable::DrawCallback> drawCB = new SingleScreenCB(cd.camera, cd.channelNum, cd.second);

   cd.fixedGeo = new osg::Geometry();
   ushort vertices[4] = { 0, 1, 2, 3 };
   osg::DrawElementsUShort *plane = new osg::DrawElementsUShort(osg::PrimitiveSet::QUADS, 4, vertices);

   cd.fixedGeo->addPrimitiveSet(plane);
   cd.fixedGeo->setColorArray(color);
   cd.fixedGeo->setColorBinding(osg::Geometry::BIND_OVERALL);
   cd.fixedGeo->setNormalArray(normal);
   cd.fixedGeo->setNormalBinding(osg::Geometry::BIND_OVERALL);
   cd.fixedGeo->setTexCoordArray(0, cd.texcoord);
   {
      osg::Vec3Array *coord  = new osg::Vec3Array(4);
      (*coord)[0 ].set(-1., -1., 0.);
      (*coord)[1 ].set( 1., -1., 0.);
      (*coord)[2 ].set( 1.,  1., 0.);
      (*coord)[3 ].set(-1.,  1., 0.);
      cd.fixedGeo->setVertexArray(coord);

      cd.fixedGeo->setDrawCallback(drawCB);
      cd.fixedGeo->setUseDisplayList( false ); // required for DrawCallback
      osg::StateSet *stateSet = cd.fixedGeo->getOrCreateStateSet();
      //stateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
      //stateSet->setRenderBinDetails(-20,"RenderBin");
      //stateSet->setNestRenderBins(false);
      osg::Depth* depth = new osg::Depth;
      depth->setFunction(osg::Depth::LEQUAL);
      depth->setRange(0.0,1.0);
      stateSet->setAttribute(depth);
      stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
      stateSet->setTextureAttributeAndModes(0, cd.colorTex, osg::StateAttribute::ON);
      stateSet->setTextureAttribute(0, texEnv);
      stateSet->setTextureAttributeAndModes(1, cd.depthTex, osg::StateAttribute::ON);
      stateSet->setTextureAttribute(1, texEnv);

      osg::Uniform* colSampler = new osg::Uniform("col", 0);
      osg::Uniform* depSampler = new osg::Uniform("dep", 1);
      stateSet->addUniform(colSampler);
      stateSet->addUniform(depSampler);

      osg::Program *depthProgramObj = new osg::Program;
      osg::Shader *depthFragmentObj = new osg::Shader( osg::Shader::FRAGMENT );
      depthProgramObj->addShader(depthFragmentObj);
      depthFragmentObj->setShaderSource(
            "uniform sampler2DRect col;"
            "uniform sampler2DRect dep;"
            "void main(void) {"
            "   vec4 color = texture2DRect(col, gl_TexCoord[0].xy);"
            "   gl_FragColor = color;"
            "   gl_FragDepth = texture2DRect(dep, gl_TexCoord[0].xy).x;"
            "}"
            );
      stateSet->setAttributeAndModes(depthProgramObj, osg::StateAttribute::ON);
      cd.fixedGeo->setStateSet(stateSet);
   }

   cd.reprojGeo = new osg::Geometry();
   {
      cd.reprojGeo->setDrawCallback(drawCB);
      cd.reprojGeo->setUseDisplayList( false );
      osg::BoundingBox bb(-0.5,0.,-0.5, 0.5,0.,0.5);
      cd.reprojGeo->setInitialBound(bb);
      cd.pointCoord = new osg::Vec2Array(1);
      (*cd.pointCoord)[0].set(0., 0.);
      cd.reprojGeo->setVertexArray(cd.pointCoord);
      cd.quadCoord = new osg::Vec2Array(0);
      cd.reprojGeo->setColorArray(color);
      cd.reprojGeo->setColorBinding(osg::Geometry::BIND_OVERALL);
      cd.reprojGeo->setNormalArray(normal);
      cd.reprojGeo->setNormalBinding(osg::Geometry::BIND_OVERALL);
      // required for instanced rendering and also for SingleScreenCB
      cd.reprojGeo->setUseDisplayList( false );
      cd.reprojGeo->setUseVertexBufferObjects( true );

      osg::StateSet *stateSet = cd.reprojGeo->getOrCreateStateSet();
      osg::Depth* depth = new osg::Depth;
      depth->setFunction(osg::Depth::LEQUAL);
      depth->setRange(0.0,1.0);
      stateSet->setAttribute(depth);
      stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
      stateSet->setTextureAttributeAndModes(0, cd.colorTex, osg::StateAttribute::ON);
      stateSet->setTextureAttribute(0, texEnv);
      stateSet->setTextureAttributeAndModes(1, cd.depthTex, osg::StateAttribute::ON);
      stateSet->setTextureAttribute(1, texEnv);

      osg::Uniform *colSampler = new osg::Uniform("col", 0);
      osg::Uniform *depSampler = new osg::Uniform("dep", 1);
      cd.size = new osg::Uniform(osg::Uniform::FLOAT_VEC2, "size");
      cd.pixelOffset = new osg::Uniform(osg::Uniform::FLOAT_VEC2, "offset");
      cd.withNeighbors = new osg::Uniform(osg::Uniform::BOOL, "withNeighbors");
      cd.withHoles = new osg::Uniform(osg::Uniform::BOOL, "withHoles");
      cd.reprojMat = new osg::Uniform(osg::Uniform::FLOAT_MAT4, "ReprojectionMatrix");
      cd.reprojMat->set(osg::Matrix::identity());
      stateSet->addUniform(colSampler);
      stateSet->addUniform(depSampler);
      stateSet->addUniform(cd.size);
      stateSet->addUniform(cd.pixelOffset);
      stateSet->addUniform(cd.withNeighbors);
      stateSet->addUniform(cd.withHoles);
      stateSet->addUniform(cd.reprojMat);

      {
         cd.reprojConstProgram = new osg::Program;
         osg::Shader *reprojVertexObj = new osg::Shader( osg::Shader::VERTEX );
         reprojVertexObj->setShaderSource(reprojVert);
         cd.reprojConstProgram->addShader(reprojVertexObj);

         osg::Shader *reprojFragmentObj = new osg::Shader( osg::Shader::FRAGMENT );
         reprojFragmentObj->setShaderSource(reprojFrag);
         cd.reprojConstProgram->addShader(reprojFragmentObj);
      }

      {
         cd.reprojAdaptProgram = new osg::Program;
         osg::Shader *reprojVertexObj = new osg::Shader( osg::Shader::VERTEX );
         reprojVertexObj->setShaderSource(reprojAdaptVert);
         cd.reprojAdaptProgram->addShader(reprojVertexObj);

         osg::Shader *reprojFragmentObj = new osg::Shader( osg::Shader::FRAGMENT );
         reprojFragmentObj->setShaderSource(reprojFrag);
         cd.reprojAdaptProgram->addShader(reprojFragmentObj);
      }

      {
          cd.reprojMeshProgram = new osg::Program;
          osg::Shader *reprojVertexObj = new osg::Shader( osg::Shader::VERTEX );
          reprojVertexObj->setShaderSource(reprojMeshVert);
          cd.reprojMeshProgram->addShader(reprojVertexObj);

          osg::Shader *reprojGeoObj = new osg::Shader( osg::Shader::GEOMETRY );
          reprojGeoObj->setShaderSource(reprojMeshGeo);
          cd.reprojMeshProgram->addShader(reprojGeoObj);
          cd.reprojMeshProgram->setParameter( GL_GEOMETRY_VERTICES_OUT_EXT, 4 );
          cd.reprojMeshProgram->setParameter( GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS );
          cd.reprojMeshProgram->setParameter( GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP );

          osg::Shader *reprojFragmentObj = new osg::Shader( osg::Shader::FRAGMENT );
          reprojFragmentObj->setShaderSource(reprojFrag);
          cd.reprojMeshProgram->addShader(reprojFragmentObj);
      }

      cd.reprojGeo->setStateSet(stateSet);
   }

   cd.geode = geometryNode;
   cd.geode->setNodeMask(cd.geode->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));
}


void MultiChannelDrawer::initChannelData(ChannelData &cd) {

   cd.camera = coVRConfig::instance()->channels[cd.channelNum].camera;

   cd.colorTex = new osg::TextureRectangle;
   osg::Image *cimg = new osg::Image();
   cd.colorTex->setImage(new osg::Image());
   cimg->setPixelBufferObject(new osg::PixelBufferObject(cimg));
   cd.colorTex->setInternalFormat( GL_RGBA );
   cd.colorTex->setBorderWidth( 0 );
   cd.colorTex->setFilter( osg::Texture::MIN_FILTER, osg::Texture::NEAREST );
   cd.colorTex->setFilter( osg::Texture::MAG_FILTER, osg::Texture::NEAREST );

   cd.depthTex = new osg::TextureRectangle;
   osg::Image *dimg = new osg::Image();
   cd.depthTex->setImage(dimg);
   dimg->setPixelBufferObject(new osg::PixelBufferObject(dimg));
   cd.depthTex->setInternalFormat( GL_DEPTH_COMPONENT32F );
   cd.depthTex->setBorderWidth( 0 );
   cd.depthTex->setFilter( osg::Texture::MIN_FILTER, osg::Texture::NEAREST );
   cd.depthTex->setFilter( osg::Texture::MAG_FILTER, osg::Texture::NEAREST );

   createGeometry(cd);

   //std::cout << "vp: " << vp->width() << "," << vp->height() << std::endl;

   osg::MatrixTransform *imageMat = new osg::MatrixTransform();
   imageMat->setMatrix(osg::Matrix::identity());
   imageMat->addChild(cd.geode);
   imageMat->setName("VncClient_imageMat");

   cd.scene = imageMat;
}

void MultiChannelDrawer::clearChannelData() {

    for (size_t view=0; view<m_channelData.size(); ++view) {
       ChannelData &cd = m_channelData[view];

       osg::Image *depth =  cd.depthTex->getImage();
       memset(depth->data(), 0xff, depth->getTotalSizeInBytes());
       depth->dirty();

       osg::Image *color =  cd.colorTex->getImage();
       memset(color->data(), 0, color->getTotalSizeInBytes());
       color->dirty();
    }
}

void MultiChannelDrawer::swapFrame() {
   for (size_t s=0; s<m_channelData.size(); ++s) {
      ChannelData &cd = m_channelData[s];

      cd.curView = cd.newView;
      cd.curProj = cd.newProj;
      cd.curModel = cd.newModel;

      cd.depthTex->getImage()->dirty();
      cd.colorTex->getImage()->dirty();
   }
}

void MultiChannelDrawer::updateMatrices(int idx, const osg::Matrix &model, const osg::Matrix &view, const osg::Matrix &proj) {

   ChannelData &cd = m_channelData[idx];
   cd.newModel = model;
   cd.newView = view;
   cd.newProj = proj;
}

void MultiChannelDrawer::resizeView(int idx, int w, int h, GLenum depthFormat) {
    ChannelData &cd = m_channelData[idx];
    osg::Image *cimg = cd.colorTex->getImage();
    if (cimg->s() != w || cimg->t() != h) {
        cimg->allocateImage(w, h, 1, GL_RGBA, GL_UNSIGNED_BYTE);

        if (m_flipped) {
            (*cd.texcoord)[0].set(0., h);
            (*cd.texcoord)[1].set(w, h);
            (*cd.texcoord)[2].set(w, 0.);
            (*cd.texcoord)[3].set(0., 0.);
        } else {
            (*cd.texcoord)[0].set(0., 0.);
            (*cd.texcoord)[1].set(w, 0.);
            (*cd.texcoord)[2].set(w, h);
            (*cd.texcoord)[3].set(0., h);
        }
        cd.fixedGeo->setTexCoordArray(0, cd.texcoord);
    }

    osg::Image *dimg = cd.depthTex->getImage();
    if (depthFormat != 0) {
        if (dimg->s() != w || dimg->t() != h || dimg->getDataType() != depthFormat) {
            dimg->allocateImage(w, h, 1, GL_DEPTH_COMPONENT, depthFormat);

            osg::Geometry *geo = cd.reprojGeo;
#ifdef INSTANCED
            if (geo->getNumPrimitiveSets() > 0) {
                geo->setPrimitiveSet(0, new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, 1, w*h));
            } else {
                geo->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, 1, w*h));
            }
#else
            cd.pointCoord->resizeArray(w*h);
            for (int y=0; y<h; ++y) {
                for (int x=0; x<w; ++x) {
                    (*cd.pointCoord)[y*w+x].set(x+0.5f, m_flipped ? y+0.5f : h-y+0.5f);
                }
            }
            cd.pointCoord->dirty();
            cd.pointArr = new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, w*h);

            cd.quadCoord->resizeArray((w-1)*(h-1));
            size_t idx=0;
            for (int y=0; y<h-1; ++y) {
                for (int x=0; x<w-1; ++x) {
                    (*cd.quadCoord)[idx++].set(x+0.5f, m_flipped ? y+0.5f : h-y+0.5f);
                }
            }
            cd.quadCoord->dirty();
            cd.quadArr = new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, (w-1)*(h-1));

            osg::ref_ptr<osg::DrawArrays> arr = (m_mode==ReprojectMesh||m_mode==ReprojectMeshWithHoles) ? cd.quadArr : cd.pointArr;

            if (geo->getNumPrimitiveSets() > 0) {
                geo->setPrimitiveSet(0, arr);
            } else {
                geo->addPrimitiveSet(arr);
            }
            geo->dirtyDisplayList();
#endif
            cd.size->set(osg::Vec2(w, h));
            cd.pixelOffset->set(osg::Vec2((w+1)%2*0.5f, (h+1)%2*0.5f));
        }
    }
}

void MultiChannelDrawer::reproject(int idx, const osg::Matrix &model, const osg::Matrix &view, const osg::Matrix &proj) {

    ChannelData &cd = m_channelData[idx];

    osg::Matrix cur = model * view * proj;
    osg::Matrix old = cd.curModel * cd.curView * cd.curProj;
    osg::Matrix oldInv = osg::Matrix::inverse(old);
    osg::Matrix reproj = oldInv * cur;
    cd.reprojMat->set(reproj);
}

unsigned char *MultiChannelDrawer::rgba(int idx) const {
    const ChannelData &cd = m_channelData[idx];
    return cd.colorTex->getImage()->data();
}

unsigned char *MultiChannelDrawer::depth(int idx) const {
    const ChannelData &cd = m_channelData[idx];
    return cd.depthTex->getImage()->data();
}

MultiChannelDrawer::Mode MultiChannelDrawer::mode() const {

    return m_mode;
}

void MultiChannelDrawer::setMode(MultiChannelDrawer::Mode mode) {

   m_mode = mode;
   for (size_t i=0; i<m_channelData.size(); ++i) {
       ChannelData &cd = m_channelData[i];
       osg::StateSet *state = cd.reprojGeo->getStateSet();
       assert(state);

       cd.geode->removeDrawable(cd.fixedGeo);
       cd.geode->removeDrawable(cd.reprojGeo);

       if (mode == AsIs) {
           cd.geode->addDrawable(cd.fixedGeo);
       } else {
           cd.geode->addDrawable(cd.reprojGeo);
       }

       switch(mode) {
           case AsIs:
               break;
           case Reproject:
               cd.reprojGeo->setVertexArray(cd.pointCoord);
               cd.reprojGeo->setPrimitiveSet(0, cd.pointArr);
               state->setMode(GL_PROGRAM_POINT_SIZE, osg::StateAttribute::OFF);
               state->setAttributeAndModes(cd.reprojAdaptProgram, osg::StateAttribute::OFF);
               state->setAttributeAndModes(cd.reprojMeshProgram, osg::StateAttribute::OFF);
               state->setAttributeAndModes(cd.reprojConstProgram, osg::StateAttribute::ON);
               break;
           case ReprojectAdaptive:
           case ReprojectAdaptiveWithNeighbors:
               cd.reprojGeo->setVertexArray(cd.pointCoord);
               cd.reprojGeo->setPrimitiveSet(0, cd.pointArr);
               state->setMode(GL_PROGRAM_POINT_SIZE, osg::StateAttribute::ON);
               state->setAttributeAndModes(cd.reprojConstProgram, osg::StateAttribute::OFF);
               state->setAttributeAndModes(cd.reprojMeshProgram, osg::StateAttribute::OFF);
               state->setAttributeAndModes(cd.reprojAdaptProgram, osg::StateAttribute::ON);
               cd.withNeighbors->set(mode == ReprojectAdaptiveWithNeighbors ? true : false);
               break;
           case ReprojectMesh:
           case ReprojectMeshWithHoles:
               cd.reprojGeo->setVertexArray(cd.quadCoord);
               cd.reprojGeo->setPrimitiveSet(0, cd.quadArr);
               state->setMode(GL_PROGRAM_POINT_SIZE, osg::StateAttribute::OFF);
               state->setAttributeAndModes(cd.reprojConstProgram, osg::StateAttribute::OFF);
               state->setAttributeAndModes(cd.reprojAdaptProgram, osg::StateAttribute::OFF);
               state->setAttributeAndModes(cd.reprojMeshProgram, osg::StateAttribute::ON);
               cd.withHoles->set(mode == ReprojectMeshWithHoles ? true : false);
               break;
       }
   }
}

}
