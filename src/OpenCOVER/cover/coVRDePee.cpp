/*
  Steffen Frey
  Fachpraktikum Graphik-Programmierung 2007
  Institut fuer Visualisierung und Interaktive Systeme
  Universitaet Stuttgart
 */

//export OSG_NOTIFY_LEVEL=DEBUG_INFO

#include "coVRDePee.h"

#include <stdio.h>

#include <osg/GLExtensions>
#include <osg/Node>
#include <osg/MatrixTransform>
#include <osg/Projection>
#include <osg/Geode>

#include "coVRFileManager.h"

#include <osg/ShapeDrawable>
#include <osg/Geometry>
#include <assert.h>
#include <iostream>

#include <stdio.h>
#include <osg/Geometry>
#include <osg/Geode>
#include <osgDB/FileUtils>
#include <osgDB/fstream>

using namespace opencover;

coVRDePee::coVRDePee(osg::Group* parent, osg::Node* subgraph, unsigned width, unsigned height)
{
  _renderToFirst = false;

  _isSketchy =false;
  _isColored = false;
  _isEdgy = true;
  _isCrayon = false;

  _normalDepthMapProgram = createProgram("share/covise/shaders/depthpeel_normaldepthmap.vert","share/covise/shaders/depthpeel_normaldepthmap.frag");
  _colorMapProgram = createProgram("share/covise/shaders/depthpeel_colormap.vert","share/covise/shaders/depthpeel_colormap.frag" );
  _edgeMapProgram = createProgram("share/covise/shaders/depthpeel_edgemap.vert", "share/covise/shaders/depthpeel_edgemap.frag");

  _parent = new osg::Group;
  parent->addChild(_parent.get());
  _subgraph = subgraph;

  _width = width;
  _height = height;
  _texWidth = width;
  _texHeight = height;

  assert(parent);
  assert(subgraph);

  _fps = 0;
  _colorCamera = 0;

  _sketchy = new osg::Uniform("sketchy", false);
  _colored = new osg::Uniform("colored", false);
  _edgy = new osg::Uniform("edgy", true);
  _sketchiness = new osg::Uniform("sketchiness", (float) 1.0);

  _normalDepthMap0  = newColorTexture2D(_texWidth, _texHeight, 32);
  _normalDepthMap1  = newColorTexture2D(_texWidth, _texHeight, 32);
  _edgeMap = newColorTexture2D(_texWidth, _texHeight, 8);
  _colorMap = newColorTexture2D(_texWidth, _texHeight, 8);

  //create a noise map...this doesn't end up in a new rendering pass
  (void) createMap(NOISE_MAP);

  //the viewport aligned quad
  _quadGeode = getCanvasQuad(_width, _height);


  //!!!Getting problems if assigning unit to texture in depth peeling subgraph and removing depth peeling steps!!!
  //That's why it is done here
  osg::StateSet* stateset = _parent->getOrCreateStateSet();
  stateset->setTextureAttributeAndModes(1, _normalDepthMap0.get(), osg::StateAttribute::ON);
  stateset->setTextureAttributeAndModes(2, _normalDepthMap1.get(), osg::StateAttribute::ON);
  stateset->setTextureAttributeAndModes(3, _edgeMap.get(), osg::StateAttribute::ON);
  stateset->setTextureAttributeAndModes(4, _colorMap.get(), osg::StateAttribute::ON);
  stateset->setTextureAttributeAndModes(5, _noiseMap.get(), osg::StateAttribute::ON);

  // render the final thing
  (void) createFinal();

    //take one step initially
  addcoVRDePeePass();

  //render head up display
  (void) createHUD();
}

coVRDePee::~coVRDePee()
{
}

void coVRDePee::setSketchy(bool sketchy)
  {
    _sketchy->set(sketchy);
    _isSketchy = sketchy;
  }

void coVRDePee::setCrayon(bool crayon)
{
  if(_isCrayon != crayon)
    {
      _isCrayon = crayon;
      createMap(NOISE_MAP);
    }
}

void coVRDePee::setSketchiness(double sketchiness)
  {
    _sketchiness->set((float)sketchiness);
  }

void coVRDePee::setColored(bool colored)
{
  if(colored == !_isColored)
    {
      if(colored)
	{
	  (void) createMap(COLOR_MAP, false);
	}
      else
	{
	  _coVRDePeePasses.back()->remRenderPass(COLOR_MAP);
	}
      _colored->set(colored);
      _isColored = colored;
    }
}

void coVRDePee::setEdgy(bool edgy)
{

  if(edgy != _isEdgy)
    {

      _isEdgy = edgy;
      unsigned int n = 0;
      while(remcoVRDePeePass())
	{
	  ++n;
	}

      if(edgy)
	{
	  (void) createMap(EDGE_MAP,_coVRDePeePasses.size() == 1);
	}
      else
	{
	  _coVRDePeePasses.back()->remRenderPass(EDGE_MAP);
	}

      for(unsigned int i=0; i < n; i++)
	{
	  addcoVRDePeePass();
	}
    }
  _edgy->set(edgy);
}



void coVRDePee::setFPS(double* fps)
{
  _fps = fps;
}

unsigned int coVRDePee::getNumberOfRenderPasses()
{
  unsigned int n = 0;
  for(unsigned int i=0; i < _coVRDePeePasses.size();i++)
    n += _coVRDePeePasses.at(i)->Cameras.size();
  // add one pass for final rendering pass and one for hud
  return n+2;
}

bool coVRDePee::addcoVRDePeePass()
{

  if(_isColored)
    {
      //remove previous color pass
      _coVRDePeePasses.back()->remRenderPass(COLOR_MAP);
    }

  _coVRDePeePasses.push_back(new coVRDePeePass());
  _parent->addChild(_coVRDePeePasses.back()->root.get());

  //need to create a depth map in every case
  (void) createMap(NORMAL_DEPTH_MAP, _coVRDePeePasses.size() == 1);

  if(_isEdgy)
    {
      (void) createMap(EDGE_MAP,_coVRDePeePasses.size() == 1);
    }

  if(_isColored)
    {
      (void) createMap(COLOR_MAP, false);
    }

  return true;
}

bool coVRDePee::remcoVRDePeePass()
{
  if(_coVRDePeePasses.size() < 2)
    return false;

  _parent->removeChild(_coVRDePeePasses.back()->root.get());
  delete _coVRDePeePasses.back();
  _coVRDePeePasses.pop_back();

  _renderToFirst = !_renderToFirst;

  if(_isColored)
    {
      (void) createMap(COLOR_MAP, false);
    }

  return true;
}


//create noise map with values ranging from 0 to 255
bool coVRDePee::createNoiseMap()
{
  {
    osg::StateSet* stateset = _parent->getOrCreateStateSet();
    _noiseMap = new osg::Texture2D;
    _noiseMap->setTextureSize(_width, _height);
    _noiseMap->setFilter(osg::Texture2D::MIN_FILTER,osg::Texture2D::LINEAR);
    _noiseMap->setFilter(osg::Texture2D::MAG_FILTER,osg::Texture2D::LINEAR);
    stateset->setTextureAttributeAndModes(5, _noiseMap.get(), osg::StateAttribute::ON);
  }

  osg::Image* image = new osg::Image;
  unsigned char* data = new unsigned char[_width*_height];
  unsigned char* tmpData = new unsigned char[_width*_height];

  int random=rand() % 5000;
  for(unsigned y=0; y < _height; y++)
    for(unsigned x=0; x < _width; x++)
      data[y*_width + x] = (unsigned char) (0.5 * 255.0 + getNoise(x, y, random) * 0.5 * 255.0);

  //if style isn't crayon style, smooth the noise map
  if(!_isCrayon)
    {
      for(unsigned i=0; i < 4; i++)
        {
          for(unsigned y=0; y < _height; y++)
            for(unsigned x=0; x < _width; x++)
              tmpData[y*_width + x] = (unsigned char)smoothNoise(_width, _height,x,y, data);

          for(unsigned y=0; y < _height; y++)
            for(unsigned x=0; x < _width; x++)
              data[y*_width + x] = (unsigned char)smoothNoise(_width, _height, x,y, tmpData);
        }
    }

  image->setImage(_width, _height, 1,
		  1, GL_LUMINANCE, GL_UNSIGNED_BYTE,
		  data,
		  osg::Image::USE_NEW_DELETE);
  _noiseMap->setImage(image);
  return true;
}

bool coVRDePee::createHUD()
{
    osg::Geode* geode = new osg::Geode();

    std::string timesFont("fonts/arial.ttf");

    // turn lighting off for the text and disable depth test to ensure its always ontop.
    osg::StateSet* stateset = geode->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    stateset->setTextureAttributeAndModes(1, _normalDepthMap0.get(), osg::StateAttribute::OFF);
    stateset->setTextureAttributeAndModes(2, _normalDepthMap1.get(), osg::StateAttribute::OFF);
    stateset->setTextureAttributeAndModes(3, _edgeMap.get(), osg::StateAttribute::OFF);
    stateset->setTextureAttributeAndModes(4, _colorMap.get(), osg::StateAttribute::OFF);
    stateset->setTextureAttributeAndModes(5, _noiseMap.get(), osg::StateAttribute::OFF);

    osg::Vec3 position(5.0f,7.0f,0.0f);
    osg::Vec3 delta(0.0f,-120.0f,0.0f);

    _hudText = new  osgText::Text;

    {
      geode->addDrawable( _hudText );
      _hudText->setFont(timesFont);
      _hudText->setPosition(position);
      _hudText->setText("Head Up Display");
      _hudText->setColor(osg::Vec4(0.5, 0.5, 0.5, 1.0));
      _hudText->setCharacterSize(20.0);
      position += delta;
    }

    {
      osg::BoundingBox bb;
      for(unsigned int i=0;i<geode->getNumDrawables();++i)
        {
	  bb.expandBy(geode->getDrawable(i)->getBoundingBox());
        }

      osg::Geometry* geom = new osg::Geometry;

      osg::Vec3Array* vertices = new osg::Vec3Array;
      float depth = bb.zMin()-0.1;
      vertices->push_back(osg::Vec3(bb.xMin(),bb.yMax(),depth));
      vertices->push_back(osg::Vec3(bb.xMin(),bb.yMin(),depth));
      vertices->push_back(osg::Vec3(bb.xMax(),bb.yMin(),depth));
      vertices->push_back(osg::Vec3(bb.xMax(),bb.yMax(),depth));
      geom->setVertexArray(vertices);

      osg::Vec3Array* normals = new osg::Vec3Array;
      normals->push_back(osg::Vec3(0.0f,0.0f,1.0f));
      geom->setNormalArray(normals, osg::Array::BIND_OVERALL);

      osg::Vec4Array* colors = new osg::Vec4Array;
      colors->push_back(osg::Vec4(0.0f,0.0,0.0f,0.3f));
      geom->setColorArray(colors, osg::Array::BIND_OVERALL);

      geom->addPrimitiveSet(new osg::DrawArrays(GL_QUADS,0,4));

      osg::StateSet* stateset = geom->getOrCreateStateSet();
      stateset->setMode(GL_BLEND,osg::StateAttribute::ON);

      stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

      //geode->addDrawable(geom);
    }

    osg::Camera* camera = new osg::Camera;

    // set the projection matrix
    camera->setProjectionMatrix(osg::Matrix::ortho2D(0,1280,0,1024));

    // set the view matrix
    camera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    camera->setViewMatrix(osg::Matrix::identity());

    // only clear the depth buffer
    camera->setClearMask(GL_DEPTH_BUFFER_BIT);

    // draw subgraph after main camera view.
    camera->setRenderOrder(osg::Camera::POST_RENDER);

    camera->addChild(geode);

    _parent->addChild(camera);

    return true;

}


// then create the first camera node to do the render to texture
// render normal and depth map color map

bool coVRDePee::createMap(MapMode mapMode, bool first)
  {
    switch(mapMode)
      {
      case EDGE_MAP:
	return createEdgeMap(first);
      case NOISE_MAP:
	return createNoiseMap();
      case NORMAL_DEPTH_MAP:
      case COLOR_MAP:
	return createNormalDepthColorMap(mapMode, first);
      default:
	std::cerr << "mapMode not recognized!!!\n";
	return false;
      }
  }

bool coVRDePee::createFinal()
{
  osg::Projection* screenAlignedProjectionMatrix = new osg::Projection;

  screenAlignedProjectionMatrix->setMatrix(osg::Matrix::ortho2D(0,_width,0,_height));
  screenAlignedProjectionMatrix->setCullingActive(false);

  osg::MatrixTransform* screenAlignedModelViewMatrix = new osg::MatrixTransform;
  screenAlignedModelViewMatrix->setMatrix(osg::Matrix::identity());

  // Make sure the model view matrix is not affected by any transforms
  // above it in the scene graph:
  screenAlignedModelViewMatrix->setReferenceFrame(osg::Transform::ABSOLUTE_RF);


  // new we need to add the texture to the Drawable, we do so by creating a
  // StateSet to contain the Texture StateAttribute.
  osg::StateSet* stateset = new osg::StateSet;

  stateset->setMode(GL_DEPTH_TEST,osg::StateAttribute::OFF);

  _quadGeode->setStateSet(stateset);

  _parent->addChild(screenAlignedProjectionMatrix);
  screenAlignedProjectionMatrix->addChild(screenAlignedModelViewMatrix);
  screenAlignedModelViewMatrix->addChild(_quadGeode.get());

  //setup shader
  std::string vertSource;
  if(!readFile("share/covise/shaders/depthpeel_final.vert", vertSource))
    {
      printf("shader source not found\n");
      return false;
    }

  std::string fragSource;
  if(!readFile("share/covise/shaders/depthpeel_final.frag", fragSource))
    {
      printf("shader source not found\n");
      return false;
    }

  osg::ref_ptr<osg::Program> program = new osg::Program;
  program->addShader( new osg::Shader( osg::Shader::VERTEX, vertSource.c_str() ) );
  program->addShader( new osg::Shader( osg::Shader::FRAGMENT, fragSource.c_str() ) );

  //choose map to display
  stateset->addUniform( new osg::Uniform("normalDepthMap0", 1));
  stateset->addUniform( new osg::Uniform("normalDepthMap1", 2));
  stateset->addUniform(new osg::Uniform("edgeMap", 3));
  stateset->addUniform( new osg::Uniform("colorMap", 4));
  stateset->addUniform( new osg::Uniform("noiseMap", 5));

  stateset->addUniform(_sketchy);
  stateset->addUniform(_colored);
  stateset->addUniform(_edgy);
  stateset->addUniform(_sketchiness);

  stateset->setAttributeAndModes( program.get(), osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);

  //switch lighting off
  stateset->setMode(GL_LIGHTING,osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
  return true;
}

bool coVRDePee::createEdgeMap(bool first)
//create the edge map of the first normal and depth map
{
  _coVRDePeePasses.back()->newRenderPass(EDGE_MAP);


  // set up the background color and clear mask.
  _coVRDePeePasses.back()->Cameras[EDGE_MAP]->setClearColor(osg::Vec4(0.3,0.3f,0.3f,1.0f));
  _coVRDePeePasses.back()->Cameras[EDGE_MAP]->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  const osg::BoundingSphere& bs = _quadGeode->getBound();
  if (!bs.valid())
    {
      return false;
    }

  float znear = 1.0f*bs.radius();
  float zfar  = 3.0f*bs.radius();

  znear *= 0.9f;
  zfar *= 1.1f;


  // set up projection.
  //_coVRDePeePasses.back()->Cameras.top()->setProjectionMatrixAsFrustum(-proj_right,proj_right,-proj_top,proj_top,znear,zfar);
  _coVRDePeePasses.back()->Cameras[EDGE_MAP]->setProjectionMatrixAsOrtho(0,_width,0,_height,znear,zfar);

  //set view
  _coVRDePeePasses.back()->Cameras[EDGE_MAP]->setReferenceFrame(osg::Transform::ABSOLUTE_RF);

  _coVRDePeePasses.back()->Cameras[EDGE_MAP]->setViewMatrixAsLookAt(osg::Vec3(0.0f,0.0f,2.0f)*bs.radius(), osg::Vec3(0.0,0.0,0.0),osg::Vec3(0.0f,1.0f,0.0f));

  // set viewport
  _coVRDePeePasses.back()->Cameras[EDGE_MAP]->setViewport(0,0,_texWidth,_texHeight);

  // set the camera to render before the main camera.
  _coVRDePeePasses.back()->Cameras[EDGE_MAP]->setRenderOrder(osg::Camera::PRE_RENDER);

  // tell the camera to use OpenGL frame buffer object
  _coVRDePeePasses.back()->Cameras[EDGE_MAP]->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER);

  //switch lighting off
  osg::ref_ptr<osg::StateSet> stateset = new osg::StateSet;


  if(_renderToFirst)
    {
      stateset->addUniform(new osg::Uniform("normalDepthMap", 1));
    }
  else
    {
      stateset->addUniform(new osg::Uniform("normalDepthMap", 2));
    }

  _coVRDePeePasses.back()->Cameras[EDGE_MAP]->attach(osg::Camera::COLOR_BUFFER, _edgeMap.get());
  stateset->addUniform( new osg::Uniform("edgeMap", 3));

  stateset->setMode(GL_LIGHTING,osg::StateAttribute::OVERRIDE |
              osg::StateAttribute::OFF);
  //setup shader
  stateset->setAttributeAndModes(_edgeMapProgram.get(), osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
  stateset->addUniform(new osg::Uniform("width", (float) _width));
  stateset->addUniform(new osg::Uniform("height", (float) _height));

  if(first)
    {
      stateset->addUniform(new osg::Uniform("first", (float)1.0));
    }
  else
    {
      stateset->addUniform(new osg::Uniform("first", (float)0.0));
    }
  _coVRDePeePasses.back()->settingNodes[EDGE_MAP]->setStateSet(stateset.get());

  // add subgraph to render
  assert(_coVRDePeePasses.size() > 0);

  _coVRDePeePasses.back()->settingNodes[EDGE_MAP]->addChild(_quadGeode.get());

  return true;
}


bool coVRDePee::createNormalDepthColorMap(MapMode mapMode, bool first)
{
  coVRDePeePass* pass;

  pass = _coVRDePeePasses.back();

  pass->newRenderPass(mapMode);

  //
  // setup camera
  //

  // set up the background color and clear mask
  pass->Cameras[mapMode]->setClearColor(osg::Vec4(0.f,0.f,1.f,1.f));
  pass->Cameras[mapMode]->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  const osg::BoundingSphere& bs = _subgraph->getBound();
  if (!bs.valid())
    {
      return false;
    }

  float znear = 1.0f*bs.radius();
  float zfar  = 3.0f*bs.radius();

  // 2:1 aspect ratio as per flag geometry below.
  float projTop   = 0.25f*znear;
  float projRight = projTop * ((double)_width/(double)_height);

  znear *= 0.9f;
  zfar *= 1.1f;

  // set up projection.
  pass->Cameras[mapMode]->setProjectionMatrixAsFrustum(-projRight,projRight,-projTop,projTop, znear,zfar);

  // setup view
  pass->Cameras[mapMode]->setReferenceFrame(osg::Transform::ABSOLUTE_RF);

  pass->Cameras[mapMode]->setViewMatrixAsLookAt(bs.center()-osg::Vec3(0.0f,2.0f,0.0f)*bs.radius(),
							       bs.center(),
							       osg::Vec3(0.0f,0.0f,1.0f));
  // set viewport
  pass->Cameras[mapMode]->setViewport(0,0,_texWidth,_texHeight);

  // set the camera to render before the main camera.
  pass->Cameras[mapMode]->setRenderOrder(osg::Camera::PRE_RENDER);

  pass->Cameras[mapMode]->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);

  //
  // setup stateset
  //
  //switch lighting off
  osg::ref_ptr<osg::StateSet> stateset = new osg::StateSet;

  stateset->setMode(GL_LIGHTING,osg::StateAttribute::OVERRIDE |
		    osg::StateAttribute::OFF);

    switch(mapMode)
      {
      case NORMAL_DEPTH_MAP:

	_renderToFirst = !_renderToFirst;

	if(_renderToFirst)
	  {
	    pass->Cameras[mapMode]->attach(osg::Camera::COLOR_BUFFER, _normalDepthMap0.get());
	    stateset->addUniform(new osg::Uniform("normalDepthMap", 2));
	  }
	else
	  {
	    pass->Cameras[mapMode]->attach(osg::Camera::COLOR_BUFFER, _normalDepthMap1.get());
	    stateset->addUniform(new osg::Uniform("normalDepthMap", 1));
	  }

	stateset->setMode(GL_LIGHTING,osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
        stateset->setAttributeAndModes(_normalDepthMapProgram.get(), osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
	break;

      case COLOR_MAP:

        assert(pass == _coVRDePeePasses.back());
        pass->Cameras[mapMode]->attach(osg::Camera::COLOR_BUFFER, _colorMap.get());


	if(_renderToFirst)
	  {
	    stateset->addUniform(new osg::Uniform("normalDepthMap", 1));
	  }
	else
	  {
	    stateset->addUniform(new osg::Uniform("normalDepthMap", 2));
	  }
	pass->Cameras[mapMode]->setClearColor(osg::Vec4(1.f,1.f,1.f,1.f));
	stateset->setMode(GL_LIGHTING,osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
	stateset->setMode(GL_BLEND, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);
	stateset->setAttributeAndModes(_colorMapProgram.get(), osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
	stateset->addUniform(new osg::Uniform("tex", 0));

	break;
      default:
	return false;
      };

    // add subgraph to render

    pass->settingNodes[mapMode]->addChild(_subgraph.get());

    stateset->addUniform(new osg::Uniform("first", first));

    stateset->addUniform(new osg::Uniform("width", (float) _width));
    stateset->addUniform(new osg::Uniform("height", (float) _height));
    stateset->addUniform(new osg::Uniform("znear", znear));
    stateset->addUniform(new osg::Uniform("zfar", zfar));


    pass->settingNodes[mapMode]->setStateSet(stateset.get());

    return true;
}


bool coVRDePee::updateHUDText()
{
  if(!_fps)
    return false;
  std::string str;
  std::string tmp = coVRDePee::toString(*_fps);
  unsigned i = tmp.find_first_of('.');
  tmp = tmp.substr(0, i + 3);
  _hudText->setText(coVRDePee::toString(_coVRDePeePasses.size())
		    + " Depth Peeling Pass" + (_coVRDePeePasses.size() == 1 ? " " : "es ")
		    + "((a)dd; (r)emove) "
		    + (_isEdgy ? "+" : "-") + "(E)dgy " +
		    + (_isSketchy ? "+" : "-") + "(S)ketchy " +
		    + (_isColored ? "+" : "-") + "(C)olored " +
		    + "-> "+coVRDePee::toString(getNumberOfRenderPasses())+ " Rendering Passes "
		    + "@ "
		    + tmp + " fps");
  return true;
}


bool coVRDePee::readFile(const char* fName, std::string& s)
{
  const char *file = coVRFileManager::instance()->getName(fName);
  if (file == NULL) return false;
  std::string foundFile = file;

  osgDB::ifstream is;//(fName);
  is.open(foundFile.c_str());
  if (is.fail())
    {
      std::cerr << "Could not open " << fName << " for reading.\n";
      return false;
    }
  char ch = is.get();
  while (!is.eof())
    {
      s += ch;
      ch = is.get();
    }
  is.close();
  return true;
}

std::string coVRDePee::toString(double d)
{
  std::stringstream ostr;
  ostr << d;
  return ostr.str();
}

osg::Program* coVRDePee::createProgram(std::string vs, std::string fs)
{
  //setup shader
  std::string vertSource;
  if(!readFile((char*)vs.c_str(), vertSource))
    {
      printf("shader source not found\n");
      return 0;
    }

  std::string fragSource;
  if(!readFile((char*)fs.c_str(), fragSource))
    {
      printf("shader source not found\n");
      return 0;
    }


  osg::Program* program = new osg::Program;
  program->addShader( new osg::Shader( osg::Shader::VERTEX, vertSource.c_str() ) );
  program->addShader( new osg::Shader( osg::Shader::FRAGMENT, fragSource.c_str() ) );
  return program;
}

double coVRDePee::getNoise(unsigned x, unsigned y, unsigned random)
{
  int n = x + y * 57 + random * 131;
  n = (n<<13) ^ n;
  double noise = (1.0f - ( (n * (n * n * 15731 + 789221) +
                1376312589)&0x7fffffff)* 0.000000000931322574615478515625f);
  return noise;
}

double coVRDePee::smoothNoise(unsigned width, unsigned height, unsigned x, unsigned y, unsigned char* noise)
{
  assert(noise);

  if(x==0 || x > width -2
     || y==0 || y > height -2)
    return noise[x + y*width];

  double corners = (noise[x-1 + (y-1) *width]
            +noise[x+1 + (y-1)*width]
            +noise[x-1 + (y+1) * width]
            +noise[x+1 + (y+1) * width]) / 16.0;
  double sides   = (noise[x-1 + y*width]
            +noise[x+1 + y*width]
            +noise[x + (y-1)*width]
            +noise[x + (y+1)*width]) / 8.0;
  double center  =  noise[x + y*width] / 4.0;

  return corners + sides + center;
}

osg::Texture2D* coVRDePee::newColorTexture2D(unsigned width, unsigned height, unsigned accuracy)
{
  osg::Texture2D* texture2D = new osg::Texture2D;

  texture2D->setTextureSize(width, height);
  if(accuracy == 32)
    {
      texture2D->setInternalFormat(GL_RGBA32F_ARB);
      texture2D->setSourceFormat(GL_RGBA);
    }
  else if(accuracy == 8)
    {
      texture2D->setInternalFormat(GL_RGBA);
    }
  texture2D->setSourceType(GL_FLOAT);
  texture2D->setFilter(osg::Texture2D::MIN_FILTER,osg::Texture2D::LINEAR);
  texture2D->setFilter(osg::Texture2D::MAG_FILTER,osg::Texture2D::LINEAR);
  return texture2D;
}

osg::Geode* coVRDePee::getCanvasQuad(unsigned width, unsigned height, double depth)
{
  osg::Vec3Array* vertices = new osg::Vec3Array;
  osg::Vec2Array* texCoords = new osg::Vec2Array;
  vertices->push_back(osg::Vec3(0,0,depth));
  texCoords->push_back(osg::Vec2(0,0));

  vertices->push_back(osg::Vec3(width,0,depth));
  texCoords->push_back(osg::Vec2(1,0));

  vertices->push_back(osg::Vec3(0,height,depth));
  texCoords->push_back(osg::Vec2(0,1));

  vertices->push_back(osg::Vec3(width,height,depth));
  texCoords->push_back(osg::Vec2(1,1));

  osg::Geometry* quad = new osg::Geometry;
  quad->setVertexArray(vertices);
  quad->setTexCoordArray(1,texCoords);

  quad->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUAD_STRIP,0,vertices->size()));

  osg::Vec4Array* colors = new osg::Vec4Array;
  colors->push_back(osg::Vec4(1.0f,1.0f,1.0f,1.0f));
  quad->setColorArray(colors, osg::Array::BIND_OVERALL);

  osg::Geode* geode = new osg::Geode();
  geode->addDrawable(quad);

  return geode;

}
