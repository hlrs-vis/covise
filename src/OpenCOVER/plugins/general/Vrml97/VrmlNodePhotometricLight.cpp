/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodePhotometricLight.cpp
#ifdef _WIN32
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <winsock2.h>
#include <windows.h>
#endif
#include <util/common.h>
#include <vrml97/vrml/config.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/coEventQueue.h>

#include <vrml97/vrml/MathUtils.h>
#include <vrml97/vrml/System.h>
#include <vrml97/vrml/Viewer.h>
#include <vrml97/vrml/VrmlScene.h>
#include <cover/VRViewer.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <math.h>


#include <osg/Texture3D>
#include <util/byteswap.h>
#include <plugins/general/Vrml97/coMLB.h>

#include "VrmlNodePhotometricLight.h"

#ifdef HAVE_VRMLNODEPHOTOMETRICLIGHT

#include "ViewerOsg.h"
#include <osg/Quat>
#include <osgDB/ReadFile>
#include <cover/coVRFileManager.h>
#include <osg/GLExtensions>
#include <osg/BindImageTexture>


// static initializations
std::list<VrmlNodePhotometricLight *> VrmlNodePhotometricLight::allPhotometricLights;
osg::ref_ptr<osg::Uniform> VrmlNodePhotometricLight::photometricLightMatrix;

void VrmlNodePhotometricLight::updateAll()
{
	for (std::list<VrmlNodePhotometricLight *>::iterator it = allPhotometricLights.begin(); it != allPhotometricLights.end(); it++)
	{
		(*it)->update();
	}
}
void VrmlNodePhotometricLight::update()
{
	osg::MatrixList worldMatrices = lightNodeInSceneGraph->getWorldMatrices();
	osg::Matrixf firstMat = worldMatrices[0];
	// photometricLightMatrix->setElement(d_lightNumber.get(), firstMat); 
	osg::Matrixf invFirstMat;
	if (invFirstMat.invert_4x4(firstMat))
	{
		photometricLightMatrix->setElement(d_lightNumber.get(), invFirstMat);
	}
}
void VrmlNodePhotometricLight::updateLightTextures(osg::RenderInfo &renderInfo)
{
	for (std::list<VrmlNodePhotometricLight *>::iterator it = allPhotometricLights.begin(); it != allPhotometricLights.end(); it++)
	{
		if ((*it)->coMLB_initialized)
			(*it)->updateLightTexture(renderInfo);
	}
}
void VrmlNodePhotometricLight::updateLightTexture(osg::RenderInfo &renderInfo)
{
    // https://github.com/openscenegraph/OpenSceneGraph/blob/master/examples/osgcomputeshaders/osgcomputeshaders.cpp
	//RenderInfo is no longer needed. At the moment we dont need the openGL context
	int numHorizontalAngles = mlbFile->header.t_width;
	int numVerticalAngles = mlbFile->header.t_height;
	int num_lights = mlbFile->header.t_depth;
	int work_group_size = 16;

	// we just assume for now someone changed the configuration
    int rand_max = RAND_MAX;
	if (((float)rand() / (rand_max)) < 0.05)
	{
		configuration_changed = true;
		for (int i = 0; i < num_lights; i++)
		{
			configuration_vec[i] = ((float)rand() / (rand_max));
		}
	}

	// conditional copute shader dispatch
	if (configuration_changed)
	{
		// update texture data
		float* data = reinterpret_cast<float*>(configuration_img->data());
		for (int i = 0; i < num_lights; i++)
			data[i] = configuration_vec[i];
		configuration_img->dirty();
		// launch compute shader. Remove the following line if you dont want to use the compute shader (=Version 1)
		comp_disp->setComputeGroups(numHorizontalAngles / work_group_size + 1, numVerticalAngles / work_group_size + 1, 1);
		configuration_changed = false;
	}
	else
		comp_disp->setComputeGroups(0, 0, 0);

}

void VrmlNodePhotometricLight::initFields(VrmlNodePhotometricLight *node, vrml::VrmlNodeType *t)
{
	VrmlNodeChild::initFields(node, t);
	initFieldsHelper(node, t, 
		exposedField("lightNumber", node->d_lightNumber),
		exposedField("MLBFile", node->d_MLBFile, [node](auto f){
			node->handleMLBFile();
		}),
		exposedField("IESFile", node->d_IESFile, [node](auto f){
			node->handleIESFile();
		}));

		static bool once = false;
        if(!once)
        {
			photometricLightMatrix =new osg::Uniform(osg::Uniform::FLOAT_MAT4, "photometricLightMatrix", MAX_LIGHTS);
			osg::ref_ptr<osg::StateSet> state = cover->getObjectsRoot()->getOrCreateStateSet();
			state->addUniform(photometricLightMatrix);
            once = true;
        }
}

const char *VrmlNodePhotometricLight::name()
{
	return "PhotometricLight";
}

VrmlNodePhotometricLight::VrmlNodePhotometricLight(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , d_lightNumber(0)
    , d_viewerObject(0)
	, d_MLBFile("")
	, d_IESFile("")
{
    setModified();
    lightNodeInSceneGraph = new osg::MatrixTransform();
    allPhotometricLights.push_back(this);
}

void VrmlNodePhotometricLight::addToScene(VrmlScene *s, const char *relUrl)
{
    (void)relUrl;
    d_scene = s;
    if (!s)
    {
        cerr << "no Scene" << endl;
    }

}

// need copy constructor for new markerName (each instance definitely needs a new marker Name) ...

VrmlNodePhotometricLight::VrmlNodePhotometricLight(const VrmlNodePhotometricLight &n)
    : VrmlNodeChild(n)
    , d_lightNumber(n.d_lightNumber)
    , d_viewerObject(n.d_viewerObject)
	, d_MLBFile(n.d_MLBFile)
	, d_IESFile(n.d_IESFile)
    , lightNodeInSceneGraph(n.lightNodeInSceneGraph)
{
    allPhotometricLights.push_back(this);
    setModified();
}

VrmlNodePhotometricLight::~VrmlNodePhotometricLight()
{
	// TODO: remove textures -> fixed
	/*
	Never use a regular C++ pointer variable for long-term storage of pointers
	to objects derived from Referenced.As an exception, you can use a
	regular C++ pointer variable temporarily, as long as the heap memory
	address is eventually stored in a ref_ptr<>.However, using a ref_ptr<>
	is always the safest approach.
	*/
    allPhotometricLights.remove(this);
}

VrmlNodePhotometricLight *VrmlNodePhotometricLight::toPhotometricLight() const
{
    return (VrmlNodePhotometricLight *)this;
}

void VrmlNodePhotometricLight::render(Viewer *viewer)
{
    if (!haveToRender())
        return;

    if (d_viewerObject && isModified())
    {
        viewer->removeObject(d_viewerObject);
        d_viewerObject = 0;
    }
    if (d_viewerObject)
    {
        viewer->insertReference(d_viewerObject);
    }
    d_viewerObject = viewer->beginObject(name(), 0, this);

    ((osgViewerObject *)d_viewerObject)->pNode = lightNodeInSceneGraph;
    ((ViewerOsg *)viewer)->addToScene((osgViewerObject *)d_viewerObject);

    viewer->endObject();

    clearModified();
}

// Set the value of one of the node fields.

void VrmlNodePhotometricLight::handleMLBFile()
{
// please use osg::ref_ptr rather than normal pointer.
	mlbFile = new coMLB(d_MLBFile.get());
	int numHorizontalAngles = mlbFile->header.t_width;
	int numVerticalAngles = mlbFile->header.t_height;
	int num_lights = mlbFile->header.t_depth;
	int numValues = numHorizontalAngles * numVerticalAngles;
	std::cout << "Tex. size: " << numHorizontalAngles << " x " << numVerticalAngles << std::endl;

	// load compute shader code
	std::string buf = "share/covise/materials/MatrixLight.comp";
	std::string code = "";
	const char *fn = coVRFileManager::instance()->getName(buf.c_str());
	std::string filename = fn;
	if (!filename.empty())
	{
		std::ifstream t(filename.c_str());
		std::stringstream buffer;
		buffer << t.rdbuf();
		code = buffer.str();
	}

	//set binding numbers in the compute shader code
	for (int binding_number = 0; binding_number < 3; binding_number++)
	{
		string from = std::string("binding=") + std::to_string(binding_number);
		string to = std::string("binding=") + std::to_string(binding_number + d_lightNumber.get() * 3);
		size_t start_pos = code.find(from);
		if (start_pos != std::string::npos)
			code.replace(start_pos, from.length(), to);
		std::cout << "changed binding nr from " << binding_number << " to "<< (binding_number + d_lightNumber.get() * 3) << std::endl;
	}
	std::cout << code << "\n";

	// matrix light data as texture3D
	osg::ref_ptr<osg::Texture3D> all_lights_tex = new osg::Texture3D();
	all_lights_tex->setInternalFormat(GL_R8);
	all_lights_tex->setSourceFormat(GL_RED);
	all_lights_tex->setSourceType(GL_UNSIGNED_BYTE);
	all_lights_tex->setResizeNonPowerOfTwoHint(false);
	all_lights_tex->setFilter(osg::Texture3D::MIN_FILTER, osg::Texture3D::NEAREST);
	all_lights_tex->setFilter(osg::Texture3D::MAG_FILTER, osg::Texture3D::NEAREST);
	all_lights_tex->setWrap(osg::Texture3D::WRAP_S, osg::Texture3D::CLAMP);
	all_lights_tex->setWrap(osg::Texture3D::WRAP_T, osg::Texture3D::CLAMP);
	all_lights_tex->setImage(mlbFile->getTexture()); // 		pixelFormat = GL_LUMINANCE; type = GL_UNSIGNED_BYTE

	//output texture: sum of all matrix lights
	osg::ref_ptr<osg::Texture2D> sum_lights_tex = new osg::Texture2D();
	sum_lights_tex->setTextureSize(numHorizontalAngles, numVerticalAngles);
	sum_lights_tex->setResizeNonPowerOfTwoHint(false);
	sum_lights_tex->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::NEAREST);
	sum_lights_tex->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::NEAREST);
	// https://www.khronos.org/registry/OpenGL-Refpages/es3.0/html/glTexImage2D.xhtml
	sum_lights_tex->setInternalFormat(GL_R16F);
	sum_lights_tex->setSourceFormat(GL_RED);
	sum_lights_tex->setSourceType(GL_FLOAT);

	//texture holding the configuration of the matrix lights (to be updated!)
	configuration_vec.resize(mlbFile->header.t_depth);
	std::fill(configuration_vec.begin(), configuration_vec.end(), 1.0);

	light_conf_tex = new osg::Texture2D;
	configuration_img = new osg::Image();
	configuration_img->allocateImage(num_lights, 1, 1, GL_RED, GL_FLOAT);  //  GLenum pixelFormat, GLenum type
	light_conf_tex->setInternalFormat(GL_R32F);
	light_conf_tex->setSourceFormat(GL_RED);
	light_conf_tex->setSourceType(GL_FLOAT);
	light_conf_tex->setResizeNonPowerOfTwoHint(false);
	light_conf_tex->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR);
	light_conf_tex->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
	light_conf_tex->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
	light_conf_tex->setWrap(osg::Texture::WRAP_T, osg::Texture::CLAMP_TO_EDGE);
	light_conf_tex->setImage(configuration_img.get());


	// The compute shader can't work with other kinds of shaders
	computeProg = new osg::Program;
	computeProg->addShader(new osg::Shader(osg::Shader::COMPUTE, code));
	// Create a node for outputting to the texture.
	comp_disp = new osg::DispatchCompute(0, 0, 0); // launch 0 work groups, wich disables the compute shader for now
	osg::ref_ptr<osg::Node> sourceNode = comp_disp;
	osg::ref_ptr<osg::StateSet> state = sourceNode->getOrCreateStateSet();
	sourceNode->setDataVariance(osg::Object::DYNAMIC);

	state->setAttributeAndModes(computeProg.get());  //get()); if otherNode is a ref_ptr, without if the node is a raw pointer. Don't use raw pointer!

	std::cout << "light number: " << d_lightNumber.get() << std::endl;
	state->addUniform(new osg::Uniform("configuration", (int)(0 + d_lightNumber.get() * 3)));
	state->addUniform(new osg::Uniform("targetTex", (int)(2 + d_lightNumber.get() * 3)));
	state->addUniform(new osg::Uniform("AllPhotometricLightsTexX", (int)(1 + d_lightNumber.get() * 3)));

	// textures are read only, images not. we need to bind the image of a texture to a binding point.
	osg::ref_ptr<osg::BindImageTexture> imagbinding1 = new osg::BindImageTexture((0 + d_lightNumber.get() * 3), light_conf_tex, osg::BindImageTexture::READ_ONLY, GL_R32F);
	osg::ref_ptr<osg::BindImageTexture> imagbinding2 = new osg::BindImageTexture((1 + d_lightNumber.get() * 3), all_lights_tex, osg::BindImageTexture::READ_ONLY, GL_R8); //, 0, GL_TRUE, 0
	osg::ref_ptr<osg::BindImageTexture> imagbinding3 = new osg::BindImageTexture((2 + d_lightNumber.get() * 3), sum_lights_tex, osg::BindImageTexture::WRITE_ONLY, GL_R16F);  // GLenum format = GL_RGBA8
	//https://stackoverflow.com/questions/17015132/compute-shader-not-modifying-3d-texture
	//<osg::GLExtensions>()->glBindImageTexture(0, 6, 0, /*layered=*/GL_TRUE, 0, GL_READ_WRITE, GL_R8);
	state->setTextureAttributeAndModes((0 + d_lightNumber.get() * 3), light_conf_tex, osg::StateAttribute::ON);
	state->setTextureAttributeAndModes((1 + d_lightNumber.get() * 3), all_lights_tex, osg::StateAttribute::ON);
	state->setTextureAttributeAndModes((2 + d_lightNumber.get() * 3), sum_lights_tex, osg::StateAttribute::ON);
	state->setAttributeAndModes(imagbinding1.get());
	state->setAttributeAndModes(imagbinding2.get());
	state->setAttributeAndModes(imagbinding3.get());
	
	// prepare state for all objects in the scene
	state = cover->getObjectsRoot()->getOrCreateStateSet();  // Object Root
	state->setTextureAttributeAndModes((5 + d_lightNumber.get() * 3), light_conf_tex, osg::StateAttribute::ON);  // needs to be done. otherwise the first frame after this texture changes is buggy
	state->setTextureAttributeAndModes((6 + d_lightNumber.get() * 3), all_lights_tex, osg::StateAttribute::ON);
	state->setTextureAttributeAndModes((7 + d_lightNumber.get() * 3), sum_lights_tex, osg::StateAttribute::ON);
	// only the targetTex is needed if you use compute shader. If you dont want to use it (=version 1) uncomment the following lines
	/*
	state->addUniform(new osg::Uniform("configuration0", (int)(5 + 0 * 3)));
	state->addUniform(new osg::Uniform("configuration1", (int)(5 + 1 * 3)));
	state->addUniform(new osg::Uniform("configuration2", (int)(5 + 2 * 3)));
	state->addUniform(new osg::Uniform("configuration3", (int)(5 + 3 * 3)));
	state->addUniform(new osg::Uniform("AllPhotometricLightsTex0", (int)(6 + 0 * 3)));
	state->addUniform(new osg::Uniform("AllPhotometricLightsTex1", (int)(6 + 1 * 3)));
	state->addUniform(new osg::Uniform("AllPhotometricLightsTex2", (int)(6 + 2 * 3)));
	state->addUniform(new osg::Uniform("AllPhotometricLightsTex3", (int)(6 + 3 * 3)));
	*/
	state->addUniform(new osg::Uniform("targetTex0", (int)(7 + 0 * 3)));
	state->addUniform(new osg::Uniform("targetTex1", (int)(7 + 1 * 3)));
	state->addUniform(new osg::Uniform("targetTex2", (int)(7 + 2 * 3)));
	state->addUniform(new osg::Uniform("targetTex3", (int)(7 + 3 * 3)));

	// the fragemnt shader needs to know the size of the light:
	// TODO: use vectors rather than a osg::vec4 if you want to have more than 4 matrix lights. Take care of the shader accordingly.
	if (d_lightNumber.get() >= 4) {
		throw std::invalid_argument("light number must be below 4!");
	}

	osg::Vec4f tmp;

	state->getOrCreateUniform(std::string("left"), osg::Uniform::FLOAT_VEC4)->get(tmp);
	tmp[d_lightNumber.get()] = (mlbFile->header.left);
	state->addUniform(new osg::Uniform("left", tmp));
	std::cout << "left = [" << tmp[0] << "\t" << tmp[1] << "\t" << tmp[2] << "\t" << tmp[3] << "]\n";

	state->getOrCreateUniform(std::string("bottom"), osg::Uniform::FLOAT_VEC4)->get(tmp);
	tmp[d_lightNumber.get()] = (mlbFile->header.bottom);
	state->addUniform(new osg::Uniform("bottom", tmp));
	std::cout << "bottom = [" << tmp[0] << "\t" << tmp[1] << "\t" << tmp[2] << "\t" << tmp[3] << "]\n";

	state->getOrCreateUniform(std::string("width"), osg::Uniform::FLOAT_VEC4)->get(tmp);
	tmp[d_lightNumber.get()] = (mlbFile->header.width);
	state->addUniform(new osg::Uniform("width", tmp));
	std::cout << "width = [" << tmp[0] << "\t" << tmp[1] << "\t" << tmp[2] << "\t" << tmp[3] << "]\n";

	state->getOrCreateUniform(std::string("height"), osg::Uniform::FLOAT_VEC4)->get(tmp);
	tmp[d_lightNumber.get()] = (mlbFile->header.height);
	state->addUniform(new osg::Uniform("height", tmp));
	std::cout << "height = [" << tmp[0] << "\t" << tmp[1] << "\t" << tmp[2] << "\t" << tmp[3] << "]\n";

	state->getOrCreateUniform(std::string("is_active"), osg::Uniform::FLOAT_VEC4)->get(tmp); // bool or int not implemented in osg v 
	if (tmp[d_lightNumber.get()] == 1.0) {
		throw std::invalid_argument("this light already exists");
	}
	tmp[d_lightNumber.get()] = 1.0;
	state->addUniform(new osg::Uniform("is_active", tmp));
	std::cout << "is_active = [" << tmp[0] << "\t" << tmp[1] << "\t" << tmp[2] << "\t" << tmp[3] << "]\n";

	state->addUniform(new osg::Uniform("MAX_LIGHTS", (MAX_LIGHTS)));

	// Create the scene graph and start the viewer
	cover->getScene()->addChild(sourceNode);
	
	coMLB_initialized = true;
}

void VrmlNodePhotometricLight::handleIESFile()
{
	std::cout << "IESFile" << std::endl;
	iesFile = new coIES(d_IESFile.get());
	osg::ref_ptr<osg::Texture2D> lightTexture = new osg::Texture2D();
	lightTexture->setResizeNonPowerOfTwoHint(false);
	lightTexture->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::NEAREST);
	lightTexture->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::NEAREST);
	lightTexture->setWrap(osg::Texture2D::WRAP_S, osg::Texture2D::CLAMP);
	lightTexture->setWrap(osg::Texture2D::WRAP_T, osg::Texture2D::CLAMP);
	lightTexture->setImage(iesFile->getTexture());
	osg::ref_ptr<osg::StateSet> state = cover->getObjectsRoot()->getOrCreateStateSet();

	state->setTextureAttributeAndModes(5+d_lightNumber.get(), lightTexture, osg::StateAttribute::ON);
}

#endif // HAVE_VRMLNODEPHOTOMETRICLIGHT
