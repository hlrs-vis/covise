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
#include <osg/BindImageTexture>

#include <osg/Version>
#if OSG_VERSION_GREATER_OR_EQUAL(3, 7, 0)
#include <osg/ComputeDispatch>
#else
#include <osg/DispatchCompute>
namespace osg { typedef DispatchCompute ComputeDispatch; }
#endif

#include <osg/Texture3D>
#include <util/byteswap.h>
#include <plugins/general/Vrml97/coMLB.h>

#include "VrmlNodePhotometricLight.h"
#include "ViewerOsg.h"
#include <osg/Quat>
#include <osgDB/ReadFile>
#include <cover/coVRFileManager.h>
#include <osg/GLExtensions>


// static initializations
std::list<VrmlNodePhotometricLight *> VrmlNodePhotometricLight::allPhotometricLights;
osg::ref_ptr<osg::Uniform> VrmlNodePhotometricLight::photometricLightMatrix;

// PhotometricLight factory.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodePhotometricLight(scene);
}

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
		(*it)->updateLightTexture(renderInfo);
	}
}
void VrmlNodePhotometricLight::updateLightTexture(osg::RenderInfo &renderInfo)
{
    // https://github.com/openscenegraph/OpenSceneGraph/blob/master/examples/osgcomputeshaders/osgcomputeshaders.cpp
	//std::cout << "preDraw: ";
	int numHorizontalAngles = mlbFile->header.t_width;
	int numVerticalAngles = mlbFile->header.t_height;
	int numValues = numHorizontalAngles * numVerticalAngles;
	int work_group_size = 16;

	GLuint n_cs = 8;   // TODO: where do we get this number from???????
	bool condition = true;
	if ( condition ) {
		renderInfo.getState()->get<osg::GLExtensions>()->glUseProgram(n_cs);
		renderInfo.getState()->get<osg::GLExtensions>()->glDispatchCompute(numHorizontalAngles / work_group_size + 1, numVerticalAngles / work_group_size + 1, 1);
	}
}


void osg::ComputeDispatch::drawImplementation(RenderInfo& renderInfo) const
{
	// usually this dispatches the compute shader. We dont want to do this here.
	// Eventually (!) want to dispatch it in preDraw()

}


// Define the built in VrmlNodeType:: "PhotometricLight" fields

VrmlNodeType *VrmlNodePhotometricLight::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("PhotometricLight", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class

    t->addExposedField("lightNumber", VrmlField::SFINT32);
	t->addExposedField("MLBFile", VrmlField::SFSTRING);
	t->addExposedField("IESFile", VrmlField::SFSTRING);
    static osg::Matrixf lightMatrices[MAX_LIGHTS];
    photometricLightMatrix =new osg::Uniform(osg::Uniform::FLOAT_MAT4, "photometricLightMatrix", MAX_LIGHTS);
    osg::StateSet *state = cover->getObjectsRoot()->getOrCreateStateSet();
    state->addUniform(photometricLightMatrix);

    return t;
}

VrmlNodeType *VrmlNodePhotometricLight::nodeType() const
{
    return defineType(0);
}

VrmlNodePhotometricLight::VrmlNodePhotometricLight(VrmlScene *scene)
    : VrmlNodeChild(scene)
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
    if (s)
    {
    }
    else
    {
        cerr << "no Scene" << endl;
    }
}

// need copy constructor for new markerName (each instance definitely needs a new marker Name) ...

VrmlNodePhotometricLight::VrmlNodePhotometricLight(const VrmlNodePhotometricLight &n)
    : VrmlNodeChild(n.d_scene)
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
    allPhotometricLights.remove(this);
}

VrmlNode *VrmlNodePhotometricLight::cloneMe() const
{
    return new VrmlNodePhotometricLight(*this);
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

ostream &VrmlNodePhotometricLight::printFields(ostream &os, int indent)
{
    if (!d_lightNumber.get())
        PRINT_FIELD(lightNumber);
	if (!d_MLBFile.get())
		PRINT_FIELD(MLBFile);
	if (!d_IESFile.get())
		PRINT_FIELD(IESFile);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodePhotometricLight::setField(const char *fieldName,
                                 const VrmlField &fieldValue)
{
    if
        TRY_FIELD(lightNumber, SFInt)
	else if
		TRY_FIELD(MLBFile, SFString)
	else if
		TRY_FIELD(IESFile, SFString)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);

    if (strcmp(fieldName, "lightNumber") == 0)
    {
    }
	if (strcmp(fieldName, "MLBFile") == 0)
	{
		mlbFile = new coMLB(d_MLBFile.get());

		osg::ref_ptr<osg::Texture3D> lightTextures = new osg::Texture3D();
		lightTextures->setResizeNonPowerOfTwoHint(false);
		lightTextures->setFilter(osg::Texture3D::MIN_FILTER, osg::Texture3D::NEAREST);
		lightTextures->setFilter(osg::Texture3D::MAG_FILTER, osg::Texture3D::NEAREST);
		lightTextures->setWrap(osg::Texture3D::WRAP_S, osg::Texture3D::CLAMP);
		lightTextures->setWrap(osg::Texture3D::WRAP_T, osg::Texture3D::CLAMP);
		lightTextures->setImage(mlbFile->getTexture());

		osg::StateSet *state = cover->getObjectsRoot()->getOrCreateStateSet();
		std::cout << "Texture 3D set to: " << 6 + d_lightNumber.get() << std::endl;
		//state->setTextureAttributeAndModes(6 + d_lightNumber.get(), lightTextures, osg::StateAttribute::ON);
		state->addUniform(new osg::Uniform("left", (mlbFile->header.left)));
		state->addUniform(new osg::Uniform("bottom", (mlbFile->header.bottom)));
		state->addUniform(new osg::Uniform("width", (mlbFile->header.width)));
		state->addUniform(new osg::Uniform("height", (mlbFile->header.height)));
		//state->addUniform(new osg::Uniform("num_lights", (mlbFile->header.t_depth)));
		int numHorizontalAngles = mlbFile->header.t_width;
		int numVerticalAngles = mlbFile->header.t_height;
		int numValues = numHorizontalAngles * numVerticalAngles;
		std::cout << "Tex. size: " << numHorizontalAngles << " x " << numVerticalAngles << std::endl;

		std::string buf = "share/covise/materials/MatrixLight.compute";
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
		//osg::ref_ptr<osg::Image> texImg = osgDB::readImageFile("D:/covise_stuff/sample_512.png");
		//osg::ref_ptr<osg::Texture2D> tex2D = new osg::Texture2D(texImg);
		// Create the texture as both the output of compute shader and the input of a normal quad
		osg::ref_ptr<osg::Texture2D> tex2D = new osg::Texture2D();
		tex2D->setTextureSize(numHorizontalAngles, numVerticalAngles);
		tex2D->setResizeNonPowerOfTwoHint(false);
		tex2D->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::NEAREST);
		tex2D->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::NEAREST);
		tex2D->setInternalFormat(GL_R32F);
		tex2D->setSourceFormat(GL_RED);
		tex2D->setSourceType(GL_FLOAT);

		// So we can use 'image2D' in the compute shader. Texzure bindung is very expensive!
		osg::ref_ptr<osg::BindImageTexture> imagbinding1 = new osg::BindImageTexture(7, tex2D, osg::BindImageTexture::WRITE_ONLY, GL_R32F);
		osg::ref_ptr<osg::BindImageTexture> imagbinding2 = new osg::BindImageTexture(6, lightTextures, osg::BindImageTexture::READ_ONLY, GL_R32F);

	    // The compute shader can't work with other kinds of shaders
		// It also requires the work group numbers. Setting them to 0 will disable the compute shader
		computeProg = new osg::Program;
		computeProg->addShader(new osg::Shader(osg::Shader::COMPUTE, code));

		// Create a node for outputting to the texture.
		// It is OK to have just an empty node here, but seems inbuilt uniforms like osg_FrameTime won't work then.
		// TODO: maybe we can have a custom drawable which also will implement glMemoryBarrier?
		osg::ref_ptr<osg::Node> sourceNode = new osg::ComputeDispatch(numHorizontalAngles/2 / 16 + 1, numVerticalAngles/2 / 16 + 1, 1);  // +1 because of 
		state = sourceNode->getOrCreateStateSet(); // source node
		sourceNode->setDataVariance(osg::Object::DYNAMIC);
		state->setAttributeAndModes(computeProg.get());  //get()); if otherNode is a ref_ptr, without if the node is a raw pointer
		state->addUniform(new osg::Uniform("AllPhotometricLights", (int)6));  // works
		state->addUniform(new osg::Uniform("targetTex", (int)7));  // works
		// uniforms
		state->setTextureAttributeAndModes(6, lightTextures, osg::StateAttribute::ON);  // works   // what does "osg::StateAttribute::ON" do?
		state->setTextureAttributeAndModes(7, tex2D, osg::StateAttribute::ON);  // works   // what does "osg::StateAttribute::ON" do?


		// Display the texture on a quad. We will also be able to operate on the data if reading back to CPU side
		state = cover->getObjectsRoot()->getOrCreateStateSet();  // Object Root
		//state->setMode(GL_LIGHTING, osg::StateAttribute::OFF); // not needed
		state->setTextureAttributeAndModes(6, lightTextures, osg::StateAttribute::ON);  // what is osg::StateAttribute::ON
		state->setTextureAttributeAndModes(7, tex2D, osg::StateAttribute::ON);  // what is osg::StateAttribute::ON
		state->setAttributeAndModes(imagbinding1.get());
		state->setAttributeAndModes(imagbinding2.get());
		state->addUniform(new osg::Uniform("AllPhotometricLights", (int) 6));
		state->addUniform(new osg::Uniform("targetTex", (int)7));

		// Create the scene graph and start the viewer
		cover->getScene()->addChild(sourceNode);
		int i = computeProg->getNumShaders();
		std::cout << "num shaders after setting up compute shader:" << i << std::endl;

	}

	if (strcmp(fieldName, "IESFile") == 0)
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
        osg::StateSet *state = cover->getObjectsRoot()->getOrCreateStateSet();

        state->setTextureAttributeAndModes(5+d_lightNumber.get(), lightTexture, osg::StateAttribute::ON);
    }
}

const VrmlField *VrmlNodePhotometricLight::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "lightNumber") == 0)
        return &d_lightNumber;
	if (strcmp(fieldName, "MLBFile") == 0)
		return &d_MLBFile;
	if (strcmp(fieldName, "IESFile") == 0)
		return &d_IESFile;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}
