/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodePhotometricLight.h

#ifndef _VRMLNODEPhotometricLight_
#define _VRMLNODEPhotometricLight_

#include <util/coTypes.h>
#include "coIES.h"

#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlSFBool.h>
#include <vrml97/vrml/VrmlSFInt.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFVec3f.h>
#include <vrml97/vrml/VrmlSFString.h>
#include <vrml97/vrml/VrmlSFRotation.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>
#include <cover/coVRPluginSupport.h>

#include <osg/Version>
#if OSG_VERSION_GREATER_OR_EQUAL(3, 6, 0)
#define HAVE_VRMLNODEPHOTOMETRICLIGHT
// all OSG releases have DispatchCompute
#include <osg/DispatchCompute>
#elif OSG_VERSION_GREATER_OR_EQUAL(3, 5, 0)
// only some dev versions use ComputeDispatch
#define HAVE_VRMLNODEPHOTOMETRICLIGHT
#include <osg/ComputeDispatch>
namespace osg { typedef ComputeDispatch DispatchCompute; }
#endif

#include <osg/Texture2D>



#ifdef HAVE_VRMLNODEPHOTOMETRICLIGHT
using namespace opencover;
using namespace vrml;
class coMLB;

class VRML97COVEREXPORT VrmlNodePhotometricLight : public VrmlNodeChild
{

public:
    // Define the fields of PhotometricLight nodes
    static void initFields(VrmlNodePhotometricLight *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodePhotometricLight(VrmlScene *scene = 0);
    VrmlNodePhotometricLight(const VrmlNodePhotometricLight &n);
    virtual ~VrmlNodePhotometricLight();
    virtual void addToScene(VrmlScene *s, const char *);

    virtual VrmlNodePhotometricLight *toPhotometricLight() const;

    virtual void render(Viewer *);

	static void updateAll();
	void update();
	static void updateLightTextures(osg::RenderInfo &);
	void updateLightTexture(osg::RenderInfo &);
    
    static std::list<VrmlNodePhotometricLight *> allPhotometricLights;
	osg::ref_ptr<osg::Program> computeProg;
	osg::ref_ptr<osg::DispatchCompute> comp_disp;
	bool coMLB_initialized = false;
	bool configuration_changed = false;
	std::vector<float> configuration_vec;
	osg::ref_ptr<osg::Texture2D> light_conf_tex;
	osg::ref_ptr<osg::Image> configuration_img;  
	/*
	Never use a regular C++ pointer variable for long-term storage of pointers
	to objects derived from Referenced.As an exception, you can use a
	regular C++ pointer variable temporarily, as long as the heap memory
	address is eventually stored in a ref_ptr<>.However, using a ref_ptr<>
	is always the safest approach.
	*/


private:
    // Fields
    VrmlSFInt d_lightNumber;
	VrmlSFString d_MLBFile;
	VrmlSFString d_IESFile;

	coMLB *mlbFile;
	coIES *iesFile;
    static osg::ref_ptr<osg::Uniform> photometricLightMatrix;
    Viewer::Object d_viewerObject;
    osg::ref_ptr<osg::MatrixTransform> lightNodeInSceneGraph;
    static const int MAX_LIGHTS = 4;  // must always be 4 or less!
    void handleMLBFile();
    void handleIESFile();

};
#endif

#endif //_VRMLNODEPhotometricLight_
