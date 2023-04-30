#include "ViewerOsg.h"
/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) Uwe Woessner
//
//  %W% %G%
//  ViewerOsg.cpp
//  Display of VRML models using Performer/COVER.
//
#ifndef NOMINMAX
#define NOMINMAX
#endif
static const int NUM_TEXUNITS = 4;

#ifdef _WIN32
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <winsock2.h>
#include <windows.h>
#endif
#include <util/unixcompat.h>
#include <vrml97/vrml/config.h>

#include <vrml97/vrml/MathUtils.h>
#include <vrml97/vrml/System.h>
#include <vrml97/vrml/VrmlScene.h>
#include <vrml97/vrml/VrmlNodeNavigationInfo.h>
#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlNodeLight.h>
#include <vrml97/vrml/VrmlNodeMovieTexture.h>
#include <vrml97/vrml/Player.h>

#include <cover/coTabletUI.h>
#include <cover/coVRTui.h>
#include <cover/coVRLighting.h>
#include <cover/coVRShadowManager.h>

#include <osg/BlendFunc>
#include <osg/AlphaFunc>
#include <osg/LineWidth>
#include <osg/GL>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/TexEnv>
#include <osg/Texture>
#include <osg/TextureCubeMap>
#include <osg/Texture2D>
#include <osg/Texture3D>
#include <osg/Geode>
#include <osg/Switch>
#include <osg/Geometry>
#include <osg/PrimitiveSet>
#include <osg/CullFace>
#include <osg/BlendFunc>
#include <osg/Light>
#include <osg/LightSource>
#include <osg/Depth>
#include <osg/Fog>
#include <osg/AlphaFunc>
#include <osg/ColorMask>
#include <osg/Program>
#include <osg/Shader>
#include <osg/Point>
#include <osg/ImageStream>
#include <osg/PolygonOffset>

#include <osgDB/ReadFile>
#include <osgDB/Registry>

#include <osgUtil/Tessellator>
#include <osg/KdTree>
#include <osgUtil/TangentSpaceGenerator>

#ifdef HAVE_OSGNV
#include <osgNVCg/Context>
#include <osgNVCg/Program>

#include <osgNV/Version>
#include <osgNV/VectorParameterValue>
#include <osgNV/StateMatrixParameterValue>
#endif

#include <osgText/Font>
#include <osgText/Text>

#include "ViewerOsg.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/coVRFileManager.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <cover/coBillboard.h>
#include <cover/VRViewer.h>
#include <cover/coVRLighting.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRShader.h>

#include <iterator>

#include <config/CoviseConfig.h>

#include <OpenVRUI/osg/OSGVruiUserDataCollection.h>

using namespace std;
using covise::coCoviseConfig;

#ifdef HAVE_SDL
#include "SDLAudio.h"
#endif

MirrorInfo::MirrorInfo()
{
    geometry = NULL;
    CameraID = -1; // no camera, normal Mirror per default
}

MirrorInfo::~MirrorInfo()
{
}

static const char *combineTexturesFragSource = {
    "uniform sampler2D tex0;"
    "uniform sampler2D tex1;"
    "uniform sampler2D tex2;"
    ""
    "void main(void)"
    "{"
    "vec4 t0 = texture2D(tex0, gl_TexCoord[0].st);"
    "vec4 t1 = texture2D(tex1, gl_TexCoord[1].st);"
    "vec4 t2 = texture2D(tex2, gl_TexCoord[2].st);"
    "vec4 v = t0*t1 + t2*(1.0-t1);"
    "gl_FragColor = v * gl_Color;"
    "}"
};

static const char *combineEnvTexturesFragSource = {
    "uniform sampler2D tex0;"
    "uniform sampler2D tex1;"
    "uniform sampler2D tex2;"
    ""
    "void main(void)"
    "{"
    "vec4 t0 = texture2D(tex0, gl_TexCoord[0].st);"
    "vec4 t1 = texture2D(tex1, gl_TexCoord[1].st);"
    "vec4 t2 = texture2D(tex2, gl_TexCoord[2].st);"
    "vec4 v = t0 + t2*t1;"
    "gl_FragColor = v * gl_Color;"
    "}"
};

static const char *bumpCubeShaderVertSource = {
    "varying vec3 lightVec;"
    "varying vec3 eyeVec;"
    "varying vec2 texCoord;"
    "attribute vec3 vTangent; "
    "mat3 GetLinearPart( mat4 m )"
    "{"
    "	mat3 result;"
    "	result[0][0] = m[0][0];"
    "	result[0][1] = m[0][1];"
    "	result[0][2] = m[0][2];"
    "	result[1][0] = m[1][0];"
    "	result[1][1] = m[1][1];"
    "	result[1][2] = m[1][2];"
    "	result[2][0] = m[2][0];"
    "	result[2][1] = m[2][1];"
    "	result[2][2] = m[2][2];"
    "	return result;"
    "}"
    "void main(void)"
    "{"
    "	gl_Position = ftransform();"
    "	texCoord = gl_MultiTexCoord0.xy;"
    "	vec3 n = normalize(gl_NormalMatrix * gl_Normal);"
    "	vec3 t = normalize(gl_NormalMatrix * vTangent);"
    "	vec3 b = cross(n, t);"
    "	vec3 vVertex = vec3(gl_ModelViewMatrix * gl_Vertex);"
    "	vec3 tmpVec = normalize(gl_LightSource[0].position.xyz - vVertex);"
    "	lightVec.x = dot(tmpVec, t);"
    "	lightVec.y = dot(tmpVec, b);"
    "	lightVec.z = dot(tmpVec, n);"
    "	tmpVec = -vVertex;"
    "	eyeVec.x = dot(tmpVec, t);"
    "	eyeVec.y = dot(tmpVec, b);"
    "	eyeVec.z = dot(tmpVec, n);"
    "	mat3 ModelView3x3 = GetLinearPart( gl_ModelViewMatrix );"
    "	vec3 N = normalize( ModelView3x3 * gl_Normal );"
    "	gl_TexCoord[2].xyz = reflect( vVertex, N );"
    "}"
};

static const char *bumpCubeShaderFragSource = {
    "varying vec3 lightVec;"
    "varying vec3 eyeVec;"
    "varying vec2 texCoord;"
    "uniform sampler2D colorMap;"
    "uniform sampler2D normalMap;"
    "uniform samplerCube cubeMap;"
    "uniform float invRadius;"
    ""
    "void main (void)"
    "{"
    "	float distSqr = dot(lightVec, lightVec);"
    "	float att = clamp(1.0 - invRadius * sqrt(distSqr), 0.0, 1.0);"
    "	vec3 lVec = lightVec * inversesqrt(distSqr);"
    ""
    "	vec3 vVec = normalize(eyeVec);"
    "	"
    "	vec4 base = texture2D(colorMap, texCoord);"
    "	"
    /*"	vec3 bump = normalize( texture2D(normalMap, texCoord).xyz * 2.0 - 1.0);"
   ""
   "	vec4 vAmbient = gl_LightSource[0].ambient * gl_FrontMaterial.ambient;"
   ""
   "	float diffuse = max( dot(lVec, bump), 0.0 );"
   "	"
   "	vec4 vDiffuse = gl_LightSource[0].diffuse * gl_FrontMaterial.diffuse * "
   "					diffuse;	"
   ""
   "	float specular = pow(clamp(dot(reflect(-vVec, bump), lVec), 0.0, 1.0), "
   "	                 gl_FrontMaterial.shininess );"
   "	vec4 vCu = textureCube(cubeMap, gl_TexCoord[2].xyz);"
   "	vec4 vSpecular = gl_LightSource[0].specular * gl_FrontMaterial.specular * "
   "					 specular;	"
   "	gl_FragColor = ( vAmbient*base + vDiffuse*base + vSpecular) * att;"*/

    "	vec4 vCu = textureCube(cubeMap, gl_TexCoord[2].xyz);"
    "	gl_FragColor = ( base) * att;"
    "}"
};

static const char *bumpEnvShaderVertSource = {
    "varying vec3 lightVec;"
    "varying vec3 eyeVec;"
    "varying vec2 texCoord;"
    "attribute vec3 vTangent; "
    "mat3 GetLinearPart( mat4 m )"
    "{"
    "	mat3 result;"
    "	result[0][0] = m[0][0];"
    "	result[0][1] = m[0][1];"
    "	result[0][2] = m[0][2];"
    "	result[1][0] = m[1][0];"
    "	result[1][1] = m[1][1];"
    "	result[1][2] = m[1][2];"
    "	result[2][0] = m[2][0];"
    "	result[2][1] = m[2][1];"
    "	result[2][2] = m[2][2];"
    "	return result;"
    "}"
    "void main(void)"
    "{"
    "	gl_Position = ftransform();"
    "	texCoord = gl_MultiTexCoord0.xy;"
    "	vec3 n = normalize(gl_NormalMatrix * gl_Normal);"
    "	vec3 t = normalize(gl_NormalMatrix * vTangent);"
    "	vec3 b = cross(n, t);"
    "	vec3 vVertex = vec3(gl_ModelViewMatrix * gl_Vertex);"
    "	vec3 tmpVec = normalize(gl_LightSource[0].position.xyz - vVertex);"
    "	lightVec.x = dot(tmpVec, t);"
    "	lightVec.y = dot(tmpVec, b);"
    "	lightVec.z = dot(tmpVec, n);"
    "	tmpVec = -vVertex;"
    "	eyeVec.x = dot(tmpVec, t);"
    "	eyeVec.y = dot(tmpVec, b);"
    "	eyeVec.z = dot(tmpVec, n);"
    "	mat3 ModelView3x3 = GetLinearPart( gl_ModelViewMatrix );"
    "	vec3 N = normalize( ModelView3x3 * gl_Normal );"
    "	vec3 r = reflect( vVertex, N );"
    "	float m = 2.0 * sqrt( r.x*r.x + r.y*r.y + (r.z+1.0)*(r.z+1.0) );"
    "	gl_TexCoord[2].s = r.x/m + 0.5;"
    "	gl_TexCoord[2].t = r.y/m + 0.5;"
    "}"
};

static const char *bumpEnvShaderFragSource = {
    "varying vec3 lightVec;"
    "varying vec3 eyeVec;"
    "varying vec2 texCoord;"
    "uniform sampler2D colorMap;"
    "uniform sampler2D normalMap;"
    "uniform sampler2D sphereTexture;"
    "uniform float invRadius;"
    ""
    "void main (void)"
    "{"
    "	float distSqr = dot(lightVec, lightVec);"
    "	float att = clamp(1.0 - invRadius * sqrt(distSqr), 0.0, 1.0);"
    "	vec3 lVec = lightVec * inversesqrt(distSqr);"
    "	vec3 vVec = normalize(eyeVec);"
    "	vec4 base = texture2D(colorMap, texCoord);"
    "	vec4 vCu = texture2D(sphereTexture, gl_TexCoord[2]);"
    "	vec4 vAmbient =  gl_FrontMaterial.ambient;"
    "	vec3 bump = normalize( texture2D(normalMap, texCoord).xyz * 2.0 - 1.0);"
    "	float diffuse = max( dot(lVec, bump), 0.0 );"
    "	vec4 vDiffuse = gl_LightSource[0].diffuse * gl_FrontMaterial.diffuse * diffuse;"
    "	float specular = pow(clamp(dot(reflect(-vVec, bump), lVec), 0.0, 1.0), gl_FrontMaterial.shininess );"
    "	vec4 vSpecular = gl_LightSource[0].specular * gl_FrontMaterial.specular * specular;"
    "	gl_FragColor = ( 0.8*diffuse* vCu + vAmbient*base + vDiffuse*base + vSpecular) * att;"
    "}"
};

///////////////////////////////////////////////////////////////////////////
// OpenGL Shading Language source code for the "bump Mapping" example,

static const char *bumpshaderVertSource = {
    "varying vec3 lightVec;"
    "varying vec3 eyeVec;"
    "varying vec2 texCoord;"
    "attribute vec3 vTangent; "
    "void main(void)"
    "{"
    "	gl_Position = ftransform();"
    "	texCoord = gl_MultiTexCoord0.xy;"
    "	vec3 n = normalize(gl_NormalMatrix * gl_Normal);"
    "	vec3 t = normalize(gl_NormalMatrix * vTangent);"
    "	vec3 b = cross(n, t);"
    "	vec3 vVertex = vec3(gl_ModelViewMatrix * gl_Vertex);"
    "	vec3 tmpVec = normalize(gl_LightSource[0].position.xyz - vVertex);"
    "	lightVec.x = dot(tmpVec, t);"
    "	lightVec.y = dot(tmpVec, b);"
    "	lightVec.z = dot(tmpVec, n);"
    "	tmpVec = -vVertex;"
    "	eyeVec.x = dot(tmpVec, t);"
    "	eyeVec.y = dot(tmpVec, b);"
    "	eyeVec.z = dot(tmpVec, n);"
    "}"
};

static const char *bumpshaderFragSource = {
    "varying vec3 lightVec;"
    "varying vec3 eyeVec;"
    "varying vec2 texCoord;"
    "uniform sampler2D colorMap;"
    "uniform sampler2D normalMap;"
    "uniform float invRadius;"
    ""
    "void main (void)"
    "{"
    "	float distSqr = dot(lightVec, lightVec);"
    "	float att = clamp(1.0 - invRadius * sqrt(distSqr), 0.0, 1.0);"
    "	vec3 lVec = lightVec * inversesqrt(distSqr);"
    ""
    "	vec3 vVec = normalize(eyeVec);"
    "	"
    "	vec4 base = texture2D(colorMap, texCoord);"
    "	"
    "	vec3 bump = normalize( texture2D(normalMap, texCoord).xyz * 2.0 - 1.0);"
    ""
    "	vec4 vAmbient = gl_LightSource[0].ambient * gl_FrontMaterial.ambient;"
    ""
    "	float diffuse = max( dot(lVec, bump), 0.0 );"
    "	"
    "	vec4 vDiffuse = gl_LightSource[0].diffuse * gl_FrontMaterial.diffuse * "
    "					diffuse;	"
    ""
    "	float specular = pow(clamp(dot(reflect(-vVec, bump), lVec), 0.0, 1.0), "
    "	                 gl_FrontMaterial.shininess );"
    "	"
    "	vec4 vSpecular = gl_LightSource[0].specular * gl_FrontMaterial.specular * "
    "					 specular;	"
    "	"
    "	gl_FragColor = ( vAmbient*base + vDiffuse*base + vSpecular) * att;"
    "}"
};

static const char *testshaderVertSource = {
    "varying vec3  Normal;"
    "varying vec3  EyeDir;"
    "varying vec4  EyePos;"
    "varying float LightIntensity;"
    "varying vec2 texCoord;"
    ""
    "varying vec3  LightPos;"
    ""
    "void main(void)"
    "{"
    "    gl_Position    = ftransform();"
    "	texCoord = gl_MultiTexCoord0.xy;"
    "    Normal         = normalize(gl_NormalMatrix * gl_Normal);"
    "    vec4 pos       = gl_ModelViewMatrix * gl_Vertex;"
    "    EyeDir         = pos.xyz;"
    "    EyePos		   = gl_ModelViewProjectionMatrix * gl_Vertex;"
    "	 LightPos = gl_LightSource[0].position.xyz;"
    "    LightIntensity = max(dot(normalize(LightPos - EyeDir), Normal), 0.0);"
    "}"
};

static const char *testshaderFragSource = {
    "const vec3 Xunitvec = vec3 (1.0, 0.0, 0.0);"
    "const vec3 Yunitvec = vec3 (0.0, 1.0, 0.0);"
    "const vec3 BaseColor = vec3 (1.0, 1.0, 1.0);"
    "varying vec2 texCoord;"
    "uniform float Depth;"
    "uniform float MixRatio;"
    "uniform float FrameWidth;"
    "uniform float FrameHeight;"
    ""
    "uniform sampler2D EnvMap;"
    "uniform sampler2D RefractionMap;"
    ""
    "varying vec3  Normal;"
    "varying vec3  EyeDir;"
    "varying vec4  EyePos;"
    "varying float LightIntensity;"
    "void main (void)"
    "{"
    "vec3 reflectDir = reflect(EyeDir, Normal);"
    "    vec2 index;"
    "    index.y = dot(normalize(reflectDir), Yunitvec);"
    "    reflectDir.y = 0.0;"
    "    index.x = dot(normalize(reflectDir), Xunitvec) * 0.5;"
    "    if (reflectDir.z >= 0.0)"
    "        index = (index + 1.0) * 0.5;"
    "    else"
    "    {"
    "        index.t = (index.t + 1.0) * 0.5;"
    "        index.s = (-index.s) * 0.5 + 1.0;"
    "    }"
    "    vec3 envColor = vec3 (texture2D(EnvMap, index));"
    "    float fresnel = abs(dot(normalize(EyeDir), Normal));"
    "    fresnel *= MixRatio;"
    "    fresnel = clamp(fresnel, 0.1, 0.9);"
    "	vec3 refractionDir = normalize(EyeDir) - normalize(Normal);"
    "	float depthVal = Depth / -refractionDir.z;"
    "	float recipW = 1.0 / EyePos.w;"
    "	vec2 eye = EyePos.xy * vec2(recipW);"
    "	index.s = (eye.x + refractionDir.x * depthVal);"
    "	index.t = (eye.y + refractionDir.y * depthVal);"
    "	index.s = index.s / 2.0 + 0.5;"
    "	index.t = index.t / 2.0 + 0.5;"
    "	float recip1k = 1.0 / 2048.0;"
    "	index.s = clamp(index.s, 0.0, 1.0 - recip1k);"
    "	index.t = clamp(index.t, 0.0, 1.0 - recip1k);"
    "	index.s = index.s * FrameWidth * recip1k;"
    "	index.t = index.t * FrameHeight * recip1k;"
    "    vec3 RefractionColor = vec3 (texture2D(RefractionMap, index));"
    "    vec3 base = LightIntensity * BaseColor;"
    "    envColor = mix(envColor, RefractionColor, fresnel);"
    "    envColor = mix(envColor, base, 0.2);"
    "    gl_FragColor = vec4 (RefractionColor, 1.0);"
    "}"
};
static const char *glassShaderVertSource = {
    "varying vec3  Normal;"
    "varying vec3  EyeDir;"
    "varying vec4  EyePos;"
    "varying float LightIntensity;"
    "varying vec2 texCoord;"
    ""
    "varying vec3  LightPos;"
    ""
    "void main(void)"
    "{"
    "    gl_Position    = ftransform();"
    "	texCoord = gl_MultiTexCoord0.xy;"
    "    Normal         = normalize(gl_NormalMatrix * gl_Normal);"
    "    vec4 pos       = gl_ModelViewMatrix * gl_Vertex;"
    "    EyeDir         = pos.xyz;"
    "    EyePos		   = gl_ModelViewProjectionMatrix * gl_Vertex;"
    "	 LightPos = gl_LightSource[0].position.xyz;"
    "    LightIntensity = max(dot(normalize(LightPos - EyeDir), Normal), 0.0);"
    "}"
};

static const char *glassShaderFragSource = {
    "const vec3 Xunitvec = vec3 (1.0, 0.0, 0.0);"
    "const vec3 Yunitvec = vec3 (0.0, 1.0, 0.0);"
    "const vec3 BaseColor = vec3 (1.0, 1.0, 1.0);"
    "varying vec2 texCoord;"
    "uniform float Depth;"
    "uniform float MixRatio;"
    "uniform float FrameWidth;"
    "uniform float FrameHeight;"
    ""
    "uniform sampler2D EnvMap;"
    "uniform sampler2D RefractionMap;"
    ""
    "varying vec3  Normal;"
    "varying vec3  EyeDir;"
    "varying vec4  EyePos;"
    "varying float LightIntensity;"
    ""
    "void main (void)"
    "{"
    "    vec3 reflectDir = reflect(EyeDir, Normal);"
    "    vec2 index;"
    "    index.y = dot(normalize(reflectDir), Yunitvec);"
    "    reflectDir.y = 0.0;"
    "    index.x = dot(normalize(reflectDir), Xunitvec) * 0.5;"
    "    if (reflectDir.z >= 0.0)"
    "        index = (index + 1.0) * 0.5;"
    "    else"
    "    {"
    "        index.t = (index.t + 1.0) * 0.5;"
    "        index.s = (-index.s) * 0.5 + 1.0;"
    "    }"
    "    vec3 envColor = vec3 (texture2D(EnvMap, index));"
    "    float fresnel = abs(dot(normalize(EyeDir), Normal));"
    "    fresnel *= MixRatio;"
    "    fresnel = clamp(fresnel, 0.1, 0.9);"
    "	vec3 refractionDir = normalize(EyeDir) - normalize(Normal);"
    "	float depthVal = Depth / -refractionDir.z;"
    "	float recipW = 1.0 / EyePos.w;"
    "	vec2 eye = EyePos.xy * vec2(recipW);"
    "	index.s = (eye.x + refractionDir.x * depthVal);"
    "	index.t = (eye.y + refractionDir.y * depthVal);"
    "	index.s = index.s / 2.0 + 0.5;"
    "	index.t = index.t / 2.0 + 0.5;"
    "	float recip1k = 1.0 / 2048.0;"
    "	index.s = clamp(index.s, 0.0, 1.0 - recip1k);"
    "	index.t = clamp(index.t, 0.0, 1.0 - recip1k);"
    "	index.s = index.s * FrameWidth * recip1k;"
    "	index.t = index.t * FrameHeight * recip1k;"
    "    vec3 RefractionColor = vec3 (texture2D(RefractionMap, index));"
    "    vec3 base = LightIntensity * BaseColor;"
    "    envColor = mix(envColor, RefractionColor, fresnel);"
    "    envColor = mix(envColor, base, 0.2);"
    "    gl_FragColor = vec4 (envColor, 1.0);"
    "}"
};

using namespace osg;

ViewerOsg *ViewerOsg::viewer = NULL;
int Sorted = 0;
int Blended = 0;
int AlphaTest = 0;
static int Crease = 0;
static bool backFaceCulling = true;
static bool reTesselate = true;
static bool countTextures = true;
static int texSize = 0;
static int oldTexSize = 0;
static bool countGeometry = true;
static int numVert = 0;
static int oldNumVert = 0;
static int numPoly = 0;
static int oldNumPoly = 0;
static bool UseFieldOfViewForScaling = false;

int textureMode = -1;
static int textureQuality = 0;

osg::ref_ptr<osg::TexEnv> tEnvAdd;
osg::ref_ptr<osg::TexEnv> tEnvBlend;
osg::ref_ptr<osg::TexEnv> tEnvDecal;
osg::ref_ptr<osg::TexEnv> tEnvReplace;
osg::ref_ptr<osg::TexEnv> tEnvModulate;

// current transformation matrix while traversing the Tree.
// it is updated, in an endObject call, the setTransform call
osg::ref_ptr<MatrixTransform> ViewerOsg::VRMLCaveRoot;

struct CopyTextureCallback : public osg::Drawable::DrawCallback
{

    CopyTextureCallback()
    {
    }
    virtual void drawImplementation(osg::RenderInfo &renderInfo, const osg::Drawable *drawable) const
    {
        if (!ViewerOsg::viewer->framebufferTextureUpdated)
        {
            ViewerOsg::viewer->framebufferTexture->copyTexImage2D(*renderInfo.getState(), 0, 0, 2048, 2048);
            ViewerOsg::viewer->framebufferTextureUpdated = true;
        }
        drawable->drawImplementation(renderInfo);
    }
};

static void worldChangedCB(int reason)
{
    switch (reason)
    {
    case VrmlScene::DESTROY_WORLD:
        if (ViewerOsg::viewer)
            ViewerOsg::viewer->restart();
        if (cover->debugLevel(1))
            cerr << "DESTROY_WORLD" << endl;
        break;

    case VrmlScene::REPLACE_WORLD:
        if (ViewerOsg::viewer)
            ViewerOsg::viewer->restart();
        if (cover->debugLevel(1))
            cerr << "REPLACE_WORLD" << endl;
        break;
    }
}

void ViewerOsg::setRootNode(osg::Group *group)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::setRootNode\n";
    if (VRMLRoot->getNumParents() > 0)
    {
        if (VRMLRoot->getParent(0) == group)
            return;
    }
    VRMLRoot = new MatrixTransform();
    VRMLRoot->setName("VRMLRoot");
    Matrix tmpMat;
    tmpMat.makeRotate(M_PI / 2.0, 1.0, 0.0, 0.0);
    //tmpMat.scale(Vec3(1000.f, 1000.f, 1000.f));
    VRMLRoot->setMatrix(tmpMat);
    group->addChild(VRMLRoot);

    if (d_root)
    {
        d_root->pNode = VRMLRoot;
        d_currentObject = d_root;
        if (d_scene->getRoot()->toGroup()->getViewerObject() != 0)
        {
            ((osgViewerObject *)(d_scene->getRoot()->toGroup()->getViewerObject()))->pNode = VRMLRoot;
        }
    }
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::setRootNode\n";
}

#ifdef DEBUG_LINES
osg::Vec3Array *LineVerts;
osg::Vec4Array *LineColors;
coTUIToggleButton *updateCameraButton;
#endif
//  Construct a viewer for the specified scene. I'm not happy with the
//  mutual dependencies between VrmlScene/VrmlNodes and Viewers...
//  Maybe a static callback function pointer should be passed in rather
//  than a class object pointer. Currently, the scene is used to access
//  the VrmlScene::render() method. Also, the static VrmlScene::update
//  is called from the idle function. A way to pass mouse/keyboard sensor
//  events back to the scene is also needed.

ViewerOsg::ViewerOsg(VrmlScene *s, Group *rootNode)
    : Viewer(s)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::ViewerOsg\n";

#ifdef DEBUG_LINES
    coTUITab *vrmlTab = new coTUITab("Mirror", coVRTui::instance()->mainFolder->getID());
    vrmlTab->setPos(0, 0);

    updateCameraButton = new coTUIToggleButton("UpdateCamera", vrmlTab->getID());
    //updateCameraButton->setEventListener(this);
    updateCameraButton->setPos(0, 0);
    viewerPos.set(0, 0, 0);
#endif

    viewer = this;
    numCameras = 0;
    textureNumber = 0;
    //NoFrameBuffer = new osg::ColorMask(false,false,false,false);
    NoDepthBuffer = new osg::Depth(osg::Depth::LESS, 0.0, 1.0, false);
    NoDepthBuffer->setWriteMask(false);
    numVP = -1;
    //viewpoints = NULL;
    // Don't make any GL calls here since the window probably doesn't exist.

    d_selectMode = false;
    UseFieldOfViewForScaling = coCoviseConfig::isOn("COVER.Plugin.Vrml97.UseFieldOfViewForScaling", false);
    SubdivideThreshold=coCoviseConfig::getInt("COVER.Plugin.Vrml97.SubdivideThreshold", 10000);

    currentTransform.makeIdentity();

    d_renderTime = 1.0;
    d_renderTime1 = 1.0;

    d_root = new osgViewerObject(NULL);
    d_currentObject = d_root;
    globalmtl = NULL;
    startLoadTime = 0;
    framebufferTextureUpdated = false;
    framebufferTexture = new osg::Texture2D();

    osg::Uniform *tex0 = new osg::Uniform("tex0", 0);
    osg::Uniform *tex1 = new osg::Uniform("tex1", 1);
    osg::Uniform *tex2 = new osg::Uniform("tex2", 2);

    combineTexturesState = new osg::StateSet;
    combineTextures = new osg::Program;
    combineTextures->addShader(new osg::Shader(osg::Shader::FRAGMENT, combineTexturesFragSource));
    combineTexturesState->setAttributeAndModes(combineTextures.get(), osg::StateAttribute::ON);
    combineTexturesState->addUniform(tex0);
    combineTexturesState->addUniform(tex1);
    combineTexturesState->addUniform(tex2);

    combineEnvTexturesState = new osg::StateSet;
    combineEnvTextures = new osg::Program;
    combineEnvTextures->addShader(new osg::Shader(osg::Shader::FRAGMENT, combineEnvTexturesFragSource));
    combineEnvTexturesState->setAttributeAndModes(combineEnvTextures.get(), osg::StateAttribute::ON);
    combineEnvTexturesState->addUniform(tex0);
    combineEnvTexturesState->addUniform(tex1);
    combineEnvTexturesState->addUniform(tex2);

    bumpProgram = new osg::Program;
    bumpProgram->addShader(new osg::Shader(
        osg::Shader::FRAGMENT, bumpshaderFragSource));
    bumpProgram->addShader(new osg::Shader(
        osg::Shader::VERTEX, bumpshaderVertSource));
    bumpProgram->addBindAttribLocation("vTangent", 6);

    bumpCubeProgram = new osg::Program;
    bumpCubeProgram->addShader(new osg::Shader(
        osg::Shader::FRAGMENT, bumpCubeShaderFragSource));
    bumpCubeProgram->addShader(new osg::Shader(
        osg::Shader::VERTEX, bumpCubeShaderVertSource));
    bumpCubeProgram->addBindAttribLocation("vTangent", 6);

    bumpEnvProgram = new osg::Program;
    bumpEnvProgram->addShader(new osg::Shader(
        osg::Shader::FRAGMENT, bumpEnvShaderFragSource));
    bumpEnvProgram->addShader(new osg::Shader(
        osg::Shader::VERTEX, bumpEnvShaderVertSource));
    bumpEnvProgram->addBindAttribLocation("vTangent", 6);

    glassProgram = new osg::Program;
    glassProgram->addShader(new osg::Shader(
        osg::Shader::VERTEX, glassShaderVertSource));
    glassProgram->addShader(new osg::Shader(
        osg::Shader::FRAGMENT, glassShaderFragSource));

    testProgram = new osg::Program;
    testProgram->addShader(new osg::Shader(
        osg::Shader::VERTEX, testshaderVertSource));
    testProgram->addShader(new osg::Shader(
        osg::Shader::FRAGMENT, testshaderFragSource));

    BumpCubeState = new osg::StateSet();
    BumpCubeState->setAttributeAndModes(bumpCubeProgram.get(), osg::StateAttribute::ON);

    osg::Uniform *lightPosU = new osg::Uniform("LightPosition", osg::Vec3(0.0, -10000.0, 10000.0));
    osg::Uniform *normalMapU = new osg::Uniform("normalMap", 1);
    osg::Uniform *baseTextureU = new osg::Uniform("baseTexture", 0);
    osg::Uniform *cubeTextureU = new osg::Uniform("cubeTexture", 2);

    BumpCubeState->addUniform(lightPosU);
    BumpCubeState->addUniform(normalMapU);
    BumpCubeState->addUniform(baseTextureU);
    BumpCubeState->addUniform(cubeTextureU);

    BumpEnvState = new osg::StateSet();
    BumpEnvState->setAttributeAndModes(bumpEnvProgram.get(), osg::StateAttribute::ON);

    osg::Uniform *lightPosUE = new osg::Uniform("LightPosition", osg::Vec3(0.0, -10000.0, 10000.0));
    osg::Uniform *baseTextureUE = new osg::Uniform("baseTexture", 0);
    osg::Uniform *normalMapUE = new osg::Uniform("normalMap", 1);
    osg::Uniform *cubeTextureUE = new osg::Uniform("sphereTexture", 2);

    BumpEnvState->addUniform(lightPosUE);
    BumpEnvState->addUniform(baseTextureUE);
    BumpEnvState->addUniform(normalMapUE);
    BumpEnvState->addUniform(cubeTextureUE);

    textureMode = TexEnv::MODULATE;
    std::string entry = coCoviseConfig::getEntry("mode", "COVER.Plugin.Vrml97.Texture");
    if (!entry.empty())
    {
        if (strncasecmp(entry.c_str(), "MODULATE", 8) == 0)
        {
            textureMode = TexEnv::MODULATE;
            if (cover->debugLevel(1))
                cerr << "TextureMode: MODULATE" << endl;
        }
        else if (strncasecmp(entry.c_str(), "BLEND", 5) == 0)
        {
            textureMode = TexEnv::BLEND;
            if (cover->debugLevel(1))
                cerr << "TextureMode: BLEND" << endl;
        }
        else if (strncasecmp(entry.c_str(), "DECAL", 5) == 0)
        {
            textureMode = TexEnv::DECAL;
            if (cover->debugLevel(1))
                cerr << "TextureMode: DECAL" << endl;
        }
        else if (strncasecmp(entry.c_str(), "REPLACE", 7) == 0)
        {
            textureMode = TexEnv::REPLACE;
            if (cover->debugLevel(1))
                cerr << "TextureMode: REPLACE" << endl;
        }
        else if (strncasecmp(entry.c_str(), "ADD", 3) == 0)
        {
            textureMode = TexEnv::ADD;
            if (cover->debugLevel(1))
                cerr << "TextureMode: ADD" << endl;
        }
        else if (strncasecmp(entry.c_str(), "ALPHA", 5) == 0)
        {
            textureMode = TexEnv::BLEND;
            cerr << "TextureMode ALPHA is not supported by OpenSceneGraph\n" << endl;
        }
    }
    textureQuality = 1;
    entry = coCoviseConfig::getEntry("quality", "COVER.Plugin.Vrml97.Texture");
    if (!entry.empty())
    {
        if (strncasecmp(entry.c_str(), "HIGH", 4) == 0)
        {
            textureQuality = 1;
            if (cover->debugLevel(1))
                cerr << "TextureQuality: High" << endl;
        }
        if (strncasecmp(entry.c_str(), "LOW", 3) == 0)
        {
            textureQuality = 2;
            if (cover->debugLevel(1))
                cerr << "TextureQuality: Low" << endl;
        }
        if (strncasecmp(entry.c_str(), "NORMAL", 6) == 0)
        {
            textureQuality = 0;
            if (cover->debugLevel(1))
                cerr << "TextureQuality: Normal" << endl;
        }
    }

    tEnvModulate = new TexEnv;
    tEnvModulate->setMode(TexEnv::MODULATE);
    tEnvBlend = new TexEnv;
    tEnvBlend->setMode(TexEnv::BLEND);
    tEnvDecal = new TexEnv;
    tEnvDecal->setMode(TexEnv::DECAL);
    tEnvReplace = new TexEnv;
    tEnvReplace->setMode(TexEnv::REPLACE);
    tEnvAdd = new TexEnv;
    tEnvAdd->setMode(TexEnv::ADD);

    // Create a scene
    VRMLRoot = new MatrixTransform();
    VRMLRoot->setName("VRMLRoot");
    Matrix tmpMat;
    tmpMat.makeRotate(M_PI / 2.0, 1.0, 0.0, 0.0);
    //tmpMat.scale(Vec3(1000.f, 1000.f, 1000.f));
    VRMLRoot->setMatrix(tmpMat);
	VRMLRoot->setNodeMask(VRMLRoot->getNodeMask() & ~Isect::Update);
    VRMLCaveRoot = new MatrixTransform();
    VRMLCaveRoot->setName("VRMLCaveRoot");
    VRMLCaveRoot->setMatrix(tmpMat);
    cover->getScene()->addChild(VRMLCaveRoot);
    //tgenNode->addChild( VRMLRoot );
    //rootNode->addChild(tgenNode);
    rootNode->addChild(VRMLRoot);

    s->addWorldChangedCallback(worldChangedCB);

    // Create a lit scene pfGeoState for the scene
    //pfGeoState *gstate = new pfGeoState;
    // gstate->setMode(PFSTATE_ENLIGHTING, PF_ON);
    // attach the pfGeoState to the scene
    //scene->setGState(gstate);

    // put a default light source in the scene
    //scene->addChild(new pfLightSource);

    std::string buf = coCoviseConfig::getEntry("value", "COVER.Plugin.Vrml97.TransparencyMode", "sorted_blended_alphatest");
    std::transform(buf.begin(), buf.end(), buf.begin(), ::tolower);
    cerr << "testit " << endl;
    cerr << buf << endl;
    if (!buf.empty())
    {
        if (buf == "off")
        {
            Sorted = 0;
            Blended = 0;
            AlphaTest = 0;
        }
        else
        {
            if (buf.find("blended") != std::string::npos)
                Blended = 1;
            if (buf.find("sorted") != std::string::npos)
                Sorted = 1;
            if (buf.find("alphatest") != std::string::npos)
                AlphaTest = 1;
        }
        cerr << "TransparencyMode: " << buf << endl;
        //default Sorted=0 Blended=1 AlphaTest=1
    }
    else
    {
        if (cover->debugLevel(1))
        {
            cerr << "Transparency mode: default" << endl;
        }
#ifndef __sgi
        Sorted = 1;
        Blended = 1;
#endif
    }

    Crease = coCoviseConfig::isOn("COVER.Plugin.Vrml97.Crease", true);
    enableLights = coCoviseConfig::isOn("COVER.Plugin.Vrml97.Lights", true);
    reTesselate = coCoviseConfig::isOn("COVER.Plugin.Vrml97.ReTesselate", false);
    countTextures = coCoviseConfig::isOn("counter", "COVER.Plugin.Vrml97.Texture", true);
    countGeometry = coCoviseConfig::isOn("COVER.Plugin.Vrml97.GeometryCounter", true);
    startLoadTime = cover->frameTime();
    //backFaceCulling = coCoviseConfig::isOn("COVER.Plugin.Vrml97.CorrectBackfaceCulling");
    backFaceCulling = System::the->isCorrectBackFaceCulling();

    /*if(coCoviseConfig::getEntry("COVER.Plugin.Vrml97.NoViewpoints"))
     {
     noViewpoints = 1;
     if(cover->debugLevel(1))
     cerr << "Viewpoints OFF!" << endl;
     }
     else
     {
     if(cover->debugLevel(1))
     cerr << "Viewpoints ON!" << endl;
     }*/

    font = coVRFileManager::instance()->loadFont(NULL);

    d_player = NULL;

    if (cover->debugLevel(3))
        CERR << "d_player: " << d_player << endl;

    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::ViewerOsg\n";
#ifdef DEBUG_LINES

    osg::Geometry *lines = new osg::Geometry();
    lines->setUseDisplayList(false);
    lines->setUseVertexBufferObjects(false);

    // set up geometry
    LineVerts = new osg::Vec3Array;
    osg::DrawArrays *primitives = new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, 100 * 2);
    LineColors = new osg::Vec4Array();
    for (int i = 0; i < 100; i++)
    {
        LineVerts->push_back(osg::Vec3(0, 0, 0));
        LineVerts->push_back(osg::Vec3(0, 0, 0));
        LineColors->push_back(osg::Vec4(1, 1, 1, 1));
        LineColors->push_back(osg::Vec4(1, 1, 1, 1));
    }
    lines->setVertexArray(LineVerts);
    lines->addPrimitiveSet(primitives);
    lines->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    lines->setColorArray(LineColors);
    osg::Geode *geode = new osg::Geode();

    osg::StateSet *geoState = geode->getOrCreateStateSet();
    //setDefaultMaterial(geoState, transparent);
    geoState->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    osg::BlendFunc *blendFunc = new osg::BlendFunc();
    blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
    geoState->setAttributeAndModes(blendFunc, osg::StateAttribute::ON);
    osg::AlphaFunc *alphaFunc = new osg::AlphaFunc();
    alphaFunc->setFunction(osg::AlphaFunc::ALWAYS, 1.0);
    geoState->setAttributeAndModes(alphaFunc, osg::StateAttribute::OFF);

    osg::LineWidth *lineWidth = new osg::LineWidth(2);
    geoState->setAttributeAndModes(lineWidth, osg::StateAttribute::ON);

    geode->setStateSet(geoState);

    geode->addDrawable(lines);
    cover->getScene()->addChild(geode);
#endif
}

void ViewerOsg::removeMovieTexture()
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::removeMovieTexture\n";
    std::list<movieImageData *>::iterator it = moviePs.begin();
    for (; it != moviePs.end(); it++)
    {
        movieImageData *movieProp = (*it);
        //movieProp->imageStream->unref();
        delete movieProp;
    }
    moviePs.clear();
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::removeMovieTexture\n";
}

ViewerOsg::~ViewerOsg()
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::~ViewerOsg\n";

    viewer = NULL;
    delete d_root;
    d_root = NULL;
    removeMovieTexture();
    if (VRMLRoot->getNumParents() > 0 && VRMLRoot->getParent(0) != NULL)
        VRMLRoot->getParent(0)->removeChild(VRMLRoot);
    VRMLRoot = NULL;

    cover->getScene()->removeChild(VRMLCaveRoot);
    VRMLCaveRoot = NULL;

    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::~ViewerOsg\n";
}

void ViewerOsg::removeChild(Object obj)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::removeChild\n";
    if (d_currentObject)
    {
        d_currentObject->removeChild((osgViewerObject *)obj);
    }
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::removeChild\n";
}

void ViewerOsg::restart()
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::restart\n";
    numVP = -1;
    delete d_root;
    d_root = new osgViewerObject(NULL);
    d_currentObject = d_root;
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::restart\n";
}

void ViewerOsg::initialize()
{
    if (cover->debugLevel(1))
        cerr << "ViewerOsg::initialize" << endl;
}

//
//  beginObject/endObject should correspond to grouping nodes.
//  Group-level scoping for directional lights, anchors, sensors
//  are handled here. Display lists can optionally be created
//  (but the retain flag is just a hint, not guaranteed). Retained
//  objects can be referred to later to avoid duplicating geometry.
//  OpenGL doesn't allow nested objects. The top-down approach of
//  putting entire groups in display lists is faster for static
//  scenes but uses more memory and means that if anything is changed,
//  the whole object must be tossed.
//  The bottom-up model wraps each piece of geometry in a dlist but
//  requires traversal of the entire scene graph to reference each dlist.
//  The decision about what groups to stuff in an object is punted to
//  the object itself, as it can decide whether it is mutable.
//
//  The OpenGL viewer never puts objects in display lists, so the
//  retain hint is ignored.

void ViewerOsg::setChoice(int which)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::setChoice\n";
    //fprintf(stderr, "setChoice(%d)\n", which);
    if (d_currentObject->whichChoice == which)
    {
        return;
    }

    Switch *pSwitch = dynamic_cast<Switch *>(d_currentObject->pNode.get());
    //fprintf(stderr, "setChoice(which=%d), pNode=%p, pSwitch=%p\n", which,
    //d_currentObject->pNode.get(), pSwitch);

    int saveChoice = d_currentObject->whichChoice;
    d_currentObject->whichChoice = which;
    //cerr << "setChoice" << which << "ChoiceMap[which]:" <<
    //d_currentObject->choiceMap[d_currentObject->whichChoice]<<
    //"numChildren:" << ((Switch*)(d_currentObject->pNode.get()))->getNumChildren()<< endl;
    if (pSwitch && which >= -2)
    {
        if (which == -1)
        {
//
#if 0
         // performer should now do intersections correctly
         d_currentObject->pNode.get()->setTravMask(PFTRAV_ISECT, 0x00, PFTRAV_DESCEND, PF_SET);
#endif

            pSwitch->setAllChildrenOff();
        }
        else if (d_currentObject->choiceMap[which] == -1 || pSwitch->getNumChildren() > (unsigned int)d_currentObject->choiceMap[which])
        {
            // it's important to compare signed to signed...
            if (d_currentObject->choiceMap[which] >= 0)
            {
#if 0
            // performer should now do intersections correctly
            d_currentObject->pNode.get()->setTravMask(PFTRAV_ISECT, 0x00, PFTRAV_DESCEND, PF_SET);
            ((Switch*)(d_currentObject->pNode.get()))->getChild(d_currentObject->choiceMap[d_currentObject->whichChoice])->setTravMask(PFTRAV_ISECT, 0xffff, PFTRAV_DESCEND | PFTRAV_SELF, PF_SET);
#endif
                pSwitch->setSingleChildOn(d_currentObject->choiceMap[which]);
            }
            else
            {
                pSwitch->setAllChildrenOn();
            }
        }
        else
        {
            d_currentObject->whichChoice = saveChoice;
        }
        pSwitch->dirtyBound();
    }
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::setChoice\n";
}

Viewer::Object ViewerOsg::beginObject(const char *name,
                                      bool retain,
                                      VrmlNode *node)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::beginObject\n";
    (void)name;
    (void)retain;

    //fprintf(stderr, "beginObject(name=%s, retain=%d, node=%p\n", name, (int)retain, node);
    if (node == NULL)
    {
        d_currentObject = d_root;
    }
    else if (d_currentObject->node == node)
    {
        d_currentObject->incLevel();
    }
    else
    {

        osgViewerObject *to = d_currentObject->getChild(node);
        if (!to)
        {
            to = new osgViewerObject(node);
            d_currentObject->addChild(to);
        }
        else
        {
            // make sure, we come back this way (needed for used nodes)
            //CERR << "already have child" << endl;
            to->parent = d_currentObject;
        }
        d_currentObject = to;
        
        d_currentObject->name = name;
    }
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::beginObject\n";
    return (Object)d_currentObject;
}

// End of group scope

void ViewerOsg::endObject()
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::endObject\n";
    if (d_currentObject->pNode.get() == NULL && d_currentObject->getLevel() == 0)
    {
        // this must be a group node (I think)
        if (d_currentObject->whichChoice != -2)
        {
            //cerr << "endObject(): added Switch: choice="  << d_currentObject->whichChoice << endl;
            if (cover->debugLevel(1))
                cerr << "Switch";
            d_currentObject->pNode = new Switch();
            d_currentObject->pNode->setName(d_currentObject->name.c_str());
        }
        else
        {
            d_currentObject->pNode = new Group();
            d_currentObject->pNode->setName(d_currentObject->name.c_str());
        }
        setModesByName();
        addToScene(d_currentObject);

        d_currentObject->addChildrensNodes();
        // add sensors that have not been added yet
        if (d_currentObject->sensor == NULL && d_currentObject->sensorObjectToAdd != NULL)
        {
            d_currentObject->sensor = new coSensiveSensor(d_currentObject->pNode.get(), d_currentObject, d_currentObject->sensorObjectToAdd, d_scene, VRMLRoot);
            sensors.push_back(d_currentObject->sensor);
            sensorList.append(d_currentObject->sensor);
        }
    }
    d_currentObject = d_currentObject->getParent();
    if (d_currentObject == NULL)
        d_currentObject = d_root;
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::endObject\n";
}

// These attributes need to be reset per geometry. Any attribute
// modified in a geometry function should be reset here. This is
// called after Appearance/Material has been set. Any attribute
// that can be modified by an Appearance node should not be modified
// here since these settings will be put in dlists with the geometry.

void ViewerOsg::beginGeometry()
{

    //cerr << "ViewerOsg::beginGeometry" << endl;
}

// This should be called BEFORE ending any dlists containing geometry,
// otherwise, attributes changed in the geometry insertion will impact
// the geometries rendered after the dlist is executed.

void ViewerOsg::endGeometry()
{

    //cerr << "ViewerOsg::endGeometry" << endl;
}

// Queries

void ViewerOsg::getViewerMat(double *M)
{
    Matrix m = cover->getViewerMat();
    for (int i = 0; i < 16; i++)
    {
        M[i] = m.ptr()[i];
    }
}

void ViewerOsg::getCurrentTransform(double *M)
{
    for (int i = 0; i < 16; i++)
    {
        M[i] = currentTransform.ptr()[i];
    }
}

void ViewerOsg::getVrmlBaseMat(double *M)
{
    for (int i = 0; i < 16; i++)
    {
        M[i] = vrmlBaseMat.ptr()[i];
    }
}

void ViewerOsg::removeSensor(coSensiveSensor *s)
{
    for (int i = 0; i < sensors.size(); i++)
    {
        if (sensors[i] == s)
        {
            sensors.erase(sensors.begin() + i);
            break;
        }
    }
    if (sensorList.find(s))
        sensorList.remove();
}

int ViewerOsg::getRenderMode()
{
    //cerr << "ViewerOsg::getRenderMode" << endl;
    return d_selectMode ? RENDER_MODE_PICK : RENDER_MODE_DRAW;
}

double ViewerOsg::getFrameRate()
{
    //cerr << "ViewerOsg::getFrameRate" << endl;
    return 1.0 / d_renderTime;
}

//

//
//  Geometry insertion.
//

Viewer::Object ViewerOsg::insertBumpMapping()
{
    if (d_currentObject->pNode.get())
    {
        //d_currentObject->pNode.get()->setStateSet(BumpCubeState.get());
        cerr << "ViewerOsg::insertBumpMapping oops" << endl;
#if 0
      coEffectHandler *eh = coEffectHandler::getEffectHandler(d_currentObject->pNode.get());
      coCgBumpMapping *bumpMapping;
      if((bumpMapping = (coCgBumpMapping*)eh->getEffect("CgBumpMapping"))==NULL)
      {
         bumpMapping = new coCgBumpMapping();
         eh->addEffect(bumpMapping);
      }
#endif
    }
    return (Object)0;
}

Viewer::Object ViewerOsg::insertWave(
    float Time,
    float Speed1,
    float Speed2,
    float Freq1,
    float Height1,
    float Damping1,
    float dir1[3],
    float Freq2,
    float Height2,
    float Damping2,
    float dir2[3],
    float *coeffSin,
    float *coeffCos,
    const char *fileName)
{
    (void)Time;
    (void)Speed1;
    (void)Speed2;
    (void)Freq1;
    (void)Height1;
    (void)Damping1;
    (void)dir1;
    (void)Freq2;
    (void)Height2;
    (void)Damping2;
    (void)dir2;
    (void)coeffSin;
    (void)coeffCos;
    (void)fileName;

    if (cover->debugLevel(2))
        cerr << "insertWave: not implemented" << endl;
    return (Object)0;
#if 0
   if(d_currentObject->pNode.get())
   {
      coEffectHandler *eh = coEffectHandler::getEffectHandler(d_currentObject->pNode);
      coCgWave *wave;
      if((wave = (coCgWave *)eh->getEffect("CgWave"))==NULL)
      {
         wave = new coCgWave();
         strcpy(wave->fileName,fileName);
         eh->addEffect(wave);
         cerr << "ViewerOsg::insertWave" << endl;
      }
      wave->Time = Time;
      wave->Speed1 = Speed1;
      wave->Speed2 = Speed2;
      wave->Freq1=Freq1;
      wave->Height1=Height1;
      wave->Damping1=Damping1;
      wave->dir1=Vec3(dir1[0], dir1[1], dir1[2]);
      wave->Freq2=Freq2;
      wave->Height2=Height2;
      wave->Damping2=Damping2;
      wave->dir2=Vec3(dir2[0], dir2[1], dir2[2]);
      int i;
      for(i=0;i<4;i++)
      {
         wave->coeffSin[i]=coeffSin[i];
         wave->coeffCos[i]=coeffCos[i];
      }
   }
#endif
}

Viewer::Object ViewerOsg::insertBackground(int /*nGroundAngles*/,
                                           float * /*groundAngle*/,
                                           float * /*groundColor*/,
                                           int /*nSkyAngles*/,
                                           float * /*skyAngle*/,
                                           float *skyColor,
                                           int * /*whc*/,
                                           unsigned char ** /*pixels*/)
{
    if (skyColor)
    {
        VRViewer::instance()->setClearColor(osg::Vec4(skyColor[0], skyColor[1], skyColor[2], 1));
    }
    //cerr << "ViewerOsg::insertBackground" << endl;
    return (Object)0;
}

Viewer::Object ViewerOsg::insertNode(Node *node)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::insertNode\n";
    d_currentObject->pNode = node;
    setModesByName();

    addToScene(d_currentObject);
    if (cover->debugLevel(1))
        cerr << "N";
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::insertNode\n";
    return (Object)d_currentObject;
}

Viewer::Object ViewerOsg::insertLineSet(int npoints,
                                        float *points,
                                        int nlines,
                                        int *lines,
                                        bool colorPerVertex,
                                        float *color,
                                        int componentsPerColor,
                                        int nci,
                                        int *ci, const char *name)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::insertLineSet\n";
    if (strncmp(name, "coOccluder", 10) == 0)
    {
        // create and occluder which will site along side the loadmodel model.
        osg::OccluderNode *occluderNode = new osg::OccluderNode;

        // create the convex planer occluder
        osg::ConvexPlanarOccluder *cpo = new osg::ConvexPlanarOccluder;

        // attach it to the occluder node.
        occluderNode->setOccluder(cpo);
        occluderNode->setName(name);

        d_currentObject->pNode = occluderNode;
        // set the occluder up for the front face of the bounding box.
        osg::ConvexPlanarPolygon &occluder = cpo->getOccluder();

        int lineStart = lines[0];
        int iStart = 0;
        int i;
        for (i = 0; i < nlines; i++)
        {
            if (lines[i] != -1)
            {
                int v = lines[i];
                if (i == iStart || v != lineStart)
                    occluder.add(Vec3(points[v * 3 + 0], points[v * 3 + 1], points[v * 3 + 2]));
            }
            else
            {
                lineStart = lines[++i];
                iStart = i;
                break;
            }
        }

        osg::ConvexPlanarPolygon *hole = new osg::ConvexPlanarPolygon;
        for (; i < nlines; i++)
        {
            if (lines[i] != -1)
            {
                int v = lines[i];
                if (i == iStart || v != lineStart)
                    hole->add(Vec3(points[v * 3 + 0], points[v * 3 + 1], points[v * 3 + 2]));
            }
            else
            {
                iStart = i + 1;
                lineStart = lines[i + 1];
                cpo->addHole(*hole);
                delete hole;
                hole = new osg::ConvexPlanarPolygon;
            }
        }
        delete hole;
    }
    else
    {
        //fprintf(stderr, "insertLineSet(np=%d, nl=%d, nci=%d)\n", npoints, nlines, nci);
        Geode *geode = new Geode();

        Geometry *geom = new Geometry();
        cover->setRenderStrategy(geom);
        geode->addDrawable(geom);
        StateSet *geoState = geode->getOrCreateStateSet();
        geoState->setNestRenderBins(false);
        setDefaultMaterial(geoState);
        geoState->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
        geode->setStateSet(geoState);

        Vec4Array *colArr = new Vec4Array();
        if ((componentsPerColor != 4) && (componentsPerColor != 3))
            cerr << "Error: wrong number of color components " << name << endl;

        // set up geometry and colorPerVertex color
        Vec3Array *vert = new Vec3Array;
        DrawArrayLengths *primitives = new DrawArrayLengths(PrimitiveSet::LINE_STRIP);
        int lineStart = 0;
        if (npoints > 0)
        {
            int ind = 0;
            for (int i = 0; i < nlines; i++)
            {
                if (lines[i] != -1)
                {
                    int v = lines[i];
                    vert->push_back(Vec3(points[v * 3 + 0], points[v * 3 + 1], points[v * 3 + 2]));
                    if (ci && (ind < nci))
                    {
                        if (componentsPerColor == 3)
                            colArr->push_back(Vec4(color[ci[ind] * 3], color[ci[ind] * 3 + 1], color[ci[ind] * 3 + 2], 1));
                        else
                            colArr->push_back(Vec4(color[ci[ind] * 4], color[ci[ind] * 4 + 1], color[ci[ind] * 4 + 2], color[ci[ind] * 4 + 3]));
                    }
                    if (colorPerVertex)
                        ++ind;
                }
                else
                {
                    primitives->push_back(i - lineStart);
                    lineStart = i + 1;
                    ++ind;
                }
            }
            if (lineStart != nlines + 1)
            {
                primitives->push_back(nlines - lineStart);
            }
            geom->setVertexArray(vert);
            geom->addPrimitiveSet(primitives);

            // associate colors if !colorPerVertex
            if (ci && !colorPerVertex)
            {
                for (int i = 0; i < primitives->size(); i++)
                {
                    if (componentsPerColor == 3)
                        colArr->push_back(Vec4(color[i * 3], color[i * 3 + 1], color[i * 3 + 2], 1));
                    else
                        colArr->push_back(Vec4(color[ci[i] * 4], color[ci[i] * 4 + 1], color[ci[i] * 4 + 2], color[ci[i] * 4 + 3]));
                }
            }
            if (ci)
            {
                geom->setColorArray(colArr);
                geom->setColorBinding(Geometry::BIND_PER_VERTEX);
            }
            else
            {
                geom->setColorBinding(Geometry::BIND_OFF);
            }
        }

        geode->setNodeMask(geode->getNodeMask() & (~Isect::Intersection));
        d_currentObject->pNode = geode;
        geom->setName(name);

        geode->setName(name);
        setModesByName(name);
        //d_currentObject->updateTexData(numActiveTextures);
        if (componentsPerColor == 4)
            d_currentObject->transparent = true;
        d_currentObject->updateMaterial();
        //d_currentObject->updateTexture();

        if (cover->debugLevel(1))
            cerr << "L";
        //d_currentObject->updateTMat();
        d_currentObject->updateBin();
    }
    addToScene(d_currentObject);
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::insertLineSet\n";
    return (Object)d_currentObject;
}

Viewer::Object ViewerOsg::insertPointSet(int npoints,
                                         float *points,
                                         float *color,
                                         int componentsPerColor)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::insertPointSet" << endl;
    Geode *geode = new Geode();

    Geometry *geom = new Geometry();
    cover->setRenderStrategy(geom);
    geode->addDrawable(geom);
    StateSet *geoState = geode->getOrCreateStateSet();
    setDefaultMaterial(geoState);
    geoState->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    osg::Point *point = new osg::Point();
    point->setSize(2.);
    geoState->setAttributeAndModes(point, StateAttribute::ON);
    geode->setStateSet(geoState);

    // set up geometry
    Vec3Array *vert = new Vec3Array;
    DrawArrayLengths *primitives = new DrawArrayLengths(PrimitiveSet::POINTS);
    primitives->push_back(npoints);
    for (int i = 0; i < npoints; i++)
    {
        vert->push_back(Vec3(points[i * 3 + 0], points[i * 3 + 1], points[i * 3 + 2]));
    }
    geom->setVertexArray(vert);
    geom->addPrimitiveSet(primitives);

    // associate colors
    if (color)
    {
        Vec4Array *colArr = new Vec4Array();
        if ((componentsPerColor != 4) && (componentsPerColor != 3))
            cerr << "Error: wrong number of color components "
                 << "PointSet" << endl;
        for (int i = 0; i < npoints; i++)
        {
            if (componentsPerColor == 3)
                colArr->push_back(Vec4(color[i * 3 + 0], color[i * 3 + 1], color[i * 3 + 2], 1));
            else
                colArr->push_back(Vec4(color[i * 4 + 0], color[i * 4 + 1], color[i * 4 + 2], color[i * 4 + 3]));
        }
        geom->setColorArray(colArr);
        geom->setColorBinding(Geometry::BIND_PER_VERTEX);
    }
    else
    {
        geom->setColorBinding(Geometry::BIND_OFF);
    }

    geode->setNodeMask(geode->getNodeMask() & (~Isect::Intersection));

    d_currentObject->pNode = geode;
    setModesByName();

    if (componentsPerColor == 4)
        d_currentObject->transparent = true;
    d_currentObject->updateMaterial();
    d_currentObject->updateBin();
    addToScene(d_currentObject);

    if (cover->debugLevel(1))
        cerr << "P";
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::insertPointSet" << endl;
    return (Object)d_currentObject;
}

void
ViewerOsg::applyTextureCoordinateGenerator(int textureUnit, int mode,
                                           float *parameter, int numParameter)
{
    if (mode == TEXTURE_COORDINATE_GENERATOR_MODE_SPHERE)
        d_currentObject->setTexGen(1, textureUnit, 1);
    else if (mode == TEXTURE_COORDINATE_GENERATOR_MODE_CAMERASPACEREFLECTIONVECTOR)
        d_currentObject->setTexGen(2, textureUnit, 1);
    else if (mode == TEXTURE_COORDINATE_GENERATOR_MODE_COORD)
        d_currentObject->setTexGen(3, textureUnit, 1);
    else if (mode == TEXTURE_COORDINATE_GENERATOR_MODE_CAMERASPACENORMAL)
        d_currentObject->setTexGen(4, textureUnit, 1);
    else if (mode == TEXTURE_COORDINATE_GENERATOR_MODE_COORD_EYE)
        d_currentObject->setTexGen(5, textureUnit, 1);
}
namespace
{
void printInts(int num, int *ints)
{
    copy(ints, ints + num, ostream_iterator<int>(cerr, ", "));
    cerr << endl;
}

void printInts(const vector<int> &ints)
{
    copy(ints.begin(), ints.end(), ostream_iterator<int>(cerr, ", "));
    cerr << endl;
}

void printVec3s(int num, float *values)
{
    for (int i = 0; i < num; i++)
    {
        cerr << "(" << values[i * 3] << ", " << values[i * 3 + 1] << ", " << values[i * 3 + 2] << ")\t ";
    }
    cerr << endl;
}
}

#define HAVE_TO_INSERT(perVertex, endFace) (!(perVertex) || !(endFace))

namespace
{
template <typename A>
A *createPolygonsIndexed(DrawArrayLengths *primitives, const vector<int> &faces)
{
    int polyStart = 0;
    A *indices = new A();
    for (int i = 0; i < faces.size(); i++)
    {
        if (faces[i] != -1)
        {
            indices->push_back(faces[i]);
            numVert++;
        }
        else
        {
            if (i - polyStart < 3)
            {
                cerr << "Degenerated Poly !" << endl;
            }
            primitives->push_back(i - polyStart);
            polyStart = i + 1;
            numPoly++;
        }
    }
    return indices;
}

template <typename A>
A *createAttributesIndexed(bool attributesPerPrimitive, const vector<int> &faces, const vector<int> &attributeIndices)
{
    A *indices = new A();
    if (attributesPerPrimitive)
    {
        int faceIndex = 0;
        for (int i = 0; i < faces.size(); i++)
        {
            if (faces[i] == -1)
            {
                faceIndex++;
            }
            else
            {
                indices->push_back(attributeIndices[faceIndex]);
            }
        }
    }
    else
    {
        for (int i = 0; i < attributeIndices.size(); i++)
        {
            if (attributeIndices[i] != -1)
                indices->push_back(attributeIndices[i]);
        }
    }
    return indices;
}

void fillIndexVec(int mask, bool isPerPrimitive, int nindices, int *indices, vector<int> &indicesVec)
{
    if ((mask & Viewer::MASK_CCW) || isPerPrimitive)
    {
        copy(indices, indices + nindices, back_inserter(indicesVec));
        return;
    }

    vector<int> polyIndices;
    for (int i = 0; i < nindices; ++i)
    {
        if (indices[i] == -1)
        {
            reverse(polyIndices.begin(), polyIndices.end());
            copy(polyIndices.begin(), polyIndices.end(), back_inserter(indicesVec));
            indicesVec.push_back(-1);
            polyIndices.clear();
        }
        else
        {
            polyIndices.push_back(indices[i]);
        }
    }
}
}

// There are too many arguments to this...

Viewer::Object
ViewerOsg::insertShell(unsigned int mask,
                       int npoints,
                       float *points,
                       int nfaces,
                       int *faces, // face list (-1 ends each face)
                       float **tc, // texture coordinates for all units
                       int *ntci, // # of texture coordinate indices for all units
                       int **tci, // texture coordinate indices for all units

                       float *normal, // normals
                       int nni, // # of normal indices
                       int *ni, // normal indices
                       float *color, // colors
                       int nci,
                       int *ci,
                       const char *objName,
                       int *texCoordGeneratorMode,
                       float **texCoordGeneratorParameter,
                       int *numTexCoordGeneratorParameter)
{
    if (cover->debugLevel(5))
        cerr << "InsertShell " << objName << "\n";
    if (points == NULL)
    {
        if (cover->debugLevel(3))
            fprintf(stderr, "insertShell: points NULL\n");
        return (Object)d_currentObject;
    }
    int componentsPerColor = 3; // number of components per color (3 for Color node, 4 for ColorRGBA node)
    if (mask & MASK_COLOR_RGBA)
        componentsPerColor = 4;
    // optimisation for animated objects: only set new vertex coordinates
    if (d_currentObject->pNode.get() && (mask & MASK_CONVEX) && !strncmp(objName, "Animated", 8))
    {
        Geode *geode = dynamic_cast<Geode *>(d_currentObject->pNode.get());
        if (geode)
        {
            Geometry *geom = dynamic_cast<Geometry *>(geode->getDrawable(0));
            if (geom)
            {
                Vec3Array *vert = dynamic_cast<Vec3Array *>(geom->getVertexArray());
                if (vert)
                {
                    //int polyStart=0;
                    int vertnum = 0;
                    for (int i = 0; i < nfaces; i++)
                    {
                        if (faces[i] != -1)
                        {
                            int v = faces[i];
                            (*vert)[vertnum].set(Vec3(points[v * 3 + 0], points[v * 3 + 1], points[v * 3 + 2]));
                            vertnum++;
                        }
                        else
                        {
                            //primitives->push_back(i-polyStart);
                            //polyStart=i+1;
                        }
                    }
                    if (ni)
                    {
                        Vec3Array *normalArray = dynamic_cast<Vec3Array *>(geom->getNormalArray());
                        if (normalArray)
                        {
                            int normalind = 0;
                            for (int i = 0; i < nni; i++)
                            {
                                if (HAVE_TO_INSERT(mask & MASK_NORMAL_PER_VERTEX, faces[i] == -1))
                                {
                                    if (mask & MASK_CCW)
                                    {
                                        (*normalArray)[normalind].set(Vec3(normal[ni[i] * 3], normal[ni[i] * 3 + 1], normal[ni[i] * 3 + 2]));
                                    }
                                    else
                                    {
                                        (*normalArray)[normalind].set(Vec3(normal[ni[i] * 3] * -1, normal[ni[i] * 3 + 1] * -1, normal[ni[i] * 3 + 2] * -1));
                                    }
                                    normalind++;
                                }
                            }
							normalArray->dirty();
                        }
                    }
					vert->dirty();
                    geom->dirtyDisplayList();

                    return (Object)d_currentObject;
                }
            }
        }
    }

    osg::Node::NodeMask nodeMask = ~Isect::Intersection;

    if (d_currentObject->pNode.get())
    {
        // save old Mask if object already exists
        nodeMask = d_currentObject->pNode->getNodeMask();

        if (auto geode = d_currentObject->pNode->asGeode())
        {
            for (unsigned i=0; i<geode->getNumDrawables(); ++i)
            {
                if (auto geom = geode->getDrawable(i)->asGeometry())
                {
                    auto vert = geom->getVertexArray();
                    if (vert)
                        numVert -= vert->getNumElements();
                    for (unsigned j=0; j<geom->getNumPrimitiveSets(); ++j)
                    {
                        auto prim = geom->getPrimitiveSet(j);
                        if (prim)
                            numPoly -= prim->getNumPrimitives();
                    }
                }
            }
        }

        while (d_currentObject->pNode->getNumParents())
        {
            Group *parentNode = d_currentObject->pNode->getParent(0);
            if (!parentNode)
                break;
            parentNode->removeChild(d_currentObject->pNode.get());
        }
        d_currentObject->pNode = NULL;
    }

    Geode *geode = new Geode();

    //geode->setName(objName);

    ref_ptr<Geometry> geom = new Geometry();
    cover->setRenderStrategy(geom);

    StateSet *geoState = geode->getOrCreateStateSet();
    setDefaultMaterial(geoState);
    // backFaceCulling nur dann, wenn es im CoviseConfig enabled ist
    if (backFaceCulling && (mask & MASK_SOLID))
    {
        CullFace *cullFace = new CullFace(); // da viele Modelle backface Culling nicht vertragen (nicht richtig modelliert sind)
        cullFace->setMode(CullFace::BACK);
        geoState->setAttributeAndModes(cullFace, StateAttribute::ON);
    }

    geode->setStateSet(geoState);

    bool indexed = true, tris = true, quads = true;
    std::vector<int *> indices;
    int numInd = 0;
    if (nni > 0)
    {
        numInd = nni;
        indices.push_back(ni);
    }
    if (nci > 0)
    {
        if (numInd == 0)
            numInd = nci;
        else if (numInd != nci)
            indexed = false;
        indices.push_back(ci);
    }
    for (int i=0; i<numTextures; ++i)
    {
        if (ntci[i] > 0)
        {
            if (numInd == 0)
                numInd = ntci[i];
            else if (numInd != ntci[i])
                indexed = false;
            indices.push_back(tci[i]);
        }
    }
    int nverts = 0;
    int n = 0;
    int numFaces = 0;
    for (int i=0; i<nfaces; ++i)
    {
        if (faces[i] == -1 && nverts > 0)
        {
            ++numFaces;
            if (nverts != 3)
                tris = false;
            if (nverts != 4)
                quads = false;
            nverts = 0;
        }
        else
        {
            if (indexed)
            {
                int idx = faces[i];
                for (auto i: indices)
                {
                    if (i[n] != idx)
                    {
                        indexed = false;
                        break;
                    }
                }
            }

            ++nverts;
            ++n;
        }

        if (!indexed && !tris && !quads)
            break;
    }

    // set up geometry
    DrawArrayLengths *polygons = nullptr;
    DrawArrays *primitives = nullptr;
    if (tris)
        primitives = new DrawArrays(PrimitiveSet::TRIANGLES, 0, numFaces*3);
    else if (quads)
        primitives = new DrawArrays(PrimitiveSet::QUADS, 0, numFaces*4);
    else
        polygons = new DrawArrayLengths(PrimitiveSet::POLYGON);
    Vec3Array *vert = new Vec3Array;
    int polyStart = 0;
    if (mask & MASK_CCW)
    {
        for (int i = 0; i < nfaces; i++)
        {
            if (faces[i] != -1)
            {
                int v = faces[i] * 3;
                vert->push_back(Vec3(points[v], points[v + 1], points[v + 2]));
                numVert++;
            }
            else
            {
                if (polygons)
                    polygons->push_back(i - polyStart);
                polyStart = i + 1;
                numPoly++;
            }
        }
    }
    else
    {
        int i = 0;
        int polyEnd = 0;
        while (i < nfaces)
        {
            polyStart = i;
            while (faces[i] != -1 && i < nfaces)
            {
                i++;
            }
            polyEnd = i;
            if (polygons)
                polygons->push_back(i - polyStart);
            numPoly++;
            while (i > polyStart)
            {
                i--;
                int v = faces[i] * 3;
                vert->push_back(Vec3(points[v], points[v + 1], points[v + 2]));
                numVert++;
            }
            i = polyEnd + 1;
        }
    }
    geom->setVertexArray(vert);
    if (polygons)
        geom->addPrimitiveSet(polygons);
    else
        geom->addPrimitiveSet(primitives);

    // associate colors
    if (ci && nci>1)
    {
        Vec4Array *colArr = new Vec4Array();
        if ((componentsPerColor != 4) && (componentsPerColor != 3))
            cerr << "Error: wrong number of color components " << objName << endl;
        if (mask & MASK_CCW || !(mask & MASK_COLOR_PER_VERTEX))
        {
            int ind = 0;
            for (int i = 0; i < nfaces; ++i)
            {
                if (ind >= nci)
                {
                    cerr << "Error: not enough color indices " << objName << endl;
                    break;
                }
                if (faces[i] != -1)
                {
                    if (componentsPerColor == 3)
                        colArr->push_back(Vec4(color[ci[ind] * 3], color[ci[ind] * 3 + 1], color[ci[ind] * 3 + 2], 1));
                    else
                        colArr->push_back(Vec4(color[ci[ind] * 4], color[ci[ind] * 4 + 1], color[ci[ind] * 4 + 2], color[ci[ind] * 4 + 3]));
                }
                if ((mask & MASK_COLOR_PER_VERTEX) || faces[i] == -1)
                {
                    ++ind;
                }
            }
        }
        else
        {
            int i = 0;
            int polyEnd = 0;
            int polyStart = 0;
            while (i < nci)
            {
                polyStart = i;
                while (faces[i] != -1 && i < nci)
                {
                    i++;
                }
                polyEnd = i;
                while (i > polyStart)
                {
                    i--;
                    int v = ci[i] * componentsPerColor;
                    if (componentsPerColor == 3)
                        colArr->push_back(Vec4(color[v], color[v + 1], color[v + 2], 1));
                    else
                        colArr->push_back(Vec4(color[v], color[v + 1], color[v + 2], color[v + 3]));
                }
                i = polyEnd + 1;
            }
        }
        geom->setColorArray(colArr);
        geom->setColorBinding(Geometry::BIND_PER_VERTEX);
    }
    else
    {
        geom->setColorBinding(Geometry::BIND_OFF);
    }

    // associate normals
    if (ni)
    {
        Vec3Array *normalArray = new Vec3Array();
        if (mask & MASK_CCW || !(mask & MASK_NORMAL_PER_VERTEX))
        {
            int ind = 0;
            for (int i = 0; i < nfaces; ++i)
            {
                if (ind >= nni)
                {
                    cerr << "Error: not enough normal indices " << objName << endl;
                    break;
                }
                if (faces[i] != -1)
                {
                    if (mask & MASK_CCW)
                        normalArray->push_back(Vec3(normal[ni[ind] * 3], normal[ni[ind] * 3 + 1], normal[ni[ind] * 3 + 2]));
                    else
                        normalArray->push_back(Vec3(normal[ni[ind] * 3] * -1, normal[ni[ind] * 3 + 1] * -1, normal[ni[ind] * 3 + 2] * -1));
                }
                if ((mask & MASK_NORMAL_PER_VERTEX) || faces[i] == -1)
                {
                    ++ind;
                }
            }
        }
        else
        {
            int i = 0;
            int polyEnd = 0;
            int polyStart = 0;
            while (i < nni)
            {
                polyStart = i;
                while (faces[i] != -1 && i < nni)
                {
                    i++;
                }
                polyEnd = i;
                while (i > polyStart)
                {
                    i--;
                    int v = ni[i] * 3;
                    normalArray->push_back(Vec3(normal[v] * -1, normal[v + 1] * -1, normal[v + 2] * -1));
                }
                i = polyEnd + 1;
            }
        }
        geom->setNormalArray(normalArray);
        geom->setNormalBinding(Geometry::BIND_PER_VERTEX);
    }

    // texture coords
    int numActiveTextures = 0;
    bool hasTexCoordGenerator = false;
    if (texCoordGeneratorMode != NULL)
        hasTexCoordGenerator = true;
    // not a long term solution, only for testing
    for (int unit = 0; unit < numTextures; unit++)
    {
        if (hasTexCoordGenerator)
        {
            if (texCoordGeneratorMode[unit] != 0)
            {
                applyTextureCoordinateGenerator(unit, texCoordGeneratorMode[unit],
                                                texCoordGeneratorParameter[unit],
                                                numTexCoordGeneratorParameter[unit]);
                numActiveTextures++;
                continue;
            }
        }

        int ntcoords = 0;
        if (ntci[unit])
        {
            numActiveTextures = unit + 1;
        }
        else
        {
            continue;
        }
        if (tci && tci[unit] && tc[unit])
        {
            for (int i = 0; i < ntci[unit]; i++)
            {
                if (tci[unit][i] > ntcoords)
                    ntcoords = tci[unit][i];
            }
            ntcoords++;
        }

        if (ntcoords > 0)
        {
            Vec2Array *tcArray = new Vec2Array();
			tcArray->reserve(ntci[unit]);
            int j = 0;
            if (mask & MASK_CCW)
            {
                for (int i = 0; i < ntci[unit]; i++)
                {
                    if (faces[i] != -1)
                    {
                        tcArray->push_back(Vec2(tc[unit][tci[unit][i] * 2], tc[unit][tci[unit][i] * 2 + 1]));
                        j++;
                    }
                }
            }
            else
            {
                int i = 0;
                int startPoly = 0;
                int endPoly = 0;
                while (i < ntci[unit])
                {
                    startPoly = i;
                    while (faces[i] != -1 && i < ntci[unit])
                        i++;
                    endPoly = i;
                    while (i > startPoly)
                    {
                        i--;
						tcArray->push_back(Vec2(tc[unit][tci[unit][i] * 2], tc[unit][tci[unit][i] * 2 + 1]));
                        j++;
                    }
                    i = endPoly + 1;
                }
            }
            geom->setTexCoordArray(unit, tcArray);
        }
        
    }

    if (strncmp(objName, "Animated", 8) != 0)
    {
        if ((!(mask & MASK_CONVEX)) || reTesselate)
        {
            osgUtil::Tessellator tess;
            tess.retessellatePolygons(*geom);
            //cerr << "Convex";
        }
        
        geom->setName(objName);
        geode->addDrawable(geom.get());
        
        if(((unsigned int)nfaces > SubdivideThreshold) && !((d_currentObject->pNode.get() && (mask & MASK_CONVEX) && !strncmp(objName, "Animated", 8))))
        {
            splitGeometry(geode,SubdivideThreshold);
            if (cover->debugLevel(2))
                cerr << "-" << nfaces  << "/" << geode->getNumDrawables() << ":";
        }
        if (cover->debugLevel(1))
        {
            cerr << "P";
            if (tris)
                cerr << "t";
            if (quads)
                cerr << "q";
            if (indexed)
                cerr << "i";
        }
    }
    else
    {
        geom->setName(objName);
        geode->addDrawable(geom.get());
    }

#if (OSG_VERSION_GREATER_OR_EQUAL(3, 4, 0))
    osg::ref_ptr<osg::KdTreeBuilder> kdb = new osg::KdTreeBuilder;
    for(unsigned int i=0;i<geode->getNumDrawables();i++)
    {
        osg::Geometry *geo = dynamic_cast<osg::Geometry *>(geode->getDrawable(i));
        kdb->apply(*geo);
    }
#endif

    geode->setName(objName);
    geode->setNodeMask(nodeMask);

    d_currentObject->pNode = geode;
    setModesByName(objName);
    d_currentObject->updateTexData(numActiveTextures);
    if (componentsPerColor == 4)
        d_currentObject->transparent = true;
    d_currentObject->updateMaterial();
    d_currentObject->updateTexture();

    d_currentObject->updateTMat();
    d_currentObject->updateBin();

    addToScene(d_currentObject);

    d_currentObject->nodeType = NODE_IFS;

    if (cover->debugLevel(5))
        cerr << "   return current Object\n";

    return (Object)d_currentObject;
}

void ViewerOsg::splitGeometry(osg::Geode *geode, unsigned int threshold)
{
    bool didSplit=true;
    int numLevels = 0;
    while(didSplit && numLevels < 8)
    {
        didSplit = false;
        numLevels++;
        std::vector<osg::ref_ptr<osg::Geometry>> newDrawables;
        std::vector<osg::Drawable *> oldDrawables;
        for(unsigned int i=0;i<geode->getNumDrawables();i++)
        {
            osg::Drawable *d = geode->getDrawable(i);
            osg::Geometry *geom = dynamic_cast<osg::Geometry *>(d);
            if(geom!=NULL)
            {
                osg::PrimitiveSet *ps = geom->getPrimitiveSet(0);
                ps->getNumPrimitives();
                if(ps!=NULL)
                {
                    if(ps->getNumPrimitives()>threshold)
                    {
                        didSplit=true;
                        osg::Geometry *geometries[2];
                        splitDrawable(geometries,geom);
                        oldDrawables.push_back(d);
                        for(int n=0;n<2;n++)
                        {
                            if(geometries[n]!=NULL)
                            {
                                newDrawables.push_back(geometries[n]);
                            }
                        }
                    }
                }
            }
        }
        
        for(auto it = oldDrawables.begin(); it != oldDrawables.end();it++)
        {
            geode->removeDrawable(*it);
        }
        for(auto it = newDrawables.begin(); it != newDrawables.end();it++)
        {
            geode->addDrawable(*it);
        }
        newDrawables.clear();
    }
}

void ViewerOsg::splitDrawable(osg::Geometry *(&geometries)[2],osg::Geometry *geom)
{
    osg::PrimitiveSet *ps = geom->getPrimitiveSet(0);
    osg::DrawArrayLengths *polygons = dynamic_cast<osg::DrawArrayLengths *>(ps);
    osg::DrawArrays *drawarray = dynamic_cast<osg::DrawArrays *>(ps);

    Vec3Array *vertexArray = dynamic_cast<Vec3Array *>(geom->getVertexArray());
    Vec3Array *normalArray = dynamic_cast<Vec3Array *>(geom->getNormalArray());
    Vec4Array *colorArray = dynamic_cast<Vec4Array *>(geom->getColorArray());
    int numTexCoords = geom->getNumTexCoordArrays();
    Vec2Array **tcArray = new Vec2Array *[numTexCoords];
    for(int i=0;i<numTexCoords;i++)
    {
        tcArray[i] = dynamic_cast<Vec2Array *>(geom->getTexCoordArray(i));
    }
    osg::BoundingBox bb;
    bb.expandBy(geom->getBound());
    float xs = bb.xMax() - bb.xMin(); 
    float ys = bb.yMax() - bb.yMin(); 
    float zs = bb.zMax() - bb.zMin(); 
    float xm = bb.xMin() + xs/2.0;
    float ym = bb.yMin() + ys/2.0;
    float zm = bb.zMin() + zs/2.0;
    int index= 0;
    float splitAt;
    if(xs > ys && xs > zs)
    {
        index = 0;
        splitAt = xm;
    }
    else if(ys > xs && ys > zs)
    {
        index = 1;
        splitAt = ym;
    }
    else
    {
        index = 2;
        splitAt = zm;
    }

    osg::DrawArrayLengths *polygonsMin = nullptr, *polygonsMax = nullptr;
    osg::DrawArrays *primitivesMin = nullptr, *primitivesMax = nullptr;
    if(polygons || drawarray)
    {
        // count new num vertices and polygons
        int nv=0;
        int np=0;
        int vpf = 1;
        if (drawarray)
        {
            if (drawarray->getMode() == osg::PrimitiveSet::TRIANGLES)
                vpf = 3;
            if (drawarray->getMode() == osg::PrimitiveSet::QUADS)
                vpf = 4;
        }

        //std::cerr << "S" << vpf;

        int vertNum = drawarray ? drawarray->getFirst() : 0;
        for(unsigned int i=0;i<ps->getNumPrimitives();i++)
        {
            int v = polygons ? (*polygons)[i] : vpf;
            if((*vertexArray)[vertNum][index] < splitAt)
            {
                nv +=v;
                np++;
            }
            vertNum += v;
        }

        if (polygons)
        {
            polygonsMin = new osg::DrawArrayLengths(PrimitiveSet::POLYGON);
            polygonsMax = new osg::DrawArrayLengths(PrimitiveSet::POLYGON);
            polygonsMin->reserve(np);
            polygonsMax->reserve(polygons->getNumPrimitives() - np);
        }
        else if (drawarray)
        {
            primitivesMin = new osg::DrawArrays(drawarray->getMode(), 0, nv);
            primitivesMax = new osg::DrawArrays(drawarray->getMode(), 0, drawarray->getCount()-nv);
        }

        Vec3Array *vertexArrayMin = new Vec3Array();
        Vec3Array *vertexArrayMax = new Vec3Array();
        vertexArrayMin->reserve(nv);
        vertexArrayMax->reserve(vertexArray->getNumElements() - nv);
        Vec3Array *normalArrayMin=NULL;
        Vec3Array *normalArrayMax=NULL;
        if(normalArray!=NULL)
        {
            normalArrayMin = new Vec3Array();
            normalArrayMax = new Vec3Array();
            normalArrayMin->reserve(nv);
            normalArrayMax->reserve(normalArray->getNumElements() - nv);
        }
        
        Vec4Array *colorArrayMin=NULL;
        Vec4Array *colorArrayMax=NULL;
        if(colorArray!=NULL)
        {
            colorArrayMin = new Vec4Array();
            colorArrayMax = new Vec4Array();
            colorArrayMin->reserve(nv);
            colorArrayMax->reserve(colorArray->getNumElements() - nv);
        }

        Vec2Array **tcArrayMin =new Vec2Array *[numTexCoords];
        Vec2Array **tcArrayMax =new Vec2Array *[numTexCoords];
        for(int i=0;i<numTexCoords;i++)
        {
            tcArrayMin[i] = new Vec2Array();
            tcArrayMax[i] = new Vec2Array();
            tcArrayMin[i]->reserve(nv);
            tcArrayMax[i]->reserve(tcArray[i]->getNumElements() - nv);
            tcArrayMin[i]->reserve(nv);
            tcArrayMax[i]->reserve(tcArray[i]->getNumElements() - nv);
        }
        nv=0;
        vertNum = drawarray ? drawarray->getFirst() : 0;
        for(unsigned int i=0;i<ps->getNumPrimitives();i++)
        {
            int v = polygons ? (*polygons)[i] : vpf;
            if((*vertexArray)[vertNum][index] < splitAt)
            {
                if (polygonsMin)
                    polygonsMin->push_back(v);
                for(int n=0;n<v;n++)
                {
                    vertexArrayMin->push_back((*vertexArray)[nv]);
                    if(colorArrayMin!=NULL)
                    {
                        colorArrayMin->push_back((*colorArray)[nv]);
                    }
                    if(normalArrayMin!=NULL)
                    {
                        normalArrayMin->push_back((*normalArray)[nv]);
                    }
                    for(int m=0;m<numTexCoords;m++)
                    {
                        tcArrayMin[m]->push_back((*tcArray[m])[nv]);
                    }
                    nv++;
                }
                np++;
            }
            else
            {
                if (polygonsMax)
                    polygonsMax->push_back(v);
                for(int n=0;n<v;n++)
                {
                    vertexArrayMax->push_back((*vertexArray)[nv]);
                    if(colorArrayMax!=NULL)
                    {
                        colorArrayMax->push_back((*colorArray)[nv]);
                    }
                    if(normalArrayMax!=NULL)
                    {
                        normalArrayMax->push_back((*normalArray)[nv]);
                    }
                    for(int m=0;m<numTexCoords;m++)
                    {
                        tcArrayMax[m]->push_back((*tcArray[m])[nv]);
                    }
                    nv++;
                }
                np++;
            }
            vertNum += v;
        }
        geometries[0] = new osg::Geometry;
        geometries[1] = new osg::Geometry;
        geometries[0]->setStateSet(geom->getStateSet());
        geometries[1]->setStateSet(geom->getStateSet());
        if (polygons)
        {
            geometries[0]->addPrimitiveSet(polygonsMin);
            geometries[1]->addPrimitiveSet(polygonsMax);
        }
        else
        {
            geometries[0]->addPrimitiveSet(primitivesMin);
            geometries[1]->addPrimitiveSet(primitivesMax);
        }
        geometries[0]->setVertexArray(vertexArrayMin);
        geometries[1]->setVertexArray(vertexArrayMax);
        if(normalArrayMin!=NULL)
        {
            geometries[0]->setNormalArray(normalArrayMin);
            geometries[0]->setNormalBinding(Geometry::BIND_PER_VERTEX);
        }
        if(normalArrayMax!=NULL)
        {
            geometries[1]->setNormalArray(normalArrayMax);
            geometries[1]->setNormalBinding(Geometry::BIND_PER_VERTEX);
        }
        if(colorArrayMin!=NULL)
        {
            geometries[0]->setColorArray(colorArrayMin);
            geometries[0]->setColorBinding(Geometry::BIND_PER_VERTEX);
        }
        if(colorArrayMax!=NULL)
        {
            geometries[1]->setColorArray(colorArrayMax);
            geometries[1]->setColorBinding(Geometry::BIND_PER_VERTEX);
        }
        for(int m=0;m<numTexCoords;m++)
        {
            geometries[0]->setTexCoordArray(m,tcArrayMin[m]);
            geometries[1]->setTexCoordArray(m,tcArrayMax[m]);
        }
        delete[] tcArrayMin;
        delete[] tcArrayMax;
    }
    delete[] tcArray;
}

void ViewerOsg::setDefaultMaterial(StateSet *geoState)
{

    geoState->setNestRenderBins(false);
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::setDefaultMaterial\n";
    if (globalmtl.get() == NULL)
    {
        globalmtl = new Material;
        globalmtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
        globalmtl->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.0));
        globalmtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0));
        globalmtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0));
        globalmtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0));
        globalmtl->setShininess(Material::FRONT_AND_BACK, 16.0f);
    }

    geoState->setAttributeAndModes(globalmtl.get(), StateAttribute::ON);

    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::setDefaultMaterial\n";
}

void ViewerOsg::addObj(osgViewerObject *obj, osg::Group *group)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::addObj\n";
    LightSource *pLightSource = dynamic_cast<LightSource *>(obj->getNode());
    if (pLightSource)
    {
        obj->nodeType = NODE_LIGHT;
        if (cover->debugLevel(1))
        {
            if (cover->debugLevel(1))
                cerr << "L";
            cerr.flush();
        }

        Group *group = dynamic_cast<Group *>(obj->parent->pNode.get());
        if (obj->lightedNode == NULL)
        {
            obj->lightedNode = group;
        }
        coVRLighting::instance()->addLight(pLightSource, group, obj->lightedNode.get(), obj->node->name());
        const VrmlField *intensity = obj->node->getField("intensity");
        if (intensity)
        {
            coVRLighting::instance()->switchLight(pLightSource, intensity->toSFFloat()->get() != 0.);
        }
    }
    else
    {
        group->addChild(obj->getNode());
    }
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::addObj\n";
}

bool ViewerOsg::addToScene(osgViewerObject *obj)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::addToScene\n";
    bool ret = true;
    obj->viewer = this;

    if (obj->billBoard.get())
    {
        obj->billBoard->addChild(obj->pNode.get());
    }

    if (d_currentObject->parent)
    {
        if (d_currentObject->parent->pNode.get())
        {
            if (obj->node && strncmp(obj->node->name(), "StaticCave", 10) == 0)
            {

                obj->setRootNode(VRMLCaveRoot);
                addObj(obj, VRMLCaveRoot);
                if (cover->debugLevel(1))
                    cerr << "add node " << obj->node->name() << " to Cave root!" << endl;
                if (cover->debugLevel(1))
                    cerr << "_SC_";
            }
            else
            {
                Group *pGroup = dynamic_cast<Group *>(d_currentObject->parent->pNode.get());
                if (pGroup)
                {
                    addObj(obj, pGroup);
                }
                else
                {
                    if (cover->debugLevel(2))
                        CERR << "Kacke" << endl;
                }
            }
            //cerr << "add a Child to Parent" << d_currentObject->node->name() << endl;
            if (d_currentObject->parent->whichChoice >= 0)
            {
                //cerr << "addSwitchChild" << d_currentObject->parent->whichChoice <<"\n";

                Group *pGroup = dynamic_cast<Group *>(d_currentObject->parent->pNode.get());
                if (pGroup)
                {
                    d_currentObject->parent->choiceMap[d_currentObject->parent->whichChoice] = pGroup->getChildIndex(obj->getNode());
                }
                else
                {
                    if (cover->debugLevel(3))
                        fprintf(stderr, "Could not cast %p to Group*\n", d_currentObject->parent->pNode.get());
                }

                Switch *pSwitch = dynamic_cast<Switch *>(d_currentObject->parent->pNode.get());
                if (pSwitch)
                {
                    pSwitch->setSingleChildOn(d_currentObject->parent->choiceMap[d_currentObject->parent->whichChoice]);
                }
                else
                {
                    if (cover->debugLevel(2))
                        CERR << "Oberkacke" << endl;
                }
//cerr << "choiceMap : " <<  d_currentObject->parent->choiceMap[d_currentObject->parent->whichChoice] << endl;

// XXX
#if 0
            d_currentObject->parent->pNode.get()->setTravMask(PFTRAV_ISECT, 0x00, PFTRAV_DESCEND, PF_SET);
            ((Switch*)(d_currentObject->parent->pNode.get()))->getChild(d_currentObject->parent->choiceMap[d_currentObject->parent->whichChoice])->setTravMask(PFTRAV_ISECT, 0x01, PFTRAV_DESCEND | PFTRAV_SELF, PF_SET);
#endif
            }
        }
        else
        {
            ret = false;
            obj->haveToAdd++;
        }
    }
    else
    {
        //cerr << "add a Child to Root" << obj->node->name() << endl;
        if (obj->node && strncmp(obj->node->name(), "StaticCave", 10) == 0)
        {
            obj->setRootNode(VRMLCaveRoot);
            addObj(obj, VRMLCaveRoot);
            //cerr << "add node " << obj->node->name() << " to Cave root!" << endl;
            //cerr << "_SC_";
        }
        else
        {
            addObj(obj, VRMLRoot);
        }
    }
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::addToScene\n";

    return ret;
}

// Not fully implemented... need font, extents
Viewer::Object ViewerOsg::insertText(int *justify,
                                     float fontSize,
                                     int stringNumber,
                                     char **strings,
                                     const char *objName)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::insertText\n";

    if (objName != NULL)
    {
        d_currentObject->node->setName(objName);
    }

    if (NULL == font)
    {
        if (cover->debugLevel(2))
            fprintf(stderr, "insertText(%s): font==NULL\n", stringNumber > 0 ? strings[0] : "");
        return 0L;
    }

    osg::ref_ptr<Geode> pGeode = dynamic_cast<Geode *>(d_currentObject->pNode.get());
    if (d_currentObject->pNode.get())
    {
        if (!pGeode)
        {
            d_currentObject->pNode->unref();
            d_currentObject->pNode = NULL;
        }
    }

    if (!pGeode)
    {
        pGeode = new Geode();
    }
    pGeode->setNodeMask(pGeode->getNodeMask() & (~Isect::Intersection));

    for (unsigned int i = 0; i < (unsigned int)stringNumber; ++i)
    {
        osg::ref_ptr<osgText::Text> pText = NULL;
        if (i < pGeode->getNumDrawables())
        {
            pText = dynamic_cast<osgText::Text *>(pGeode->getDrawable(i));
            if (!pText)
            {
                pText = new osgText::Text();
                pGeode->setDrawable(i, pText);
            }
        }
        if (!pText)
        {
            pText = new osgText::Text;
            pGeode->setDrawable(i, pText);
            pGeode->addDrawable(pText);
            pText->setFont(font);
        }

        if (justify[0] > 0)
            pText->setAlignment(osgText::Text::LEFT_BASE_LINE);
        else if (justify[0] == 0)
            pText->setAlignment(osgText::Text::CENTER_BASE_LINE);
        else
            pText->setAlignment(osgText::Text::RIGHT_BASE_LINE);

        std::string vrmlText(strings[i]);
        osgText::String osgStr(vrmlText, osgText::String::ENCODING_UTF8);
        pText->setText(osgStr);

        if (d_currentObject->mtl.get() != NULL)
        {
            const osg::Vec4f dfColor = d_currentObject->mtl->getDiffuse(osg::Material::FRONT);
            pText->setColor(dfColor);

            osg::ref_ptr<osg::StateSet> stateSet = NULL;
            if ((stateSet = pGeode->getStateSet()) == NULL)
            {
                stateSet = new osg::StateSet();

                if (dfColor.a() < 1.0f)
                {
                    stateSet->setMode(GL_BLEND, osg::StateAttribute::ON);
                    stateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
                }
                else
                {
                    stateSet->setMode(GL_BLEND, osg::StateAttribute::OFF);
                    stateSet->setRenderingHint(osg::StateSet::OPAQUE_BIN);
                }

                osg::AlphaFunc *alphaFunc = new osg::AlphaFunc();
                alphaFunc->setFunction(osg::AlphaFunc::GEQUAL, 0.1f);

                stateSet->setAttributeAndModes(alphaFunc, osg::StateAttribute::ON);

                pGeode->setStateSet(stateSet);
            }

            osg::ref_ptr<osg::Material> mat = NULL;
            if ((mat = dynamic_cast<osg::Material *>(pGeode->getStateSet()->getAttribute(osg::StateAttribute::MATERIAL))) == NULL)
            {
                mat = new osg::Material();
                pGeode->getStateSet()->setAttribute(mat);
            }
            *mat = *(d_currentObject->mtl);
        }
        else
            pText->setColor(Vec4(1.0, 1.0, 1.0, 1.0));

        pText->setFontResolution(coCoviseConfig::getFloat("COVER.Plugin.Vrml97.FontResolution", 32.0f), coCoviseConfig::getFloat("COVER.Plugin.Vrml97.FontResolution", 32.0f));
        pText->setPosition(Vec3(0.0, -fontSize * i, 0.0));
    }

    if ((unsigned int)stringNumber < pGeode->getNumDrawables())
        pGeode->removeDrawables(stringNumber, pGeode->getNumDrawables() - stringNumber);

    if (d_currentObject->pNode.get() == NULL)
    {
        d_currentObject->pNode = (Node *)pGeode;
        setModesByName();
        addToScene(d_currentObject);
    }
    // don't simply update the StateSet here because this ruines all other text nodes which share this stateset, it would have to be copied first
    //d_currentObject->updateMaterial();
    d_currentObject->transparent = true;
    d_currentObject->updateBin();
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::insertText" << endl;
    return (Object)d_currentObject;
}

// Lights

Viewer::Object ViewerOsg::insertDirLight(float /*ambient*/,
                                         float intensity,
                                         float rgb[],
                                         float direction[])
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::insertDirLight\n";
    if (enableLights)
    {
        if (d_currentObject->pNode.get() == NULL)
        {
            LightSource *ls = new LightSource();
            d_currentObject->pNode = ls;
            d_currentObject->lightedNode = NULL; // this means our parent (see addObj)
            setModesByName();
            addToScene(d_currentObject);
        }

        LightSource *pLightSource = dynamic_cast<LightSource *>(d_currentObject->pNode.get());
        if (pLightSource)
        {
            Light *light = pLightSource->getLight();
            light->setPosition(Vec4(-direction[0], -direction[1], -direction[2], 0.0));
            light->setSpecular(Vec4(rgb[0] * intensity, rgb[1] * intensity, rgb[2] * intensity, 1.0));
            light->setDiffuse(Vec4(rgb[0] * intensity, rgb[1] * intensity, rgb[2] * intensity, 1.0));
            coVRLighting::instance()->switchLight(pLightSource, (intensity != 0.));
        }
        else
        {
            if (cover->debugLevel(2))
                cerr << "insertDirLight with wrong performer Node";
        }

        if (cover->debugLevel(5))
            cerr << "END 1 ViewerOsg::insertDirLight\n";
        return (Object)d_currentObject;
    }
    if (cover->debugLevel(5))
        cerr << "END 2 ViewerOsg::insertDirLight\n";

    return 1;
}

//
//  Only objects within radius should be lit by each PointLight.
//  Test each object drawn against each point light and enable
//  the lights accordingly? Get light and geometry into consistent
//  coordinates first...
//

Viewer::Object ViewerOsg::insertPointLight(float /*ambient*/,
                                           float attenuation[],
                                           float rgb[],
                                           float intensity,
                                           float location[],
                                           float /*radius*/)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::insertPointLight\n";
    if (enableLights)
    {
        if (d_currentObject->pNode.get() == NULL)
        {
            LightSource *ls = new LightSource();
            d_currentObject->pNode = ls;
            d_currentObject->lightedNode = cover->getObjectsRoot();
            setModesByName();
            addToScene(d_currentObject);
        }

        LightSource *pLightSource = dynamic_cast<LightSource *>(d_currentObject->pNode.get());
        if (pLightSource)
        {

            Light *light = pLightSource->getLight();
            light->setPosition(Vec4(location[0], location[1], location[2], 1.0));
            light->setSpecular(Vec4(rgb[0] * intensity, rgb[1] * intensity, rgb[2] * intensity, 1.0));
            light->setDiffuse(Vec4(rgb[0] * intensity, rgb[1] * intensity, rgb[2] * intensity, 1.0));
            light->setConstantAttenuation(attenuation[0]);
            light->setLinearAttenuation(attenuation[1]);
            light->setQuadraticAttenuation(attenuation[2]);
            light->setSpotExponent(1.0);
            light->setSpotCutoff(180);
            coVRLighting::instance()->switchLight(pLightSource, (intensity != 0.));
        }
        else
        {
            if (cover->debugLevel(2))
                cerr << "insertPointLight with wrong osg Node";
        }

        if (cover->debugLevel(2))
            cerr << "END 1 ViewerOsg::insertPointLight\n";
        return (Object)d_currentObject;
    }
    if (cover->debugLevel(5))
        cerr << "END 2 ViewerOsg::insertPointLight\n";
    return 1;
}

// same comments as for PointLight apply here...
Viewer::Object ViewerOsg::insertSpotLight(float /*ambient*/,
                                          float attenuation[],
                                          float beamWidth,
                                          float rgb[],
                                          float cutOffAngle,
                                          float direction[],
                                          float intensity,
                                          float location[],
                                          float /*radius*/)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::insertSpotLight\n";
    if (enableLights)
    {
        if (d_currentObject->pNode.get() == NULL)
        {
            LightSource *ls = new LightSource();
            d_currentObject->pNode = ls;
            d_currentObject->lightedNode = cover->getObjectsRoot();
            setModesByName();
            addToScene(d_currentObject);
        }

        LightSource *pLightSource = dynamic_cast<LightSource *>(d_currentObject->pNode.get());
        if (pLightSource)
        {
            Light *light = pLightSource->getLight();
            light->setPosition(Vec4(location[0], location[1], location[2], 1.0));
            light->setSpecular(Vec4(rgb[0] * intensity, rgb[1] * intensity, rgb[2] * intensity, 1.0));
            light->setDiffuse(Vec4(rgb[0] * intensity, rgb[1] * intensity, rgb[2] * intensity, 1.0));
            light->setConstantAttenuation(attenuation[0]);
            light->setLinearAttenuation(attenuation[1]);
            light->setQuadraticAttenuation(attenuation[2]);
            float exp = 128 * (1 - ((beamWidth) / (M_PI / 2.0)));
            light->setSpotExponent(exp);
            light->setSpotCutoff(cutOffAngle * 180.0 / M_PI);
            light->setDirection(Vec3(direction[0], direction[1], direction[2]));
            coVRLighting::instance()->switchLight(pLightSource, (intensity != 0.));
        }
        else
        {
            if (cover->debugLevel(2))
                cerr << "insertSpotLight with wrong performer Node";
        }

        if (cover->debugLevel(5))
            cerr << "END 1 ViewerOsg::insertSpotLight\n";
        return (Object)d_currentObject;
    }
    if (cover->debugLevel(5))
        cerr << "END 1 ViewerOsg::insertSpotLight\n";
    return 1;
}

// Lightweight copy

Viewer::Object ViewerOsg::insertReference(Object existingObject)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::insertReference\n";
    osgViewerObject *obj = (osgViewerObject *)existingObject;

    if (d_currentObject->hasChild(obj) || d_currentObject == obj)
    {
        //CERR << "insertReference: nothing to do" << endl;
    }
    else
    {
        bool doAddToScene = false;
        if (obj == d_currentObject)
        {
            if (cover->debugLevel(2))
                cerr << "Oops, recursion" << endl;
            return 0;
        }
        if (d_currentObject->parent && (obj == d_currentObject->parent))
        {
            if (cover->debugLevel(2))
                cerr << "Oops, recursion2" << endl;
            return 0;
        }
        if (cover->debugLevel(2))
        {
            cerr << "check obj" << endl;
            cerr << obj->numTextures << endl;
            cerr << "addChild" << endl;
        }
        d_currentObject->addChild(obj);

        if (cover->debugLevel(2))
            cerr << "after addChild" << endl;

        if (d_currentObject->pNode.get() == NULL)
        {
            if (cover->debugLevel(2))
                cerr << "ViewerOsg::insertReference Create Group (I assume, that empty viewerObjects are Groups at this stage)" << endl;
            if (cover->debugLevel(1))
                cerr << "_RG_";
            if (d_currentObject->whichChoice != -2)
            {
                if (cover->debugLevel(2))
                    cerr << "insertReference(): added Switch: choice=" << d_currentObject->whichChoice << endl;
                if (cover->debugLevel(1))
                    cerr << "S";
                d_currentObject->pNode = new Switch();
            }
            else
            {
                if (cover->debugLevel(2))
                    cerr << "insertReference(): added group" << endl;
                d_currentObject->pNode = new Group();
            }
            if (cover->debugLevel(2))
                cerr << "insertReference(): setModes add to Scene" << endl;
            setModesByName();
            doAddToScene = true; // do it later, so that sensors can be activated
            d_currentObject->addChildrensNodes();
            // add sensors that have not been added yet
            if (d_currentObject->sensor == NULL && d_currentObject->sensorObjectToAdd != NULL)
            {
                d_currentObject->sensor = new coSensiveSensor(d_currentObject->pNode.get(), d_currentObject, d_currentObject->sensorObjectToAdd, d_scene, VRMLRoot);
                sensors.push_back(d_currentObject->sensor);
                sensorList.append(d_currentObject->sensor);
            }
        }
        if (cover->debugLevel(2))
            cerr << "insertReference(): pGroup" << endl;
        Group *pGroup = dynamic_cast<Group *>(d_currentObject->pNode.get());
        if (pGroup)
        {
            if (cover->debugLevel(2))
                cerr << "ViewerOsg::insertReference to group node" << endl;
            if (cover->debugLevel(1))
                cerr << "RG";
            if (d_currentObject->pNode.get())
            {
                pGroup->addChild(obj->getNode());
                if (cover->debugLevel(2))
                    cerr << "add a Child to Parent\n";
                if (d_currentObject->whichChoice >= 0)
                {
                    if (cover->debugLevel(2))
                        cerr << "addSwitchChild" << d_currentObject->whichChoice << "\n";
                    d_currentObject->choiceMap[d_currentObject->whichChoice] = pGroup->getChildIndex(obj->getNode());
                    if (cover->debugLevel(2))
                        cerr << "choiceMap : " << d_currentObject->choiceMap[d_currentObject->whichChoice] << endl;
                    ((Switch *)(d_currentObject->pNode.get()))->setSingleChildOn(d_currentObject->choiceMap[d_currentObject->whichChoice]);
                }
            }
        }
        else if (d_currentObject->pNode.get() == NULL)
        {
            obj->haveToAdd++;
        }

        if (doAddToScene)
        {
            if (cover->debugLevel(2))
                cerr << "insertRef addToScene\n";
            addToScene(d_currentObject);
        }
    }
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::insertReference\n";
    return 0;
}

// Remove an object from the display list

void ViewerOsg::removeObject(Object /*key*/)
{
    /*osgViewerObject *obj=(osgViewerObject *)key;
     if((d_currentObject->hasChild(obj))||(d_currentObject==obj))
     {
     if(obj->nodeType==NODE_IFS) // only remove indexed face sets
     delete obj;
     }*/
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::removeObject" << endl;
}

void ViewerOsg::enableLighting(bool /*lightsOn*/)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::enableLighting" << endl;
}

// Set attributes

void ViewerOsg::setColor(float r, float g, float b, float a)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::setColor\n";
    if ((d_currentObject->dC[0] != r) || (d_currentObject->dC[1] != g) || (d_currentObject->dC[2] != b) || (d_currentObject->trans != (1.0 - a)))
    {
        d_currentObject->dC[0] = r;
        d_currentObject->dC[1] = g;
        d_currentObject->dC[2] = b;
        d_currentObject->trans = (1.0 - a);
        if (d_currentObject->mtl == NULL)
        {
            d_currentObject->mtl = new Material;
            d_currentObject->mtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
            d_currentObject->mtl->setAmbient(Material::FRONT_AND_BACK, Vec4(0.1 * r, 0.1 * g, 0.1 * b, a));
            d_currentObject->mtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(r, g, b, a));
            d_currentObject->mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.5 + 0.5 * r, 0.5 + 0.5 * g, 0.5 + 0.5 * b, a));
            d_currentObject->mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0, 0.0, 0.0, a));
            d_currentObject->mtl->setShininess(Material::FRONT_AND_BACK, 16.0);
            d_currentObject->mtl->setAlpha(Material::FRONT_AND_BACK, a);
            d_currentObject->updateMaterial();
        }
        else
        {
            d_currentObject->mtl->setAmbient(Material::FRONT_AND_BACK, Vec4(0.1 * r, 0.1 * g, 0.1 * b, a));
            d_currentObject->mtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(r, g, b, a));
            d_currentObject->mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.5 + 0.5 * r, 0.5 + 0.5 * g, 0.5 + 0.5 * b, a));
            d_currentObject->mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0, 0.0, 0.0, a));
            d_currentObject->mtl->setShininess(Material::FRONT_AND_BACK, 16.0);
            d_currentObject->mtl->setAlpha(Material::FRONT_AND_BACK, a);
            d_currentObject->updateMaterial();
        }
        bool oldTransparency = d_currentObject->transparent;
        /* bool textured = false;
         for(i=0;i<d_currentObject->numTextures;i++)
         {
      // if texture available, texture defines transparency, otherwise we could set this to opaque
      if(d_currentObject->texData[i].texture!=NULL)
      {
      textured = true;
      }
      }
      if(!textured)
      {*/
        if (a == 1.0)
        {
            d_currentObject->transparent = false;
        }
        /*    else
            {
            d_currentObject->transparent = true;
            }
            }*/
        if (oldTransparency != d_currentObject->transparent)
        {
            d_currentObject->updateBin();
        }

        if (cover->debugLevel(1))
            cerr << "C" << endl;
    }
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::setColor\n";
}

void ViewerOsg::setFog(float *color,
                       float visibilityRange,
                       const char *fogType)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::setFog\n";
#if 0
    (void)color;
    (void)visibilityRange;
    (void)fogType;
// nebel tut nicht weil er in Z-Richtung geht.
   // XXX
   fprintf(stderr, "ViewerOsg::setFog(color=(%f %f %f), range=%f, type=%s)\n",
      color[0], color[1], color[2], visibilityRange, fogType);

   Vec3 scale = currentTransform.getScale();
   fprintf(stderr, "current scale: (%f %f %f)\n", scale[0], scale[1], scale[2]);
   
#endif
#if 1
   
   StateSet *state = VRMLRoot->getOrCreateStateSet();
   Fog *fog = new Fog();
   if(visibilityRange == 0)
   {
       state->setAttributeAndModes(fog, StateAttribute::OFF);
       return;
   }
   fog->setColor(Vec4(color[0], color[1], color[2], 1.0));
   fog->setStart(0.1);
   fog->setEnd(0.1+visibilityRange*1000.0);
   fog->setFogCoordinateSource(Fog::FRAGMENT_DEPTH);
   //fog->setFogCoordinateSource(Fog::FOG_COORDINATE);
   fog->setUseRadialFog(true);
   Fog::Mode fogMode = Fog::LINEAR;
   if(!strcmp(fogType, "LINEAR"))
   {
      fogMode = Fog::LINEAR;
   }
   else if(!strcmp(fogType, "EXPONENTIAL"))
   {
      fogMode = Fog::EXP;
   }
   else
   {
      CERR << "unknown fog mode " << fogType << endl;
   }
   fog->setMode(fogMode);
   state->setAttributeAndModes(fog, StateAttribute::ON);

   state->setAttributeAndModes(fog, StateAttribute::ON);
#endif
}

// This hack is necessary because setting the color mode needs to know
// about the appearance (presence & components of texture) and the geometry
// (presence of colors). Putting this stuff in either insertTexture or
// insert<geometry> causes problems when the texture or geometry node is
// USE'd with a different context.

void ViewerOsg::setMaterialMode(int /*textureComponents*/,
                                bool /*colors*/)
{
    //cerr << "ViewerOsg::setMaterialMode" << endl;
}

void ViewerOsg::setNameModes(const char *modes, const char *relURL)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::setNameModes\n";
    if (d_currentObject)
    {
        if (modes)
        {
            d_currentObject->modeNames = new char[strlen(modes) + 1];
            strcpy(d_currentObject->modeNames, modes);
        }
        if (relURL)
        {
            d_currentObject->MyDoc = relURL;
        }
    }
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::setNameModes\n";
}

#ifdef HAVE_OSGNV
class TimeCallback : public osgNV::ParameterValueCallback
{
public:
    TimeCallback() {}
    TimeCallback(const TimeCallback &copy, const osg::CopyOp &copyop = osg::CopyOp::SHALLOW_COPY)
        : osgNV::ParameterValueCallback(copy, copyop)
    {
    }

    META_Object(test, TimeCallback);

    void operator()(osgNV::ParameterValue *param, osg::State &state) const
    {
        (void)state;
        osgNV::VectorParameterValue *cgp = dynamic_cast<osgNV::VectorParameterValue *>(param);
        if (cgp)
        {
            cgp->set(cover->frameTime());
        }
    }
};
#endif
struct coMirrorCullCallback : public osg::Drawable::CullCallback
{
    int myMirror;
    ViewerOsg *theViewer;
    coMirrorCullCallback(int m, ViewerOsg *v)
    {
        myMirror = m;
        theViewer = v;
    }
    virtual bool cull(osg::NodeVisitor *, osg::Drawable *, osg::RenderInfo *) const
    {
        /*fprintf(stderr,"isVisible RenderInfo %d\n",myMirror);*/ theViewer->mirrors[myMirror].isVisible = true;
		if (theViewer->mirrors[myMirror].camera->getNumParents() == 0)
			return true; // don't render if it just became visible, otherwise the camera was not active this frame and rendering will crash
        return false;
    }
    virtual bool cull(osg::NodeVisitor *, osg::Drawable *, osg::State *) const
    {
        theViewer->mirrors[myMirror].isVisible = true;
        return false;
    }
};

void ViewerOsg::setModesByName(const char *objectName)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::setModesByName\n";
    if (d_currentObject->node && d_currentObject->pNode.get())
    {

        const char *name;
        if (objectName)
            name = objectName;
        else
        {
            name = d_currentObject->node->name();
            d_currentObject->pNode->setName(name);
        }

        bool animated = false;
        if (strncmp(name, "Animated", 8) == 0)
        {
            name += 8;
            animated = true;
        }

        Geode *pGeode = dynamic_cast<Geode *>(d_currentObject->pNode.get());
        if (pGeode)
        {
            for (unsigned int i = 0; i < pGeode->getNumDrawables(); i++)
            {
                osg::Drawable *drawable = pGeode->getDrawable(i);
                if (drawable)
                {
                    cover->setRenderStrategy(drawable, false);

                    StateSet *stateset = drawable->getOrCreateStateSet();
                    stateset->setNestRenderBins(false);
                    if (strncmp(name, "coDontMirror", 12) == 0)
                    {
						pGeode->setNodeMask(pGeode->getNodeMask() & (~Isect::NoMirror));
					}
					else if (strncmp(name, "coPolygonOffset", 15) == 0)
					{
						osg::StateSet* stateset;
						stateset = drawable->getOrCreateStateSet();
						float units=1.0;
                        float factor = 1.0;
                        sscanf(name, "coPolygonOffset_%f_%f", &factor,&units);
						osg::PolygonOffset* po = new osg::PolygonOffset();
						po->setFactor(0);
                        po->setUnits(units);
						stateset->setAttributeAndModes(po, osg::StateAttribute::ON);
                        fprintf(stderr, "po %f %f\n", factor, units);
					}
                    else if (strncmp(name, "coMirror", 8) == 0)
                    {

                        pGeode->setNodeMask(pGeode->getNodeMask() & (~Isect::NoMirror));
                        osg::Drawable *d = pGeode->getDrawable(0);
                        if (d)
                        {
                            d->setCullCallback(new coMirrorCullCallback(numCameras, this));
                        }
                        mirrors[numCameras].shader = NULL;
                        mirrors[numCameras].CameraID = -1;
                        char *shaderName = new char[strlen(name + 8) + 1];
                        strcpy(shaderName, name + 8);
                        char *c = shaderName;
                        while (*c != '\0')
                        {
                            if (*c == '_' || *c == '-')
                            {
                                *c = '\0';
                                break;
                            }
                            c++;
                        }
                        if (strncmp(shaderName, "Camera", 6) == 0)
                        {
                            sscanf(shaderName, "Camera%d", &mirrors[numCameras].CameraID);
                            shaderName = c;
                            while (*c != '\0')
                            {
                                if (*c == '_' || *c == '-')
                                {
                                    *c = '\0';
                                    break;
                                }
                                c++;
                            }
                        }
                        coVRShader *shader;
                        mirrors[numCameras].shader = shader = coVRShaderList::instance()->get(shaderName);
                        if (shader)
                        {
                            if (shader->isTransparent())
                                d_currentObject->transparent = true;
                            mirrors[numCameras].instance = shader->apply(pGeode, drawable);
                        }
                        else
                        {
                            if (mirrors[numCameras].CameraID < 0)
                                cerr << "Mirror without a shader (-->flat mirror)" << shaderName << endl;
                            else
                                cerr << "rear view Camera " << mirrors[numCameras].CameraID << endl;
                        }

                        if (textureNumber == 0)
                        {
                            if (shader)
                                d_currentObject->updateTexData(textureNumber + 2);
                            else
                                d_currentObject->updateTexData(textureNumber + 1);
                        }
                        int tex_width = coCoviseConfig::getInt("COVER.Plugin.Vrml97.MirrorWidth", 512);
                        int tex_height = coCoviseConfig::getInt("COVER.Plugin.Vrml97.MirrorWidth", 256);

                        osg::Camera::RenderTargetImplementation renderImplementation = osg::Camera::FRAME_BUFFER_OBJECT;

                        std::string buf = coCoviseConfig::getEntry("COVER.Plugin.Vrml97.RTTImplementation");
                        if (!buf.empty())
                        {
                            if (cover->debugLevel(2))
                                cerr << "renderImplementation: " << buf << endl;
                            if (strcasecmp(buf.c_str(), "fbo") == 0)
                                renderImplementation = osg::Camera::FRAME_BUFFER_OBJECT;
                            if (strcasecmp(buf.c_str(), "pbuffer") == 0)
                                renderImplementation = osg::Camera::PIXEL_BUFFER;
                            if (strcasecmp(buf.c_str(), "pbuffer-rtt") == 0)
                                renderImplementation = osg::Camera::PIXEL_BUFFER_RTT;
                            if (strcasecmp(buf.c_str(), "fb") == 0)
                                renderImplementation = osg::Camera::FRAME_BUFFER;
#if OSG_VERSION_GREATER_OR_EQUAL(3, 4, 0)
                            if (strcasecmp(buf.c_str(), "window") == 0)
                                renderImplementation = osg::Camera::SEPARATE_WINDOW;
#else
                            if (strcasecmp(buf.c_str(), "window") == 0)
                                renderImplementation = osg::Camera::SEPERATE_WINDOW;
#endif
                        }

                        osg::Texture *texture = NULL;
                        osg::Texture *texture2 = NULL;
                        {
                            osg::Texture2D *texture2D = new osg::Texture2D;
                            texture2D->setTextureSize(tex_width, tex_height);
                            texture2D->setInternalFormat(GL_RGBA);
                            texture2D->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR);
                            texture2D->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);

                            texture = texture2D;
                            if (shader)
                            {
                                osg::Texture2D *texture2D = new osg::Texture2D;
                                texture2D->setTextureSize(tex_width, tex_height);
                                texture2D->setInternalFormat(GL_DEPTH_COMPONENT);
                                texture2D->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::NEAREST);
                                texture2D->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::NEAREST);
                                texture2D->setSourceType(GL_UNSIGNED_SHORT);

                                texture2 = texture2D;
                            }
                        }

                        if ((d_currentObject->texData.size() > 0) && d_currentObject->texData[textureNumber].texture != NULL)
                            d_currentObject->texData[textureNumber].texture.get()->ref(); // make sure the old texture is not deleted because it might be reused better would probably be to keep a reference to it
                        if ((d_currentObject->texData.size() > 1) && d_currentObject->texData[textureNumber + 1].texture != NULL)
                            d_currentObject->texData[textureNumber + 1].texture.get()->ref(); // make sure the old texture is not deleted because it might be reused better would probably be to keep a reference to it
                        d_currentObject->texData[textureNumber].texture = texture;
                        if (shader)
                        {
                            if (d_currentObject->texData.size() < 2)
                            {
                                fprintf(stderr, "oops not enought texture objects\n");
                            }
                            d_currentObject->texData[textureNumber + 1].texture = texture2;
                        }

                        d_currentObject->setTexEnv(false, textureNumber, 1, 4);
                        d_currentObject->setTexGen(false, textureNumber, 1);
                        
                                fprintf(stderr, "textureNumber %d \n",textureNumber);
                        d_currentObject->texData[textureNumber].mirror = 2;
                        d_currentObject->updateTexture();

                        // then create the camera node to do the render to texture
                        {
                            osg::Camera *camera;
                            mirrors[numCameras].camera = camera = new osg::Camera;
							camera->setName("mirrorCamera");
                            mirrors[numCameras].geometry = pGeode;
                            mirrors[numCameras].isVisible = true;

                            // set up the background color and clear mask.
                            camera->setClearColor(osg::Vec4(0.0f, 0.0f, 0.0f, 1.0f));
                            camera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                            camera->setCullMask(Isect::NoMirror);


                            // we assume a planar mapping from min to max in x/z  plane
                            // find out x/z min and max.
                            const osg::BoundingBox &bb = pGeode->getBoundingBox();
                            mirrors[numCameras].coords[0].set(bb.xMin(), bb.yMax(), bb.zMin());
                            mirrors[numCameras].coords[1].set(bb.xMax(), bb.yMax(), bb.zMin());
                            mirrors[numCameras].coords[2].set(bb.xMax(), bb.yMax(), bb.zMax());
                            mirrors[numCameras].coords[3].set(bb.xMin(), bb.yMax(), bb.zMax());
                            //von vorne
                            // 0     1
                            // 3     2

                            // set up projection.
                            camera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
                            camera->setComputeNearFarMode(osg::Camera::DO_NOT_COMPUTE_NEAR_FAR);

                            // set viewport
                            camera->setViewport(0, 0, tex_width, tex_height);

                            // set the camera to render before the main camera.
                            camera->setRenderOrder(osg::Camera::PRE_RENDER);

                            // tell the camera to use OpenGL frame buffer object where supported.
                            camera->setRenderTargetImplementation(renderImplementation);

                            camera->attach(osg::Camera::COLOR_BUFFER, texture);
							if(texture2)
                                camera->attach(osg::Camera::DEPTH_BUFFER, texture2);

                            mirrors[numCameras].statesetGroup = new osg::Group;
							mirrors[numCameras].statesetGroup->setName("cameraStatesetGroup");
                            osg::StateSet *dstate = mirrors[numCameras].statesetGroup->getOrCreateStateSet();
                            dstate->setNestRenderBins(false);
                            dstate->setMode(GL_CULL_FACE, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);

                            mirrors[numCameras].statesetGroup->addChild(cover->getObjectsXform());

                            // add subgraph to render
                            camera->addChild(mirrors[numCameras].statesetGroup.get());

                            cover->getScene()->addChild(camera);
                            numCameras++;
                            if (numCameras > MAX_MIRRORS)
                            {
                                cerr << "numMirrors = " << numCameras << " MAX_MIRRORS is " << MAX_MIRRORS << endl;
                                exit(-1);
                            }
                        }
                    }
                    else if (strncmp(name, "coDepthOnly", 11) == 0)
                    {
                        // after Video but before all normal geometry

                        //stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);
                        stateset->setNestRenderBins(false);
                        stateset->setRenderBinDetails(-1, "RenderBin");
                        stateset->setAttributeAndModes(cover->getNoFrameBuffer().get(), StateAttribute::ON);
                    }
                    else if (strncmp(name, "coNoDepthTest", 13) == 0)
                    {   
                        // do no depth test and render as late as possible
                        stateset->setMode(GL_DEPTH_TEST,osg::StateAttribute::OFF);
                        stateset->setRenderBinDetails( 11, "RenderBin");
                    }
                    else if (strncmp(name, "coNoDepth", 9) == 0)
                    {
                        stateset->setRenderBinDetails(-2, "RenderBin");
                        stateset->setAttributeAndModes(NoDepthBuffer, StateAttribute::ON);
                        if (cover->debugLevel(2))
                            fprintf(stderr, "coNoDepth\n");
                    }
                    else if (strncmp(name, "coShader", 8) == 0)
                    {
                      applyShader(name+8,pGeode,drawable);   
                    }

                    else if (strncmp(name, "coBumpCube", 10) == 0)
                    {
                        if (d_currentObject->pNode.get())
                        {
                            d_currentObject->pNode.get()->setStateSet(BumpCubeState.get());
                            if (cover->debugLevel(2))
                                cerr << "ViewerOsg::insertBumpCubeMapping" << endl;
                            osg::Geometry *geo = dynamic_cast<osg::Geometry *>(drawable);
                            if (geo)
                            {
                                osg::ref_ptr<osgUtil::TangentSpaceGenerator> tsg = new osgUtil::TangentSpaceGenerator;
                                tsg->generate(geo, 1);
                                //vTangent
                                if (!geo->getVertexAttribArray(6))
                                    geo->setVertexAttribArray(6, tsg->getTangentArray(), osg::Array::BIND_PER_VERTEX);
                                /*if (!geo->getVertexAttribArray(7))
                           geo->setVertexAttribData(7, osg::Geometry::ArrayData(tsg->getBinormalArray(), osg::Geometry::BIND_PER_VERTEX, GL_FALSE));
                        if (!geo->getVertexAttribArray(15))
                           geo->setVertexAttribData(15, osg::Geometry::ArrayData(tsg->getNormalArray(), osg::Geometry::BIND_PER_VERTEX, GL_FALSE));
                                   */
                            }
                        }
                    }
                    else if (strncmp(name, "coBumpEnv", 9) == 0)
                    {
                        if (d_currentObject->pNode.get())
                        {
                            d_currentObject->pNode.get()->setStateSet(BumpEnvState.get());
                            if (cover->debugLevel(2))
                                cerr << "ViewerOsg::insertBumpEnvMapping" << endl;
                            osg::Geometry *geo = dynamic_cast<osg::Geometry *>(drawable);
                            if (geo)
                            {
                                osg::ref_ptr<osgUtil::TangentSpaceGenerator> tsg = new osgUtil::TangentSpaceGenerator;
                                tsg->generate(geo, 1);
                                //vTangent
                                if (!geo->getVertexAttribArray(6))
                                    geo->setVertexAttribArray(6, tsg->getTangentArray(), osg::Array::BIND_PER_VERTEX);
                            }
                        }
                    }
                    else if (strncmp(name, "coBump", 6) == 0)
                    {
                        int _normal_unit = 1; // normal texture is unit 1
                        osg::Geometry *geo = dynamic_cast<osg::Geometry *>(drawable);
                        if (geo)
                        {
                            StateSet *stateset = d_currentObject->pNode->getOrCreateStateSet();
                            stateset->setNestRenderBins(false);
                            stateset->setAttributeAndModes(bumpProgram.get(), osg::StateAttribute::ON);

                            osg::Uniform *lightPosU = new osg::Uniform("LightPosition", osg::Vec3(0, 0, 1));
                            osg::Uniform *normalMapU = new osg::Uniform("normalMap", 1);
                            osg::Uniform *baseTextureU = new osg::Uniform("baseTexture", 0);

                            stateset->addUniform(lightPosU);
                            stateset->addUniform(normalMapU);
                            stateset->addUniform(baseTextureU);

                            osg::ref_ptr<osgUtil::TangentSpaceGenerator> tsg = new osgUtil::TangentSpaceGenerator;
                            tsg->generate(geo, _normal_unit);
                            //vTangent
                            if (!geo->getVertexAttribArray(6))
                                geo->setVertexAttribArray(6, tsg->getTangentArray(), osg::Array::BIND_PER_VERTEX);
                            /*if (!geo->getVertexAttribArray(7))
                        geo->setVertexAttribData(7, osg::Geometry::ArrayData(tsg->getBinormalArray(), osg::Geometry::BIND_PER_VERTEX, GL_FALSE));
                     if (!geo->getVertexAttribArray(15))
                        geo->setVertexAttribData(15, osg::Geometry::ArrayData(tsg->getNormalArray(), osg::Geometry::BIND_PER_VERTEX, GL_FALSE));
                                */
                        }
                    }
                    else if (strncmp(name, "coGlass", 7) == 0)
                    {
                        osg::Geometry *geo = dynamic_cast<osg::Geometry *>(drawable);
                        if (geo)
                        {
                            // replace or set texture number 1 to framebuffer
                            d_currentObject->updateTexData(2);
                            d_currentObject->texData[1].texture = framebufferTexture.get();
                            if (d_currentObject->numTextures < 2)
                            {
                                numTextures = 2;
                            }
                            d_currentObject->updateTexture();

                            geo->setDrawCallback(new CopyTextureCallback);
                            StateSet *stateset = d_currentObject->pNode->getOrCreateStateSet();
                            stateset->setNestRenderBins(false);
                            stateset->setAttributeAndModes(glassProgram.get(), osg::StateAttribute::ON);
                            stateset->addUniform(new osg::Uniform("Depth", 0.5f));
                            stateset->addUniform(new osg::Uniform("MixRatio", 0.9f));
                            stateset->addUniform(new osg::Uniform("FrameWidth", 1024.0f));
                            stateset->addUniform(new osg::Uniform("FrameHeight", 768.0f));
                            stateset->addUniform(new osg::Uniform("RefractionMap", 1));
                            stateset->addUniform(new osg::Uniform("EnvMap", 0));
                        }
                    }
                    else if (strncmp(name, "coTest", 6) == 0)
                    {
                        osg::Geometry *geo = dynamic_cast<osg::Geometry *>(drawable);
                        if (geo)
                        {
                            osg::Program *testProgram = new osg::Program;
                            testProgram->addShader(osg::Shader::readShaderFile(
                                osg::Shader::VERTEX, coVRFileManager::instance()->getName("test.vert")));
                            testProgram->addShader(osg::Shader::readShaderFile(
                                osg::Shader::FRAGMENT, coVRFileManager::instance()->getName("test.frag")));
                            testProgram->addBindAttribLocation("vTangent", 6);
                            StateSet *testState = d_currentObject->pNode->getOrCreateStateSet();
                            testState->setAttributeAndModes(testProgram, osg::StateAttribute::ON);

                            osg::Uniform *lightPosU = new osg::Uniform("LightPosition", osg::Vec3(0.0, -10000.0, 10000.0));
                            osg::Uniform *baseTextureU = new osg::Uniform("baseTexture", 0);
                            osg::Uniform *normalMapU = new osg::Uniform("normalMap", 1);
                            osg::Uniform *cubeTextureU = new osg::Uniform("cubeTexture", 2);

                            testState->addUniform(lightPosU);
                            testState->addUniform(baseTextureU);
                            testState->addUniform(normalMapU);
                            testState->addUniform(cubeTextureU);
                            osg::ref_ptr<osgUtil::TangentSpaceGenerator> tsg = new osgUtil::TangentSpaceGenerator;
                            tsg->generate(geo, 1);
                            //vTangent
                            if (!geo->getVertexAttribArray(6))
                                geo->setVertexAttribArray(6, tsg->getTangentArray(), osg::Array::BIND_PER_VERTEX);

                            /*
                                       d_currentObject->updateTexData(2); // replace or set texture number 1 to framebuffer
                                       d_currentObject->texData[1].texture=framebufferTexture.get();
                                       if(d_currentObject->numTextures<2)
                                       {
                                         numTextures = 2;
                                       }
                                       d_currentObject->updateTexture();

                                       geo->setDrawCallback(new CopyTextureCallback);
                                       StateSet *stateset = d_currentObject->pNode->getOrCreateStateSet();
                                       stateset->setAttributeAndModes( testProgram.get(), osg::StateAttribute::ON );
                                       stateset->addUniform(new osg::Uniform( "Depth", 10.0f));
                                       stateset->addUniform(new osg::Uniform( "MixRatio", 0.1f));
                                       stateset->addUniform( new osg::Uniform( "FrameWidth", 1024.0f ) );
                                       stateset->addUniform( new osg::Uniform( "FrameHeight", 768.0f ) );
                                       stateset->addUniform( new osg::Uniform( "RefractionMap", 1 ) );
                                       stateset->addUniform( new osg::Uniform( "EnvMap", 0 ) );*/
                        }
                    }
                    else if (strncmp(name, "coLeftOnly", 10) == 0)
                    {
                        d_currentObject->pNode.get()->setNodeMask(Isect::Left);
                    }
                    else if (strncmp(name, "coRightOnly", 11) == 0)
                    {
                        d_currentObject->pNode.get()->setNodeMask(Isect::Right);
                    }
                    else if (strncmp(name, "combineTextures", 15) == 0)
                    {
                        d_currentObject->pNode.get()->setStateSet(combineTexturesState.get());
                    }
                    else if (strncmp(name, "combineEnvTextures", 15) == 0)
                    {
                        d_currentObject->pNode.get()->setStateSet(combineEnvTexturesState.get());
                    }
                }
            }
        }
        else
        {

            if (strncmp(name, "coDepthOnly", 11) == 0)
            {
                StateSet *stateset = d_currentObject->pNode->getOrCreateStateSet();
                stateset->setNestRenderBins(false);
                // after Video but before all normal geometry
                stateset->setRenderBinDetails(-1, "RenderBin");
                stateset->setAttributeAndModes(cover->getNoFrameBuffer().get(), StateAttribute::ON);
            }
            else if (strncmp(name, "coNoDepth", 9) == 0)
            {
                StateSet *stateset = d_currentObject->pNode->getOrCreateStateSet();
                stateset->setNestRenderBins(false);
                stateset->setAttributeAndModes(NoDepthBuffer, StateAttribute::ON);
                if (cover->debugLevel(2))
                    fprintf(stderr, "coNoDepth\n");
            }
            else if (strncmp(name, "coGroupShader", 13) == 0)
            {
                applyShader(name+13,NULL,NULL);   
            }
            else if (strncmp(name, "coShader", 8) == 0)
            {
                applyShader(name+8,NULL,NULL);   
            }
            else if (strncmp(name, "coCgShader", 10) == 0)
            {
#ifdef HAVE_OSGNV

                StateSet *stateset = d_currentObject->pNode->getOrCreateStateSet();

                // create a Cg Context object (this will become the
                // default context for all new Cg Programs).
                // NOTE: if you don't create a context explicitly,
                // the library will create one by itself; in this
                // case you can access the default context by calling
                // the static method Context::getDefaultContext().
                osgNVCg::Context *context = osgNVCg::Context::getDefaultContext();

                // create a CG Program object. If you don't specify
                // which context the program belongs to, the default
                // context will be used. Note that you must pass the
                // program profile to the constructor.
                osgNVCg::Program *program = new osgNVCg::Program(context, osgNVCg::Program::VP20);

                // tell osgNVCg to load a program from file.
                // If you call the readCodeFromFile() method instead,
                // the source file will be read immediately and its
                // content will be passed to Program::setCode().
                // By calling setFileName() you are creating a link
                // between the program and the source file, so if
                // you save the scene graph you will save only the
                // file name, not the code itself.
                program->setFileName("osgnvcg2.cg");

                // add some parameters using the "shortcut" methods provided
                // by osgNVCg::Program
                program->addVectorParameter("LightVec")->set(0, 0, 1, 0);
                program->addVectorParameter("time")->setCallback(new TimeCallback);
                program->addStateMatrixParameter("ModelViewProj")->set(osgNV::StateMatrixParameterValue::MODELVIEW_PROJECTION);
                program->addStateMatrixParameter("ModelViewIT")->set(osgNV::StateMatrixParameterValue::MODELVIEW, osgNV::StateMatrixParameterValue::INVERSE_TRANSPOSE);

                // apply the CG program to our scene graph.
                stateset->setAttributeAndModes(program);
#else
                if (cover->debugLevel(2))
                    cerr << "no osgNV" << endl;
#endif
            }
            else if (strncmp(name, "coNoColli", 9) == 0)
                d_currentObject->pNode->setNodeMask(d_currentObject->pNode->getNodeMask() & ~Isect::Collision);
            else if (strncmp(name, "coNoIsect", 9) == 0)
                d_currentObject->pNode->setNodeMask(d_currentObject->pNode->getNodeMask() & ~Isect::Intersection);
            else if (strncmp(name, "coNoWalk", 8) == 0)
                d_currentObject->pNode->setNodeMask(d_currentObject->pNode->getNodeMask() & ~Isect::Walk);
            else if (strncmp(name, "coNoIntersection", 16) == 0)
                d_currentObject->pNode->setNodeMask(d_currentObject->pNode->getNodeMask() & ~(Isect::Walk | Isect::Intersection | Isect::Collision | Isect::Touch | Isect::Pick));
            else if (strncmp(name, "coLeftOnly", 10) == 0)
            {
                d_currentObject->pNode.get()->setNodeMask(Isect::Left);
            }
            else if (strncmp(name, "coRightOnly", 11) == 0)
            {
                d_currentObject->pNode.get()->setNodeMask(Isect::Right);
            }
            else if (strncmp(name, "coReceiveShadow", 15) == 0)
            {
                d_currentObject->pNode.get()->setNodeMask(d_currentObject->pNode->getNodeMask() | Isect::ReceiveShadow);
            }
            else if (strncmp(name, "coCastShadow", 12) == 0)
            {
                d_currentObject->pNode.get()->setNodeMask(d_currentObject->pNode->getNodeMask() | Isect::CastShadow);
            }
        }
    }
    if (cover->debugLevel(5))
        cerr << "EN DViewerOsg::setModesByName\n";
}

void ViewerOsg::applyShader(const char *shaderNameAndValues,osg::Geode *pGeode, osg::Drawable *drawable)
{
    char *shaderName = new char[strlen(shaderNameAndValues) + 1];
    strcpy(shaderName, shaderNameAndValues);
    char *c = shaderName;
    // % between parameter name and value
    while (*c != '\0')
    {
        if (*c == '%')
            *c = '=';
        else if (*c == '$')
            *c = '-';
        else if (*c == '&')
            *c = '.';
        c++;
    }

    // terminate shaderName and advance past it
    c = shaderName;
    while (*c != '\0')
    {
        if (*c == '_' || *c == '-')
        {
            *c = '\0';
            ++c;
            break;
        }
        ++c;
    }

    coVRShader *shader = coVRShaderList::instance()->get(shaderName);
    if (shader == NULL)
    { // try to find a local shader definition
        if (d_currentObject->MyDoc)
        {
            char *dirName = new char[strlen(d_currentObject->MyDoc) + 1];
            strcpy(dirName, d_currentObject->MyDoc);
            char *pos = strrchr(dirName, '/');
#ifdef _WIN32
            char *pos2 = strrchr(dirName, '\\');
            if (pos2 > pos)
                pos = pos2;
#endif
            if (pos != NULL)
            {
                *pos = '\0';
                std::string dir(dirName);
                shader = coVRShaderList::instance()->add(shaderName, dir);
            }
            else
            {
                std::string dir(".");
                shader = coVRShaderList::instance()->add(shaderName, dir);
            }
            delete[] dirName;
        }
    }
    if (shader == NULL)
    {
        if (cover->debugLevel(1))
            cerr << "ERROR: no shader found with name:" << shaderName << endl;
        return;
    }

    std::list<coVRUniform *> uniformList = shader->getUniforms();
    auto applyParam = [uniformList](const char *n, const char *value)
    {
        std::string name(n);
        //std::cerr << "setting param: " << name << "=" << value << std::endl;
        auto it = std::find_if(uniformList.begin(), uniformList.end(),
                               [name](const coVRUniform *uni) { return uni->getName() == name; });
        if (it != uniformList.end())
            (*it)->setValue(value);
    };

    //std::cerr << "parsing params for " << shaderName << ": " << c << std::endl;
    // now parse for parameters, _ between params, = between name and value
    const char *paramName = c;
    const char *value = nullptr;
    while (*c != '\0')
    {
        // next parameter
        if (*c == '_' /* || *c == '-'*/)
        {
            // terminate value
            *c = '\0';

            if (value)
            {
                applyParam(paramName, value);
            }

            paramName = c + 1;
            value = nullptr;
        }
        // value for current parameter
        else if (*c == '=')
        {
            // terminate current parameter name
            *c = '\0';
            value = c + 1;
        }

        c++;
    }
    // apply last parameter
    if (value)
    {
        applyParam(paramName, value);
    }

    if (shader->isTransparent())
        d_currentObject->transparent = true;
    if (pGeode != NULL && drawable != NULL)
    {
        shader->apply(pGeode, drawable);
    }
    else
    {
        shader->apply(d_currentObject->pNode->getOrCreateStateSet());
    }
}

void ViewerOsg::setCollision(bool collide)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::setCollision\n";
    if (!d_currentObject->pNode.get())
    {
        d_currentObject->pNode = new Group();

        setModesByName();
        addToScene(d_currentObject);
        d_currentObject->addChildrensNodes();
    }
    if (collide == false)
    {
        d_currentObject->pNode->setNodeMask(d_currentObject->pNode->getNodeMask() & ~Isect::Collision);
    }
    else
    {
        d_currentObject->pNode->setNodeMask(d_currentObject->pNode->getNodeMask() | Isect::Collision);
    }
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::setCollision\n";
}

void ViewerOsg::setSensitive(void *object)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::setSensitive\n";
    if (!object)
        return;
    if (d_currentObject->sensor == NULL)
    {
        if (d_currentObject->pNode.get() != NULL)
        {
            d_currentObject->sensor = new coSensiveSensor(d_currentObject->pNode.get(), d_currentObject, object, d_scene, VRMLRoot);
            sensors.push_back(d_currentObject->sensor);
            sensorList.append(d_currentObject->sensor);
        }
        else
        {
            d_currentObject->sensorObjectToAdd = object;
        }
        //cerr << "ViewerOsg::setSensitive " << d_currentObject->node->name()<< endl;
        //cerr << "SE" ;
    }
    //cerr << "ViewerOsg::setSensitive" << object << endl;
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::setSensitive\n";
}

//
// Pixels are lower left to upper right by row.
//

Viewer::TextureObject
ViewerOsg::insertTexture(int w, int h, int nc,
                         bool repeat_s,
                         bool repeat_t,
                         unsigned char *pixels,
                         const char* filename,
                         bool /*retainHint*/,
                         bool environment, int blendMode, int anisotropy, int filter)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::insertTexture" << endl;

#if 0
   const char *name="(null)";
   if(d_currentObject->pNode.get())
   {
      name = d_currentObject->pNode->getName().c_str();
   }
   fprintf(stderr, "insertTexture(texNum=%d, w=%d, h=%d, nc=%d, filter=%d, name=%s)\n",
      textureNumber, w, h, nc, filter, name);
#endif
    /* if(textureNumber==0)
    {
       Geode *pGeode = dynamic_cast<Geode *>(d_currentObject->pNode.get());
       if(pGeode)
       {
          const char *name = d_currentObject->pNode->getName().c_str();
          Drawable *geoset = pGeode->getDrawable(0);
          if(name && geoset)
          {
             if(strncmp(name,"coMirror",8)==0)
             {
                  return (TextureObject) 1; //texture; // we generate a texture, dont use the normal diffuse texture
             }
          }
       }
    }*/
    d_currentObject->updateTexData(textureNumber + 1);
    //fprintf(stderr, "Texture %x\n",pixels);
    // set transparency based on first texture
    if (textureNumber == 0)
    {
        if (nc == 4 || nc == 2)
        {
            if (!d_currentObject->transparent)
            {
                d_currentObject->transparent = true;
                d_currentObject->updateBin();
            }
        }
    }

    // Uwe's tiff hack (nc=40 means 4 byte/pixel but without transparency)
    if (nc > 4)
        nc = 4;

    int breite = 1;
    int hoehe = 1;
    for (int i = 0; i < 100; i++)
    {
        if (breite > w)
            break;
        else
            breite *= 2;
    }
    for (int i = 0; i < 100; i++)
    {
        if (hoehe > h)
            break;
        else
            hoehe *= 2;
    }
    breite /= 2;
    hoehe /= 2;

    // Pixel information is stored with rows rounded to 32bit boundaries
    int rowSize = (breite * nc + 3) / 4 * 4;

    /*
      fprintf(stderr, "insertTexture: w=%d, h=%d, b=%d, h=%d, rowsize=%d, nc=%d\n",
      w, h, breite, hoehe, rowSize, nc);
    */

    Texture *texture = d_currentObject->texData[textureNumber].texture.get();

    if (texture == NULL)
    {

        if (w == 0 && h == 0)
        {

            osg::Image *image;
            osgDB::ReaderWriter::Options *options = 0;

            // Flip DDS images per default
            if (coCoviseConfig::isOn("COVER.Plugin.Vrml97.DDSFlip", true))
                options = new osgDB::ReaderWriter::Options("dds_flip");

            d_currentObject->texData[textureNumber].texImage = image = osgDB::readImageFile((const char *)pixels, options);
            if (image)
            {
                if (image->r() > 1 && image->s() > 1 && image->t() > 1)
                {
                    d_currentObject->texData[textureNumber].texture = new Texture3D;
                    texture = NULL;
                }
                if (image->getPixelSizeInBits() == 32)
                {
                    if (!d_currentObject->transparent)
                    {
                        d_currentObject->transparent = true;
                        d_currentObject->updateBin();
                    }
                }
                texSize += image->getTotalSizeInBytesIncludingMipmaps();
            }
            else
            {

                return (TextureObject)-1;
            }
        }
        else
        {
            texSize += rowSize * hoehe * sizeof(char);
            d_currentObject->texData[textureNumber].texImage = new osg::Image();
        }
        if (d_currentObject->texData[textureNumber].texture.get() == NULL)
        {
            d_currentObject->texData[textureNumber].texture = texture = new Texture2D;
        }
        d_currentObject->texData[textureNumber].texture->setDataVariance(osg::Object::DYNAMIC);

        if (d_currentObject->texData[textureNumber].texImage == NULL)
        {
            return (TextureObject)-1;
        }
    }
    else
    {
        //fprintf(stderr, "updating Texture\n");
        if (cover->debugLevel(1))
            cerr << "T";
    }
    if (d_currentObject->texData[textureNumber].texImage)
    {
        d_currentObject->texData[textureNumber].texImage->setFileName(filename);
    }
    d_currentObject->texData[textureNumber].texture->setMaxAnisotropy(anisotropy);
    if (filter)
    {
        if (filter == 1)
        {
            // XXX: bilinear
            d_currentObject->texData[textureNumber].texture->setFilter(Texture::MIN_FILTER, Texture::LINEAR);
        }
        else if (filter == 2)
        {
            d_currentObject->texData[textureNumber].texture->setFilter(Texture::MIN_FILTER, Texture::LINEAR_MIPMAP_LINEAR);
        }
        else if (filter == 3)
        {
            // XXX: bilinear
            d_currentObject->texData[textureNumber].texture->setFilter(Texture::MIN_FILTER, Texture::LINEAR_MIPMAP_NEAREST);
        }
        else if (filter == 4)
        {
            // XXX: trilinear
            d_currentObject->texData[textureNumber].texture->setFilter(Texture::MIN_FILTER, Texture::LINEAR_MIPMAP_LINEAR);
        }
        else if (filter == 5)
        {
            // XXX: point ?
            d_currentObject->texData[textureNumber].texture->setFilter(Texture::MIN_FILTER, Texture::NEAREST);
        }
    }
    else
        d_currentObject->texData[textureNumber].texture->setFilter(Texture::MIN_FILTER, Texture::LINEAR_MIPMAP_LINEAR);
    d_currentObject->texData[textureNumber].texture->setFilter(Texture::MAG_FILTER, Texture::LINEAR);

    ////texture->setFilter(PFTEX_MINFILTER, PFTEX_BILINEAR);
    // Set the filter modes of the texture
    if (repeat_s)
        d_currentObject->texData[textureNumber].texture->setWrap(Texture::WRAP_S, Texture::REPEAT);
    else
        d_currentObject->texData[textureNumber].texture->setWrap(Texture::WRAP_S, Texture::CLAMP_TO_EDGE);
    if (repeat_t)
        d_currentObject->texData[textureNumber].texture->setWrap(Texture::WRAP_T, Texture::REPEAT);
    else
        d_currentObject->texData[textureNumber].texture->setWrap(Texture::WRAP_T, Texture::CLAMP_TO_EDGE);
    if (dynamic_cast<osg::Texture3D *>(d_currentObject->texData[textureNumber].texture.get()))
    {
        if (repeat_s)
            d_currentObject->texData[textureNumber].texture->setWrap(Texture::WRAP_R, Texture::REPEAT);
        else
            d_currentObject->texData[textureNumber].texture->setWrap(Texture::WRAP_R, Texture::CLAMP_TO_EDGE);
    }

#if 0
   if(textureQuality==1)
   {
      texture->setFormat(PFTEX_INTERNAL_FORMAT,PFTEX_RGBA_8);
      if(nc == 4)
         texture->setFormat(PFTEX_IMAGE_FORMAT,PFTEX_RGBA);
      else if(nc == 3)
         texture->setFormat(PFTEX_IMAGE_FORMAT,PFTEX_RGB);
   }
#endif
   osg::Image *pImage = d_currentObject->texData[textureNumber].texImage.get();
    if (w != 0 || h != 0)
    {
        const unsigned char *vrmlImage = pixels;
        //vrmlImage += breite*hoehe*nc;
        unsigned char *imageData = new unsigned char[nc * rowSize * hoehe];

        for (int i = 0; i < hoehe; i++)
        {
            int y = (int)((float)h * (float)((hoehe - i) - 1) / (float)hoehe);
            if (y > h - 1)
                y = h - 1;
            for (int j = 0; j < breite; j++)
            {
                int x = (int)((float)w * (float)j / (float)breite);
                if (x > w - 1)
                    x = w - 1;
                // XXX
                int irj = i * rowSize + j * nc;
                for (int comp = 0; comp < nc; comp++)
                    imageData[irj + comp] = vrmlImage[y * w * nc + x * nc + comp];
            }
        }

        GLint internalFormat = nc;
        if (textureQuality == 1) // High, 32bit texels
        {
            if (nc == 3)
            {
                internalFormat = GL_RGB8;
            }
            else if (nc == 4)
            {
                internalFormat = GL_RGBA8;
            }
        }
        else if (textureQuality == 2) // Low
        {
            if (nc == 3)
            {
                internalFormat = GL_RGB5;
            }
            else if (nc == 4)
            {
                internalFormat = GL_RGBA4;
            }
        }
        GLint format = GL_LUMINANCE;
        switch (nc)
        {
        case 1:
            format = GL_LUMINANCE;
            break;
        case 2:
            format = GL_LUMINANCE_ALPHA;
            break;
        case 3:
            format = GL_RGB;
            break;
        case 4:
            format = GL_RGBA;
            break;
        }
        pImage->setImage(breite, hoehe, 1, internalFormat, format, GL_UNSIGNED_BYTE, imageData, osg::Image::USE_NEW_DELETE /*, nc,breite, hoehe, 1*/);
    }
    //pImage = osgDB::readImageFile("c:\\src\\uwexp\\covise\\icons\\UI\\frame.rgb");
    d_currentObject->texData[textureNumber].texture->setImage(0, pImage);

#if 0
   if(textureQuality==1)
   {
      texture->setFormat(PFTEX_INTERNAL_FORMAT,PFTEX_RGBA_8);
      if(nc == 4)
         texture->setFormat(PFTEX_IMAGE_FORMAT,PFTEX_RGBA);
      else if(nc == 3)
         texture->setFormat(PFTEX_IMAGE_FORMAT,PFTEX_RGB);
   }
#endif

    d_currentObject->setTexEnv(environment, textureNumber, blendMode, nc);
    d_currentObject->setTexGen(environment, textureNumber, blendMode);

    //cerr << "ViewerOsg::insertTexture" << endl;
    if (cover->debugLevel(1))
        cerr << "o";
    cerr.flush();
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::insertTexture" << endl;
    return (TextureObject)d_currentObject->texData[textureNumber].texture.get();
}

osg::Image *openMovie(char *filename, movieImageData *movDat)
{
    static bool playerLibLoaded = false;
    //static bool ffmpegSound = false;

    if (!playerLibLoaded)
    {
        std::string playerLib = coCoviseConfig::getEntry("COVER.Plugin.Vrml97.MoviePlayer");
        if (playerLib.empty())
            playerLib = "ffmpeg";
        if (playerLib == "ffmpeg")
#ifdef _WIN32
        {
            //ffmpegSound = coCoviseConfig::isOn("COVER.Plugin.Vrml97.FFMPEGSound",false);
            std::string libName = osgDB::Registry::instance()->createLibraryNameForExtension(playerLib);
            playerLibLoaded = osgDB::Registry::instance()->loadLibrary(libName);
        }
        else
        {
            osgDB::Registry::instance()->addFileExtensionAlias("mpg", "qt");
            osgDB::Registry::instance()->addFileExtensionAlias("avi", "qt");
        }
#else
        {
            // ffmpegSound = coCoviseConfig::isOn("COVER.Plugin.Vrml97.FFMPEGSound",false);
            std::string libName = osgDB::Registry::instance()->createLibraryNameForExtension(playerLib);
            playerLibLoaded = osgDB::Registry::instance()->loadLibrary(libName);
        }
#endif
    }

    osg::Image *image = osgDB::readImageFile(filename);

    if (!image)
    {
        if (cover->debugLevel(2))
            cerr << "ViewerOsg::openMovie Reading " << filename << " failed" << endl;
        return (NULL);
    }

    osg::ImageStream *imageS = dynamic_cast<osg::ImageStream *>(image);
    if (imageS)
    {
        movDat->imageStream = (osg::ImageStream *)imageS;
        if (movDat->movieProp->loop)
            imageS->setLoopingMode(osg::ImageStream::LOOPING);
        else
            imageS->setLoopingMode(osg::ImageStream::NO_LOOPING);

        imageS->setReferenceTime(0);

#ifdef HAVE_SDL
        if (ffmpegSound)
        {
            osg::ImageStream::AudioStreams &audioStreams = imageS->getAudioStreams();
            if (!audioStreams.empty())
            {
                osg::AudioStream *audioStream = audioStreams[0].get();
                audioStream->setAudioSink(new SDLAudioSink(audioStream));
            }
        }
#endif
    }

#ifndef _WIN32
//   image->setPixelBufferObject(NULL);
#endif

    return (image);
}

Viewer::TextureObject
ViewerOsg::insertMovieTexture(char *filename,
                              movieProperties *movProp, int nc,
                              bool /*retainHint*/,
                              bool environment, int blendMode, int anisotropy, int filter)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::insertMovieTexture" << endl;

    Texture *tex = NULL;
    movieImageData *dataSet = new movieImageData();
    dataSet->movieProp = movProp;
    osg::Image *image = openMovie(filename, dataSet);

    d_currentObject->updateTexData(textureNumber + 1);

    if (image)
    {
        {
            Texture *texture = (Texture *)d_currentObject->texData[textureNumber].texture.get();
            if (texture == NULL)
            {
                d_currentObject->texData[textureNumber].texture = texture = new Texture2D();
                texture->setDataVariance(osg::Object::DYNAMIC);

            }
                if (cover->debugLevel(2))
                    cerr << "ViewerOsg::insertMovieTexture(" << filename << ")" << endl;

                d_currentObject->texData[textureNumber].texImage = image;
                moviePs.push_back(dataSet);
                d_currentObject->texData[textureNumber].texture = texture = new Texture2D(image);
				texture->setResizeNonPowerOfTwoHint(false);

                d_currentObject->texData[textureNumber].mirror = 1;

                if (image->getPixelSizeInBits() == 32)
                {
                    if (!d_currentObject->transparent)
                    {
                        d_currentObject->transparent = true;
                        d_currentObject->updateBin();
                    }
                }
                texSize += image->getTotalSizeInBytesIncludingMipmaps();

            if (filter)
            {
                if (filter == 1)
                {
                    // XXX: bilinear
                    texture->setFilter(Texture::MIN_FILTER, Texture::LINEAR);
                }
                else if (filter == 2)
                {
                    texture->setFilter(Texture::MIN_FILTER, Texture::LINEAR_MIPMAP_LINEAR);
                }
                else if (filter == 3)
                {
                    // XXX: bilinear
                    texture->setFilter(Texture::MIN_FILTER, Texture::LINEAR_MIPMAP_NEAREST);
                }
                else if (filter == 4)
                {
                    // XXX: trilinear
                    texture->setFilter(Texture::MIN_FILTER, Texture::LINEAR_MIPMAP_LINEAR);
                }
                else if (filter == 5)
                {
                    // XXX: point ?
                    texture->setFilter(Texture::MIN_FILTER, Texture::NEAREST);
                }
            }
            else
                texture->setFilter(Texture::MIN_FILTER, Texture::LINEAR_MIPMAP_LINEAR);
            texture->setFilter(Texture::MAG_FILTER, Texture::LINEAR);

            ////texture->setFilter(PFTEX_MINFILTER, PFTEX_BILINEAR);

            // Set the filter modes of the texture

            if (movProp->repeatS)
                texture->setWrap(Texture::WRAP_S, Texture::REPEAT);
            else
                texture->setWrap(Texture::WRAP_S, Texture::CLAMP_TO_BORDER);
            if (movProp->repeatT)
                texture->setWrap(Texture::WRAP_T, Texture::REPEAT);
            else
                texture->setWrap(Texture::WRAP_T, Texture::CLAMP_TO_BORDER);

            tex = texture;
        }

        tex->setMaxAnisotropy(anisotropy);

        d_currentObject->setTexEnv(environment, textureNumber, blendMode, nc);
        d_currentObject->setTexGen(environment, textureNumber, blendMode);

        //cerr << "ViewerOsg::insertMovieTexture" << endl;
        if (cover->debugLevel(1))
            cerr << "o";
    }

    cerr.flush();
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::insertMovieTexture" << endl;

    return (TextureObject)tex;
}

void ViewerOsg::insertMovieReference(TextureObject t, int nc, bool environment, int blendMode)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::insertMovieReference" << endl;

    d_currentObject->updateTexData(textureNumber + 1);

    if (d_currentObject->texData[textureNumber].texture == NULL)
    {
        {
            d_currentObject->texData[textureNumber].texture = (Texture *)t;
			d_currentObject->texData[textureNumber].texture->setResizeNonPowerOfTwoHint(false);
            d_currentObject->setTexEnv(environment, textureNumber, blendMode, nc);
            d_currentObject->setTexGen(environment, textureNumber, blendMode);
            osg::Texture *tex = d_currentObject->texData[textureNumber].texture.get();
			if (osg::Texture2D *tex2d = dynamic_cast<osg::Texture2D *>(tex))
			{
				d_currentObject->texData[textureNumber].texImage = tex2d->getImage();
			}
            else
                d_currentObject->texData[textureNumber].texImage = NULL;
            d_currentObject->texData[0].mirror = 1;
        }
    }
    if (cover->debugLevel(1))
        cerr << "x";

    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::insertMovieReference" << endl;
    //cerr.flush();
}

void ViewerOsg::insertTextureReference(TextureObject t, int nc, bool environment, int blendMode)
{
    //fprintf(stderr, "insertTextureReference(t=%ld, nc=%d)\n", t, nc);
    if (t == (TextureObject)-1)
        return;
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::insertTextureReference" << endl;

    d_currentObject->updateTexData(textureNumber + 1);
    if (d_currentObject->texData[textureNumber].texture == NULL)
    {
        d_currentObject->texData[textureNumber].texture = (Texture *)t;
        d_currentObject->setTexEnv(environment, textureNumber, blendMode, nc);
        d_currentObject->setTexGen(environment, textureNumber, blendMode);
        osg::Texture *tex = d_currentObject->texData[textureNumber].texture.get();
        if (osg::Texture2D *tex2d = dynamic_cast<osg::Texture2D *>(tex))
            d_currentObject->texData[textureNumber].texImage = tex2d->getImage();
        else
            d_currentObject->texData[textureNumber].texImage = NULL;

        d_currentObject->updateTexture();
        if (textureNumber == 0) // only first texture defines transparency, actually depends on texenv but...
        {
            if (nc == 4 || nc == 2)
            {
                if (!d_currentObject->transparent)
                {
                    d_currentObject->transparent = true;
                    d_currentObject->updateBin();
                }
            }
        }
        if (cover->debugLevel(1))
            cerr << "x";
    }

    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::insertTextureReference" << endl;
    //cerr << "ViewerOsg::insertTextureReference" << endl;
    //cerr.flush();
}

Viewer::TextureObject
ViewerOsg::insertCubeTexture(int w, int h, int nc,
                             bool /*repeat_s*/,
                             bool /*repeat_t*/,
                             unsigned char *pixelsXP,
                             unsigned char *pixelsXN,
                             unsigned char *pixelsYP,
                             unsigned char *pixelsYN,
                             unsigned char *pixelsZP,
                             unsigned char *pixelsZN,
                             bool /*retainHint*/,
                             int blendMode)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::insertCubeTexture" << endl;
    d_currentObject->updateTexData(textureNumber + 1);
    //fprintf(stderr, "Texture %x\n",pixels);
    // set transparency based on first texture
    if (textureNumber == 0)
    {
        if (nc == 4 || nc == 2)
        {
            if (!d_currentObject->transparent)
            {
                d_currentObject->transparent = true;
                d_currentObject->updateBin();
            }
        }
    }

    // Uwe's tiff hack (nc=40 means 4 byte/pixel but without transparency)
    if (nc > 4)
        nc = 4;

    int breite = 1;
    int hoehe = 1;
    for (int i = 0; i < 100; i++)
    {
        if (breite > w)
            break;
        else
            breite *= 2;
    }
    for (int i = 0; i < 100; i++)
    {
        if (hoehe > h)
            break;
        else
            hoehe *= 2;
    }
    breite /= 2;
    hoehe /= 2;

    // Pixel information is stored with rows rounded to 32bit boundaries
    int rowSize = (breite * nc + 3) / 4 * 4;

    /*
      fprintf(stderr, "insertTexture: w=%d, h=%d, b=%d, h=%d, rowsize=%d, nc=%d\n",
      w, h, breite, hoehe, rowSize, nc);
    */

    TextureCubeMap *cubemap = (TextureCubeMap *)(d_currentObject->texData[textureNumber].texture.get());

    if (cubemap == NULL)
    {
        d_currentObject->texData[textureNumber].texture = cubemap = new TextureCubeMap;
        //texture->setDataVariance(osg::Object::DYNAMIC);
    }
    else
    {
        //fprintf(stderr, "updating Texture\n");
        if (cover->debugLevel(1))
            cerr << "T";
    }

    GLint internalFormat = nc;
    if (textureQuality == 1) // High, 32bit texels
    {
        if (nc == 3)
        {
            internalFormat = GL_RGB8;
        }
        else if (nc == 4)
        {
            internalFormat = GL_RGBA8;
        }
    }
    else if (textureQuality == 2) // Low
    {
        if (nc == 3)
        {
            internalFormat = GL_RGB5;
        }
        else if (nc == 4)
        {
            internalFormat = GL_RGBA4;
        }
    }
    GLint format = GL_LUMINANCE;
    switch (nc)
    {
    case 1:
        format = GL_LUMINANCE;
        break;
    case 2:
        format = GL_LUMINANCE_ALPHA;
        break;
    case 3:
        format = GL_RGB;
        break;
    case 4:
        format = GL_RGBA;
        break;
    }

    const unsigned char *vrmlImage = NULL;
    //vrmlImage += breite*hoehe*nc;

    for (int n = 0; n < 6; n++)
    {
        unsigned char *imageData = new unsigned char[nc * rowSize * hoehe];
        if (n == 0)
            vrmlImage = pixelsXP;
        else if (n == 1)
            vrmlImage = pixelsXN;
        else if (n == 2)
            vrmlImage = pixelsYP;
        else if (n == 3)
            vrmlImage = pixelsYN;
        else if (n == 4)
            vrmlImage = pixelsZP;
        else if (n == 5)
            vrmlImage = pixelsZN;
        if (true) // 3dsMax
        {
            if (n == 5)
            {
                for (int i = 0; i < hoehe; i++)
                {
                    int y = h - ((int)((float)h * (float)((hoehe - i) - 1) / (float)hoehe));
                    if (y > h - 1)
                        y = h - 1;
                    for (int j = 0; j < breite; j++)
                    {
                        int x = w - ((int)((float)w * (float)j / (float)breite));
                        if (x > w - 1)
                            x = w - 1;
                        // XXX
                        int irj = i * rowSize + j * nc;
                        for (int comp = 0; comp < nc; comp++)
                            imageData[irj + comp] = vrmlImage[y * w * nc + x * nc + comp];
                    }
                }
            }
            else if (n == 3)
            {
                for (int i = 0; i < hoehe; i++)
                {
                    int y = h - ((int)((float)h * (float)((hoehe - i) - 1) / (float)hoehe));
                    if (y > h - 1)
                        y = h - 1;
                    for (int j = 0; j < breite; j++)
                    {
                        int x = w - ((int)((float)w * (float)j / (float)breite));
                        if (x > w - 1)
                            x = w - 1;
                        // XXX
                        int irj = i * rowSize + j * nc;
                        for (int comp = 0; comp < nc; comp++)
                            imageData[irj + comp] = vrmlImage[y * w * nc + x * nc + comp];
                    }
                }
            }
            else if (n == 0)
            {

                // Pixel information is stored with rows rounded to 32bit boundaries
                int rowSize2 = (hoehe * nc + 3) / 4 * 4;
                for (int i = 0; i < breite; i++)
                {
                    //int x=(int)( (float)w*(float)((breite-i)-1) /(float)breite );
                    int x = (int)((float)w * (float)i / (float)breite);
                    if (x > w - 1)
                        x = w - 1;
                    for (int j = 0; j < hoehe; j++)
                    {
                        int y = (int)((float)h * (float)j / (float)hoehe);
                        if (y > h - 1)
                            y = h - 1;
                        // XXX
                        int irj = i * rowSize2 + j * nc;
                        for (int comp = 0; comp < nc; comp++)
                            imageData[irj + comp] = vrmlImage[y * w * nc + x * nc + comp];
                    }
                }
            }
            else if (n == 1)
            {

                // Pixel information is stored with rows rounded to 32bit boundaries
                int rowSize2 = (hoehe * nc + 3) / 4 * 4;
                for (int i = 0; i < breite; i++)
                {
                    int x = (int)((float)w * (float)((breite - i) - 1) / (float)breite);
                    if (x > w - 1)
                        x = w - 1;
                    for (int j = 0; j < hoehe; j++)
                    {
                        int y = (int)((float)h * (float)((hoehe - j) - 1) / (float)hoehe);
                        if (y > h - 1)
                            y = h - 1;
                        // XXX
                        int irj = i * rowSize2 + j * nc;
                        for (int comp = 0; comp < nc; comp++)
                            imageData[irj + comp] = vrmlImage[y * w * nc + x * nc + comp];
                    }
                }
            }
            else
            {
                for (int i = 0; i < hoehe; i++)
                {
                    int y = w - (int)((float)w * (float)i / (float)hoehe);
                    if (y > h - 1)
                        y = h - 1;
                    for (int j = 0; j < breite; j++)
                    {
                        int x = h - (int)((float)h * (float)((breite - j) - 1) / (float)breite);
                        if (x > w - 1)
                            x = w - 1;
                        // XXX
                        int irj = i * rowSize + j * nc;
                        for (int comp = 0; comp < nc; comp++)
                            imageData[irj + comp] = vrmlImage[y * w * nc + x * nc + comp];
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < hoehe; i++)
            {
                int y = (int)((float)w * (float)i / (float)hoehe);
                if (y > h - 1)
                    y = h - 1;
                for (int j = 0; j < breite; j++)
                {
                    int x = (int)((float)h * (float)((breite - j) - 1) / (float)breite);
                    if (x > w - 1)
                        x = w - 1;
                    // XXX
                    int irj = i * rowSize + j * nc;
                    for (int comp = 0; comp < nc; comp++)
                        imageData[irj + comp] = vrmlImage[y * w * nc + x * nc + comp];
                }
            }
        }
        osg::Image *pImage = new osg::Image();
        if (true) //3dsmax
        {
            if (n == 0 || n == 1)
            {
                pImage->setImage(hoehe, breite, 1, internalFormat, format, GL_UNSIGNED_BYTE, imageData, osg::Image::USE_NEW_DELETE /*, nc,breite, hoehe, 1*/);
            }
            else
            {
                pImage->setImage(breite, hoehe, 1, internalFormat, format, GL_UNSIGNED_BYTE, imageData, osg::Image::USE_NEW_DELETE /*, nc,breite, hoehe, 1*/);
            }
        }
        else
        {
            pImage->setImage(breite, hoehe, 1, internalFormat, format, GL_UNSIGNED_BYTE, imageData, osg::Image::USE_NEW_DELETE /*, nc,breite, hoehe, 1*/);
        }
        if (n == 0)
            cubemap->setImage(osg::TextureCubeMap::POSITIVE_X, pImage);
        else if (n == 1)
            cubemap->setImage(osg::TextureCubeMap::NEGATIVE_X, pImage);
        else if (n == 2)
            cubemap->setImage(osg::TextureCubeMap::POSITIVE_Y, pImage);
        else if (n == 3)
            cubemap->setImage(osg::TextureCubeMap::NEGATIVE_Y, pImage);
        else if (n == 4)
            cubemap->setImage(osg::TextureCubeMap::POSITIVE_Z, pImage);
        else if (n == 5)
            cubemap->setImage(osg::TextureCubeMap::NEGATIVE_Z, pImage);
    }

    cubemap->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
    cubemap->setWrap(osg::Texture::WRAP_T, osg::Texture::CLAMP_TO_EDGE);
    cubemap->setWrap(osg::Texture::WRAP_R, osg::Texture::CLAMP_TO_EDGE);

    cubemap->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR_MIPMAP_LINEAR);
    cubemap->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);

    /*   if(repeat_s)
         cubemap->setWrap(Texture::WRAP_S, Texture::REPEAT);
      else
         cubemap->setWrap(Texture::WRAP_S, Texture::CLAMP);
      if(repeat_t)
         cubemap->setWrap(Texture::WRAP_T, Texture::REPEAT);
      else
         cubemap->setWrap(Texture::WRAP_T, Texture::CLAMP);*/
    d_currentObject->setTexEnv(false, textureNumber, blendMode, nc);
    d_currentObject->setTexGen(2, textureNumber, blendMode);

    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::insertCubeTexture" << endl;
    return (TextureObject)cubemap;
}

void ViewerOsg::insertCubeTextureReference(TextureObject t, int nc, int blendMode)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::insertCubeTextureReference" << endl;
    d_currentObject->updateTexData(textureNumber + 1);
    if (d_currentObject->texData[textureNumber].texture == NULL)
    {
        d_currentObject->texData[textureNumber].texture = (Texture *)t;
        d_currentObject->setTexEnv(false, textureNumber, blendMode, nc);
        d_currentObject->setTexGen(2, textureNumber, blendMode);
        //d_currentObject->texData[textureNumber].texImage = d_currentObject->texData[textureNumber].texture->getImage();

        d_currentObject->updateTexture();
        if (textureNumber == 0) // only first texture defines transparency, actually depends on texenv but...
        {
            if (nc == 4 || nc == 2)
            {
                if (!d_currentObject->transparent)
                {
                    d_currentObject->transparent = true;
                    d_currentObject->updateBin();
                }
            }
        }
        if (cover->debugLevel(1))
            cerr << "x";
    }
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::insertCubeTextureReference" << endl;
}

void ViewerOsg::removeCubeTextureObject(TextureObject)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::removeCubeTextureObject" << endl;
    //cerr << "t";
    cerr.flush();
}

void ViewerOsg::removeTextureObject(TextureObject t)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::removeTextureObject" << endl;
    osg::Texture *tex = (osg::Texture *)t;
    osg::Image *image = NULL;
    if (dynamic_cast<osg::Texture2D *>(tex))
        image = ((osg::Texture2D *)tex)->getImage();
    std::list<movieImageData *>::iterator it = moviePs.begin();
    for (; it != moviePs.end(); it++)
    {
        movieImageData *movieProp = (*it);
        if (movieProp->imageStream == (osg::ImageStream *)image)
        {
            movieProp->imageStream->quit(true);
            moviePs.erase(it);
            delete movieProp;
            break;
        }
    }

    d_currentObject->texData.resize(numTextures);
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::removeTextureObject" << endl;
}

// Texture coordinate transform
// Tc' = -C x S x R x C x T x Tc

void ViewerOsg::setTextureTransform(float *center,
                                    float rotation,
                                    float *scale,
                                    float *translation)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::setTextureTransform" << endl;

    d_currentObject->updateTexData(textureNumber + 1);
    if (center && scale && translation)
    {

        Matrix m2;
        d_currentObject->texData[textureNumber].newTMat.makeTranslate(-center[0], -center[1], 0.0);
        if (!FPEQUAL(scale[0], 1.0) || !FPEQUAL(scale[1], 1.0))
        {
            if (!FPEQUAL(scale[0], 0.0) && !FPEQUAL(scale[1], 0.0))
            {
                m2.makeScale(scale[0], scale[1], 1.0);
                d_currentObject->texData[textureNumber].newTMat.preMult(m2);
            }
        }
        if (rotation != 0.0)
        {
            m2.makeRotate(rotation, 0, 0, 1);
            d_currentObject->texData[textureNumber].newTMat.preMult(m2);
        }
        m2.makeTranslate(center[0], center[1], 0.0);
        d_currentObject->texData[textureNumber].newTMat.preMult(m2);
        m2.makeTranslate(translation[0], translation[1], 0.0);
        d_currentObject->texData[textureNumber].newTMat.preMult(m2);
    }
    else
    {
        d_currentObject->texData[textureNumber].newTMat.makeIdentity();
    }
    d_currentObject->updateTMat();

    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::setTextureTransform" << endl;
}

void ViewerOsg::setClip(float *pos,
                        float *ori,
                        int number,
                        bool enabled)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::setClip" << endl;
    osg::ClipNode *cn = (osg::ClipNode *)d_currentObject->pNode.get();
    osg::ClipPlane *cp = NULL;
    if (cn)
    {
        if (cn->getNumClipPlanes() > 0)
            cp = cn->getClipPlane(0);
    }
    if (!d_currentObject->pNode)
    {
        cn = new ClipNode();
        d_currentObject->pNode = cn;
        cp = new osg::ClipPlane();
        cp->setClipPlaneNum(number);
        cn->addClipPlane(cp);
        setModesByName();
        addToScene(d_currentObject);
        d_currentObject->addChildrensNodes();
    }

    if (enabled)
    {
        if (cp == NULL)
        {
            cp = new osg::ClipPlane();
        }
        cp->setClipPlaneNum(number);
        osg::Quat q(ori[3], osg::Vec3(ori[0], ori[1], ori[2]));
        osg::Vec3 normal(0, 0, 1);
        normal = q * normal;
        osg::Plane p(normal, osg::Vec3(pos[0], pos[1], pos[2]));
        cp->setClipPlane(p);
        if (cn->getNumClipPlanes() == 0)
            cn->addClipPlane(cp);
    }
    else
    {
        while (cn->getNumClipPlanes() > 0)
            cn->removeClipPlane((unsigned int)0);
    }
}


void ViewerOsg::setShadow(const std::string &technique)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::setShadow" << endl;
    osg::Group *ShadowGroup = (osg::Group *)d_currentObject->pNode.get();
    if (!d_currentObject->pNode)
    {
        ShadowGroup = new osg::Group();
        d_currentObject->pNode = ShadowGroup;
        setModesByName();
        addToScene(d_currentObject);
        d_currentObject->addChildrensNodes();
    }
    
    coVRShadowManager::instance()->setTechnique(technique);
}

// Transforms
// P' = T x C x R x SR x S x -SR x -C x P

void ViewerOsg::setTransform(float *center,
                             float *rotation,
                             float *scale,
                             float *scaleOrientation,
                             float *translation, bool changed)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::setTransform" << endl;
    if (!d_currentObject->pNode)
    {
        d_currentObject->pNode = new MatrixTransform();
        setModesByName();
        addToScene(d_currentObject);
        d_currentObject->addChildrensNodes();
    }
    /*Matrix mat,m2;
     mat.makeTranslate(translation[0],translation[1],translation[2]);
     m2.makeTranslate(center[0],center[1],center[2]);
     mat.postMult(m2);
     if (! FPZERO(rotation[3]) )
     {
     m2.makeRotate(rotation[3] ,rotation[0],rotation[1],rotation[2]);
     mat.postMult(m2);
     }
     if (! FPEQUAL(scale[0], 1.0) || ! FPEQUAL(scale[1], 1.0) || ! FPEQUAL(scale[2], 1.0) )
     {
     if (! FPZERO(scaleOrientation[3]) )
     {
     m2.makeRotate(-scaleOrientation[3],scaleOrientation[0],scaleOrientation[1],scaleOrientation[2]);
     mat.postMult(m2);
     m2.makeScale(scale[0],scale[1],scale[2]);
     mat.postMult(m2);
     m2.makeRotate(scaleOrientation[3] ,scaleOrientation[0],scaleOrientation[1],scaleOrientation[2]);
     mat.postMult(m2);
     }
     else
     {
     m2.makeScale(scale[0],scale[1],scale[2]);
     mat.postMult(m2);
     }
     }
     m2.makeTranslate(-center[0],-center[1],-center[2]);
     mat.postMult(m2);*/

    Matrix mat, m2;
    mat.makeTranslate(translation[0], translation[1], translation[2]);
    m2.makeTranslate(center[0], center[1], center[2]);
    mat.preMult(m2);
    if (!FPZERO(rotation[3]))
    {
        m2.makeRotate(rotation[3], rotation[0], rotation[1], rotation[2]);
        mat.preMult(m2);
    }
    if (!FPEQUAL(scale[0], 1.0) || !FPEQUAL(scale[1], 1.0) || !FPEQUAL(scale[2], 1.0))
    {
        if (!FPZERO(scaleOrientation[3]))
        {
            m2.makeRotate(scaleOrientation[3], scaleOrientation[0], scaleOrientation[1], scaleOrientation[2]);
            mat.preMult(m2);
            m2.makeScale(scale[0], scale[1], scale[2]);
            mat.preMult(m2);
            m2.makeRotate(-scaleOrientation[3], scaleOrientation[0], scaleOrientation[1], scaleOrientation[2]);
            mat.preMult(m2);
        }
        else
        {
            m2.makeScale(scale[0], scale[1], scale[2]);
            mat.preMult(m2);
        }
    }
    m2.makeTranslate(-center[0], -center[1], -center[2]);
    mat.preMult(m2);

    d_currentObject->parentTransform = currentTransform;
    currentTransform.preMult(mat);

    void *info = (void *)vrui::OSGVruiUserDataCollection::getUserData(d_currentObject->pNode.get(), "MoveInfo");
    if (info == NULL)
    {
        // leave alone moved nodes
        if (changed) // only move objects if the matrix actually changed, otherwise modifications in the tabletUI are reverted immediately
        {
            ((MatrixTransform*)d_currentObject->pNode.get())->setMatrix(mat);
        }
    }
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::setTransform" << endl;
}

// I used to just do a glPushMatrix()/glPopMatrix() in beginObject()/endObject().
// This is a hack to work around the glPushMatrix() limit (32 deep on Mesa).
// It has some ugly disadvantages: it is slower and the resulting transform
// after a setTransform/unsetTransform may not be identical to the original.
// It might be better to just build our own matrix stack...

void ViewerOsg::unsetTransform(float * /*center*/,
                               float * /*rotation*/,
                               float * /*scale*/,
                               float * /*scaleOrientation*/,
                               float * /*translation*/)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::unsetTransform" << endl;

    currentTransform = d_currentObject->parentTransform;
}

// The matrix gets popped at endObject() - Not anymore. I added
// an explicit unsetBillboardTransform to work around the matrix
// depth limit of 32 in mesa. Now the limit only applies to
// nested billboards.

void ViewerOsg::setBillboardTransform(float *axisOfRotation)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::setBillboardTransform" << endl;
    if (d_currentObject->billBoard == NULL)
    {
        d_currentObject->billBoard = new coBillboard();

        if (d_currentObject->pNode.get())
        {
            if (d_currentObject->pNode->getNumParents())
            {
                int i, nump;
                nump = d_currentObject->pNode->getNumParents();
                for (i = 0; i < nump; i++)
                {
                    Group *parentNode;
                    parentNode = (Group *)d_currentObject->pNode->getParent(0);
                    parentNode->removeChild(d_currentObject->pNode.get());
                    parentNode->addChild(d_currentObject->billBoard.get());
                }
            }
            d_currentObject->billBoard->addChild(d_currentObject->pNode.get());
        }
    }

    if (axisOfRotation[0] == 0 && axisOfRotation[1] == 0 && axisOfRotation[2] == 0)
    {
        d_currentObject->billBoard->setMode(coBillboard::POINT_ROT_WORLD);
        d_currentObject->billBoard->setAxis(Vec3(0, 0, 1));
        d_currentObject->billBoard->setNormal(Vec3(0, -1, 0));
    }
    else
    {
        Vec3 rotAxis(axisOfRotation[0], axisOfRotation[1], axisOfRotation[2]);
        d_currentObject->billBoard->setMode(coBillboard::AXIAL_ROT);
        d_currentObject->billBoard->setAxis(rotAxis);
        d_currentObject->billBoard->setNormal(Vec3(0, 0, 1));
    }
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::setBillboardTransform" << endl;
}

void ViewerOsg::unsetBillboardTransform(float * /*axisOfRotation*/)
{
    if (cover->debugLevel(2))
        cerr << "ViewerOsg::unsetBillboardTransform" << endl;
}

void ViewerOsg::setViewpoint(float *position,
                             float *orientation,
                             float fieldOfView,
                             float avatarSize,
                             float /*visibilityLimit*/,
                             const char *type,
                             float ascaleFactor)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::setViewpoint" << endl;
    static float pos[3] = { 0, 0, 0 };
    static float ori[4] = { 1, 0, 0, 0 };
    static float aS = 0.0;
    static float oldScale = 0.0;
    float scaleFactor = UseFieldOfViewForScaling ? fieldOfView : ascaleFactor;
    if (fieldOfView < -1000) // resetViewpointHack
    {
        if (cover->debugLevel(1))
            cerr << "reset Viewpoint" << endl;
        aS = 0.0;
        oldScale = 0.0;
    }
    if (orientation[0] == 0.0 && orientation[1] == 0.0 && orientation[2] == 0.0)
    {
        orientation[0] = 1.0;
        orientation[3] = 0.0;
    }
    if (avatarSize <= 0)
        avatarSize = 1.6;
    if (ascaleFactor <= 0)
        scaleFactor = 1000;
    if ((oldScale != scaleFactor) || (aS != avatarSize) || (position[0] != pos[0]) || (position[1] != pos[1]) || (position[2] != pos[2])
        || (orientation[0] != ori[0]) || (orientation[1] != ori[1]) || (orientation[2] != ori[2]) || (orientation[3] != ori[3]))
    {
        int i;
        for (i = 0; i < 3; i++)
        {
            pos[i] = position[i];
        }
        for (i = 0; i < 4; i++)
        {
            ori[i] = orientation[i];
        }
        oldScale = scaleFactor;
        aS = avatarSize;
        Matrix mat, rotMat;
        mat.makeTranslate(-pos[0], -pos[1], -pos[2]);
        rotMat.makeRotate(-ori[3], Vec3(ori[0], ori[1], ori[2]));

    //fprintf(stderr,"orientCamera: %f %f %f %f\n",ori[0], ori[1], ori[2], ori[3]);
        mat.postMult(rotMat);

        //get rid of scale part of the matrix
        /* Matrix invTransNoScale;
         Vec3 tmpvec;
         invTrans.getRow(0,tmpvec);
         tmpvec.normalize();
         invTransNoScale.setRow(0,tmpvec);
         invTrans.getRow(1,tmpvec);
         tmpvec.normalize();
         invTransNoScale.setRow(1,tmpvec);
         invTrans.getRow(2,tmpvec);
         tmpvec.normalize();
         invTransNoScale.setRow(2,tmpvec);
         invTrans.getRow(3,tmpvec);
         invTransNoScale.setRow(3,tmpvec);*/

        /*Matrix TMat;
      // get rid of non uniform scaling
      TMat.invert(mat);

      Vec3 v1(1,0,0),v2(0,1,0),v3(0,0,1),v4(0,0,0);
      Vec3 v1x(1,0,0),v2x(0,1,0),v3x(0,0,1),v4x(0,0,0);
      v1x = Matrix::transform3x3(v1, TMat);
      v1x.normalize();
      v2x = Matrix::transform3x3(v2, TMat);
      v2x.normalize();
      v3x = Matrix::transform3x3(v3, TMat);
      v3x.normalize();
      v4x = TMat.preMult(v4);
      TMat.set(v1x[0], v1x[1], v1x[2], 0.0,
      v2x[0], v2x[1], v2x[2], 0.0,
      v3x[0], v3x[1], v3x[2], 0.0,
      v4x[0], v4x[1], v4x[2], 1.0);
       */
        osg::Matrix CamTrans;
        CamTrans.makeIdentity();
        if (strcmp(type, "horizontal") == 0)
        {
            float pitch = asin(mat(1, 2));
            float cp = cos(pitch);
            float roll = acos(mat(2, 2) / cp);
            mat.makeRotate(roll, Vec3(0.0, 1.0, 0.0));
            coCoord coord = mat;
            coord.hpr[1] = 0.0;
            coord.hpr[0] = 0.0;
            coord.makeMat(mat);
        }
        else if (strncmp(type, "standard", 8) == 0) // no CAVE mode, viepoint in viewers position
        {
            if (strcmp(type, "standardNoFov") != 0 && fieldOfView > 0) // adjust field of view
            {
                float screen = std::min(coVRConfig::instance()->screens[0].hsize, coVRConfig::instance()->screens[0].vsize);
                float vd = (screen / 2) / tan(fieldOfView / 2);
                osg::Matrix tmp = osg::Matrix::translate(0, -vd, 0);
                VRViewer::instance()->setViewerMat(tmp);
            }
            osg::Vec3 vPos = VRViewer::instance()->getViewerPos();
            CamTrans = osg::Matrix::translate(vPos);
        }
        /*
         mat.invert(TMat);*/

        rotMat.makeRotate(M_PI / 2.0, Vec3(1.0, 0.0, 0.0));
        mat.postMult(rotMat);
        rotMat.makeRotate(-M_PI / 2.0, Vec3(1.0, 0.0, 0.0));
        mat.preMult(rotMat);

        Matrix scMat;
        Matrix iscMat;
        scMat.makeScale(scaleFactor, scaleFactor, scaleFactor);
        iscMat.makeScale(1.0 / scaleFactor, 1.0 / scaleFactor, 1.0 / scaleFactor);
        mat.postMult(scMat);
        mat.preMult(iscMat);

        cover->setXformMat(mat * CamTrans);
        //cerr << "oldScale " << cover->getScale() << " newScale: " << 50/aS <<endl;
        //cerr << "pos " << pos[0] << ";" <<pos[1] << ";" <<pos[2] << endl;
        //cerr << "or " << or[0] << ";" <<or[1] << ";" <<or[2] << endl;
        //cerr << "as " << aS << endl;
        cover->setScale(scaleFactor);
        //cerr << "New Scale:" << scale << endl;
    }
    //cerr << "ViewerOsg::setViewpoint" << endl;
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::setViewpoint" << endl;
}

// The viewer knows the current viewpoint

void ViewerOsg::transformPoints(int /*np*/, float * /*p*/)
{

    if (cover->debugLevel(5))
        cerr << "ViewerOsg::transformPoints" << endl;
}

osg::Vec3 reflect(osg::Vec3 L, osg::Vec3 N)
{
    return (N * ((N * L) * 2)) - L;
}

osg::Vec3 closestPoint(osg::Vec3 a1, osg::Vec3 b1, osg::Vec3 a2, osg::Vec3 b2)
{
    osg::Vec3 p1, cd;
    osg::Matrix m;
    cd = b1 ^ b2;
    m.makeIdentity();
    m(0, 0) = b1[0];
    m(0, 1) = b1[1];
    m(0, 2) = b1[2];
    m(1, 0) = -b2[0];
    m(1, 1) = -b2[1];
    m(1, 2) = -b2[2];
    m(2, 0) = cd[0];
    m(2, 1) = cd[1];
    m(2, 2) = cd[2];
    osg::Matrix im;
    im.invert(m);
    osg::Vec3 rst = (a2 - a1) * im;
    p1 = a1 + b1 * rst[0];
    return p1;
}

//
//  Viewer callbacks (called from window system specific functions)
//

// update is called from a timer callback
bool ViewerOsg::update(double timeNow)
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::update" << endl;
    bool updated = false;
    sensorList.update();
    std::list<movieImageData *>::iterator it = moviePs.begin();
    for (; it != moviePs.end(); it++)
    {
        movieImageData *movDat = (*it);
        osg::ImageStream *imageS = movDat->imageStream.get();
        if (imageS->getStatus() != osg::ImageStream::PLAYING)
        {
            if (movDat->movieProp->playing && !(movDat->movieProp->start > 0))
            {
                movDat->movieProp->playing = false;
                movDat->movieProp->mtNode->stopped();
            }
            if (movDat->movieProp->start > 0)
            {
                //     	                	imageS->setReferenceTime((cover->frameTime()-startLoadTime)*movieProp->speed*1000);
                imageS->setReferenceTime(0);
                imageS->rewind();
                imageS->play();
                movDat->movieProp->playing = true;
                imageS->setTimeMultiplier((double)fabs(movDat->movieProp->speed));
                if(movDat->movieProp->stop <movDat->movieProp->start)
                    movDat->movieProp->stop = -2;
                movDat->movieProp->start = -2;
                movDat->movieProp->speed = -movDat->movieProp->speed;
            }
            else if (!movDat->movieProp->loop && (movDat->movieProp->stop > 0))
            {
#ifndef _WIN32
                movDat->movieProp->stop = -2;
#else
                imageS->setReferenceTime(0);
#endif
            }
        }
        else
        {
#ifdef NEW_OSG
            if (imageS->getStatus() != osg::ImageStream::PAUSED)
            {
                if (movDat->movieProp->stop > 0)
                {
#ifdef _WIN32
                    if (!movDat->movieProp->loop && (imageS->getLength() - imageS->getReferenceTime() < 0.05))
                        imageS->setReferenceTime(0);
#endif
                    imageS->pause();
                    movDat->movieProp->stop = -2;
                }
                else if (movDat->movieProp->speed > 0)
                {
                    imageS->setTimeMultiplier((double)movDat->movieProp->speed);
                    movDat->movieProp->speed = -movDat->movieProp->speed;
                }
            }
#endif
        }
    }

    framebufferTextureUpdated = false;
    if (countTextures && texSize > oldTexSize)
    {
        if (cover->debugLevel(1))
            cerr << endl << endl << "Texture Size: " << texSize << endl;

        //pfdPrintSceneGraphStats((Node *)cover->getScene(),  cover->frameTime() - startLoadTime);
        oldTexSize = texSize;
    }

    if (countGeometry && (numVert > oldNumVert || numPoly > oldNumPoly))
    {
        if (cover->debugLevel(1))
            cerr << endl << endl << "Vertices: " << numVert << ", Polygons: " << numPoly << endl;
        oldNumVert = numVert;
        oldNumPoly = numPoly;
    }
    if (d_scene)
    {
        currentTransform.makeIdentity();
        updated = d_scene->update(timeNow);
        redraw();
    }
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::update" << endl;

    return updated;
}

void ViewerOsg::preFrame()
{

	//,j,k;
	for (int i = 0; i < numCameras; i++)
	{
		osg::Camera *camera = mirrors[i].camera;
		if (mirrors[i].isVisible)
		{
			if (camera->getCullMask() == 0)
			{
				fprintf(stderr, "\nCamera on %d\n", i);
				camera->setCullMask(Isect::NoMirror);
			}
			/*
			if (mirrors[i].camera->getNumParents() == 0)
			{
				fprintf(stderr, "Camera on %d\n", i);
				cover->getScene()->addChild(mirrors[i].camera.get());
			}*/
		}
		else
		{
			if (camera->getCullMask() != 0)
			{
				fprintf(stderr, "\nCamera off %d\n", i);
				camera->setCullMask(0);
			}
			/*
			if (mirrors[i].camera->getNumParents() != 0)
			{
				fprintf(stderr, "Camera off %d\n", i);
				//cover->getScene()->removeChild(mirrors[i].camera.get());
			}
			*/
		}
		mirrors[i].isVisible = false;
		// getPlane in wc
		osg::Node *currentNode;
		osg::Matrix geoToWC, tmpMat;
		currentNode = mirrors[i].geometry->getParent(0);
		geoToWC.makeIdentity();
		while (currentNode != NULL)
		{
			if (dynamic_cast<osg::MatrixTransform *>(currentNode))
			{
				tmpMat = ((osg::MatrixTransform *)currentNode)->getMatrix();
				geoToWC.postMult(tmpMat);
			}
			if (currentNode->getNumParents() > 0)
				currentNode = currentNode->getParent(0);
			else
				currentNode = NULL;
		}

		if (mirrors[i].CameraID < 0)
		{

			osg::Matrix WCToGeo;
			//osg::Vec3 wcCoords[4];
			//for(j=0;j<4;j++)
			//   wcCoords[j]=geoToWC.preMult(mirrors[i].coords[j]);
			osg::Vec3 viewerPosMirror, mirrorViewerInMirrorCS;

			if (coVRConfig::instance()->stereoMode() == osg::DisplaySettings::RIGHT_EYE)
				viewerPos.set(VRViewer::instance()->getSeparation() / 2.0f, 0.0f, 0.0f);
			else if (coVRConfig::instance()->stereoMode() == osg::DisplaySettings::LEFT_EYE)
				viewerPos.set(-(VRViewer::instance()->getSeparation() / 2.0f), 0.0f, 0.0f);
			else if (coVRConfig::instance()->monoView() == coVRConfig::MONO_RIGHT)
				viewerPos.set(VRViewer::instance()->getSeparation() / 2.0f, 0.0f, 0.0f);
			else if (coVRConfig::instance()->monoView() == coVRConfig::MONO_LEFT)
				viewerPos.set(-(VRViewer::instance()->getSeparation() / 2.0f), 0.0f, 0.0f);
			else if (coVRConfig::instance()->channels[0].stereoMode == osg::DisplaySettings::RIGHT_EYE)
				viewerPos.set(VRViewer::instance()->getSeparation() / 2.0f, 0.0f, 0.0f);
			else if (coVRConfig::instance()->channels[0].stereoMode == osg::DisplaySettings::LEFT_EYE)
				viewerPos.set(-(VRViewer::instance()->getSeparation() / 2.0f), 0.0f, 0.0f);

			else
				viewerPos.set(0.0, 0.0, 0.0);
			osg::Matrix viewerMat = cover->getViewerMat();
			viewerPos = viewerMat.preMult(viewerPos);
			WCToGeo.invert(geoToWC);

#ifdef DEBUG_LINES
			if (updateCameraButton->getState())
			{
				viewerInMirrorCS = WCToGeo.preMult(viewerPos);
			}
#else

			viewerInMirrorCS = WCToGeo.preMult(viewerPos);
#endif

			if (mirrors[i].shader) // compute viewing frustum for spherical/aspherical mirrors
			{
				osg::Uniform *radiusU = mirrors[i].shader->getUniform("Radius");
				osg::Uniform *aU = mirrors[i].shader->getUniform("a");
				osg::Uniform *KU = mirrors[i].shader->getUniform("K");
				float Radius;
				radiusU->get(Radius);
				float a;
				aU->get(a);
				float K;
				KU->get(K);
				osg::Vec3 cornerPos[4];
				osg::Vec3 flatCornerPos[4];
				osg::Vec3 normal;
				osg::Vec3 reflectVecs[4];
				osg::Vec3 normalVecs[4];
				for (int n = 0; n < 4; n++)
				{
					float xpos = mirrors[i].coords[n][0];
					float zpos = mirrors[i].coords[n][2];
					cornerPos[n] = mirrors[i].coords[n];
					flatCornerPos[n] = mirrors[i].coords[n];
					flatCornerPos[n][1] = 0;

					cornerPos[n][1] = Radius * -2 + sqrt(Radius * Radius - xpos * xpos) + sqrt(Radius * Radius - zpos * zpos);
					if (a > 0 && xpos > a)
					{
						float tmpf = xpos - a;
						cornerPos[n][1] = cornerPos[n][1] - K * tmpf * tmpf * tmpf;
						normal[0] = xpos / sqrt(Radius * Radius - xpos * xpos) + (3 * K * (xpos - a) * (xpos - a));
						normal[1] = 1;
						normal[2] = zpos / sqrt(Radius * Radius - zpos * zpos);
					}
					else if (a < 0 && xpos < a)
					{
						float tmpf = -(xpos - a);
						cornerPos[n][1] = cornerPos[n][1] - K * tmpf * tmpf * tmpf;
						normal[0] = xpos / sqrt(Radius * Radius - xpos * xpos) - (3 * K * (xpos - a) * (xpos - a));
						normal[1] = 1;
						normal[2] = zpos / sqrt(Radius * Radius - zpos * zpos);
					}
					else
					{
						normal[0] = xpos / sqrt(Radius * Radius - xpos * xpos);
						normal[1] = 1;
						normal[2] = zpos / sqrt(Radius * Radius - zpos * zpos);
					}
					normal.normalize();
					osg::Vec3 toViewer = viewerInMirrorCS - cornerPos[n];
					toViewer.normalize();
					reflectVecs[n] = reflect(toViewer, normal);
					normalVecs[n] = normal;
				}
#ifdef oldMethod
				osg::Vec3 points[4];
				points[0] = closestPoint(cornerPos[0], reflectVecs[0], cornerPos[1], reflectVecs[1]);
				points[1] = closestPoint(cornerPos[1], reflectVecs[1], cornerPos[2], reflectVecs[2]);
				points[2] = closestPoint(cornerPos[2], reflectVecs[2], cornerPos[3], reflectVecs[3]);
				points[3] = closestPoint(cornerPos[3], reflectVecs[3], cornerPos[0], reflectVecs[0]);
				mirrorViewerInMirrorCS.set(0, 0, 0);
				float min = fabs(points[0][1]);
				mirrorViewerInMirrorCS = points[0];
				mirrorViewerInMirrorCS += points[2];
				mirrorViewerInMirrorCS /= 2.0;

				for (int n = 0; n < 4; n++)
				{
					if (fabs(points[n][1]) < min)
					{
						if (n == 1 || n == 3)
						{
							mirrorViewerInMirrorCS = points[1];
							mirrorViewerInMirrorCS += points[3];
							mirrorViewerInMirrorCS /= 2.0;
						}
						else
						{
							mirrorViewerInMirrorCS = points[0];
							mirrorViewerInMirrorCS += points[2];
							mirrorViewerInMirrorCS /= 2.0;
						}
						//viewerInMirrorCS = points[n];
						min = fabs(points[n][1]);
					}
					//viewerInMirrorCS += points[n];
				}
				//viewerInMirrorCS /=4.0;
#else

#ifdef TaugtNichts
				// 3     2
				// 0     1
				osg::Vec3 cposlr;
				osg::Vec3 cposud;
				if (reflectVecs[0][0] < reflectVecs[3][0])
				{
					osg::Vec3 d = cornerPos[1] - cornerPos[0];
					float f1 = ((d[0] * reflectVecs[1][1]) - (d[1] * reflectVecs[1][0])) / ((reflectVecs[0][0] * reflectVecs[1][1]) - (reflectVecs[1][0] * reflectVecs[0][1]));
					cposlr = cornerPos[0] + (reflectVecs[0] * f1);
				}
				else
				{
					osg::Vec3 d = cornerPos[2] - cornerPos[3];
					float f1 = ((d[0] * reflectVecs[2][1]) - (d[1] * reflectVecs[2][0])) / ((reflectVecs[3][0] * reflectVecs[2][1]) - (reflectVecs[2][0] * reflectVecs[3][1]));
					cposlr = cornerPos[3] + (reflectVecs[3] * f1);
				}

				if (reflectVecs[0][2] > reflectVecs[1][2])
				{
					osg::Vec3 d = cornerPos[3] - cornerPos[0];
					float f1 = ((d[0] * reflectVecs[3][1]) - (d[1] * reflectVecs[3][0])) / ((reflectVecs[0][0] * reflectVecs[3][1]) - (reflectVecs[3][0] * reflectVecs[0][1]));
					cposud = cornerPos[0] + (reflectVecs[0] * f1);
				}
				else
				{
					osg::Vec3 d = cornerPos[2] - cornerPos[1];
					float f1 = ((d[0] * reflectVecs[2][1]) - (d[1] * reflectVecs[2][0])) / ((reflectVecs[1][0] * reflectVecs[2][1]) - (reflectVecs[2][0] * reflectVecs[1][1]));
					cposud = cornerPos[1] + (reflectVecs[1] * f1);
				}
				if (cposlr[1] < cposud[1])
				{
					mirrorViewerInMirrorCS = cposlr;
				}
				else
				{
					mirrorViewerInMirrorCS = cposud;
				}
#endif

				// von vorne
				// 0     1
				// 3     2
				// von hinten
				// 1     0
				// 2     3

				//left/right
				osg::Vec3 leftR, rightR, cposlr, cposud, pl, pr;
				if (reflectVecs[3][0] / reflectVecs[3][1] < reflectVecs[2][0] / reflectVecs[2][1])
				{
					leftR = reflectVecs[3];
					pl = cornerPos[3];
				}
				else
				{
					leftR = reflectVecs[2];
					pl = cornerPos[2];
				}

				if (reflectVecs[0][0] / reflectVecs[0][1] > reflectVecs[1][0] / reflectVecs[1][1])
				{
					rightR = reflectVecs[0];
					pr = cornerPos[0];
				}
				else
				{
					rightR = reflectVecs[1];
					pr = cornerPos[1];
				}
				leftR[2] = rightR[2] = pr[2] = pl[2] = 0;
				cposlr = closestPoint(pl, leftR, pr, rightR);

				// von vorne
				// 0     1
				// 3     2
				// von hinten
				// 1     0
				// 2     3

				//up/down
				osg::Vec3 up, down, pu, pd;
				if (reflectVecs[0][2] / reflectVecs[0][1] > reflectVecs[1][2] / reflectVecs[1][1])
				{
					down = reflectVecs[0];
					pd = cornerPos[0];
				}
				else
				{
					down = reflectVecs[1];
					pd = cornerPos[1];
				}

				if (reflectVecs[3][2] / reflectVecs[3][1] < reflectVecs[2][2] / reflectVecs[2][1])
				{
					up = reflectVecs[3];
					pu = cornerPos[3];
				}
				else
				{
					up = reflectVecs[2];
					pu = cornerPos[2];
				}
				up[0] = down[0] = pu[0] = pd[0] = 0;
				cposud = closestPoint(pu, up, pd, down);

				if (cposlr[1] > cposud[1])
				{
					mirrorViewerInMirrorCS = cposlr;
					mirrorViewerInMirrorCS[2] = pd[2] + down[2] * (cposlr[1] - pd[1]) / (down[1]);
				}
				else
				{
					mirrorViewerInMirrorCS = cposud;
					mirrorViewerInMirrorCS[0] = pl[0] + leftR[0] * (cposud[1] - pl[1]) / (leftR[1]);
				}

#endif

				//mirrors[i].shader->setVec3Uniform("viewerInMirrorCS",viewerInMirrorCS);

				viewerPosMirror = geoToWC.preMult(mirrorViewerInMirrorCS);

				osg::Vec3 tmpV;
				tmpV.set(WCToGeo(0, 0), WCToGeo(0, 1), WCToGeo(0, 2));
				float scale = tmpV.length();
				float miny = std::min(cornerPos[0][1], cornerPos[1][1]);
				miny = std::min(miny, cornerPos[2][1]);
				miny = std::min(miny, cornerPos[3][1]);
				float dist = -(mirrorViewerInMirrorCS[1] - miny);

				float distl = -(mirrorViewerInMirrorCS[1] - pr[1]);
				float distr = -(mirrorViewerInMirrorCS[1] - pl[1]);
				float distu = -(mirrorViewerInMirrorCS[1] - pd[1]);
				float distd = -(mirrorViewerInMirrorCS[1] - pu[1]);

				// relation near plane to screen plane
				float n_over_d = 1.0;
				//float dx=mirrors[i].coords[1][0]-mirrors[i].coords[0][0];
				//float dz=mirrors[i].coords[3][2]-mirrors[i].coords[0][2];
				// parameter of right channel

				// 0     1
				// 3     2

				float right = -n_over_d * (pl[0] - mirrorViewerInMirrorCS[0]) / scale * (dist / distr);
				float left = -n_over_d * (pr[0] - mirrorViewerInMirrorCS[0]) / scale * (dist / distl);
				float top = -n_over_d * (pd[2] - mirrorViewerInMirrorCS[2]) / scale * (dist / distu);
				float bottom = -n_over_d * (pu[2] - mirrorViewerInMirrorCS[2]) / scale * (dist / distd);
				float nearPlane = dist * 1.0 / scale;

				mirrors[i].camera->setProjectionMatrixAsFrustum(left, right, bottom, top, nearPlane, coVRConfig::instance()->farClip());

#ifdef DEBUG_LINES
				//ObjektY in WC
				osg::Vec3 objectY_WC;
				objectY_WC.set(geoToWC(1, 0), geoToWC(1, 1), geoToWC(1, 2));
				objectY_WC.normalize();
				//objectY_WC*=-1;

				//Up in WC
				osg::Vec3 up_WC;
				up_WC.set(geoToWC(2, 0), geoToWC(2, 1), geoToWC(2, 2));
				up_WC.normalize();
				up_WC *= -1;

				mirrors[i].camera->setViewMatrixAsLookAt(viewerPosMirror, viewerPosMirror + objectY_WC, up_WC);
				osg::Matrix proj = mirrors[i].camera->getProjectionMatrix();
				osg::Matrix mv = mirrors[i].camera->getViewMatrix();

				osg::Vec3 pos;
				osg::Vec3 pos2;
				osg::Vec4f c(1, 1, 1, 1);
				int vnum = 0;
#define addLine(a, b, c)       \
    (*LineVerts)[vnum].set(a); \
    (*LineColors)[vnum++] = c; \
    (*LineVerts)[vnum].set(b); \
    (*LineColors)[vnum++] = c
				pos = geoToWC.preMult(cornerPos[0]);
				addLine(viewerPosMirror, pos, c);
				pos = geoToWC.preMult(cornerPos[1]);
				addLine(viewerPosMirror, pos, osg::Vec4(1, 1, 1, 1));
				pos = geoToWC.preMult(cornerPos[2]);
				addLine(viewerPosMirror, pos, osg::Vec4(1, 1, 1, 1));
				pos = geoToWC.preMult(cornerPos[3]);
				addLine(viewerPosMirror, pos, osg::Vec4(1, 1, 1, 1));
				pos = geoToWC.preMult(cornerPos[0] - reflectVecs[0] * 1000);
				pos2 = geoToWC.preMult(cornerPos[0] + reflectVecs[0] * 1000);
				addLine(pos, pos2, osg::Vec4(1, 0, 0, 1));
				pos = geoToWC.preMult(cornerPos[1] - reflectVecs[1] * 1000);
				pos2 = geoToWC.preMult(cornerPos[1] + reflectVecs[1] * 1000);
				addLine(pos, pos2, osg::Vec4(1, 0, 0, 1));
				pos = geoToWC.preMult(cornerPos[2] - reflectVecs[2] * 1000);
				pos2 = geoToWC.preMult(cornerPos[2] + reflectVecs[2] * 1000);
				addLine(pos, pos2, osg::Vec4(1, 0, 0, 1));
				pos = geoToWC.preMult(cornerPos[3] - reflectVecs[3] * 1000);
				pos2 = geoToWC.preMult(cornerPos[3] + reflectVecs[3] * 1000);
				addLine(pos, pos2, osg::Vec4(1, 0, 0, 1));
				pos = geoToWC.preMult(cornerPos[0]);
				pos2 = geoToWC.preMult(cornerPos[1]);
				addLine(pos, pos2, osg::Vec4(0, 1, 0, 1));
				pos = geoToWC.preMult(cornerPos[1]);
				pos2 = geoToWC.preMult(cornerPos[2]);
				addLine(pos, pos2, osg::Vec4(0, 1, 0, 1));
				pos = geoToWC.preMult(cornerPos[2]);
				pos2 = geoToWC.preMult(cornerPos[3]);
				addLine(pos, pos2, osg::Vec4(0, 1, 0, 1));
				pos = geoToWC.preMult(cornerPos[3]);
				pos2 = geoToWC.preMult(cornerPos[0]);
				addLine(pos, pos2, osg::Vec4(0, 1, 0, 1));

				pos = geoToWC.preMult(flatCornerPos[0]);
				pos2 = geoToWC.preMult(flatCornerPos[1]);
				addLine(pos, pos2, osg::Vec4(0, 0, 1, 1));
				pos = geoToWC.preMult(flatCornerPos[1]);
				pos2 = geoToWC.preMult(flatCornerPos[2]);
				addLine(pos, pos2, osg::Vec4(0, 0, 1, 1));
				pos = geoToWC.preMult(flatCornerPos[2]);
				pos2 = geoToWC.preMult(flatCornerPos[3]);
				addLine(pos, pos2, osg::Vec4(0, 0, 1, 1));
				pos = geoToWC.preMult(flatCornerPos[3]);
				pos2 = geoToWC.preMult(flatCornerPos[0]);
				addLine(pos, pos2, osg::Vec4(0, 0, 1, 1));

				pos = geoToWC.preMult(cornerPos[0]);
				pos2 = geoToWC.preMult(cornerPos[0] + normalVecs[0] * 100);
				addLine(pos, pos2, osg::Vec4(1, 1, 0, 1));
				pos = geoToWC.preMult(cornerPos[1]);
				pos2 = geoToWC.preMult(cornerPos[1] + normalVecs[1] * 100);
				addLine(pos, pos2, osg::Vec4(1, 1, 0, 1));
				pos = geoToWC.preMult(cornerPos[2]);
				pos2 = geoToWC.preMult(cornerPos[2] + normalVecs[2] * 100);
				addLine(pos, pos2, osg::Vec4(1, 1, 0, 1));
				pos = geoToWC.preMult(cornerPos[3]);
				pos2 = geoToWC.preMult(cornerPos[3] + normalVecs[3] * 100);
				addLine(pos, pos2, osg::Vec4(1, 1, 0, 1));

				pos = viewerPosMirror;
				pos2 = viewerPosMirror + objectY_WC;
				addLine(pos, pos2, osg::Vec4(0, 1, 1, 1));
				pos = viewerPosMirror;
				pos2 = viewerPosMirror + up_WC;
				addLine(pos, pos2, osg::Vec4(0, 1, 1, 1));
#endif
			}
			else
			{
				viewerInMirrorCS = WCToGeo.preMult(viewerPos);
				viewerInMirrorCS[1] = -viewerInMirrorCS[1];
				viewerPosMirror = geoToWC.preMult(viewerInMirrorCS);

				osg::Vec3 tmpV;
				tmpV.set(WCToGeo(0, 0), WCToGeo(0, 1), WCToGeo(0, 2));

				// von vorne
				// 0     1
				// 3     2
				// von hinten
				// 1     0
				// 2     3

				float scale = tmpV.length();
				float dist = -(viewerInMirrorCS[1] - mirrors[i].coords[0][1]);

				// relation near plane to screen plane
				float n_over_d = 1.2;
				//float dx=mirrors[i].coords[1][0]-mirrors[i].coords[0][0];
				//float dz=mirrors[i].coords[3][2]-mirrors[i].coords[0][2];
				// parameter of right channel
				float right = -n_over_d * (mirrors[i].coords[0][0] - viewerInMirrorCS[0]) / scale;
				float left = -n_over_d * (mirrors[i].coords[1][0] - viewerInMirrorCS[0]) / scale;
				float top = -n_over_d * (mirrors[i].coords[0][2] - viewerInMirrorCS[2]) / scale;
				float bottom = -n_over_d * (mirrors[i].coords[3][2] - viewerInMirrorCS[2]) / scale;
				float nearPlane = dist * 1.2 / scale;

				mirrors[i].camera->setProjectionMatrixAsFrustum(left, right, bottom, top, nearPlane, coVRConfig::instance()->farClip());
			}

			// Now in World coordinates for rendering
			//osg::Matrix baseMat = cover->getBaseMat();
			//ObjektY in WC

			osg::Vec3 objectY_WC;
			objectY_WC.set(geoToWC(1, 0), geoToWC(1, 1), geoToWC(1, 2));
			objectY_WC.normalize();
			//objectY_WC*=-1;

			//Up in WC
			osg::Vec3 up_WC;
			up_WC.set(geoToWC(2, 0), geoToWC(2, 1), geoToWC(2, 2));
			up_WC.normalize();
			up_WC *= -1;

			mirrors[i].camera->setViewMatrixAsLookAt(viewerPosMirror, viewerPosMirror + objectY_WC, up_WC);
			//update all mirror cameras

			osg::Matrix proj = mirrors[i].camera->getProjectionMatrix();
			osg::Matrix mv = mirrors[i].camera->getViewMatrix();

			osg::Matrix nmv;
			osg::Matrix npm;
			if (coVRConfig::instance()->getEnvMapMode() == coVRConfig::NONE)
			{
				nmv = mv;
				npm = proj;
			}
			else
			{
				osg::Matrix rotonly = mv;
				rotonly(3, 0) = 0;
				rotonly(3, 1) = 0;
				rotonly(3, 2) = 0;
				rotonly(3, 3) = 1;
				osg::Matrix invRot;

				invRot.invert(rotonly);
				nmv = (mv * invRot) * cover->invEnvCorrectMat;
				npm = cover->envCorrectMat * rotonly * proj;
			}

			mirrors[i].camera->setViewMatrix(nmv);
			mirrors[i].camera->setProjectionMatrix(npm);
			if (mirrors[i].shader)
			{
				osg::Matrixf nmvf = nmv;
				osg::Matrixf npmf = npm;
				osg::Matrixf geoToWCf = geoToWC;
				mirrors[i].shader->setMatrixUniform("ViewMatrix", nmvf);
				mirrors[i].shader->setMatrixUniform("ProjectionMatrix", npmf);
				mirrors[i].shader->setMatrixUniform("ModelMatrix", geoToWCf);
				if (mirrors[i].instance)
				{
					osg::Uniform *U;
					U = mirrors[i].instance->getUniform("ViewMatrix");
					if (U)
					{
						U->set(nmv);
					}
					U = mirrors[i].instance->getUniform("ProjectionMatrix");
					if (U)
					{
						U->set(npm);
					}
					U = mirrors[i].instance->getUniform("ModelMatrix");
					if (U)
					{
						U->set(geoToWC);
					}
				}
				/*osg::Vec3 tmpv(1,1,1);
				osg::Vec3 tmpv2;
				tmpv2 = tmpv * ProjInMirrorCS ;
				fprintf(stderr,"%f %f %f\n",tmpv2[0],tmpv2[1],tmpv2[2]);*/
			}
		}
		else
		{
			// this is a camera based mirror, get the view and projection matrix from the appropriate rear view camera.

			osg::Matrix nmv = mirrors[i].vm;
			osg::Matrix npm = mirrors[i].pm;
			mirrors[i].camera->setViewMatrix(mirrors[i].vm);
			mirrors[i].camera->setProjectionMatrix(mirrors[i].pm);
			// this is a camera based mirror, get the view and projection matrix from the appropriate rear view camera.
			// todo fix environment map orientation in mirror view
			/*
			osg::Matrix proj = mirrors[i].camera->getProjectionMatrix();
			osg::Matrix mv = mirrors[i].camera->getViewMatrix();
			osg::Matrix rotonly = mv;
			rotonly(3,0)=0;
			rotonly(3,1)=0;
			rotonly(3,2)=0;
			rotonly(3,3)=1;
			osg::Matrix invRot;

			invRot.invert(rotonly);
			nmv=((mv * invRot) * cover->invEnvCorrectMat);
			npm=(cover->envCorrectMat *rotonly * proj) ;
			mirrors[i].camera->setViewMatrix(mirrors[i].vm);
			mirrors[i].camera->setProjectionMatrix(mirrors[i].pm);
			*/
			if (mirrors[i].shader)
			{
				osg::Matrixf nmvf = nmv;
				osg::Matrixf npmf = npm;
				osg::Matrixf geoToWCf = geoToWC;
				mirrors[i].shader->setMatrixUniform("ViewMatrix", nmvf);
				mirrors[i].shader->setMatrixUniform("ProjectionMatrix", npmf);
				mirrors[i].shader->setMatrixUniform("ModelMatrix", geoToWCf);
				if (mirrors[i].instance)
				{
					osg::Uniform *U;
					U = mirrors[i].instance->getUniform("ViewMatrix");
					if (U)
					{
						U->set(nmv);
					}
					U = mirrors[i].instance->getUniform("ProjectionMatrix");
					if (U)
					{
						U->set(npm);
					}
					U = mirrors[i].instance->getUniform("ModelMatrix");
					if (U)
					{
						U->set(geoToWC);
					}
				}
				/*osg::Vec3 tmpv(1,1,1);
				osg::Vec3 tmpv2;
				tmpv2 = tmpv * ProjInMirrorCS ;
				fprintf(stderr,"%f %f %f\n",tmpv2[0],tmpv2[1],tmpv2[2]);*/
			}
		}
	}
}

void ViewerOsg::redraw()
{
    if (cover->debugLevel(5))
        cerr << "ViewerOsg::redraw" << endl;
    //double start = System::the->time();

    vrmlBaseMat = cover->getBaseMat();
    Matrix transformMat = VRMLRoot->getMatrix();
    vrmlBaseMat.preMult(transformMat);

    //cerr << "ViewerOsg::redraw" << endl;
    d_scene->render(this);

    //pfuTravPrintNodes(scene,"testit");

    /* static float oldRadius=0.0;
   // Determine extent of scene's geometry.
   pfSphere bsphere;
   VRMLRoot->getBound(&bsphere);
   if(bsphere.radius+bsphere.center.length()>oldRadius)
   oldRadius = bsphere.radius+bsphere.center.length();
   chan->setNearFar(1.0f, 10.0f*oldRadius);

   // Spin text for 15 seconds.
   static double oldt=0.0;
   double t;
   t = pfGetTime()- oldt;
   if(t > 15)
   {
   oldt = pfGetTime();
   t = 0.0;
   pfExit();
   }
   pfCoord	   view;
   float      s, c;

   // Compute new view position, rotating around text.
   t=1.0;
   pfSinCos(45.0f*t, &s, &c);
   view.hpr.set(45.0f*t, -5.0f, 0.0f);
   view.xyz.set(
   2.0f * oldRadius * s,
   -2.0f * oldRadius * c,
   0.3f * oldRadius);
   chan->setView(view.xyz, view.hpr);

   // Initiate cull/draw processing for this frame.
   pfFrame();

   d_renderTime1 = d_renderTime;
   d_renderTime = System::the->time() - start;*/

    // SpinMenu (Viewpoints)

    /*  if(noViewpoints==0)
       {
       if(numVP<0 || numVP!=d_scene->nViewpoints())
       {
       delete viewpoints;
       numVP=d_scene->nViewpoints();
       viewpoints = NULL;
       if(numVP>0)
       {
       int i;
       viewpoints = new SpinMenu(cover->getScene(),numVP,SM_SNAP_ON,1,0);
       for(i=0;i<numVP;i++)
       {
       const char *name,*desc;
       d_scene->getViewpoint(i,&name,&desc);
       SpinMenuItem *item = new SpinMenuItem(name);
       item->setUserData(desc);
       viewpoints->addMenuItem(item);
       }
       }
       }
       }
       if(viewpoints)
       {
       int state=0;
       int states=0;
       cover->getBuiltInFunctionState("XForm",&state);
       states = states & state;
       cover->getBuiltInFunctionState("Drive",&state);
       states = states & state;
       cover->getBuiltInFunctionState("Scale",&state);
       states = states & state;
       cover->getBuiltInFunctionState("Walk",&state);
       states = states & state;
       cover->getBuiltInFunctionState("Fly",&state);
       states = states & state;
       if(noViewpoints==0)
       {
       switch(viewpoints->update())
       {
       case SM_ACTIVE:
       break;

       case SM_SELECTION:
       d_scene->setViewpoint(viewpoints->getSelectionIndex());
       break;

       case SM_NOACTION:
       break;

       case SM_CANCEL:
       viewpoints->hide();
       return;

       default:
       break;
       }
       }
       }*/
    if (cover->debugLevel(5))
        cerr << "END ViewerOsg::redraw" << endl;
}

void ViewerOsg::getTransform(double *M)
{
    matToVrml(M, currentTransform);
}

void ViewerOsg::matToPf(Matrix *mat, const double *M)
{
    for (int i = 0; i < 16; i++)
    {
        mat->ptr()[i] = M[i];
    }
}

void ViewerOsg::matToVrml(double *M, const Matrix &mat)
{
    for (int i = 0; i < 16; i++)
    {
        M[i] = mat.ptr()[i];
    }
}

int ViewerOsg::getBlendModeForVrmlNode(const char *modeString)
{
    return osgViewerObject::getBlendModeForVrmlNode(modeString);
}
