/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  ViewerOsg.h
//  Class for display of VRML models using OpenSceneGraph.
//

#ifndef _VIEWEROSG_H_
#define _VIEWEROSG_H_

#include "ViewerObject.h"
#include <osg/Version>
#include <osg/Camera>

#include <osg/GL>
#include <osg/Node>
#include <osg/Matrix>
#include <osg/Material>
#include <osg/Billboard>
#include <osg/Texture>
#include <osg/ImageStream>
#include <osgText/Font>

#include <util/common.h>
#include <vrml97/vrml/config.h>
#include <vrml97/vrml/Viewer.h>
#include <vrml97/vrml/VrmlNode.h>

#include <util/DLinkList.h>
#include "coSensiveSensor.h"

#ifdef HAVE_OSGNV
#include <osgNVExt/RegisterCombiners>
#include <osgNVExt/CombinerInput>
#include <osgNVExt/CombinerOutput>
#include <osgNVExt/FinalCombinerInput>
#endif
#include <osg/ref_ptr>

#define MAX_MIRRORS 32
namespace opencover
{
class coVRShader;
class coVRShaderInstance;
}

struct movieImageData
{
    movieProperties *movieProp;
    osg::ref_ptr<osg::ImageStream> imageStream;
};

class VRML97COVEREXPORT MirrorInfo
{
public:
    MirrorInfo();
    ~MirrorInfo();
    osg::Vec3 coords[4];
    osg::ref_ptr<osg::Camera> camera;
    osg::ref_ptr<osg::Group> statesetGroup;
    osg::Geode *geometry;
    coVRShader *shader;
    coVRShaderInstance *instance;
    bool isVisible;
    osg::Matrix vm;
    osg::Matrix pm;
    int CameraID;
};

namespace osg
{
class TexEnv;
class Geometry;
class Texture;
class Texture2D;
class StateSet;
class Depth;
class ColorMask;
};
namespace osgText
{
class Font;
};
namespace osgNVExt
{
class RegisterCombiners;
};

extern int Sorted;
extern int textureMode;
extern osg::ref_ptr<osg::TexEnv> tEnvAdd;
extern osg::ref_ptr<osg::TexEnv> tEnvBlend;
extern osg::ref_ptr<osg::TexEnv> tEnvDecal;
extern osg::ref_ptr<osg::TexEnv> tEnvReplace;
extern osg::ref_ptr<osg::TexEnv> tEnvModulate;

class coCubeMap;
class osgViewerObject;

class VRML97COVEREXPORT ViewerOsg : public Viewer
{
    friend class osgViewerObject;

private:
    osg::Vec3 viewerPos;
    osg::Vec3 viewerInMirrorCS;

public:
    osg::ref_ptr<osgText::Font> font;
    int numVP;
    //SpinMenu *viewpoints;
    static ViewerOsg *viewer;
    enum
    {
        MAX_LIGHTS = 8
    };
    osgViewerObject *d_root;
    osgViewerObject *d_currentObject;
    std::vector<coSensiveSensor *> sensors; // hold all sensors for later access

    //osg::ColorMask *NoFrameBuffer;
    osg::Depth *NoDepthBuffer;
    void setRootNode(osg::Group *group);
    osg::MatrixTransform *VRMLRoot;
    static osg::MatrixTransform *VRMLCaveRoot;
    double startLoadTime;
    osg::Matrix vrmlBaseMat;
    osg::Matrix currentTransform;
    osg::ref_ptr<osg::StateSet> BumpCubeState;
    osg::ref_ptr<osg::Program> bumpCubeProgram;
    osg::ref_ptr<osg::Program> bumpEnvProgram;
    osg::ref_ptr<osg::StateSet> BumpEnvState;
    osg::ref_ptr<osg::Program> bumpProgram;
    osg::ref_ptr<osg::Program> glassProgram;
    osg::ref_ptr<osg::Program> testProgram;
    std::list<movieImageData *> moviePs;
    osg::ref_ptr<osg::StateSet> combineTexturesState;
    osg::ref_ptr<osg::StateSet> combineEnvTexturesState;
    osg::ref_ptr<osg::Program> combineTextures;
    osg::ref_ptr<osg::Program> combineEnvTextures;
    int numCameras;
    MirrorInfo mirrors[MAX_MIRRORS];

public:
    ViewerOsg(VrmlScene *, osg::Group *rootNode);
    virtual ~ViewerOsg();

    void setModesByName(const char *objectName = NULL);
    void setDefaultMaterial(osg::StateSet *geoState);

    bool addToScene(osgViewerObject *obj);
    void restart();

    virtual void getViewerMat(double *M);
    virtual void getVrmlBaseMat(double *M);
    virtual void getCurrentTransform(double *M);

    virtual int getRenderMode();
    virtual double getFrameRate();

    // Scope dirlights, open/close display lists
    virtual Object beginObject(const char *name, bool retain, VrmlNode *node);
    virtual void endObject();

    // Insert objects into the display list
    virtual Object insertBumpMapping();

    virtual Object insertWave(
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
        const char *fileName);
    virtual Object insertBackground(int nGroundAngles = 0,
                                    float *groundAngle = 0,
                                    float *groundColor = 0,
                                    int nSkyAngles = 0,
                                    float *skyAngle = 0,
                                    float *skyColor = 0,
                                    int *whc = 0,
                                    unsigned char **pixels = 0);

    Object insertNode(osg::Node *node);

    virtual Object insertLineSet(int, float *, int, int *,
                                 bool colorPerVertex,
                                 float *color, int componentsPerColor,
                                 int nColorIndex,
                                 int *colorIndex, const char *name);

    virtual Object insertPointSet(int npts, float *points, float *colors,
                                  int componentsPerColor);

    void applyTextureCoordinateGenerator(int textureUnit, int mode,
                                         float *parameters, int numParameters);

    virtual Object insertShell(unsigned int mask,
                               int npoints, float *points,
                               int nfaces, int *faces,
                               float **tc,
                               int *ntci, int **tci,
                               float *normal,
                               int nni, int *ni,
                               float *color,
                               int nci, int *ci,
                               const char *objName,
                               int *texCoordGeneratorName = NULL,
                               float **texCoordGeneratorParameter = NULL,
                               int *numTexCoordGeneratorParameter = NULL);

    virtual Object insertText(int *, float, int n, char **s, const char *objName);

    // Lights
    virtual Object insertDirLight(float a, float i, float rgb[], float xyz[]);

    virtual Object insertPointLight(float ambientIntensity,
                                    float attenuation[],
                                    float color[],
                                    float intensity,
                                    float location[],
                                    float radius);

    virtual Object insertSpotLight(float ambientIntensity,
                                   float attenuation[],
                                   float beamWidth,
                                   float color[],
                                   float cutOffAngle,
                                   float direction[],
                                   float intensity,
                                   float location[],
                                   float radius);

    // Lightweight copy
    virtual Object insertReference(Object existingObject);

    // Remove an object from the display list
    virtual void removeObject(Object key);

    virtual void removeChild(Object obj);

    virtual void enableLighting(bool);

    // Set attributes
    virtual void setColor(float r, float g, float b, float a = 1.0);

    virtual void setFog(float *color,
                        float visibilityRange,
                        const char *fogType);

    virtual void setMaterial(float ambientIntensity,
                             float diffuseColor[],
                             float emissiveColor[],
                             float shininess,
                             float specularColor[],
                             float transparency)
    {
        d_currentObject->setMaterial(ambientIntensity,
                                     diffuseColor, emissiveColor, shininess, specularColor, transparency);
    };

    virtual void setNameModes(const char *modes, const char *relURL = NULL);

    virtual void setMaterialMode(int nTexComponents, bool geometryColor);

    virtual void setSensitive(void *object);

    virtual void setCollision(bool collide);

    virtual TextureObject insertTexture(int w, int h, int nc,
                                        bool repeat_s,
                                        bool repeat_t,
                                        unsigned char *pixels,
                                        bool retainHint = false,
                                        bool environment = false, int blendMode = 0, int anisotropy = 1, int filter = 0);

    virtual TextureObject insertMovieTexture(char *filename,
                                             movieProperties *movProp, int nc,
                                             bool retainHint = false,
                                             bool environment = false, int blendMode = 0, int anisotropy = 1, int filter = 0);

    virtual void insertMovieReference(TextureObject, int nc, bool environment, int blendMode);

    virtual TextureObject insertCubeTexture(int /*w*/, int /*h*/, int /*nc*/,
                                            bool /*repeat_s*/,
                                            bool /*repeat_t*/,
                                            unsigned char * /*pixels*/,
                                            unsigned char * /*pixels*/,
                                            unsigned char * /*pixels*/,
                                            unsigned char * /*pixels*/,
                                            unsigned char * /*pixels*/,
                                            unsigned char * /*pixels*/,
                                            bool retainHint = false, int blendMode = 0);

    // Reference/remove a texture object
    virtual void insertTextureReference(TextureObject, int, bool environment = false, int blendMode = 0);
    virtual void insertCubeTextureReference(TextureObject, int, int blendMode = 0);
    virtual void removeTextureObject(TextureObject);
    virtual void removeCubeTextureObject(TextureObject);
    virtual void removeMovieTexture();

    virtual void setTextureTransform(float * /*center*/,
                                     float /*rotation*/,
                                     float * /*scale*/,
                                     float * /*translation*/);

    virtual void setTransform(float * /*center*/,
                              float * /*rotation*/,
                              float * /*scale*/,
                              float * /*scaleOrientation*/,
                              float * /*translation*/,
                              bool changed);

    virtual void setClip(float *pos,
                         float *ori,
                         int number,
                         bool enabled);

    virtual void unsetTransform(float * /*center*/,
                                float * /*rotation*/,
                                float * /*scale*/,
                                float * /*scaleOrientation*/,
                                float * /*translation*/);

    virtual void getTransform(double *M);

    virtual void setBillboardTransform(float * /*axisOfRotation*/);
    virtual void unsetBillboardTransform(float * /*axisOfRotation*/);

    virtual void setViewpoint(float *position,
                              float *orientation,
                              float fieldOfView,
                              float avatarSize,
                              float visLimit,
                              const char *type,
                              float scaleFactor);

    // The viewer knows the current viewpoint
    virtual void transformPoints(int nPoints, float *points);

    virtual void setChoice(int which);

    //
    // Viewer callbacks (not for public consumption)

    // Update the model.
    void update(double time = 0.0);

    // Redraw the screen.
    virtual void redraw();

    static void matToPf(osg::Matrix *mat, const double *M);
    static void matToVrml(double *M, const osg::Matrix &mat);

    osg::ref_ptr<osg::Texture2D> framebufferTexture;
    bool framebufferTextureUpdated;

    virtual int getBlendModeForVrmlNode(const char *modeString);
    void removeSensor(coSensiveSensor *s);

private:
    osg::ref_ptr<osg::Material> globalmtl;

    // Initialize OpenGL state
    void initialize();

    // Geometry insertion setup & cleanup methods
    void beginGeometry();
    void endGeometry();

    bool d_selectMode;

    double d_renderTime, d_renderTime1;
    void addObj(osgViewerObject *obj, osg::Group *group);

    std::string localizeString(const std::string &stringToLocalize) const;
};
#endif
