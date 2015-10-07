/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  Viewer.h
//  Abstract base class for display of VRML models
//

#ifndef _VIEWER_
#define _VIEWER_

#include "Player.h"

namespace vrml
{

typedef struct
{
    bool loop;
    float speed;
    double start;
    double stop;
    bool repeatS;
    bool repeatT;
} movieProperties;

class VRMLEXPORT VrmlNode;
class VRMLEXPORT VrmlScene;

class VRMLEXPORT Viewer
{

protected:
    // Explicitly instantiate a subclass object
    Viewer(VrmlScene *scene);

public:
    virtual ~Viewer();

    // Options flags
    enum
    {
        MASK_NONE = 0,
        MASK_CCW = 1,
        MASK_CONVEX = 2,
        MASK_SOLID = 4,
        MASK_BOTTOM = 8,
        MASK_TOP = 16,
        MASK_SIDE = 32,
        MASK_COLOR_PER_VERTEX = 64,
        MASK_NORMAL_PER_VERTEX = 128,
        MASK_COLOR_RGBA = 256
    };

    enum
    {
        TEXTURE_COORDINATE_GENERATOR_MODE_SPHERE = 1,
        TEXTURE_COORDINATE_GENERATOR_MODE_CAMERASPACEREFLECTIONVECTOR = 2,
        TEXTURE_COORDINATE_GENERATOR_MODE_COORD = 3,
        TEXTURE_COORDINATE_GENERATOR_MODE_CAMERASPACENORMAL = 4,
        TEXTURE_COORDINATE_GENERATOR_MODE_COORD_EYE = 5
    };

    enum
    {
        NUM_TEXUNITS = 4
    };

    // number of active textures
    int numTextures;

    // current texture to work on
    int textureNumber;
    double startLoadTime;

    // Object and texture keys. Don't mix them up.
    // is used to store pointers to nodes in OpenCOVER thus needs to pe ptr_size
    typedef int64_t Object;
    typedef int64_t TextureObject;

    //
    VrmlScene *scene()
    {
        return d_scene;
    }

    // Query
    virtual void getViewerMat(double *M) = 0;
    virtual void getCurrentTransform(double *M) = 0;
    virtual void getVrmlBaseMat(double *M) = 0;

    virtual void getPosition(float *x, float *y, float *z);
    virtual void getOrientation(float *orientation);
    virtual void getPositionWC(float *x, float *y, float *z);
    virtual void getWC(float px, float py, float pz, float *x, float *y, float *z);

    enum
    {
        RENDER_MODE_DRAW,
        RENDER_MODE_PICK
    };

    // Return renderMode
    virtual int getRenderMode() = 0;

    virtual double getFrameRate() = 0;

    virtual void resetUserNavigation()
    { /*std::cerr << "resetUserNavigation not implemented!" << std::endl;*/
    }

    // Open/close display lists
    virtual Object beginObject(const char *, bool, VrmlNode *node) = 0;
    virtual void endObject() = 0;

    // Insert objects into the display list
    virtual Object insertBumpMapping() = 0;

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
        const char *fileName) = 0;
    virtual Object insertBackground(int nGroundAngles = 0,
                                    float *groundAngle = NULL,
                                    float *groundColor = NULL,
                                    int nSkyAngles = 0,
                                    float *skyAngle = NULL,
                                    float *skyColor = NULL,
                                    int *whc = NULL,
                                    unsigned char **pixels = NULL) = 0;

    virtual Object insertBox(float, float, float);
    virtual Object insertCone(float, float, bool, bool);
    virtual Object insertCylinder(float h, float r, bool bottom, bool side, bool top);

    virtual Object insertElevationGrid(unsigned int mask,
                                       int nx,
                                       int nz,
                                       float *height,
                                       float dx,
                                       float dz,
                                       float *tc,
                                       float *normals,
                                       float *colors,
                                       float creaseAngle);

    virtual Object insertExtrusion(unsigned int mask,
                                   int nOrientation,
                                   float *orientation,
                                   int nScale,
                                   float *scale,
                                   int nCrossSection,
                                   float *crossSection,
                                   int nSpine,
                                   float *spine,
                                   float creaseAngle);

    virtual Object insertLineSet(int nCoords, float *coord,
                                 int nCoordIndex, int *coordIndex,
                                 bool colorPerVertex,
                                 float *color, int componentsPerColor,
                                 int nColorIndex,
                                 int *colorIndex, const char *name) = 0;

    virtual Object insertPointSet(int nv, float *v,
                                  float *c, int componentsPerColor) = 0;

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
                               float **texCoordGeneratorParameters = NULL,
                               int *numTexCoordGeneratorParameters = NULL) = 0;

    static void computeNormals(const float *coords,
                               int numCoordInd, const int *coordInd,
                               float *normal, int *nind,
                               float creaseAngle, bool ccw);

    virtual Object insertSphere(float radius);

    virtual Object insertText(int *, float size, int n, char **s, const char *objName) = 0;

    // Lights
    virtual Object insertDirLight(float, float, float[], float[]) = 0;

    virtual Object insertPointLight(float, float[], float[],
                                    float, float[], float) = 0;

    virtual Object insertSpotLight(float ambientIntensity,
                                   float attenuation[],
                                   float beamWidth,
                                   float color[],
                                   float cutOffAngle,
                                   float direction[],
                                   float intensity,
                                   float location[],
                                   float radius) = 0;

    // Lightweight copy
    virtual Object insertReference(Object existingObject) = 0;

    // Remove an object from the display list
    virtual void removeObject(Object) = 0;

    virtual void removeChild(Object){};

    virtual void enableLighting(bool) = 0;

    // Set attributes
    virtual void setFog(float *color,
                        float visibilityRange,
                        const char *fogType) = 0;

    virtual void setColor(float r, float g, float b, float a = 1.0) = 0;

    virtual void setMaterial(float, float[], float[], float, float[], float) = 0;

    virtual void setNameModes(const char *modes, const char *relURL = NULL);

    virtual void setMaterialMode(int nTexComponents, bool geometryColor) = 0;

    virtual void setSensitive(void *object) = 0;

    virtual void setCollision(bool){};

    virtual void scaleTexture(int w, int h,
                              int newW, int newH,
                              int nc,
                              unsigned char *pixels);

    // Create a texture object
    virtual TextureObject insertTexture(int w, int h, int nc,
                                        bool repeat_s,
                                        bool repeat_t,
                                        unsigned char *pixels,
                                        const char* filename = "",
                                        bool retainHint = false,
                                        bool environment = false, int blendMode = 0, int anisotropy = 1, int filter = 0) = 0;
    virtual TextureObject insertMovieTexture(char *filename, movieProperties *movProp, int nc,
                                             bool retainHint = false,
                                             bool environment = false, int blendMode = 0, int anisotropy = 1, int filter = 0) = 0;
    virtual TextureObject insertCubeTexture(int w, int h, int nc,
                                            bool repeat_s,
                                            bool repeat_t,
                                            unsigned char *pixelsXP,
                                            unsigned char *pixelsXN,
                                            unsigned char *pixelsYP,
                                            unsigned char *pixelsYN,
                                            unsigned char *pixelsZP,
                                            unsigned char *pixelsZN,
                                            bool retainHint = false, int blendMode = 0) = 0;

    // Reference/remove a texture object
    virtual void insertMovieReference(TextureObject, int, bool environment, int blendMode) = 0;
    virtual void insertTextureReference(TextureObject, int, bool environment = false, int blendMode = 0) = 0;
    virtual void insertCubeTextureReference(TextureObject, int, int blendMode = 0) = 0;
    virtual void removeTextureObject(TextureObject) = 0;
    virtual void removeCubeTextureObject(TextureObject) = 0;

    virtual void setTextureTransform(float *center,
                                     float rotation,
                                     float *scale,
                                     float *translation) = 0;

    virtual void setTransform(float *center,
                              float *rotation,
                              float *scale,
                              float *scaleOrientation,
                              float *translation,
                              bool changed) = 0;

    virtual void setClip(float *pos,
                         float *ori,
                         int number,
                         bool enabled) = 0;

    virtual void setShadow(int number,
                         bool enabled) = 0;

    // This is a hack to work around the glPushMatrix() limit (32 deep on Mesa).
    // It has some ugly disadvantages: it is slower and the resulting transform
    // after a setTransform/unsetTransform may not be identical to the original.
    // It might be better to just build our own matrix stack...
    virtual void unsetTransform(float * /*center*/,
                                float * /*rotation*/,
                                float * /*scale*/,
                                float * /*scaleOrientation*/,
                                float * /*translation*/) = 0;

    virtual void getTransform(double *M) = 0;

    virtual void setBillboardTransform(float * /*axisOfRotation*/) = 0;

    virtual void unsetBillboardTransform(float * /*axisOfRotation*/) = 0;

    virtual void setViewpoint(float * /*position*/,
                              float * /*orientation*/,
                              float /*fieldOfView*/,
                              float /*avatarSize*/,
                              float /*visLimit*/,
                              const char * /*type*/,
                              float /*scale*/) = 0;

    // The viewer knows the current viewpoint
    virtual void transformPoints(int nPoints, float *points) = 0;

    // specify which child of a switch node to draw
    virtual void setChoice(int which) = 0;

    // retrieve the Player
    virtual Player *getPlayer()
    {
        return d_player;
    }

    // set Player
    virtual void setPlayer(Player *player)
    {
        d_player = player;
    }

    // set number of active textures
    virtual void setNumTextures(int numTex);

    virtual int getBlendModeForVrmlNode(const char *)
    {
        return 0;
    }

protected:
    VrmlScene *d_scene;

    Player *d_player;

private:
    Viewer(); // Don't allow default constructors
};
}
#endif // _VIEWER_
