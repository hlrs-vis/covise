/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_SPHERE_H
#define CO_SPHERE_H

/*! \file
 \brief osg sphere node

 \author Thomas van Reimersdahl <reimersdahl@uni-koeln.de>
 \author Sebastian Breuers <breuerss@uni-koeln.de>
 \author (C) 2005
         ZAIK Center for Applied Informatics,
         Robert-Koch-Str. 10, Geb. 52,
         D-50931 Koeln,
         Germany

 \date 
 */

#include <util/common.h>
#include <osg/Drawable>
#include <osg/Geometry>
#include <OpenThreads/Mutex>
#include <vector>
#include <osg/Version>

typedef struct _CGprogram *CGprogram;
typedef struct _CGparameter *CGparameter;

namespace opencover
{
class PLUGIN_UTILEXPORT coSphere : public osg::Drawable
{
public:
    // keep in sync with Sphere module
    enum RenderMethod
    {
        RENDER_METHOD_CPU_BILLBOARDS = 0,
        RENDER_METHOD_CG_SHADER = 1,
        RENDER_METHOD_ARB_POINT_SPRITES = 2,
        RENDER_METHOD_PARTICLE_CLOUD = 4,
        RENDER_METHOD_DISC = 5,
        RENDER_METHOD_TEXTURE = 6,
        RENDER_METHOD_CG_SHADER_INVERTED = 7
    };

    coSphere();

    void setRenderMethod(RenderMethod rm);
    void setNumberOfSpheres(int n);
    void updateCoords(const float *x_c, const float *y_c, const float *z_c);
    void updateCoordsFromMatrices(float *const *matrices);
    void setCoords(int no_of_points, const float *x, const float *y, const float *z, const float *r);
    void setCoords(int no_of_points, const float *x, const float *y, const float *z, float r=1.f);
    void updateRadii(const float *r);
    void updateRadii(const double *r);
    void updateNormals(const float *nx, const float *ny, const float *nz);
    void setColorBinding(int colorbinding);
    void setColor(int index, float r, float g, float b, float a);
    void updateColors(const float *r, const float *g, const float *b, const float *a = NULL);
    void updateColors(const int *pc);
    static void setScale(float scale);
    static void setTransparency(float alpha);
    static void enableTransparencyOverride(bool);
    void overrideBoundingBox(const osg::BoundingBox &bb);

    virtual void drawImplementation(osg::RenderInfo &renderInfo) const;
    void setMaxRadius(float m)
    {
        m_extMaxRadius = m;
    };
    void setVertexAttribArray(unsigned int index, const osg::Array *array, osg::Array::Binding binding = osg::Array::BIND_UNDEFINED);

protected:
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
    virtual osg::BoundingBox computeBoundingBox() const;
#else
    virtual osg::BoundingBox computeBound() const;
#endif
    //virtual osg::BoundingBox getBoundingBox() const;

private:
    virtual ~coSphere();

    virtual osg::Object *cloneType() const
    {
        return new coSphere();
    }
    virtual osg::Object *clone(const osg::CopyOp &copyop) const
    {
        return new coSphere(*this, copyop);
    }

    coSphere(const coSphere &, const osg::CopyOp &copyop = osg::CopyOp::SHALLOW_COPY);

    struct BMPImage
    {
        int width;
        int height;
        unsigned char *data;
    };
    static void setTexture(const char *chTexFile);

    static void getBitmapImageData(const char *pFileName, BMPImage *pImage);
    static bool loadTexture(int context, const char *pFilename, int iTextureMode = 0, int colorR = 0, int colorG = 0, int colorB = 0);

    float m_defaultColor[4];
    int m_colorBinding;
    static bool s_overrideTransparency;
    float *m_coord;
    float *m_color;
    float *m_normals;
    static float s_alpha;
    static float s_scale;
    int m_numSpheres;
    bool m_useVertexArrays;
    std::vector<int> *m_sortedRadiusIndices;
    int m_maxPointSize;

    float m_maxRadius;
    float m_extMaxRadius;
    float *m_radii;
    mutable RenderMethod m_renderMethod;
    bool m_overrideBounds;

    static bool loadCgPrograms(int context);
    bool unbindProgramAndParams(int context) const;
    void bindMatrices(int context) const;
    bool bindProgramAndParams(int context) const;
    static void initTexture(int context);

    static bool *s_CgChecked;
    static bool s_pointSpritesChecked;
    static bool s_configured; // prevent using coConfig before main
    static bool s_useVertexArrays;

    static CGprogram *s_VertexProgram;
    static CGprogram *s_FragmentProgram;
    static CGprogram *s_FragmentProgramParticleCloud;
    static CGprogram *s_VertexProgramDisc;
    static CGprogram *s_FragmentProgramDisc;
    static CGprogram *s_FragmentProgramTexture;
    static CGprogram *s_FragmentProgramInverted;

    static CGparameter *s_CGVertexParam_modelViewProj;
    static CGparameter *s_CGVertexParam_modelView;
    static CGparameter *s_CGVertexParam_modelViewIT;
    static CGparameter *s_CGVertexParam_lightPos;
    static CGparameter *s_CGVertexParam_viewerPos;
    static CGparameter *s_CGFragParam_glAmbient;
    static CGparameter *s_CGFragParam_glDiffuse;
    static CGparameter *s_CGFragParam_glSpecular;

    static CGparameter *s_CGFragParamInverted_glAmbient;
    static CGparameter *s_CGFragParamInverted_glDiffuse;
    static CGparameter *s_CGFragParamInverted_glSpecular;

    static CGparameter *s_CGVertexParamDisc_modelViewProj;
    static CGparameter *s_CGVertexParamDisc_modelView;
    static CGparameter *s_CGVertexParamDisc_modelViewIT;
    static CGparameter *s_CGVertexParamDisc_lightPos;
    static CGparameter *s_CGVertexParamDisc_viewerPos;
    static CGparameter *s_CGFragParamDisc_glAmbient;
    static CGparameter *s_CGFragParamDisc_glDiffuse;
    static CGparameter *s_CGFragParamDisc_glSpecular;
    static CGparameter *s_CGFragParamTexture_glAmbient;
    static CGparameter *s_CGFragParamTexture_glDiffuse;
    static CGparameter *s_CGFragParamTexture_glSpecular;
    static GLuint *s_textureID;
    static char *s_chTexFile;
    static void(APIENTRY *s_glPointParameterfARB)(GLenum, GLfloat);
    static void(APIENTRY *s_glPointParameterfvARB)(GLenum, const GLfloat *);

    static OpenThreads::Mutex *mutex()
    {
        static OpenThreads::Mutex mutex;
        return &mutex;
    }

    static int s_maxcontext;
};
}
#endif
