/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/CoviseConfig.h>
#include <util/common.h>
#include <util/byteswap.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/coIntersection.h>
#include "coSphere.h"
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osgUtil/LineSegmentIntersector>
#include <osg/io_utils>

#include <osg/GLExtensions>

#include <cover/RenderObject.h>

#ifdef HAVE_CG
#include <Cg/cg.h>
#include <Cg/cgGL.h>
#endif

using namespace covise;

#ifdef WIN32
template <class T>
inline const T &fmax(const T &a, const T &b)
{
    return a < b ? b : a;
}
#endif

namespace opencover
{

#ifndef _USE_TEXTURE_LOADING_MS
typedef struct _AUX_RGBImageRec
{
    GLint sizeX, sizeY;
    unsigned char *data;
} AUX_RGBImageRec;
#else
#include <GL/glaux.h>
#endif

//constants taken from gl extensions registry (file : "glext.h")
#ifndef GL_POINT_FADE_THRESHOLD_SIZE_ARB
#define GL_POINT_FADE_THRESHOLD_SIZE_ARB 0x8128
#endif

#ifndef GL_POINT_DISTANCE_ATTENUATION_ARB
#define GL_POINT_DISTANCE_ATTENUATION_ARB 0x8129
#endif

#ifndef GL_POINT_SPRITE_ARB
#define GL_POINT_SPRITE_ARB 0x8861
#endif

#ifndef GL_COORD_REPLACE_ARB
#define GL_COORD_REPLACE_ARB 0x8862
#endif

#ifndef GL_SMOOTH_POINT_SIZE_RANGE
#define GL_SMOOTH_POINT_SIZE_RANGE 0x0B12
#endif

/*initialisation of static member variables*/
bool coSphere::s_overrideTransparency = false;
float coSphere::s_alpha = 1.0;
float coSphere::s_scale = 1.0;

#ifdef HAVE_CG
static CGprofile *s_VertexProfile = NULL; //CG_PROFILE_UNKNOWN;
static CGprofile *s_FragmentProfile = NULL; //CG_PROFILE_UNKNOWN;
#endif

CGprogram *coSphere::s_VertexProgram = NULL;
CGprogram *coSphere::s_VertexProgramDisc = NULL;
CGprogram *coSphere::s_FragmentProgram = NULL;
CGprogram *coSphere::s_FragmentProgramParticleCloud = NULL;
CGprogram *coSphere::s_FragmentProgramDisc = NULL;
CGprogram *coSphere::s_FragmentProgramTexture = NULL;
CGprogram *coSphere::s_FragmentProgramInverted = NULL;

CGparameter *coSphere::s_CGVertexParam_modelViewProj = NULL;
CGparameter *coSphere::s_CGVertexParam_modelView = NULL;
CGparameter *coSphere::s_CGVertexParam_modelViewIT = NULL;
CGparameter *coSphere::s_CGVertexParam_lightPos = NULL;
CGparameter *coSphere::s_CGVertexParam_viewerPos = NULL;
CGparameter *coSphere::s_CGFragParam_glAmbient = NULL;
CGparameter *coSphere::s_CGFragParam_glDiffuse = NULL;
CGparameter *coSphere::s_CGFragParam_glSpecular = NULL;
CGparameter *coSphere::s_CGFragParamInverted_glAmbient = NULL;
CGparameter *coSphere::s_CGFragParamInverted_glDiffuse = NULL;
CGparameter *coSphere::s_CGFragParamInverted_glSpecular = NULL;

CGparameter *coSphere::s_CGVertexParamDisc_modelViewProj = NULL;
CGparameter *coSphere::s_CGVertexParamDisc_modelView = NULL;
CGparameter *coSphere::s_CGVertexParamDisc_modelViewIT = NULL;
CGparameter *coSphere::s_CGVertexParamDisc_lightPos = NULL;
CGparameter *coSphere::s_CGVertexParamDisc_viewerPos = NULL;
CGparameter *coSphere::s_CGFragParamDisc_glAmbient = NULL;
CGparameter *coSphere::s_CGFragParamDisc_glDiffuse = NULL;
CGparameter *coSphere::s_CGFragParamDisc_glSpecular = NULL;
CGparameter *coSphere::s_CGFragParamTexture_glAmbient = NULL;
CGparameter *coSphere::s_CGFragParamTexture_glDiffuse = NULL;
CGparameter *coSphere::s_CGFragParamTexture_glSpecular = NULL;

bool *coSphere::s_CgChecked = NULL;
bool coSphere::s_pointSpritesChecked = false;
int coSphere::s_maxcontext = -1;

void(APIENTRY *coSphere::s_glPointParameterfARB)(GLenum, GLfloat) = NULL;
void(APIENTRY *coSphere::s_glPointParameterfvARB)(GLenum, const GLfloat *) = NULL;

GLuint *coSphere::s_textureID = NULL;
char *coSphere::s_chTexFile = 0;

// this is hacky, problems:
// - will crash when changing render mode after having set the data
// - won't work on multi-head systems where one head has, the other doesn't have Cg support

bool coSphere::s_configured = false;
bool coSphere::s_useVertexArrays = false;

class SphereIntersector: public opencover::IntersectionHandler
{
public:
    bool canHandleDrawable(osg::Drawable *drawable) const
    {
        auto s = dynamic_cast<coSphere *>(drawable);
        if (s)
            return true;
        return false;
    }

    void intersect(osgUtil::IntersectionVisitor &iv, coIntersector &is, osg::Drawable *drawable)
    {
        osg::Vec3d s(is.getStart()), e(is.getEnd());
#if (OSG_VERSION_GREATER_OR_EQUAL(3, 4, 0))
        osg::BoundingBox bb = drawable->getBoundingBox();
        if (!is.intersectAndClip(s, e, bb)) return;
#endif
        if (iv.getDoDummyTraversal()) return;

        //std::cerr << "SPHERE isect start=" << s << ", end=" << e << std::endl;

        coSphere *sphere = dynamic_cast<coSphere *>(drawable);
        assert(sphere);

        const osg::Vec3d se = e-s;
        double a = se.length2();

        for (int i=0; i<sphere->m_numSpheres; ++i)
        {
            osg::Vec3 center(sphere->m_coord[i*3], sphere->m_coord[i*3+1], sphere->m_coord[i*3+2]);
            float radius = sphere->m_radii[i];

            osg::Vec3 sm = s - center;
            double c = sm.length2()-radius*radius;
            if (c<0.0)
            {
                // inside sphere
#if 0
                osgUtil::LineSegmentIntersector::Intersection hit;
                hit.ratio = 0;
                hit.nodePath = iv.getNodePath();
                hit.drawable = drawable;
                hit.matrix = iv.getModelMatrix();
                hit.localIntersectionPoint = s;
                is.insertIntersection(hit);
#endif
                continue;
            }

            double b = (sm*se)*2.0;
            double disc = b*b-4.0*a*c;

            if (disc<0.0)
                continue;

            double d = sqrt(disc);

            double div = 1.0/(2.0*a);

            double r1 = (-b-d)*div;
            double r2 = (-b+d)*div;

            double ratio = r1 <= 0. ? r2 : r1;
            if (ratio <= 0. || ratio >= 1.)
                continue;

            //std::cerr << "sphere " << i << ", dist=" << sqrt(c) << ", ratio=" << ratio << std::endl;

#if 0
            if (ratio >= is.getIntersections().begin()->ratio)
                continue;
#endif

            osgUtil::LineSegmentIntersector::Intersection hit;
            hit.ratio = ratio;
            hit.nodePath = iv.getNodePath();
            hit.drawable = drawable;
            hit.primitiveIndex = i;
            hit.matrix = iv.getModelMatrix();
            hit.localIntersectionPoint = s + se*ratio;
            osg::Vec3d norm = (hit.localIntersectionPoint-center);
            norm.normalize();
            hit.localIntersectionNormal = norm;
            is.insertIntersection(hit);
        }
    }
};

coSphere::coSphere()
{
    if (!s_configured)
    {
        coIntersection::instance()->addHandler(new SphereIntersector);

        s_useVertexArrays = coCoviseConfig::isOn("COVER.Spheres.UseVertexArrays", false);
        s_configured = true;
    }

    m_useVertexArrays = s_useVertexArrays;

    m_extMaxRadius = 0;

    /*color stuff*/
    m_defaultColor[0] = 1.f;
    m_defaultColor[1] = 0.f;
    m_defaultColor[2] = 0.f;
    m_defaultColor[3] = 1.f;
    m_colorBinding = Bind::PerVertex;
    m_color = NULL;
    m_overrideBounds = false;

    /*geometry stuff*/
    m_numSpheres = -1;
    m_coord = NULL;
    m_radii = NULL;
    m_normals = NULL;
    m_maxRadius = FLT_MIN;
    GLint range[2] = { 0, 32 };
#if 0
   glGetIntegerv(GL_SMOOTH_POINT_SIZE_RANGE, range);
#endif
    m_maxPointSize = range[1];
    m_sortedRadiusIndices = new std::vector<int>[m_maxPointSize + 1];
    for (int i = 0; i <= m_maxPointSize; ++i)
        m_sortedRadiusIndices[i] = std::vector<int>();

    setSupportsDisplayList(false);

#ifdef HAVE_CG
    m_renderMethod = RENDER_METHOD_CG_SHADER; // default value
#else
    m_renderMethod = RENDER_METHOD_CPU_BILLBOARDS;
#endif
    m_renderMethod = RENDER_METHOD_ARB_POINT_SPRITES;

    if (!s_pointSpritesChecked && !s_glPointParameterfvARB && !s_glPointParameterfARB)
    {
        s_glPointParameterfARB = (void(APIENTRY *)(GLenum, GLfloat))osg::getGLExtensionFuncPtr("glPointParameterf", "glPointParameterfARB");
        s_glPointParameterfvARB = (void(APIENTRY *)(GLenum, const GLfloat *))osg::getGLExtensionFuncPtr("glPointParameterfv", "glPointParameterfvARB");
        if (!s_glPointParameterfARB || !s_glPointParameterfARB)
            s_pointSpritesChecked = true;
    }
}

coSphere::coSphere(const coSphere &s, const osg::CopyOp &copyop)
    : Drawable(s, copyop)
{
    /*color stuff*/
    m_defaultColor[0] = s.m_defaultColor[0];
    m_defaultColor[1] = s.m_defaultColor[1];
    m_defaultColor[2] = s.m_defaultColor[2];
    m_defaultColor[3] = s.m_defaultColor[3];
    m_colorBinding = s.m_colorBinding;
    m_useVertexArrays = s.m_useVertexArrays;
    m_renderMethod = s.m_renderMethod;
    m_overrideBounds = s.m_overrideBounds;

    m_coord = NULL;
    m_radii = NULL;
    m_color = NULL;
    m_normals = NULL;
    setNumberOfSpheres(s.m_numSpheres);
    if (m_useVertexArrays)
    {
        memcpy(m_coord, s.m_coord, sizeof(float) * s.m_numSpheres * 3 * 4);
        memcpy(m_radii, s.m_radii, sizeof(float) * s.m_numSpheres * 3 * 4);
        memcpy(m_color, s.m_color, sizeof(float) * s.m_numSpheres * 16);
        if (s.m_normals)
        {
            m_normals = new float[m_numSpheres * 3];
            memcpy(m_normals, s.m_normals, sizeof(float) * s.m_numSpheres * 3 * 4);
        }
    }
    else
    {
        memcpy(m_coord, s.m_coord, sizeof(float) * s.m_numSpheres * 3);
        memcpy(m_radii, s.m_radii, sizeof(float) * s.m_numSpheres);
        memcpy(m_color, s.m_color, sizeof(float) * s.m_numSpheres * 4);
        if (s.m_normals)
        {
            m_normals = new float[m_numSpheres * 3];
            memcpy(m_normals, s.m_normals, sizeof(float) * s.m_numSpheres * 3);
        }
    }
    m_maxRadius = s.m_maxRadius;
    m_maxPointSize = s.m_maxPointSize;
    m_sortedRadiusIndices = new std::vector<int>[m_maxPointSize + 1];
    for (int i = 0; i <= m_maxPointSize; ++i)
        m_sortedRadiusIndices[i] = std::vector<int>(s.m_sortedRadiusIndices[i]);

    setSupportsDisplayList(false);

    if (!s_pointSpritesChecked)
    {
        s_glPointParameterfARB = (void(APIENTRY *)(GLenum, GLfloat))osg::getGLExtensionFuncPtr("glPointParameterf", "glPointParameterfARB");
        s_glPointParameterfvARB = (void(APIENTRY *)(GLenum, const GLfloat *))osg::getGLExtensionFuncPtr("glPointParameterfv", "glPointParameterfvARB");
    }
}

coSphere::~coSphere()
{
    delete[] m_radii;
    delete[] m_coord;
    delete[] m_color;
    delete[] m_normals;
    delete[] m_sortedRadiusIndices;
}

bool coSphere::loadTexture(int context, const char *pFilename, int iTextureMode, int colorR, int colorG, int colorB)
{
    bool bSuccess = false;

    if (iTextureMode == 0)
    {
        //
        // Load up the texture...
        //
        BMPImage textureImage;

        getBitmapImageData(pFilename, &textureImage);
        AUX_RGBImageRec *pImage_RGB = new AUX_RGBImageRec;
        pImage_RGB->sizeX = textureImage.width;
        pImage_RGB->sizeY = textureImage.height;
        pImage_RGB->data = textureImage.data;

        unsigned char *pImage_RGBA = NULL;

        if (pImage_RGB != NULL)
        {
            int imageSize_RGB = pImage_RGB->sizeX * pImage_RGB->sizeY * 3;
            int imageSize_RGBA = pImage_RGB->sizeX * pImage_RGB->sizeY * 4;

            // allocate buffer for a RGBA image
            pImage_RGBA = new unsigned char[imageSize_RGBA];

            //
            // Loop through the original RGB image buffer and copy it over to the
            // new RGBA image buffer setting each pixel that matches the key color
            // transparent.
            //

            int i, j;

            for (i = 0, j = 0; i < imageSize_RGB; i += 3, j += 4)
            {
                // Does the current pixel match the selected color key?
                if (pImage_RGB->data[i] <= colorR && pImage_RGB->data[i + 1] <= colorG && pImage_RGB->data[i + 2] <= colorB)
                {
                    pImage_RGBA[j + 3] = 0; // If so, set alpha to fully transparent.
                }
                else
                {
                    pImage_RGBA[j + 3] = 255; // If not, set alpha to fully opaque.
                }

                pImage_RGBA[j] = pImage_RGB->data[i];
                pImage_RGBA[j + 1] = pImage_RGB->data[i + 1];
                pImage_RGBA[j + 2] = pImage_RGB->data[i + 2];
            }

            if (s_textureID[context] == 0)
                glGenTextures(1, &s_textureID[context]);
            glBindTexture(GL_TEXTURE_2D, s_textureID[context]);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );

            // Don't forget to use GL_RGBA for our new image data... we support Alpha transparency now!
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, pImage_RGB->sizeX, pImage_RGB->sizeY, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, pImage_RGBA);

            bSuccess = true;
        }
        else
        {
            stringstream cInfo;
            cInfo << "Texture file " << pFilename << "not found.\n";
            fprintf(stderr, "%s", cInfo.str().c_str());

            bSuccess = false;
        }

        if (pImage_RGB)
        {
            if (pImage_RGB->data)
                delete[] pImage_RGB -> data;
            delete pImage_RGB;
            pImage_RGB = NULL;
        }

        if (pImage_RGBA)
        {
            delete[] pImage_RGBA;
            pImage_RGBA = NULL;
        }
    }
    else if (iTextureMode == 1)
    {
        //
        // Load up the texture...
        //

        BMPImage textureImage;

        getBitmapImageData(pFilename, &textureImage);
        AUX_RGBImageRec *pTextureImage = new AUX_RGBImageRec;
        pTextureImage->sizeX = textureImage.width;
        pTextureImage->sizeY = textureImage.height;
        memcpy(pTextureImage->data, textureImage.data, sizeof(textureImage.data[0]) * textureImage.width * textureImage.height);

        if (pTextureImage != NULL)
        {
            if (s_textureID[context] == 0)
                glGenTextures(1, &s_textureID[context]);

            glBindTexture(GL_TEXTURE_2D, s_textureID[context]);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            glTexImage2D(GL_TEXTURE_2D, 0, 3, pTextureImage->sizeX, pTextureImage->sizeY, 0,
                         GL_RGB, GL_UNSIGNED_BYTE, pTextureImage->data);

            bSuccess = true;
        }
        else
            bSuccess = false;

        if (pTextureImage)
        {
            delete[] pTextureImage -> data;
            delete pTextureImage;
        }
    }
    return bSuccess;
}

void coSphere::setVertexAttribArray(unsigned int /*index*/, const osg::Array * /*array*/, osg::Array::Binding /*binding*/)
{
}
//-----------------------------------------------------------------------------
// Name: getBitmapImageData()
// Desc: Simply image loader for 24 bit BMP files.
//-----------------------------------------------------------------------------
void coSphere::getBitmapImageData(const char *pFileName, BMPImage *pImage)
{
    FILE *pFile = fopen(pFileName, "rb");
    if (pFile == NULL)
    {
        printf("ERROR: getBitmapImageData - %s not found\n", pFileName);
        return;
    }

    // Seek forward to width and height info
    fseek(pFile, 18, SEEK_CUR);
    if (fread(&pImage->width, 4, 1, pFile) != 1)
    {
        printf("ERROR: getBitmapImageData - Couldn't read width from %s.\n", pFileName);
        return;
    }

    if (fread(&pImage->height, 4, 1, pFile) != 1)
    {
        printf("ERROR: getBitmapImageData - Couldn't read height from %s.\n", pFileName);
        return;
    }

    uint16_t nNumPlanes;
    if (fread(&nNumPlanes, 2, 1, pFile) != 1)
        printf("ERROR: getBitmapImageData - Couldn't read plane count from %s.\n", pFileName);

#ifndef BYTESWAP
    byteSwap(pImage->width);
    byteSwap(pImage->height);
    byteSwap(nNumPlanes);
#endif
    if (nNumPlanes != 1)
    {
        printf("ERROR: getBitmapImageData - Plane count from %s is not 1: %u\n", pFileName, nNumPlanes);
        return;
    }

    uint16_t nNumBPP;
    if (fread(&nNumBPP, 2, 1, pFile) != 1)
        printf("ERROR: getBitmapImageData - Couldn't read BPP from %s.\n", pFileName);
#ifndef BYTESWAP
    byteSwap(nNumBPP);
#endif
    if (nNumBPP != 24)
    {
        printf("ERROR: getBitmapImageData - BPP from %s is not 24: %u\n", pFileName, nNumBPP);
        return;
    }

    // Seek forward to image data
    fseek(pFile, 24, SEEK_CUR);

    // Calculate the image's total size in bytes. Note how we multiply the
    // result of (width * height) by 3. This is because a 24 bit color BMP
    // file will give you 3 bytes per pixel.
    int nTotalImagesize = (pImage->width * pImage->height) * 3;

    pImage->data = new unsigned char[nTotalImagesize];

    if (fread(pImage->data, nTotalImagesize, 1, pFile) != 1)
        printf("ERROR: getBitmapImageData - Couldn't read image data from %s.\n", pFileName);

    //
    // Finally, rearrange BGR to RGB
    //

    for (int i = 0; i < nTotalImagesize; i += 3)
    {
        unsigned char charTemp = pImage->data[i];
        pImage->data[i] = pImage->data[i + 2];
        pImage->data[i + 2] = charTemp;
    }

    fclose(pFile);
}

void coSphere::setColorBinding(int colorbinding)
{
    m_colorBinding = colorbinding;
}

void coSphere::setTransparency(float alpha)
{
    s_overrideTransparency = true;
    s_alpha = alpha;
}

void coSphere::enableTransparencyOverride(bool override)
{
    s_overrideTransparency = override;
}

void coSphere::setScale(float scale)
{
    s_scale = scale;
}

//=====================================================
// sphere stuff
//=====================================================
void coSphere::drawImplementation(osg::RenderInfo &renderInfo) const
{
    mutex()->lock();
    if (s_maxcontext < 0)
    {
        s_maxcontext = renderInfo.getState()->getGraphicsContext()->getMaxContextID();
    }
    /*texture stuff*/
    if (s_textureID == NULL)
    {
        s_textureID = new GLuint[s_maxcontext + 1];
        for (int i = 0; i < s_maxcontext + 1; i++)
        {
            s_textureID[i] = 0;
        }
    }
    mutex()->unlock();
    int thiscontext = renderInfo.getContextID();
    mutex()->lock();
    if (s_textureID[thiscontext] == 0)
        initTexture(thiscontext);
    mutex()->unlock();

    glPushMatrix();
    glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);
    glPushAttrib(GL_ENABLE_BIT);
    glPushAttrib(GL_COLOR_BUFFER_BIT);
    glPushAttrib(GL_POINT_BIT);

    glEnable(GL_TEXTURE_2D);
    glEnable(GL_NORMALIZE);

    glEnable(GL_BLEND);
    if (m_renderMethod == RENDER_METHOD_PARTICLE_CLOUD)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    else
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    GLfloat mat[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, mat);
    osg::Matrix mMat(mat);
    osg::Matrix imMat;
    imMat.invert(mMat);
    osg::Vec3 vPos = imMat.getTrans();

    if (m_renderMethod == RENDER_METHOD_CG_SHADER
        || m_renderMethod == RENDER_METHOD_PARTICLE_CLOUD
        || m_renderMethod == RENDER_METHOD_DISC
        || m_renderMethod == RENDER_METHOD_TEXTURE
        || m_renderMethod == RENDER_METHOD_CG_SHADER_INVERTED)
    {
#ifdef HAVE_CG
        if (s_CgChecked == NULL)
        {
            s_CgChecked = new bool[s_maxcontext + 1];
            for (int i = 0; i < s_maxcontext + 1; i++)
            {
                s_CgChecked[i] = false;
            }
        }
        mutex()->lock();
        if (!s_CgChecked[thiscontext])
            loadCgPrograms(thiscontext);
        mutex()->unlock();

        if (m_renderMethod == RENDER_METHOD_PARTICLE_CLOUD)
            glDepthMask(GL_FALSE);

        bindProgramAndParams(thiscontext);
        bindMatrices(thiscontext);

        GLint maxLights;
        glGetIntegerv(GL_MAX_LIGHTS, &maxLights);
        GLfloat glLightPos[4], viewerPos[4], glAmbient[4], glDiffuse[4], glSpecular[4];
        viewerPos[0] = vPos[0];
        viewerPos[1] = vPos[1];
        viewerPos[2] = vPos[2];
        viewerPos[3] = 1.0;
        for (int i = 0; i < maxLights; i++)
        {
            GLboolean lightIsOn;
            glGetBooleanv(GL_LIGHT0 + i, &lightIsOn);
            if (lightIsOn)
            {
                glGetLightfv(GL_LIGHT0 + i, GL_POSITION, glLightPos);
                glGetLightfv(GL_LIGHT0 + i, GL_AMBIENT, glAmbient);
                glGetLightfv(GL_LIGHT0 + i, GL_DIFFUSE, glDiffuse);
                glGetLightfv(GL_LIGHT0 + i, GL_SPECULAR, glSpecular);
                //printf("LightPos%d: %f, %f, %f, %f\n", i, glLightPos[0], glLightPos[1], glLightPos[2], glLightPos[3]);
                //printf("glAmbient: %f, %f, %f, %f\n", glAmbient[0], glAmbient[1], glAmbient[2], glAmbient[3]);
                //printf("glDiffuse: %f, %f, %f, %f\n", glDiffuse[0], glDiffuse[1], glDiffuse[2], glDiffuse[3]);
                //printf("glSpecular: %f, %f, %f, %f\n", glSpecular[0], glSpecular[1], glSpecular[2], glSpecular[3]);
                break;
            }
        }

        if (m_renderMethod == RENDER_METHOD_DISC || m_renderMethod == RENDER_METHOD_TEXTURE)
        {
            cgGLSetParameter4fv(s_CGVertexParamDisc_lightPos[thiscontext], glLightPos);
            cgGLSetParameter4fv(s_CGVertexParamDisc_viewerPos[thiscontext], viewerPos);
            if (m_renderMethod == RENDER_METHOD_DISC)
            {
                cgGLSetParameter4fv(s_CGFragParamDisc_glAmbient[thiscontext], glAmbient);
                cgGLSetParameter4fv(s_CGFragParamDisc_glDiffuse[thiscontext], glDiffuse);
                cgGLSetParameter4fv(s_CGFragParamDisc_glSpecular[thiscontext], glSpecular);
            }
            else
            {
                cgGLSetParameter4fv(s_CGFragParamTexture_glAmbient[thiscontext], glAmbient);
                cgGLSetParameter4fv(s_CGFragParamTexture_glDiffuse[thiscontext], glDiffuse);
                cgGLSetParameter4fv(s_CGFragParamTexture_glSpecular[thiscontext], glSpecular);
            }
        }
        else
        {
            cgGLSetParameter4fv(s_CGVertexParam_lightPos[thiscontext], glLightPos);
            cgGLSetParameter4fv(s_CGVertexParam_viewerPos[thiscontext], viewerPos);

            if (m_renderMethod == RENDER_METHOD_CG_SHADER_INVERTED)
            {
                cgGLSetParameter4fv(s_CGFragParamInverted_glAmbient[thiscontext], glAmbient);
                cgGLSetParameter4fv(s_CGFragParamInverted_glDiffuse[thiscontext], glDiffuse);
                cgGLSetParameter4fv(s_CGFragParamInverted_glSpecular[thiscontext], glSpecular);
            }
            else
            {
                cgGLSetParameter4fv(s_CGFragParam_glAmbient[thiscontext], glAmbient);
                cgGLSetParameter4fv(s_CGFragParam_glDiffuse[thiscontext], glDiffuse);
                cgGLSetParameter4fv(s_CGFragParam_glSpecular[thiscontext], glSpecular);
            }
        }

        if (m_useVertexArrays)
        {
            glEnableClientState(GL_VERTEX_ARRAY); // Enable Vertex Arrays
            glEnableClientState(GL_COLOR_ARRAY); // Enable Color Arrays
            glEnableClientState(GL_TEXTURE_COORD_ARRAY); // Enable Texture Coord Arrays
            glVertexPointer(3, GL_FLOAT, 0, m_coord);
            glColorPointer(4, GL_FLOAT, 0, m_color);
            glTexCoordPointer(3, GL_FLOAT, 0, m_radii);
            glDrawArrays(GL_QUADS, 0, m_numSpheres * 4); // Disable Pointers
            glDisableClientState(GL_VERTEX_ARRAY); // Disable Vertex Arrays
            glDisableClientState(GL_COLOR_ARRAY); // Disable Color Arrays
            glDisableClientState(GL_TEXTURE_COORD_ARRAY); // Disable Texture Coord Arrays
        }
        else
        {
            glBegin(GL_QUADS);
            for (int i = 0; i < m_numSpheres; i++)
            {
                {
                    if (s_overrideTransparency)
                        glColor4f(m_color[i * 4 + 0], m_color[i * 4 + 1], m_color[i * 4 + 2], s_alpha * m_color[i * 4 + 3]);
                    else
                        glColor4f(m_color[i * 4 + 0], m_color[i * 4 + 1], m_color[i * 4 + 2], m_color[i * 4 + 3]);

                    if (m_renderMethod == RENDER_METHOD_DISC || m_renderMethod == RENDER_METHOD_TEXTURE)
                    {
                        if (m_normals)
                            glNormal3f(m_normals[i * 3], m_normals[i * 3 + 1], m_normals[i * 3 + 2]);
                        else
                            glNormal3f(0., 1., 0.);
                    }

                    glTexCoord3f(-1.0f, -1.0f, m_radii[i] * s_scale);
                    glVertex3fv(&m_coord[i * 3]);

                    glTexCoord3f(1.0f, -1.0f, m_radii[i] * s_scale);
                    glVertex3fv(&m_coord[i * 3]);

                    glTexCoord3f(1.0f, 1.0f, m_radii[i] * s_scale);
                    glVertex3fv(&m_coord[i * 3]);

                    glTexCoord3f(-1.0f, 1.0f, m_radii[i] * s_scale);
                    glVertex3fv(&m_coord[i * 3]);
                }
            }
            glEnd();
        }
        unbindProgramAndParams(thiscontext);
#else
        if (!m_useVertexArrays)
            m_renderMethod = RENDER_METHOD_CPU_BILLBOARDS;
#endif
    }
    else if (m_renderMethod == RENDER_METHOD_ARB_POINT_SPRITES)
    {
        glPushAttrib(GL_LIGHTING);
        glDisable(GL_LIGHTING);
        if (!s_pointSpritesChecked)
        {
            s_pointSpritesChecked = true;
            unsigned ctx = renderInfo.getState()->getContextID();
            if (!osg::isGLExtensionSupported(ctx, "GL_ARB_point_sprite") || !osg::isGLExtensionSupported(ctx, "GL_ARB_point_parameters"))
            {
                s_glPointParameterfvARB = NULL;
                s_glPointParameterfARB = NULL;
            }
        }

        if (s_glPointParameterfARB && s_glPointParameterfvARB)
        {
            glEnable(GL_ALPHA_TEST);
            glAlphaFunc(GL_GREATER, s_overrideTransparency ? s_alpha * 0.8f : 0.8f);

            s_glPointParameterfARB(GL_POINT_FADE_THRESHOLD_SIZE_ARB, 1.0f);

            // the point size will be modified according to scale factor,
            // and glPointSize correction factor (see below glPointSize)
            float scale = cover->getScale();
            float glViewport[4];
            glGetFloatv(GL_VIEWPORT, glViewport);
            //printf("x: %f, y: %f, width: %f, height: %f\n", glViewport[0], glViewport[1], glViewport[2], glViewport[3]);
            float correction = 1 / (m_maxRadius * m_maxRadius * glViewport[2] * glViewport[2] * scale * scale);
            float quadratic[] = { 0.0f, 0.0f, correction };
            s_glPointParameterfvARB(GL_POINT_DISTANCE_ATTENUATION_ARB, quadratic);

            // Specify point sprite texture coordinate replacement mode for each texture unit
            glTexEnvf(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
            //
            // Render point sprites...
            //
            glBindTexture(GL_TEXTURE_2D, s_textureID[thiscontext]);
            // Render each particle...
            GLboolean pointSprites = GL_FALSE;
            glGetBooleanv(GL_POINT_SPRITE_ARB, &pointSprites);
            glEnable(GL_POINT_SPRITE_ARB);
            float minRad = m_maxRadius * 0.0001; // avoid opengl errors

            bool array = false;

            if (array)
            {
                glEnableClientState(GL_COLOR_ARRAY);
                glColorPointer(4, GL_FLOAT, 0, m_color);
                glEnableClientState(GL_VERTEX_ARRAY);
                glVertexPointer(3, GL_FLOAT, 0, m_coord);

                for (int j = 0; j <= m_maxPointSize; ++j)
                {
                    // setting pointSize once per bucket
                    float radius = (j * m_maxRadius) / m_maxPointSize;
                    glPointSize(fmax(minRad, 2 * (radius) / m_maxRadius));
                    //std::cerr << "bucket " << j << "#=" << m_sortedRadiusIndices[j].size() << ": rad=" << radius << "/" << fmax(minRad, 2*radius/m_maxRadius) << std::endl;

                    glDrawElements(GL_POINTS, m_sortedRadiusIndices[j].size(), GL_UNSIGNED_INT, m_sortedRadiusIndices[j].data());
                }

                glDisableClientState(GL_VERTEX_ARRAY);
                glDisableClientState(GL_COLOR_ARRAY);
            }
            else
            {
                for (int j = 0; j <= m_maxPointSize; ++j)
                {
                    // setting pointSize once per bucket
                    float radius = (j * m_maxRadius) / m_maxPointSize;
                    glPointSize(fmax(minRad, 2 * (radius) / m_maxRadius));

                    glBegin(GL_POINTS);
                    for (std::vector<int>::iterator it = m_sortedRadiusIndices[j].begin();
                         it != m_sortedRadiusIndices[j].end(); ++it)
                    {
                        int i = *it;
                        assert(i >= 0);
                        assert(i < m_numSpheres);
                        glColor4f(m_color[i * 4 + 0], m_color[i * 4 + 1], m_color[i * 4 + 2], m_color[i * 4 + 3]);
                        glVertex3f(m_coord[i * 3 + 0], m_coord[i * 3 + 1], m_coord[i * 3 + 2]);
                    }
                    glEnd();
                }
            }

            for (int i = 0; i < m_numSpheres; i++)
            {
                // adapting size of point to radius
                // using window resolution in pixel and
                // window resolution in millimeter
            }

            if (!pointSprites)
                glDisable(GL_POINT_SPRITE_ARB);
        }
        else
        {
            m_renderMethod = RENDER_METHOD_CPU_BILLBOARDS;
        }
        glPopAttrib();//GL_LIGHTING
    }

    if (m_renderMethod == RENDER_METHOD_CPU_BILLBOARDS)
    {
        glEnable(GL_ALPHA_TEST);
        glAlphaFunc(GL_GREATER, s_overrideTransparency ? s_alpha * 0.8f : 0.8f);

        // Recompute the quad points with respect to the view matrix
        // and the quad's center point. The quad's center is always
        // at the origin...

        glBindTexture(GL_TEXTURE_2D, s_textureID[thiscontext]);

        // Render each particle...

        glBegin(GL_QUADS);
        osg::Vec3 normal = vPos;
        normal.normalize();
        glNormal3f(normal[0], normal[1], normal[2]); // eine Normale ist besser als keine und kostet nichts (eine Normale pro

        for (int i = 0; i < m_numSpheres; i++)
        {
            if (s_overrideTransparency)
                glColor4f(m_color[i * 4 + 0], m_color[i * 4 + 1], m_color[i * 4 + 2], s_alpha);
            else
                glColor4f(m_color[i * 4 + 0], m_color[i * 4 + 1], m_color[i * 4 + 2], m_color[i * 4 + 3]);

            //-------------------------------------------------------------
            //
            // vPoint3                vPoint2
            //         +------------+
            //         |            |
            //         |     +      |
            //         |  vCenter   |
            //         |            |
            //         +------------+
            // vPoint0                vPoint1
            //
            // Now, build a quad around the center point based on the vRight
            // and vUp vectors. This will guarantee that the quad will be
            // orthogonal to the view.
            //
            //-------------------------------------------------------------
            osg::Vec3 pos(m_coord[i * 3 + 0], m_coord[i * 3 + 1], m_coord[i * 3 + 2]);
            // this toPoint is in viewer coordsosg::Vec3 toPoint( mMat.preMult(pos) );
            //we need it in object coords
            osg::Vec3 toPoint = pos - vPos; // from viewer to pos
            toPoint.normalize();

            //osg::Vec3 up( imMat(0,2), imMat(1,2), imMat(2,2) ); // viewer up, could also use object up....
            //                                                         up.normalize();
            osg::Vec3 up(0, 0, 1);

            osg::Vec3 right(up ^ toPoint);
            right.normalize();

            up = right ^ toPoint;

            osg::Vec3 vPoint0(pos + ((-right - up) * m_radii[i]) * s_scale);
            osg::Vec3 vPoint1(pos + ((right - up) * m_radii[i]) * s_scale);
            osg::Vec3 vPoint2(pos + ((right + up) * m_radii[i]) * s_scale);
            osg::Vec3 vPoint3(pos + ((-right + up) * m_radii[i]) * s_scale);

            glTexCoord2f(0.0f, 0.0f);
            glVertex3fv(vPoint0.ptr());

            glTexCoord2f(1.0f, 0.0f);
            glVertex3fv(vPoint1.ptr());

            glTexCoord2f(1.0f, 1.0f);
            glVertex3fv(vPoint2.ptr());

            glTexCoord2f(0.0f, 1.0f);
            glVertex3fv(vPoint3.ptr());
        }
        glEnd();
    }
    //
    // Reset OpenGL states...
    //
    glPopClientAttrib(); // GL_CLIENT_VERTEX_ARRAY_BIT
    glPopAttrib(); // GL_POINT_BIT
    glPopAttrib(); // GL_COLOR_BUFFER_BIT
    glPopAttrib(); // GL_ENABLE_BIT
    glPopMatrix();
}

void coSphere::initTexture(int context)
{
    if (const char *fn = coVRFileManager::instance()->getName("share/covise/materials/textures/particle.bmp"))
    {
        setTexture(fn);

        loadTexture(context, s_chTexFile, 0, 30, 30, 30);
    }
}

#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
osg::BoundingBox coSphere::computeBoundingBox() const
#else
osg::BoundingBox coSphere::computeBound() const
#endif
{
    if (m_overrideBounds)
        return _boundingBox;

    if (m_numSpheres <= 0)
        return osg::BoundingBox();

    float xmin = FLT_MAX, ymin = FLT_MAX, zmin = FLT_MAX, xmax = -FLT_MAX, ymax = -FLT_MAX, zmax = -FLT_MAX;
    if (m_useVertexArrays)
    {
        for (int i = 0; i < m_numSpheres; i++)
        {
            if (m_coord[i * 12 + 0] + m_radii[i * 12 + 2] > xmax)
                xmax = m_coord[i * 12 + 0] + m_radii[i * 12 + 2];
            if (m_coord[i * 12 + 1] + m_radii[i * 12 + 2] > ymax)
                ymax = m_coord[i * 12 + 1] + m_radii[i * 12 + 2];
            if (m_coord[i * 12 + 2] + m_radii[i * 12 + 2] > zmax)
                zmax = m_coord[i * 12 + 2] + m_radii[i * 12 + 2];
            if (m_coord[i * 12 + 0] - m_radii[i * 12 + 2] < xmin)
                xmin = m_coord[i * 12 + 0] - m_radii[i * 12 + 2];
            if (m_coord[i * 12 + 1] - m_radii[i * 12 + 2] < ymin)
                ymin = m_coord[i * 12 + 1] - m_radii[i * 12 + 2];
            if (m_coord[i * 12 + 2] - m_radii[i * 12 + 2] < zmin)
                zmin = m_coord[i * 12 + 2] - m_radii[i * 12 + 2];
        }
    }
    else
    {
        for (int i = 0; i < m_numSpheres; i++)
        {
            if (m_coord[i * 3 + 0] + m_radii[i] * s_scale > xmax)
                xmax = m_coord[i * 3 + 0] + m_radii[i] * s_scale;
            if (m_coord[i * 3 + 1] + m_radii[i] * s_scale > ymax)
                ymax = m_coord[i * 3 + 1] + m_radii[i] * s_scale;
            if (m_coord[i * 3 + 2] + m_radii[i] * s_scale > zmax)
                zmax = m_coord[i * 3 + 2] + m_radii[i] * s_scale;
            if (m_coord[i * 3 + 0] - m_radii[i] * s_scale < xmin)
                xmin = m_coord[i * 3 + 0] - m_radii[i] * s_scale;
            if (m_coord[i * 3 + 1] - m_radii[i] * s_scale < ymin)
                ymin = m_coord[i * 3 + 1] - m_radii[i] * s_scale;
            if (m_coord[i * 3 + 2] - m_radii[i] * s_scale < zmin)
                zmin = m_coord[i * 3 + 2] - m_radii[i] * s_scale;
        }
    }

    return osg::BoundingBox(osg::Vec3(xmin, ymin, zmin), osg::Vec3(xmax, ymax, zmax));
}

void coSphere::setTexture(const char *chTexFile)
{
    // Deallocate the memory that was previously reserved for this string.
    if (s_chTexFile)
    {
        delete[] s_chTexFile;
        s_chTexFile = NULL;
    }

    // Dynamically allocate the correct amount of memory.
    s_chTexFile = new char[strlen(chTexFile) + 1];
    strcpy(s_chTexFile, chTexFile);
}

void
coSphere::setNumberOfSpheres(int no_of_points)
{
    if (no_of_points < 0)
        no_of_points = 0;

    if (m_numSpheres != no_of_points)
    {
        delete[] m_coord;
        delete[] m_radii;
        delete[] m_color;
        delete[] m_normals;
        m_normals = NULL;
        if (m_useVertexArrays)
        {
            m_coord = new float[4 * 3 * no_of_points];
            m_radii = new float[4 * 3 * no_of_points];
            m_color = new float[16 * no_of_points];
        }
        else
        {
            m_coord = new float[3 * no_of_points];
            m_radii = new float[no_of_points];
            m_color = new float[4 * no_of_points];
        }
    }
    m_numSpheres = no_of_points;
}

void coSphere::overrideBoundingBox(const osg::BoundingBox &bb)
{
    m_overrideBounds = true;
    setInitialBound(bb);
    dirtyBound();
}

//=========================================================================
// set the coordinates and radius of the point object
//=========================================================================
void
coSphere::setCoords(int no_of_points, const float *x_c, const float *y_c,
                    const float *z_c, const float *radii)
{
    if (no_of_points < 0)
        no_of_points = 0;
    setNumberOfSpheres(no_of_points);

    dirtyBound();

    m_maxRadius = FLT_MIN;
    if (m_useVertexArrays)
    {
        for (int i = 0; i < m_numSpheres; i++)
        {
            m_coord[i * 12 + 0] = x_c[i];
            m_coord[i * 12 + 1] = y_c[i];
            m_coord[i * 12 + 2] = z_c[i];
            m_coord[i * 12 + 3] = x_c[i];
            m_coord[i * 12 + 4] = y_c[i];
            m_coord[i * 12 + 5] = z_c[i];
            m_coord[i * 12 + 6] = x_c[i];
            m_coord[i * 12 + 7] = y_c[i];
            m_coord[i * 12 + 8] = z_c[i];
            m_coord[i * 12 + 9] = x_c[i];
            m_coord[i * 12 + 10] = y_c[i];
            m_coord[i * 12 + 11] = z_c[i];
            m_radii[i * 12 + 0] = -1.0f;
            m_radii[i * 12 + 1] = -1.0f;
            m_radii[i * 12 + 2] = radii[i];
            m_radii[i * 12 + 3] = 1.0f;
            m_radii[i * 12 + 4] = -1.0f;
            m_radii[i * 12 + 5] = radii[i];
            m_radii[i * 12 + 6] = 1.0f;
            m_radii[i * 12 + 7] = 1.0f;
            m_radii[i * 12 + 8] = radii[i];
            m_radii[i * 12 + 9] = -1.0f;
            m_radii[i * 12 + 10] = 1.0f;
            m_radii[i * 12 + 11] = radii[i];

            m_maxRadius = m_maxRadius < radii[i] ? radii[i] : m_maxRadius;
        }
    }
    else
    {
        for (int i = 0; i < m_numSpheres; i++)
        {
            m_maxRadius = m_maxRadius < radii[i] ? radii[i] : m_maxRadius;
            m_coord[i * 3 + 0] = x_c[i];
            m_coord[i * 3 + 1] = y_c[i];
            m_coord[i * 3 + 2] = z_c[i];
            m_radii[i] = radii[i];
        }
    }
    if (m_extMaxRadius != 0)
        m_maxRadius = m_extMaxRadius;

    // sorting by radius
    for (int i = 0; i <= m_maxPointSize; ++i)
        m_sortedRadiusIndices[i].clear();
    for (int i = 0; i < m_numSpheres; ++i)
    {
        int bucket = (int)floor(0.5 + m_radii[i] / m_maxRadius * m_maxPointSize);
        if (bucket < 0)
            bucket = 0;
        if (bucket >= m_maxPointSize)
            bucket = m_maxPointSize - 1;
        m_sortedRadiusIndices[bucket].push_back(i);
    }
}

void
coSphere::setCoords(int no_of_points, const float *x_c, const float *y_c,
                    const float *z_c, float r)
{
    if (no_of_points < 0)
        no_of_points = 0;
    setNumberOfSpheres(no_of_points);

    dirtyBound();

    m_maxRadius = r;
    if (m_useVertexArrays)
    {
        for (int i = 0; i < m_numSpheres; i++)
        {
            m_coord[i * 12 + 0] = x_c[i];
            m_coord[i * 12 + 1] = y_c[i];
            m_coord[i * 12 + 2] = z_c[i];
            m_coord[i * 12 + 3] = x_c[i];
            m_coord[i * 12 + 4] = y_c[i];
            m_coord[i * 12 + 5] = z_c[i];
            m_coord[i * 12 + 6] = x_c[i];
            m_coord[i * 12 + 7] = y_c[i];
            m_coord[i * 12 + 8] = z_c[i];
            m_coord[i * 12 + 9] = x_c[i];
            m_coord[i * 12 + 10] = y_c[i];
            m_coord[i * 12 + 11] = z_c[i];
            m_radii[i * 12 + 0] = -1.0f;
            m_radii[i * 12 + 1] = -1.0f;
            m_radii[i * 12 + 2] = r;
            m_radii[i * 12 + 3] = 1.0f;
            m_radii[i * 12 + 4] = -1.0f;
            m_radii[i * 12 + 5] = r;
            m_radii[i * 12 + 6] = 1.0f;
            m_radii[i * 12 + 7] = 1.0f;
            m_radii[i * 12 + 8] = r;
            m_radii[i * 12 + 9] = -1.0f;
            m_radii[i * 12 + 10] = 1.0f;
            m_radii[i * 12 + 11] = r;
        }
    }
    else
    {
        for (int i = 0; i < m_numSpheres; i++)
        {
            m_coord[i * 3 + 0] = x_c[i];
            m_coord[i * 3 + 1] = y_c[i];
            m_coord[i * 3 + 2] = z_c[i];
            m_radii[i] = r;
        }
    }
    if (m_extMaxRadius != 0)
        m_maxRadius = m_extMaxRadius;

    // sorting by radius
    for (int i = 0; i <= m_maxPointSize; ++i)
        m_sortedRadiusIndices[i].clear();
    int bucket = (int)floor(0.5 + r / m_maxRadius * m_maxPointSize);
    if (bucket < 0)
        bucket = 0;
    if (bucket >= m_maxPointSize)
            bucket = m_maxPointSize - 1;
    for (int i = 0; i < m_numSpheres; ++i)
    {
        m_sortedRadiusIndices[bucket].push_back(i);
    }
}

void
coSphere::setCoords(int no_of_points, const osg::Vec3Array* coords, const float *r)
{
    if (no_of_points < 0)
        no_of_points = 0;
    setNumberOfSpheres(no_of_points);

    dirtyBound();

    osg::Vec3Array::const_iterator coord = coords->begin();

    m_maxRadius = FLT_MIN;
    if (m_useVertexArrays)
    {
        for (int i = 0; i < m_numSpheres; i++)
        {
            const osg::Vec3 &pos = *coord;
            m_coord[i * 12 + 0] = pos.x();
            m_coord[i * 12 + 1] = pos.y();
            m_coord[i * 12 + 2] = pos.z();
            m_coord[i * 12 + 3] = pos.x();
            m_coord[i * 12 + 4] = pos.y();
            m_coord[i * 12 + 5] = pos.z();
            m_coord[i * 12 + 6] = pos.x();
            m_coord[i * 12 + 7] = pos.y();
            m_coord[i * 12 + 8] = pos.z();
            m_coord[i * 12 + 9] = pos.x();
            m_coord[i * 12 + 10] = pos.y();
            m_coord[i * 12 + 11] = pos.z();
            m_radii[i * 12 + 0] = -1.0f;
            m_radii[i * 12 + 1] = -1.0f;
            m_radii[i * 12 + 2] = r[i];
            m_radii[i * 12 + 3] = 1.0f;
            m_radii[i * 12 + 4] = -1.0f;
            m_radii[i * 12 + 5] = r[i];
            m_radii[i * 12 + 6] = 1.0f;
            m_radii[i * 12 + 7] = 1.0f;
            m_radii[i * 12 + 8] = r[i];
            m_radii[i * 12 + 9] = -1.0f;
            m_radii[i * 12 + 10] = 1.0f;
            m_radii[i * 12 + 11] = r[i];
            ++coord;

            m_maxRadius = m_maxRadius < r[i] ? r[i] : m_maxRadius;
        }
    }
    else
    {
        for (int i = 0; i < m_numSpheres; i++)
        {
            const osg::Vec3 &pos = *coord;
            m_maxRadius = m_maxRadius < r[i] ? r[i] : m_maxRadius;
            m_coord[i * 3 + 0] = pos.x();
            m_coord[i * 3 + 1] = pos.y();
            m_coord[i * 3 + 2] = pos.z();
            m_radii[i] = r[i];
            ++coord;
        }
    }

    if (m_extMaxRadius != 0)
        m_maxRadius = m_extMaxRadius;

    // sorting by radius
    for (int i = 0; i <= m_maxPointSize; ++i)
        m_sortedRadiusIndices[i].clear();
    for (int i = 0; i < m_numSpheres; ++i)
    {
        int bucket = (int)floor(0.5 + m_radii[i] / m_maxRadius * m_maxPointSize);
        if (bucket < 0)
            bucket = 0;
        if (bucket >= m_maxPointSize)
            bucket = m_maxPointSize - 1;
        m_sortedRadiusIndices[bucket].push_back(i);
    }
}

void coSphere::updateNormals(const float *nx, const float *ny, const float *nz)
{
    if (m_useVertexArrays)
    {
        if (!m_normals)
        {
            m_normals = new float[m_numSpheres * 3 * 4];
        }
        for (int i = 0; i < m_numSpheres * 3; i++)
        {
            // TODO
            m_normals[i * 12 + 0] = -1.0f;
            m_normals[i * 12 + 1] = -1.0f;
            m_normals[i * 12 + 2] = nz[i];
            m_normals[i * 12 + 3] = 1.0f;
            m_normals[i * 12 + 4] = -1.0f;
            m_normals[i * 12 + 5] = nz[i];
            m_normals[i * 12 + 6] = 1.0f;
            m_normals[i * 12 + 7] = 1.0f;
            m_normals[i * 12 + 8] = nz[i];
            m_normals[i * 12 + 9] = -1.0f;
            m_normals[i * 12 + 10] = 1.0f;
            m_normals[i * 12 + 11] = nz[i];
        }
    }
    else
    {
        if (!m_normals)
        {
            m_normals = new float[m_numSpheres * 3];
        }
        for (int i = 0; i < m_numSpheres; i++)
        {
            m_normals[i * 3 + 0] = nx[i];
            m_normals[i * 3 + 1] = ny[i];
            m_normals[i * 3 + 2] = nz[i];
        }
    }
}

void coSphere::updateRadii(const double *radii)
{
    dirtyBound();

    m_maxRadius = FLT_MIN;
    if (m_useVertexArrays)
    {
        for (int i = 0; i < m_numSpheres; i++)
        {
            m_radii[i * 12 + 0] = -1.0f;
            m_radii[i * 12 + 1] = -1.0f;
            m_radii[i * 12 + 2] = radii[i];
            m_radii[i * 12 + 3] = 1.0f;
            m_radii[i * 12 + 4] = -1.0f;
            m_radii[i * 12 + 5] = radii[i];
            m_radii[i * 12 + 6] = 1.0f;
            m_radii[i * 12 + 7] = 1.0f;
            m_radii[i * 12 + 8] = radii[i];
            m_radii[i * 12 + 9] = -1.0f;
            m_radii[i * 12 + 10] = 1.0f;
            m_radii[i * 12 + 11] = radii[i];

            m_maxRadius = m_maxRadius < radii[i] ? radii[i] : m_maxRadius;
        }
    }
    else
    {
        for (int i = 0; i < m_numSpheres; i++)
        {
            m_radii[i] = radii[i];
            m_maxRadius = m_maxRadius < m_radii[i] ? m_radii[i] : m_maxRadius;
        }
    }

    // sorting by radius
    for (int i = 0; i <= m_maxPointSize; ++i)
        m_sortedRadiusIndices[i].clear();
    for (int i = 0; i < m_numSpheres; ++i)
    {
        int bucket = (int)floor(0.5 + m_radii[i] / m_maxRadius * m_maxPointSize);
        if (bucket < 0)
            bucket = 0;
        if (bucket >= m_maxPointSize)
            bucket = m_maxPointSize - 1;
        m_sortedRadiusIndices[bucket].push_back(i);
    }
}

void coSphere::updateRadii(const float *radii)
{
    dirtyBound();

    m_maxRadius = FLT_MIN;
    if (m_useVertexArrays)
    {
        for (int i = 0; i < m_numSpheres; i++)
        {
            m_radii[i * 12 + 0] = -1.0f;
            m_radii[i * 12 + 1] = -1.0f;
            m_radii[i * 12 + 2] = radii[i];
            m_radii[i * 12 + 3] = 1.0f;
            m_radii[i * 12 + 4] = -1.0f;
            m_radii[i * 12 + 5] = radii[i];
            m_radii[i * 12 + 6] = 1.0f;
            m_radii[i * 12 + 7] = 1.0f;
            m_radii[i * 12 + 8] = radii[i];
            m_radii[i * 12 + 9] = -1.0f;
            m_radii[i * 12 + 10] = 1.0f;
            m_radii[i * 12 + 11] = radii[i];

            m_maxRadius = m_maxRadius < radii[i] ? radii[i] : m_maxRadius;
        }
    }
    else
    {
        for (int i = 0; i < m_numSpheres; i++)
        {
            m_radii[i] = radii[i];
            m_maxRadius = m_maxRadius < m_radii[i] ? m_radii[i] : m_maxRadius;
        }
    }

    // sorting by radius
    for (int i = 0; i <= m_maxPointSize; ++i)
        m_sortedRadiusIndices[i].clear();
    for (int i = 0; i < m_numSpheres; ++i)
    {
        int bucket = (int)floor(0.5 + m_radii[i] / m_maxRadius * m_maxPointSize);
        if (bucket < 0)
            bucket = 0;
        if (bucket >= m_maxPointSize)
            bucket = m_maxPointSize - 1;
        m_sortedRadiusIndices[bucket].push_back(i);
    }
}

void coSphere::updateCoordsFromMatrices(float *const *matrices)
{
    dirtyBound();

    if (m_useVertexArrays)
    {
        for (int i = 0; i < m_numSpheres; i++)
        {
            m_coord[i * 12 + 0] = matrices[i][12];
            m_coord[i * 12 + 1] = matrices[i][13];
            m_coord[i * 12 + 2] = matrices[i][14];
            m_coord[i * 12 + 3] = matrices[i][12];
            m_coord[i * 12 + 4] = matrices[i][13];
            m_coord[i * 12 + 5] = matrices[i][14];
            m_coord[i * 12 + 6] = matrices[i][12];
            m_coord[i * 12 + 7] = matrices[i][13];
            m_coord[i * 12 + 8] = matrices[i][14];
            m_coord[i * 12 + 9] = matrices[i][12];
            m_coord[i * 12 + 10] = matrices[i][13];
            m_coord[i * 12 + 11] = matrices[i][14];
        }
    }
    else
    {
        for (int i = 0; i < m_numSpheres; i++)
        {
            m_coord[i * 3 + 0] = matrices[i][12];
            m_coord[i * 3 + 1] = matrices[i][13];
            m_coord[i * 3 + 2] = matrices[i][14];
        }
    }
}

void coSphere::updateCoords(const float *x_c, const float *y_c, const float *z_c)
{
    dirtyBound();

    if (m_useVertexArrays)
    {
        for (int i = 0; i < m_numSpheres; i++)
        {
            m_coord[i * 12 + 0] = x_c[i];
            m_coord[i * 12 + 1] = y_c[i];
            m_coord[i * 12 + 2] = z_c[i];
            m_coord[i * 12 + 3] = x_c[i];
            m_coord[i * 12 + 4] = y_c[i];
            m_coord[i * 12 + 5] = z_c[i];
            m_coord[i * 12 + 6] = x_c[i];
            m_coord[i * 12 + 7] = y_c[i];
            m_coord[i * 12 + 8] = z_c[i];
            m_coord[i * 12 + 9] = x_c[i];
            m_coord[i * 12 + 10] = y_c[i];
            m_coord[i * 12 + 11] = z_c[i];
        }
    }
    else
    {
        for (int i = 0; i < m_numSpheres; i++)
        {
            m_coord[i * 3 + 0] = x_c[i];
            m_coord[i * 3 + 1] = y_c[i];
            m_coord[i * 3 + 2] = z_c[i];
        }
    }
}

void coSphere::updateCoords(int i, const osg::Vec3 &pos)
{
    dirtyBound();

    if (m_useVertexArrays)
    {
            m_coord[i * 12 + 0] = pos.x();
            m_coord[i * 12 + 1] = pos.y();
            m_coord[i * 12 + 2] = pos.z();
            m_coord[i * 12 + 3] = pos.x();
            m_coord[i * 12 + 4] = pos.y();
            m_coord[i * 12 + 5] = pos.z();
            m_coord[i * 12 + 6] = pos.x();
            m_coord[i * 12 + 7] = pos.y();
            m_coord[i * 12 + 8] = pos.z();
            m_coord[i * 12 + 9] = pos.x();
            m_coord[i * 12 + 10] = pos.y();
            m_coord[i * 12 + 11] = pos.z();

    }
    else
    {
            m_coord[i * 3 + 0] = pos.x();
            m_coord[i * 3 + 1] = pos.y();
            m_coord[i * 3 + 2] = pos.z();
    }
}


void coSphere::setRenderMethod(RenderMethod rm)
{
    m_useVertexArrays = false;
    if (rm == RENDER_METHOD_CG_SHADER
        || rm == RENDER_METHOD_PARTICLE_CLOUD
        || rm == RENDER_METHOD_DISC
        || rm == RENDER_METHOD_TEXTURE
        || rm == RENDER_METHOD_CG_SHADER_INVERTED)
    {
        m_renderMethod = rm;
#ifdef HAVE_CG

        if (s_CgChecked != NULL && s_CgChecked[0]
            && s_VertexProgram != NULL && s_VertexProgram[0]
            && s_FragmentProgram != NULL && s_FragmentProgram[0]
            && s_FragmentProgramParticleCloud != NULL && s_FragmentProgramParticleCloud[0]
            && s_VertexProgramDisc != NULL && s_VertexProgramDisc[0]
            && s_FragmentProgramDisc != NULL && s_FragmentProgramDisc[0]
            && s_FragmentProgramTexture != NULL && s_FragmentProgramTexture[0]
            && s_FragmentProgramInverted != NULL && s_FragmentProgramInverted[0])
            m_useVertexArrays = s_useVertexArrays;
#endif
    }
    else if (rm == RENDER_METHOD_ARB_POINT_SPRITES && s_glPointParameterfARB && s_glPointParameterfvARB)
        m_renderMethod = rm;
    else
        m_renderMethod = RENDER_METHOD_CPU_BILLBOARDS;
}

void coSphere::setColor(const int index, float fR, float fG, float fB, float fA)
{
    if (m_useVertexArrays)
    {
        m_color[index * 16 + 0] = fR;
        m_color[index * 16 + 1] = fG;
        m_color[index * 16 + 2] = fB;
        m_color[index * 16 + 3] = fA;
        m_color[index * 16 + 4] = fR;
        m_color[index * 16 + 5] = fG;
        m_color[index * 16 + 6] = fB;
        m_color[index * 16 + 7] = fA;
        m_color[index * 16 + 8] = fR;
        m_color[index * 16 + 9] = fG;
        m_color[index * 16 + 10] = fB;
        m_color[index * 16 + 11] = fA;
        m_color[index * 16 + 12] = fR;
        m_color[index * 16 + 13] = fG;
        m_color[index * 16 + 14] = fB;
        m_color[index * 16 + 15] = fA;
    }
    else
    {
        m_color[index * 4 + 0] = fR;
        m_color[index * 4 + 1] = fG;
        m_color[index * 4 + 2] = fB;
        m_color[index * 4 + 3] = fA;
    }
}

void coSphere::updateColors(const float *r, const float *g, const float *b, const float *a)
{
    if (m_useVertexArrays)
    {
        for (int k = 0; k < m_numSpheres; k++)
        {
            int i = m_colorBinding == Bind::OverAll ? 0 : k;
            if (r && g && b)
            {
                m_color[k * 16 + 0] = r[i];
                m_color[k * 16 + 1] = g[i];
                m_color[k * 16 + 2] = b[i];
                m_color[k * 16 + 4] = r[i];
                m_color[k * 16 + 5] = g[i];
                m_color[k * 16 + 6] = b[i];
                m_color[k * 16 + 8] = r[i];
                m_color[k * 16 + 9] = g[i];
                m_color[k * 16 + 10] = b[i];
                m_color[k * 16 + 12] = r[i];
                m_color[k * 16 + 13] = g[i];
                m_color[k * 16 + 14] = b[i];
            }
            else
            {
                m_color[k * 16 + 0] = m_defaultColor[0];
                m_color[k * 16 + 1] = m_defaultColor[1];
                m_color[k * 16 + 2] = m_defaultColor[2];
                m_color[k * 16 + 4] = m_defaultColor[0];
                m_color[k * 16 + 5] = m_defaultColor[1];
                m_color[k * 16 + 6] = m_defaultColor[2];
                m_color[k * 16 + 8] = m_defaultColor[0];
                m_color[k * 16 + 9] = m_defaultColor[1];
                m_color[k * 16 + 10] = m_defaultColor[2];
                m_color[k * 16 + 12] = m_defaultColor[0];
                m_color[k * 16 + 13] = m_defaultColor[1];
                m_color[k * 16 + 14] = m_defaultColor[2];
            }
            if (a)
            {
                m_color[k * 16 + 3] = a[i];
                m_color[k * 16 + 7] = a[i];
                m_color[k * 16 + 11] = a[i];
                m_color[k * 16 + 15] = a[i];
            }
            else
            {
                m_color[k * 16 + 3] = m_defaultColor[3];
                m_color[k * 16 + 7] = m_defaultColor[3];
                m_color[k * 16 + 11] = m_defaultColor[3];
                m_color[k * 16 + 15] = m_defaultColor[3];
            }
        }
    }
    else
    {
        for (int k = 0; k < m_numSpheres; k++)
        {
            int i = m_colorBinding == Bind::OverAll ? 0 : k;
            if (r && g && b)
            {
                m_color[k * 4 + 0] = r[i];
                m_color[k * 4 + 1] = g[i];
                m_color[k * 4 + 2] = b[i];
            }
            else
            {
                m_color[k * 4 + 0] = m_defaultColor[0];
                m_color[k * 4 + 1] = m_defaultColor[1];
                m_color[k * 4 + 2] = m_defaultColor[2];
            }
            if (a)
            {
                m_color[k * 4 + 3] = a[i];
            }
            else
            {
                m_color[k * 4 + 3] = m_defaultColor[3];
            }
        }
    }
}

void coSphere::updateColors(const int *pc)
{
    if (m_useVertexArrays)
    {
        for (int k = 0; k < m_numSpheres; k++)
        {
            int i = m_colorBinding == Bind::OverAll ? 0 : k;
            if (pc)
            {
                float r, g, b, a;
                unpackRGBA(pc, i, &r, &g, &b, &a);
                m_color[k * 16 + 0] = r;
                m_color[k * 16 + 1] = g;
                m_color[k * 16 + 2] = b;
                m_color[k * 16 + 3] = a;
                m_color[k * 16 + 4] = r;
                m_color[k * 16 + 5] = g;
                m_color[k * 16 + 6] = b;
                m_color[k * 16 + 7] = a;
                m_color[k * 16 + 8] = r;
                m_color[k * 16 + 9] = g;
                m_color[k * 16 + 10] = b;
                m_color[k * 16 + 11] = a;
                m_color[k * 16 + 12] = r;
                m_color[k * 16 + 13] = g;
                m_color[k * 16 + 14] = b;
                m_color[k * 16 + 15] = a;
            }
            else
            {
                m_color[k * 16 + 0] = m_defaultColor[0];
                m_color[k * 16 + 1] = m_defaultColor[1];
                m_color[k * 16 + 2] = m_defaultColor[2];
                m_color[k * 16 + 3] = m_defaultColor[3];
                m_color[k * 16 + 4] = m_defaultColor[0];
                m_color[k * 16 + 5] = m_defaultColor[1];
                m_color[k * 16 + 6] = m_defaultColor[2];
                m_color[k * 16 + 7] = m_defaultColor[3];
                m_color[k * 16 + 8] = m_defaultColor[0];
                m_color[k * 16 + 9] = m_defaultColor[1];
                m_color[k * 16 + 10] = m_defaultColor[2];
                m_color[k * 16 + 11] = m_defaultColor[3];
                m_color[k * 16 + 12] = m_defaultColor[0];
                m_color[k * 16 + 13] = m_defaultColor[1];
                m_color[k * 16 + 14] = m_defaultColor[2];
                m_color[k * 16 + 15] = m_defaultColor[3];
            }
        }
    }
    else
    {
        for (int k = 0; k < m_numSpheres; k++)
        {
            int i = m_colorBinding == Bind::OverAll ? 0 : k;
            if (pc)
            {
                float r, g, b, a;
                unpackRGBA(pc, i, &r, &g, &b, &a);
                m_color[k * 4 + 0] = r;
                m_color[k * 4 + 1] = g;
                m_color[k * 4 + 2] = b;
                m_color[k * 4 + 3] = a;
            }
            else
            {
                m_color[k * 4 + 0] = m_defaultColor[0];
                m_color[k * 4 + 1] = m_defaultColor[1];
                m_color[k * 4 + 2] = m_defaultColor[2];
                m_color[k * 4 + 3] = m_defaultColor[3];
            }
        }
    }
}

bool coSphere::loadCgPrograms(int context)
{
#ifndef HAVE_CG
    (void)context;
#else

    s_CgChecked[context] = true;
    if (s_VertexProgram == NULL || s_FragmentProgram == NULL
        || s_FragmentProgramParticleCloud == NULL
        || s_VertexProgramDisc == NULL || s_FragmentProgramDisc == NULL
        || s_FragmentProgramTexture == NULL || s_FragmentProgramInverted == NULL)

    {
        s_VertexProgram = new CGprogram[s_maxcontext + 1];
        s_FragmentProgram = new CGprogram[s_maxcontext + 1];
        s_FragmentProgramParticleCloud = new CGprogram[s_maxcontext + 1];
        s_VertexProgramDisc = new CGprogram[s_maxcontext + 1];
        s_FragmentProgramDisc = new CGprogram[s_maxcontext + 1];
        s_FragmentProgramTexture = new CGprogram[s_maxcontext + 1];
        s_FragmentProgramInverted = new CGprogram[s_maxcontext + 1];

        s_CGVertexParam_modelViewProj = new CGparameter[s_maxcontext + 1];
        s_CGVertexParam_modelView = new CGparameter[s_maxcontext + 1];
        s_CGVertexParam_modelViewIT = new CGparameter[s_maxcontext + 1];
        s_CGVertexParam_lightPos = new CGparameter[s_maxcontext + 1];
        s_CGVertexParam_viewerPos = new CGparameter[s_maxcontext + 1];
        s_CGFragParam_glAmbient = new CGparameter[s_maxcontext + 1];
        s_CGFragParam_glDiffuse = new CGparameter[s_maxcontext + 1];
        s_CGFragParam_glSpecular = new CGparameter[s_maxcontext + 1];
        s_CGFragParamInverted_glAmbient = new CGparameter[s_maxcontext + 1];
        s_CGFragParamInverted_glDiffuse = new CGparameter[s_maxcontext + 1];
        s_CGFragParamInverted_glSpecular = new CGparameter[s_maxcontext + 1];

        s_CGVertexParamDisc_modelViewProj = new CGparameter[s_maxcontext + 1];
        s_CGVertexParamDisc_modelView = new CGparameter[s_maxcontext + 1];
        s_CGVertexParamDisc_modelViewIT = new CGparameter[s_maxcontext + 1];
        s_CGVertexParamDisc_lightPos = new CGparameter[s_maxcontext + 1];
        s_CGVertexParamDisc_viewerPos = new CGparameter[s_maxcontext + 1];
        s_CGFragParamDisc_glAmbient = new CGparameter[s_maxcontext + 1];
        s_CGFragParamDisc_glDiffuse = new CGparameter[s_maxcontext + 1];
        s_CGFragParamDisc_glSpecular = new CGparameter[s_maxcontext + 1];
        s_CGFragParamTexture_glAmbient = new CGparameter[s_maxcontext + 1];
        s_CGFragParamTexture_glDiffuse = new CGparameter[s_maxcontext + 1];
        s_CGFragParamTexture_glSpecular = new CGparameter[s_maxcontext + 1];

        s_VertexProfile = new CGprofile[s_maxcontext + 1];
        s_FragmentProfile = new CGprofile[s_maxcontext + 1];

        for (int i = 0; i < s_maxcontext + 1; i++)
        {
            s_VertexProgram[i] = NULL;
            s_FragmentProgram[i] = NULL;
            s_FragmentProgramParticleCloud[i] = NULL;
            s_VertexProgramDisc[i] = NULL;
            s_FragmentProgramDisc[i] = NULL;
            s_FragmentProgramTexture[i] = NULL;
            s_VertexProfile[i] = CG_PROFILE_UNKNOWN;
            s_FragmentProfile[i] = CG_PROFILE_UNKNOWN;
            s_FragmentProgramInverted[i] = NULL;
        }
    }
    if (!s_VertexProgram[context] || !s_FragmentProgram[context]
        || !s_FragmentProgramParticleCloud[context]
        || !s_VertexProgramDisc[context] || !s_FragmentProgramDisc[context]
        || !s_FragmentProgramTexture[context] || !s_FragmentProgramInverted[context])
    {
        s_FragmentProfile[context] = cgGLGetLatestProfile(CG_GL_FRAGMENT);
        s_VertexProfile[context] = cgGLGetLatestProfile(CG_GL_VERTEX);

        if (cgGLIsProfileSupported(CG_PROFILE_ARBVP1))
            s_VertexProfile[context] = CG_PROFILE_ARBVP1;
        if (cgGLIsProfileSupported(CG_PROFILE_ARBFP1))
            s_FragmentProfile[context] = CG_PROFILE_ARBFP1;

        if (s_FragmentProfile[context] == CG_PROFILE_UNKNOWN || s_VertexProfile[context] == CG_PROFILE_UNKNOWN)
        {
            printf("CG profile unknown\n");
            return false;
        }

        CGcontext m_CGcontext = cgCreateContext();

        char *pCovisePath = getenv("COVISEDIR");

        string vertexProgramFile = pCovisePath;
        vertexProgramFile += "/share/covise/materials/Cg/billboard_spheres_vertex.cg";
        s_VertexProgram[context] = cgCreateProgramFromFile(m_CGcontext, CG_SOURCE,
                                                           vertexProgramFile.c_str(),
                                                           s_VertexProfile[context], 0, 0);
        if (s_VertexProgram[context] == NULL)
        {
            CGerror Error = cgGetError();
            printf("CGError in loading vertex shader program: %s\n", cgGetErrorString(Error));
            return false;
        }
        cgGLLoadProgram(s_VertexProgram[context]);
        s_CGVertexParam_modelViewProj[context] = cgGetNamedParameter(s_VertexProgram[context], "modelViewProj");
        s_CGVertexParam_modelView[context] = cgGetNamedParameter(s_VertexProgram[context], "modelView");
        s_CGVertexParam_modelViewIT[context] = cgGetNamedParameter(s_VertexProgram[context], "modelViewIT");
        s_CGVertexParam_lightPos[context] = cgGetNamedParameter(s_VertexProgram[context], "lightPos");
        s_CGVertexParam_viewerPos[context] = cgGetNamedParameter(s_VertexProgram[context], "viewerPos");

        vertexProgramFile = pCovisePath;
        vertexProgramFile += "/share/covise/materials/Cg/billboard_spheres_disc_vertex.cg";
        s_VertexProgramDisc[context] = cgCreateProgramFromFile(m_CGcontext, CG_SOURCE,
                                                               vertexProgramFile.c_str(),
                                                               s_VertexProfile[context], 0, 0);
        if (s_VertexProgram[context] == NULL)
        {
            CGerror Error = cgGetError();
            printf("CGError in loading vertex shader program: %s\n", cgGetErrorString(Error));
            return false;
        }
        cgGLLoadProgram(s_VertexProgramDisc[context]);
        s_CGVertexParamDisc_modelViewProj[context] = cgGetNamedParameter(s_VertexProgramDisc[context], "modelViewProj");
        s_CGVertexParamDisc_modelView[context] = cgGetNamedParameter(s_VertexProgramDisc[context], "modelView");
        s_CGVertexParamDisc_modelViewIT[context] = cgGetNamedParameter(s_VertexProgramDisc[context], "modelViewIT");
        s_CGVertexParamDisc_lightPos[context] = cgGetNamedParameter(s_VertexProgramDisc[context], "lightPos");
        s_CGVertexParamDisc_viewerPos[context] = cgGetNamedParameter(s_VertexProgramDisc[context], "viewerPos");

        string fragmentProgramFile = pCovisePath;
        fragmentProgramFile += "/share/covise/materials/Cg/billboard_spheres_fragment.cg";
        s_FragmentProgram[context] = cgCreateProgramFromFile(m_CGcontext, CG_SOURCE,
                                                             fragmentProgramFile.c_str(),
                                                             s_FragmentProfile[context], NULL, NULL);
        if (s_FragmentProgram[context] == NULL)
        {
            CGerror Error = cgGetError();
            printf("CGError in loading fragment shader program %s: %s\n",
                   fragmentProgramFile.c_str(),
                   cgGetErrorString(Error));
            return false;
        }
        cgGLLoadProgram(s_FragmentProgram[context]);
        s_CGFragParam_glAmbient[context] = cgGetNamedParameter(s_FragmentProgram[context], "lightAmbient");
        s_CGFragParam_glDiffuse[context] = cgGetNamedParameter(s_FragmentProgram[context], "lightDiffuse");
        s_CGFragParam_glSpecular[context] = cgGetNamedParameter(s_FragmentProgram[context], "lightSpecular");

        fragmentProgramFile = pCovisePath;
        fragmentProgramFile += "/share/covise/materials/Cg/billboard_spheres_fragment_inverted.cg";
        s_FragmentProgramInverted[context] = cgCreateProgramFromFile(m_CGcontext, CG_SOURCE,
                                                                     fragmentProgramFile.c_str(),
                                                                     s_FragmentProfile[context], NULL, NULL);
        if (s_FragmentProgramInverted[context] == NULL)
        {
            CGerror Error = cgGetError();
            printf("CGError in loading fragment shader program %s: %s\n",
                   fragmentProgramFile.c_str(),
                   cgGetErrorString(Error));
            return false;
        }
        cgGLLoadProgram(s_FragmentProgramInverted[context]);
        s_CGFragParamInverted_glAmbient[context] = cgGetNamedParameter(s_FragmentProgramInverted[context], "lightAmbient");
        s_CGFragParamInverted_glDiffuse[context] = cgGetNamedParameter(s_FragmentProgramInverted[context], "lightDiffuse");
        s_CGFragParamInverted_glSpecular[context] = cgGetNamedParameter(s_FragmentProgramInverted[context], "lightSpecular");

        fragmentProgramFile = pCovisePath;
        fragmentProgramFile += "/share/covise/materials/Cg/billboard_spheres_particle_cloud_fragment.cg";
        s_FragmentProgramParticleCloud[context] = cgCreateProgramFromFile(m_CGcontext, CG_SOURCE,
                                                                          fragmentProgramFile.c_str(),
                                                                          s_FragmentProfile[context], NULL, NULL);
        if (s_FragmentProgramParticleCloud[context] == NULL)
        {
            CGerror Error = cgGetError();
            printf("CGError in loading fragment shader program %s: %s\n",
                   fragmentProgramFile.c_str(),
                   cgGetErrorString(Error));
            return false;
        }
        cgGLLoadProgram(s_FragmentProgramParticleCloud[context]);

        fragmentProgramFile = pCovisePath;
        fragmentProgramFile += "/share/covise/materials/Cg/billboard_spheres_disc_fragment.cg";
        s_FragmentProgramDisc[context] = cgCreateProgramFromFile(m_CGcontext, CG_SOURCE,
                                                                 fragmentProgramFile.c_str(),
                                                                 s_FragmentProfile[context], NULL, NULL);
        if (s_FragmentProgramDisc[context] == NULL)
        {
            CGerror Error = cgGetError();
            printf("CGError in loading fragment shader program %s: %s\n",
                   fragmentProgramFile.c_str(),
                   cgGetErrorString(Error));
            return false;
        }
        cgGLLoadProgram(s_FragmentProgramDisc[context]);
        s_CGFragParamDisc_glAmbient[context] = cgGetNamedParameter(s_FragmentProgramDisc[context], "lightAmbient");
        s_CGFragParamDisc_glDiffuse[context] = cgGetNamedParameter(s_FragmentProgramDisc[context], "lightDiffuse");
        s_CGFragParamDisc_glSpecular[context] = cgGetNamedParameter(s_FragmentProgramDisc[context], "lightSpecular");

        fragmentProgramFile = pCovisePath;
        fragmentProgramFile += "/share/covise/materials/Cg/billboard_spheres_texture_fragment.cg";
        s_FragmentProgramTexture[context] = cgCreateProgramFromFile(m_CGcontext, CG_SOURCE,
                                                                    fragmentProgramFile.c_str(),
                                                                    s_FragmentProfile[context], NULL, NULL);
        if (s_FragmentProgramTexture[context] == NULL)
        {
            CGerror Error = cgGetError();
            printf("CGError in loading fragment shader program %s: %s\n",
                   fragmentProgramFile.c_str(),
                   cgGetErrorString(Error));
            return false;
        }
        cgGLLoadProgram(s_FragmentProgramTexture[context]);
        s_CGFragParamTexture_glAmbient[context] = cgGetNamedParameter(s_FragmentProgramTexture[context], "lightAmbient");
        s_CGFragParamTexture_glDiffuse[context] = cgGetNamedParameter(s_FragmentProgramTexture[context], "lightDiffuse");
        s_CGFragParamTexture_glSpecular[context] = cgGetNamedParameter(s_FragmentProgramTexture[context], "lightSpecular");

        return true;
    }
#endif
    return false;
}

bool coSphere::bindProgramAndParams(int context) const
{
#ifndef HAVE_CG
    (void)context;
    return false;
#else
    // Set up for ball rendering
    cgGLEnableProfile(s_VertexProfile[context]);
    cgGLEnableProfile(s_FragmentProfile[context]);

    switch (m_renderMethod)
    {
    case RENDER_METHOD_CG_SHADER:
        cgGLBindProgram(s_VertexProgram[context]);
        cgGLBindProgram(s_FragmentProgram[context]);
        break;
    case RENDER_METHOD_PARTICLE_CLOUD:
        cgGLBindProgram(s_VertexProgram[context]);
        cgGLBindProgram(s_FragmentProgramParticleCloud[context]);
        break;
    case RENDER_METHOD_DISC:
        cgGLBindProgram(s_VertexProgramDisc[context]);
        cgGLBindProgram(s_FragmentProgramDisc[context]);
        break;
    case RENDER_METHOD_TEXTURE:
        cgGLBindProgram(s_VertexProgramDisc[context]);
        cgGLBindProgram(s_FragmentProgramTexture[context]);
        break;
    case RENDER_METHOD_CG_SHADER_INVERTED:
        cgGLBindProgram(s_VertexProgram[context]);
        cgGLBindProgram(s_FragmentProgramInverted[context]);
        break;
    default:
        ;
    }
    return true;
#endif
}

bool coSphere::unbindProgramAndParams(int context) const
{
#ifndef HAVE_CG
    (void)context;
    return false;
#else
    cgGLDisableProfile(s_VertexProfile[context]);
    cgGLDisableProfile(s_FragmentProfile[context]);
    return true;
#endif
}

void coSphere::bindMatrices(int context) const
{
#ifndef HAVE_CG
    (void)context;
#else
    if (m_renderMethod == RENDER_METHOD_DISC || m_renderMethod == RENDER_METHOD_TEXTURE)
    {
        cgGLSetStateMatrixParameter(s_CGVertexParamDisc_modelViewProj[context], CG_GL_MODELVIEW_PROJECTION_MATRIX, CG_GL_MATRIX_IDENTITY);
        cgGLSetStateMatrixParameter(s_CGVertexParamDisc_modelView[context], CG_GL_MODELVIEW_MATRIX, CG_GL_MATRIX_IDENTITY);
        cgGLSetStateMatrixParameter(s_CGVertexParamDisc_modelViewIT[context], CG_GL_MODELVIEW_MATRIX, CG_GL_MATRIX_INVERSE_TRANSPOSE);
    }
    else
    {
        cgGLSetStateMatrixParameter(s_CGVertexParam_modelViewProj[context], CG_GL_MODELVIEW_PROJECTION_MATRIX, CG_GL_MATRIX_IDENTITY);
        cgGLSetStateMatrixParameter(s_CGVertexParam_modelView[context], CG_GL_MODELVIEW_MATRIX, CG_GL_MATRIX_IDENTITY);
        cgGLSetStateMatrixParameter(s_CGVertexParam_modelViewIT[context], CG_GL_MODELVIEW_MATRIX, CG_GL_MATRIX_INVERSE_TRANSPOSE);
    }
#endif
}

}
