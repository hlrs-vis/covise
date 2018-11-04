/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
#define GL_GLEXT_LEGACY
#include "InvObjects.h"
#undef GL_VERSION_1_4
#undef GL_VERSION_1_3
#undef __glext_h
#include <sysdep/khronos-glext.h>

#ifndef APIENTRY
#define APIENTRY
#endif

#ifndef WITHOUT_VIRVO
#include <virvo/vvvecmath.h>
#include <virvo/vvdynlib.h> // for glXGetProcAddress replacement
#endif

#include <util/coLog.h>
#include <InvViewer.h>

//#define _USE_TEXTURE_LOADING_MS
//#define _USE_TEXTURE_LOADING_COVISE
#define _USE_MY_LOADING

using covise::print_comment;

#ifndef _USE_TEXTURE_LOADING_MS
typedef struct _AUX_RGBImageRec
{
    GLint sizeX, sizeY;
    unsigned char *data;
} AUX_RGBImageRec;
#else
#include <GL/glaux.h>
#endif

void(APIENTRYP f_glPointParameterfARB)(GLenum pname, GLfloat param) = NULL;
void(APIENTRYP f_glPointParameterfvARB)(GLenum pname, const GLfloat *param) = NULL;

inline void unpackRGBA(uint32_t *pc, int pos, float *r, float *g, float *b, float *a)
{
    unsigned char *chptr = (unsigned char *)&pc[pos];
    // RGBA switched 12.03.96 due to color bug in Inventor Renderer
    // D. Rantzau
    *r = ((float)(*chptr)) / 255.0;
    chptr++;
    *g = ((float)(*chptr)) / 255.0;
    chptr++;
    *b = ((float)(*chptr)) / 255.0;
    chptr++;
    *a = ((float)(*chptr)) / 255.0;
}

int smoothNormalsEnabled = 0;
#define CREASEANGLE 1.2

///////////////////////////////////////////////////////////////////////////////////
//
// class InvGeoObj
//
///////////////////////////////////////////////////////////////////////////////////
InvGeoObj::InvGeoObj(int colorpacking)
    : colPack_(colorpacking)
    , colBind_(INV_OVERALL)
    , geoShape_(NULL)
    , normbind_(NULL)
{

    top_switch_ = new SoSwitch;
    root_ = new SoSeparator;
    objName_ = new SoLabel;
    rObjName_ = new SoLabel;
    transform_ = new SoTransform;
    drawstyle_ = new SoDrawStyle;
    normbind_ = new SoNormalBinding;
    normal_ = new SoNormal;
    matbind = new SoMaterialBinding;
    material_ = new SoMaterial;
    geoGrp_ = new SoGroup;

    top_switch_->addChild(root_);
    root_->addChild(objName_);
    root_->addChild(transform_);
    root_->addChild(drawstyle_);
    root_->addChild(normbind_);
    root_->addChild(normal_);
    root_->addChild(matbind);
    root_->addChild(material_);

    geoGrp_->addChild(rObjName_);

    top_switch_->whichChild.setValue(0);
}

// set obj. name
void
InvGeoObj::setName(const char *name)
{
    gName_ = new char[strlen(name) + 3];
    strcpy(gName_, "G_");
    strcat(gName_, name);

    if (geoShape_)
    {
        geoShape_->setName(SbName(gName_));
    }

    strcpy(gName_, "S_");
    strcat(gName_, name);

    top_switch_->setName(SbName(gName_));
    objName_->label.setValue(name);
}

void
InvGeoObj::setRealObjName(const char *rn)
{
    rObjName_->setName(SbName(rn));
    geoGrp_->setName(SbName(rn));
}

void
InvGeoObj::setGrpLabel(const char *lb)
{

    rObjName_->label.setValue(lb);
    //cerr << "InvGeoObj::setGrpLabel(..) set label to " << lb << endl;
}

void
InvGeoObj::setNormals(int no_of_normals, float *nx, float *ny, float *nz)
{
    float *n;
    int k, j;

    //
    // store normals
    //
    n = new float[no_of_normals * 3];
    if (n != NULL)
    {
        k = 0;
        for (j = 0; j < no_of_normals; j++)
        {
            *(n + k) = *(nx + j);
            *(n + k + 1) = *(ny + j);
            *(n + k + 2) = *(nz + j);
            k = k + 3;
        }
        normal_->vector.setValues(0, no_of_normals, (float(*)[3])n);
        delete[] n;
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
}

// get top node of geometry obj
SoGroup *
InvGeoObj::getTopNode()
{
    return top_switch_;
}

// return root_ separator
SoSeparator *
InvGeoObj::getSeparator()
{
    return root_;
}

// return transformation node
SoTransform *
InvGeoObj::getTransform()
{
    return transform_;
}

SoDrawStyle *
InvGeoObj::getDrawStyle()
{
    return drawstyle_;
}

//==========================================================================
// set the colors of a geometric object
//==========================================================================
void
InvGeoObj::setColors(int no_of_colors, float *r, float *g, float *b)
{
    int k = 0;

    if (colPack_ == INV_NONE)
    {
        //
        // store colors
        //
        float *colors = new float[no_of_colors * 3];
        if (colors != NULL)
        {
            for (int j = 0; j < no_of_colors; j++)
            {
                *(colors + k) = r ? *(r + j) : 0.f;
                *(colors + k + 1) = g ? *(g + j) : 0.f;
                *(colors + k + 2) = b ? *(b + j) : 0.f;
                k = k + 3;
            }
            material_->diffuseColor.setValues(0, no_of_colors, (float(*)[3])colors);
            delete[] colors;
        }
        else
        {
            print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
        }
    }
}

//==========================================================================
// set the colors of a geometric object (packed rgba)
//==========================================================================
void
InvGeoObj::setColors(int no_of_colors, uint32_t *pc)
{

    if (colPack_ == INV_RGBA && pc)
    {
        float *c = new float[no_of_colors * 3];
        float *t = new float[no_of_colors];
        if (c != NULL)
        {
            int k = 0;
            for (int j = 0; j < no_of_colors; j++)
            {
                unpackRGBA(pc, j, c + k, c + k + 1, (c + k + 2), t + j);

                t[j] = 1.0 - t[j];
                k = k + 3;
            }
            material_->diffuseColor.setValues(0, no_of_colors, (float(*)[3])c);
            material_->transparency.setValues(0, no_of_colors, t);
            delete[] c;
            delete[] t;
        }
        else
        {
            print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
        }
    }
}

//=========================================================================
// set the material
//=========================================================================
void
InvGeoObj::setMaterial(covise::coMaterial *m)
{

    material_->transparency.setValue(m->transparency);
    material_->shininess.setValue(m->shininess);
    material_->ambientColor.setValue(m->ambientColor);
    material_->diffuseColor.setValue(m->diffuseColor);
    material_->specularColor.setValue(m->specularColor);
    material_->emissiveColor.setValue(m->emissiveColor);
}

//=========================================================================
// set the colorbinding of the object
//=========================================================================
void
InvGeoObj::setColorBinding(const int &type)
{
    colBind_ = type;

    if (colPack_ != INV_TEXTURE)
    {

        if (type == INV_PER_VERTEX)
            matbind->value = SoMaterialBinding::PER_VERTEX_INDEXED;
        else if (type == INV_PER_FACE)
            matbind->value = SoMaterialBinding::PER_FACE;
        else if (type == INV_NONE)
            matbind->value = SoMaterialBinding::OVERALL;
        else if (type == INV_OVERALL)
            matbind->value = SoMaterialBinding::OVERALL;
    }
}

//=========================================================================
// set transparency of geom obj.
//=========================================================================
void
InvGeoObj::setTransparency(const float &transparency)
{
    if (colPack_ == INV_NONE)
    {
        material_->transparency.setValue(transparency);
    }
}

//=========================================================================
// set normal binding (for line, polyg. and quadmesh)
//=========================================================================
void
InvGeoObj::setNormalBinding(const int &type)
{
    if (normbind_)
    {
        if (type == INV_PER_VERTEX)
        {
            normbind_->value = SoNormalBinding::PER_VERTEX_INDEXED;
        }
        else if (type == INV_PER_FACE)
        {
            normbind_->value = SoNormalBinding::PER_FACE;
        }
        else
        {
            normbind_->value = SoNormalBinding::DEFAULT;
        }
    }
}

InvGeoObj::~InvGeoObj()
{
    delete gName_;
}

bool InvSphere::loadTexture(const char *pFilename, int iTextureMode, int colorR, int colorG, int colorB)
{
    bool bSuccess = false;

    if (iTextureMode == 0)
    {
//
// Load up the texture...
//
#ifdef _USE_TEXTURE_LOADING_MS
        AUX_RGBImageRec *pImage_RGB = auxDIBImageLoad(pFilename);
#elif defined(_USE_MY_LOADING)
        BMPImage textureImage;

        getBitmapImageData(pFilename, &textureImage);
        AUX_RGBImageRec *pImage_RGB = new AUX_RGBImageRec;
        pImage_RGB->sizeX = textureImage.width;
        pImage_RGB->sizeY = textureImage.height;
        pImage_RGB->data = textureImage.data;
#elif defined(_USE_TEXTURE_LOADING_COVISE)
        //doesnot work
        //Windows: failure in PNG library routines!?
        AUX_RGBImageRec *pImage_RGB;

        coImage *pImage_RGB_tmp = NULL;
        pImage_RGB_tmp = new coImage(pFilename);

        pImage_RGB->data = (unsigned char *)(pImage_RGB_tmp->getBitmap(pImage_RGB_tmp->getNumFrames()));
        pImage_RGB->sizeX = pImage_RGB_tmp->getWidth();
        pImage_RGB->sizeY = pImage_RGB_tmp->getHeight();
#endif
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

            glGenTextures(1, &m_textureID);
            glBindTexture(GL_TEXTURE_2D, m_textureID);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            // Don't forget to use GL_RGBA for our new image data... we support Alpha transparency now!
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, pImage_RGB->sizeX, pImage_RGB->sizeY, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, pImage_RGBA);

            bSuccess = true;
        }
        else
        {
            fprintf(stderr, "Texture file %s not found.\n", pFilename);

            bSuccess = false;
        }

        if (pImage_RGB)
        {
            if (pImage_RGB->data)
                free(pImage_RGB->data);

            free(pImage_RGB);
        }

        if (pImage_RGBA)
            delete[] pImage_RGBA;
    }
    else if (iTextureMode == 1)
    {
//
// Load up the texture...
//

#ifdef _USE_TEXTURE_LOADING_MS
        AUX_RGBImageRec *pTextureImage = auxDIBImageLoad(m_chTexFile);
#elif defined(_USE_MY_LOADING)
        BMPImage textureImage;

        getBitmapImageData(pFilename, &textureImage);
        AUX_RGBImageRec *pTextureImage = new AUX_RGBImageRec;
        pTextureImage->sizeX = textureImage.width;
        pTextureImage->sizeY = textureImage.height;
        memcpy(pTextureImage->data, textureImage.data, sizeof(textureImage.data[0]) * textureImage.width * textureImage.height);
#elif defined(_USE_TEXTURE_LOADING_COVISE)
        AUX_RGBImageRec *pTextureImage; // = auxDIBImageLoad( pFilename );

        coImage *pImage_RGB_tmp = NULL;
        pImage_RGB_tmp = new coImage(pFilename);

        pTextureImage->data = const_cast<unsigned char *>(pImage_RGB_tmp->getBitmap());
        pTextureImage->sizeX = pImage_RGB_tmp->getWidth();
        pTextureImage->sizeY = pImage_RGB_tmp->getHeight();
#endif

        if (pTextureImage != NULL)
        {
            glGenTextures(1, &m_textureID);

            glBindTexture(GL_TEXTURE_2D, m_textureID);

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
            if (pTextureImage->data)
                free(pTextureImage->data);

            free(pTextureImage);
        }
    }

    return bSuccess;
}

void InvSphere::setViewer(InvViewer *pViewer)
{
    if (!m_pViewer)
    {
        m_pViewer = pViewer;
    }
}

InvViewer *InvSphere::getViewer()
{
    return m_pViewer;
}

//=====================================================
// sphere stuff
//=====================================================
void InvSphere::Render()
{
    //list<VF3>::iterator itColor;

    //   COpenGLDebug pDebugOgl;
    if (this->m_pViewer)
    {
#ifdef INVENTORRENDERER
        m_renderMethod = RENDER_METHOD_MANUAL_CPU_BILLBOARDS;
        m_bBlendOutColorKey = false;
#else
        m_renderMethod = (InvSphere::RENDER_METHOD)m_pViewer->getBillboardRenderingMethod();
        m_bBlendOutColorKey = m_pViewer->getBillboardRenderingBlending();
#endif
    }

    glPushMatrix();
    glPushAttrib(GL_ENABLE_BIT);
    glPushAttrib(GL_COLOR_BUFFER_BIT);

    glEnable(GL_TEXTURE_2D);
    //glEnable(GL_COLOR_MATERIAL);
    //glEnable(GL_LIGHTING);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
    glShadeModel(GL_SMOOTH);
    glEnable(GL_NORMALIZE);

    //
    // Enabling GL_DEPTH_TEST and setting glDepthMask to GL_FALSE makes the
    // Z-Buffer read-only, which helps remove graphical artifacts generated
    // from  rendering a list of particles that haven't been sorted by
    // distance to the eye.
    //
    // Enabling GL_BLEND and setting glBlendFunc to GL_DST_ALPHA with GL_ONE
    // allows particles, which overlap, to alpha blend with each other
    // correctly.
    //

    glEnable(GL_DEPTH_TEST);

    glEnable(GL_BLEND);
    if (m_bBlendOutColorKey == true)
    {
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_ALPHA_TEST);
        glDepthFunc(GL_LEQUAL);
        glAlphaFunc(GL_GREATER, 0);
    }
    else
    {
        glDepthMask(GL_FALSE);
        glBlendFunc(GL_DST_ALPHA, GL_ONE);
    }

// here we go

#ifdef WITHOUT_VIRVO
    std::cerr << "sphere rendering requires Virvo" << std::endl;
#else
    if (m_renderMethod == RENDER_METHOD_MANUAL_CPU_BILLBOARDS || m_renderMethod == RENDER_METHOD_ARB_POINT_SPRITES || m_renderMethod == RENDER_METHOD_CG_VERTEX_SHADER)
    {
        // Recompute the quad points with respect to the view matrix
        // and the quad's center point. The quad's center is always
        // at the origin...

        float mat[16];
        glGetFloatv(GL_MODELVIEW_MATRIX, mat);

        vvVector3 vRight(mat[0], mat[4], mat[8]);
        vvVector3 vUp(mat[1], mat[5], mat[9]);
        vvVector3 vPoint0;
        vvVector3 vPoint1;
        vvVector3 vPoint2;
        vvVector3 vPoint3;
        vvVector3 vCenter;

        float fAdjustedSize = 0.66f;

        glBindTexture(GL_TEXTURE_2D, m_textureID);

        // Render each particle...
        list<VF3>::iterator itCoord;
        list<VF3>::iterator itColor;

        int i = 0;
        for (itCoord = coord.begin(), itColor = m_vf3Color.begin();
             itCoord != coord.end() && itColor != m_vf3Color.end();
             itCoord++, itColor++, i++)
        {
            glBegin(GL_QUADS);
            {
                vCenter[0] = (*itCoord)[0];
                vCenter[1] = (*itCoord)[1];
                vCenter[2] = (*itCoord)[2];
                glColor3f((*itColor)[0], (*itColor)[1], (*itColor)[2]);

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

                vPoint0 = vCenter + ((-vRight - vUp) * fAdjustedSize * m_pfRadii[i]);
                vPoint1 = vCenter + ((vRight - vUp) * fAdjustedSize * m_pfRadii[i]);
                vPoint2 = vCenter + ((vRight + vUp) * fAdjustedSize * m_pfRadii[i]);
                vPoint3 = vCenter + ((-vRight + vUp) * fAdjustedSize * m_pfRadii[i]);

                glTexCoord2f(0.0f, 0.0f);
                glVertex3f(vPoint0[0], vPoint0[1], vPoint0[2]);

                glTexCoord2f(1.0f, 0.0f);
                glVertex3f(vPoint1[0], vPoint1[1], vPoint1[2]);

                glTexCoord2f(1.0f, 1.0f);
                glVertex3f(vPoint2[0], vPoint2[1], vPoint2[2]);

                glTexCoord2f(0.0f, 1.0f);
                glVertex3f(vPoint3[0], vPoint3[1], vPoint3[2]);
            }
            glEnd();
        }
    }
#endif

    //
    // Compute billboard vertices on the GPU using ARB_point_sprites...
    //

    /*   if( m_renderMethod == RENDER_METHOD_ARB_POINT_SPRITES )
   {
      		this->m_bBlendOutColorKey=false;
                        loadTexture(m_chTexFile,1);
  
      // This is how our point sprite's size will be modified by
      // distance from the viewer.
      float quadratic[] =  { 0.5f, .5f, .5f };
      f_glPointParameterfvARB( GL_POINT_DISTANCE_ATTENUATION_ARB, quadratic );

      // The alpha of a point is calculated to allow the fading of points
      // instead of shrinking them past a defined threshold size. The threshold
      // is defined by GL_POINT_FADE_THRESHOLD_SIZE_ARB and is not clamped to
      // the minimum and maximum point sizes.
      f_glPointParameterfARB( GL_POINT_FADE_THRESHOLD_SIZE_ARB, 60.0f );

      float fAdjustedSize = 100.0f;                //m_fSize / 4.0f;

      f_glPointParameterfARB( GL_POINT_SIZE_MIN_ARB, 12.0f );
      f_glPointParameterfARB( GL_POINT_SIZE_MAX_ARB, fAdjustedSize );


      // Specify point sprite texture coordinate replacement mode for each texture unit
      glTexEnvf( GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE );
      //	vvGLTools::printGLError("5");

      //
      // Render point sprites...
      //

      glBindTexture( GL_TEXTURE_2D, m_textureID );

      // Render each particle...
      list<VF3>::iterator itCoord;
      list<VF3>::iterator itColor;

      int i=0;
      glEnable( GL_POINT_SPRITE_ARB );
      //      glBegin( GL_POINTS );
      for (itCoord = coord.begin(),itColor = m_vf3Color.begin();
            itCoord != coord.end() ;
            itCoord++,itColor++,i++)
      {
         glPointSize( m_pfRadii[i]*fAdjustedSize );
         glBegin( GL_POINTS );
         {
            glColor3f((*itColor)[0],(*itColor)[1],(*itColor)[2]);
            glVertex3f( (*itCoord)[0],
                  (*itCoord)[1],
                  (*itCoord)[2] );
         }
         glEnd();
      }
      i=0;
      //      glEnd();

      //	pDebugOgl.DebugOglError();
      glDisable( GL_POINT_SPRITE_ARB );
   }

   //
   // Compute billboard vertices on the GPU using a Cg based shader...
   //

#ifdef HAVE_CG
   if( m_renderMethod == RENDER_METHOD_CG_VERTEX_SHADER )
   {
      cgGLSetStateMatrixParameter( m_CGparam_modelViewProj,
            CG_GL_MODELVIEW_PROJECTION_MATRIX,
            CG_GL_MATRIX_IDENTITY);
      cgGLSetStateMatrixParameter( m_CGparam_modelView,
            CG_GL_MODELVIEW_MATRIX,
            CG_GL_MATRIX_IDENTITY);

      cgGLBindProgram( m_CGprogram );
      cgGLEnableProfile( m_CGprofile );

      glBindTexture( GL_TEXTURE_2D, m_textureID );
      float fAdjustedSize = 1.00f;                  //m_fSize / 300.0f;

      int i=0;
      list<VF3>::iterator itCoord;
      list<VF3>::iterator itColor;

      for (itCoord = coord.begin(),itColor = m_vf3Color.begin();
            itCoord != coord.end(), itColor != m_vf3Color.end();
            itCoord++,itColor++,i++)
      {
         glBegin( GL_QUADS );
         {
            float x, y, z;
            x = (*itCoord)[0];
            y = (*itCoord)[1];
            z = (*itCoord)[2];

            glColor3f((*itColor)[0],(*itColor)[1],(*itColor)[2]);

            glTexCoord3f( 0.0f,  0.0f, m_pfRadii[i]*fAdjustedSize );
            glVertex3f( x, y, z );

            glTexCoord3f( 1.0f,  0.0f, m_pfRadii[i]*fAdjustedSize );
            glVertex3f( x, y, z );

            glTexCoord3f( 1.0f,  1.0f, m_pfRadii[i]*fAdjustedSize );
            glVertex3f( x, y, z );

            glTexCoord3f( 0.0f,  1.0f, m_pfRadii[i]*fAdjustedSize );
            glVertex3f( x, y, z );

         }
         glEnd();
      }
   }
#endif
*/
    //
    // Reset OpenGL states...
    //

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

    //
    // Report frames per second and the number of objects culled...
    //
    /*   static char billboardMethodString[50];

        if( m_renderMethod == RENDER_METHOD_MANUAL_CPU_BILLBOARDS )
        sprintf( billboardMethodString, "Billboard Method: Manual billboards on the CPU"  );
        else if( m_renderMethod == RENDER_METHOD_ARB_POINT_SPRITES )
        sprintf( billboardMethodString, "Billboard Method: GL_ARB_point_sprite extension"  );
        else if( m_renderMethod == RENDER_METHOD_CG_VERTEX_SHADER )
        sprintf( billboardMethodString, "Billboard Method: Cg shader billboards" );

        strcpy ((m_pViewer->vInfoText[0]).pInfoText,billboardMethodString);
    */
    /* 
      beginRenderText( 680, 480 );
      {
      glColor3f( 1.0f, 1.0f, 1.0f );
      renderText( 5, 15, GLUT_BITMAP_HELVETICA_12, billboardMethodString );
      renderText( 5, 30, GLUT_BITMAP_HELVETICA_12, fpsString );
      }
      endRenderText();
    */

    glPopAttrib(); // GL_COLOR_BUFFER_BIT
    glPopAttrib(); // GL_ENABLE_BIT
    glPopMatrix();
}

void InvSphere::RenderCallback(void *d, SoAction *action)
{

    if (action->isOfType(SoGLRenderAction::getClassTypeId()))
    {
        // Make my custom GL calls
        ((InvSphere *)d)->Render();

        // Invalidate the state so that a cache is not made
        SoCacheElement::invalidate(action->getState());
    }
}

InvSphere::InvSphere(int colorpacking)
    : InvGeoObj(colorpacking)
    , empty_(true)
{
    coord.resize(0);
    cbfunction = NULL;

    m_pfRadii = NULL;

    m_textureID = 0;
    m_renderMethod = RENDER_METHOD_ARB_POINT_SPRITES;

    m_fSize = 1.0f;
    m_bBlendOutColorKey = true;

    m_fMaxPointSize = 0.0f;
    m_chTexFile = NULL;
    const char *pCovisePath;
    char pFile[256];
    pCovisePath = getenv("COVISEDIR");
#ifdef WIN32
    sprintf(pFile, "%s\\share\\covise\\materials\\textures\\%s", pCovisePath, "particle.bmp");
#else
    sprintf(pFile, "%s/share/covise/materials/textures/%s", pCovisePath, "particle.bmp");
#endif
    SetTexture(pFile);

    m_pViewer = NULL;
}

void InvSphere::EnableBlendOutColorKey(bool bBlendOutColorKey)
{
    m_bBlendOutColorKey = bBlendOutColorKey;
}

bool InvSphere::IsEnabledBlendOutColorKey()
{
    return m_bBlendOutColorKey;
}

//=========================================================================
// initialize the InvSphere instance
//=========================================================================
bool InvSphere::Init()
{
    cbfunction = new SoCallback();
    cbfunction->setCallback(InvSphere::RenderCallback, (void *)this);
    root_->addChild(cbfunction);

    return InitBillboards();
}

void InvSphere::SetTexture(const char *chTexFile)
{
    if (empty_)
        return;
    // Deallocate the memory that was previously reserved for this string.
    if (m_chTexFile != NULL)
    {
        delete[] m_chTexFile;
        m_chTexFile = NULL;
    }

    // Dynamically allocate the correct amount of memory.
    m_chTexFile = new char[strlen(chTexFile) + 1];

    // If the allocation succeeds, copy the initialization string.
    if (m_chTexFile != NULL)
        strcpy(m_chTexFile, chTexFile);
}

bool InvSphere::InitBillboards()
{
    loadTexture(m_chTexFile, 0, 20, 20, 20);

    //
    // If the required extensions are present, get the addresses of thier
    // functions that we wish to use...
    //

    const char *ext = reinterpret_cast<const char *>(glGetString(GL_EXTENSIONS));

    if (strstr(ext, "GL_ARB_point_parameters") == NULL)
    {
        //        MessageBox(NULL,"GL_ARB_point_parameters extension was not found",
        //            "ERROR",MB_OK|MB_ICONEXCLAMATION);
        std::cerr << "GL_ARB_point_parameters extension was not found" << std::endl;
        return 0;
    }
    else
    {
#ifdef WIN32
        f_glPointParameterfARB = (PFNGLPOINTPARAMETERFARBPROC)wglGetProcAddress("glPointParameterfARB");
        f_glPointParameterfvARB = (PFNGLPOINTPARAMETERFVARBPROC)wglGetProcAddress("glPointParameterfvARB");
#else
#ifdef WITHOUT_VIRVO
        f_glPointParameterfARB = NULL;
        f_glPointParameterfvARB = NULL;
#else
        f_glPointParameterfARB = (void(APIENTRY *)(GLenum, GLfloat))vvDynLib::glSym("glPointParameterfARB");
        f_glPointParameterfvARB = (void(APIENTRY *)(GLenum, const GLfloat *))vvDynLib::glSym("glPointParameterfvARB");
#endif
#endif
        if (!f_glPointParameterfARB || !f_glPointParameterfvARB)
        {
            //            MessageBox(NULL,"One or more GL_ARB_point_parameters functions were not found",
            //                "ERROR",MB_OK|MB_ICONEXCLAMATION);
            std::cerr << "One or more GL_ARB_point_parameters functions were not found" << std::endl;
            return 0;
        }
    }

    //
    // If you want to know the max size that a point sprite can be set
    // to, do this.
    //

    glGetFloatv(GL_POINT_SIZE_MAX_ARB, &m_fMaxPointSize);
    glPointSize(m_fMaxPointSize);

//
// Init the Cg shader
//

//
// Search for a valid vertex shader profile in this order:
//
// CG_PROFILE_ARBVP1 - GL_ARB_vertex_program
// CG_PROFILE_VP30   - GL_NV_vertex_program2
// CG_PROFILE_VP20   - GL_NV_vertex_program
//
#if 0
#ifdef HAVE_CG
   if( cgGLIsProfileSupported(CG_PROFILE_ARBVP1) )
      m_CGprofile = CG_PROFILE_ARBVP1;
   else if( cgGLIsProfileSupported(CG_PROFILE_VP30) )
      m_CGprofile = CG_PROFILE_VP30;
   else if( cgGLIsProfileSupported(CG_PROFILE_VP20) )
      m_CGprofile = CG_PROFILE_VP20;
   else
   {
      std::cerr << "Failed to initialize vertex shader! Hardware doesn't support" << std::endl;
      std::cerr << "any of the vertex shading extensions!" << std::endl;
      return false;
   }

   // Create the context...
   m_CGcontext = cgCreateContext();

   //
   // Create the vertex shader...
   //

   char *pCovisePath, pFile[256];
   pCovisePath = getenv("COVISEDIR");
#ifdef WIN32
   sprintf (pFile, "%s\\CgPrograms\\%s", pCovisePath,  "cg_billboard_spheres.cg");
#else
   sprintf (pFile, "%s/CgPrograms/%s", pCovisePath, "cg_billboard_spheres.cg");
#endif

   m_CGprogram = cgCreateProgramFromFile( m_CGcontext,
      CG_SOURCE,
      pFile,
      m_CGprofile,
      NULL,
      NULL );

   //
   // Load the program using Cg's expanded interface...
   //

   
   cgGLLoadProgram( m_CGprogram );

   //
   // Bind some parameters by name so we can set them later...
   //

   m_CGparam_modelViewProj  = cgGetNamedParameter(m_CGprogram, "modelViewProj");
   m_CGparam_modelView     = cgGetNamedParameter(m_CGprogram, "modelView");
   //m_CGparam_preRotatedQuad = cgGetNamedParameter(m_CGprogram, "preRotatedQuad");
   //m_CGparam_size           = cgGetNamedParameter(m_CGprogram, "size");
   /*CGerror error = cgGetError();
   const char* errorstring = cgGetErrorString(error);
   printf("Fehler: %s\n", errorstring);*/
#else
   std::cout << "HAVE_CG is false." << std::endl;
#endif
#endif
    return 1;
}

//==================================================================
// set the coordinates of the point object
//=========================================================================
void
InvSphere::setCoords(const int no_of_points, float *x_c, float *y_c,
                     float *z_c)
{
    if (no_of_points > 0)
        empty_ = false;
    unsigned int j = 0;

    list<VF3>::iterator itCoord;
    list<VF3>::iterator itColor;

    //
    // store coordinates
    //
    if (no_of_points > 0)
    {
        m_iNoOfSpheres = no_of_points;

        if (coord.size() != m_iNoOfSpheres)
        {
            coord.resize(m_iNoOfSpheres);
            m_vf3Color.resize(m_iNoOfSpheres);
        }
        if (coord.size() != m_iNoOfSpheres)
            print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");

        j = 0;
        for (itCoord = coord.begin(), itColor = m_vf3Color.begin(); itCoord != coord.end() && itColor != m_vf3Color.end(); itCoord++, itColor++, j++)
        {
            (*itCoord).resize(3);
            (*itCoord)[0] = *(x_c + j);
            (*itCoord)[1] = *(y_c + j);
            (*itCoord)[2] = *(z_c + j);

            (*itColor).resize(3);
            (*itColor)[0] = 1.0f;
            (*itColor)[1] = .0f;
            (*itColor)[2] = .0f;
        }

        if (j != m_iNoOfSpheres)
            print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR: number of points not greater zero");
}

void InvSphere::setRadii(float *radii_c)
{
    m_pfRadii = radii_c;
}

void InvSphere::setRenderMethod(RENDER_METHOD rm)
{
    m_renderMethod = rm;
#ifndef INVENTORRENDERER
    m_pViewer->setBillboardRenderingMethod(rm);
#endif
}

void InvSphere::setColors(float *r, float *g, float *b)
{
    int j = 0;
    std::list<VF3>::iterator itColor;

    if (r != NULL && g != NULL && b != NULL)
        for (itColor = m_vf3Color.begin(); itColor != m_vf3Color.end(); itColor++, j++)
        {
            (*itColor).resize(3);
            (*itColor)[0] = r[j];
            (*itColor)[1] = g[j];
            (*itColor)[2] = b[j];
        }
}

void InvSphere::setColors(uint32_t *pc)
{

    if (colPack_ == INV_RGBA && pc)
    {
        float dummy;
        int j = 0;
        std::list<VF3>::iterator itColor;
        for (itColor = m_vf3Color.begin(); itColor != m_vf3Color.end(); itColor++, j++)
        {
            (*itColor).resize(3);
            unpackRGBA(pc, j, &(*itColor)[0], &(*itColor)[1], &(*itColor)[2], &dummy);
        }
    }
}

InvSphere::~InvSphere()
{
    if (cbfunction)
    {
        root_->removeChild(cbfunction);
    }
    glDeleteTextures(1, &m_textureID);
}

//=====================================================
// point stuff
//=====================================================
InvPoint::InvPoint(int colorpacking)
    : InvGeoObj(colorpacking)
{
    /// --- these parts are common to all points

    coord = new SoCoordinate3;
    lightmodel = new SoLightModel;
    geoShape_ = new SoPointSet;

    root_->addChild(coord);
    root_->addChild(lightmodel);
    //root_->addChild(geoShape_);

    defaultColor[0] = 1.;
    defaultColor[1] = 0.;
    defaultColor[2] = 0.;
    material_->diffuseColor.set1Value(0, defaultColor);

    geoGrp_->addChild(geoShape_);
    root_->addChild(geoGrp_);
}

//=========================================================================
// set point size hint
//=========================================================================
void InvPoint::setSize(float pointsize)
{
    getDrawStyle()->pointSize.setValue(pointsize);
}

//=========================================================================
// set the coordinates of the point object
//=========================================================================
void
InvPoint::setCoords(int no_of_points, float *x_c, float *y_c,
                    float *z_c)
{
    int k = 0;
    //
    // store coordinates
    //
    float *coord_points = new float[no_of_points * 3];
    if (coord_points != NULL)
    {
        for (int j = 0; j < no_of_points; j++)
        {
            *(coord_points + k) = *(x_c + j);
            *(coord_points + k + 1) = *(y_c + j);
            *(coord_points + k + 2) = *(z_c + j);
            k = k + 3;
        }
        coord->point.setValues(0, no_of_points, (float(*)[3])coord_points);
        delete[] coord_points;
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
}

///////////////////////////////////////////////////////////////////////////////////
//
// class InvLine
//
///////////////////////////////////////////////////////////////////////////////////
InvLine::InvLine(int colorpacking)
    : InvGeoObj(colorpacking)

{
    //
    // allocate nodes for inventor stuff
    coord = new SoCoordinate3;
    geoShape_ = new SoIndexedLineSet;

    //
    // create object tree
    //
    root_->addChild(coord);

    //
    // default settings
    //
    top_switch_->whichChild.setValue(0);
    drawstyle_->lineWidth.setValue(2);
    matbind->value = SoMaterialBinding::OVERALL;
    normbind_->value = SoNormalBinding::DEFAULT;
    defaultColor[0] = 1.;
    defaultColor[1] = 0.;
    defaultColor[2] = 1.;
    material_->diffuseColor.setValues(0, 1, (const float(*)[3])defaultColor);
    defaultColor[0] = 0.;
    defaultColor[1] = 0.;
    defaultColor[2] = 0.;
    material_->emissiveColor.setValues(0, 1, (float(*)[3])defaultColor);
    material_->ambientColor.setValues(0, 1, (float(*)[3])defaultColor);
    material_->transparency.setValues(0, 1, defaultColor);

    geoGrp_->addChild(geoShape_);
    root_->addChild(geoGrp_);
}

//=========================================================================
// set the coordinates for the line object
//=========================================================================
void
InvLine::setCoords(int no_of_lines, int no_of_vertices,
                   int no_of_coords, float *x_c, float *y_c, float *z_c,
                   int *vertex_list, int *index_list)
{
    long no_lines;
    long j, k;
    float *coord_points;
    int32_t *vertices;

    coord_points = new float[no_of_coords * 3];
    vertices = new int32_t[no_of_vertices + no_of_lines];

    // we do nothing if we have an empty line
    if ((no_of_lines) && (no_of_vertices) && (no_of_coords))
    {

        if (coord_points != NULL && vertices != NULL)
        {
            k = 0;
            for (j = 0; j < no_of_coords; j++)
            {
                *(coord_points + k) = *(x_c + j);
                *(coord_points + k + 1) = *(y_c + j);
                *(coord_points + k + 2) = *(z_c + j);
                k = k + 3;
            }
            coord->point.setValues(0, no_of_coords, (float(*)[3])coord_points);

            //
            // store vertices list ( used by normal and coordinate field )
            //
            j = 0;
            for (no_lines = 0; no_lines < no_of_lines - 1; no_lines++)
            {
                for (k = *(index_list + no_lines); k < *(index_list + no_lines + 1); k++)
                {
                    *(vertices + j) = *(vertex_list + k);
                    //fprintf(stderr, "%d ", *(vertices +j));
                    j++;
                }
                *(vertices + j) = SO_END_LINE_INDEX;
                //fprintf(stderr, "END: j=%d, v[j]=%d\n", j, *(vertices +j));
                j++;
            }
            // last line
            for (k = *(index_list + no_of_lines - 1); k < no_of_vertices; k++)
            {
                *(vertices + j) = *(vertex_list + k);
                j++;
            }
            if (no_of_vertices)
            {
                *(vertices + j) = SO_END_LINE_INDEX;
            }

            SoIndexedLineSet *lp = NULL;
            lp = (SoIndexedLineSet *)(geoShape_);
            if (lp)
            {
                lp->coordIndex.setValues(0, no_of_vertices + no_of_lines, vertices);
            }
            else
            {
                // this should never occur
                cerr << "FATAL error: InvLine::setCoords(..) dynamic cast failed in line "
                     << __LINE__ << "file " << __FILE__ << endl;
            }
            delete[] coord_points;
            delete[] vertices;
        }
        else
            print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
    }
}

//=========================================================================
//
//=========================================================================
void
InvLine::setDrawstyle(int)
{
}

//=========================================================================
//
//=========================================================================
InvPolygon::InvPolygon(int colorpacking)
    : InvGeoObj(colorpacking)
    , texture(NULL)
    , empty_(true)
{
    gType = UNDEF;

    geoShape_ = new SoIndexedFaceSet;
    coord = new SoCoordinate3;
    lightmodel = new SoLightModel;
    shapehints = new SoShapeHints;

    root_->addChild(material_);
    root_->addChild(matbind);
    root_->addChild(coord);
    root_->addChild(shapehints);
    root_->addChild(lightmodel);

    drawstyle_->style = SoDrawStyle::FILLED;
    matbind->value = SoMaterialBinding::OVERALL;
    normbind_->value = SoNormalBinding::DEFAULT;

    if (colorpacking == INV_TEXTURE)
    {
        // allocate nodes for inventor stuff
        texture = new SoTexture2;
        texCoord = new SoTextureCoordinate2;
        texbind = new SoTextureCoordinateBinding;

        // create object tree
        root_->addChild(texture);
        root_->addChild(texbind);
        root_->addChild(texCoord);
    }
    else
    {
        defaultColor[0] = 1.;
        defaultColor[1] = 1.;
        defaultColor[2] = 1.;
        material_->diffuseColor.setValues(0, 1, (const float(*)[3])defaultColor);
        defaultColor[0] = 0.;
        defaultColor[1] = 0.;
        defaultColor[2] = 0.;
        material_->emissiveColor.setValues(0, 1, (float(*)[3])defaultColor);
        material_->ambientColor.setValues(0, 1, (float(*)[3])defaultColor);
    }

    if (smoothNormalsEnabled)
    {
        shapehints->creaseAngle = (float)CREASEANGLE;
    }
    geoGrp_->addChild(geoShape_);
    root_->addChild(geoGrp_);
}

//=========================================================================
//
//=========================================================================
SoTexture2 *
InvPolygon::getTexture()
{
    //
    // return texture node of objects
    //
    return texture;
}

//=========================================================================
//
//=========================================================================
void
InvPolygon::setCoords(int no_of_polygons, int no_of_vertices,
                      int no_of_coords, float *x_c, float *y_c, float *z_c,
                      int *vertex_list, int *index_list)
{
    if ((no_of_polygons > 0) && (no_of_vertices > 0) && (no_of_coords > 0))
    {
        empty_ = false;
    }

    int no_poly;
    long j, k;
    float *coord_points;
    int32_t *vertices;

    //
    // store coordinates
    //
    coord_points = new float[no_of_coords * 3];
    vertices = new int32_t[no_of_vertices + no_of_polygons];

    if (coord_points != NULL && vertices != NULL)
    {
        k = 0;
        for (j = 0; j < no_of_coords; j++)
        {
            *(coord_points + k) = *(x_c + j);
            *(coord_points + k + 1) = *(y_c + j);
            *(coord_points + k + 2) = *(z_c + j);
            k = k + 3;
        }
        coord->point.setValues(0, no_of_coords, (float(*)[3])coord_points);

        // store vertices list
        //
        j = 0;
        for (no_poly = 0; no_poly < no_of_polygons - 1; no_poly++)
        {
            for (k = *(index_list + no_poly); k < *(index_list + no_poly + 1); k++)
            {
                *(vertices + j) = *(vertex_list + k);
                j++;
            }
            *(vertices + j) = SO_END_FACE_INDEX;
            j++;
        }
        // last polygon
        for (k = *(index_list + no_of_polygons - 1); k < no_of_vertices; k++)
        {
            *(vertices + j) = *(vertex_list + k);
            j++;
        }
        if (no_of_vertices)
        {
            *(vertices + j) = SO_END_FACE_INDEX;
        }

        SoIndexedFaceSet *pt = (SoIndexedFaceSet *)(geoShape_);
        if (pt)
        {
            pt->coordIndex.setValues(0, no_of_vertices + no_of_polygons, vertices);
        }
        else
        {
            // this should never occur
            cerr << "FATAL error: InvPolygon::setCoords(..) dynamic cast failed in line "
                 << __LINE__ << "file " << __FILE__ << endl;
        }

        //	polygon->coordIndex.setValues(0 ,no_of_vertices + no_of_polygons,vertices);
        delete[] coord_points;
        delete[] vertices;
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
}

//=========================================================================
// set texture for polygon
//=========================================================================
void InvPolygon::setTexture(int texWidth, int texHeight, int pixelSize, unsigned char *image)
{
    if (empty_)
        return;
    if (texture)
    {
        texture->image.setValue(SbVec2s(texWidth, texHeight), pixelSize, image);
        if (gType != MXI)
        {
            // Notice: clamping doesn't work correctly on MXI.
            texture->wrapS.setValue(SoTexture2::CLAMP);
            texture->wrapT.setValue(SoTexture2::CLAMP);
        }
        texture->model.setValue(SoTexture2::MODULATE);
    }
}

void InvPolygon::setTextureCoordinateBinding(int type)
{
    if (type == INV_PER_VERTEX)
        texbind->value = SoTextureCoordinateBinding::PER_VERTEX_INDEXED;
    else if (type == INV_NONE)
        texbind->value = SoTextureCoordinateBinding::DEFAULT;
}

//=========================================================================
//
//=========================================================================
void InvPolygon::setTexCoords(int no_of_texCoords, float *tx, float *ty)
{
    if (empty_)
        return;
    long j, k;
    float *tc;
    float texX;

    if (no_of_texCoords == 0)
    {
        texture = NULL;
        return;
    }
    //
    // store texture coordinates
    //
    tc = new float[no_of_texCoords * 2];
    if (tc != NULL)
    {
        k = 0;
        for (j = 0; j < no_of_texCoords; j++)
        {
            if (gType != MXI)
            {
                *(tc + k) = *(tx + j);
                *(tc + k + 1) = *(ty + j);
            }
            else
            {
                texX = *(tx + j);

                // Workaround
                // Notice: clamping doesn't work correctly on MXI.
                if (texX <= BORDER)
                    *(tc + k) = (float)BORDER;
                else if (texX >= 1.0f - BORDER)
                    *(tc + k) = 1.0f - (float)BORDER;
                else
                    *(tc + k) = texX;

                *(tc + k + 1) = *(ty + j);
            }

            k = k + 2;
        }
        texCoord->point.setValues(0, no_of_texCoords, (float(*)[2])tc);
        delete[] tc;
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
}

//=========================================================================
//
//=========================================================================
void InvPolygon::setDrawstyle(int)
{
}

//=========================================================================
//
//=========================================================================
void InvPolygon::setVertexOrdering(int ordering)
{
    if (empty_)
        return;
    switch (ordering)
    {
    case 0:
        shapehints->vertexOrdering = SoShapeHints::UNKNOWN_ORDERING;
        break;
    case 1:
        shapehints->vertexOrdering = SoShapeHints::CLOCKWISE;
        break;
    case 2:
        shapehints->vertexOrdering = SoShapeHints::COUNTERCLOCKWISE;
        break;
    }
}

//=========================================================================
//
//=========================================================================
InvTriangleStrip::InvTriangleStrip(int colorpacking, const char *rName)
    : InvGeoObj(colorpacking)
    , empty_(true)
{
    (void)rName;
    gType = UNDEF;

    geoShape_ = new SoIndexedTriangleStripSet;

    lightmodel = new SoLightModel;
    shapehints = new SoShapeHints;

    root_->addChild(shapehints);

    drawstyle_->style = SoDrawStyle::FILLED;
    matbind->value = SoMaterialBinding::OVERALL;

    if (smoothNormalsEnabled)
    {
        shapehints->creaseAngle = (float)CREASEANGLE;
    }

    if (colorpacking == INV_RGBA)
    {
        //
        // allocate nodes for inventor stuff
        coord = new SoCoordinate3;

        //
        // create object tree
        //
        root_->addChild(normbind_);
        root_->addChild(coord);
        normbind_->value = SoNormalBinding::DEFAULT;
    }
    else if (colorpacking == INV_TEXTURE)
    {
        //
        // allocate nodes for inventor stuff
        texture = new SoTexture2;
        //texCoord       = new SoTextureCoordinate2;
        //texbind        = new SoTextureCoordinateBinding;
        vertexp = new SoVertexProperty;

        //
        // create object tree
        //
        root_->addChild(texture);
        //root->addChild(texbind);
        root_->addChild(vertexp);
    }
    else
    {
        //
        // allocate nodes for inventor stuff
        coord = new SoCoordinate3;

        //
        // create object tree
        //
        root_->addChild(normbind_);
        root_->addChild(coord);

        normbind_->value = SoNormalBinding::DEFAULT;
        defaultColor[0] = 1.;
        defaultColor[1] = 1.;
        defaultColor[2] = 1.;
        material_->diffuseColor.setValues(0, 1, (const float(*)[3])defaultColor);
        defaultColor[0] = 0.;
        defaultColor[1] = 0.;
        defaultColor[2] = 0.;
        material_->emissiveColor.setValues(0, 1, (float(*)[3])defaultColor);
        material_->ambientColor.setValues(0, 1, (float(*)[3])defaultColor);
    }

    geoGrp_->addChild(geoShape_);
    root_->addChild(geoGrp_);
}

//=========================================================================
//
//=========================================================================
SoTexture2 *
InvTriangleStrip::getTexture()
{
    //
    // return texture node of objects
    //
    return texture;
}

//=========================================================================
//
//=========================================================================
void InvTriangleStrip::setCoords(int no_of_strips, int no_of_vertices,
                                 int no_of_coords, float *x_c, float *y_c, float *z_c,
                                 int *vertex_list, int *index_list)
{
    if ((no_of_strips > 0) && (no_of_vertices > 0) && (no_of_coords > 0))
    {
        empty_ = false;
    }

    long no_strip;
    long j, k;
    float *coord_points;
    int32_t *vertices;

    //
    // store coordinates
    //
    coord_points = new float[no_of_coords * 3];
    vertices = new int32_t[no_of_vertices + no_of_strips];

    if (coord_points != NULL && vertices != NULL)
    {
        k = 0;
        for (j = 0; j < no_of_coords; j++)
        {
            *(coord_points + k) = *(x_c + j);
            *(coord_points + k + 1) = *(y_c + j);
            *(coord_points + k + 2) = *(z_c + j);
            k = k + 3;
        }
        if (colPack_ == INV_TEXTURE)
            vertexp->vertex.setValues(0, no_of_coords, (float(*)[3])coord_points);
        else
            coord->point.setValues(0, no_of_coords, (float(*)[3])coord_points);

        // store vertices list
        //
        j = 0;
        for (no_strip = 0; no_strip < no_of_strips - 1; no_strip++)
        {
            for (k = *(index_list + no_strip); k < *(index_list + no_strip + 1); k++)
            {
                *(vertices + j) = *(vertex_list + k);
                j++;
            }
            *(vertices + j) = SO_END_STRIP_INDEX;
            j++;
        }
        // last strip
        for (k = *(index_list + no_of_strips - 1); k < no_of_vertices; k++)
        {
            *(vertices + j) = *(vertex_list + k);
            j++;
        }
        if (no_of_vertices)
            *(vertices + j) = SO_END_STRIP_INDEX;

        SoIndexedTriangleStripSet *pt = (SoIndexedTriangleStripSet *)(geoShape_);
        if (pt)
        {
            pt->coordIndex.setValues(0, no_of_vertices + no_of_strips, vertices);
        }
        else
        {
            // this should never occur
            cerr << "FATAL error: InvPolygon::setCoords(..) dynamic cast failed in line "
                 << __LINE__ << "file " << __FILE__ << endl;
        }

        delete[] coord_points;
        delete[] vertices;
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
}

//=========================================================================
//
//=========================================================================
void
InvTriangleStrip::setNormalBinding(const int &type)
{
    if (colPack_ == INV_TEXTURE)
    {
        if (type == INV_PER_VERTEX)
            vertexp->normalBinding = SoVertexProperty::PER_VERTEX_INDEXED;
        else if (type == INV_PER_FACE)
            vertexp->normalBinding = SoVertexProperty::PER_FACE;
        else if (type == INV_NONE)
            vertexp->normalBinding = SoNormalBinding::DEFAULT;
    }
    else
    {
        if (type == INV_PER_VERTEX)
            normbind_->value = SoNormalBinding::PER_VERTEX_INDEXED;
        else if (type == INV_PER_FACE)
            normbind_->value = SoNormalBinding::PER_FACE;
        else if (type == INV_NONE)
            normbind_->value = SoNormalBinding::DEFAULT;
    }
}

//=========================================================================
//
//=========================================================================
void InvTriangleStrip::setTexture(int texWidth, int texHeight, int pixelSize, unsigned char *image)
{
    if (empty_)
        return;

    texture->image.setValue(SbVec2s(texWidth, texHeight), pixelSize, image);
    if (gType != MXI)
    {
        // Notice: clamping doesn't work correctly on MXI.
        texture->wrapS.setValue(SoTexture2::CLAMP);
        texture->wrapT.setValue(SoTexture2::CLAMP);
    }
    texture->model.setValue(SoTexture2::MODULATE);
}

void InvTriangleStrip::setTextureCoordinateBinding(int type)
{
    (void)type;
    //if ( type == INV_PER_VERTEX)
    //  vertexp->textureBinding = SoVertexProperty::PER_VERTEX_INDEXED;
    //else if ( type == INV_NONE )
    //  vertexp->textureBinding = SoVertexProperty::OVERALL;
}

//=========================================================================
//
//=========================================================================
void InvTriangleStrip::setTexCoords(int no_of_texCoords, float *tx, float *ty)
{
    if (empty_)
        return;
    long j, k;
    float *tc;
    float texX;

    //
    // store texture coordinates
    //
    tc = new float[no_of_texCoords * 2];
    if (tc != NULL)
    {
        k = 0;
        for (j = 0; j < no_of_texCoords; j++)
        {

            if (gType != MXI)
            {
                *(tc + k) = *(tx + j);
                *(tc + k + 1) = *(ty + j);
            }
            else
            {
                texX = *(tx + j);

                // Workaround
                // Notice: clamping doesn't work correctly on MXI.
                if (texX <= BORDER)
                    *(tc + k) = (float)BORDER;
                else if (texX >= 1.0f - BORDER)
                    *(tc + k) = 1.0f - (float)BORDER;
                else
                    *(tc + k) = texX;

                *(tc + k + 1) = *(ty + j);
            }

            k = k + 2;
        }
        vertexp->texCoord.setValues(0, no_of_texCoords, (float(*)[2])tc);
        delete[] tc;
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
}

//=========================================================================
//
//=========================================================================
void InvTriangleStrip::setDrawstyle(int)
{
}

//=========================================================================
//
//=========================================================================
void InvTriangleStrip::setVertexOrdering(int ordering)
{
    if (empty_)
        return;
    switch (ordering)
    {
    case 0:
        shapehints->vertexOrdering = SoShapeHints::UNKNOWN_ORDERING;
        break;
    case 1:
        shapehints->vertexOrdering = SoShapeHints::CLOCKWISE;
        break;
    case 2:
        shapehints->vertexOrdering = SoShapeHints::COUNTERCLOCKWISE;
        break;
    }
}

//=========================================================================
//
//=========================================================================
InvQuadmesh::InvQuadmesh(int colorpacking)
    : InvGeoObj(colorpacking)
{
    //cerr << "InvQuadmesh::InvQuadmesh(..) called" << endl;

    geoShape_ = new SoQuadMesh;
    coord = new SoCoordinate3;
    shapehints = new SoShapeHints;
    lightmodel = new SoLightModel;

    root_->addChild(coord);
    root_->addChild(shapehints);
    root_->addChild(lightmodel);

    drawstyle_->style = SoDrawStyle::FILLED;
    matbind->value = SoMaterialBinding::OVERALL;
    normbind_->value = SoNormalBinding::DEFAULT;
    if (smoothNormalsEnabled)
    {
        shapehints->creaseAngle = (float)CREASEANGLE;
    }

    defaultColor[0] = 1.;
    defaultColor[1] = 1.;
    defaultColor[2] = 1.;
    material_->diffuseColor.setValues(0, 1, (float(*)[3])defaultColor);
    defaultColor[0] = 0.;
    defaultColor[1] = 0.;
    defaultColor[2] = 0.;
    material_->emissiveColor.setValues(0, 1, (float(*)[3])defaultColor);
    material_->ambientColor.setValues(0, 1, (float(*)[3])defaultColor);

    //root_->addChild(geoShape_);
    geoGrp_->addChild(geoShape_);
    root_->addChild(geoGrp_);
}

//=========================================================================
//
//=========================================================================
void InvQuadmesh::setVertexOrdering(int ordering)
{
    switch (ordering)
    {
    case 0:
        shapehints->vertexOrdering = SoShapeHints::UNKNOWN_ORDERING;
        break;
    case 1:
        shapehints->vertexOrdering = SoShapeHints::CLOCKWISE;
        break;
    case 2:
        shapehints->vertexOrdering = SoShapeHints::COUNTERCLOCKWISE;
        break;
    }
}

//=========================================================================
//
//=========================================================================
void InvQuadmesh::setCoords(int VerticesPerRow, int VerticesPerColumn,
                            float *x_c, float *y_c, float *z_c)
{
    int num_vec;
    long j, k;
    float *vertices;

    //
    // store coordinates
    //
    num_vec = VerticesPerRow * VerticesPerColumn;
    vertices = new float[num_vec * 3];

    if (vertices != NULL)
    {
        k = 0;
        for (j = 0; j < num_vec; j++)
        {
            *(vertices + k) = *(x_c + j);
            *(vertices + k + 1) = *(y_c + j);
            *(vertices + k + 2) = *(z_c + j);
            k += 3;
        }
        coord->point.setValues(0, num_vec, (float(*)[3])vertices);

        SoQuadMesh *pt = (SoQuadMesh *)(geoShape_);
        if (pt)
        {
            pt->verticesPerRow = VerticesPerRow;
            pt->verticesPerColumn = VerticesPerColumn;
        }
        else
        {
            // this should never occur
            cerr << "FATAL error: InvPolygon::setCoords(..) dynamic cast failed in line "
                 << __LINE__ << "file " << __FILE__ << endl;
        }

        delete[] vertices;
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
}

//=========================================================================
//
//=========================================================================
void InvQuadmesh::setDrawstyle(int)
{
}

//-----------------------------------------------------------------------------
// Name: getBitmapImageData()
// Desc: Simply image loader for 24 bit BMP files.
//-----------------------------------------------------------------------------
void InvSphere::getBitmapImageData(const char *pFileName, BMPImage *pImage)
{
    FILE *pFile = NULL;
    unsigned short nNumPlanes;
    unsigned short nNumBPP;
    size_t i;

    if ((pFile = fopen(pFileName, "rb")) == NULL)
    {
        printf("ERROR: getBitmapImageData - %s not found\n", pFileName);
        return;
    }

    // Seek forward to width and height info
    fseek(pFile, 18, SEEK_CUR);

    if ((i = fread(&pImage->width, 4, 1, pFile)) != 1)
        printf("ERROR: getBitmapImageData - Couldn't read width from %s.\n", pFileName);

    if ((i = fread(&pImage->height, 4, 1, pFile)) != 1)
        printf("ERROR: getBitmapImageData - Couldn't read height from %s.\n", pFileName);

    if ((fread(&nNumPlanes, 2, 1, pFile)) != 1)
        printf("ERROR: getBitmapImageData - Couldn't read plane count from %s.\n", pFileName);

    if (nNumPlanes != 1)
        printf("ERROR: getBitmapImageData - Plane count from %s is not 1: %u\n", pFileName, nNumPlanes);

    if ((i = fread(&nNumBPP, 2, 1, pFile)) != 1)
        printf("ERROR: getBitmapImageData - Couldn't read BPP from %s.\n", pFileName);

    if (nNumBPP != 24)
        printf("ERROR: getBitmapImageData - BPP from %s is not 24: %u\n", pFileName, nNumBPP);

    // Seek forward to image data
    fseek(pFile, 24, SEEK_CUR);

    // Calculate the image's total size in bytes. Note how we multiply the
    // result of (width * height) by 3. This is becuase a 24 bit color BMP
    // file will give you 3 bytes per pixel.
    int nTotalImagesize = (pImage->width * pImage->height) * 3;

    pImage->data = (unsigned char *)malloc(nTotalImagesize);

    if ((i = fread(pImage->data, nTotalImagesize, 1, pFile)) != 1)
        printf("ERROR: getBitmapImageData - Couldn't read image data from %s.\n", pFileName);

    //
    // Finally, rearrange BGR to RGB
    //

    unsigned char charTemp;
    for (i = 0; i < nTotalImagesize; i += 3)
    {
        charTemp = pImage->data[i];
        pImage->data[i] = pImage->data[i + 2];
        pImage->data[i + 2] = charTemp;
    }
    if (pFile)
        fclose(pFile);
}
