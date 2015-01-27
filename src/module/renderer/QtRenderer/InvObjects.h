/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_OBJECTS_H
#define _INV_OBJECTS_H

/* $Id: InvObjects.h /main/vir_main/2 22-Aug-2001.15:25:53 ralfm_te $ */

/* $Log:  $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

// **************************************************************************
//
// * Description    : Inventor related stuff for drawing points, lines and
//                    polygons
//
//                    This is the class description
//
// * Class(es)      : InvPoint, InvLine, InvPolygon
//
//
// * inherited from : none
//
//
// * Author  : Dirk Rantzau
//
//
// * History : 29.03.94 V 1.0
//
//
//
// **************************************************************************
//
// Inventor stuff
//

#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoNormal.h>
#include <Inventor/nodes/SoNormalBinding.h>
#include <Inventor/nodes/SoTexture2.h>
#include <Inventor/nodes/SoTextureCoordinate2.h>
#include <Inventor/nodes/SoTextureCoordinateBinding.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoPackedColor.h>
#include <Inventor/nodes/SoMaterialBinding.h>
#include <Inventor/nodes/SoGroup.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoPointSet.h>
#include <Inventor/nodes/SoDrawStyle.h>
#include <Inventor/nodes/SoNormal.h>
#include <Inventor/nodes/SoLabel.h>
#include <Inventor/nodes/SoTransform.h>
#include <Inventor/nodes/SoNormalBinding.h>
#include <Inventor/nodes/SoLightModel.h>
#include <Inventor/nodes/SoIndexedLineSet.h>
#include <Inventor/nodes/SoFaceSet.h>
#include <Inventor/nodes/SoQuadMesh.h>
#include <Inventor/nodes/SoIndexedFaceSet.h>
#include <Inventor/nodes/SoIndexedTriangleStripSet.h>
#include <Inventor/nodes/SoShapeHints.h>
#include <Inventor/nodes/SoCallback.h>
#include <Inventor/actions/SoAction.h>
#include <Inventor/actions/SoGLRenderAction.h>
#include <Inventor/elements/SoCacheElement.h>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#if 0
#ifdef HAVE_CG
#include <Cg/cg.h>
#include <Cg/cgGL.h>
#endif
#endif

//
// C stuff
//
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <list>
#include <vector>

//
// Renderer stuff
//
#include "InvDefs.h"
#include "InvError.h"

#include <util/coMaterial.h>

//
// ec stuff
//
#include <covise/covise_process.h>

#ifdef __sgi
#include <invent.h>
#endif

enum GraphicsType // compare settings from <invent.h>
{
    UNDEF,
    RE = 12, // RealityEngine
    MXI = 15, // MaximumImpact (Mardigras graphics)
    IR = 16, // InfiniteReality
    IR2 = 18 // InfiniteReality2
};

#define BORDER 0.0025

//
// forward declarations
//
class InvViewer;

//
// Base class of all Inventor geometry objs.
//
class InvGeoObj
{
public:
    InvGeoObj(int colorpacking = INV_NONE);

    void setName(const char *name);
    void setNormals(int no_of_normals, float *nx, float *ny, float *nz);
    void setMaterial(covise::coMaterial *m);
    void setColors(int no_of_colors, float *r, float *g, float *b);
    void setColors(int no_of_colors, uint32_t *pc);
    void setColorBinding(const int &type);
    void setTransparency(const float &transparency);
    void setRealObjName(const char *rn);
    virtual void setNormalBinding(const int &type);
    void setGrpLabel(const char *lb);

    SoGroup *getTopNode();
    SoSeparator *getSeparator();
    SoTransform *getTransform();
    SoDrawStyle *getDrawStyle();

    virtual ~InvGeoObj();

protected:
    int colPack_;
    int colBind_;
    char *gName_;

    SoSeparator *root_;
    SoDrawStyle *drawstyle_;
    SoSwitch *top_switch_;
    SoShape *geoShape_;
    SoMaterial *material_;
    SoMaterialBinding *matbind;
    SoNormalBinding *normbind_;

    // private:

    SoGroup *geoGrp_;
    SoTransform *transform_;
    SoLabel *objName_;
    SoLabel *rObjName_;
    SoNormal *normal_;
};

class InvPoint : public InvGeoObj
{
private:
    float defaultColor[3];
    SoCoordinate3 *coord_bbox;
    SoCoordinate3 *coord;
    SoPointSet *points;
    SoLightModel *lightmodel;

    int colorSwap_;

public:
    InvPoint(int colorpacking = INV_NONE);

    void setCoords(int no_of_points, float *x_c, float *y_c, float *z_c);

    void setSize(float pointsize);

    ~InvPoint(){};
};

class InvSphere : public InvGeoObj
{
public:
    typedef vector<float> VF3;

    enum RENDER_METHOD
    {
        RENDER_METHOD_MANUAL_CPU_BILLBOARDS = 0,
        RENDER_METHOD_ARB_POINT_SPRITES,
        RENDER_METHOD_CG_VERTEX_SHADER
    };

    InvSphere(int colorpacking = INV_NONE);
    bool Init();
    bool InitBillboards();

    void setCoords(const int no_of_points, float *x_c, float *y_c, float *z_c);
    void setRadii(float *radii_c);
    void setColors(float *fR, float *fG, float *fB);
    void setColors(uint32_t *pc);
    void setRenderMethod(RENDER_METHOD rm);
    void setViewer(InvViewer *pViewer);
    InvViewer *getViewer();
    void Render();
    static void RenderCallback(void *d, SoAction *action);
    void SetTexture(const char *chTexFile);
    void EnableBlendOutColorKey(bool bBlendOutColorKey);
    bool IsEnabledBlendOutColorKey();

    struct BMPImage
    {
        int width;
        int height;
        unsigned char *data;
    };
    void getBitmapImageData(const char *pFileName, BMPImage *pImage);
    bool loadTexture(const char *pFilename, int iTextureMode = 0, int colorR = 0, int colorG = 0, int colorB = 0);

    virtual ~InvSphere();

private:
    list<VF3> coord;
    list<VF3> m_vf3Color;
    unsigned int m_iNoOfSpheres;
    SoCallback *cbfunction;

    InvViewer *m_pViewer;

    float *m_pfRadii;
    float m_fSize;
    RENDER_METHOD m_renderMethod;
    GLuint m_textureID;
    bool m_bBlendOutColorKey;
#if 0
#ifdef HAVE_CG
	CGprofile   m_CGprofile;
	CGcontext   m_CGcontext;
	CGprogram   m_CGprogram;
	CGparameter m_CGparam_modelViewProj;
   CGparameter m_CGparam_modelView;
	CGparameter m_CGparam_preRotatedQuad;
#endif
#endif

    char *m_chTexFile;
    float m_fMaxPointSize;
    bool empty_;
};

//****************************************************************************
// InvLine
//****************************************************************************

class InvLine : public InvGeoObj
{
private:
    float defaultColor[3];
    SoCoordinate3 *coord;
    SoIndexedLineSet *lines;
    SoLightModel *lightmodel;

    int colorSwap_;

public:
    InvLine(int colorpacking = INV_NONE);

    void setCoords(int no_of_lines, int no_of_vertices,
                   int no_of_coords, float *x_c, float *y_c, float *z_c,
                   int *vertex_list, int *index_list);

    void setDrawstyle(int type);

    ~InvLine(){};
};

//****************************************************************************
// InvPolygon
//****************************************************************************

class InvPolygon : public InvGeoObj
{
private:
    float defaultColor[3];
    SoCoordinate3 *coord;
    SoTexture2 *texture;
    SoTextureCoordinate2 *texCoord;
    SoTextureCoordinateBinding *texbind;
    SoLightModel *lightmodel;
    SoShapeHints *shapehints;
    GraphicsType gType;
    bool empty_;

public:
    InvPolygon(int colorpacking = INV_NONE);

    SoTexture2 *getTexture();

    void setCoords(int no_of_polygons, int no_of_vertices,
                   int no_of_coords, float *x_c, float *y_c, float *z_c,
                   int *vertex_list, int *index_list);

    void setTexture(int texWidth, int texHeight, int pixelSize, unsigned char *image);

    void setTextureCoordinateBinding(int type);

    void setTexCoords(int no_of_texCoords, float *tx, float *ty);

    void setDrawstyle(int type);

    void setVertexOrdering(int ordering);

    ~InvPolygon(){};
};

//****************************************************************************
// InvTriangleStrip
//****************************************************************************

class InvTriangleStrip : public InvGeoObj
{
private:
    float defaultColor[3];
    SoCoordinate3 *coord;
    SoTexture2 *texture;
    SoTextureCoordinate2 *texCoord;
    SoTextureCoordinateBinding *texbind;
    SoLightModel *lightmodel;
    SoShapeHints *shapehints;
    SoVertexProperty *vertexp;
    GraphicsType gType;
    bool empty_;

public:
    InvTriangleStrip(int colorpacking = INV_NONE, const char *rName = NULL);

    SoTexture2 *getTexture();

    void setCoords(int no_of_strips, int no_of_vertices,
                   int no_of_coords, float *x_c, float *y_c, float *z_c,
                   int *vertex_list, int *index_list);

    void setNormalBinding(const int &type);

    void setTexture(int texWidth, int texHeight, int pixelSize, unsigned char *image);

    void setTextureCoordinateBinding(int type);

    void setTexCoords(int no_of_texCoords, float *tx, float *ty);

    void setDrawstyle(int type);

    void setVertexOrdering(int ordering);

    ~InvTriangleStrip(){};
};

//****************************************************************************
// InvQuadmesh
//****************************************************************************

class InvQuadmesh : public InvGeoObj
{
private:
    float defaultColor[3];
    SoCoordinate3 *coord;
    SoLightModel *lightmodel;
    SoShapeHints *shapehints;

public:
    InvQuadmesh(int colorpacking = INV_NONE);

    void setCoords(int VerticesPerRow, int VerticesPerColumn,
                   float *x_c, float *y_c, float *z_c);

    void setDrawstyle(int type);

    void setVertexOrdering(int ordering);

    ~InvQuadmesh(){};
};
#endif
