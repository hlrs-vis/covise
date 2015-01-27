/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
	FILE:			EvalCol.cpp
	DESCRIPTION:	Vertex Color Renderer
	CREATED BY:		Christer Janson
	HISTORY:		Created Monday, December 12, 1996

 *>	Copyright (c) 1997 Kinetix, All Rights Reserved.
 **********************************************************************/
//***************************************************************************
// December 12/13   1996	CCJ
// January  8		1997	CCJ  Bugfix
// June		1		1997	CCJ  Port to MAX 2.0
// June		6		1997	CCJ  Implemented full ShadeContext
//
// Description:
// These functions calculates the diffuse, ambient or pre-lit color at each
// vertex or face of an INode.
//
// Exports:
// BOOL calcMixedVertexColors(INode*, TimeValue, int, ColorTab&);
//      This function calculates the interpolated diffuse or ambient
//      color at each vetex of an INode.
//      Usage: Pass in a node pointer and the TimeValue to generate
//      a list of Colors corresponding to each vertex in the mesh
//      Use the int flag to specify if you want to have diffuse or
//      ambient colors, or if you want to use the scene lights.
//      Note:
//        You are responsible for deleting the Color objects in the table.
//      Additional note:
//        Since materials are assigned by face, this function renders each
//        face connected to the specific vertex (at the point of the vertex)
//        and mixes the colors afterwards. If this is not what you want
//        you can use the calcFaceColors() to calculate the color at the
//        centerpoint of each face.
//
//***************************************************************************

#include "max.h"
#include "bmmlib.h"
#include "evalcol.h"

// Enable this to print out debug information
// #define EVALCOL_DEBUG

class SContext;
class RefEnumProc;

Point3 interpVertexNormal(Mesh *mesh, Matrix3 tm, unsigned int vxNo, BitArray &faceList);
void AddSceneLights(SContext *sc, MtlBaseLib *mtls);
int LoadMapFiles(INode *node, SContext *sc, MtlBaseLib &mtls, TimeValue t);
void EnumRefs(ReferenceMaker *rm, RefEnumProc &proc);

SingleVertexColor::~SingleVertexColor()
{
    for (int i = 0; i < vertexColors.Count(); i++)
    {
        delete vertexColors[i];
    }
}

//***************************************************************************
//* The is the map enumerator class used for collecting projector lights for
//* spotlights
//***************************************************************************

class GetMaps : public RefEnumProc
{
    MtlBaseLib *mlib;

public:
#if MAX_PRODUCT_VERSION_MAJOR > 8
    int proc(ReferenceMaker *rm);
#else
    void proc(ReferenceMaker *rm);
#endif
    GetMaps(MtlBaseLib *mbl);
};

//***************************************************************************
//* The is the Light descriptor object for default lights
//***************************************************************************
class DefObjLight : public ObjLightDesc
{
public:
    Color intensCol; // intens*color
    DefObjLight(DefaultLight *l);
    void DeleteThis() { delete this; }
    int Update(TimeValue t, const RendContext &rc, RenderGlobalContext *rgc, BOOL shadows, BOOL shadowGeomChanged);
    int UpdateViewDepParams(const Matrix3 &worldToCam);
    BOOL Illuminate(ShadeContext &sc, Point3 &normal, Color &color, Point3 &dir, float &dot_nl, float &diffCoef);
};

class LightInfo
{
public:
    LightInfo(INode *node, MtlBaseLib *mtls);
    LightInfo(DefaultLight *l);
    ~LightInfo();

    ObjLightDesc *lightDesc;
    LightObject *light;
};

typedef Tab<LightInfo *> LightTab;

//***************************************************************************
//* RendContext is used to evaluate the lights
//***************************************************************************

class RContext : public RendContext
{
public:
    Matrix3 WorldToCam() const { return Matrix3(1); }
    ShadowBuffer *NewShadowBuffer() const;
    ShadowQuadTree *NewShadowQuadTree() const;
    Color GlobalLightLevel() const;
    int Progress(int done, int total)
    {
        return 1;
    }
};

//***************************************************************************
// ShadeContext for evaluating materials
//***************************************************************************

class SContext : public ShadeContext
{
public:
    SContext();
    ~SContext();

    TimeValue CurTime();
    int NodeID();
    INode *Node();
    Point3 BarycentricCoords();
    int FaceNumber();
    Point3 Normal();
    float Curve();

    LightDesc *Light(int lightNo);
    Point3 GNormal(void);
    Point3 ReflectVector(void);
    Point3 RefractVector(float ior);
    Point3 CamPos(void);
    Point3 V(void);
    Point3 P(void);
    Point3 DP(void);
    Point3 PObj(void);
    Point3 DPObj(void);
    Box3 ObjectBox(void);
    Point3 PObjRelBox(void);
    Point3 DPObjRelBox(void);
    void ScreenUV(Point2 &uv, Point2 &duv);
    IPoint2 ScreenCoord(void);
    Point3 UVW(int chan);
    Point3 DUVW(int chan);
    void DPdUVW(Point3[], int chan);
    void GetBGColor(Color &bgCol, Color &transp, int fogBG);
    Point3 PointTo(const Point3 &p, RefFrame ito);
    Point3 PointFrom(const Point3 &p, RefFrame ito);
    Point3 VectorTo(const Point3 &p, RefFrame ito);
    Point3 VectorFrom(const Point3 &p, RefFrame ito);
    int InMtlEditor();
    void SetView(Point3 v);

    int ProjType();
    void SetNodeAndTime(INode *n, TimeValue tm);
    void SetMesh(Mesh *m);
    void SetBaryCoord(Point3 bary);
    void SetFaceNum(int f);
    void SetMtlNum(int mNo);
    void SetTargetPoint(Point3 tp);
    void SetViewPoint(Point3 vp);
    void SetViewDir(Point3 vd);
    void CalcNormals();
    void CalcBoundObj();
    void ClearLights();
    void AddLight(LightInfo *li);
    void SetAmbientLight(Color c);
    void UpdateLights();
    void calc_size_ratio();
    float RayDiam() { return 0.1f; }
    void getTVerts(int chan);
    void getObjVerts();

public:
    LightTab lightTab;
    Matrix3 tmAfterWSM;

private:
    INode *node;
    Mesh *mesh;
    Point3 baryCoord;
    int faceNum;
    Point3 targetPt;
    Point3 viewDir;
    Point3 viewPoint;
    TimeValue t;
    Point3 vxNormals[3];
    UVVert tv[2][3];
    Point3 bumpv[2][3];
    Box3 boundingObj;
    RContext rc;
    Point3 obpos[3];
    Point3 dobpos;
    float ratio;
    float curve;
};

//***************************************************************************
//* Dummy Material : Simple Phong shader using Node color
//* This material is assigned to each node that does not have a material
//* previously assigned. The diffuse color is assigned based on the
//* wireframe color.
//* This way we can assume that all nodes have a material assigned.
//***************************************************************************

#define DUMMTL_CLASS_ID Class_ID(0x4efd2694, 0x37c809f4)

#define DUMSHINE .25f
#define DUMSPEC .50f

class DumMtl : public Mtl
{
    Color diff, spec;
    float phongexp;

public:
    DumMtl(Color c);
    void Update(TimeValue t, Interval &valid);
    void Reset();
    Interval Validity(TimeValue t);
    ParamDlg *CreateParamDlg(HWND hwMtlEdit, IMtlParams *imp);
    Color GetAmbient(int mtlNum = 0, BOOL backFace = FALSE);
    Color GetDiffuse(int mtlNum = 0, BOOL backFace = FALSE);
    Color GetSpecular(int mtlNum = 0, BOOL backFace = FALSE);
    float GetShininess(int mtlNum = 0, BOOL backFace = FALSE);
    float GetShinStr(int mtlNum = 0, BOOL backFace = FALSE);
    float GetXParency(int mtlNum = 0, BOOL backFace = FALSE);
    void SetAmbient(Color c, TimeValue t);
    void SetDiffuse(Color c, TimeValue t);
    void SetSpecular(Color c, TimeValue t);
    void SetShininess(float v, TimeValue t);
    Class_ID ClassID();
    void DeleteThis();
#if MAX_PRODUCT_VERSION_MAJOR > 16
    RefResult NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                               PartID &partID, RefMessage message, BOOL propagate);
#else
    RefResult NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                               PartID &partID, RefMessage message);
#endif
    void Shade(ShadeContext &sc);
};

// Return a pointer to a TriObject given an INode or return NULL
// if the node cannot be converted to a TriObject
TriObject *GetTriObjectFromNode(INode *node, TimeValue t, int &deleteIt)
{
    deleteIt = FALSE;
    Object *obj = node->EvalWorldState(t).obj;
    if (obj->CanConvertToType(Class_ID(TRIOBJ_CLASS_ID, 0)))
    {
        TriObject *tri = (TriObject *)obj->ConvertToType(t,
                                                         Class_ID(TRIOBJ_CLASS_ID, 0));
        // Note that the TriObject should only be deleted
        // if the pointer to it is not equal to the object
        // pointer that called ConvertToType()
        if (obj != tri)
            deleteIt = TRUE;
        return tri;
    }
    else
    {
        return NULL;
    }
}

//***************************************************************************
// Calculate ambient or diffuse color at each vertex.
//***************************************************************************
BOOL calcVertexColors(INode *node, TimeValue t, int lightModel, VertexColorTab &vxColTab, EvalColProgressCallback *fn)
{
    ObjectState ostate;
    BOOL deleteTri;
    Mesh *mesh;
    SContext sc;
    DefaultLight dl1, dl2;
    MtlBaseLib mtls;
    Matrix3 tm;

    sc.SetNodeAndTime(node, t);
    tm = sc.tmAfterWSM;

    TriObject *tri = GetTriObjectFromNode(node, t, deleteTri);

    // We will only work on GeomObjects
    if (!tri)
    {
        return FALSE;
    }

    // Get the mesh from the object
    mesh = &tri->GetMesh();
    if (!mesh)
    {
        return FALSE;
    }

    // If the node doesn't have a material attached,
    // we create a dummy material.
    Mtl *mtl = node->GetMtl();
    if (!mtl)
    {
        mtl = new DumMtl((Color)node->GetWireColor());
    }

    mesh->buildRenderNormals();

    vxColTab.ZeroCount();
    vxColTab.Shrink();

    sc.SetMesh(mesh);
    sc.CalcBoundObj();

    // Add the material to the list
    mtls.AddMtl(mtl);

    // Setup ambient light
    if (lightModel == LIGHT_AMBIENT)
    {
        sc.SetAmbientLight(Color(1.0f, 1.0f, 1.0f));
    }
    else
    {
        sc.SetAmbientLight(Color(0.0f, 0.0f, 0.0f));
    }

    // If we're using the real lights, we need to find them first
    if (lightModel == LIGHT_SCENELIGHT)
    {
        AddSceneLights(&sc, &mtls);

        // Add default lights if there are no lights in the scene
        if (sc.lightTab.Count() == 0)
        {
            dl1.ls.intens = 1.0f;
            dl1.ls.color = Color(0.8f, 0.8f, 0.8f);
            dl1.ls.type = OMNI_LGT;
            dl1.tm = TransMatrix(1000.0f * Point3(-900.0f, -1000.0f, 1500.0f));

            dl2.ls.intens = 1.0f;
            dl2.ls.color = Color(0.8f, 0.8f, 0.8f);
            dl2.ls.type = OMNI_LGT;
            dl2.tm = TransMatrix(-1000.0f * Point3(-900.0f, -1000.0f, 1500.0f));

            sc.AddLight(new LightInfo(&dl1));
            sc.AddLight(new LightInfo(&dl2));
        }

        sc.SetAmbientLight(GetCOREInterface()->GetAmbient(t, FOREVER));
    }

    sc.UpdateLights();
    // Update material
    mtl->Update(t, FOREVER);

    int numVerts = mesh->numVerts;
    for (unsigned int v = 0; v < (unsigned)numVerts; v++)
    {

        if (fn)
        {
            if (fn->progress(float(v) / float(numVerts)))
            {
                if (deleteTri)
                {
                    delete tri;
                }

                mtls.Empty();

                if (mtl->ClassID() == DUMMTL_CLASS_ID)
                {
                    delete mtl;
                }

                // What to return here is up for discussion.
                // 1) We are aborting so FALSE might be in order.
                // 2) We have calculated some colors. Let's use what we've got so far.
                return TRUE;
            }
        }

        BitArray faceList;
        faceList.SetSize(mesh->numFaces, 0);

        // Get vertex normal
        // We also pass in a BitArray that will be filled in with
        // to inform us to which faces this vertex belongs.
        // We could do this manually, but we need to do it to get
        // the vertex normal anyway so this is done to speed things
        // up a bit.
        Point3 vxNormal = interpVertexNormal(mesh, tm, v, faceList);
        Point3 viewDir = -vxNormal;
        Point3 viewPoint = tm * mesh->verts[v] + 5.0f * vxNormal;
        Point3 lightPos = viewPoint;
        Point3 viewTarget = tm * mesh->verts[v];

        // We now have a viewpoint and a view target.
        // Now we just have to shade this point on the mesh in order
        // to get it's color.
        // Note:
        // Since materials are assigned on Face basis we need to render each
        // vertex as many times as it has connecting faces.

        SingleVertexColor *svc = new SingleVertexColor();

        for (int nf = 0; nf < faceList.GetSize(); nf++)
        {
            if (faceList[nf])
            {
                // render vertex for this face.
                sc.SetViewPoint(viewPoint);
                sc.SetTargetPoint(viewTarget);
                sc.SetViewDir(viewDir);
                sc.SetFaceNum(nf);
                Face *f = &mesh->faces[nf];
                sc.SetMtlNum(f->getMatID());
                sc.CalcNormals();

                // Setup the barycentric coordinate
                if (mesh->faces[nf].v[0] == v)
                    sc.SetBaryCoord(Point3(1.0f, 0.0f, 0.0f));
                else if (mesh->faces[nf].v[1] == v)
                    sc.SetBaryCoord(Point3(0.0f, 1.0f, 0.0f));
                else if (mesh->faces[nf].v[2] == v)
                    sc.SetBaryCoord(Point3(0.0f, 0.0f, 1.0f));

                // Use diffuse color instead of ambient
                // The only difference is that we create a special light
                // located at the viewpoint and we set the ambient light to black.
                if (lightModel == LIGHT_DIFFUSE)
                {
                    dl1.ls.intens = 1.0f;
                    dl1.ls.color = Color(0.8f, 0.8f, 0.8f);
                    dl1.ls.type = OMNI_LGT;
                    dl1.tm = TransMatrix(lightPos);

                    sc.ClearLights();
                    sc.AddLight(new LightInfo(&dl1));
                    sc.UpdateLights();
                }

                // Shade the vertex
                mtl->Shade(sc);

                Color *tmpCol = new Color();

                tmpCol->r += sc.out.c.r;
                tmpCol->g += sc.out.c.g;
                tmpCol->b += sc.out.c.b;

                tmpCol->ClampMinMax();

                svc->vertexColors.Append(1, &tmpCol, 2);
            }
        }

        // Append the Color to the table. If the array needs
        // to be realloc'ed, allocate extra space for 100 points.
        vxColTab.Append(1, &svc, 100);
    }

    // Some objects gives us a temporary mesh that we need to delete afterwards.
    if (deleteTri)
    {
        delete tri;
    }

    mtls.Empty();

    if (mtl->ClassID() == DUMMTL_CLASS_ID)
    {
        delete mtl;
    }

    return TRUE;
}

//***************************************************************************
// Calculate ambient or diffuse color at each vertex.
// Pass in TRUE as the "diffuse" parameter to calculate the diffuse color.
// If FALSE is passed in, ambient color is calculated.
//***************************************************************************
BOOL calcMixedVertexColors(INode *node, TimeValue t, int lightModel, ColorTab &vxColTab, EvalColProgressCallback *fn)
{
    ObjectState ostate;
    BOOL deleteTri;
    Mesh *mesh;
    SContext sc;
    DefaultLight dl1, dl2;
    MtlBaseLib mtls;
    Matrix3 tm;

    sc.SetNodeAndTime(node, t);
    tm = sc.tmAfterWSM;

    TriObject *tri = GetTriObjectFromNode(node, t, deleteTri);

    // We will only work on GeomObjects
    if (!tri)
    {
        return FALSE;
    }

    // Get the mesh from the object
    mesh = &tri->GetMesh();
    if (!mesh)
    {
        return FALSE;
    }

    // If the node doesn't have a material attached,
    // we create a dummy material.
    Mtl *mtl = node->GetMtl();
    if (!mtl)
    {
        mtl = new DumMtl((Color)node->GetWireColor());
    }

    mesh->buildRenderNormals();

    vxColTab.ZeroCount();
    vxColTab.Shrink();

    sc.SetMesh(mesh);
    sc.CalcBoundObj();

    // Add the material to the list
    mtls.AddMtl(mtl);

    // Setup ambient light
    if (lightModel == LIGHT_AMBIENT)
    {
        sc.SetAmbientLight(Color(1.0f, 1.0f, 1.0f));
    }
    else
    {
        sc.SetAmbientLight(Color(0.0f, 0.0f, 0.0f));
    }

    // If we're using the real lights, we need to find them first
    if (lightModel == LIGHT_SCENELIGHT)
    {
        AddSceneLights(&sc, &mtls);

        // Add default lights if there are no lights in the scene
        if (sc.lightTab.Count() == 0)
        {
            dl1.ls.intens = 1.0f;
            dl1.ls.color = Color(0.8f, 0.8f, 0.8f);
            dl1.ls.type = OMNI_LGT;
            dl1.tm = TransMatrix(1000.0f * Point3(-900.0f, -1000.0f, 1500.0f));

            dl2.ls.intens = 1.0f;
            dl2.ls.color = Color(0.8f, 0.8f, 0.8f);
            dl2.ls.type = OMNI_LGT;
            dl2.tm = TransMatrix(-1000.0f * Point3(-900.0f, -1000.0f, 1500.0f));

            sc.AddLight(new LightInfo(&dl1));
            sc.AddLight(new LightInfo(&dl2));
        }

        sc.SetAmbientLight(GetCOREInterface()->GetAmbient(t, FOREVER));
    }

    sc.UpdateLights();
    // Update material
    mtl->Update(t, FOREVER);

    int numVerts = mesh->numVerts;
    for (unsigned int v = 0; v < (unsigned)numVerts; v++)
    {

        if (fn)
        {
            if (fn->progress(float(v) / float(numVerts)))
            {
                if (deleteTri)
                {
                    delete tri;
                }

                mtls.Empty();

                if (mtl->ClassID() == DUMMTL_CLASS_ID)
                {
                    delete mtl;
                }

                // What to return here is up for discussion.
                // 1) We are aborting so FALSE might be in order.
                // 2) We have calculated some colors. Let's use what we've got so far.
                return TRUE;
            }
        }

        // Create a new entry
        Color *vxCol = new Color;
        Point3 tmpCol(0.0f, 0.0f, 0.0f);

        int numShades = 0;
        BitArray faceList;
        faceList.SetSize(mesh->numFaces, 0);

        // Get vertex normal
        // We also pass in a BitArray that will be filled in with
        // to inform us to which faces this vertex belongs.
        // We could do this manually, but we need to do it to get
        // the vertex normal anyway so this is done to speed things
        // up a bit.
        Point3 vxNormal = interpVertexNormal(mesh, tm, v, faceList);
        Point3 viewDir = -vxNormal;
        Point3 viewPoint = tm * mesh->verts[v] + 5.0f * vxNormal;
        Point3 lightPos = viewPoint;
        Point3 viewTarget = tm * mesh->verts[v];

        // We now have a viewpoint and a view target.
        // Now we just have to shade this point on the mesh in order
        // to get it's color.
        // Note:
        // Since materials are assigned on Face basis we need to render each
        // vertex as many times as it has connecting faces.
        // the colors collected are mixed to get the resulting
        // color at each vertex.

        for (int nf = 0; nf < faceList.GetSize(); nf++)
        {
            if (faceList[nf])
            {
                // render vertex for this face.
                sc.SetViewPoint(viewPoint);
                sc.SetTargetPoint(viewTarget);
                sc.SetViewDir(viewDir);
                sc.SetFaceNum(nf);
                Face *f = &mesh->faces[nf];
                sc.SetMtlNum(f->getMatID());
                sc.CalcNormals();

                // Setup the barycentric coordinate
                if (mesh->faces[nf].v[0] == v)
                    sc.SetBaryCoord(Point3(1.0f, 0.0f, 0.0f));
                else if (mesh->faces[nf].v[1] == v)
                    sc.SetBaryCoord(Point3(0.0f, 1.0f, 0.0f));
                else if (mesh->faces[nf].v[2] == v)
                    sc.SetBaryCoord(Point3(0.0f, 0.0f, 1.0f));

                // Use diffuse color instead of ambient
                // The only difference is that we create a special light
                // located at the viewpoint and we set the ambient light to black.
                if (lightModel == LIGHT_DIFFUSE)
                {
                    dl1.ls.intens = 1.0f;
                    dl1.ls.color = Color(0.8f, 0.8f, 0.8f);
                    dl1.ls.type = OMNI_LGT;
                    dl1.tm = TransMatrix(lightPos);

                    sc.ClearLights();
                    sc.AddLight(new LightInfo(&dl1));
                    sc.UpdateLights();
                }

                // Shade the vertex
                mtl->Shade(sc);

                tmpCol.x += sc.out.c.r;
                tmpCol.y += sc.out.c.g;
                tmpCol.z += sc.out.c.b;
                numShades++;
            }
        }

        // The color mixes. We just add the colors together and
        // then divide with as many colors as we added.
        if (numShades > 0)
        {
            tmpCol = tmpCol / (float)numShades;
        }

        vxCol->r = tmpCol.x;
        vxCol->g = tmpCol.y;
        vxCol->b = tmpCol.z;

        vxCol->ClampMinMax();

        // Append the Color to the table. If the array needs
        // to be realloc'ed, allocate extra space for 100 points.
        vxColTab.Append(1, &vxCol, 100);
    }

    // Some objects gives us a temporary mesh that we need to delete afterwards.
    if (deleteTri)
    {
        delete tri;
    }

    mtls.Empty();

    if (mtl->ClassID() == DUMMTL_CLASS_ID)
    {
        delete mtl;
    }

    return TRUE;
}

// Since vertices might have different normals depending on the face
// you are accessing it through, we get the normal for each face that
// connects to this vertex and interpolate these normals to get a single
// vertex normal fairly perpendicular to the mesh at the point of
// this vertex.
Point3 interpVertexNormal(Mesh *mesh, Matrix3 tm, unsigned int vxNo, BitArray &faceList)
{
    Point3 iNormal = Point3(0.0f, 0.0f, 0.0f);
    int numNormals = 0;

    for (int f = 0; f < mesh->numFaces; f++)
    {
        for (int fi = 0; fi < 3; fi++)
        {
            if (mesh->faces[f].v[fi] == vxNo)
            {
                Point3 &fn = VectorTransform(tm, mesh->getFaceNormal(f));
                iNormal += fn;
                numNormals++;
                faceList.Set(f);
            }
        }
    }

    iNormal = iNormal / (float)numNormals;

    return Normalize(iNormal);
}

//***************************************************************************
// LightInfo encapsulates the light descriptor for standard and default lights
//***************************************************************************

LightInfo::LightInfo(INode *node, MtlBaseLib *mtls)
{
    ObjectState ostate = node->EvalWorldState(0);

    light = (LightObject *)ostate.obj;
    lightDesc = light->CreateLightDesc(node);

    // Process projector maps
    GetMaps getmaps(mtls);
    EnumRefs(light, getmaps);
}

LightInfo::LightInfo(DefaultLight *l)
{
    lightDesc = new DefObjLight(l);
    light = NULL;
}

LightInfo::~LightInfo()
{
    if (lightDesc)
    {
        delete lightDesc;
    }
}

//***************************************************************************
// Light Descriptor for the diffuse light we use
//***************************************************************************

DefObjLight::DefObjLight(DefaultLight *l)
    : ObjLightDesc(NULL)
{
    inode = NULL;
    ls = l->ls;
    lightToWorld = l->tm;
    worldToLight = Inverse(lightToWorld);
}

//***************************************************************************
// Update
//***************************************************************************

int DefObjLight::Update(TimeValue t, const RendContext &rc, RenderGlobalContext *rgc, BOOL shadows, BOOL shadowGeomChanged)
{
    intensCol = ls.intens * ls.color;
    return 1;
}

//***************************************************************************
// Update viewdependent parameters
//***************************************************************************

int DefObjLight::UpdateViewDepParams(const Matrix3 &worldToCam)
{
    lightToCam = lightToWorld * worldToCam;
    camToLight = Inverse(lightToCam);
    lightPos = lightToCam.GetRow(3); // light pos in camera space
    return 1;
}

//***************************************************************************
// Illuminate method for default lights
// This is a special illumination method in order to evaluate diffuse color
// only, with no specular etc.
//***************************************************************************

BOOL DefObjLight::Illuminate(ShadeContext &sc, Point3 &normal, Color &color, Point3 &dir, float &dot_nl, float &diffCoef)
{
    dir = Normalize(lightPos - sc.P());
    diffCoef = dot_nl = DotProd(normal, dir);
    color = intensCol;

    return (dot_nl <= 0.0f) ? 0 : 1;
}

static inline Point3 pabs(Point3 p) { return Point3(fabs(p.x), fabs(p.y), fabs(p.z)); }

// The ShadeContext used to shade a material a a specific point.
// This ShadeContext is setup to have full ambient light and no other
// lights until you call SetLight(). This will cause the ambient light to
// go black.
SContext::SContext()
{
    mode = SCMODE_NORMAL;
    doMaps = TRUE;
    filterMaps = FALSE;
    shadow = FALSE;
    backFace = FALSE;
    ambientLight = Color(1.0f, 1.0f, 1.0f);
    mtlNum = 0;

    nLights = 0;
}

SContext::~SContext()
{
    ClearLights();
}

// When the mesh and face number is specified we calculate
// and store the vertex normals
void SContext::CalcNormals()
{
    RVertex *rv[3];
    Face *f = &mesh->faces[faceNum];
    DWORD smGroup = f->smGroup;
    int numNormals;

    // Get the vertex normals
    for (int i = 0; i < 3; i++)
    {
        rv[i] = mesh->getRVertPtr(f->getVert(i));

        // Is normal specified
        // SPCIFIED is not currently used, but may be used in future versions.
        if (rv[i]->rFlags & SPECIFIED_NORMAL)
        {
            vxNormals[i] = rv[i]->rn.getNormal();
        }
        // If normal is not specified it's only available if the face belongs
        // to a smoothing group
        else if ((numNormals = rv[i]->rFlags & NORCT_MASK) && smGroup)
        {
            // If there is only one vertex is found in the rn member.
            if (numNormals == 1)
            {
                vxNormals[i] = rv[i]->rn.getNormal();
            }
            else
            {
                // If two or more vertices are there you need to step through them
                // and find the vertex with the same smoothing group as the current face.
                // You will find multiple normals in the ern member.
                for (int j = 0; j < numNormals; j++)
                {
                    if (rv[i]->ern[j].getSmGroup() & smGroup)
                    {
                        vxNormals[i] = rv[i]->ern[j].getNormal();
                    }
                }
            }
        }
        else
        {
            vxNormals[i] = mesh->getFaceNormal(faceNum);
        }
    }
    vxNormals[0] = Normalize(VectorTransform(tmAfterWSM, vxNormals[0]));
    vxNormals[1] = Normalize(VectorTransform(tmAfterWSM, vxNormals[1]));
    vxNormals[2] = Normalize(VectorTransform(tmAfterWSM, vxNormals[2]));
}

void SContext::SetBaryCoord(Point3 bary)
{
    baryCoord = bary;
}

int SContext::ProjType()
{
    return PROJ_PARALLEL;
}

void SContext::SetNodeAndTime(INode *n, TimeValue tv)
{
    node = n;
    t = tv;
    tmAfterWSM = node->GetObjTMAfterWSM(t, NULL);
}

void SContext::SetFaceNum(int f)
{
    faceNum = f;
}

void SContext::SetMtlNum(int mNo)
{
    mtlNum = mNo;
}

void SContext::SetViewPoint(Point3 vp)
{
    viewPoint = vp;
}

void SContext::SetTargetPoint(Point3 tp)
{
    targetPt = tp;
}

void SContext::SetViewDir(Point3 vd)
{
    viewDir = vd;
}

void SContext::SetMesh(Mesh *m)
{
    mesh = m;
}

void SContext::AddLight(LightInfo *li)
{
    lightTab.Append(1, &li);
}

void SContext::ClearLights()
{
    for (int i = 0; i < lightTab.Count(); i++)
    {
        delete lightTab[i];
    }
    lightTab.ZeroCount();
    lightTab.Shrink();
    nLights = 0;
}

void SContext::UpdateLights()
{
    for (int i = 0; i < lightTab.Count(); i++)
    {
        ((LightInfo *)lightTab[i])->lightDesc->Update(t, rc, NULL, FALSE, TRUE);
        ((LightInfo *)lightTab[i])->lightDesc->UpdateViewDepParams(Matrix3(1));
    }

    nLights = lightTab.Count();
}

void SContext::SetAmbientLight(Color c)
{
    ambientLight = c;
}

void SContext::CalcBoundObj()
{
    if (!mesh)
        return;

    boundingObj.Init();

    // Include each vertex in the bounding box
    for (int nf = 0; nf < mesh->numFaces; nf++)
    {
        Face *f = &(mesh->faces[nf]);

        boundingObj += mesh->getVert(f->getVert(0));
        boundingObj += mesh->getVert(f->getVert(1));
        boundingObj += mesh->getVert(f->getVert(2));
    }
}

// Return current time
TimeValue SContext::CurTime()
{
    return t;
}

int SContext::NodeID()
{
    return -1;
}

INode *SContext::Node()
{
    return node;
}

Point3 SContext::BarycentricCoords()
{
    return baryCoord;
}

int SContext::FaceNumber()
{
    return faceNum;
}

// Interpolated normal
Point3 SContext::Normal()
{
    return Normalize(baryCoord.x * vxNormals[0] + baryCoord.y * vxNormals[1] + baryCoord.z * vxNormals[2]);
}

// Geometric normal (face normal)
Point3 SContext::GNormal(void)
{
    // The face normals are already in camera space
    return VectorTransform(tmAfterWSM, mesh->getFaceNormal(faceNum));
}

// Return a Light descriptor
LightDesc *SContext::Light(int lightNo)
{
    return ((LightInfo *)lightTab[lightNo])->lightDesc;
}

// Return reflection vector at this point.
// We do it like this to avoid specular color to show up.
Point3 SContext::ReflectVector(void)
{
    return -Normal();
}

// Foley & vanDam: Computer Graphics: Principles and Practice,
//     2nd Ed. pp 756ff.
Point3 SContext::RefractVector(float ior)
{
    Point3 N = Normal();
    float VN, nur, k;
    VN = DotProd(-viewDir, N);
    if (backFace)
        nur = ior;
    else
        nur = (ior != 0.0f) ? 1.0f / ior : 1.0f;
    k = 1.0f - nur * nur * (1.0f - VN * VN);
    if (k <= 0.0f)
    {
        // Total internal reflection:
        return ReflectVector();
    }
    else
    {
        return (nur * VN - (float)sqrt(k)) * N + nur * viewDir;
    }
}

Point3 SContext::CamPos(void)
{
    return viewPoint;
}

// Screen coordinate beeing rendered
IPoint2 SContext::ScreenCoord(void)
{
    return IPoint2(0, 0);
}

// Background color
void SContext::GetBGColor(class Color &bgCol, class Color &transp, int fogBG)
{
    bgCol = Color(0.0f, 0.0f, 0.0f);
    transp = Color(0.0f, 0.0f, 0.0f);
}

// Transforms the specified point from internal camera space to the specified space.
Point3 SContext::PointTo(const class Point3 &p, RefFrame ito)
{
    if (ito == REF_OBJECT)
    {
        return Inverse(tmAfterWSM) * p;
    }

    return p;
}

// Transforms the specified point from the specified coordinate system
// to internal camera space.
Point3 SContext::PointFrom(const class Point3 &p, RefFrame ito)
{
    if (ito == REF_OBJECT)
    {
        return tmAfterWSM * p;
    }
    return p;
}

// Transform the vector from internal camera space to the specified space.
Point3 SContext::VectorTo(const class Point3 &p, RefFrame ito)
{
    if (ito == REF_OBJECT)
    {
        return VectorTransform(Inverse(tmAfterWSM), p);
    }
    return p;
}

// Transform the vector from the specified space to internal camera space.
Point3 SContext::VectorFrom(const class Point3 &p, RefFrame ito)
{
    if (ito == REF_OBJECT)
    {
        return VectorTransform(tmAfterWSM, p);
    }
    return p;
}

// This method returns the unit view vector, from the camera towards P,
// in camera space.
Point3 SContext::V(void)
{
    return viewDir;
}

// Returns the point to be shaded in camera space.
Point3 SContext::P(void)
{
    return targetPt;
}

// This returns the derivative of P, relative to the pixel.
// This gives the renderer or shader information about how fast the position
// is changing relative to the screen.
// TBD

#define DFACT .1f

Point3 SContext::DP(void)
{
    float d = (1.0f + DFACT) * (RayDiam()) / (DFACT + (float)fabs(DotProd(Normal(), viewDir)));
    return Point3(d, d, d);
}

// Retrieves the point relative to the screen where the lower left
// corner is 0,0 and the upper right corner is 1,1.
void SContext::ScreenUV(class Point2 &uv, class Point2 &duv)
{
    Point2 p;

    uv.x = .5f;
    uv.y = .5f;
    duv.x = 1.0f;
    duv.y = 1.0f;
}

// Bounding box in object coords
Box3 SContext::ObjectBox(void)
{
    return boundingObj;
}

// Returns the point to be shaded relative to the object box where each
// component is in the range of -1 to +1.
Point3 SContext::PObjRelBox(void)
{
    Point3 q;
    Point3 p = PObj();
    Box3 b = ObjectBox();
    q.x = 2.0f * (p.x - b.pmin.x) / (b.pmax.x - b.pmin.x) - 1.0f;
    q.y = 2.0f * (p.y - b.pmin.y) / (b.pmax.y - b.pmin.y) - 1.0f;
    q.z = 2.0f * (p.z - b.pmin.z) / (b.pmax.z - b.pmin.z) - 1.0f;
    return q;
}

// Returns the derivative of PObjRelBox().
// This is the derivative of the point relative to the object box where
// each component is in the range of -1 to +1.
Point3 SContext::DPObjRelBox(void)
{
    Box3 b = ObjectBox();
    Point3 d = DPObj();
    d.x *= 2.0f / (b.pmax.x - b.pmin.x);
    d.y *= 2.0f / (b.pmax.y - b.pmin.y);
    d.z *= 2.0f / (b.pmax.z - b.pmin.z);
    return d;
}

// Returns the point to be shaded in object coordinates.
Point3 SContext::PObj(void)
{
    return Inverse(tmAfterWSM) * P();
}

// Returns the derivative of PObj(), relative to the pixel.
// TBD
Point3 SContext::DPObj(void)
{
    Point3 d = DP();
    return VectorTransform(Inverse(tmAfterWSM), d);
}

// Returns the UVW coordinates for the point.
Point3 SContext::UVW(int)
{
    Point3 uvw = Point3(0.0f, 0.0f, 0.0f);
    UVVert tverts[3];

    if (mesh->numTVerts > 0)
    {
        TVFace *tvf = &mesh->tvFace[faceNum];
        tverts[0] = mesh->tVerts[tvf->getTVert(0)];
        tverts[1] = mesh->tVerts[tvf->getTVert(1)];
        tverts[2] = mesh->tVerts[tvf->getTVert(2)];

        uvw = baryCoord.x * tverts[0] + baryCoord.y * tverts[1] + baryCoord.z * tverts[2];
    }

    return uvw;
}

static Point3 basic_tva[3] = { Point3(0.0, 0.0, 0.0), Point3(1.0, 0.0, 0.0), Point3(1.0, 1.0, 0.0) };
static Point3 basic_tvb[3] = { Point3(1.0, 1.0, 0.0), Point3(0.0, 1.0, 0.0), Point3(0.0, 0.0, 0.0) };
static int nextpt[3] = { 1, 2, 0 };
static int prevpt[3] = { 2, 0, 1 };

void MakeFaceUV(Face *f, UVVert *tv)
{
    int na, nhid, i;
    Point3 *basetv;
    /* make the invisible edge be 2->0 */
    nhid = 2;
    if (!(f->flags & EDGE_A))
        nhid = 0;
    else if (!(f->flags & EDGE_B))
        nhid = 1;
    else if (!(f->flags & EDGE_C))
        nhid = 2;
    na = 2 - nhid;
    basetv = (f->v[prevpt[nhid]] < f->v[nhid]) ? basic_tva : basic_tvb;
    for (i = 0; i < 3; i++)
    {
        tv[i] = basetv[na];
        na = nextpt[na];
    }
}

void SContext::getTVerts(int chan)
{
    if (chan == 0 && (node->GetMtl()->Requirements(mtlNum) & MTLREQ_FACEMAP))
    {
        MakeFaceUV(&mesh->faces[faceNum], tv[0]);
    }
    else
    {
        Mesh *m = mesh;
        if (chan == 0)
        {
            UVVert *tverts;
            TVFace *tvf;
            tverts = m->tVerts;
            tvf = m->tvFace;
            if (tverts == 0 || tvf == 0)
                return;
            tvf = &tvf[faceNum];
            tv[0][0] = tverts[tvf->t[0]];
            tv[0][1] = tverts[tvf->t[1]];
            tv[0][2] = tverts[tvf->t[2]];
        }
        else
        {
            VertColor *vc;
            TVFace *tvf;
            vc = m->vertCol;
            tvf = m->vcFace;
            if (vc == 0 || tvf == 0)
                return;
            tvf = &tvf[faceNum];
            tv[1][0] = vc[tvf->t[0]];
            tv[1][1] = vc[tvf->t[1]];
            tv[1][2] = vc[tvf->t[2]];
        }
    }
}

void SContext::getObjVerts()
{
    // TBD
}

// Returns the UVW derivatives for the point.
Point3 SContext::DUVW(int chan)
{
    getTVerts(chan);
    calc_size_ratio();
    return 0.5f * (pabs(tv[chan][1] - tv[chan][0]) + pabs(tv[chan][2] - tv[chan][0])) * ratio;
}

// This returns the bump basis vectors for UVW in camera space.
void SContext::DPdUVW(Point3 dP[3], int chan)
{
    getTVerts(chan);
    calc_size_ratio();
    Point3 bv[3];
    getObjVerts();
    ComputeBumpVectors(tv[chan], obpos, bv);
    bumpv[chan][0] = Normalize(bv[0]);
    bumpv[chan][1] = Normalize(bv[1]);
    bumpv[chan][2] = Normalize(bv[2]);
    dP[0] = bumpv[chan][0];
    dP[1] = bumpv[chan][1];
    dP[2] = bumpv[chan][2];
}

//--------------------------------------------------------------------
// Computes the average curvature per unit surface distance in the face
//--------------------------------------------------------------------
float ComputeFaceCurvature(Point3 *n, Point3 *v, Point3 bc)
{
    Point3 nc = (n[0] + n[1] + n[2]) / 3.0f;
    Point3 dn0 = n[0] - nc;
    Point3 dn1 = n[1] - nc;
    Point3 dn2 = n[2] - nc;
    Point3 c = (v[0] + v[1] + v[2]) / 3.0f;
    Point3 v0 = v[0] - c;
    Point3 v1 = v[1] - c;
    Point3 v2 = v[2] - c;
    float d0 = DotProd(dn0, v0) / LengthSquared(v0);
    float d1 = DotProd(dn1, v1) / LengthSquared(v1);
    float d2 = DotProd(dn2, v2) / LengthSquared(v2);
    float ad0 = (float)fabs(d0);
    float ad1 = (float)fabs(d1);
    float ad2 = (float)fabs(d2);
    return (ad0 > ad1) ? (ad0 > ad2 ? d0 : d2) : ad1 > ad2 ? d1 : d2;
}

static inline float size_meas(Point3 a, Point3 b, Point3 c)
{
    double d = fabs(b.x - a.x);
    d += fabs(b.y - a.y);
    d += fabs(b.z - a.z);
    d += fabs(c.x - a.x);
    d += fabs(c.y - a.y);
    d += fabs(c.z - a.z);
    return float(d / 6.0);
}

// This is an estimate of how fast the normal is varying.
// For example if you are doing enviornment mapping this value may be used to
// determine how big an area of the environment to sample.
// If the normal is changing very fast a large area must be sampled otherwise
// you'll get aliasing.  This is an estimate of dN/dsx, dN/dsy put into a
// single value.
// Signed curvature:
float SContext::Curve()
{
    Point3 tpos[3];
    Face &f = mesh->faces[faceNum];
    tpos[0] = mesh->verts[f.v[0]];
    tpos[1] = mesh->verts[f.v[1]];
    tpos[2] = mesh->verts[f.v[2]];
    float d = ComputeFaceCurvature(vxNormals, tpos, baryCoord);
    curve = d * RayDiam();
    return backFace ? -curve : curve;
}

#define SZFACT 1.5f

// Approximate how big fragment is relative to whole face.
void SContext::calc_size_ratio()
{
    Point3 dp = DP();
    Point3 cv[3];
    cv[0] = *mesh->getVertPtr((&mesh->faces[faceNum])->v[0]);
    cv[1] = *mesh->getVertPtr((&mesh->faces[faceNum])->v[1]);
    cv[2] = *mesh->getVertPtr((&mesh->faces[faceNum])->v[2]);
    float d = size_meas(cv[0], cv[1], cv[2]);
    ratio = SZFACT * (float)fabs(dp.x) / d;
}

int SContext::InMtlEditor()
{
    return FALSE;
}

void SContext::SetView(Point3 v)
{
    viewPoint = v;
}

/****************************************************************************
// Shadow buffer
 ***************************************************************************/

ShadowBuffer *RContext::NewShadowBuffer() const
{
    return NULL;
}

ShadowQuadTree *RContext::NewShadowQuadTree() const
{
    return NULL;
}

Color RContext::GlobalLightLevel() const
{
    return Color(1, 1, 1); // TBD
}

/****************************************************************************
// Scan the scene for all lights and add them for the ShadeContext's lightTab
 ***************************************************************************/

void sceneLightEnum(INode *node, SContext *sc, MtlBaseLib *mtls)
{
    // For each child of this node, we recurse into ourselves
    // until no more children are found.
    for (int c = 0; c < node->NumberOfChildren(); c++)
    {
        sceneLightEnum(node->GetChildNode(c), sc, mtls);
    }

    // Get the ObjectState.
    // The ObjectState is the structure that flows up the pipeline.
    // It contains a matrix, a material index, some flags for channels,
    // and a pointer to the object in the pipeline.
    ObjectState ostate = node->EvalWorldState(0);
    if (ostate.obj == NULL)
        return;

    // Examine the superclass ID in order to figure out what kind
    // of object we are dealing with.
    if (ostate.obj->SuperClassID() == LIGHT_CLASS_ID)
    {
        // Get the light object from the ObjectState
        LightObject *light = (LightObject *)ostate.obj;

        // Is this light turned on?
        if (light->GetUseLight())
        {
            // Create a RenderLight and append it to our list of lights
            // to fix compiler error
            LightInfo *li = new LightInfo(node, mtls);
            sc->lightTab.Append(1, &(li));
            //sc->lightTab.Append(1, &(new LightInfo(node, mtls)));
        }
    }
}

void AddSceneLights(SContext *sc, MtlBaseLib *mtls)
{
    INode *scene = GetCOREInterface()->GetRootNode();
    for (int i = 0; i < scene->NumberOfChildren(); i++)
    {
        sceneLightEnum(scene->GetChildNode(i), sc, mtls);
    }
}

/****************************************************************************
// Material enumerator functions
// Before evaluating a material we need to load the maps used by the material
// and then tell the material to prepare for evaluation.
 ***************************************************************************/
#if MAX_PRODUCT_VERSION_MAJOR > 11
class CheckFileNames : public AssetEnumCallback
{
public:
    NameTab *missingMaps;
    BitmapInfo bi;
    CheckFileNames(NameTab *n);
    void RecordAsset(const MaxSDK::AssetManagement::AssetUser &asset);
};

//***************************************************************************
// Add a name to the list if it's not already there
//***************************************************************************

void CheckFileNames::RecordAsset(const MaxSDK::AssetManagement::AssetUser &asset)
{

    MSTR name = asset.GetFullFilePath();
    if (name)
    {
        if (name[0] != 0)
        {
            if (missingMaps->FindName(name) < 0)
            {
                missingMaps->AddName(name);
            }
        }
    }
}
#else
class CheckFileNames : public NameEnumCallback
{
public:
    NameTab *missingMaps;
    BitmapInfo bi;
    CheckFileNames(NameTab *n);
    void RecordName(TCHAR *name);
};

//***************************************************************************
// Add a name to the list if it's not already there
//***************************************************************************

void CheckFileNames::RecordName(TCHAR *name)
{
    if (name)
    {
        if (name[0] != 0)
        {
            if (missingMaps->FindName(name) < 0)
            {
                missingMaps->AddName(name);
            }
        }
    }
}
#endif

//***************************************************************************
// Class to manage names of missing maps
//***************************************************************************

CheckFileNames::CheckFileNames(NameTab *n)
{
    missingMaps = n;
}

class MtlEnum
{
public:
    virtual int proc(MtlBase *m, int subMtlNum) = 0;
};

class MapLoadEnum : public MtlEnum
{
public:
    TimeValue t;

    MapLoadEnum(TimeValue time);
    virtual int proc(MtlBase *m, int subMtlNum);
};

//***************************************************************************
// Constructor of map loader
//***************************************************************************

MapLoadEnum::MapLoadEnum(TimeValue time)
{
    t = time;
}

//***************************************************************************
// Map loader enum proc
//***************************************************************************

int MapLoadEnum::proc(MtlBase *m, int subMtlNum)
{
    Texmap *tm = (Texmap *)m;
    tm->LoadMapFiles(t);
    return 1;
}

int EnumMaps(MtlBase *mb, int subMtl, MtlEnum &tenum)
{
    if (IsTex(mb))
    {
        if (!tenum.proc(mb, subMtl))
        {
            return 0;
        }
    }
    int i;
    for (i = 0; i < mb->NumSubTexmaps(); i++)
    {
        Texmap *st = mb->GetSubTexmap(i);
        if (st)
        {
            int subm = (mb->IsMultiMtl() && subMtl < 0) ? i : subMtl;
            if (mb->SubTexmapOn(i))
            {
                if (!EnumMaps(st, subm, tenum))
                {
                    return 0;
                }
            }
        }
    }
    if (IsMtl(mb))
    {
        Mtl *m = (Mtl *)mb;
        for (i = 0; i < m->NumSubMtls(); i++)
        {
            Mtl *sm = m->GetSubMtl(i);
            if (sm)
            {
                int subm = (mb->IsMultiMtl() && subMtl < 0) ? i : subMtl;
                if (!EnumMaps(sm, subm, tenum))
                {
                    return 0;
                }
            }
        }
    }
    return 1;
}

void EnumRefs(ReferenceMaker *rm, RefEnumProc &proc)
{
    proc.proc(rm);
    for (int i = 0; i < rm->NumRefs(); i++)
    {
        ReferenceMaker *srm = rm->GetReference(i);
        if (srm)
        {
            EnumRefs(srm, proc);
        }
    }
}

//***************************************************************************
// Constructor of map enumerator
//***************************************************************************

GetMaps::GetMaps(MtlBaseLib *mbl)
{
    mlib = mbl;
}

//***************************************************************************
// Implementation of the map enumerator
//***************************************************************************

#if MAX_PRODUCT_VERSION_MAJOR > 8
int GetMaps::proc(ReferenceMaker *rm)
#else
void GetMaps::proc(ReferenceMaker *rm)
#endif
{
    if (IsTex((MtlBase *)rm))
    {
        mlib->AddMtl((MtlBase *)rm);
    }
#if MAX_PRODUCT_VERSION_MAJOR > 8
    return REF_ENUM_CONTINUE;
#endif
}

int LoadMapFiles(INode *node, SContext *sc, MtlBaseLib &mtls, TimeValue t)
{
    NameTab mapFiles;
    CheckFileNames checkNames(&mapFiles);

    node->EnumAuxFiles(checkNames, FILE_ENUM_MISSING_ONLY | FILE_ENUM_1STSUB_MISSING);

    // Check the lights
    int i;
    for (i = 0; i < sc->lightTab.Count(); i++)
    {
        if (((LightInfo *)sc->lightTab[i])->light != NULL)
        {
            ((LightInfo *)sc->lightTab[i])->light->EnumAuxFiles(checkNames, FILE_ENUM_MISSING_ONLY | FILE_ENUM_1STSUB_MISSING);
        }
    }

    if (mapFiles.Count())
    {
        // Error! Missing maps.
        // not sure how to handle this so we gladly continue.

        //if (MessageBox(hWnd, "There are missing maps.\nDo you want to render anyway?", "Warning!", MB_YESNO) != IDYES) {
        //	return 0;
        //}
    }

    // Load the maps
    MapLoadEnum mapload(t);
    for (i = 0; i < mtls.Count(); i++)
    {
        EnumMaps(mtls[i], -1, mapload);
    }

    return 1;
}

//***************************************************************************
// This material is used when a node does not have a material assigned.
//***************************************************************************

DumMtl::DumMtl(Color c)
{
    diff = c;
    spec = Color(DUMSPEC, DUMSPEC, DUMSPEC);
    phongexp = (float)pow(2.0, DUMSHINE * 10.0);
}

void DumMtl::Update(TimeValue t, Interval &valid)
{
}

void DumMtl::Reset()
{
}

Interval DumMtl::Validity(TimeValue t)
{
    return FOREVER;
}

ParamDlg *DumMtl::CreateParamDlg(HWND hwMtlEdit, IMtlParams *imp)
{
    return NULL;
}

Color DumMtl::GetAmbient(int mtlNum, BOOL backFace)
{
    return diff;
}

Color DumMtl::GetDiffuse(int mtlNum, BOOL backFace)
{
    return diff;
}

Color DumMtl::GetSpecular(int mtlNum, BOOL backFace)
{
    return spec;
}

float DumMtl::GetShininess(int mtlNum, BOOL backFace)
{
    return 0.0f;
}

float DumMtl::GetShinStr(int mtlNum, BOOL backFace)
{
    return 0.0f;
}

float DumMtl::GetXParency(int mtlNum, BOOL backFace)
{
    return 0.0f;
}

void DumMtl::SetAmbient(Color c, TimeValue t)
{
}

void DumMtl::SetDiffuse(Color c, TimeValue t)
{
}

void DumMtl::SetSpecular(Color c, TimeValue t)
{
}

void DumMtl::SetShininess(float v, TimeValue t)
{
}

Class_ID DumMtl::ClassID()
{
    return DUMMTL_CLASS_ID;
}

void DumMtl::DeleteThis()
{
    delete this;
}

#if MAX_PRODUCT_VERSION_MAJOR > 16
RefResult DumMtl::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget,
                                   PartID &partID, RefMessage message, BOOL propagate)
#else
RefResult DumMtl::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                                   PartID &partID, RefMessage message)
#endif
{
    return REF_SUCCEED;
}

//***************************************************************************
// Shade method for the dummy material
// If a node does not have a material assigned we create
// a dummy material that inherits the wireframe color of
// the node
//***************************************************************************

void DumMtl::Shade(ShadeContext &sc)
{
    Color lightCol;
    Color diffwk(0.0f, 0.0f, 0.0f);
    Color specwk(0.0f, 0.0f, 0.0f);
    Point3 N = sc.Normal();
    Point3 R = sc.ReflectVector();
    LightDesc *l;
    for (int i = 0; i < sc.nLights; i++)
    {
        l = sc.Light(i);
        register float NL, diffCoef;
        Point3 L;
        if (!l->Illuminate(sc, N, lightCol, L, NL, diffCoef))
            continue;

        // diffuse
        if (l->affectDiffuse)
            diffwk += diffCoef * lightCol;
        // specular
        if (l->affectSpecular)
        {
            float c = DotProd(L, R);
            if (c > 0.0f)
            {
                c = (float)pow((double)c, (double)phongexp);
                specwk += c * lightCol * NL; // multiply by NL to SOFTEN
            }
        }
    }
    sc.out.t = Color(0.0f, 0.0f, 0.0f);
    sc.out.c = (.3f * sc.ambientLight + diffwk) * diff + specwk * spec;
}
