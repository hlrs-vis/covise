/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
	FILE: vrmlexp.cpp

	DESCRIPTION:  VRML/VRBL .WRL file export module

	CREATED BY: Scott Morrison

	HISTORY: created 15 February, 1996

 *>	Copyright (c) 1996, 1997 All Rights Reserved.
 **********************************************************************/

#include <time.h>
#include "vrml.h"
#include "simpobj.h"
#include "istdplug.h"
#include "inline.h"
#include "COVISEObject.h"
#include "lod.h"
#include "inlist.h"
#include "notetrck.h"
#include "bookmark.h"
#include "stdmat.h"
#include "normtab.h"
#include "vrml_api.h"
#include "vrmlexp.h"
#include "appd.h"
#include "timer.h"
#include "navinfo.h"
#include "backgrnd.h"
#include "fog.h"
#include "sky.h"
#include "sound.h"
#include "touch.h"
#include "prox.h"
#include "vrml2.h"
#include "helpsys.h"
#include "defuse.h"

#include <windows.h>
#include <Winuser.h>

#if MAX_PRODUCT_VERSION_MAJOR > 14
#define STRTOUTF8(x) x.ToUTF8().data()
#else
#define STRTOUTF8(x) x
#endif

extern TCHAR *GetString(int id);

// Returns TRUE if an object or one of its ancestors in animated
static BOOL IsEverAnimated(INode *node);

// Round numbers near zero to zero.  This help reduce VRML file size.
inline float
round(float f)
{
    // This is used often, so we avoid calling fabs
    if (f < 0.0f)
    {
        if (f > -1.0e-5)
            return 0.0f;
        return f;
    }
    if (f < 1.0e-5)
        return 0.0f;
    return f;
}

// Function for writing values into a string for output.
// These functions take care of rounding values near zero, and flipping
// Y and Z for VRML output.

// Format a 3D coordinate value.
TCHAR *
VRBLExport::point(Point3 &p)
{
    static TCHAR buf[50];
    TCHAR format[20];
    _stprintf(format, _T("%%.%dg %%.%dg %%.%dg"), mDigits, mDigits, mDigits);
    if (mZUp)
        _stprintf(buf, format, round(p.x), round(p.y), round(p.z));
    else
        _stprintf(buf, format, round(p.x), round(p.z), round(-p.y));
    CommaScan(buf);
    return buf;
}

TCHAR *
VRBLExport::color(Color &c)
{
    static TCHAR buf[50];
    TCHAR format[20];
    _stprintf(format, _T("%%.%dg %%.%dg %%.%dg"), mDigits, mDigits, mDigits);
    _stprintf(buf, format, round(c.r), round(c.g), round(c.b));
    CommaScan(buf);
    return buf;
}

TCHAR *
VRBLExport::color(Point3 &c)
{
    static TCHAR buf[50];
    TCHAR format[20];
    _stprintf(format, _T("%%.%dg %%.%dg %%.%dg"), mDigits, mDigits, mDigits);
    _stprintf(buf, format, round(c.x), round(c.y), round(c.z));
    CommaScan(buf);
    return buf;
}

TCHAR *
VRBLExport::floatVal(float f)
{
    static TCHAR buf[50];
    TCHAR format[20];
    _stprintf(format, _T("%%.%dg"), mDigits);
    _stprintf(buf, format, round(f));
    CommaScan(buf);
    return buf;
}

TCHAR *
VRBLExport::texture(UVVert &uv)
{
    static TCHAR buf[50];
    TCHAR format[20];
    _stprintf(format, _T("%%.%dg %%.%dg"), mDigits, mDigits, mDigits);
    _stprintf(buf, format, round(uv.x), round(uv.y));
    CommaScan(buf);
    return buf;
}

// Format a scale value
TCHAR *
VRBLExport::scalePoint(Point3 &p)
{
    static TCHAR buf[50];
    TCHAR format[20];
    _stprintf(format, _T("%%.%dg %%.%dg %%.%dg"), mDigits, mDigits, mDigits);
    //if (mZUp)
    //    _stprintf(buf, format, round(p.x), round( p.y), round(p.z));
    //else
    _stprintf(buf, format, round(p.x), round(p.z), round(p.y));
    CommaScan(buf);
    return buf;
}

// Format a normal vector
TCHAR *
VRBLExport::normPoint(Point3 &p)
{
    static TCHAR buf[50];
    TCHAR format[20];
    _stprintf(format, _T("%%.%dg %%.%dg %%.%dg"), mDigits, mDigits, mDigits);
    if (mZUp)
        _stprintf(buf, format, round(p.x), round(p.y), round(p.z));
    else
        _stprintf(buf, format, round(p.x), round(p.z), round(-p.y));
    CommaScan(buf);
    return buf;
}

// Format an axis value
TCHAR *
VRBLExport::axisPoint(Point3 &p, float angle)
{
    static TCHAR buf[50];
    TCHAR format[20];
    _stprintf(format, _T("%%.%dg %%.%dg %%.%dg %%.%dg"),
              mDigits, mDigits, mDigits, mDigits);
    if (mZUp)
        _stprintf(buf, format, round(p.x), round(p.y), round(p.z),
                  round(angle));
    else
        _stprintf(buf, format, round(p.x), round(p.z), round(-p.y),
                  round(angle));
    CommaScan(buf);
    return buf;
}

// Get the tranform matrix that take a point from its local coordinate
// system to it's parent's coordinate system
static Matrix3
GetLocalTM(INode *node, TimeValue t)
{
    Matrix3 tm;
    tm = node->GetObjTMAfterWSM(t);
    if (!node->GetParentNode()->IsRootNode())
    {
        Matrix3 ip = Inverse(node->GetParentNode()->GetObjTMAfterWSM(t));
        tm = tm * ip;
    }
    return tm;
}

class VRBLClassDesc : public ClassDesc
{
public:
    int IsPublic() { return TRUE; }
    void *Create(BOOL loading = FALSE) { return new VRBLExport; }
    const TCHAR *ClassName() { return GetString(IDS_VRML_EXPORT_CLASS); }
    SClass_ID SuperClassID() { return SCENE_EXPORT_CLASS_ID; }
    Class_ID ClassID() { return Class_ID(VRBL_EXPORT_CLASS_ID, 0); }
    const TCHAR *Category() { return GetString(IDS_TH_SCENEEXPORT); }
};

static VRBLClassDesc VRBLDesc;

ClassDesc *GetVRBLDesc() { return &VRBLDesc; }

////////////////////////////////////////////////////////////////////////
// VRBL Export implementation
////////////////////////////////////////////////////////////////////////

// Indent to the given level.
void
VRBLExport::Indent(int level)
{
    if (!mIndent)
        return;
    assert(level >= 0);
    for (; level; level--)
        MSTREAMPRINTF  _T("  "));
}

extern TCHAR *VRMLName(const TCHAR *name);
// Translates name (if necessary) to VRML compliant name.
// Returns name in static buffer, so calling a second time trashes
// the previous contents.
#define CTL_CHARS 31
TCHAR *VRMLName(const TCHAR *name)
{
    static TCHAR buffer[256];
    TCHAR *cPtr;
    int firstCharacter = 1;

    _tcscpy(buffer, name);
    cPtr = buffer;
    while (*cPtr)
    {
        if (*cPtr <= CTL_CHARS || *cPtr == ' ' || *cPtr == '\'' || *cPtr == '"' || *cPtr == '\\' || *cPtr == '{' || *cPtr == '}' || *cPtr == ',' || *cPtr == '.' || *cPtr == '[' || *cPtr == ']' || *cPtr == '.' || *cPtr == '#' || *cPtr >= 127 || (firstCharacter && (*cPtr >= '0' && *cPtr <= '9')))
            *cPtr = '_';
        firstCharacter = 0;
        cPtr++;
    }

    return buffer;
}

// Return true if it has a mirror transform
BOOL
VRBLExport::IsMirror(INode *node)
{
    Matrix3 tm = GetLocalTM(node, mStart);
    AffineParts parts;
    decomp_affine(tm, &parts);

    return parts.f < 0.0f;
}

// Write beginning of the Separator node.
void
VRBLExport::StartNode(INode *node, Object *obj, int level, BOOL outputName)
{
    if (node->IsRootNode())
    {
        MSTREAMPRINTF  _T("Separator {\n"));
        return;
    }
    Indent(level);
    if (obj->SuperClassID() == CAMERA_CLASS_ID || obj->SuperClassID() == LIGHT_CLASS_ID || !outputName)
    {
        // Lights and cameras need different top-level names for triggers
        MSTREAMPRINTF  _T("DEF %s_TopLevel Separator {\n"), mNodes.GetNodeName(node));
    }
    else
    {
        MSTREAMPRINTF  _T("DEF %s Separator {\n"), mNodes.GetNodeName(node));
        if (IsMirror(node))
        {
            Indent(level + 1);
            MSTREAMPRINTF  _T("ShapeHints {\n"));
            Indent(level + 2);
            MSTREAMPRINTF  _T("vertexOrdering CLOCKWISE\n"));
            Indent(level + 1);
            MSTREAMPRINTF  _T("}\n"));
        }
    }

    if (!obj)
        return;

    // Put note tracks as info nodes
    int numNotes = node->NumNoteTracks();
    for (int i = 0; i < numNotes; i++)
    {
        DefNoteTrack *nt = (DefNoteTrack *)node->GetNoteTrack(i);
        for (int j = 0; j < nt->keys.Count(); j++)
        {
            NoteKey *nk = nt->keys[j];
            TSTR note = nk->note;
            if (note.Length() > 0)
            {
                Indent(level + 1);
                MSTREAMPRINTF  _T("Info { string \"frame %d: %s\" }\n"),
                        nk->time/GetTicksPerFrame(), note.data());
            }
        }
    }
}

// Write end of the separator node.
void
VRBLExport::EndNode(INode *node, int level, BOOL lastChild)
{
    Indent(level);
    MSTREAMPRINTF  _T("}\n"));
}

// Write out the transform from the local coordinate system to the
// parent coordinate system
void
VRBLExport::OutputNodeTransform(INode *node, int level)
{
    // Root node is always identity
    if (node->IsRootNode())
        return;

    Matrix3 tm = GetLocalTM(node, mStart);
    int i, j;
    Point3 p;

    // Check for scale and rotation part of matrix being identity.
    BOOL isIdentity = TRUE;
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            if (i == j)
            {
                if (tm.GetRow(i)[j] != 1.0f)
                {
                    isIdentity = FALSE;
                    goto done;
                }
            }
            else if (fabs(tm.GetRow(i)[j]) > 1.0e-05)
            {
                isIdentity = FALSE;
                goto done;
            }
        }
    }

done:
    if (isIdentity)
    {
        p = tm.GetTrans();
        Indent(level);
        MSTREAMPRINTF  _T("Translation { translation %s }\n"), point(p));
    }
    else
    {
        // If not identity, decompose matrix into scale, rotation and
        // translation components.
        Point3 s, axis;
        Quat q;
        float ang;

        AffineParts parts;
        decomp_affine(tm, &parts);
        p = parts.t;
        q = parts.q;
        AngAxisFromQ(q, &ang, axis);
        Indent(level);
        MSTREAMPRINTF  _T("Translation { translation %s }\n"), point(p));
        if (ang != 0.0f && ang != -0.0f)
        {
            Indent(level);
            // VRML angle convention is opposite of MAX convention,
            // so we negate the angle.
            MSTREAMPRINTF  _T("Rotation { rotation %s }\n"),
                    axisPoint(axis, -ang));
        }
        ScaleValue sv(parts.k, parts.u);
        s = sv.s;
        if (parts.f < 0.0f)
            s = -s;
        if (s.x != 1.0f || s.y != 1.0f || s.z != 1.0f)
        {
            //double tmpy= s.y;
            //s.y=s.z;
            //s.z=tmpy; // axes are different in Inventor so exchange y and z
            Indent(level);
            MSTREAMPRINTF  _T("Scale { scaleFactor %s }\n"), scalePoint(s));
        }
    }
}

// Initialize the normal table
NormalTable::NormalTable()
{
    tab.SetCount(NORM_TABLE_SIZE);
    for (int i = 0; i < NORM_TABLE_SIZE; i++)
        tab[i] = NULL;
}

NormalTable::~NormalTable()
{
    // Delete the buckets in the normal table hash table.
    for (int i = 0; i < NORM_TABLE_SIZE; i++)
        delete tab[i];
}

// Add a normal to the hash table
void
NormalTable::AddNormal(Point3 &norm)
{
    // Truncate normals to a value that brings close normals into
    // the same bucket.
    Point3 n = NormalizeNorm(norm);
    DWORD code = HashCode(n);
    NormalDesc *nd;
    for (nd = tab[code]; nd; nd = nd->next)
    {
        if (nd->n == n) // Equality OK because of normalization procedure.
            return;
    }
    NormalDesc *newNorm = new NormalDesc(norm);
    newNorm->next = tab[code];
    tab[code] = newNorm;
}

// Get the index of a normal in the IndexedFaceSet
int
NormalTable::GetIndex(Point3 &norm)
{
    Point3 n = NormalizeNorm(norm);
    DWORD code = HashCode(n);
    NormalDesc *nd;
    for (nd = tab[code]; nd; nd = nd->next)
    {
        if (nd->n == n)
            return nd->index;
    }
    return -1;
}

// Produce a hash code for a normal
DWORD
NormalTable::HashCode(Point3 &norm)
{
    union
    {
        float p[3];
        DWORD i[3];
    } u;
    u.p[0] = norm.x;
    u.p[1] = norm.y;
    u.p[2] = norm.z;
    return ((u.i[0] >> 8) + (u.i[1] >> 16) + u.i[2]) % NORM_TABLE_SIZE;
}

// Print the hash table statistics for the normal table
void
NormalTable::PrintStats(MAXSTREAM mStream)
{
    int slots = 0;
    int buckets = 0;
    int i;
    NormalDesc *nd;

    for (i = 0; i < NORM_TABLE_SIZE; i++)
    {
        if (tab[i])
        {
            slots++;
            for (nd = tab[i]; nd; nd = nd->next)
                buckets++;
        }
    }
    MSTREAMPRINTF _T("# slots = %d, buckets = %d, avg. chain length = %.5g\n"),
            slots, buckets, ((double) buckets / (double) slots));
}

// Returns true IFF the mesh is all in the same smoothing group
static BOOL
MeshIsAllOneSmoothingGroup(Mesh &mesh)
{
    int numfaces = mesh.getNumFaces();
    unsigned int sg;
    int i;

    for (i = 0; i < numfaces; i++)
    {
        if (i == 0)
        {
            sg = mesh.faces[i].getSmGroup();
            if (sg == 0) // Smoothing group of 0 means faceted
                return FALSE;
        }
        else
        {
            if (sg != mesh.faces[i].getSmGroup())
                return FALSE;
        }
    }
    return TRUE;
}

// Write out the indices of the normals for the IndexedFaceSet
void
VRBLExport::OutputNormalIndices(Mesh &mesh, NormalTable *normTab, int level)
{
    Point3 n;
    int numfaces = mesh.getNumFaces();
    int i, j, v, norCnt;

    Indent(level);

    MSTREAMPRINTF  _T("normalIndex [\n"));
    for (i = 0; i < numfaces; i++)
    {
        int smGroup = mesh.faces[i].getSmGroup();
        Indent(level + 1);
        for (v = 0; v < 3; v++)
        {
            int cv = mesh.faces[i].v[v];
            RVertex *rv = mesh.getRVertPtr(cv);
            if (rv->rFlags & SPECIFIED_NORMAL)
            {
                n = rv->rn.getNormal();
                continue;
            }
            else if ((norCnt = (int)(rv->rFlags & NORCT_MASK)) && smGroup)
            {
                if (norCnt == 1)
                    n = rv->rn.getNormal();
                else
                    for (j = 0; j < norCnt; j++)
                    {
                        if (rv->ern[j].getSmGroup() & smGroup)
                        {
                            n = rv->ern[j].getNormal();
                            break;
                        }
                    }
            }
            else
                n = mesh.getFaceNormal(i);
            int index = normTab->GetIndex(n);
            assert(index != -1);
            MSTREAMPRINTF  _T("%d, "), index);
        }
        MSTREAMPRINTF  _T("-1,\n"));
    }
    Indent(level);
    MSTREAMPRINTF  _T("]\n"));
}

// Create the hash table of normals for the given mesh, and
// write out the normal values.
NormalTable *
VRBLExport::OutputNormals(Mesh &mesh, int level)
{
    int i, j, norCnt;
    int numverts = mesh.getNumVerts();
    int numfaces = mesh.getNumFaces();
    NormalTable *normTab;

    mesh.buildRenderNormals();

    if (MeshIsAllOneSmoothingGroup(mesh))
    {
        // No need for normals when whole object is smooth.
        // VRML Browsers compute normals automatically in this case.
        return NULL;
    }

    normTab = new NormalTable();

    // Otherwise we have several smoothing groups
    int index;
    for (index = 0; index < numfaces; index++)
    {
        int smGroup = mesh.faces[index].getSmGroup();
        for (i = 0; i < 3; i++)
        {
            // Now get the normal for each vertex of the face
            // Given the smoothing group
            int cv = mesh.faces[index].v[i];
            RVertex *rv = mesh.getRVertPtr(cv);
            if (rv->rFlags & SPECIFIED_NORMAL)
            {
                normTab->AddNormal(rv->rn.getNormal());
            }
            else if ((norCnt = (int)(rv->rFlags & NORCT_MASK)) && smGroup)
            {
                if (norCnt == 1) // 1 normal, stored in rn
                    normTab->AddNormal(rv->rn.getNormal());
                else
                    for (j = 0; j < norCnt; j++)
                    {
                        // More than one normal, stored in ern.
                        normTab->AddNormal(rv->ern[j].getNormal());
                    }
            }
            else
                normTab->AddNormal(mesh.getFaceNormal(index));
        }
    }

    // Now write out the table
    index = 0;
    NormalDesc *nd;
    Indent(level);
    MSTREAMPRINTF  _T("Normal { vector [\n"));

    for (i = 0; i < NORM_TABLE_SIZE; i++)
    {
        for (nd = normTab->Get(i); nd; nd = nd->next)
        {
            nd->index = index++;
            Indent(level + 1);
            Point3 p = nd->n / NUM_NORMS;
            MSTREAMPRINTF  _T("%s,\n"), normPoint(p));
        }
    }
    Indent(level);
    MSTREAMPRINTF  _T("] }\n"));

    Indent(level);
    MSTREAMPRINTF  _T("NormalBinding { value PER_VERTEX_INDEXED }\n"));

#ifdef DEBUG_NORM_HASH
    normTab->PrintStats(mStream);
#endif

    return normTab;
}

// Write out the data for a single triangle mesh
void
VRBLExport::OutputTriObject(INode *node, TriObject *obj, BOOL isMulti,
                            BOOL twoSided, int level)
{
    assert(obj);
    Mesh &mesh = obj->GetMesh();
    int numverts = mesh.getNumVerts();
    int numtverts = mesh.getNumTVerts();
    int numfaces = mesh.getNumFaces();
    int i;
    NormalTable *normTab = NULL;
    TextureDesc *td = GetMatTex(node);

    if (numfaces == 0)
    {
        delete td;
        return;
    }

    if (isMulti)
    {
        Indent(level);
        MSTREAMPRINTF  _T("MaterialBinding { value PER_FACE_INDEXED }\n"));
    }

    // Output the vertices
    Indent(level);
    MSTREAMPRINTF  _T("Coordinate3 { point [\n"));

    for (i = 0; i < numverts; i++)
    {
        Point3 p = mesh.verts[i];
        Indent(level + 1);
        MSTREAMPRINTF  _T("%s"), point(p));
        if (i == numverts - 1)
        {
            MSTREAMPRINTF  _T("]\n"));
            Indent(level);
            MSTREAMPRINTF  _T("}\n"));
        }
        else
            MSTREAMPRINTF  _T(",\n"));
    }

    // Output the normals
    if (mGenNormals)
    {
        normTab = OutputNormals(mesh, level);
    }

    // Output Texture coordinates
    if (numtverts > 0 && td)
    {
        Indent(level);
        MSTREAMPRINTF  _T("TextureCoordinate2 { point [\n"));

        for (i = 0; i < numtverts; i++)
        {
            UVVert p = mesh.getTVert(i);
            Indent(level + 1);
            MSTREAMPRINTF  _T("%s"), texture(p));
            if (i == numtverts - 1)
            {
                MSTREAMPRINTF  _T("]\n"));
                Indent(level);
                MSTREAMPRINTF  _T("}\n"));
            }
            else
                MSTREAMPRINTF  _T(",\n"));
        }
    }

    if (twoSided)
    {
        Indent(level);
        MSTREAMPRINTF  _T("ShapeHints {\n"));
        Indent(level + 1);
        MSTREAMPRINTF  _T("shapeType UNKNOWN_SHAPE_TYPE\n"));
        Indent(level);
        MSTREAMPRINTF  _T("}\n"));
    }
    // Output the triangles
    Indent(level);
    MSTREAMPRINTF  _T("IndexedFaceSet { coordIndex [\n"));
    for (i = 0; i < numfaces; i++)
    {
        if (!(mesh.faces[i].flags & FACE_HIDDEN))
        {
            Indent(level + 1);
            MSTREAMPRINTF  _T("%d, %d, %d, -1"), mesh.faces[i].v[0],
                    mesh.faces[i].v[1], mesh.faces[i].v[2]);
            if (i != numfaces - 1)
                MSTREAMPRINTF  _T(", \n"));
        }
    }
    MSTREAMPRINTF  _T("]\n"));

    // Output the texture coordinate indices
    if (numtverts > 0 && td)
    {
        Indent(level);
        MSTREAMPRINTF  _T("textureCoordIndex [\n"));
        for (i = 0; i < numfaces; i++)
        {
            if (!(mesh.faces[i].flags & FACE_HIDDEN))
            {
                Indent(level + 1);
                MSTREAMPRINTF  _T("%d, %d, %d, -1"), mesh.tvFace[i].t[0],
                        mesh.tvFace[i].t[1], mesh.tvFace[i].t[2]);
                if (i != numfaces - 1)
                MSTREAMPRINTF  _T(", \n"));
            }
        }
        MSTREAMPRINTF  _T("]\n"));
    }

    // Output the material indices
    if (isMulti)
    {
        Indent(level);
        MSTREAMPRINTF  _T("materialIndex [\n"));
        for (i = 0; i < numfaces; i++)
        {
            if (!(mesh.faces[i].flags & FACE_HIDDEN))
            {
                Indent(level + 1);
                MSTREAMPRINTF  _T("%d"), mesh.faces[i].getMatID());
                if (i != numfaces - 1)
                    MSTREAMPRINTF  _T(", \n"));
            }
        }
        MSTREAMPRINTF  _T("]\n"));
    }

    // Output the normal indices
    if (mGenNormals && normTab)
    {
        OutputNormalIndices(mesh, normTab, level);
        delete normTab;
    }

    Indent(level);
    MSTREAMPRINTF  _T("}\n"));
    delete td;
}

// Returns TRUE iff the node has an attached standard material with
// a texture map on the diffuse color
BOOL
VRBLExport::HasTexture(INode *node)
{
    TextureDesc *td = GetMatTex(node);
    if (!td)
        return FALSE;
    delete td;
    return TRUE;
}

// Get the name of the texture file of the texure on the diffuse
// color of the material attached to the given node.
TextureDesc *
VRBLExport::GetMatTex(INode *node)
{
    Mtl *mtl = node->GetMtl();
    if (!mtl)
        return NULL;

    // We only handle standard materials.
    if (mtl->ClassID() != Class_ID(DMTL_CLASS_ID, 0))
        return NULL;

    StdMat *sm = (StdMat *)mtl;
    // Check for texture map
    Texmap *tm = (BitmapTex *)sm->GetSubTexmap(ID_DI);
    if (!tm)
        return NULL;

    // We only handle bitmap textures in VRML
    if (tm->ClassID() != Class_ID(BMTEX_CLASS_ID, 0))
        return NULL;
    BitmapTex *bm = (BitmapTex *)tm;

    TSTR bitmapFile;
    TSTR fileName;

    bitmapFile = bm->GetMapName();
    if (bitmapFile.data() == NULL)
        return NULL;
    int l = bitmapFile.Length() - 1;
    if (l < 0)
        return NULL;

    // Split the name up
    TSTR path;
    SplitPathFile(bitmapFile, &path, &fileName);

    TSTR url;
    if (mUsePrefix && mUrlPrefix.Length() > 0)
    {
        if (mUrlPrefix[mUrlPrefix.Length() - 1] != '/')
        {
            TSTR slash = _T("/");
            url = mUrlPrefix + slash + fileName;
        }
        else
            url = mUrlPrefix + fileName;
    }
    else
        url = fileName;
    int cNum = 1;
    if (bm->GetTheUVGen())
        cNum = bm->GetTheUVGen()->GetMapChannel();
    TextureDesc *td = new TextureDesc(bm, fileName, url, cNum);
    return td;
}

// Write out the colors for a multi/sub-object material
void
VRBLExport::OutputMultiMtl(Mtl *mtl, int level)
{
    int i;
    Mtl *sub;
    Color c;
    float f;

    Indent(level);
    MSTREAMPRINTF  _T("Material {\n"));
    int num = mtl->NumSubMtls();

    Indent(level + 1);
    MSTREAMPRINTF  _T("ambientColor [ "));
    for (i = 0; i < num; i++)
    {
        sub = mtl->GetSubMtl(i);
        // Some slots might be empty!
        if (!sub)
            continue;
        c = sub->GetAmbient(mStart);
        if (i == num - 1)
            MSTREAMPRINTF  _T("%s "), color(c));
        else
            MSTREAMPRINTF  _T("%s, "), color(c));
    }
    MSTREAMPRINTF  _T("]\n"));
    Indent(level + 1);
    MSTREAMPRINTF  _T("diffuseColor [ "));
    for (i = 0; i < num; i++)
    {
        sub = mtl->GetSubMtl(i);
        if (!sub)
            continue;
        c = sub->GetDiffuse(mStart);
        if (i == num - 1)
            MSTREAMPRINTF  _T("%s "), color(c));
        else
            MSTREAMPRINTF  _T("%s, "), color(c));
    }
    MSTREAMPRINTF  _T("]\n"));

    Indent(level + 1);
    MSTREAMPRINTF  _T("specularColor [ "));
    for (i = 0; i < num; i++)
    {
        sub = mtl->GetSubMtl(i);
        if (!sub)
            continue;
        c = sub->GetSpecular(mStart);
        if (i == num - 1)
            MSTREAMPRINTF  _T("%s "), color(c));
        else
            MSTREAMPRINTF  _T("%s, "), color(c));
    }
    MSTREAMPRINTF  _T("]\n"));

    Indent(level + 1);
    MSTREAMPRINTF  _T("shininess [ "));
    for (i = 0; i < num; i++)
    {
        sub = mtl->GetSubMtl(i);
        if (!sub)
            continue;
        f = sub->GetShininess(mStart);
        if (i == num - 1)
            MSTREAMPRINTF  _T("%s "), floatVal(f));
        else
            MSTREAMPRINTF  _T("%s, "), floatVal(f));
    }
    MSTREAMPRINTF  _T("]\n"));

    Indent(level + 1);
    MSTREAMPRINTF  _T("emissiveColor [ "));
    for (i = 0; i < num; i++)
    {
        sub = mtl->GetSubMtl(i);
        if (!sub)
            continue;
        c = sub->GetDiffuse(mStart);
        float si;
        if (sub->ClassID() == Class_ID(DMTL_CLASS_ID, 0))
        {
            StdMat *stdMtl = (StdMat *)sub;
            si = stdMtl->GetSelfIllum(mStart);
        }
        else
            si = 0.0f;
        Point3 p = si * Point3(c.r, c.g, c.b);
        if (i == num - 1)
            MSTREAMPRINTF  _T("%s "), color(p));
        else
            MSTREAMPRINTF  _T("%s, "), color(p));
    }
    MSTREAMPRINTF  _T("]\n"));

    Indent(level);
    MSTREAMPRINTF  _T("}\n"));
}

void
VRBLExport::OutputNoTexture(int level)
{
    Indent(level);
    MSTREAMPRINTF  _T("Texture2 {}\n"));
}

// Output the matrial definition for a node.
BOOL
VRBLExport::OutputMaterial(INode *node, BOOL &twoSided, int level)
{
    Mtl *mtl = node->GetMtl();
    twoSided = FALSE;

    // If no material is assigned, use the wire color
    if (!mtl || (mtl->ClassID() != Class_ID(DMTL_CLASS_ID, 0) && !mtl->IsMultiMtl()))
    {
        Color col(node->GetWireColor());
        Indent(level);
        MSTREAMPRINTF  _T("Material {\n"));
        Indent(level + 1);
        MSTREAMPRINTF  _T("diffuseColor %s\n"), color(col));
        Indent(level + 1);
        MSTREAMPRINTF  _T("specularColor .9 .9 .9\n"));
        Indent(level);
        MSTREAMPRINTF  _T("}\n"));
        OutputNoTexture(level);
        return FALSE;
    }

    if (mtl->IsMultiMtl())
    {
        OutputMultiMtl(mtl, level);
        OutputNoTexture(level);
        return TRUE;
    }

    StdMat *sm = (StdMat *)mtl;
    twoSided = sm->GetTwoSided();
    Interval i = FOREVER;
    sm->Update(0, i);
    Indent(level);
    MSTREAMPRINTF  _T("Material {\n"));
    Color c;

    Indent(level + 1);
    c = sm->GetAmbient(mStart);
    MSTREAMPRINTF  _T("ambientColor %s\n"), color(c));
    Indent(level + 1);
    c = sm->GetDiffuse(mStart);
    MSTREAMPRINTF  _T("diffuseColor %s\n"), color(c));
    Indent(level + 1);
    c = sm->GetSpecular(mStart);
    MSTREAMPRINTF  _T("specularColor %s\n"), color(c));
    Indent(level + 1);
    MSTREAMPRINTF  _T("shininess %s\n"),
            floatVal(sm->GetShininess(mStart)));
    Indent(level + 1);
    MSTREAMPRINTF  _T("transparency %s\n"),
            floatVal(1.0f - sm->GetOpacity(mStart)));
    float si = sm->GetSelfIllum(mStart);
    if (si > 0.0f)
    {
        Indent(level + 1);
        c = sm->GetDiffuse(mStart);
        Point3 p = si * Point3(c.r, c.g, c.b);
        MSTREAMPRINTF  _T("emissiveColor %s\n"), color(p));
    }
    Indent(level);
    MSTREAMPRINTF  _T("}\n"));

    TextureDesc *td = GetMatTex(node);
    if (!td)
    {
        OutputNoTexture(level);
        return FALSE;
    }

    Indent(level);
    MSTREAMPRINTF  _T("Texture2 {\n"));
    Indent(level + 1);
    MSTREAMPRINTF  _T("filename \"%s\"\n"), td->url);
    Indent(level);
    MSTREAMPRINTF  _T("}\n"));

    BitmapTex *bm = td->tex;
    delete td;

    StdUVGen *uvGen = bm->GetUVGen();
    if (!uvGen)
    {
        return FALSE;
    }

    // Get the UV offset and scale value for Texture2Transform
    float uOff = uvGen->GetUOffs(mStart);
    float vOff = uvGen->GetVOffs(mStart);
    float uScl = uvGen->GetUScl(mStart);
    float vScl = uvGen->GetVScl(mStart);
    float ang = uvGen->GetAng(mStart);

    if (uOff == 0.0f && vOff == 0.0f && uScl == 1.0f && vScl == 1.0f && ang == 0.0f)
    {
        return FALSE;
    }

    Indent(level);
    MSTREAMPRINTF  _T("Texture2Transform {\n"));
    if (uOff != 0.0f || vOff != 0.0f)
    {
        Indent(level + 1);
        UVVert p = UVVert(uOff, vOff, 0.0f);
        MSTREAMPRINTF  _T("translation %s\n"), texture(p));
    }
    if (ang != 0.0f)
    {
        Indent(level + 1);
        MSTREAMPRINTF  _T("rotation %s\n"), floatVal(ang));
    }
    if (uScl != 1.0f || vScl != 1.0f)
    {
        Indent(level + 1);
        UVVert p = UVVert(uScl, vScl, 0.0f);
        MSTREAMPRINTF  _T("scaleFactor %s\n"), texture(p));
    }
    Indent(level);
    MSTREAMPRINTF  _T("}\n"));

    return FALSE;
}

// Create a VRMNL primitive sphere, if appropriate.
// Returns TRUE if a primitive is created
BOOL
VRBLExport::VrblOutSphere(INode *node, Object *obj, int level)
{
    SimpleObject *so = (SimpleObject *)obj;
    float radius, hemi;
    int basePivot, genUV, smooth;
    BOOL td = HasTexture(node);

    // Reject "base pivot" mapped, non-smoothed and hemisphere spheres
    so->pblock->GetValue(SPHERE_RECENTER, mStart, basePivot, FOREVER);
    so->pblock->GetValue(SPHERE_GENUVS, mStart, genUV, FOREVER);
    so->pblock->GetValue(SPHERE_HEMI, mStart, hemi, FOREVER);
    so->pblock->GetValue(SPHERE_SMOOTH, mStart, smooth, FOREVER);
    if (!smooth || basePivot || (genUV && td) || hemi > 0.0f)
        return FALSE;

    so->pblock->GetValue(SPHERE_RADIUS, mStart, radius, FOREVER);

    Indent(level);

    MSTREAMPRINTF  _T("Sphere { radius %s }\n"), floatVal(radius));

    return TRUE;
}

// Create a VRMNL primitive cylinder, if appropriate.
// Returns TRUE if a primitive is created
BOOL
VRBLExport::VrblOutCylinder(INode *node, Object *obj, int level)
{
    SimpleObject *so = (SimpleObject *)obj;
    float radius, height;
    int sliceOn, genUV, smooth;
    BOOL td = HasTexture(node);

    // Reject sliced, non-smooth and mapped cylinders
    so->pblock->GetValue(CYLINDER_GENUVS, mStart, genUV, FOREVER);
    so->pblock->GetValue(CYLINDER_SLICEON, mStart, sliceOn, FOREVER);
    so->pblock->GetValue(CYLINDER_SMOOTH, mStart, smooth, FOREVER);
    if (sliceOn || (genUV && td) || !smooth)
        return FALSE;

    so->pblock->GetValue(CYLINDER_RADIUS, mStart, radius, FOREVER);
    so->pblock->GetValue(CYLINDER_HEIGHT, mStart, height, FOREVER);
    Indent(level);
    MSTREAMPRINTF  _T("Separator {\n"));
    Indent(level + 1);
    if (mZUp)
    {
        MSTREAMPRINTF  _T("Rotation { rotation 1 0 0 %s }\n"),
                floatVal(float(PI/2.0)));
        Indent(level + 1);
        MSTREAMPRINTF  _T("Translation { translation 0 %s 0 }\n"),
                floatVal(float(height/2.0)));
    }
    else
    {
        Point3 p = Point3(0.0f, 0.0f, height / 2.0f);
        MSTREAMPRINTF  _T("Translation { translation %s }\n"), point(p));
    }
    Indent(level + 1);
    MSTREAMPRINTF  _T("Cylinder { radius %s "), floatVal(radius));
    MSTREAMPRINTF  _T("height %s }\n"), floatVal(float(fabs(height))));
    Indent(level);
    MSTREAMPRINTF  _T("}\n"));

    return TRUE;
}

// Create a VRMNL primitive cone, if appropriate.
// Returns TRUE if a primitive is created
BOOL
VRBLExport::VrblOutCone(INode *node, Object *obj, int level)
{
    SimpleObject *so = (SimpleObject *)obj;
    float radius1, radius2, height;
    int sliceOn, genUV, smooth;
    BOOL td = HasTexture(node);

    // Reject sliced, non-smooth and mappeded cones
    so->pblock->GetValue(CONE_GENUVS, mStart, genUV, FOREVER);
    so->pblock->GetValue(CONE_SLICEON, mStart, sliceOn, FOREVER);
    so->pblock->GetValue(CONE_SMOOTH, mStart, smooth, FOREVER);
    so->pblock->GetValue(CONE_RADIUS2, mStart, radius2, FOREVER);
    if (sliceOn || (genUV && td) || !smooth || radius2 > 0.0f)
        return FALSE;

    so->pblock->GetValue(CONE_RADIUS1, mStart, radius1, FOREVER);
    so->pblock->GetValue(CONE_HEIGHT, mStart, height, FOREVER);
    Indent(level);

    MSTREAMPRINTF  _T("Separator {\n"));
    Indent(level + 1);
    if (mZUp)
    {
        if (height > 0.0f)
            MSTREAMPRINTF  _T("Rotation { rotation 1 0 0 %s }\n"),
                    floatVal(float(PI/2.0)));
        else
            MSTREAMPRINTF  _T("Rotation { rotation 1 0 0 %s }\n"),
                    floatVal(float(-PI/2.0)));
        Indent(level + 1);
        MSTREAMPRINTF  _T("Translation { translation 0 %s 0 }\n"),
                floatVal(float(fabs(height)/2.0)));
    }
    else
    {
        Point3 p = Point3(0.0f, 0.0f, (float)fabs(height) / 2.0f);
        MSTREAMPRINTF  _T("Translation { translation %s }\n"), point(p));
    }
    Indent(level + 1);

    MSTREAMPRINTF  _T("Cone { bottomRadius %s "), floatVal(radius1));
    MSTREAMPRINTF  _T("height %s }\n"), floatVal(float(fabs(height))));

    Indent(level);
    MSTREAMPRINTF  _T("}\n"));
    return TRUE;
}

// Create a VRMNL primitive cube, if appropriate.
// Returns TRUE if a primitive is created
BOOL
VRBLExport::VrblOutCube(INode *node, Object *obj, int level)
{
    Mtl *mtl = node->GetMtl();
    // Multi materials need meshes
    if (mtl && mtl->IsMultiMtl())
        return FALSE;

    SimpleObject *so = (SimpleObject *)obj;
    float length, width, height;
    BOOL td = HasTexture(node);

    int genUV, lsegs, wsegs, hsegs;
    so->pblock->GetValue(BOXOBJ_GENUVS, mStart, genUV, FOREVER);
    so->pblock->GetValue(BOXOBJ_LSEGS, mStart, lsegs, FOREVER);
    so->pblock->GetValue(BOXOBJ_WSEGS, mStart, hsegs, FOREVER);
    so->pblock->GetValue(BOXOBJ_HSEGS, mStart, wsegs, FOREVER);
    if ((genUV && td) || lsegs > 1 || hsegs > 1 || wsegs > 1)
        return FALSE;

    so->pblock->GetValue(BOXOBJ_LENGTH, mStart, length, FOREVER);
    so->pblock->GetValue(BOXOBJ_WIDTH, mStart, width, FOREVER);
    so->pblock->GetValue(BOXOBJ_HEIGHT, mStart, height, FOREVER);
    Indent(level);
    MSTREAMPRINTF  _T("Separator {\n"));
    Indent(level + 1);
    Point3 p = Point3(0.0f, 0.0f, height / 2.0f);
    // VRML cubes grow from the middle, MAX grows from z=0
    MSTREAMPRINTF  _T("Translation { translation %s }\n"), point(p));
    Indent(level + 1);

    if (mZUp)
    {
        MSTREAMPRINTF  _T("Cube { width %s "),
                floatVal(float(fabs(width))));
        MSTREAMPRINTF  _T("height %s "),
                floatVal(float(fabs(length))));
        MSTREAMPRINTF  _T(" depth %s }\n"),
                floatVal(float(fabs(height))));
    }
    else
    {
        MSTREAMPRINTF  _T("Cube { width %s "),
                floatVal(float(fabs(width))));
        MSTREAMPRINTF  _T("height %s "),
                floatVal(float(fabs(height))));
        MSTREAMPRINTF  _T(" depth %s }\n"),
                floatVal(float(fabs(length))));
    }
    Indent(level);
    MSTREAMPRINTF  _T("}\n"));

    return TRUE;
}

// Output a perspective camera
BOOL
VRBLExport::VrblOutCamera(INode *node, Object *obj, int level)
{
    // compute camera transform
    ViewParams vp;
    CameraState cs;
    Interval iv;
    CameraObject *cam = (CameraObject *)obj;
    cam->EvalCameraState(0, iv, &cs);
    vp.fov = cs.fov / 1.3333f;

    Indent(level);
    MSTREAMPRINTF  _T("DEF %s_Animated PerspectiveCamera {\n"), mNodes.GetNodeName(node));
    Indent(level + 1);
    MSTREAMPRINTF  _T("position 0 0 0\n"));
    Indent(level + 1);
    MSTREAMPRINTF  _T("heightAngle %s\n"), floatVal(vp.fov));
    if (!mZUp)
    {
        Indent(level + 1);
        MSTREAMPRINTF  _T("orientation 1 0 0 %s\n"),
                floatVal(float(-PI/2.0)));
    }
    Indent(level);
    MSTREAMPRINTF  _T("}\n"));

    return TRUE;
}

// Output an omni light
BOOL
VRBLExport::VrblOutPointLight(INode *node, LightObject *light, int level)
{
    LightState ls;
    Interval iv = FOREVER;

    light->EvalLightState(mStart, iv, &ls);

    Indent(level);
    MSTREAMPRINTF  _T("DEF %s PointLight {\n"), mNodes.GetNodeName(node));
    Indent(level + 1);
    MSTREAMPRINTF  _T("intensity %s\n"),
            floatVal(light->GetIntensity(mStart, FOREVER)));
    Indent(level + 1);
    Point3 col = light->GetRGBColor(mStart, FOREVER);
    MSTREAMPRINTF  _T("color %s\n"), color(col));
    Indent(level + 1);
    MSTREAMPRINTF  _T("location 0 0 0\n"));

    Indent(level + 1);
    MSTREAMPRINTF  _T("on %s\n"), ls.on ? _T("TRUE") : _T("FALSE"));
    Indent(level);
    MSTREAMPRINTF  _T("}\n"));
    return TRUE;
}

// Output a directional light
BOOL
VRBLExport::VrblOutDirectLight(INode *node, LightObject *light, int level)
{
    LightState ls;
    Interval iv = FOREVER;

    light->EvalLightState(mStart, iv, &ls);

    Indent(level);
    MSTREAMPRINTF  _T("DEF %s DirectionalLight {\n"),  mNodes.GetNodeName(node));
    Indent(level + 1);
    MSTREAMPRINTF  _T("intensity %s\n"),
            floatVal(light->GetIntensity(mStart, FOREVER)));
    Indent(level + 1);
    Point3 col = light->GetRGBColor(mStart, FOREVER);

    MSTREAMPRINTF  _T("color %s\n"), color(col));

    Indent(level + 1);
    MSTREAMPRINTF  _T("on %s\n"), ls.on ? _T("TRUE") : _T("FALSE"));
    Indent(level);
    MSTREAMPRINTF  _T("}\n"));
    return TRUE;
}

// Output a Spot Light
BOOL
VRBLExport::VrblOutSpotLight(INode *node, LightObject *light, int level)
{
    LightState ls;
    Interval iv = FOREVER;

    Point3 dir(0, 0, -1);
    light->EvalLightState(mStart, iv, &ls);
    Indent(level);
    MSTREAMPRINTF  _T("DEF %s SpotLight {\n"),  mNodes.GetNodeName(node));
    Indent(level + 1);
    MSTREAMPRINTF  _T("intensity %s\n"),
            floatVal(light->GetIntensity(mStart,FOREVER)));
    Indent(level + 1);
    Point3 col = light->GetRGBColor(mStart, FOREVER);
    MSTREAMPRINTF  _T("color %s\n"), color(col));
    Indent(level + 1);
    MSTREAMPRINTF  _T("location 0 0 0\n"));
    Indent(level + 1);
    MSTREAMPRINTF  _T("direction %s\n"), normPoint(dir));
    Indent(level + 1);
    MSTREAMPRINTF  _T("cutOffAngle %s\n"),
            floatVal(DegToRad(ls.fallsize)));
    Indent(level + 1);
    MSTREAMPRINTF  _T("dropOffRate %s\n"),
            floatVal(1.0f - ls.hotsize/ls.fallsize));
    Indent(level + 1);
    MSTREAMPRINTF  _T("on %s\n"), ls.on ? _T("TRUE") : _T("FALSE"));
    Indent(level);
    MSTREAMPRINTF  _T("}\n"));
    return TRUE;
}

// Output an omni light at the top-level Separator
BOOL
VRBLExport::VrblOutTopPointLight(INode *node, LightObject *light)
{
    LightState ls;
    Interval iv = FOREVER;

    light->EvalLightState(mStart, iv, &ls);

    Indent(1);
    MSTREAMPRINTF  _T("DEF %s PointLight {\n"),  mNodes.GetNodeName(node));
    Indent(2);
    MSTREAMPRINTF  _T("intensity %s\n"),
            floatVal(light->GetIntensity(mStart, FOREVER)));
    Indent(2);
    Point3 col = light->GetRGBColor(mStart, FOREVER);
    MSTREAMPRINTF  _T("color %s\n"), color(col));
    Indent(2);
    Point3 p = node->GetObjTMAfterWSM(mStart).GetTrans();
    MSTREAMPRINTF  _T("location %s\n"), point(p));

    Indent(2);
    MSTREAMPRINTF  _T("on %s\n"), ls.on ? _T("TRUE") : _T("FALSE"));
    Indent(1);
    MSTREAMPRINTF  _T("}\n"));
    return TRUE;
}

// Output a directional light at the top-level Separator
BOOL
VRBLExport::VrblOutTopDirectLight(INode *node, LightObject *light)
{
    LightState ls;
    Interval iv = FOREVER;

    light->EvalLightState(mStart, iv, &ls);

    Indent(1);
    MSTREAMPRINTF  _T("DEF %s DirectionalLight {\n"),  mNodes.GetNodeName(node));
    Indent(2);
    MSTREAMPRINTF  _T("intensity %s\n"),
            floatVal(light->GetIntensity(mStart, FOREVER)));
    Indent(2);
    Point3 col = light->GetRGBColor(mStart, FOREVER);
    MSTREAMPRINTF  _T("color %s\n"), color(col));
    Point3 p = Point3(0, 0, -1);

    Matrix3 tm = node->GetObjTMAfterWSM(mStart);
    Point3 trans, s;
    Quat q;
    AffineParts parts;
    decomp_affine(tm, &parts);
    q = parts.q;
    Matrix3 rot;
    q.MakeMatrix(rot);
    p = p * rot;

    Indent(2);
    MSTREAMPRINTF  _T("direction %s\n"), normPoint(p));
    Indent(2);
    MSTREAMPRINTF  _T("on %s\n"), ls.on ? _T("TRUE") : _T("FALSE"));
    Indent(1);
    MSTREAMPRINTF  _T("}\n"));
    return TRUE;
}

// Output a spot light at the top-level Separator
BOOL
VRBLExport::VrblOutTopSpotLight(INode *node, LightObject *light)
{
    LightState ls;
    Interval iv = FOREVER;

    light->EvalLightState(mStart, iv, &ls);
    Indent(1);
    MSTREAMPRINTF  _T("DEF %s SpotLight {\n"),  mNodes.GetNodeName(node));
    Indent(2);
    MSTREAMPRINTF  _T("intensity %s\n"),
            floatVal(light->GetIntensity(mStart,FOREVER)));
    Indent(2);
    Point3 col = light->GetRGBColor(mStart, FOREVER);
    MSTREAMPRINTF  _T("color %s\n"), color(col));
    Indent(2);
    Point3 p = node->GetObjTMAfterWSM(mStart).GetTrans();
    MSTREAMPRINTF  _T("location %s\n"), point(p));

    Matrix3 tm = node->GetObjTMAfterWSM(mStart);
    p = Point3(0, 0, -1);
    Point3 trans, s;
    Quat q;
    AffineParts parts;
    decomp_affine(tm, &parts);
    q = parts.q;
    Matrix3 rot;
    q.MakeMatrix(rot);
    p = p * rot;

    Indent(2);
    MSTREAMPRINTF  _T("direction %s\n"), normPoint(p));
    Indent(2);
    MSTREAMPRINTF  _T("cutOffAngle %s\n"),
            floatVal( DegToRad(ls.fallsize)));
    Indent(2);
    MSTREAMPRINTF  _T("dropOffRate %s\n"),
            floatVal(1.0f - ls.hotsize/ls.fallsize));
    Indent(2);
    MSTREAMPRINTF  _T("on %s\n"), ls.on ? _T("TRUE") : _T("FALSE"));
    Indent(1);
    MSTREAMPRINTF  _T("}\n"));
    return TRUE;
}

// Create a light at the top-level of the file
void
VRBLExport::OutputTopLevelLight(INode *node, LightObject *light)
{
    Class_ID id = light->ClassID();
    if (id == Class_ID(OMNI_LIGHT_CLASS_ID, 0))
        VrblOutTopPointLight(node, light);
    else if (id == Class_ID(DIR_LIGHT_CLASS_ID, 0))
        VrblOutTopDirectLight(node, light);
    else if (id == Class_ID(SPOT_LIGHT_CLASS_ID, 0) || id == Class_ID(FSPOT_LIGHT_CLASS_ID, 0))
        VrblOutTopSpotLight(node, light);
}

// Output a VRML Inline node.
BOOL
VRBLExport::VrblOutInline(VRMLInsObject *obj, int level)
{
    Indent(level);
    MSTREAMPRINTF  _T("WWWInline {\n"));
    Indent(level + 1);
    MSTREAMPRINTF  _T("name %s\n"), obj->GetUrl().data());
    float size = obj->GetSize() * 2.0f;
    Indent(level + 1);
    Point3 p = Point3(size, size, size);
    MSTREAMPRINTF  _T("bboxSize %s\n"), scalePoint(p));
    Indent(level);
    MSTREAMPRINTF  _T("}\n"));
    return TRUE;
}

// Distance comparison function for sorting LOD lists.
static int
DistComp(LODObj **obj1, LODObj **obj2)
{
    float diff = (*obj1)->dist - (*obj2)->dist;
    if (diff < 0.0f)
        return -1;
    if (diff > 0.0f)
        return 1;
    return 0;
}

// Create a level-of-detail object.
BOOL
VRBLExport::VrblOutLOD(INode *node, LODObject *obj, int level)
{
    int numLod = obj->NumRefs();
    Tab<LODObj *> lodObjects = obj->GetLODObjects();
    int i;

    if (numLod == 0)
        return TRUE;

    lodObjects.Sort((CompareFnc)DistComp);

    if (numLod > 1)
    {
        Indent(level);
        MSTREAMPRINTF  _T("LOD {\n"));
        Indent(level + 1);
        Point3 p = node->GetObjTMAfterWSM(mStart).GetTrans();
        MSTREAMPRINTF  _T("center %s\n"), point(p));
        Indent(level + 1);
        MSTREAMPRINTF  _T("range [ "));
        for (i = 0; i < numLod - 1; i++)
        {
            if (i < numLod - 2)
                MSTREAMPRINTF  _T("%s, "), floatVal(lodObjects[i]->dist));
            else
                MSTREAMPRINTF  _T("%s ]\n"), floatVal(lodObjects[i]->dist));
        }
    }

    for (i = 0; i < numLod; i++)
    {
        INode *node = lodObjects[i]->node;
        INode *parent = node->GetParentNode();
        VrblOutNode(node, parent, level + 1, TRUE, FALSE);
    }

    if (numLod > 1)
    {
        Indent(level);
        MSTREAMPRINTF  _T("}\n"));
    }

    return TRUE;
}

// Output an AimTarget.
BOOL
VRBLExport::VrblOutTarget(INode *node, int level)
{
    INode *lookAt = node->GetLookatNode();
    if (!lookAt)
        return TRUE;
    Object *lookAtObj = lookAt->EvalWorldState(mStart).obj;
    Class_ID id = lookAtObj->ClassID();
    // Only generate aim targets for targetted spot lights and cameras
    if (id != Class_ID(SPOT_LIGHT_CLASS_ID, 0) && id != Class_ID(LOOKAT_CAM_CLASS_ID, 0))
        return TRUE;
    Indent(level);
    MSTREAMPRINTF  _T("AimTarget_ktx_com {\n"));
    if (mGenFields)
    {
        Indent(level + 1);
        MSTREAMPRINTF  _T("fields [ SFString aimer ]\n"));
    }
    Indent(level + 1);
    if ((id == Class_ID(LOOKAT_CAM_CLASS_ID, 0)) && IsEverAnimated(lookAt))
                MSTREAMPRINTF  _T("aimer \"%s_Animated\"\n"), mNodes.GetNodeName(lookAt));
    else
                MSTREAMPRINTF  _T("aimer \"%s\"\n"), mNodes.GetNodeName(lookAt));
    Indent(level);
    MSTREAMPRINTF  _T("}\n"));
    return TRUE;
}

// Write out the VRML for nodes we know about, including VRML helper nodes,
// lights, cameras and VRML primitives
BOOL
VRBLExport::VrblOutSpecial(INode *node, INode *parent,
                           Object *obj, int level)
{
    Class_ID id = obj->ClassID();

    /*
    if (id == Class_ID(MR_BLUE_CLASS_ID1, MR_BLUE_CLASS_ID2)) {
        level++;
        VrblOutMrBlue(node, parent, (MrBlueObject*) obj,
                      &level, FALSE);
    }
    */

    if (id == Class_ID(OMNI_LIGHT_CLASS_ID, 0))
        return VrblOutPointLight(node, (LightObject *)obj, level + 1);

    if (id == Class_ID(DIR_LIGHT_CLASS_ID, 0))
        return VrblOutDirectLight(node, (LightObject *)obj, level + 1);

    if (id == Class_ID(SPOT_LIGHT_CLASS_ID, 0) || id == Class_ID(FSPOT_LIGHT_CLASS_ID, 0))
        return VrblOutSpotLight(node, (LightObject *)obj, level + 1);

    if (id == Class_ID(VRML_INS_CLASS_ID1, VRML_INS_CLASS_ID2))
        return VrblOutInline((VRMLInsObject *)obj, level + 1);

    if (id == Class_ID(LOD_CLASS_ID1, LOD_CLASS_ID2))
        return VrblOutLOD(node, (LODObject *)obj, level + 1);

    if (id == Class_ID(SIMPLE_CAM_CLASS_ID, 0) || id == Class_ID(LOOKAT_CAM_CLASS_ID, 0))
        return VrblOutCamera(node, obj, level + 1);

    if (id == Class_ID(TARGET_CLASS_ID, 0))
        return VrblOutTarget(node, level + 1);

    // If object has modifiers or WSMs attached, do not output as
    // a primitive
    SClass_ID sid = node->GetObjectRef()->SuperClassID();
    if (sid == WSM_DERIVOB_CLASS_ID || sid == DERIVOB_CLASS_ID)
        return FALSE;

    if (!mPrimitives)
        return FALSE;

    // Otherwise look for the primitives we know about
    if (id == Class_ID(SPHERE_CLASS_ID, 0))
        return VrblOutSphere(node, obj, level + 1);

    if (id == Class_ID(CYLINDER_CLASS_ID, 0))
        return VrblOutCylinder(node, obj, level + 1);

    if (id == Class_ID(CONE_CLASS_ID, 0))
        return VrblOutCone(node, obj, level + 1);

    if (id == Class_ID(BOXOBJ_CLASS_ID, 0))
        return VrblOutCube(node, obj, level + 1);

    return FALSE;
}

static BOOL
IsLODObject(Object *obj)
{
    return obj->ClassID() == Class_ID(LOD_CLASS_ID1, LOD_CLASS_ID2);
}

// Returns TRUE iff an object or one of its ancestors in animated
static BOOL
IsEverAnimated(INode *node)
{
    // need to sample transform
    Class_ID id = node->EvalWorldState(0).obj->ClassID();
    if (id == Class_ID(SIMPLE_CAM_CLASS_ID, 0) || id == Class_ID(LOOKAT_CAM_CLASS_ID, 0))
        return TRUE;

    for (; !node->IsRootNode(); node = node->GetParentNode())
        if (node->IsAnimated())
            return TRUE;
    return FALSE;
}

// Returns TRUE for object that we want a VRML node to occur
// in the file.
BOOL
VRBLExport::isVrblObject(INode *node, Object *obj, INode *parent)
{
    if (!obj)
        return FALSE;

    Class_ID id = obj->ClassID();
    // Mr Blue nodes only 1st class if stand-alone

    // only animated light come out in scene graph
    if (IsLight(node) || (id == Class_ID(SIMPLE_CAM_CLASS_ID, 0) || id == Class_ID(LOOKAT_CAM_CLASS_ID, 0)))
        return IsEverAnimated(node);

    return (obj->IsRenderable() || id == Class_ID(LOD_CLASS_ID1, LOD_CLASS_ID2) || node->NumberOfChildren() > 0 //||
            ) && (mExportHidden || !node->IsHidden());
}

// Write the VRML for a single object.
void
VRBLExport::VrblOutObject(INode *node, INode *parent, Object *obj, int level)
{
    BOOL isTriMesh = obj->CanConvertToType(triObjectClassID);

    BOOL multiMat = FALSE, twoSided = FALSE;
    // Output the material
    if (obj->IsRenderable())
        multiMat = OutputMaterial(node, twoSided, level + 1);

    // First check for VRML primitives and other special objects
    if (VrblOutSpecial(node, parent, obj, level))
    {
        return;
    }

    // Otherwise output as a triangle mesh
    if (isTriMesh)
    {
        TriObject *tri = (TriObject *)obj->ConvertToType(0, triObjectClassID);
        OutputTriObject(node, tri, multiMat, twoSided, level + 1);
        if (obj != (Object *)tri)
            tri->DeleteThis();
    }
}

// Get the distance to the line of sight target
float
GetLosProxDist(INode *node, TimeValue t)
{
    Point3 p0 = node->GetObjTMAfterWSM(t).GetTrans();
    Matrix3 tmat;
    node->GetTargetTM(t, tmat);
    Point3 p1 = tmat.GetTrans();
    return Length(p1 - p0);
}

// Get the vector to the line of sight target
Point3
GetLosVector(INode *node, TimeValue t)
{
    Point3 p0 = node->GetObjTMAfterWSM(t).GetTrans();
    Matrix3 tmat;
    node->GetTargetTM(t, tmat);
    Point3 p1 = tmat.GetTrans();
    return p1 - p0;
}

// Return TRUE iff the controller is a TCB controller
static BOOL
IsTCBControl(Control *cont)
{
    return (cont && (cont->ClassID() == Class_ID(TCBINTERP_FLOAT_CLASS_ID, 0) || cont->ClassID() == Class_ID(TCBINTERP_POSITION_CLASS_ID, 0) || cont->ClassID() == Class_ID(TCBINTERP_ROTATION_CLASS_ID, 0) || cont->ClassID() == Class_ID(TCBINTERP_POINT3_CLASS_ID, 0) || cont->ClassID() == Class_ID(TCBINTERP_SCALE_CLASS_ID, 0)));
}

// Return TRUE iff the keys are different in any way.
static BOOL
TCBIsDifferent(ITCBKey *k, ITCBKey *oldK)
{
    return k->tens != oldK->tens || k->cont != oldK->cont || k->bias != oldK->bias || k->easeIn != oldK->easeIn || k->easeOut != oldK->easeOut;
}

// returns TRUE iff the position keys are exactly the same
static BOOL
PosKeysSame(ITCBPoint3Key &k1, ITCBPoint3Key &k2)
{
    if (TCBIsDifferent(&k1, &k2))
        return FALSE;
    return k1.val == k2.val;
}

// returns TRUE iff the rotation keys are exactly the same
static BOOL
RotKeysSame(ITCBRotKey &k1, ITCBRotKey &k2)
{
    if (TCBIsDifferent(&k1, &k2))
        return FALSE;
    return k1.val.axis == k2.val.axis && k1.val.angle == k2.val.angle;
}

// returns TRUE iff the scale keys are exactly the same
static BOOL
ScaleKeysSame(ITCBScaleKey &k1, ITCBScaleKey &k2)
{
    if (TCBIsDifferent(&k1, &k2))
        return FALSE;
    return k1.val.s == k2.val.s;
}

// Write out all the keyframe data for the TCB given controller
BOOL
VRBLExport::WriteTCBKeys(INode *node, Control *cont,
                         int type, int level)
{
    ITCBFloatKey fkey, ofkey;
    ITCBPoint3Key pkey, opkey;
    ITCBRotKey rkey, orkey;
    ITCBScaleKey skey, oskey;
    ITCBKey *k, *oldK;
    int num = cont->NumKeys();
    Point3 pval;
    Quat q, qLast = IdentQuat();
    AngAxis rval;
    ScaleValue sval;
    Interval valid;
    Point3 p, po;

    // Get the keyframe interface
    IKeyControl *ikeys = GetKeyControlInterface(cont);

    // Gotta have some keys
    if (num == NOT_KEYFRAMEABLE || num == 0 || !ikeys)
    {
        return FALSE;
    }

    // Set up 'k' to point at the right derived class
    switch (type)
    {
    case KEY_FLOAT:
        k = &fkey;
        oldK = &ofkey;
        break;
    case KEY_POS:
        k = &pkey;
        oldK = &opkey;
        break;
    case KEY_ROT:
        k = &rkey;
        oldK = &orkey;
        break;
    case KEY_SCL:
        k = &skey;
        oldK = &oskey;
        break;
    case KEY_COLOR:
        k = &pkey;
        oldK = &opkey;
        break;
    default:
        return FALSE;
    }

    for (int i = 0; i < ikeys->GetNumKeys(); i++)
    {
        ikeys->GetKey(i, k);
        if (k->time < mStart)
            continue;

        if (i == 0 || TCBIsDifferent(k, oldK))
        {
            Indent(level);
            MSTREAMPRINTF  _T("AnimationStyle_ktx_com {\n"));
            Indent(level + 1);
            if (mGenFields)
                MSTREAMPRINTF  _T("fields [ SFBool loop, SFBitMask splineUse, SFFloat tension, SFFloat continuity, SFFloat bias, SFFloat easeTo, SFFloat easeFrom, SFVec3f pivotOffset ]\n"));
            Indent(level + 1);
            MSTREAMPRINTF  _T("splineUse ("));

            // Write flags
            BOOL hadOne = FALSE;
            if (k->tens != 0.0f)
            {
                MSTREAMPRINTF  _T("TENSION"));
                hadOne = TRUE;
            }
            if (k->cont != 0.0f)
            {
                if (hadOne)
                    MSTREAMPRINTF  _T(" | "));
                MSTREAMPRINTF  _T("CONTINUITY"));
                hadOne = TRUE;
            }
            if (k->bias != 0.0f)
            {
                if (hadOne)
                    MSTREAMPRINTF  _T(" | "));
                MSTREAMPRINTF  _T("BIAS"));
                hadOne = TRUE;
            }
            if (k->easeIn != 0.0f)
            {
                if (hadOne)
                    MSTREAMPRINTF  _T(" | "));
                MSTREAMPRINTF  _T("EASE_TO"));
                hadOne = TRUE;
            }
            if (k->easeOut != 0.0f)
            {
                if (hadOne)
                    MSTREAMPRINTF  _T(" | "));
                MSTREAMPRINTF  _T("EASE_FROM"));
                hadOne = TRUE;
            }
            MSTREAMPRINTF  _T(")\n"));

            // Write TCB and ease
            if (k->tens != 0.0f)
            {
                Indent(level + 1);
                MSTREAMPRINTF  _T("tension %s\n"), floatVal(k->tens));
            }
            if (k->cont != 0.0f)
            {
                Indent(level + 1);
                MSTREAMPRINTF  _T("continuity %s\n"), floatVal(k->cont));
            }
            if (k->bias != 0.0f)
            {
                Indent(level + 1);
                MSTREAMPRINTF  _T("bias %s\n"), floatVal(k->bias));
            }
            if (k->easeIn != 0.0f)
            {
                Indent(level + 1);
                MSTREAMPRINTF  _T("easeTo %s\n"), floatVal(k->easeIn));
            }
            if (k->easeOut != 0.0f)
            {
                Indent(level + 1);
                MSTREAMPRINTF  _T("easeFrom %s\n"), floatVal(k->easeOut));
            }

            // get the pivot offset and remove the rotational component
            Matrix3 m = Matrix3(TRUE);
            Quat q = node->GetObjOffsetRot();
            q.MakeMatrix(m);
            p = -node->GetObjOffsetPos();
            m = Inverse(m);
            po = VectorTransform(m, p);

            Indent(level + 1);
            if (type != KEY_POS) MSTREAMPRINTF  _T("pivotOffset %s\n"), point(po));
            Indent(level);
            MSTREAMPRINTF  _T("}\n"));
        }
        // Write values
        switch (type)
        {
        case KEY_FLOAT:
            assert(FALSE);
            break;

        case KEY_SCL:
        {
            if (i == 0 && (k->time - mStart) != 0)
            {
                WriteScaleKey0(node, mStart, level, TRUE);
                WriteScaleKey0(node,
                               k->time - GetTicksPerFrame(), level, TRUE);
            }
            Matrix3 tm = GetLocalTM(node, mStart);
            AffineParts parts;
            decomp_affine(tm, &parts);
            ScaleValue sv(parts.k, parts.u);
            Point3 s = sv.s;
            if (parts.f < 0.0f)
                s = -s;
            else
                s = skey.val.s;
            if (i != 0 && ScaleKeysSame(skey, oskey))
                continue;
            mHadAnim = TRUE;
            Indent(level);
            MSTREAMPRINTF  _T("ScaleKey_ktx_com {\n"));
            Indent(level + 1);
            if (mGenFields)
                MSTREAMPRINTF 
                        _T("fields [ SFLong frame, SFVec3f scale ]\n"));
            Indent(level + 1);
            MSTREAMPRINTF  _T("frame %d\n"), (k->time - mStart)/GetTicksPerFrame());
            Indent(level + 1);
            MSTREAMPRINTF  _T("scale %s\n"), scalePoint(s));
            Indent(level);
            MSTREAMPRINTF  _T("}\n"));
            memcpy(oldK, k, sizeof(skey));
            break;
        }

        case KEY_COLOR:
            if (i == 0 && k->time != 0)
            {
                WritePositionKey0(node, mStart, level, TRUE);
                WritePositionKey0(node,
                                  k->time - GetTicksPerFrame(), level, TRUE);
            }
            if (i != 0 && PosKeysSame(pkey, opkey))
                continue;
            mHadAnim = TRUE;
            Indent(level);
            MSTREAMPRINTF  _T("ColorKey_ktx_com {\n"));
            Indent(level + 1);
            if (mGenFields)
                MSTREAMPRINTF 
                        _T("fields [ SFLong frame, SFColor color ]\n"));
            Indent(level + 1);
            MSTREAMPRINTF  _T("frame %d\n"), (k->time - mStart)/GetTicksPerFrame());
            Indent(level + 1);
            MSTREAMPRINTF  _T("color %s\n"), color(pkey.val));
            Indent(level);
            MSTREAMPRINTF  _T("}\n"));
            memcpy(oldK, k, sizeof(pkey));
            break;

        case KEY_POS:
            if (i == 0 && (k->time - mStart) != 0)
            {
                WritePositionKey0(node, mStart, level, TRUE);
                WritePositionKey0(node,
                                  k->time - GetTicksPerFrame(), level, TRUE);
            }
            if (i != 0 && PosKeysSame(pkey, opkey))
                continue;
            mHadAnim = TRUE;
            Indent(level);
            MSTREAMPRINTF  _T("PositionKey_ktx_com {\n"));
            Indent(level + 1);
            if (mGenFields)
                MSTREAMPRINTF 
                        _T("fields [ SFLong frame, SFVec3f translation ]\n"));
            Indent(level + 1);
            MSTREAMPRINTF  _T("frame %d\n"), (k->time - mStart)/GetTicksPerFrame());
            p = pkey.val;
            Indent(level + 1);
            MSTREAMPRINTF  _T("translation %s\n"), point(p));
            Indent(level);
            MSTREAMPRINTF  _T("}\n"));
            memcpy(oldK, k, sizeof(pkey));
            break;

        case KEY_ROT:
        {
            // Note rotation keys are cummulative unlike other keys.
            if (i == 0 && (k->time - mStart) != 0)
            {
                WriteRotationKey0(node, mStart, level, TRUE);
                WriteRotationKey0(node,
                                  k->time - GetTicksPerFrame(), level, TRUE);
            }
            Matrix3 tm = GetLocalTM(node, k->time);
            Point3 axis;
            Quat q;
            float ang;

            AffineParts parts;
            decomp_affine(tm, &parts);
            q = parts.q;
            AngAxisFromQ(q / qLast, &ang, axis);
            // this removes rotational direction errors when rotating PI
            // and reduces negative rotational errors
            if (!round(axis.x + rkey.val.axis.x) && !round(axis.y + rkey.val.axis.y) && !round(axis.z + rkey.val.axis.z))
            {
                ang = rkey.val.angle;
                axis = rkey.val.axis;
            }

            // this removes errors if q = (0 0 0 0) for rkey (1 0 0 360)
            if (axis.x == 0.0 && axis.y == 0.0 && axis.z == 0.0 && ang == 0.0)
            {
                ang = rkey.val.angle;
                axis = rkey.val.axis;
            }

            if (i != 0 && ang == 0.0f)
                continue;
            qLast = q;
            mHadAnim = TRUE;
            Indent(level);
            MSTREAMPRINTF  _T("RotationKey_ktx_com {\n"));
            Indent(level + 1);
            if (mGenFields)
                MSTREAMPRINTF 
                        _T("fields [ SFLong frame, SFRotation rotation ]\n"));
            Indent(level + 1);
            MSTREAMPRINTF  _T("frame %d\n"), (k->time - mStart)/GetTicksPerFrame());
            Indent(level + 1);
            MSTREAMPRINTF  _T("rotation %s\n"),
                    axisPoint(axis, ang));
            Indent(level);
            MSTREAMPRINTF  _T("}\n"));
            memcpy(oldK, k, sizeof(rkey));
            break;
        }
        }
    }
    return TRUE;
}

// Write out all the keyframe data for an arbitrary PRS controller
void
VRBLExport::WriteLinearKeys(INode *node,
                            Tab<TimeValue> &posTimes,
                            Tab<Point3> &posKeys,
                            Tab<TimeValue> &rotTimes,
                            Tab<AngAxis> &rotKeys,
                            Tab<TimeValue> &sclTimes,
                            Tab<Point3> &sclKeys,
                            int type, int level)
{
    AngAxis rval;
    Point3 p, po, s;
    int i;
    TimeValue t;
    Tab<TimeValue> &timeVals = posTimes;

    // Set up 'k' to point at the right derived class
    switch (type)
    {
    case KEY_POS:
    case KEY_COLOR:
        timeVals = posTimes;
        break;
    case KEY_ROT:
        timeVals = rotTimes;
        break;
    case KEY_SCL:
        timeVals = sclTimes;
        break;
    default:
        return;
    }

    Indent(level);
    MSTREAMPRINTF  _T("AnimationStyle_ktx_com {\n"));
    Indent(level + 1);
    if (mGenFields)
        MSTREAMPRINTF  _T("fields [ SFVec3f pivotOffset ]\n"));
    Indent(level + 1);
    // get the pivot offset and remove rotational component
    Matrix3 m = Matrix3(TRUE);
    Quat q = node->GetObjOffsetRot();
    q.MakeMatrix(m);
    p = -node->GetObjOffsetPos();
    m = Inverse(m);
    po = VectorTransform(m, p);

    Indent(level + 1);
    if (type != KEY_POS) MSTREAMPRINTF  _T("pivotOffset %s\n"), point(po));
    Indent(level);
    MSTREAMPRINTF  _T("}\n"));

    for (i = 0; i < timeVals.Count(); i++)
    {
        t = timeVals[i];
        if (t < mStart)
            continue;

        // Write values
        switch (type)
        {
        case KEY_POS:
            mHadAnim = TRUE;
            p = posKeys[i];
            Indent(level);
            MSTREAMPRINTF  _T("PositionKey_ktx_com {\n"));
            Indent(level + 1);
            if (mGenFields)
                MSTREAMPRINTF 
                        _T("fields [ SFLong frame, SFVec3f translation ]\n"));
            Indent(level + 1);
            MSTREAMPRINTF  _T("frame %d\n"),
                    (timeVals[i]-mStart)/GetTicksPerFrame());
            Indent(level + 1);
            MSTREAMPRINTF  _T("translation %s\n"), point(p));
            Indent(level);
            MSTREAMPRINTF  _T("}\n"));
            break;

        case KEY_ROT:
            mHadAnim = TRUE;
            rval = rotKeys[i];
            if (rval.angle == 0.0f || fabs(rval.angle - 2.0 * PI) < 1.0e-5)
                break;
            Indent(level);
            MSTREAMPRINTF  _T("RotationKey_ktx_com {\n"));
            Indent(level + 1);
            if (mGenFields)
                MSTREAMPRINTF 
                        _T("fields [ SFLong frame, SFRotation rotation ]\n"));
            Indent(level + 1);
            MSTREAMPRINTF  _T("frame %d\n"),
                    (timeVals[i]-mStart)/GetTicksPerFrame());
            Indent(level + 1);
            MSTREAMPRINTF  _T("rotation %s\n"),
                    axisPoint(rval.axis, rval.angle));
            Indent(level);
            MSTREAMPRINTF  _T("}\n"));
            break;
        case KEY_SCL:
            mHadAnim = TRUE;
            s = sclKeys[i];
            Indent(level);
            MSTREAMPRINTF  _T("ScaleKey_ktx_com {\n"));
            Indent(level + 1);
            if (mGenFields)
                MSTREAMPRINTF 
                        _T("fields [ SFLong frame, SFVec3f scale ]\n"));
            Indent(level + 1);
            MSTREAMPRINTF  _T("frame %d\n"),
                    (timeVals[i]-mStart)/GetTicksPerFrame());
            Indent(level + 1);
            MSTREAMPRINTF  _T("scale %s\n"), scalePoint(s));
            Indent(level);
            MSTREAMPRINTF  _T("}\n"));
            break;

        case KEY_COLOR:
            mHadAnim = TRUE;
            Indent(level);
            MSTREAMPRINTF  _T("ColorKey_ktx_com {\n"));
            Indent(level + 1);
            if (mGenFields)
                MSTREAMPRINTF 
                        _T("fields [ SFLong frame, SFColor color ]\n"));
            Indent(level + 1);
            MSTREAMPRINTF  _T("frame %d\n"),
                    (timeVals[i]-mStart)/GetTicksPerFrame());
            Indent(level + 1);
            p = posKeys[i];
            MSTREAMPRINTF  _T("color %s\n"), color(p));
            Indent(level);
            MSTREAMPRINTF  _T("}\n"));
            break;
        }
    }

    return;
}

int
VRBLExport::WriteAllControllerData(INode *node, int flags, int level,
                                   Control *lc)
{
    int i;
    TimeValue t;
    TimeValue end = mIp->GetAnimRange().End();
    int frames = (end - mStart) / GetTicksPerFrame();
    Point3 p, axis, s;
    Quat q, qLast = IdentQuat();
    Matrix3 tm, ip;
    int retVal = 0;

    // Tables of keyframe values
    Tab<Point3> posKeys;
    Tab<TimeValue> posTimes;
    Tab<Point3> scaleKeys;
    Tab<TimeValue> scaleTimes;
    Tab<AngAxis> rotKeys;
    Tab<TimeValue> rotTimes;
    BOOL keys;

    // Set up 'k' to point at the right derived class
    if (flags & KEY_POS)
    {
        Control *pc = node->GetTMController()->GetPositionController();
        if (IsTCBControl(pc))
        {
            keys = WriteTCBKeys(node, pc, KEY_POS, level);
            flags &= ~KEY_POS;
            if (keys)
                retVal |= KEY_POS;
        }
        else
        {
            posKeys.SetCount(frames + 1);
            posTimes.SetCount(frames + 1);
        }
    }
    if (flags & KEY_COLOR)
    {
        posKeys.SetCount(frames + 1);
        posTimes.SetCount(frames + 1);
    }
    if (flags & KEY_ROT)
    {
        Control *rc = node->GetTMController()->GetRotationController();
        // disabling writing tcb rotation keys because position controller
        // like path controller also change rotation so you have a tcbrotation controller
        // with no keys.
        if (IsTCBControl(rc) && rc->NumKeys() && FALSE)
        {
            keys = WriteTCBKeys(node, rc, KEY_ROT, level);
            flags &= ~KEY_ROT;
            if (keys)
                retVal |= KEY_ROT;
        }
        else
        {
            rotKeys.SetCount(frames + 1);
            rotTimes.SetCount(frames + 1);
        }
    }
    if (flags & KEY_SCL)
    {
        Control *sc = node->GetTMController()->GetScaleController();
        if (IsTCBControl(sc))
        {
            keys = WriteTCBKeys(node, sc, KEY_SCL, level);
            flags &= ~KEY_SCL;
            if (keys)
                retVal |= KEY_SCL;
        }
        else
        {
            scaleKeys.SetCount(frames + 1);
            scaleTimes.SetCount(frames + 1);
        }
    }

    if (!flags)
        return retVal;

    // Sample the controller at every frame
    for (i = 0, t = mStart; i <= frames; i++, t += GetTicksPerFrame())
    {
        if (flags & KEY_COLOR)
        {
            lc->GetValue(t, &posKeys[i], FOREVER);
            posTimes[i] = t;
            continue;
        }
        tm = GetLocalTM(node, t);
        AffineParts parts;
        decomp_affine(tm, &parts);
        if (flags & KEY_SCL)
        {
            s = ScaleValue(parts.k, parts.u).s;
            if (parts.f < 0.0f)
                s = -s;
            scaleTimes[i] = t;
            scaleKeys[i] = s;
        }

        if (flags & KEY_POS)
        {
            p = parts.t;
            posTimes[i] = t;
            posKeys[i] = p;
        }

        if (flags & KEY_ROT)
        {
            q = parts.q;
            rotTimes[i] = t;
            rotKeys[i] = AngAxis(q / qLast);
            qLast = q;
        }
    }

    int newKeys;
    float eps;
    if (flags & KEY_POS)
    {
        eps = float(1.0e-5);
        newKeys = reducePoint3Keys(posTimes, posKeys, eps);
        if (newKeys != 0)
        {
            retVal |= KEY_POS;
            WriteLinearKeys(node,
                            posTimes, posKeys,
                            rotTimes, rotKeys,
                            scaleTimes, scaleKeys,
                            KEY_POS, level);
        }
    }
    if (flags & KEY_ROT)
    {
        eps = float(1.0e-5);
        newKeys = reduceAngAxisKeys(rotTimes, rotKeys, eps);
        if (newKeys != 0)
        {
            retVal |= KEY_ROT;
            WriteLinearKeys(node,
                            posTimes, posKeys,
                            rotTimes, rotKeys,
                            scaleTimes, scaleKeys,
                            KEY_ROT, level);
        }
    }
    if (flags & KEY_SCL)
    {
        eps = float(1.0e-5);
        newKeys = reducePoint3Keys(scaleTimes, scaleKeys, eps);
        if (newKeys != 0)
        {
            retVal |= KEY_SCL;
            WriteLinearKeys(node,
                            posTimes, posKeys,
                            rotTimes, rotKeys,
                            scaleTimes, scaleKeys,
                            KEY_SCL, level);
        }
    }
    if (flags & KEY_COLOR)
    {
        eps = float(1.0e-5);
        newKeys = reducePoint3Keys(posTimes, posKeys, eps);
        if (newKeys != 0)
        {
            retVal |= KEY_SCL;
            WriteLinearKeys(node,
                            posTimes, posKeys,
                            rotTimes, rotKeys,
                            scaleTimes, scaleKeys,
                            KEY_COLOR, level);
        }
    }
    return retVal;
}

// Write out the initial position key, relative to the parent.
void
VRBLExport::WritePositionKey0(INode *node, TimeValue t, int level, BOOL force)
{
    Matrix3 tm = GetLocalTM(node, mStart);
    Point3 p = tm.GetTrans();

    // Don't need a key for identity translate
    if (!force && (p.x == 0.0f && p.y == 0.0f && p.z == 0.0f))
        return;

    mHadAnim = TRUE;
    Indent(level);
    MSTREAMPRINTF  _T("PositionKey_ktx_com {\n"));
    Indent(level + 1);
    if (mGenFields)
        MSTREAMPRINTF  _T("fields [ SFLong frame, SFVec3f translation ]\n"));
    Indent(level + 1);
    MSTREAMPRINTF  _T("frame %d\n"), (t-mStart)/GetTicksPerFrame());
    Indent(level + 1);
    MSTREAMPRINTF  _T("translation %s\n"), point(p));
    Indent(level);
    MSTREAMPRINTF  _T("}\n"));
}

// Write out the initial rotation key, relative to the parent.
void
VRBLExport::WriteRotationKey0(INode *node, TimeValue t, int level, BOOL force)
{
    Matrix3 tm = GetLocalTM(node, mStart);
    Point3 p, s, axis;
    Quat q;
    float ang;

    AffineParts parts;
    decomp_affine(tm, &parts);
    p = parts.t;
    q = parts.q;
    AngAxisFromQ(q, &ang, axis);

    // Dont't need a ket for identity rotate
    if (!force && ang == 0.0f)
        return;

    mHadAnim = TRUE;
    Indent(level);
    MSTREAMPRINTF  _T("RotationKey_ktx_com {\n"));
    Indent(level + 1);
    if (mGenFields)
        MSTREAMPRINTF 
                _T("fields [ SFLong frame, SFRotation rotation ]\n"));
    Indent(level + 1);
    MSTREAMPRINTF  _T("frame %d\n"), (t-mStart)/GetTicksPerFrame());
    Indent(level + 1);
    MSTREAMPRINTF  _T("rotation %s\n"), axisPoint(axis, ang));
    Indent(level);
    MSTREAMPRINTF  _T("}\n"));
}

// Write out the initial scale key, relative to the parent.
void
VRBLExport::WriteScaleKey0(INode *node, TimeValue t, int level, BOOL force)
{
    Matrix3 tm = GetLocalTM(node, mStart);
    AffineParts parts;
    decomp_affine(tm, &parts);
    ScaleValue sv(parts.k, parts.u);
    Point3 s = sv.s;
    if (parts.f < 0.0f)
        s = -s;

    // Don't need a key for identity scale
    if (!force && (s.x == 1.0f && s.y == 1.0f && s.z == 1.0f))
        return;

    mHadAnim = TRUE;
    Indent(level);
    MSTREAMPRINTF  _T("ScaleKey_ktx_com {\n"));
    Indent(level + 1);
    if (mGenFields)
        MSTREAMPRINTF  _T("fields [ SFLong frame, SFVec3f scale ]\n"));
    Indent(level + 1);
    MSTREAMPRINTF  _T("frame %d\n"), (t-mStart)/GetTicksPerFrame());
    Indent(level + 1);
    MSTREAMPRINTF  _T("scale %s\n"), scalePoint(s));
    Indent(level);
    MSTREAMPRINTF  _T("}\n"));
}

void
VRBLExport::WriteVisibilityData(INode *node, int level)
{
    int i;
    TimeValue t;
    int frames = mIp->GetAnimRange().End() / GetTicksPerFrame();
    BOOL lastVis = TRUE, vis;

    // Now generate the Hide keys
    for (i = 0, t = mStart; i <= frames; i++, t += GetTicksPerFrame())
    {
        vis = node->GetVisibility(t) <= 0.0f ? FALSE : TRUE;
        if (vis != lastVis)
        {
            mHadAnim = TRUE;
            Indent(level);
            MSTREAMPRINTF  _T("HideKey_ktx_com {\n"));
            if (mGenFields)
            {
                Indent(level + 1);
                MSTREAMPRINTF  _T("fields [ SFLong frame] \n"));
            }
            Indent(level + 1);
            MSTREAMPRINTF  _T("frame %d\n"), i);
            Indent(level);
            MSTREAMPRINTF  _T("}\n"));
        }
        lastVis = vis;
    }
}

BOOL
VRBLExport::IsLight(INode *node)
{
    Object *obj = node->EvalWorldState(mStart).obj;
    if (!obj)
        return FALSE;

    SClass_ID sid = obj->SuperClassID();
    return sid == LIGHT_CLASS_ID;
}

Control *
VRBLExport::GetLightColorControl(INode *node)
{
    if (!IsLight(node))
        return NULL;
    Object *obj = node->EvalWorldState(mStart).obj;
    IParamBlock *pblock = (IParamBlock *)obj->SubAnim(0);
    Control *cont = pblock->GetController(0); // I know color is index 0!
    return cont;
}

#define NeedsKeys(nkeys) ((nkeys) > 0 || (nkeys) == NOT_KEYFRAMEABLE)

// Write out all PRS keyframe data, if it exists
void
VRBLExport::VrblOutControllers(INode *node, int level)
{
    Control *pc, *rc, *sc, *vc, *lc;
    int npk = 0, nrk = 0, nsk = 0, nvk = 0, nlk = 0;

    if (mType != Export_VRBL)
        return;

    pc = node->GetTMController()->GetPositionController();
    if (pc)
        npk = pc->NumKeys();
    rc = node->GetTMController()->GetRotationController();
    if (rc)
        nrk = rc->NumKeys();
    sc = node->GetTMController()->GetScaleController();
    if (sc)
        nsk = sc->NumKeys();
    vc = node->GetVisController();
    if (vc)
        nvk = vc->NumKeys();
    lc = GetLightColorControl(node);
    if (lc)
        nlk = lc->NumKeys();
    if (NeedsKeys(nlk))
        WriteAllControllerData(node, KEY_COLOR, level, lc);

    Class_ID id = node->GetTMController()->ClassID();
    int flags = 0;

    if (id != Class_ID(PRS_CONTROL_CLASS_ID, 0))
        flags = KEY_POS | KEY_ROT | KEY_SCL;
    else
    {
        pc = node->GetTMController()->GetPositionController();
        if (pc)
            npk = pc->NumKeys();
        rc = node->GetTMController()->GetRotationController();
        if (rc)
            nrk = rc->NumKeys();
        sc = node->GetTMController()->GetScaleController();
        if (sc)
            nsk = sc->NumKeys();
        if (NeedsKeys(npk))
            flags |= KEY_POS | KEY_ROT; // pos controllers can affect rot
        if (NeedsKeys(nrk))
            flags |= KEY_ROT;
        if (NeedsKeys(nsk))
            flags |= KEY_SCL;
    }
    if (flags)
    {
        int newFlags = WriteAllControllerData(node, flags, level, NULL);
        if (!(newFlags & KEY_POS))
            WritePositionKey0(node, mStart, level, FALSE);
        if (!(newFlags & KEY_ROT))
            WriteRotationKey0(node, mStart, level, FALSE);
        if (!(newFlags & KEY_SCL))
            WriteScaleKey0(node, mStart, level, FALSE);
    }
    if (NeedsKeys(nvk))
        WriteVisibilityData(node, level);
#if 0
    // FIXME add this back!
    if (NeedsKeys(nlk))
        WriteControllerData(node, lc, KEY_COLOR, level);
#endif
}

// Output a camera at the top level of the file
void
VRBLExport::VrmlOutTopLevelCamera(int level, INode *node, BOOL topLevel)
{
    if (!topLevel && node == mCamera)
        return;
    if (!(mExportHidden || !node->IsHidden()))
        return;

    CameraObject *cam = (CameraObject *)node->EvalWorldState(mStart).obj;
    Matrix3 tm = node->GetObjTMAfterWSM(mStart);
    Point3 p, s, axis;
    Quat q;
    float ang;

    AffineParts parts;
    decomp_affine(tm, &parts);
    p = parts.t;
    q = parts.q;
    if (!mZUp)
    {
        // Now rotate around the X Axis PI/2
        Matrix3 rot = RotateXMatrix(PI / 2);
        Quat qRot(rot);
        if (qRot == q)
        {
            axis.x = axis.z = 0.0f;
            axis.y = 1.0f;
            ang = 0.0f;
        }
        else
            AngAxisFromQ(q / qRot, &ang, axis);
    }
    else
        AngAxisFromQ(q, &ang, axis);

    ViewParams vp;
    CameraState cs;
    Interval iv;
    cam->EvalCameraState(0, iv, &cs);
    vp.fov = cs.fov / 1.3333f;

    Indent(level);
    MSTREAMPRINTF  _T("DEF %s PerspectiveCamera {\n"), mNodes.GetNodeName(node));
    Indent(level + 1);
    MSTREAMPRINTF  _T("position %s\n"), point(p));
    Indent(level + 1);
    MSTREAMPRINTF  _T("orientation %s\n"), axisPoint(axis, -ang));
    Indent(level + 1);
    MSTREAMPRINTF  _T("heightAngle %s\n"), floatVal(vp.fov));
    Indent(level);
    MSTREAMPRINTF  _T("}\n"));
}

// From dllmain.cpp
extern HINSTANCE hInstance;

// Write out some comments at the top of the file.
void
VRBLExport::VrblOutFileInfo()
{
    TCHAR filename[MAX_PATH];
    DWORD size, dummy;
    float vernum = 1.0f;
    float betanum = 0.0f;

    GetModuleFileName(hInstance, filename, MAX_PATH);
    size = GetFileVersionInfoSize(filename, &dummy);
    if (size)
    {
        char *buf = (char *)malloc(size);
        GetFileVersionInfo(filename, NULL, size, buf);
        VS_FIXEDFILEINFO *qbuf;
        UINT len;
        if (VerQueryValue(buf, _T("\\"), (void **)&qbuf, &len))
        {
            // got the version information
            DWORD ms = qbuf->dwProductVersionMS;
            DWORD ls = qbuf->dwProductVersionLS;
            vernum = HIWORD(ms) + (LOWORD(ms) / 100.0f);
            betanum = HIWORD(ls) + (LOWORD(ls) / 100.0f);
        }
        free(buf);
    }
    Indent(1);
    MSTREAMPRINTF  _T("Info { string \"Produced by Uwes VRML 1.0/2.0 exporter, Version 8, Revision 05\" }\n"), vernum, betanum);

    time_t ltime;
    time(&ltime);
    char *time = ctime(&ltime);
    // strip the CR
    time[strlen(time) - 1] = '\0';
    const TCHAR *fn = mIp->GetCurFileName().data();
    if (fn && _tcslen(fn) > 0)
    {
        Indent(1);
        MSTREAMPRINTF  _T("Info { string \"MAX File: %s, Date: %s\" }\n"), fn, time);
    }
    else
    {
        Indent(1);
        MSTREAMPRINTF  _T("Info { string \"Date: %s\" }\n"), time);
    }
    Indent(1);
    MSTREAMPRINTF  _T("ShapeHints {\n"));
    Indent(2);
    MSTREAMPRINTF  _T("shapeType SOLID\n"));
    Indent(2);
    MSTREAMPRINTF  _T("vertexOrdering COUNTERCLOCKWISE\n"));
    Indent(2);
    MSTREAMPRINTF  _T("faceType CONVEX\n"));
    Indent(1);
    MSTREAMPRINTF  _T("}\n"));
}

// Output a single node as VRML and recursively output the children of
// the node.
void
VRBLExport::VrblOutNode(INode *node, INode *parent, int level, BOOL isLOD,
                        BOOL lastChild)
{
    // Don't gen code for LOD references, only LOD nodes
    if (!isLOD && ObjectIsLODRef(node))
        return;

    Object *obj = node->EvalWorldState(mStart).obj;
    BOOL outputName = TRUE;
    int numChildren = node->NumberOfChildren();
    BOOL isVrml = isVrblObject(node, obj, parent);

    if (node->IsRootNode() || (obj && isVrml))
    {
        StartNode(node, obj, level, outputName);
        if (node->IsRootNode())
        {
            VrblOutFileInfo();
            // Collect a list of all LOD nodes and textures for later use.
            if (mCamera)
                VrmlOutTopLevelCamera(level + 1, mCamera, TRUE);
            ScanSceneGraph();
        }
    }

    if (obj && isVrml)
    {
        if (!IsLODObject(obj))
        {
            OutputNodeTransform(node, level + 1);

            // If the node has a controller, output the data
            VrblOutControllers(node, level + 1);
        }
        // Output the data for the object at this node
        VrblOutObject(node, parent, obj, level);
    }

    // Now output the children
    for (int i = 0; i < numChildren; i++)
    {
        VrblOutNode(node->GetChildNode(i), node, level + 1,
                    FALSE, i == numChildren - 1);
    }

    if (node->IsRootNode() || (obj && (isVrblObject(node, obj, parent))))
    {
        if (node->IsRootNode())
            VrblOutAnimationFrames();
        EndNode(node, level, lastChild);
    }
}

// Write the "AnimationFrames" VRBL node
void
VRBLExport::VrblOutAnimationFrames()
{
    if (mType == Export_VRBL && mHadAnim)
    {
        Indent(1);
        MSTREAMPRINTF  _T("AnimationFrames_ktx_com {\n"));
        Indent(2);
        if (mGenFields)
            MSTREAMPRINTF  _T("fields [ SFLong length, SFLong segmentStart, SFLong segmentEnd, SFLong current, SFFloat rate ]\n"));
        Indent(2);
        int frames = (mIp->GetAnimRange().End() - mIp->GetAnimRange().Start()) / GetTicksPerFrame() + 1;
        MSTREAMPRINTF  _T("length %d\n"), frames);
        Indent(2);
        MSTREAMPRINTF  _T("rate %d\n"), GetFrameRate());
        Indent(1);
        MSTREAMPRINTF  _T("}\n"));
    }
}

// Traverse the scene graph looking for LOD nodes.
void
VRBLExport::TraverseNode(INode *node)
{
    if (!node)
        return;
    Object *obj = node->EvalWorldState(mStart).obj;

    if (obj && obj->ClassID() == Class_ID(LOD_CLASS_ID1, LOD_CLASS_ID2))
        mLodList = mLodList->AddNode(node);

    if (IsLight(node) && !IsEverAnimated(node))
    {
        OutputTopLevelLight(node, (LightObject *)obj);
    }

    if (obj)
    {
        Class_ID id = obj->ClassID();
        if ((id == Class_ID(SIMPLE_CAM_CLASS_ID, 0) || id == Class_ID(LOOKAT_CAM_CLASS_ID, 0)) && !IsEverAnimated(node))
            VrmlOutTopLevelCamera(1, node, FALSE);
    }

    int n = node->NumberOfChildren();
    for (int i = 0; i < n; i++)
        TraverseNode(node->GetChildNode(i));
}

void
VRBLExport::ComputeWorldBoundBox(INode *node, ViewExp *vpt)
{
    if (!node)
        return;
    Object *obj = node->EvalWorldState(mStart).obj;
    Class_ID id;

    if (obj)
    {
        Box3 bb;
        obj->GetWorldBoundBox(mStart, node, vpt, bb);
        mBoundBox += bb;
    }

    int n = node->NumberOfChildren();
    for (int i = 0; i < n; i++)
        ComputeWorldBoundBox(node->GetChildNode(i), vpt);
}

// Make a list of al the LOD objects in the scene.
void
VRBLExport::ScanSceneGraph()
{
    //    ViewExp *vpt = mIp->GetViewport(NULL);
    INode *node = mIp->GetRootNode();
    //    ComputeWorldBoundBox(node, vpt);
    TraverseNode(node);
}

// Return TRUE iff the node is referenced by the LOD node.
static BOOL
ObjectIsReferenced(INode *lodNode, INode *node)
{
    Object *obj = lodNode->GetObjectRef();
    int numRefs = obj->NumRefs();

    for (int i = 0; i < numRefs; i++)
        if (node == (INode *)obj->GetReference(i))
            return TRUE;

    return FALSE;
}

// Return TRUE iff the node is referenced by ANY LOD node.
BOOL
VRBLExport::ObjectIsLODRef(INode *node)
{
    INodeList *l = mLodList;

    for (; l; l = l->GetNext())
        if (ObjectIsReferenced(l->GetNode(), node))
            return TRUE;

    return FALSE;
}

// Dialog procedures

// Collect up a table with pointers to all the camera nodes in it
void
VRBLExport::GetCameras(INode *inode, Tab<INode *> *camList,
                       Tab<INode *> *navInfoList,
                       Tab<INode *> *backgrounds,
                       Tab<INode *> *fogs,
                       Tab<INode *> *skys)
{
    const ObjectState &os = inode->EvalWorldState(mStart);
    Object *ob = os.obj;
    if (ob != NULL)
    {
        if (ob->SuperClassID() == CAMERA_CLASS_ID)
            camList->Append(1, &inode);

        if (ob->ClassID() == NavInfoClassID)
            navInfoList->Append(1, &inode);

        if (ob->ClassID() == BackgroundClassID)
            backgrounds->Append(1, &inode);

        if (ob->ClassID() == FogClassID)
            fogs->Append(1, &inode);

        if (ob->ClassID() == SkyClassID)
            skys->Append(1, &inode);
    }
    int count = inode->NumberOfChildren();
    for (int i = 0; i < count; i++)
        GetCameras(inode->GetChildNode(i), camList, navInfoList,
                   backgrounds, fogs, skys);
}

// Get a chunk of app data off the sound object
void
GetAppData(Interface *ip, int id, TCHAR *def, TCHAR *val, int len)
{
    SoundObj *node = ip->GetSoundObject();
    AppDataChunk *ad = node->GetAppDataChunk(Class_ID(VRBL_EXPORT_CLASS_ID, 0),
                                             SCENE_EXPORT_CLASS_ID, id);
    if (!ad)
        _tcscpy(val, def);
    else
        _tcscpy(val, (TCHAR *)ad->data);
}

// Write a chunk of app data on the sound object
void
WriteAppData(Interface *ip, int id, TCHAR *val)
{
    SoundObj *node = ip->GetSoundObject();
    node->RemoveAppDataChunk(Class_ID(VRBL_EXPORT_CLASS_ID, 0),
                             SCENE_EXPORT_CLASS_ID, id);
    int size = static_cast<int>((_tcslen(val) + 1) * sizeof(TCHAR));
    TCHAR *buf = (TCHAR *)MAX_malloc(size);
    ////TCHAR* buf = (TCHAR*) malloc(_tcslen(val)+1);
    _tcscpy(buf, val);
    node->AddAppDataChunk(Class_ID(VRBL_EXPORT_CLASS_ID, 0),
                          SCENE_EXPORT_CLASS_ID, id,
                          size, buf);
    SetSaveRequiredFlag(TRUE);
}

extern HINSTANCE hInstance;

ISpinnerControl *VRBLExport::tformSpin = NULL;
ISpinnerControl *VRBLExport::coordSpin = NULL;
ISpinnerControl *VRBLExport::flipbookSpin = NULL;

static INT_PTR CALLBACK
    SampleRatesDlgProc(HWND hDlg, UINT msg, WPARAM wParam, LPARAM lParam)
{
    TCHAR text[MAX_PATH];
    VRBLExport *exp;
    if (msg == WM_INITDIALOG)
    {
        SetWindowLongPtr(hDlg, GWLP_USERDATA, lParam);
    }
    exp = (VRBLExport *)GetWindowLongPtr(hDlg, GWLP_USERDATA);
    switch (msg)
    {
    case WM_INITDIALOG:
    {
        CenterWindow(hDlg, GetParent(hDlg));
        // transform sample rate
        GetAppData(exp->mIp, TFORM_SAMPLE_ID, _T("custom"), text, MAX_PATH);
        BOOL once = _tcscmp(text, _T("once")) == 0;
        CheckDlgButton(hDlg, IDC_TFORM_ONCE, once);
        CheckDlgButton(hDlg, IDC_TFORM_CUSTOM, !once);
        EnableWindow(GetDlgItem(hDlg, IDC_TFORM_EDIT), !once);
        EnableWindow(GetDlgItem(hDlg, IDC_TFORM_SPIN), !once);

        GetAppData(exp->mIp, TFORM_SAMPLE_RATE_ID, _T("10"), text, MAX_PATH);
        int sampleRate = _tstoi(text);

        exp->tformSpin = GetISpinner(GetDlgItem(hDlg, IDC_TFORM_SPIN));
        exp->tformSpin->SetLimits(1, 100);
        exp->tformSpin->SetValue(sampleRate, FALSE);
        exp->tformSpin->SetAutoScale();
        exp->tformSpin->LinkToEdit(GetDlgItem(hDlg, IDC_TFORM_EDIT), EDITTYPE_INT);

        // coordinate interpolator sample rate
        GetAppData(exp->mIp, COORD_SAMPLE_ID, _T("custom"), text, MAX_PATH);
        once = _tcscmp(text, _T("once")) == 0;
        CheckDlgButton(hDlg, IDC_COORD_ONCE, once);
        CheckDlgButton(hDlg, IDC_COORD_CUSTOM, !once);
        EnableWindow(GetDlgItem(hDlg, IDC_COORD_EDIT), !once);
        EnableWindow(GetDlgItem(hDlg, IDC_COORD_SPIN), !once);

        GetAppData(exp->mIp, COORD_SAMPLE_RATE_ID, _T("3"), text, MAX_PATH);
        sampleRate = _tstoi(text);

        exp->coordSpin = GetISpinner(GetDlgItem(hDlg, IDC_COORD_SPIN));
        exp->coordSpin->SetLimits(1, 100);
        exp->coordSpin->SetValue(sampleRate, FALSE);
        exp->coordSpin->SetAutoScale();
        exp->coordSpin->LinkToEdit(GetDlgItem(hDlg, IDC_COORD_EDIT), EDITTYPE_INT);

        // flipbook sample rate
        GetAppData(exp->mIp, FLIPBOOK_SAMPLE_ID, _T("custom"), text, MAX_PATH);
        once = _tcscmp(text, _T("once")) == 0;
        CheckDlgButton(hDlg, IDC_FLIPBOOK_ONCE, once);
        CheckDlgButton(hDlg, IDC_FLIPBOOK_CUSTOM, !once);
        EnableWindow(GetDlgItem(hDlg, IDC_FLIPBOOK_EDIT), !once);
        EnableWindow(GetDlgItem(hDlg, IDC_FLIPBOOK_SPIN), !once);

        GetAppData(exp->mIp, FLIPBOOK_SAMPLE_RATE_ID, _T("10"), text, MAX_PATH);
        sampleRate = _tstoi(text);

        exp->flipbookSpin = GetISpinner(GetDlgItem(hDlg, IDC_FLIPBOOK_SPIN));
        exp->flipbookSpin->SetLimits(1, 100);
        exp->flipbookSpin->SetValue(sampleRate, FALSE);
        exp->flipbookSpin->SetAutoScale();
        exp->flipbookSpin->LinkToEdit(GetDlgItem(hDlg, IDC_FLIPBOOK_EDIT), EDITTYPE_INT);

        return TRUE;
    }
    case WM_DESTROY:
        ReleaseISpinner(exp->tformSpin);
        ReleaseISpinner(exp->coordSpin);
        ReleaseISpinner(exp->flipbookSpin);
        break;
    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        case IDC_TFORM_ONCE:
            exp->tformSpin->Disable();
            return TRUE;
        case IDC_TFORM_CUSTOM:
            exp->tformSpin->Enable();
            return TRUE;
        case IDC_COORD_ONCE:
            exp->coordSpin->Disable();
            return TRUE;
        case IDC_COORD_CUSTOM:
            exp->coordSpin->Enable();
            return TRUE;
        case IDC_FLIPBOOK_ONCE:
            exp->flipbookSpin->Disable();
            return TRUE;
        case IDC_FLIPBOOK_CUSTOM:
            exp->flipbookSpin->Enable();
            return TRUE;
        case IDCANCEL:
            EndDialog(hDlg, FALSE);
            return TRUE;
            break;
        case IDOK:
        {
            BOOL once = IsDlgButtonChecked(hDlg, IDC_TFORM_ONCE);
            exp->SetTformSample(once);
            TCHAR *val = once ? _T("once") : _T("custom");
            WriteAppData(exp->mIp, TFORM_SAMPLE_ID, val);
            int rate = exp->tformSpin->GetIVal();
            exp->SetTformSampleRate(rate);
            _stprintf(text, _T("%d"), rate);
            WriteAppData(exp->mIp, TFORM_SAMPLE_RATE_ID, text);

            once = IsDlgButtonChecked(hDlg, IDC_COORD_ONCE);
            exp->SetCoordSample(once);
            val = once ? _T("once") : _T("custom");
            WriteAppData(exp->mIp, COORD_SAMPLE_ID, val);
            rate = exp->coordSpin->GetIVal();
            exp->SetCoordSampleRate(rate);
            _stprintf(text, _T("%d"), rate);
            WriteAppData(exp->mIp, COORD_SAMPLE_RATE_ID, text);

            once = IsDlgButtonChecked(hDlg, IDC_FLIPBOOK_ONCE);
            exp->SetFlipbookSample(once);
            val = once ? _T("once") : _T("custom");
            WriteAppData(exp->mIp, FLIPBOOK_SAMPLE_ID, val);
            rate = exp->flipbookSpin->GetIVal();
            exp->SetFlipbookSampleRate(rate);
            _stprintf(text, _T("%d"), rate);
            WriteAppData(exp->mIp, FLIPBOOK_SAMPLE_RATE_ID, text);

            EndDialog(hDlg, TRUE);
            return TRUE;
        }
        }
    }
    return FALSE;
}

static INT_PTR CALLBACK
    WorldInfoDlgProc(HWND hDlg, UINT msg, WPARAM wParam, LPARAM lParam)
{
    TCHAR text[MAX_PATH];
    VRBLExport *exp;
    if (msg == WM_INITDIALOG)
    {
        SetWindowLongPtr(hDlg, GWLP_USERDATA, lParam);
    }
    exp = (VRBLExport *)GetWindowLongPtr(hDlg, GWLP_USERDATA);
    switch (msg)
    {
    case WM_INITDIALOG:
    {
        CenterWindow(hDlg, GetParent(hDlg));
        GetAppData(exp->mIp, TITLE_ID, _T(""), text, MAX_PATH);
        Edit_SetText(GetDlgItem(hDlg, IDC_TITLE), text);
        GetAppData(exp->mIp, INFO_ID, _T(""), text, MAX_PATH);
        Edit_SetText(GetDlgItem(hDlg, IDC_INFO), text);
        return TRUE;
    }
    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        case IDCANCEL:
            EndDialog(hDlg, FALSE);
            return TRUE;
        case IDOK:
            Edit_GetText(GetDlgItem(hDlg, IDC_TITLE), text, MAX_PATH);
            WriteAppData(exp->mIp, TITLE_ID, text);
            Edit_GetText(GetDlgItem(hDlg, IDC_INFO), text, MAX_PATH);
            WriteAppData(exp->mIp, INFO_ID, text);
            EndDialog(hDlg, TRUE);
            return TRUE;
        }
    }
    return FALSE;
}

static BOOL CALLBACK
    AboutDlgProc(HWND hDlg, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_INITDIALOG:
    {
        CenterWindow(hDlg, GetParent(hDlg));
        return TRUE;
    }
    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        case IDOK:
            EndDialog(hDlg, TRUE);
            return TRUE;
        }
    }
    return FALSE;
}

// Dialog procedure for the export dialog.
static INT_PTR CALLBACK
    VrblExportDlgProc(HWND hDlg, UINT msg, WPARAM wParam, LPARAM lParam)
{
    TCHAR text[MAX_PATH];
    VRBLExport *exp;
    ExportType type;
    if (msg == WM_INITDIALOG)
    {
        SetWindowLongPtr(hDlg, GWLP_USERDATA, lParam);
    }
    exp = (VRBLExport *)GetWindowLongPtr(hDlg, GWLP_USERDATA);
    switch (msg)
    {
    case WM_INITDIALOG:
    {
        HWND cb = GetDlgItem(hDlg, IDC_FILE_FORMAT);
        ComboBox_AddString(cb, _T("X3D(V)"));
        ComboBox_AddString(cb, _T("Vrml97_COVER"));
        ComboBox_AddString(cb, _T("Vrml97_Standard"));
        ComboBox_AddString(cb, _T("Inventor"));
        GetAppData(exp->mIp, IDC_FILE_FORMAT, _T("X3D(V)"), text, MAX_PATH);
        ComboBox_SelectString(cb, 0, text);

        SetWindowContextHelpId(hDlg, idh_vrmlexp_export);
        CenterWindow(hDlg, GetParent(hDlg));
        GetAppData(exp->mIp, NORMALS_ID, _T("yes"), text, MAX_PATH);
        BOOL gen = _tcscmp(text, _T("yes")) == 0;
        CheckDlgButton(hDlg, IDC_GENNORMALS, gen);
        GetAppData(exp->mIp, DEFUSE_ID, _T("yes"), text, MAX_PATH);
        BOOL defu = _tcscmp(text, _T("yes")) == 0;
        CheckDlgButton(hDlg, IDC_DEF_USE, defu);
        GetAppData(exp->mIp, USELOD_ID, _T("no"), text, MAX_PATH);
        BOOL uselo = _tcscmp(text, _T("yes")) == 0;
        CheckDlgButton(hDlg, IDC_USELOD, uselo);
        GetAppData(exp->mIp, OCCLUDER_ID, _T("no"), text, MAX_PATH);
        BOOL occluder = _tcscmp(text, _T("yes")) == 0;
        CheckDlgButton(hDlg, IDC_Occluder, occluder);
        GetAppData(exp->mIp, EXPORTLIGHTS_ID, _T("no"), text, MAX_PATH);
        BOOL explights = _tcscmp(text, _T("yes")) == 0;
        CheckDlgButton(hDlg, IDC_EXPORTLIGHTS, explights);
        GetAppData(exp->mIp, COPYTEXTURES_ID, _T("yes"), text, MAX_PATH);
        BOOL copytextures = _tcscmp(text, _T("yes")) == 0;
        CheckDlgButton(hDlg, IDC_COPYTEXTURES, copytextures);
        GetAppData(exp->mIp, FORCE_WHITE_ID, _T("yes"), text, MAX_PATH);
        BOOL forceWhite = _tcscmp(text, _T("yes")) == 0;
        CheckDlgButton(hDlg, IDC_FORCE_WHITE, forceWhite);
        GetAppData(exp->mIp, INDENT_ID, _T("yes"), text, MAX_PATH);
        gen = _tcscmp(text, _T("yes")) == 0;
        CheckDlgButton(hDlg, IDC_INDENT, gen);
        GetAppData(exp->mIp, FIELDS_ID, _T("yes"), text, MAX_PATH);
        gen = _tcscmp(text, _T("yes")) == 0;
        //CheckDlgButton(hDlg, IDC_GEN_FIELDS, gen);
        GetAppData(exp->mIp, UPDIR_ID, _T("Y"), text, MAX_PATH);
        gen = _tcscmp(text, _T("Z")) == 0;
        //CheckDlgButton(hDlg, IDC_Z_UP, gen);
        //CheckDlgButton(hDlg, IDC_Y_UP, !gen);
        GetAppData(exp->mIp, COORD_INTERP_ID, _T("yes"), text, MAX_PATH);
        gen = _tcscmp(text, _T("yes")) == 0;
        CheckDlgButton(hDlg, IDC_COORD_INTERP, gen);
        GetAppData(exp->mIp, EXPORT_HIDDEN_ID, _T("no"), text, MAX_PATH);
        gen = _tcscmp(text, _T("yes")) == 0;
        CheckDlgButton(hDlg, IDC_EXPORT_HIDDEN, gen);
        GetAppData(exp->mIp, ENABLE_PROGRESS_BAR_ID, _T("yes"), text, MAX_PATH);
        gen = _tcscmp(text, _T("yes")) == 0;
        CheckDlgButton(hDlg, IDC_ENABLE_PROGRESS_BAR, gen);

        GetAppData(exp->mIp, PRIMITIVES_ID, _T("no"), text, MAX_PATH);
        gen = _tcscmp(text, _T("yes")) == 0;
        CheckDlgButton(hDlg, IDC_PRIM, gen);

        GetAppData(exp->mIp, EXPORT_PRE_LIGHT_ID, _T("no"), text, MAX_PATH);
        gen = _tcscmp(text, _T("yes")) == 0;
        CheckDlgButton(hDlg, IDC_COLOR_PER_VERTEX, gen);
        EnableWindow(GetDlgItem(hDlg, IDC_CPV_CALC), gen);
        EnableWindow(GetDlgItem(hDlg, IDC_CPV_MAX), gen);

        GetAppData(exp->mIp, CPV_SOURCE_ID, _T("max"), text, MAX_PATH);
        gen = _tcscmp(text, _T("max")) == 0;
        CheckDlgButton(hDlg, IDC_CPV_MAX, gen);
        CheckDlgButton(hDlg, IDC_CPV_CALC, !gen);

#ifdef _LEC_
        GetAppData(exp->mIp, FLIP_BOOK_ID, _T("no"), text, MAX_PATH);
        gen = _tcscmp(text, _T("yes")) == 0;
        CheckDlgButton(hDlg, IDC_FLIP_BOOK, gen);
#endif

        // Time to make a list of all the camera's in the scene
        Tab<INode *> cameras, navInfos, backgrounds, fogs, skys;
        exp->GetCameras(exp->GetIP()->GetRootNode(), &cameras, &navInfos,
                        &backgrounds, &fogs, &skys);
        int c = cameras.Count();
        int i;
        for (i = 0; i < c; i++)
        {
            // add the name to the list
            TSTR name = cameras[i]->GetName();
            int ind = (int)SendMessage(GetDlgItem(hDlg, IDC_CAMERA_COMBO),
                                       CB_ADDSTRING, 0, (LPARAM)name.data());
            SendMessage(GetDlgItem(hDlg, IDC_CAMERA_COMBO), CB_SETITEMDATA,
                        ind, (LPARAM)cameras[i]);
        }
        if (c > 0)
        {
            TSTR name;
            GetAppData(exp->mIp, CAMERA_ID, _T(""), text, MAX_PATH);
            if (_tcslen(text) == 0)
                name = cameras[0]->GetName();
            else
                name = text;
            // try to set the current selecttion to the current camera
            SendMessage(GetDlgItem(hDlg, IDC_CAMERA_COMBO), CB_SELECTSTRING,
                        0, (LPARAM)name.data());
        }

        c = navInfos.Count();
        for (i = 0; i < c; i++)
        {
            // add the name to the list
            TSTR name = navInfos[i]->GetName();
            int ind = (int)SendMessage(GetDlgItem(hDlg, IDC_NAV_INFO_COMBO),
                                       CB_ADDSTRING, 0, (LPARAM)name.data());
            SendMessage(GetDlgItem(hDlg, IDC_NAV_INFO_COMBO), CB_SETITEMDATA,
                        ind, (LPARAM)navInfos[i]);
        }
        if (c > 0)
        {
            TSTR name;
            GetAppData(exp->mIp, NAV_INFO_ID, _T(""), text, MAX_PATH);
            if (_tcslen(text) == 0)
                name = navInfos[0]->GetName();
            else
                name = text;
            // try to set the current selecttion to the current camera
            SendMessage(GetDlgItem(hDlg, IDC_NAV_INFO_COMBO), CB_SELECTSTRING,
                        0, (LPARAM)name.data());
        }

        c = backgrounds.Count();
        for (i = 0; i < c; i++)
        {
            // add the name to the list
            TSTR name = backgrounds[i]->GetName();
            int ind = (int)SendMessage(GetDlgItem(hDlg, IDC_BACKGROUND_COMBO),
                                       CB_ADDSTRING, 0, (LPARAM)name.data());
            SendMessage(GetDlgItem(hDlg, IDC_BACKGROUND_COMBO), CB_SETITEMDATA,
                        ind, (LPARAM)backgrounds[i]);
        }
        if (c > 0)
        {
            TSTR name;
            GetAppData(exp->mIp, BACKGROUND_ID, _T(""), text, MAX_PATH);
            if (_tcslen(text) == 0)
                name = backgrounds[0]->GetName();
            else
                name = text;
            // try to set the current selecttion to the current camera
            SendMessage(GetDlgItem(hDlg, IDC_BACKGROUND_COMBO),
                        CB_SELECTSTRING, 0, (LPARAM)name.data());
        }

        c = fogs.Count();
        for (i = 0; i < c; i++)
        {
            // add the name to the list
            TSTR name = fogs[i]->GetName();
            int ind = (int)SendMessage(GetDlgItem(hDlg, IDC_FOG_COMBO),
                                       CB_ADDSTRING, 0, (LPARAM)name.data());
            SendMessage(GetDlgItem(hDlg, IDC_FOG_COMBO), CB_SETITEMDATA,
                        ind, (LPARAM)fogs[i]);
        }
        if (c > 0)
        {
            TSTR name;
            GetAppData(exp->mIp, FOG_ID, _T(""), text, MAX_PATH);
            if (_tcslen(text) == 0)
                name = fogs[0]->GetName();
            else
                name = text;
            // try to set the current selecttion to the current camera
            SendMessage(GetDlgItem(hDlg, IDC_FOG_COMBO),
                        CB_SELECTSTRING, 0, (LPARAM)name.data());
        }
        c = skys.Count();
        for (i = 0; i < c; i++)
        {
            // add the name to the list
            TSTR name = skys[i]->GetName();
            int ind = (int)SendMessage(GetDlgItem(hDlg, IDC_SKY_COMBO),
                                       CB_ADDSTRING, 0, (LPARAM)name.data());
            SendMessage(GetDlgItem(hDlg, IDC_SKY_COMBO), CB_SETITEMDATA,
                        ind, (LPARAM)skys[i]);
        }
        if (c > 0)
        {
            TSTR name;
            GetAppData(exp->mIp, SKY_ID, _T(""), text, MAX_PATH);
            if (_tcslen(text) == 0)
                name = skys[0]->GetName();
            else
                name = text;
            // try to set the current selecttion to the current camera
            SendMessage(GetDlgItem(hDlg, IDC_SKY_COMBO),
                        CB_SELECTSTRING, 0, (LPARAM)name.data());
        }

        GetAppData(exp->mIp, OUTPUT_LANG_ID, _T("X3D(V)"), text, MAX_PATH);
        if (_tcscmp(text, _T("X3D(V)")) == 0)
        {
            type = Export_X3D_V;
            ComboBox_SelectString(GetDlgItem(hDlg, IDC_FILE_FORMAT), 0, _T("X3D(V)"));
        }
        else if ((_tcscmp(text, _T("Vrml97_COVER")) == 0) || (_tcscmp(text, _T("VRML97")) == 0))
        {
            type = Export_VRML_2_0_COVER;
            ComboBox_SelectString(GetDlgItem(hDlg, IDC_FILE_FORMAT), 0, _T("Vrml97_COVER"));
        }
        else if (_tcscmp(text, _T("Vrml97_Standard")) == 0)
        {
            type = Export_VRML_2_0;
            ComboBox_SelectString(GetDlgItem(hDlg, IDC_FILE_FORMAT), 0, _T("Vrml97_Standard"));
        }
        else if (_tcscmp(text, _T("Inventor")) == 0)
        {
            type = Export_VRML_1_0;
            ComboBox_SelectString(GetDlgItem(hDlg, IDC_FILE_FORMAT), 0, _T("Inventor"));
        }
        else
        {
            type = Export_X3D_V;
            ComboBox_SelectString(GetDlgItem(hDlg, IDC_FILE_FORMAT), 0, _T("X3D(V)"));
        }

#ifdef _LEC_
        EnableWindow(GetDlgItem(hDlg, IDC_FLIP_BOOK), TRUE);
#else
        EnableWindow(GetDlgItem(hDlg, IDC_FLIP_BOOK), FALSE);
#endif
        //ComboBox_SelectString(cb, 0, text);
        GetAppData(exp->mIp, USE_PREFIX_ID, _T("yes"), text, MAX_PATH);
        CheckDlgButton(hDlg, IDC_USE_PREFIX, _tcscmp(text, _T("yes")) == 0);
        GetAppData(exp->mIp, URL_PREFIX_ID, _T("maps"), text, MAX_PATH);
        Edit_SetText(GetDlgItem(hDlg, IDC_URL_PREFIX), text);
        cb = GetDlgItem(hDlg, IDC_DIGITS);
        ComboBox_AddString(cb, _T("3"));
        ComboBox_AddString(cb, _T("4"));
        ComboBox_AddString(cb, _T("5"));
        ComboBox_AddString(cb, _T("6"));
        GetAppData(exp->mIp, DIGITS_ID, _T("6"), text, MAX_PATH);
        ComboBox_SelectString(cb, 0, text);

        cb = GetDlgItem(hDlg, IDC_POLYGON_TYPE);
        ComboBox_AddString(cb, GetString(IDS_OUT_TRIANGLES));
#if TRUE // outputing higher order polygons
        ComboBox_AddString(cb, GetString(IDS_OUT_QUADS));
        ComboBox_AddString(cb, GetString(IDS_OUT_NGONS));
        ComboBox_AddString(cb, GetString(IDS_OUT_VIS_EDGES));
#endif
        GetAppData(exp->mIp, POLYGON_TYPE_ID, GetString(IDS_OUT_TRIANGLES), text, MAX_PATH);
        ComboBox_SelectString(cb, 0, text);

        // make sure the appropriate things are enabled
        /* this is not always appropriate
        BOOL checked = IsDlgButtonChecked(hDlg, IDC_PRIM);
        EnableWindow(GetDlgItem(hDlg, IDC_COLOR_PER_VERTEX), !checked);
        if (checked) CheckDlgButton(hDlg, IDC_COLOR_PER_VERTEX, FALSE);
        BOOL cpvChecked = IsDlgButtonChecked(hDlg, IDC_COLOR_PER_VERTEX);
        EnableWindow(GetDlgItem(hDlg, IDC_CPV_CALC), cpvChecked);
        EnableWindow(GetDlgItem(hDlg, IDC_CPV_MAX),  cpvChecked);
        EnableWindow(GetDlgItem(hDlg, IDC_GENNORMALS), !checked);
        if (checked) CheckDlgButton(hDlg, IDC_GENNORMALS, FALSE);
        EnableWindow(GetDlgItem(hDlg, IDC_COORD_INTERP), !checked);
        if (checked) CheckDlgButton(hDlg, IDC_COORD_INTERP, FALSE);
        */

        return TRUE;
    }
    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        case IDC_EXP_HELP:
        {
            const TCHAR *helpDir = exp->mIp->GetDir(APP_HELP_DIR);
            TCHAR helpFile[MAX_PATH];
            _tcscpy(helpFile, helpDir);
            _tcscat(helpFile, _T("\\vrmlout.hlp"));
            WinHelp(hDlg, helpFile, HELP_CONTENTS, NULL);
            break;
        }
        /*
        case IDC_PRIM: {
            BOOL checked = IsDlgButtonChecked(hDlg, IDC_PRIM);

            EnableWindow(GetDlgItem(hDlg, IDC_COLOR_PER_VERTEX), !checked);
            if (checked) CheckDlgButton(hDlg, IDC_COLOR_PER_VERTEX, FALSE);
            BOOL cpvChecked = IsDlgButtonChecked(hDlg, IDC_COLOR_PER_VERTEX);
            EnableWindow(GetDlgItem(hDlg, IDC_CPV_CALC), cpvChecked);
            EnableWindow(GetDlgItem(hDlg, IDC_CPV_MAX),  cpvChecked);

            EnableWindow(GetDlgItem(hDlg, IDC_GENNORMALS), !checked);
            if (checked) CheckDlgButton(hDlg, IDC_GENNORMALS, FALSE);

            EnableWindow(GetDlgItem(hDlg, IDC_COORD_INTERP), !checked);
            if (checked) CheckDlgButton(hDlg, IDC_COORD_INTERP, FALSE);

            break;
            }
            */
        case IDC_COLOR_PER_VERTEX:
        {
            BOOL checked = IsDlgButtonChecked(hDlg, IDC_COLOR_PER_VERTEX);
            EnableWindow(GetDlgItem(hDlg, IDC_CPV_CALC), checked);
            EnableWindow(GetDlgItem(hDlg, IDC_CPV_MAX), checked);
            break;
        }
        case IDCANCEL:
            EndDialog(hDlg, FALSE);
            break;
        case IDOK:
        {
            exp->SetGenNormals(IsDlgButtonChecked(hDlg, IDC_GENNORMALS));
            WriteAppData(exp->mIp, NORMALS_ID, exp->GetGenNormals() ? _T("yes") : _T("no"));

            exp->SetDefUse(IsDlgButtonChecked(hDlg, IDC_DEF_USE));
            WriteAppData(exp->mIp, DEFUSE_ID, exp->GetDefUse() ? _T("yes") : _T("no"));

            exp->SetExpLights(IsDlgButtonChecked(hDlg, IDC_EXPORTLIGHTS));
            WriteAppData(exp->mIp, EXPORTLIGHTS_ID, exp->GetExpLights() ? _T("yes") : _T("no"));

            exp->SetUseLod(IsDlgButtonChecked(hDlg, IDC_USELOD));
            WriteAppData(exp->mIp, USELOD_ID, exp->GetUseLod() ? _T("yes") : _T("no"));

            exp->SetExportOccluders(IsDlgButtonChecked(hDlg, IDC_Occluder));
            WriteAppData(exp->mIp, OCCLUDER_ID, exp->GetUseLod() ? _T("yes") : _T("no"));

            exp->SetCopyTextures(IsDlgButtonChecked(hDlg, IDC_COPYTEXTURES));
            WriteAppData(exp->mIp, COPYTEXTURES_ID, exp->GetCopyTextures() ? _T("yes") : _T("no"));

            exp->SetForceWhite(IsDlgButtonChecked(hDlg, IDC_FORCE_WHITE));
            WriteAppData(exp->mIp, FORCE_WHITE_ID, exp->GetForceWhite() ? _T("yes") : _T("no"));

            exp->SetIndent(IsDlgButtonChecked(hDlg, IDC_INDENT));
            WriteAppData(exp->mIp, INDENT_ID, exp->GetIndent() ? _T("yes") : _T("no"));
#if 0            
            exp->SetZUp(IsDlgButtonChecked(hDlg, IDC_Z_UP));
#else
            exp->SetZUp(FALSE);
#endif
            WriteAppData(exp->mIp, UPDIR_ID, exp->GetZUp() ? _T("Z") : _T("Y"));

            exp->SetCoordInterp(IsDlgButtonChecked(hDlg, IDC_COORD_INTERP));
            WriteAppData(exp->mIp, COORD_INTERP_ID, exp->GetCoordInterp() ? _T("yes") : _T("no"));
#ifdef _LEC_
            exp->SetFlipBook(IsDlgButtonChecked(hDlg, IDC_FLIP_BOOK));
            WriteAppData(exp->mIp, FLIP_BOOK_ID, exp->GetFlipBook() ? _T("yes") : _T("no"));
#endif

            exp->SetExportHidden(IsDlgButtonChecked(hDlg, IDC_EXPORT_HIDDEN));
            WriteAppData(exp->mIp, EXPORT_HIDDEN_ID, exp->GetExportHidden() ? _T("yes") : _T("no"));

            exp->SetEnableProgressBar(IsDlgButtonChecked(hDlg, IDC_ENABLE_PROGRESS_BAR));
            WriteAppData(exp->mIp, ENABLE_PROGRESS_BAR_ID, exp->GetEnableProgressBar() ? _T("yes") : _T("no"));

            exp->SetPrimitives(IsDlgButtonChecked(hDlg, IDC_PRIM));
            WriteAppData(exp->mIp, PRIMITIVES_ID, exp->GetPrimitives() ? _T("yes") : _T("no"));

            int index = (int)SendMessage(GetDlgItem(hDlg, IDC_CAMERA_COMBO),
                                         CB_GETCURSEL, 0, 0);
            if (index != CB_ERR)
            {
                exp->SetCamera((INode *)
                                   SendMessage(GetDlgItem(hDlg, IDC_CAMERA_COMBO),
                                               CB_GETITEMDATA, (WPARAM)index,
                                               0));
                ComboBox_GetText(GetDlgItem(hDlg, IDC_CAMERA_COMBO),
                                 text, MAX_PATH);
                WriteAppData(exp->mIp, CAMERA_ID, text);
            }
            else
                exp->SetCamera(NULL);

            index = (int)SendMessage(GetDlgItem(hDlg, IDC_NAV_INFO_COMBO),
                                     CB_GETCURSEL, 0, 0);
            if (index != CB_ERR)
            {
                exp->SetNavInfo((INode *)
                                    SendMessage(GetDlgItem(hDlg, IDC_NAV_INFO_COMBO),
                                                CB_GETITEMDATA, (WPARAM)index,
                                                0));
                ComboBox_GetText(GetDlgItem(hDlg, IDC_NAV_INFO_COMBO),
                                 text, MAX_PATH);
                WriteAppData(exp->mIp, NAV_INFO_ID, text);
            }
            else
                exp->SetNavInfo(NULL);

            index = (int)SendMessage(GetDlgItem(hDlg, IDC_BACKGROUND_COMBO),
                                     CB_GETCURSEL, 0, 0);
            if (index != CB_ERR)
            {
                exp->SetBackground((INode *)
                                       SendMessage(GetDlgItem(hDlg, IDC_BACKGROUND_COMBO),
                                                   CB_GETITEMDATA, (WPARAM)index,
                                                   0));
                ComboBox_GetText(GetDlgItem(hDlg, IDC_BACKGROUND_COMBO),
                                 text, MAX_PATH);
                WriteAppData(exp->mIp, BACKGROUND_ID, text);
            }
            else
                exp->SetBackground(NULL);

            index = (int)SendMessage(GetDlgItem(hDlg, IDC_FOG_COMBO),
                                     CB_GETCURSEL, 0, 0);
            if (index != CB_ERR)
            {
                exp->SetFog((INode *)
                                SendMessage(GetDlgItem(hDlg, IDC_FOG_COMBO),
                                            CB_GETITEMDATA, (WPARAM)index,
                                            0));
                ComboBox_GetText(GetDlgItem(hDlg, IDC_FOG_COMBO),
                                 text, MAX_PATH);
                WriteAppData(exp->mIp, FOG_ID, text);
            }
            else
                exp->SetFog(NULL);

            index = (int)SendMessage(GetDlgItem(hDlg, IDC_SKY_COMBO),
                                     CB_GETCURSEL, 0, 0);
            if (index != CB_ERR)
            {
                exp->SetSky((INode *)
                                SendMessage(GetDlgItem(hDlg, IDC_SKY_COMBO),
                                            CB_GETITEMDATA, (WPARAM)index,
                                            0));
                ComboBox_GetText(GetDlgItem(hDlg, IDC_SKY_COMBO),
                                 text, MAX_PATH);
                WriteAppData(exp->mIp, SKY_ID, text);
            }
            else
                exp->SetSky(NULL);
            index = (int)SendMessage(GetDlgItem(hDlg, IDC_FILE_FORMAT),
                                     CB_GETCURSEL, 0, 0);
            if (index != CB_ERR)
            {

                ComboBox_GetText(GetDlgItem(hDlg, IDC_FILE_FORMAT),
                                 text, MAX_PATH);
                if (_tcscmp(text, _T("X3D(V)")) == 0)
                {
                    exp->SetExportType(Export_X3D_V);
                    WriteAppData(exp->mIp, OUTPUT_LANG_ID, _T("X3D(V)"));
                }
                else if (_tcscmp(text, _T("Vrml97_COVER")) == 0)
                {
                    exp->SetExportType(Export_VRML_2_0_COVER);
                    WriteAppData(exp->mIp, OUTPUT_LANG_ID, _T("Vrml97_COVER"));
                }
                else if (_tcscmp(text, _T("Vrml97_Standard")) == 0)
                {
                    exp->SetExportType(Export_VRML_2_0);
                    WriteAppData(exp->mIp, OUTPUT_LANG_ID, _T("Vrml97_Standard"));
                }
                else
                {
                    exp->SetExportType(Export_VRML_1_0);
                    WriteAppData(exp->mIp, OUTPUT_LANG_ID, _T("Inventor"));
                }
            }
            else
                exp->SetExportType(Export_VRML_2_0_COVER);

            ComboBox_GetText(GetDlgItem(hDlg, IDC_POLYGON_TYPE), text, MAX_PATH);
            WriteAppData(exp->mIp, POLYGON_TYPE_ID, text);
            if (_tcscmp(text, _T("Visible Edges")) == 0)
                exp->SetPolygonType(OUTPUT_VISIBLE_EDGES);
            else if (_tcscmp(text, _T("Ngons")) == 0)
                exp->SetPolygonType(OUTPUT_NGONS);
            else if (_tcscmp(text, _T("Quads")) == 0)
                exp->SetPolygonType(OUTPUT_QUADS);
            else
                exp->SetPolygonType(OUTPUT_TRIANGLES);

            exp->SetPreLight(IsDlgButtonChecked(hDlg, IDC_COLOR_PER_VERTEX));
            WriteAppData(exp->mIp, EXPORT_PRE_LIGHT_ID, exp->GetPreLight() ? _T("yes") : _T("no"));

            exp->SetCPVSource(IsDlgButtonChecked(hDlg, IDC_CPV_MAX));
            WriteAppData(exp->mIp, CPV_SOURCE_ID, exp->GetCPVSource() ? _T("max") : _T("calc"));

            exp->SetUsePrefix(IsDlgButtonChecked(hDlg, IDC_USE_PREFIX));
            WriteAppData(exp->mIp, USE_PREFIX_ID, exp->GetUsePrefix()
                                                      ? _T("yes")
                                                      : _T("no"));
            Edit_GetText(GetDlgItem(hDlg, IDC_URL_PREFIX), text, MAX_PATH);
            TSTR prefix = text;
            exp->SetUrlPrefix(prefix);
            WriteAppData(exp->mIp, URL_PREFIX_ID, (TCHAR *)exp->GetUrlPrefix().data());
            ComboBox_GetText(GetDlgItem(hDlg, IDC_DIGITS), text, MAX_PATH);
            exp->SetDigits(_tstoi(text));
            WriteAppData(exp->mIp, DIGITS_ID, text);

            GetAppData(exp->mIp, TFORM_SAMPLE_ID, _T("custom"), text,
                       MAX_PATH);
            BOOL once = _tcscmp(text, _T("once")) == 0;
            exp->SetTformSample(once);
            GetAppData(exp->mIp, TFORM_SAMPLE_RATE_ID, _T("10"), text,
                       MAX_PATH);
            int sampleRate = _tstoi(text);
            exp->SetTformSampleRate(sampleRate);

            GetAppData(exp->mIp, COORD_SAMPLE_ID, _T("custom"), text,
                       MAX_PATH);
            once = _tcscmp(text, _T("once")) == 0;
            exp->SetCoordSample(once);
            GetAppData(exp->mIp, COORD_SAMPLE_RATE_ID, _T("3"), text,
                       MAX_PATH);
            sampleRate = _tstoi(text);
            exp->SetCoordSampleRate(sampleRate);

            GetAppData(exp->mIp, FLIPBOOK_SAMPLE_ID, _T("custom"), text,
                       MAX_PATH);
            once = _tcscmp(text, _T("once")) == 0;
            exp->SetFlipbookSample(once);
            GetAppData(exp->mIp, FLIPBOOK_SAMPLE_RATE_ID, _T("10"), text,
                       MAX_PATH);
            sampleRate = _tstoi(text);
            exp->SetFlipbookSampleRate(sampleRate);

            GetAppData(exp->mIp, TITLE_ID, _T(""), text, MAX_PATH);
            exp->SetTitle(text);
            GetAppData(exp->mIp, INFO_ID, _T(""), text, MAX_PATH);
            exp->SetInfo(text);
            EndDialog(hDlg, TRUE);
            break;
        }
        case IDC_SAMPLE_RATES:
            DialogBoxParam(hInstance, MAKEINTRESOURCE(IDD_SAMPLE_RATES),
                           GetActiveWindow(), SampleRatesDlgProc,
                           (LPARAM)exp);
            break;
        case IDC_WORLD_INFO:
            DialogBoxParam(hInstance, MAKEINTRESOURCE(IDD_WORLD_INFO),
                           GetActiveWindow(), WorldInfoDlgProc,
                           (LPARAM)exp);
            break;
        }
        break;
    case WM_SYSCOMMAND:
        //if ((wParam & 0xfff0) == SC_CONTEXTHELP)
        //  DoHelp(HELP_CONTEXT, idh_3dsexp_export);
        break;
    }
    return FALSE;
}

BOOL VRBLExport::SupportsOptions(int ext, DWORD options)
{
    if (options == SCENE_EXPORT_SELECTED)
        return true;
    return false;
}
// Export the current scene as VRML
int
VRBLExport::DoExport(const TCHAR *filename, ExpInterface *ei, Interface *i, BOOL suppressPrompts, DWORD options)
{
    mIp = i;
    mStart = mIp->GetAnimRange().Start();
    mExportSelected = false;

    DisableProcessWindowsGhosting(); // prevents windows from freezing the progressbar
    if (options & SCENE_EXPORT_SELECTED)
        mExportSelected = true;
    if (suppressPrompts)
        initializeDefaults();
    else if (!DialogBoxParam(hInstance, MAKEINTRESOURCE(IDD_VRBLEXP),
                             GetActiveWindow(), VrblExportDlgProc,
                             (LPARAM) this))
        return TRUE;

    if (IsVRML2())
    {
        // generate the callback table of third party dlls
        mCallbacks.GetCallbackMethods(i);

#ifdef _LEC_
        if (this->GetFlipBook())
        {
            int sampleRate;
            int end;
            int lastFrame;
            int numFrames;
            int extLoc;

            if (this->GetFlipbookSample())
                sampleRate = GetTicksPerFrame();
            else
                sampleRate = TIME_TICKSPERSEC / this->GetFlipbookSampleRate();

            mStart = i->GetAnimRange().Start();
            lastFrame = end = i->GetAnimRange().End();
            numFrames = (end - mStart) / sampleRate + 1;

            if (((end - mStart) % sampleRate) != 0)
            {
                end += sampleRate;
                numFrames++;
            }

            TSTR rfName(filename);
            extLoc = rfName.last('.');
            if (extLoc != -1)
                rfName.remove(extLoc);
            rfName.Append(_T(".txt"));
            FILE *fio = _tfopen(rfName.data(), _T("w"));
            fprintf(fio, "%s\n", filename);
            fprintf(fio, "Start Time (sec.):\t%d.0\n", mStart / TIME_TICKSPERSEC);
            fprintf(fio, "End Time (sec.):\t%d.0\n", end / TIME_TICKSPERSEC);
            fprintf(fio, "Number of Frames:\t%d\n", numFrames);
            if (fio)
            {
                fclose(fio);
                fio = NULL;
            }

            for (int frame = 0; frame < numFrames; frame++, mStart += sampleRate)
            {
                if (mStart > lastFrame)
                    break;
                VRML2Export vrml2;
                int val = vrml2.DoFBExport(filename, i, this, frame, mStart);
                if (!val)
                    return val;
            }
            return TRUE;
        }
#endif
        VRML2Export vrml2;
        int val = vrml2.DoExport(filename, i, this);

        if (mUseLod)
        {
            TSTR tmpS = filename;
            uselod(STRTOUTF8(tmpS));
        }
        if (mDefUse)
        {
            //			defuse(filename);
        }

        return val;
    }
    else
    {
        // export Inventor (VRML1)

        mIp = i;
        mStart = mIp->GetAnimRange().Start();

#if MAX_PRODUCT_VERSION_MAJOR > 14
        mStream.Open(filename, false, CP_UTF8);
#else
        mStream = _tfopen(filename, _T("a"));
#endif

#if MAX_PRODUCT_VERSION_MAJOR > 14
        if (!mStream.IsFileOpen())
        {
#else
        if (!mStream)
        {
#endif
            TCHAR msg[MAX_PATH];
            TCHAR title[MAX_PATH];
            LoadString(hInstance, IDS_OPEN_FAILED, msg, MAX_PATH);
            LoadString(hInstance, IDS_VRML_EXPORT, title, MAX_PATH);
            MessageBox(GetActiveWindow(), msg, title, MB_OK);
            return TRUE;
        }

        HCURSOR busy = LoadCursor(NULL, IDC_WAIT);
        HCURSOR normal = LoadCursor(NULL, IDC_ARROW);
        SetCursor(busy);

        // Write out the VRML header and file info
    MSTREAMPRINTF  _T("#VRML V1.0 ascii\n\n"));
    // generate the hash table of unique node names
    GenerateUniqueNodeNames(mIp->GetRootNode());

    // Write out the scene graph

    VrblOutNode(mIp->GetRootNode(), NULL, 0, FALSE, FALSE);
    delete mLodList;

    SetCursor(normal);

#if MAX_PRODUCT_VERSION_MAJOR > 14
    mStream.Close();
#else
    fclose(mStream);
#endif
    }

    return 1;
}

void
VRBLExport::initializeDefaults()
{

    SetExportType(Export_VRML_2_0);
    TCHAR text[MAX_PATH];
    GetAppData(mIp, OUTPUT_LANG_ID, _T("X3D(V)"), text, MAX_PATH);
    if (_tcscmp(text, _T("X3D(V)")) == 0)
    {
        mType = Export_X3D_V;
    }
    else if ((_tcscmp(text, _T("Vrml97_COVER")) == 0) || (_tcscmp(text, _T("VRML97")) == 0))
    {
        mType = Export_VRML_2_0_COVER;
    }
    else if (_tcscmp(text, _T("Vrml97_Standard")) == 0)
    {
        mType = Export_VRML_2_0;
    }
    else if (_tcscmp(text, _T("Inventor")) == 0)
    {
        mType = Export_VRML_1_0;
    }
    else
    {
        mType = Export_X3D_V;
    }

    GetAppData(mIp, NORMALS_ID, _T("yes"), text, MAX_PATH);
    BOOL gen = _tcscmp(text, _T("yes")) == 0;
    SetGenNormals(gen);
    GetAppData(mIp, DEFUSE_ID, _T("yes"), text, MAX_PATH);
    BOOL defu = _tcscmp(text, _T("yes")) == 0;
    SetDefUse(defu);
    GetAppData(mIp, USELOD_ID, _T("no"), text, MAX_PATH);
    BOOL uselo = _tcscmp(text, _T("yes")) == 0;
    SetUseLod(uselo);
    GetAppData(mIp, EXPORTLIGHTS_ID, _T("no"), text, MAX_PATH);
    BOOL expli = _tcscmp(text, _T("yes")) == 0;
    SetExpLights(expli);
    GetAppData(mIp, COPYTEXTURES_ID, _T("no"), text, MAX_PATH);
    BOOL copytex = _tcscmp(text, _T("yes")) == 0;
    SetCopyTextures(copytex);
    GetAppData(mIp, FORCE_WHITE_ID, _T("no"), text, MAX_PATH);
    BOOL forceWhite = _tcscmp(text, _T("yes")) == 0;
    SetForceWhite(forceWhite);
    GetAppData(mIp, INDENT_ID, _T("yes"), text, MAX_PATH);
    gen = _tcscmp(text, _T("yes")) == 0;
    SetIndent(gen);
    SetZUp(FALSE);
    GetAppData(mIp, COORD_INTERP_ID, _T("no"), text, MAX_PATH);
    gen = _tcscmp(text, _T("yes")) == 0;
    SetCoordInterp(gen);
    GetAppData(mIp, EXPORT_HIDDEN_ID, _T("no"), text, MAX_PATH);
    gen = _tcscmp(text, _T("yes")) == 0;
    SetExportHidden(gen);
    GetAppData(mIp, ENABLE_PROGRESS_BAR_ID, _T("yes"), text, MAX_PATH);
    gen = _tcscmp(text, _T("yes")) == 0;
    SetEnableProgressBar(gen);

    GetAppData(mIp, PRIMITIVES_ID, _T("no"), text, MAX_PATH);
    gen = _tcscmp(text, _T("yes")) == 0;
    SetPrimitives(gen);

    GetAppData(mIp, EXPORT_PRE_LIGHT_ID, _T("no"), text, MAX_PATH);
    gen = _tcscmp(text, _T("yes")) == 0;
    SetPreLight(gen);
    GetAppData(mIp, CPV_SOURCE_ID, _T("max"), text, MAX_PATH);
    gen = _tcscmp(text, _T("max")) == 0;
    SetCPVSource(gen);

#ifdef _LEC_
    GetAppData(mIp, FLIP_BOOK_ID, _T("no"), text, MAX_PATH);
    gen = _tcscmp(text, _T("yes")) == 0;
    SetFlipBook(gen);
#endif
    GetAppData(mIp, USE_PREFIX_ID, _T("yes"), text, MAX_PATH);
    SetUsePrefix(_tcscmp(text, _T("yes")) == 0);
    GetAppData(mIp, URL_PREFIX_ID, _T("maps"), text, MAX_PATH);
    TSTR prefix = text;
    SetUrlPrefix(prefix);
    GetAppData(mIp, DIGITS_ID, _T("4"), text, MAX_PATH);
    SetDigits(_tstoi(text));

    GetAppData(mIp, POLYGON_TYPE_ID, _T("Triangles"), text, MAX_PATH);
    if (_tcscmp(text, _T("Visible Edges")) == 0)
        SetPolygonType(OUTPUT_VISIBLE_EDGES);
    else if (_tcscmp(text, _T("Ngons")) == 0)
        SetPolygonType(OUTPUT_NGONS);
    else if (_tcscmp(text, _T("Quads")) == 0)
        SetPolygonType(OUTPUT_QUADS);
    else
        SetPolygonType(OUTPUT_TRIANGLES);

    Tab<INode *> cameras, navInfos, backgrounds, fogs, skys;
    GetCameras(GetIP()->GetRootNode(), &cameras, &navInfos,
               &backgrounds, &fogs, &skys);
    int c = cameras.Count();
    int ci;
    INode *inode = NULL;
    if (c > 0)
    {
        TSTR name;
        GetAppData(mIp, CAMERA_ID, _T(""), text, MAX_PATH);
        if (_tcslen(text) == 0)
            inode = cameras[0];
        else
        {
            name = text;
            for (ci = 0; ci < c; ci++)
                if (_tcscmp(cameras[ci]->GetName(), name) == 0)
                {
                    inode = cameras[ci];
                    break;
                }
        }
    }
    SetCamera(inode);

    c = navInfos.Count();
    inode = NULL;
    if (c > 0)
    {
        TSTR name;
        GetAppData(mIp, NAV_INFO_ID, _T(""), text, MAX_PATH);
        if (_tcslen(text) == 0)
            inode = navInfos[0];
        else
        {
            name = text;
            for (ci = 0; ci < c; ci++)
                if (_tcscmp(navInfos[ci]->GetName(), name) == 0)
                {
                    inode = navInfos[ci];
                    break;
                }
        }
    }
    SetNavInfo(inode);

    c = backgrounds.Count();
    inode = NULL;
    if (c > 0)
    {
        TSTR name;
        GetAppData(mIp, BACKGROUND_ID, _T(""), text, MAX_PATH);
        if (_tcslen(text) == 0)
            inode = backgrounds[0];
        else
        {
            name = text;
            for (ci = 0; ci < c; ci++)
                if (_tcscmp(backgrounds[ci]->GetName(), name) == 0)
                {
                    inode = backgrounds[ci];
                    break;
                }
        }
    }
    SetBackground(inode);

    c = fogs.Count();
    inode = NULL;
    if (c > 0)
    {
        TSTR name;
        GetAppData(mIp, FOG_ID, _T(""), text, MAX_PATH);
        if (_tcslen(text) == 0)
            inode = fogs[0];
        else
        {
            name = text;
            for (ci = 0; ci < c; ci++)
                if (_tcscmp(fogs[ci]->GetName(), name) == 0)
                {
                    inode = fogs[ci];
                    break;
                }
        }
    }
    SetFog(inode);

    c = skys.Count();
    inode = NULL;
    if (c > 0)
    {
        TSTR name;
        GetAppData(mIp, SKY_ID, _T(""), text, MAX_PATH);
        if (_tcslen(text) == 0)
            inode = skys[0];
        else
        {
            name = text;
            for (ci = 0; ci < c; ci++)
                if (_tcscmp(skys[ci]->GetName(), name) == 0)
                {
                    inode = skys[ci];
                    break;
                }
        }
    }
    SetSky(inode);

    GetAppData(mIp, TFORM_SAMPLE_ID, _T("custom"), text, MAX_PATH);
    BOOL once = _tcscmp(text, _T("once")) == 0;
    SetTformSample(once);
    GetAppData(mIp, TFORM_SAMPLE_RATE_ID, _T("10"), text, MAX_PATH);
    SetTformSampleRate(_tstoi(text));

    GetAppData(mIp, COORD_SAMPLE_ID, _T("custom"), text, MAX_PATH);
    once = _tcscmp(text, _T("once")) == 0;
    SetCoordSample(once);
    GetAppData(mIp, COORD_SAMPLE_RATE_ID, _T("3"), text, MAX_PATH);
    SetCoordSampleRate(_tstoi(text));

    GetAppData(mIp, FLIPBOOK_SAMPLE_ID, _T("custom"), text, MAX_PATH);
    once = _tcscmp(text, _T("once")) == 0;
    SetFlipbookSample(once);
    GetAppData(mIp, FLIPBOOK_SAMPLE_RATE_ID, _T("10"), text, MAX_PATH);
    SetFlipbookSampleRate(_tstoi(text));

    GetAppData(mIp, TITLE_ID, _T(""), text, MAX_PATH);
    SetTitle(text);
    GetAppData(mIp, INFO_ID, _T(""), text, MAX_PATH);
    SetInfo(text);
}

VRBLExport::VRBLExport()
{
    mGenNormals = FALSE;
    mDefUse = FALSE;
    mUseLod = FALSE;
    mCopyTextures = TRUE;
    mForceWhite = TRUE;
    mExpLights = FALSE;
    mHadAnim = FALSE;
    mLodList = NULL;
    mTformSample = FALSE;
    mTformSampleRate = 10;
    mCoordSample = FALSE;
    mCoordSampleRate = 3;
    mFlipbookSample = FALSE;
    mFlipbookSampleRate = 10;
}

VRBLExport::~VRBLExport()
{
}

// Number of file extensions supported by the exporter
int
VRBLExport::ExtCount()
{
    return 3;
}

// The exension supported
const TCHAR *
VRBLExport::Ext(int n)
{
    switch (n)
    {
    case 0:
        return _T("x3dv");
    case 1:
        return _T("wrl");
    case 2:
        return _T("iv");
    }
    return _T("");
}

const TCHAR *
VRBLExport::LongDesc()
{
    return _T("Autodesk VRML97/IV/X3D");
}

const TCHAR *
VRBLExport::ShortDesc()
{
    return _T("VRML97");
}

const TCHAR *
VRBLExport::AuthorName()
{
    return _T("greg finch");
}

const TCHAR *
VRBLExport::CopyrightMessage()
{
    return _T("Copyright 1997, Autodesk, Inc.");
}

const TCHAR *
VRBLExport::OtherMessage1()
{
    return _T("");
}

const TCHAR *
VRBLExport::OtherMessage2()
{
    return _T("");
}

unsigned int
VRBLExport::Version()
{
    return 100;
}

void
VRBLExport::ShowAbout(HWND hWnd)
{
}

static DWORD HashNode(DWORD o, int size)
{
    DWORD code = (DWORD)o;
    return (code >> 2) % size;
}

// form the hash value for string s
static unsigned HashName(const TCHAR *s, int size)
{
    unsigned hashVal;
    for (hashVal = 0; *s; s++)
        hashVal = *s + 31 * hashVal;
    return hashVal % size;
}

// Node Hash table lookup
NodeList *NodeTable::AddNode(INode *node)
{
    DWORD hash = HashNode((DWORD)node, NODE_HASH_TABLE_SIZE);
    NodeList *nList;

    for (nList = mTable[hash]; nList; nList = nList->next)
    {
        if (nList->node == node)
        {
            return nList;
        }
    }
    nList = new NodeList(node);
    nList->next = mTable[hash];
    mTable[hash] = nList;
    return nList;
}

// Node Name lookup
TCHAR *NodeTable::GetNodeName(INode *node)
{
    DWORD hash = HashNode((DWORD)node, NODE_HASH_TABLE_SIZE);
    NodeList *nList;

    for (nList = mTable[hash]; nList; nList = nList->next)
    {
        if (nList->node == node)
        {
            if (nList->hasName)
                return (TCHAR *)nList->name.data();
            else
                return NULL; // if it wasn't created
        }
    }
    return NULL; // if for some unknown reason we dont find it
}
bool NodeTable::findName(const TCHAR *name)
{
    unsigned hashVal = HashName(name, NODE_HASH_TABLE_SIZE);
    NameList *nList;

    for (nList = mNames[hashVal]; nList; nList = nList->next)
    {
        if (nList->name && !_tcscmp(name, nList->name))
        { // found a match
            return true;
        }
    }
    return false;
}
// Node unique name list lookup
TCHAR *NodeTable::AddName(const TCHAR *name)
{
    unsigned hashVal = HashName(name, NODE_HASH_TABLE_SIZE);
    NameList *nList;
    TCHAR buf[256];
    const TCHAR *matchStr;
    int matchVal;

    for (nList = mNames[hashVal]; nList; nList = nList->next)
    {
        if (nList->name && !_tcscmp(name, nList->name))
        { // found a match
            // checkout name for "_0xxx" that is our tag
            matchStr = _tcsrchr(name, '_');
            if (matchStr)
            { // possible additional duplicate names
                if (matchStr[1] == '0')
                { // assume additional duplicate names
                    matchVal = _tstoi(matchStr + 1); // get number
                    _tcsncpy(buf, name, _tcslen(name) - _tcslen(matchStr)); // first part
                    buf[_tcslen(name) - _tcslen(matchStr)] = '\0'; // terminate
                    //sprintf(newName.name, "%s_0%d", buf, matchVal+1);	// add one
                    int i = 0;
                    do
                    {
                        TSTR newName(buf);
                        _stprintf(buf, _T("_0%d"), matchVal + i); // add one
                        newName.Append(TSTR(buf));
                        if (!findName(newName.data()))
                        {
                            nList = new NameList(newName.data());
                            nList->next = mNames[hashVal];
                            mNames[hashVal] = nList;
                            return (TCHAR *)nList->name.data();
                        }
                    } while (1);
                }
            }
            //sprintf(newName.name, "%s_0", name);	// first duplicate name
            TSTR newName(name);
            newName.Append(_T("_0"));
            return AddName(newName.data()); // check for unique new name
        }
    }
    nList = new NameList(name);
    nList->next = mNames[hashVal];
    mNames[hashVal] = nList;
    return (TCHAR *)nList->name.data();
}

// Traverse the scene graph generating Unique Node Names
void
VRBLExport::GenerateUniqueNodeNames(INode *node)
{
    if (!node)
        return;

    NodeList *nList = mNodes.AddNode(node);
    if (!nList->hasName)
    {
        // take mangled name and get a unique name
        nList->name = mNodes.AddName(VRMLName(node->GetName()));
        nList->hasName = TRUE;
    }

    int n = node->NumberOfChildren();
    for (int i = 0; i < n; i++)
        GenerateUniqueNodeNames(node->GetChildNode(i));
}
