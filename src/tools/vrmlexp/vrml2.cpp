/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
*<
FILE: vrml2.cpp

DESCRIPTION:  VRML 2.0 .WRL file export module

CREATED BY: Scott Morrison

HISTORY: created 7 June, 1996

*>	Copyright (c) 1996, All Rights Reserved.
**********************************************************************/

#include <tchar.h>
#include <time.h>
#include <io.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <direct.h>
#include <stdlib.h>
#include <stdio.h>

#include "vrml.h"
#include "lodctrl.h"
#include "simpobj.h"
#include "istdplug.h"
#include "inline.h"
#include "COVISEObject.h"
#include "cal3dHelper.h"
#include "script.h"
#include "lod.h"
#include "inlist.h"
#include "notetrck.h"
#include "bookmark.h"
#include "stdmat.h"
#include "normtab.h"
#include "vrml_api.h"
#include "vrmlexp.h"
#include "decomp.h"
#include "timer.h"
#include "navinfo.h"
#include "backgrnd.h"
#include "fog.h"
#include "sky.h"
#include "audio.h"
#include "sound.h"
#include "touch.h"
#include "prox.h"
#include "appd.h"
#include "anchor.h"
#include "bboard.h"
#include "vrml2.h"
#include "pmesh.h"
#include "evalcol.h"
#include "string.h"
#include "arsensor.h"
#include "MultiTouchSensor.h"
#include "cover.h"
#include "switch.h"
#include "onoff.h"
#include "tabletui.h"

#include "coreexp.h"
#include "IDxMaterial.h"
#include "RTMax.h"

#include <windows.h>
#include <Winuser.h>

std::vector<ShaderEffect> shaderEffects;

enum elementTypes
{
    TUIButton,
    TUIComboBox,
    TUIEditField,
    TUIEditFloatField,
    TUIEditIntField,
    TUIFloatSlider,
    TUIFrame,
    TUILabel,
    TUIListBox,
    TUIMessageBox,
    TUISlider,
    TUISpinEditfield,
    TUISplitter,
    TUITab,
    TUITabFolder,
    TUIToggleButton
};

#define MAX_TEXTURES 6

int numShaderTextures = 0;
int shaderTextureChannel[MAX_TEXTURES];
#ifdef _DEBUG
#define FUNNY_TEST
#endif

//#define TEST_MNMESH
#ifdef TEST_MNMESH
#include "mnmath.h"
#endif

#define MIRROR_BY_VERTICES
// alternative, mirror by scale, is deprecated  --prs.

#define AEQ(a, b) (fabs(a - b) < 0.5 * pow(10.0, -mDigits))

extern TCHAR *GetString(int id);

static HWND hWndPDlg; // handle of the progress dialog
static HWND hWndPB; // handle of progress bar

static bool CoordsWritten = false;
static bool TexCoordsWritten[20];
////////////////////////////////////////////////////////////////////////
// VRML 2.0 Export implementation
////////////////////////////////////////////////////////////////////////

//#define TEMP_TEST
#ifdef TEMP_TEST
static int getworldmat = 0;
#endif

static int uniqueNumber = 0;

void AngAxisFromQa(const Quat &q, float *ang, Point3 &axis)
{
    double omega, s, x, y, z, w, l, c;
    x = (double)q.x;
    y = (double)q.y;
    z = (double)q.z;
    w = (double)q.w;
    l = sqrt(x * x + y * y + z * z + w * w);
    if (l == 0.0)
    {
        w = 1.0;
        y = z = x = 0.0;
    }
    else
    {
        c = 1.0 / l;
        x *= c;
        y *= c;
        z *= c;
        w *= c;
    }
    omega = acos(w);
    *ang = (float)(2.0 * omega);
    s = sin(omega);
    if (fabs(s) > 0.000001f)
    {
        axis[0] = (float)(x / s);
        axis[1] = (float)(y / s);
        axis[2] = (float)(z / s);
    }
    else
        axis = Point3(0, 0, 0); // RB: Added this so axis gets initialized
}

Matrix3
GetLocalTM(INode *node, TimeValue t)
{
    Matrix3 tm;
    tm = node->GetObjTMAfterWSM(t);
#ifdef TEMP_TEST
    if (getworldmat)
        return tm;
#endif
    if (!node->GetParentNode()->IsRootNode())
    {
        Matrix3 ip = node->GetParentNode()->GetObjTMAfterWSM(t);
        tm = tm * Inverse(ip);
    }
    return tm;
}

inline float
round(float f)
{
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

void
CommaScan(TCHAR *buf)
{
    for (; *buf; buf++)
        if (*buf == ',')
            *buf = '.';
}

TCHAR *
VRML2Export::point(Point3 &p)
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
VRML2Export::color(Color &c)
{
    static TCHAR buf[50];
    TCHAR format[20];
    _stprintf(format, _T("%%.%dg %%.%dg %%.%dg"), mDigits, mDigits, mDigits);
    _stprintf(buf, format, round(c.r), round(c.g), round(c.b));
    CommaScan(buf);
    return buf;
}

TCHAR *
VRML2Export::color(Point3 &c)
{
    static TCHAR buf[50];
    TCHAR format[20];
    _stprintf(format, _T("%%.%dg %%.%dg %%.%dg"), mDigits, mDigits, mDigits);
    _stprintf(buf, format, round(c.x), round(c.y), round(c.z));
    CommaScan(buf);
    return buf;
}

TCHAR *
VRML2Export::floatVal(float f)
{
    static TCHAR buf[50];
    TCHAR format[20];
    _stprintf(format, _T("%%.%dg"), mDigits);
    _stprintf(buf, format, round(f));
    CommaScan(buf);
    return buf;
}

TCHAR *
VRML2Export::texture(UVVert &uv)
{
    static TCHAR buf[50];
    TCHAR format[20];
    if (_isnan(uv.x) || _isnan(uv.y))
    {
        uv.x = 0.0;
        uv.y = 0.0;
    }
    _stprintf(format, _T("%%.%dg %%.%dg"), mDigits, mDigits);
    _stprintf(buf, format, round(uv.x), round(uv.y));
    CommaScan(buf);
    return buf;
}

// Format a scale value
TCHAR *
VRML2Export::scalePoint(Point3 &p)
{
    static TCHAR buf[50];
    TCHAR format[20];
    _stprintf(format, _T("%%.%dg %%.%dg %%.%dg"), mDigits, mDigits, mDigits);
    if (mZUp)
        _stprintf(buf, format, round(p.x), round(p.y), round(p.z));
    else
        _stprintf(buf, format, round(p.x), round(p.z), round(p.y));
    CommaScan(buf);
    return buf;
}

// Format a normal vector
TCHAR *
VRML2Export::normPoint(Point3 &p)
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
VRML2Export::axisPoint(Point3 &p, float angle)
{
    if (p == Point3(0., 0., 0.))
        p = Point3(1., 0., 0.); // default direction
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

// Indent to the given level.
void
VRML2Export::Indent(int level)
{
    if (!mIndent)
        return;
    if (level < 0)
        return;
    for (; level; level--)
      MSTREAMPRINTF  _T("  "));
}

// Translates name (if necessary) to VRML compliant name.
// Returns name in static buffer, so calling a second time trashes
// the previous contents.
#define CTL_CHARS 31
#define SINGLE_QUOTE 39
static TCHAR *VRMLName(const TCHAR *name)
{
    static TCHAR buffer[256];
    static int seqnum = 0;
    TCHAR *cPtr;
    int firstCharacter = 1;

    _tcscpy(buffer, name);
    cPtr = buffer;
    while (*cPtr)
    {
        if (*cPtr <= CTL_CHARS || *cPtr == ' ' || *cPtr == SINGLE_QUOTE || *cPtr == '"' || *cPtr == '\\' || *cPtr == '{' || *cPtr == '}' || *cPtr == ',' || *cPtr == '.' || *cPtr == '[' || *cPtr == ']' || *cPtr == '-' || *cPtr == '#' || *cPtr >= 127 || (firstCharacter && (*cPtr >= '0' && *cPtr <= '9')))
            *cPtr = '_';
        firstCharacter = 0;
        cPtr++;
    }
    if (firstCharacter)
    { // if empty name, quick, make one up!
        *cPtr++ = '_';
        *cPtr++ = '_';
        _stprintf(cPtr, _T("%d"), seqnum++);
    }

    return buffer;
}

// Write beginning of the Transform node.
void
VRML2Export::StartNode(INode *node, int level, BOOL outputName, Object *obj)
{
    TCHAR *nodnam = mNodes.GetNodeName(node);
    Class_ID id = obj->ClassID();
    Indent(level);
    if (id == Class_ID(LOD_CLASS_ID1, LOD_CLASS_ID2))
      MSTREAMPRINTF  _T("DEF %s_LoD_ Transform {\n"), nodnam);
    else
      MSTREAMPRINTF  _T("DEF %s Transform {\n"), nodnam);

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
            MSTREAMPRINTF  _T("#Info { string \"frame %d: %s\" }\n"),
               nk->time/GetTicksPerFrame(), note.data());
            }
        }
    }
}

// Write end of the Transform node.
void
VRML2Export::EndNode(INode *node, Object *obj, int level, BOOL lastChild)
{
    Indent(level);
    if (IsCamera(node) && !isSwitched(node))
      MSTREAMPRINTF  _T("\n"));
    else if (lastChild || node->GetParentNode()->IsRootNode())
      MSTREAMPRINTF  _T("}\n"));
    else
      MSTREAMPRINTF  _T("},\n"));
}

/* test
BOOL
VRML2Export::IsBBoxTrigger(INode* node)
{
Object * obj = node->EvalWorldState(mStart).obj;
if (obj->ClassID() != Class_ID(MR_BLUE_CLASS_ID1, MR_BLUE_CLASS_ID2))
return FALSE;
MrBlueObject* mbo = (MrBlueObject*) obj;
return mbo->GetBBoxEnabled();
}
*/

static BOOL
HasPivot(INode *node)
{
    Point3 p = node->GetObjOffsetPos();
    return p.x != 0.0f || p.y != 0.0f || p.z != 0.0f;
}

// Write out the transform from the parent node to this current node.
// Return true if it has a mirror transform
BOOL
VRML2Export::OutputNodeTransform(INode *node, int level, BOOL mirrored)
{
    // Root node is always identity
    if (node->IsRootNode())
        return FALSE;

    Matrix3 tm = GetLocalTM(node, mStart);
    int i, j;
    Point3 p;
    Point3 s, axis;
    Quat q;
    float ang;

    BOOL isIdentity = TRUE;
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            if (i == j)
            {
                if (tm.GetRow(i)[j] != 1.0)
                    isIdentity = FALSE;
            }
            else if (fabs(tm.GetRow(i)[j]) > 0.00001)
                isIdentity = FALSE;
        }
    }

    if (isIdentity)
    {
        p = tm.GetTrans();
#ifdef MIRROR_BY_VERTICES
        if (mirrored)
            p = -p;
#endif
        Indent(level);
      MSTREAMPRINTF  _T("translation %s\n"), point(p));
      return FALSE;
    }
    AffineParts parts;
#ifdef DDECOMP
    d_decomp_affine(tm, &parts);
#else
    decomp_affine(tm, &parts); // parts is parts
#endif
    p = parts.t;
    q = parts.q;
    AngAxisFromQa(q, &ang, axis);
#ifdef MIRROR_BY_VERTICES
    if (mirrored)
        p = -p;
#endif
    Indent(level);
   MSTREAMPRINTF  _T("translation %s\n"), point(p));
   Control *rc = node->GetTMController()->GetRotationController();

   if (ang != 0.0f && ang != -0.0f)
   {
       Indent(level);
      MSTREAMPRINTF  _T("rotation %s\n"), axisPoint(axis, -ang));
   }
   ScaleValue sv(parts.k, parts.u);
   s = sv.s;
#ifndef MIRROR_BY_VERTICES
   if (parts.f < 0.0f)
       s = -s; // this is where we mirror by scale
#endif
   if (!(AEQ(s.x, 1.0)) || !(AEQ(s.y, 1.0)) || !(AEQ(s.z, 1.0)))
   {
       Indent(level);
      MSTREAMPRINTF  _T("scale %s\n"), scalePoint(s));
      q = sv.q;
      AngAxisFromQa(q, &ang, axis);
      if (ang != 0.0f && ang != -0.0f)
      {
          Indent(level);
         MSTREAMPRINTF  _T("scaleOrientation %s\n"),
            axisPoint(axis, -ang));
      }
   }
   return parts.f < 0.0f;
}

static BOOL
MeshIsAllOneSmoothingGroup(Mesh &mesh)
{
    return FALSE; // to put out normals whenever they're called for

    int numfaces = mesh.getNumFaces();
    unsigned int sg;
    int i;

    for (i = 0; i < numfaces; i++)
    {
        if (i == 0)
        {
            sg = mesh.faces[i].getSmGroup();
            if (sg == 0)
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

#define CurrentWidth() (mIndent ? 2 * level : 0)
#define MAX_WIDTH 60

size_t
VRML2Export::MaybeNewLine(size_t width, int level)
{
    if (width > MAX_WIDTH)
    {
      MSTREAMPRINTF  _T("\n"));
      Indent(level);
      return CurrentWidth();
    }
    return width;
}

void
VRML2Export::OutputNormalIndices(Mesh &mesh, NormalTable *normTab, int level,
                                 int textureNum)
{
    Point3 n;
    int numfaces = mesh.getNumFaces();
    int i, j, v, norCnt;
    size_t width = CurrentWidth();

    Indent(level);

   MSTREAMPRINTF  _T("normalIndex [\n"));
   Indent(level + 1);
   for (i = 0; i < numfaces; i++)
   {
       int id = mesh.faces[i].getMatID();
       if (textureNum == -1 || id == textureNum)
       {
           int smGroup = mesh.faces[i].getSmGroup();
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
            width += MSTREAMPRINTF  _T("%d, "), index);
            width = MaybeNewLine(width, level + 1);
           }
         width += MSTREAMPRINTF  _T("-1, "));
         width = MaybeNewLine(width, level + 1);
       }
   }
   MSTREAMPRINTF  _T("]\n"));
}

NormalTable *
VRML2Export::OutputNormals(Mesh &mesh, int level)
{
    int i, j, norCnt;
    int numverts = mesh.getNumVerts();
    int numfaces = mesh.getNumFaces();
    NormalTable *normTab;

    //mesh.buildRenderNormals();
    mesh.buildNormals();

    if (MeshIsAllOneSmoothingGroup(mesh))
    {
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
            int cv = mesh.faces[index].v[i];
            RVertex *rv = mesh.getRVertPtr(cv);
            if (rv->rFlags & SPECIFIED_NORMAL)
            {
                normTab->AddNormal(rv->rn.getNormal());
            }
            else if ((norCnt = (int)(rv->rFlags & NORCT_MASK)) && smGroup)
            {
                if (norCnt == 1)
                    normTab->AddNormal(rv->rn.getNormal());
                else
                    for (j = 0; j < norCnt; j++)
                    {
                        normTab->AddNormal(rv->ern[j].getNormal());
                    }
            }
            else
                normTab->AddNormal(mesh.getFaceNormal(index));
        }
    }

    index = 0;
    NormalDesc *nd;
    Indent(level);
   MSTREAMPRINTF  _T("normal "));
   MSTREAMPRINTF  _T("Normal { vector [\n"));
   size_t width = CurrentWidth();
   Indent(level + 1);

   for (i = 0; i < NORM_TABLE_SIZE; i++)
   {
       for (nd = normTab->Get(i); nd; nd = nd->next)
       {
           nd->index = index++;
           Point3 p = nd->n / NUM_NORMS;
         width += MSTREAMPRINTF  _T("%s, "), normPoint(p));
         width = MaybeNewLine(width, level + 1);
       }
   }
   MSTREAMPRINTF  _T("] }\n"));

   Indent(level);
   MSTREAMPRINTF  _T("normalPerVertex TRUE\n"));

#ifdef DEBUG_NORM_HASH
   normTab->PrintStats(mStream);
#endif

   return normTab;
}

BOOL
VRML2Export::hasMaterial(TriObject *obj, int textureNum)
{
    if (textureNum == -1)
        return true;
    Mesh &mesh = obj->GetMesh();
    int numfaces = mesh.getNumFaces();
    if (numfaces == 0)
    {
        return false;
    }
    // check to see if we have faces

    for (int i = 0; i < numfaces; i++)
    {
        int id = mesh.faces[i].getMatID();
        if (id == textureNum)
        {
            if (!(mesh.faces[i].flags & FACE_HIDDEN))
            {
                return true;
            }
        }
    }
    return false;
}

void
VRML2Export::OutputPolygonObject(INode *node, TriObject *obj, BOOL isMulti,
                                 BOOL isWire, BOOL twoSided, int level,
                                 int textureNum, BOOL pMirror)
{
    assert(obj);
    size_t width;
    int i, j;
    NormalTable *normTab = NULL;
    BOOL dummy, concave;
    Mesh &mesh = obj->GetMesh();
    int numtverts[100];
    numtverts[0] = mesh.getNumTVerts();
    for (i = 1; i < 99; i++)
    {
        numtverts[i] = mesh.getNumMapVerts(i);
    }
    int numfaces = mesh.getNumFaces();

    PMPoly *poly;
    PMesh polyMesh(obj->GetMesh(), (PType)mPolygonType, numtverts);
#ifdef TEST_MNMESH
    MNMesh mnmesh(obj->mesh);
    mnmesh.MakePolyMesh();
    FILE *mnfile = fopen("mnmesh0.txt", "w");
    fprintf(mnfile, "Vertices:\n");
    for (i = 0; i < mnmesh.VNum(); i++)
        fprintf(mnfile, "  %3d)  %8.3f, %8.3f, %8.3f\n", i,
                mnmesh.P(i).x, mnmesh.P(i).y, mnmesh.P(i).z);
    fprintf(mnfile, "Faces:\n");
    for (i = 0; i < mnmesh.FNum(); i++)
    {
        fprintf(mnfile, "  ");
        MNFace *mnf = mnmesh.F(i);
        for (j = 0; j < mnf->deg; j++)
        {
            if (j > 0)
                fprintf(mnfile, ", ");
            fprintf(mnfile, "%3d", mnf->vtx[j]);
        }
        fprintf(mnfile, "\n");
    }
    fclose(mnfile);
#endif
    concave = polyMesh.GenPolygons();

    Mtl *mtl = node->GetMtl();

    if (mtl && mtl->ClassID() == Class_ID(BAKE_SHELL_CLASS_ID, 0))
    {
        mtl = mtl->GetSubMtl(1);
    }
    if (mtl && mtl->IsMultiMtl() && textureNum != -1)
    {
        if (mtl != NULL)
        {
            mtl = mtl->GetSubMtl(textureNum);
        }
    }
    if (mtl && mtl->ClassID() == Class_ID(BAKE_SHELL_CLASS_ID, 0))
    {
        mtl = mtl->GetSubMtl(1);
    }
    int numTextureDescs = 0;
    numShaderTextures = 0;
    TextureDesc *textureDescs[MAX_TEXTURES];
    GetTextures(mtl, dummy, numTextureDescs, textureDescs);

    if (!numfaces)
    {
        for (i = 0; i < numTextureDescs; i++)
            delete textureDescs[i];
        return;
    }

    Indent(level++);
    if (isWire)
    {
      MSTREAMPRINTF  _T("geometry DEF %s%s%s-FACES IndexedLineSet {\n"),
         shaderEffects[effect].getName(),shaderEffects[effect].getParamValues(),mNodes.GetNodeName(node));
    }
    else
    {
      MSTREAMPRINTF  _T("geometry DEF %s%s%s-FACES IndexedFaceSet {\n"),
         shaderEffects[effect].getName(),shaderEffects[effect].getParamValues(),mNodes.GetNodeName(node));
    }

    if (!isWire)
    {
        Indent(level);
      MSTREAMPRINTF  _T("ccw %s\n"), pMirror ? _T("FALSE") : _T("TRUE"));
      Indent(level);
      MSTREAMPRINTF  _T("solid %s\n"), twoSided ? _T("FALSE") : _T("TRUE"));
      Indent(level);
      MSTREAMPRINTF  _T("convex %s\n"), concave ? _T("FALSE") : _T("TRUE"));
    }
    bool hasColors = false;
    // color-------
    if (mPreLight)
    {
        if (!mCPVSource)
        { // 1 if MAX, 0 if we calc
            ColorTab vxColDiffTab;
            calcMixedVertexColors(node, mStart, LIGHT_SCENELIGHT, vxColDiffTab);
            int numColors = 0;
            int cfaces;
            hasColors = true;
            Color c;
            Indent(level);
         MSTREAMPRINTF  _T("colorPerVertex TRUE\n"));
         Indent(level);
         width = CurrentWidth();
         MSTREAMPRINTF  _T("color Color { color [\n"));
         Indent(level + 1);
         cfaces = vxColDiffTab.Count();
         for (i = 0; i < cfaces; i++)
         {
             c = *((Color *)vxColDiffTab[i]);
             if (i == cfaces - 1) width += MSTREAMPRINTF  _T("%s "), color(c));
             else width += MSTREAMPRINTF  _T("%s, "), color(c));
             width = MaybeNewLine(width, level + 1);
         }
         Indent(level);
         MSTREAMPRINTF  _T("] }\n"));

         Indent(level);
         MSTREAMPRINTF  _T("colorIndex [\n"));
         width = CurrentWidth();
         Indent(level + 1);
         cfaces = polyMesh.GetPolygonCnt();
         // FIXME need to add colorlist to PMesh
         for (i = 0; i < cfaces; i++)
         {
             poly = polyMesh.GetPolygon(i);
             for (j = 0; j < poly->GetVIndexCnt(); j++)
             {
               width += MSTREAMPRINTF  _T("%d, "),
                  polyMesh.LookUpVert(poly->GetVIndex(j)));
               width = MaybeNewLine(width, level + 1);
             }
            width += MSTREAMPRINTF  _T("-1"));
            if (i != polyMesh.GetPolygonCnt() - 1)
            {
               width += MSTREAMPRINTF  _T(", "));
               width = MaybeNewLine(width, level + 1);
            }
         }
         MSTREAMPRINTF  _T("]\n"));

         for (i = 0; i < vxColDiffTab.Count(); i++)
         {
             delete (Color *)vxColDiffTab[i];
         }
         vxColDiffTab.ZeroCount();
         vxColDiffTab.Shrink();
        }
        else
        {
            int numCVerts = mesh.getNumVertCol();
            if (numCVerts)
            {
                hasColors = true;
                VertColor vColor;
                Indent(level);
            MSTREAMPRINTF  _T("colorPerVertex TRUE\n"));
            Indent(level);
            width = CurrentWidth();
            MSTREAMPRINTF  _T("color Color { color [\n"));
            Indent(level + 1);

            int nVerts = polyMesh.GetVertexCnt();
            for (i = 0; i < nVerts; i++)
            {
                /*
               for (j = 0; j < poly->GetVIndexCnt(); j++) {
               width += MSTREAMPRINTF  _T("%d, "),
               polyMesh.LookUpVert(poly->GetVIndex(j)));
               width  = MaybeNewLine(width, level+1);
               }
               */
                int vIndex = polyMesh.LookUpVert(i);

                if (vIndex > numCVerts)
                {
                    assert(FALSE);
                    break;
                }

                vColor = mesh.vertCol[vIndex];
                if (i == nVerts - 1)
                  width += MSTREAMPRINTF  _T("%s "), color(vColor));
                else
                  width += MSTREAMPRINTF  _T("%s, "), color(vColor));
                width = MaybeNewLine(width, level + 1);
            }
            Indent(level);
            MSTREAMPRINTF  _T("] }\n"));

            Indent(level);
            MSTREAMPRINTF  _T("colorIndex [\n"));
            width = CurrentWidth();
            Indent(level + 1);
            int cfaces = polyMesh.GetPolygonCnt();
            // FIXME need to add colorlist to PMesh
            for (i = 0; i < cfaces; i++)
            {
                poly = polyMesh.GetPolygon(i);
                for (j = 0; j < poly->GetVIndexCnt(); j++)
                {
                  width += MSTREAMPRINTF  _T("%d, "),
                     polyMesh.LookUpVert(poly->GetVIndex(j)));
                  width = MaybeNewLine(width, level + 1);
                }
               width += MSTREAMPRINTF  _T("-1"));
               if (i != polyMesh.GetPolygonCnt() - 1)
               {
                  width += MSTREAMPRINTF  _T(", "));
                  width = MaybeNewLine(width, level + 1);
               }
            }
            MSTREAMPRINTF  _T("]\n"));
            }
            else
            {
                // why the hell should we turn colorPerVertex off, others might have cpv mPreLight = FALSE;
            }
        }
    }

    int numColors = 0;
    if (!hasColors && isMulti && textureNum == -1)
    {
        Color c;
        Indent(level);
      MSTREAMPRINTF  _T("colorPerVertex FALSE\n"));
      Mtl *sub, *mtl = node->GetMtl();
      if (mtl && mtl->ClassID() == Class_ID(BAKE_SHELL_CLASS_ID, 0))
      {
          mtl = mtl->GetSubMtl(1);
      }
      assert(mtl->IsMultiMtl());
      int num = mtl->NumSubMtls();
      Indent(level);
      width = CurrentWidth();

      MSTREAMPRINTF  _T("color Color { color [\n"));
      Indent(level + 1);
      for (i = 0; i < num; i++)
      {
          sub = mtl->GetSubMtl(i);
          if (sub && sub->ClassID() == Class_ID(BAKE_SHELL_CLASS_ID, 0))
          {
              sub = sub->GetSubMtl(1);
          }
          if (!sub)
              continue;
          numColors++;
          c = sub->GetDiffuse(mStart);
          if (i == num - 1) width += MSTREAMPRINTF  _T("%s "), color(c));
          else width += MSTREAMPRINTF  _T("%s, "), color(c));
          width = MaybeNewLine(width, level + 1);
      }
      Indent(level);
      MSTREAMPRINTF  _T("] }\n"));
    }
    if (!hasColors && isMulti && numColors > 0 && textureNum == -1)
    {
        Indent(level);
      MSTREAMPRINTF  _T("colorIndex [\n"));
      width = CurrentWidth();
      Indent(level + 1);
      numfaces = polyMesh.GetPolygonCnt();
      // FIXME need to add colorlist to PMesh
      for (i = 0; i < numfaces; i++)
      {
          poly = polyMesh.GetPolygon(i);
          int matID = mesh.faces[poly->GetTriFace(0)].getMatID();
          matID = (matID % numColors);
         width += MSTREAMPRINTF  _T("%d"), matID);
         if (i != numfaces - 1)
         {
            width += MSTREAMPRINTF  _T(", "));
            width = MaybeNewLine(width, level + 1);
         }
      }
      MSTREAMPRINTF  _T("]\n"));
    }

    // output coordinate---------
    if (textureNum < 1)
    {
        Indent(level);
      MSTREAMPRINTF  _T("coord DEF %s-COORD Coordinate { point [\n"),
         mNodes.GetNodeName(node));
      width = CurrentWidth();
      Indent(level + 1);
      int numV = polyMesh.GetVertexCnt();
      for (i = 0; i < numV; i++)
      {
          Point3 p = polyMesh.GetVertex(i);
#ifdef MIRROR_BY_VERTICES
          if (pMirror)
              p = -p;
#endif
         width += MSTREAMPRINTF  _T("%s"), point(p));
         if (i < (numV - 1))
         {
            width += MSTREAMPRINTF  _T(", "));
            width = MaybeNewLine(width, level + 1);
         }
      }
          MSTREAMPRINTF  _T("]\n"));
          Indent(level);
          MSTREAMPRINTF  _T("}\n"));
    }
    else
    {
        Indent(level);
      MSTREAMPRINTF  _T("coord USE %s-COORD\n"),
         mNodes.GetNodeName(node));
    }
    Indent(level);
   MSTREAMPRINTF  _T("coordIndex [\n"));
   Indent(level + 1);
   width = CurrentWidth();
   for (i = 0; i < polyMesh.GetPolygonCnt(); i++)
   {
       poly = polyMesh.GetPolygon(i);
       for (j = 0; j < poly->GetVIndexCnt(); j++)
       {
         width += MSTREAMPRINTF  _T("%d, "),
            polyMesh.LookUpVert(poly->GetVIndex(j)));
         width = MaybeNewLine(width, level + 1);
       }
      width += MSTREAMPRINTF  _T("-1"));
      if (i != polyMesh.GetPolygonCnt() - 1)
      {
         width += MSTREAMPRINTF  _T(", "));
         width = MaybeNewLine(width, level + 1);
      }
   }
   MSTREAMPRINTF  _T("]\n"));

   // Output Texture coordinates
   if (numtverts[0] > 0 && (numTextureDescs || textureNum == 0 || numShaderTextures) && !isWire)
   {
       if (textureNum < 1)
       {
           Indent(level);
         MSTREAMPRINTF  _T("texCoord DEF %s-TEXCOORD TextureCoordinate { point [\n"),
            mNodes.GetNodeName(node));
         width = CurrentWidth();
         Indent(level + 1);
         for (i = 0; i < polyMesh.GetTVertexCnt(); i++)
         {
             UVVert t = polyMesh.GetTVertex(i);
            width += MSTREAMPRINTF  _T("%s"), texture(t));
            if (i == polyMesh.GetTVertexCnt() - 1)
            {
               MSTREAMPRINTF  _T("]\n"));
               Indent(level);
               MSTREAMPRINTF  _T("}\n"));
            }
            else
            {
               width += MSTREAMPRINTF  _T(", "));
               width = MaybeNewLine(width, level + 1);
            }
         }
       }
       else
       {
           Indent(level);
         MSTREAMPRINTF  _T("texCoord USE %s-TEXCOORD\n"),
            mNodes.GetNodeName(node));
       }
   }
   if (numtverts[0] > 0 && (numTextureDescs || numShaderTextures) && !isWire)
   {
       Indent(level);
      MSTREAMPRINTF  _T("texCoordIndex [\n"));
      Indent(level + 1);
      width = CurrentWidth();
      int tmp = polyMesh.GetPolygonCnt();
      //for (i = 0; i < polyMesh.GetPolygonCnt(); i++) {
      for (i = 0; i < tmp; i++)
      {
          poly = polyMesh.GetPolygon(i);
          int tmp1 = poly->GetTVIndexCnt();
          //for (j = 0; j < poly->GetTVIndexCnt(); j++) {
          for (j = 0; j < tmp1; j++)
          {
              int tmp2 = poly->GetTVIndex(j);
              int tmp3 = polyMesh.LookUpTVert(tmp2);
            width += MSTREAMPRINTF  _T("%d, "),
               //polyMesh.LookUpTVert(poly->GetTVIndex(j)));
               tmp3);
            width = MaybeNewLine(width, level + 1);
          }
         width += MSTREAMPRINTF  _T("-1"));
         if (i != polyMesh.GetPolygonCnt() - 1)
         {
            width += MSTREAMPRINTF  _T(", "));
            width = MaybeNewLine(width, level + 1);
         }
      }
      MSTREAMPRINTF  _T("]\n"));
   }

   // output normals
   if (mGenNormals && !isWire && !MeshIsAllOneSmoothingGroup(mesh))
   {
       NormalTable *normTab = NULL;
       normTab = new NormalTable();

       for (j = 0; j < polyMesh.GetPolygonCnt(); j++)
       {
           //Point3 n = polyMesh.GetPolygon(j)->GetFNormal();
           for (int k = 0; k < polyMesh.GetPolygon(j)->GetVNormalCnt(); k++)
           {
               Point3 n = polyMesh.GetPolygon(j)->GetVNormal(k);
               normTab->AddNormal(n);
           }
       }

       Indent(level);
      MSTREAMPRINTF  _T("normalPerVertex TRUE\n"));
      int index = 0;
      NormalDesc *nd;
      Indent(level);
      MSTREAMPRINTF  _T("normal "));
      MSTREAMPRINTF  _T("Normal { vector [\n"));
      width = CurrentWidth();
      Indent(level + 1);
      /*
      for (i = 0; i < polyMesh.GetPolygonCnt(); i++) {
      Point3 n = polyMesh.GetPolygon(i)->GetFNormal();
      normTab->AddNormal(n);
      width += MSTREAMPRINTF  _T("%s, "), normPoint(n));
      width  = MaybeNewLine(width, level+1);
      }
      */

      for (i = 0; i < NORM_TABLE_SIZE; i++)
      {
          for (nd = normTab->Get(i); nd; nd = nd->next)
          {
              nd->index = index++;
              Point3 n = nd->n / NUM_NORMS;
            width    += MSTREAMPRINTF  _T("%s, "), normPoint(n));
            width = MaybeNewLine(width, level + 1);
          }
      }
      MSTREAMPRINTF  _T("] }\n"));

      Indent(level);
      width = CurrentWidth();
      MSTREAMPRINTF  _T("normalIndex [\n"));
      Indent(level + 1);
      width = CurrentWidth();

      for (i = 0; i < polyMesh.GetPolygonCnt(); i++)
      {
          int index;
          for (int k = 0; k < polyMesh.GetPolygon(i)->GetVNormalCnt(); k++)
          {
              Point3 n = polyMesh.GetPolygon(i)->GetVNormal(k);
              index = normTab->GetIndex(n);
            width   += MSTREAMPRINTF  _T("%d, "), index);
            width = MaybeNewLine(width, level + 1);
          }
         width += MSTREAMPRINTF  _T("-1, "));
         width = MaybeNewLine(width, level + 1);
      }
      normTab->PrintStats(mStream);

      MSTREAMPRINTF  _T("]\n"));

      delete normTab;
   }

   Indent(level);
   MSTREAMPRINTF  _T("}\n"));
   for (i = 0; i < numTextureDescs; i++)
       delete textureDescs[i];
}

// Write out the data for a single triangle mesh
void
VRML2Export::OutputTriObject(INode *node, TriObject *obj, BOOL isMulti,
                             BOOL isWire, BOOL twoSided, int level,
                             int textureNum, BOOL pMirror)
{
    assert(obj);
    Mesh &mesh = obj->GetMesh();
    int numverts = mesh.getNumVerts();
    int numfaces = mesh.getNumFaces();
    int i;
    size_t width;
    int numtverts[100];
    numtverts[0] = mesh.getNumTVerts();
    for (i = 1; i < 99; i++)
    {
        numtverts[i] = mesh.getNumMapVerts(i);
    }
    NormalTable *normTab = NULL;
    BOOL dummy;

    Mtl *mtl = node->GetMtl();
    if (mtl && mtl->ClassID() == Class_ID(BAKE_SHELL_CLASS_ID, 0))
    {
        mtl = mtl->GetSubMtl(1);
    }
    if (mtl && mtl->IsMultiMtl() && textureNum != -1)
    {
        if (mtl != NULL)
        {
            mtl = mtl->GetSubMtl(textureNum);
        }
    }
    if (mtl && mtl->ClassID() == Class_ID(BAKE_SHELL_CLASS_ID, 0))
    {
        mtl = mtl->GetSubMtl(1);
    }
    int numTextureDescs = 0;
    TextureDesc *textureDescs[MAX_TEXTURES];
    GetTextures(mtl, dummy, numTextureDescs, textureDescs);

    if (numfaces == 0)
    {
        for (i = 0; i < numTextureDescs; i++)
            delete textureDescs[i];
        return;
    }
    // check to see if we have faces
    int haveFaces = 0;

    for (i = 0; i < numfaces; i++)
    {
        int id = mesh.faces[i].getMatID();
        if (textureNum == -1 || id == textureNum)
        {
            if (!(mesh.faces[i].flags & FACE_HIDDEN))
            {
                haveFaces++;
                break;
            }
        }
    }

    if (haveFaces == 0)
    {
        for (i = 0; i < numTextureDescs; i++)
            delete textureDescs[i];
        return;
    }

    Indent(level++);
    if (isWire)
      MSTREAMPRINTF  _T("geometry DEF %s%s%s-FACES IndexedLineSet {\n"),shaderEffects[effect].getName(),shaderEffects[effect].getParamValues(), mNodes.GetNodeName(node));
    else
      MSTREAMPRINTF  _T("geometry DEF %s%s%s-FACES IndexedFaceSet {\n"),shaderEffects[effect].getName(),shaderEffects[effect].getParamValues(), mNodes.GetNodeName(node));

    if (!isWire)
    {
        Indent(level);
      MSTREAMPRINTF  _T("ccw %s\n"), pMirror ? _T("FALSE") : _T("TRUE"));
      Indent(level);
      MSTREAMPRINTF  _T("solid %s\n"),
         twoSided ? _T("FALSE") : _T("TRUE"));
    }
    bool hasColors = false;
    if (mPreLight)
    {
        if (!mCPVSource)
        { // 1 if MAX, 0 if we calc
            hasColors = true;
            ColorTab vxColDiffTab;
            calcMixedVertexColors(node, mStart, LIGHT_SCENELIGHT, vxColDiffTab);
            int numColors = 0;
            int cfaces;
            Color c;
            Indent(level);
         MSTREAMPRINTF  _T("colorPerVertex TRUE\n"));
         Indent(level);
         width = CurrentWidth();
         MSTREAMPRINTF  _T("color Color { color [\n"));
         Indent(level + 1);
         cfaces = vxColDiffTab.Count();
         for (i = 0; i < cfaces; i++)
         {
             c = *((Color *)vxColDiffTab[i]);
             if (i == cfaces - 1) width += MSTREAMPRINTF  _T("%s "), color(c));
             else width += MSTREAMPRINTF  _T("%s, "), color(c));
             width = MaybeNewLine(width, level + 1);
         }
         Indent(level);
         MSTREAMPRINTF  _T("] }\n"));

         Indent(level);
         MSTREAMPRINTF  _T("colorIndex [\n"));
         width = CurrentWidth();
         Indent(level + 1);

         for (i = 0; i < numfaces; i++)
         {
            width += MSTREAMPRINTF  _T("%d, %d, %d, -1"),
               mesh.faces[i].v[0], mesh.faces[i].v[1],
               mesh.faces[i].v[2]);
            if (i != numfaces - 1)
            {
               width += MSTREAMPRINTF  _T(", "));
               width = MaybeNewLine(width, level + 1);
            }
         }
         MSTREAMPRINTF  _T("]\n"));

         for (i = 0; i < vxColDiffTab.Count(); i++)
         {
             delete (Color *)vxColDiffTab[i];
         }
         vxColDiffTab.ZeroCount();
         vxColDiffTab.Shrink();
        }
        else
        {
            int numCVerts = mesh.getNumVertCol();
            if (numCVerts)
            {
                hasColors = true;
                VertColor vColor;
                Indent(level);
            MSTREAMPRINTF  _T("colorPerVertex TRUE\n"));
            Indent(level);
            width = CurrentWidth();
            MSTREAMPRINTF  _T("color Color { color [\n"));
            Indent(level + 1);

            /* old
            // FIXME need to add colorlist to PMesh
            for (i = 0; i < numverts; i++) {
            vColor = mesh.vertCol[i];
            if (i == numverts - 1)
            width += MSTREAMPRINTF  _T("%s "), color(vColor));
            else
            width += MSTREAMPRINTF  _T("%s, "), color(vColor));
            width = MaybeNewLine(width, level+1);
            }
            Indent(level);
            MSTREAMPRINTF  _T("] }\n"));

            Indent(level);
            MSTREAMPRINTF  _T("colorIndex [\n"));
            width = CurrentWidth();
            Indent(level+1);

            for (i = 0; i < numfaces; i++) {
            int id = mesh.faces[i].getMatID();
            if (textureNum == -1 || id == textureNum) {
            if (!(mesh.faces[i].flags & FACE_HIDDEN)) {
            width += MSTREAMPRINTF  _T("%d, %d, %d, -1"),
            mesh.faces[i].v[0], mesh.faces[i].v[1],
            mesh.faces[i].v[2]);
            if (i != numfaces-1) {
            width += MSTREAMPRINTF  _T(", "));
            width = MaybeNewLine(width, level+1);
            }
            }
            }
            }
            MSTREAMPRINTF  _T("]\n"));
            */
            // FIXME need to add colorlist to PMesh
            for (i = 0; i < numCVerts; i++)
            {
                vColor = mesh.vertCol[i];
                if (i == numCVerts - 1)
                  width += MSTREAMPRINTF  _T("%s "), color(vColor));
                else
                  width += MSTREAMPRINTF  _T("%s, "), color(vColor));
                width = MaybeNewLine(width, level + 1);
            }
            Indent(level);
            MSTREAMPRINTF  _T("] }\n"));

            Indent(level);
            MSTREAMPRINTF  _T("colorIndex [\n"));
            width = CurrentWidth();
            Indent(level + 1);

            for (i = 0; i < numfaces; i++)
            {
                int id = mesh.faces[i].getMatID();
                if (textureNum == -1 || id == textureNum)
                {
                    if (!(mesh.faces[i].flags & FACE_HIDDEN))
                    {
                     width += MSTREAMPRINTF  _T("%d, %d, %d, -1"),
                        mesh.vcFace[i].t[0], mesh.vcFace[i].t[1],
                        mesh.vcFace[i].t[2]);
                     if (i != numfaces - 1)
                     {
                        width += MSTREAMPRINTF  _T(", "));
                        width = MaybeNewLine(width, level + 1);
                     }
                    }
                }
            }
            MSTREAMPRINTF  _T("]\n"));
            }
            else
            {
                // there might be other objects with color per vertex... mPreLight = FALSE;
            }
        }
    }

    int numColors = 0;
    if (!hasColors && isMulti && textureNum == -1)
    {
        Color c;
        Indent(level);
      MSTREAMPRINTF  _T("colorPerVertex FALSE\n"));
      Mtl *sub, *mtl = node->GetMtl();
      if (mtl && mtl->ClassID() == Class_ID(BAKE_SHELL_CLASS_ID, 0))
      {
          mtl = mtl->GetSubMtl(1);
      }
      assert(mtl->IsMultiMtl());
      int num = mtl->NumSubMtls();
      Indent(level);
      width = CurrentWidth();
      MSTREAMPRINTF  _T("color Color { color [\n"));
      Indent(level + 1);
      for (i = 0; i < num; i++)
      {
          sub = mtl->GetSubMtl(i);
          if (sub && sub->ClassID() == Class_ID(BAKE_SHELL_CLASS_ID, 0))
          {
              sub = sub->GetSubMtl(1);
          }
          if (!sub)
              continue;
          numColors++;
          c = sub->GetDiffuse(mStart);
          if (i == num - 1)
            width += MSTREAMPRINTF  _T("%s "), color(c));
          else
            width += MSTREAMPRINTF  _T("%s, "), color(c));
          width = MaybeNewLine(width, level + 1);
      }
      Indent(level);
      MSTREAMPRINTF  _T("] }\n"));
    }

    if (!CoordsWritten)
    {
        CoordsWritten = true;
        // Output the vertices
        Indent(level);
      MSTREAMPRINTF  _T("coord DEF %s-COORD Coordinate { point [\n"),mNodes.GetNodeName(node));

      width = CurrentWidth();
      Indent(level + 1);
      for (i = 0; i < numverts; i++)
      {
          Point3 p = mesh.verts[i];
#ifdef MIRROR_BY_VERTICES
          if (pMirror)
              p = -p;
#endif
         width += MSTREAMPRINTF  _T("%s"), point(p));

         if (i < (numverts - 1))
         {
            width += MSTREAMPRINTF  _T(", "));
            width = MaybeNewLine(width, level + 1);
         }
      }

          MSTREAMPRINTF  _T("]\n"));
          Indent(level);
          MSTREAMPRINTF  _T("}\n"));
    }
    else
    {
        Indent(level);
      MSTREAMPRINTF  _T("coord USE %s-COORD\n"),mNodes.GetNodeName(node));
    }
    // Output the normals
    // FIXME share normals on multi-texture objects
    if (mGenNormals && !isWire)
    {
        normTab = OutputNormals(mesh, level);
    }

    int texNum = 0;
    // Output Texture coordinates

    int texStageNumber = 0;
    for (texNum = 0; texNum < numTextureDescs; texNum++)
    {
        // Output Texture coordinates
        if ((textureDescs[texNum]) && (textureDescs[texNum]->mapChannel >= 0) && numtverts[textureDescs[texNum]->mapChannel] > 0 && !isWire)
        {
            if (!TexCoordsWritten[textureDescs[texNum]->mapChannel])
            {
                TexCoordsWritten[textureDescs[texNum]->mapChannel] = true;
                Indent(level);
                if (texStageNumber == 0)
               MSTREAMPRINTF 
               _T("texCoord DEF %s-TEXCOORD%d TextureCoordinate { point [\n"),mNodes.GetNodeName(node),textureDescs[texNum]->mapChannel);
                else
               MSTREAMPRINTF 
               _T("texCoord%d DEF %s-TEXCOORD%d TextureCoordinate { point [\n"),texStageNumber+1,mNodes.GetNodeName(node),textureDescs[texNum]->mapChannel);
                width = CurrentWidth();
                Indent(level + 1);
                for (i = 0; i < numtverts[textureDescs[texNum]->mapChannel]; i++)
                {
                    UVVert p = mesh.mapVerts(textureDescs[texNum]->mapChannel)[i];
               width += MSTREAMPRINTF  _T("%s"), texture(p));
               if (i == numtverts[textureDescs[texNum]->mapChannel] - 1)
               {
                  MSTREAMPRINTF  _T("]\n"));
                  Indent(level);
                  MSTREAMPRINTF  _T("}\n"));
               }
               else
               {
                  width += MSTREAMPRINTF  _T(", "));
                  width = MaybeNewLine(width, level + 1);
               }
                }
            }
            else
            {
                Indent(level);
                if (texStageNumber == 0)
               MSTREAMPRINTF  _T("texCoord USE %s-TEXCOORD%d\n"),mNodes.GetNodeName(node),textureDescs[texNum]->mapChannel);
                else
               MSTREAMPRINTF  _T("texCoord%d USE %s-TEXCOORD%d\n"),texStageNumber+1,mNodes.GetNodeName(node),textureDescs[texNum]->mapChannel);
            }
            texStageNumber++;
        }
    }
    texStageNumber = 0;
    for (texNum = 0; texNum < numShaderTextures; texNum++)
    {
        // Output Texture coordinates for DX Materials
        if ((shaderTextureChannel[texNum] >= 0) && numtverts[shaderTextureChannel[texNum]] > 0 && !isWire)
        {
            if (!TexCoordsWritten[shaderTextureChannel[texNum]])
            {
                TexCoordsWritten[shaderTextureChannel[texNum]] = true;
                Indent(level);
                if (texStageNumber == 0)
               MSTREAMPRINTF 
               _T("texCoord DEF %s-TEXCOORD%d TextureCoordinate { point [\n"),mNodes.GetNodeName(node),shaderTextureChannel[texNum]);
                else
               MSTREAMPRINTF 
               _T("texCoord%d DEF %s-TEXCOORD%d TextureCoordinate { point [\n"),texStageNumber+1,mNodes.GetNodeName(node),shaderTextureChannel[texNum]);
                width = CurrentWidth();
                Indent(level + 1);
                for (i = 0; i < numtverts[shaderTextureChannel[texNum]]; i++)
                {
                    UVVert p = mesh.mapVerts(shaderTextureChannel[texNum])[i];
               width += MSTREAMPRINTF  _T("%s"), texture(p));
               if (i == numtverts[shaderTextureChannel[texNum]] - 1)
               {
                  MSTREAMPRINTF  _T("]\n"));
                  Indent(level);
                  MSTREAMPRINTF  _T("}\n"));
               }
               else
               {
                  width += MSTREAMPRINTF  _T(", "));
                  width = MaybeNewLine(width, level + 1);
               }
                }
            }
            else
            {
                Indent(level);
                if (texStageNumber == 0)
               MSTREAMPRINTF  _T("texCoord USE %s-TEXCOORD%d\n"),mNodes.GetNodeName(node),shaderTextureChannel[texNum]);
                else
               MSTREAMPRINTF  _T("texCoord%d USE %s-TEXCOORD%d\n"),texStageNumber+1,mNodes.GetNodeName(node),shaderTextureChannel[texNum]);
            }
            texStageNumber++;
        }
    }
    // Output the triangles
    Indent(level);
   MSTREAMPRINTF  _T("coordIndex [\n"));
   Indent(level + 1);
   width = CurrentWidth();
   for (i = 0; i < numfaces; i++)
   {
       int id = mesh.faces[i].getMatID();
       if (textureNum == -1 || id == textureNum)
       {
           if (!(mesh.faces[i].flags & FACE_HIDDEN))
           {
            width += MSTREAMPRINTF  _T("%d, %d, %d, -1"),
               mesh.faces[i].v[0], mesh.faces[i].v[1],
               mesh.faces[i].v[2]);
            if (i != numfaces - 1)
            {
               width += MSTREAMPRINTF  _T(", "));
               width = MaybeNewLine(width, level + 1);
            }
           }
       }
   }
   MSTREAMPRINTF  _T("]\n"));

   texStageNumber = 0;
   for (texNum = 0; texNum < numTextureDescs; texNum++)
   {
       if (textureDescs[texNum] && (textureDescs[texNum]->mapChannel >= 0) && numtverts[textureDescs[texNum]->mapChannel] > 0 && !isWire)
       {
           Indent(level);
           if (texStageNumber == 0)
            MSTREAMPRINTF  _T("texCoordIndex [\n"));
           else
            MSTREAMPRINTF  _T("texCoordIndex%d [\n"),texStageNumber+1);
           Indent(level + 1);
           width = CurrentWidth();
           for (i = 0; i < numfaces; i++)
           {
               int id = mesh.faces[i].getMatID();
               if (textureNum == -1 || id == textureNum)
               {
                   if (!(mesh.faces[i].flags & FACE_HIDDEN))
                   {
                  width += MSTREAMPRINTF  _T("%d, %d, %d, -1"),
                     mesh.mapFaces(textureDescs[texNum]->mapChannel)[i].t[0], mesh.mapFaces(textureDescs[texNum]->mapChannel)[i].t[1],
                     mesh.mapFaces(textureDescs[texNum]->mapChannel)[i].t[2]);
                  if (i != numfaces - 1)
                  {
                     width += MSTREAMPRINTF  _T(", "));
                     width = MaybeNewLine(width, level + 1);
                  }
                   }
               }
           }
         MSTREAMPRINTF  _T("]\n"));

         texStageNumber++;
       }
   }
   for (texNum = 0; texNum < numShaderTextures; texNum++)
   {
       if (textureDescs[texNum] && (shaderTextureChannel[texNum] >= 0) && numtverts[shaderTextureChannel[texNum]] > 0 && !isWire)
       {
           Indent(level);
           if (texStageNumber == 0)
            MSTREAMPRINTF  _T("texCoordIndex [\n"));
           else
            MSTREAMPRINTF  _T("texCoordIndex%d [\n"),texStageNumber+1);
           Indent(level + 1);
           width = CurrentWidth();
           for (i = 0; i < numfaces; i++)
           {
               int id = mesh.faces[i].getMatID();
               if (textureNum == -1 || id == textureNum)
               {
                   if (!(mesh.faces[i].flags & FACE_HIDDEN))
                   {
                  width += MSTREAMPRINTF  _T("%d, %d, %d, -1"),
                     mesh.mapFaces(shaderTextureChannel[texNum])[i].t[0], mesh.mapFaces(shaderTextureChannel[texNum])[i].t[1],
                     mesh.mapFaces(shaderTextureChannel[texNum])[i].t[2]);
                  if (i != numfaces - 1)
                  {
                     width += MSTREAMPRINTF  _T(", "));
                     width = MaybeNewLine(width, level + 1);
                  }
                   }
               }
           }
         MSTREAMPRINTF  _T("]\n"));

         texStageNumber++;
       }
   }

   if (!hasColors && isMulti && numColors > 0 && textureNum == -1)
   {
       Indent(level);
      MSTREAMPRINTF  _T("colorIndex [\n"));
      width = CurrentWidth();
      Indent(level + 1);
      for (i = 0; i < numfaces; i++)
      {
          if (!(mesh.faces[i].flags & FACE_HIDDEN))
          {
              int id = mesh.faces[i].getMatID();
              id = (id % numColors); // this is the way MAX does it
              /*
            if (id >= numColors)
            id = 0;
            */
            width += MSTREAMPRINTF  _T("%d"), id);
            if (i != numfaces - 1)
            {
               width += MSTREAMPRINTF  _T(", "));
               width = MaybeNewLine(width, level + 1);
            }
          }
      }
      MSTREAMPRINTF  _T("]\n"));
   }
   if (mGenNormals && normTab && !isWire)
   {
       OutputNormalIndices(mesh, normTab, level, textureNum);
       delete normTab;
   }

   Indent(level);
   MSTREAMPRINTF  _T("}\n"));
   for (i = 0; i < numTextureDescs; i++)
       delete textureDescs[i];
}

// Write out the data for a single PolyLine
void
VRML2Export::OutputPolyShapeObject(INode *node, PolyShape &shape, int level)
{

    Indent(level);
   MSTREAMPRINTF  _T("Shape {\n"));
   Indent(level++);
   if (mExportOccluders)
   {
       MSTREAMPRINTF  _T("geometry DEF coOccluder%s-FACES IndexedLineSet {\n"), mNodes.GetNodeName(node));
   }
   else
   {
       MSTREAMPRINTF  _T("geometry DEF %s-FACES IndexedLineSet {\n"), mNodes.GetNodeName(node));
   }

   // Output the vertices
   Indent(level);
   MSTREAMPRINTF  _T("coord DEF %s-COORD Coordinate { point [\n"),mNodes.GetNodeName(node));

   size_t width = CurrentWidth();
   Indent(level + 1);

   for (int poly = 0; poly < shape.numLines; poly++)
   {
       PolyLine &line = shape.lines[poly];

       for (int i = 0; i < line.numPts; i++)
       {

           PolyPt &pp = line.pts[i];
           Point3 p = pp.p;
         width += MSTREAMPRINTF  _T("%s"), point(p));

         if (i < line.numPts - 1)
         {
            width += MSTREAMPRINTF  _T(", "));
            width = MaybeNewLine(width, level + 1);
         }
       }
       if (poly < shape.numLines - 1)
       {
            width += MSTREAMPRINTF  _T(", "));
            width = MaybeNewLine(width, level + 1);
       }
   }
   MSTREAMPRINTF  _T("]\n"));
   Indent(level);
   MSTREAMPRINTF  _T("}\n"));

   // Output the lines
   Indent(level);
   MSTREAMPRINTF  _T("coordIndex [\n"));
   Indent(level + 1);
   width = CurrentWidth();
   int coordNum = 0;
   for (int poly = 0; poly < shape.numLines; poly++)
   {
       PolyLine &line = shape.lines[poly];
       int startVert = coordNum;
       for (int i = 0; i < line.numPts; i++)
       {

         width += MSTREAMPRINTF  _T("%d, "),
            coordNum);
         width = MaybeNewLine(width, level + 1);

         coordNum++;
       }
       if (line.IsClosed())
       {
         width += MSTREAMPRINTF  _T("%d, "),
            startVert);
       }
      width += MSTREAMPRINTF  _T("-1"));
      if (poly != shape.numLines - 1)
      {
         width += MSTREAMPRINTF  _T(", "));
         width = MaybeNewLine(width, level + 1);
      }
   }
   MSTREAMPRINTF  _T("]\n"));
   Indent(level);
   MSTREAMPRINTF  _T("}\n"));
   Indent(level);
   MSTREAMPRINTF  _T("}\n"));
}

BOOL
VRML2Export::HasTexture(INode *node, BOOL &isWire)
{
    TextureDesc *td = GetMatTex(node, isWire);
    if (!td)
        return FALSE;
    delete td;
    return TRUE;
}

TSTR
VRML2Export::PrefixUrl(TSTR &fileName)
{
    if (mUsePrefix && mUrlPrefix.Length() > 0)
    {
        if (mUrlPrefix[mUrlPrefix.Length() - 1] != '/')
        {
            TSTR slash = _T("/");
            return mUrlPrefix + slash + fileName;
        }
        else
            return mUrlPrefix + fileName;
    }
    else
        return fileName;
}

// Get the name of the texture file
TextureDesc *
VRML2Export::GetMatTex(INode *node, BOOL &isWire, int textureNumber, int askForSubTexture)
{
    Mtl *mtl = node->GetMtl();
    return GetMtlTex(mtl, isWire, textureNumber, askForSubTexture);
}

// get the first texture
void
VRML2Export::GetTextures(Mtl *mtl, BOOL &isWire, int &numTexDesks, TextureDesc **tds)
{
    int id;
    int askForSubTexture = 0;
    bool stdMap = false;
    int minMapVal = ID_DI;
    if (mtl == NULL)
        return;
    if (mtl->ClassID() == Class_ID(DMTL_CLASS_ID, 0))
    {
        stdMap = true;
    }
    else
    {
        minMapVal = 0;
    }

    for (id = minMapVal; id <= ID_DP; id++)
    {
        if (id != ID_OP)
        {
            tds[numTexDesks] = GetMtlTex(mtl, isWire, id, askForSubTexture);
            if (tds[numTexDesks])
            {
                if (tds[numTexDesks]->hasSubTextures)
                {
                    askForSubTexture++;
                    id--; // same ID again
                }
                numTexDesks++;
            }
            else
            {
                askForSubTexture = 0;
            }

            if (numTexDesks >= MAX_TEXTURES)
            {
                break;
            }
        }
    }
    if (((mType != Export_VRML_2_0_COVER) && (mType != Export_X3D_V)) && numTexDesks > 1)
        numTexDesks = 1;
}

TextureDesc *
VRML2Export::GetMtlTex(Mtl *mtl, BOOL &isWire, int textureNumber, int askForSubTexture)
{
    if (!mtl)
        return NULL;
    bool hasSubTextures = false;
    if (mtl && mtl->ClassID() == Class_ID(BAKE_SHELL_CLASS_ID, 0))
    {
        mtl = mtl->GetSubMtl(1);
    }

    StdMat *sm = NULL;
    if (mtl->ClassID() == Class_ID(DMTL_CLASS_ID, 0))
    {

        sm = (StdMat *)mtl;
        isWire = sm->GetWire();
    }

    Texmap *tm = NULL;
    if (sm)
    {
        // Check for texture map
        tm = (BitmapTex *)sm->GetSubTexmap(textureNumber);
        if (!sm->MapEnabled(textureNumber))
            return NULL;
    }
    else
    {

        // Check for texture map
        tm = (BitmapTex *)mtl->GetSubTexmap(textureNumber);
    }
    if (!tm)
        return NULL;

    Class_ID id;
    id = tm->ClassID();
#define GNORMAL_CLASS_ID Class_ID(0x243e22c6, 0x63f6a014)
    /* if(id == Class_ID(GNORMAL_CLASS_ID,0)
		int					NumSubTexmaps()			{ return NSUBTEX; }
		Texmap*				GetSubTexmap(int i)		{ return subTex[i]; }
   if(id == Class_ID(MASK_CLASS_ID,0))
   {
      tm = tm->GetSubTexmap(0);
   }*/
    Texmap *origtm = tm;
    if (tm->NumSubTexmaps() > askForSubTexture)
    {
        do
        {
            tm = origtm->GetSubTexmap(askForSubTexture);
            askForSubTexture++;
        } while (tm == NULL && askForSubTexture < origtm->NumSubTexmaps());
        if (tm)
        {
            hasSubTextures = true;
        }
        else
        {
            return NULL;
        }
    }

    if (tm->ClassID() == Class_ID(ACUBIC_CLASS_ID, 0))
    {
        StdCubic *cm = (StdCubic *)tm;

        IParamBlock2 *pblock = cm->GetParamBlock(0);

        //IParamBlock *pblock = (IParamBlock *) bm->GetTheUVGen()->SubAnim(0);
        float blurOffset = 0.0;
#define PB_UOFFS 0
#define PB_VOFFS 1
#define PB_USCL 2
#define PB_VSCL 3
#define PB_UANGLE 4
#define PB_VANGLE 5
#define PB_WANGLE 6
#define PB_BLUR 7
#define PB_NSAMT 8
#define PB_NSSIZ 9
#define PB_NSLEV 10
#define PB_NSPHS 11
#define PB_BLUROFFS 12
        pblock->GetValue(PB_BLUROFFS, mStart, blurOffset, FOREVER);

        BOOL on;
        pblock->GetValue(acubic_source, mStart, on, FOREVER);
        if (on)
        {

            TextureDesc *td = new TextureDesc(cm);
            td->repeatS = (cm->GetTextureTiling() & U_WRAP) != 0;
            td->repeatT = (cm->GetTextureTiling() & V_WRAP) != 0;
            td->hasSubTextures = hasSubTextures;
            for (int i = 0; i < 6; i++)
            {
                MCHAR *name;
                //#if MAX_PRODUCT_VERSION_MAJOR > 11
                pblock->GetValue(acubic_bitmap_names, mStart, (const MCHAR *&)name, FOREVER, i);
                //#else
                //            pblock->GetValue(acubic_bitmap_names,mStart,(MCHAR*&) name,FOREVER,i);
                //#endif
                TSTR bitmapFile, url, fileName;
                bitmapFile = name;

                td->textureID = textureNumber;
                td->blendMode = (int)(blurOffset * 10);

                if (!processTexture(bitmapFile, td->names[i], td->urls[i]))
                    return NULL;
            }
            return td;
        }
        return NULL;
    }

    if (tm->ClassID() != Class_ID(BMTEX_CLASS_ID, 0))
        return NULL;
    BitmapTex *bm = (BitmapTex *)tm;
    IParamBlock *pblock = (IParamBlock *)bm->GetTheUVGen()->SubAnim(0);
    float blurOffset = 0.0;
#define PB_UOFFS 0
#define PB_VOFFS 1
#define PB_USCL 2
#define PB_VSCL 3
#define PB_UANGLE 4
#define PB_VANGLE 5
#define PB_WANGLE 6
#define PB_BLUR 7
#define PB_NSAMT 8
#define PB_NSSIZ 9
#define PB_NSLEV 10
#define PB_NSPHS 11
#define PB_BLUROFFS 12
    pblock->GetValue(PB_BLUROFFS, mStart, blurOffset, FOREVER);

    TSTR bitmapFile, url, fileName;
    bitmapFile = bm->GetMapName();
    MaxSDK::AssetManagement::AssetUser assetUsr = bm->GetMap();
    assetUsr.GetFullFilePath(bitmapFile);

    if (!processTexture(bitmapFile, fileName, url))
        return NULL;
    TextureDesc *td = new TextureDesc(bm, fileName, url, bm->GetTheUVGen()->GetMapChannel());
    td->repeatS = (bm->GetTextureTiling() & U_WRAP) != 0;
    td->repeatT = (bm->GetTextureTiling() & V_WRAP) != 0;
    td->hasSubTextures = hasSubTextures;
    td->textureID = textureNumber;
    if (textureNumber == ID_DI)
        haveDiffuseMap = true;
    td->blendMode = (int)(blurOffset * 10);
    return td;
}

BOOL VRML2Export::processTexture(TSTR bitmapFile, TSTR &fileName, TSTR &url)
{

    if (bitmapFile.data() == NULL)
        return FALSE;
    ////int l = strlen(bitmapFile)-1;
    int l = bitmapFile.Length() - 1;
    if (l < 0)
        return FALSE;

    TSTR path;
    SplitPathFile(bitmapFile, &path, &fileName);

    if (mCopyTextures)
    {
        // check, if destination is older than source, then overwrite, otherwise ask.
        // TODO
        TSTR progressText;
        TSTR destPath;
        TSTR wrlFileName;
        TSTR wrlName = mFilename;
        TSTR slash = _T("/");

        sourceFile = bitmapFile;

        SplitPathFile(wrlName, &destPath, &wrlFileName);
        //command = "copy ";
        TSTR space = _T(" ");
        struct _tfinddata_t fileinfo;
        //try to find destdir
        TSTR destDir = destPath;
        TSTR backSlash = _T("\\");
        if (mUsePrefix && mUrlPrefix.Length() > 0)
        {
            destDir = destDir + backSlash + mUrlPrefix;
            if (mUrlPrefix[mUrlPrefix.Length() - 1] == '/')
                destDir.remove(mUrlPrefix.Length() - 1);
        }

        destFile = destDir + backSlash + fileName;
        intptr_t res = _tfindfirst(destDir, &fileinfo);
        if (res == -1)
        {
            // destdir does not exist so create it
            //command = mkdir + destDir;
            //system(command);
            _tmkdir(destDir);
        }

        bool copyToDest = false;
        bool copyFromDest = false;
        int fd = _topen(destFile, O_RDONLY);
        if (fd > 0)
        {
            int fdS = _topen(bitmapFile, O_RDONLY);
            if (fdS > 0)
            {
                struct _stat dBuf, sBuf;
                _fstat(fd, &dBuf);
                _fstat(fdS, &sBuf);
                if (dBuf.st_mtime < sBuf.st_mtime)
                {
                    if (!mReplaceAll && !mSkipAll)
                    {
                        askForConfirmation();
                    }
                    if (mReplaceAll)
                    {
                        copyToDest = true;
                    }
                    if (mReplace)
                    {
                        copyToDest = true;
                    }
                }
                _close(fdS);
            }
            else
            {
                copyFromDest = true;
            }
            _close(fd);
        }
        else
        {
            int fdS = _topen(bitmapFile, O_RDONLY);
            if (fdS > 0)
            {
                copyToDest = true;
                _close(fdS);
            }
        }
        if (copyToDest)
        {
            if (mEnableProgressBar)
            {
                progressText = TSTR(_T("copying ")) + bitmapFile + TSTR(_T(" to ")) + destFile;
                SendMessage(hWndPDlg, 666, 0, (LPARAM)(char *)progressText.data());
            }
            CopyFile(bitmapFile, destFile, FALSE);
        }
        else if (copyFromDest)
        {
            if (mEnableProgressBar)
            {
                progressText = TSTR(_T("copying ")) + destFile + TSTR(_T(" to ")) + bitmapFile;
                SendMessage(hWndPDlg, 666, 0, (LPARAM)(char *)progressText.data());
            }
            CopyFile(destFile, bitmapFile, FALSE);
        }
    }
    url = PrefixUrl(fileName);
    return TRUE;
}

string &ReplaceAll(string &context, const string &from, const string &to)
{
    size_t lookHere = 0;
    size_t foundHere;

    while ((foundHere = context.find(from, lookHere)) != string::npos)
    {
        context.replace(foundHere, from.size(), to);
        lookHere = foundHere + to.size();
    }

    return context;
}

typedef std::basic_string<TCHAR, std::char_traits<TCHAR>, std::allocator<TCHAR> > tstring;

struct testfunctor
{
    void operator()(TCHAR &c)
    {
        if (c == _T('.'))
            c = _T('_');
        if (c == _T(' '))
            c = _T('_');
        if (c == _T('/'))
            c = _T('_');
        if (c == _T('\\'))
            c = _T('_');
    }
};

TCHAR *VRML2Export::textureName(const TCHAR *name, int blendMode, bool environment)
{

    TCHAR *newName = new TCHAR[_tcslen(name) + 100];
    if (name[0] >= '0' && name[0] <= '9')
        _stprintf(newName, _T("x%s"), name);
    else
        _tcscpy(newName, name);
    //int i;
    for (int i = 0; i < _tcslen(newName); i++)
    {

        if (newName[i] == _T('.'))
            newName[i] = _T('_');
        if (newName[i] == _T(' '))
            newName[i] = _T('_');
        if (newName[i] == _T('/'))
            newName[i] = _T('_');
        if (newName[i] == _T('\\'))
            newName[i] = _T('_');
    }
    if (environment)
        _tcscat(newName, _T("_environment"));
    if (blendMode)
    {
        TCHAR num[100];
        _stprintf(num, _T("_blendMode%d"), blendMode);
        _tcscat(newName, num);
    }

    return newName;
}

BOOL
VRML2Export::OutputMaterial(INode *node, BOOL &isWire, BOOL &twoSided,
                            int level, int textureNum)
{
    Mtl *mtl = node->GetMtl();
    Mtl *origMtl = mtl;
    BOOL isMulti = FALSE;
    isWire = FALSE;
    twoSided = FALSE;
    int texNum;
    bool isBaked = false;
    numShaderTextures = 0;

    effect = NO_EFFECT;
    Indent(level++);
   MSTREAMPRINTF  _T("appearance Appearance {\n"));

   if (mtl && mtl->ClassID() == Class_ID(BAKE_SHELL_CLASS_ID, 0))
   {
       isBaked = true;
       mtl = mtl->GetSubMtl(1);
   }
   if (mtl && mtl->IsMultiMtl())
   {
       if (textureNum > -1)
       {
           mtl = mtl->GetSubMtl(textureNum);
       }
       else
       {

           int subMaterialCount = mtl->NumSubMtls(); // get first sub material (the first might be NULL)

           for (int subMaterialIndex = 0; subMaterialIndex < subMaterialCount; ++subMaterialIndex)
           {

               Mtl *material = mtl->GetSubMtl(subMaterialIndex);
               if (material)
               {
                   mtl = material;
                   break;
               }
           }
       }
       isMulti = TRUE;
       // Use first material for specular, etc.
   }
   if (mtl && mtl->ClassID() == Class_ID(BAKE_SHELL_CLASS_ID, 0))
   {
       if (isBaked)
       {
           //TCHAR msg[MAX_PATH];
           TCHAR title[MAX_PATH];
           //LoadString(hInstance, IDS_OPEN_FAILED, msg, MAX_PATH);
           TCHAR msg[500];
           _stprintf(msg, _T("%s\nnode:%s\nmaterial:%s"), _T("BakeShell within BakeShell, not supported by VRML exporter (jetzt geht's wirklich)"), mNodes.GetNodeName(node), origMtl->GetFullName());
           LoadString(hInstance, IDS_VRML_EXPORT, title, MAX_PATH);
           MessageBox(GetActiveWindow(), msg, title, MB_OK);
       }
       isBaked = true;
       mtl = mtl->GetSubMtl(1);
   }

   // If no material is assigned, use the wire color
   if (!mtl)
   {
       Color col(node->GetWireColor());
       Indent(level);
         MSTREAMPRINTF  _T("material "));
         MSTREAMPRINTF  _T(" Material {\n"));
         Indent(level + 1);
         MSTREAMPRINTF  _T("diffuseColor %s\n"), color(col));
         //        Indent(level+1);
         //        MSTREAMPRINTF  _T("specularColor .9 .9 .9\n"));
         //        MSTREAMPRINTF  _T("specularColor %s\n"), color(col));
         Indent(level);
         MSTREAMPRINTF  _T("}\n"));
         Indent(--level);
         MSTREAMPRINTF  _T("}\n"));
         return FALSE;
   }
   IDxMaterial3 *l_pIDxMaterial3 = (IDxMaterial3 *)mtl->GetInterface(IDXMATERIAL3_INTERFACE);
   if (l_pIDxMaterial3)
   {

       const MaxSDK::AssetManagement::AssetUser &effectFileAsset = l_pIDxMaterial3->GetEffectFile();
       MSTR fN = effectFileAsset.GetFileName();
       TSTR fileName = fN;

       int pos = fileName.last('.');
       if (pos > 0)
       {
           fileName = fileName.Substr(0, pos);
           pos = fileName.last('/');
           if (pos > 0)
           {
               fileName = fileName.Substr(pos + 1, fileName.length() - (pos + 1));
           }
           pos = fileName.last('\\');
           if (pos > 0)
           {
               fileName = fileName.Substr(pos + 1, fileName.length() - (pos + 1));
           }
           fileName.append(_T("coShader"));
           TSTR fn = fileName;
           fileName.append(fn);
           fileName.append(_T("_"));
       }
       ShaderEffect se(fileName);

       IParameterManager *pm = l_pIDxMaterial3->GetCurrentParameterManager();
       if (pm)
       {

           TSTR paramValues;
           for (int i = 0; i < pm->GetNumberOfParams(); i++)
           {
               int pt = pm->GetParamType(i);
               const TCHAR *name = pm->GetParamName(i);
               TCHAR buf[1000];
               switch (pt)
               {
               case IParameterManager::kPType_Float:
               {
                   float fval;
                   pm->GetParamData((void *)&fval, i);
                   _stprintf(buf, _T("%s=%s_"), name, floatVal(fval));
                   paramValues += buf;
                   //pEffect->SetFloat(pm->GetParamName(i), fval);
               }
               break;
               case IParameterManager::kPType_Color:
               case IParameterManager::kPType_Point4:
               {
                   D3DCOLORVALUE cval;
                   pm->GetParamData((void *)&cval, i);
                   _stprintf(buf, _T("%s=%f_%f_%f_%f_"), name, cval.r, cval.g, cval.b, cval.a);
                   paramValues += buf;
                   //pEffect->SetVector(pm->GetParamName(i), (D3DXVECTOR4*)&cval);
               }
               break;
               case IParameterManager::kPType_Bool:
               {
                   BOOL bval;
                   pm->GetParamData((void *)&bval, i);
                   if (bval)
                       _stprintf(buf, _T("%s=true_"), name);
                   else
                       _stprintf(buf, _T("%s=false_"), name);
                   paramValues += buf;
                   //pEffect->SetBool(pm->GetParamName(i), bval);
               }
               break;
               case IParameterManager::kPType_Int:
               {
                   int ival;
                   pm->GetParamData((void *)&ival, i);
                   _stprintf(buf, _T("%s=%d_"), name, ival);
                   paramValues += buf;
                   //pEffect->SetInt(pm->GetParamName(i), ival);
               }
               break;
               case IParameterManager::kPType_Matrix:
               {
                   float mat[16];
                   pm->GetParamData((void *)&mat, i);
                   _stprintf(buf, _T("%s=%f_%f_%f_%f_%f_%f_%f_%f_%f_%f_%f_%f_%f_%f_%f_%f_"), name, mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7], mat[8], mat[9], mat[10], mat[11], mat[12], mat[13], mat[14], mat[15]);
                   paramValues += buf;
                   //pEffect->SetInt(pm->GetParamName(i), ival);
               }
               break;
               case IParameterManager::kPType_Struct:
               {
                   char *buf;
                   int pSize = pm->GetParamSize(i);
                   buf = new char[pSize];
                   pm->GetParamData((void *)buf, i);
                   //pEffect->SetInt(pm->GetParamName(i), ival);
               }
               break;
               case IParameterManager::kPType_Texture:
               {
                   char *buf;
                   int pSize = pm->GetParamSize(i);
                   buf = new char[pSize];
                   pm->GetParamData((void *)buf, i);
                   //pEffect->SetInt(pm->GetParamName(i), ival);
               }

               break;
               default:
               {
                   _ftprintf(stderr, name);
               }
               break;
               }
           }

#if MAX_PRODUCT_VERSION_MAJOR > 14
           paramValues.Replace(_T("="), _T("%"));
           paramValues.Replace(_T("."), _T("&"));
           paramValues.Replace(_T(","), _T("&"));
           paramValues.Replace(_T("-"), _T("$"));
#else
           for (unsigned int i = 0; i < strlen(paramValues); i++)
           {
               if (paramValues[i] == '=')
                   paramValues[i] = '%';
               else if (paramValues[i] == '.')
                   paramValues[i] = '&';
               else if (paramValues[i] == ',')
                   paramValues[i] = '&';
               else if (paramValues[i] == '-')
                   paramValues[i] = '$';
           }
#endif
           se.setParamValues(paramValues);
           effect = (int)shaderEffects.size();
           shaderEffects.push_back(se);
       }
   }

   StdMat *sm = NULL;
   // If no material is assigned, use the wire color
   if (!((mtl->ClassID() != Class_ID(DMTL_CLASS_ID, 0) && mtl->ClassID() != Class_ID(0x3e0810d6, 0x603532f0))))
   {

       sm = (StdMat *)mtl;

       isWire = sm->GetWire();
       twoSided = sm->GetTwoSided();

       Interval i = FOREVER;
       sm->Update(0, i);
       Indent(level);
   }
   MSTREAMPRINTF  _T("material DEF "));
   const TCHAR *mtlName = NULL;
   if (sm)
       mtlName = sm->GetName().data();
   else
       mtlName = mtl->GetName().data();
   TCHAR *matName = new TCHAR[_tcsclen(mtlName) + 100];
   _stprintf(matName, _T("M_%s"), VRMLName(mtlName));
   /*for(unsigned int i=0;i<strlen(matName);i++)
   {
      if(matName[i]==' ')
         matName[i]='_';
      if(matName[i]=='#')
         matName[i]='_';
   }*/

   BOOL dummy = false;
   int numTextureDescs = 0;
   TextureDesc *textureDescs[MAX_TEXTURES];

   haveDiffuseMap = false;
   GetTextures(mtl, dummy, numTextureDescs, textureDescs);

   MSTREAMPRINTF  matName);
   delete[] matName;
   MSTREAMPRINTF  _T(" Material {\n"));
   Color c;
   Color diffuseColor;

   Indent(level + 1);
   if (sm)
   {
       c = sm->GetDiffuse(mStart);
       if (mForceWhite && numTextureDescs != 0 && haveDiffuseMap)
       {
           c.r = 1;
           c.g = 1;
           c.b = 1;
       }
   }
   else
   {
       c = mtl->GetDiffuse();
       if (numTextureDescs != 0 && haveDiffuseMap)
       {
           c.r = 1;
           c.g = 1;
           c.b = 1;
       }
   }
   diffuseColor = c;
   MSTREAMPRINTF  _T("diffuseColor %s\n"), color(c));
#if 1
   Indent(level + 1);
   float difin = (c.r + c.g + c.b) / 3.0f;
   if (sm)
   {
       c = sm->GetAmbient(mStart);
   }
   else
   {
       c = mtl->GetAmbient();
   }
   float ambin = (c.r + c.g + c.b) / 3.0f;
   if (ambin >= difin)
      MSTREAMPRINTF  _T("ambientIntensity 1.0\n"));
   else
      MSTREAMPRINTF  _T("ambientIntensity %s\n"), floatVal(ambin/difin));
   Indent(level + 1);
   if (sm)
   {
       c = sm->GetSpecular(mStart);
       c *= sm->GetShinStr(mStart);
   }
   else
   {
       c = mtl->GetSpecular();
       c *= mtl->GetShinStr();
       if (numTextureDescs != 0 && c.r == 0.0 && c.g == 0.0 && c.b == 0.0)
       {
           c.r = 1;
           c.g = 1;
           c.b = 1;
       }
   }
   MSTREAMPRINTF  _T("specularColor %s\n"), color(c));
#endif

   float sh = 0.0;
   if (sm)
   {
       sh = sm->GetShininess(mStart);
   }
   else
   {
       sh = mtl->GetShininess();
   }
   sh = sh * 0.95f + 0.05f;
   Indent(level + 1);
   MSTREAMPRINTF  _T("shininess %s\n"), floatVal(sh));
   Indent(level + 1);
   if (sm)
   {
   MSTREAMPRINTF  _T("transparency %s\n"),
      floatVal(1.0f - sm->GetOpacity(mStart)));
   }
   else
   {
   MSTREAMPRINTF  _T("transparency %s\n"),
      floatVal(mtl->GetXParency()));
   }
   float si;
   if (sm)
   {
       si = sm->GetSelfIllum(mStart);
   }
   else
   {
       si = mtl->GetSelfIllum();
   }
   if (isBaked)
   {
      MSTREAMPRINTF  _T("emissiveColor 1 1 1\n"));
   }
   else
   {
       if (si > 0.0f)
       {
           Indent(level + 1);
           Point3 p = si * Point3(diffuseColor.r, diffuseColor.g, diffuseColor.b);
         MSTREAMPRINTF  _T("emissiveColor %s\n"), color(p));
       }
   }
   Indent(level);
   MSTREAMPRINTF  _T("}\n"));

   if (isMulti && textureNum == -1)
   {
       Indent(--level);
      MSTREAMPRINTF  _T("}\n"));
      return TRUE;
   }

   if (numTextureDescs == 0 && l_pIDxMaterial3 == NULL)
   {
       Indent(--level);
      MSTREAMPRINTF  _T("}\n"));
      return FALSE;
   }
   if (l_pIDxMaterial3)
   {
       numShaderTextures = l_pIDxMaterial3->GetNumberOfEffectBitmaps();
       if (mType == Export_X3D_V)
       {
           Indent(level);
          MSTREAMPRINTF  _T("texture MultiTexture {\n"));
          level++;
          Indent(level);
          MSTREAMPRINTF  _T("texture ["));
          for (int texNum = 0; texNum < numShaderTextures; texNum++)
          {
              shaderTextureChannel[texNum] = l_pIDxMaterial3->GetBitmapMappingChannel(texNum);
              if (shaderTextureChannel[texNum] > MAX_TEXTURES)
                  shaderTextureChannel[texNum] = -1;

              IDxMaterial2::BitmapTypes bt = l_pIDxMaterial3->GetBitmapUsage(texNum);
              PBBitmap *bitmap = l_pIDxMaterial3->GetEffectBitmap(texNum);
              if (!bitmap)
                  continue;
              const MaxSDK::AssetManagement::AssetUser &bmFileAsset = bitmap->bi.GetAsset();
              MSTR fN = bmFileAsset.GetFileName();
              bool useTexture = false;

              TSTR url, fileName2;
              if (!processTexture(fN.data(), fileName2, url))
                  continue;
              //TextureDesc* td = new TextureDesc(bm, fileName2, url, bm->GetTheUVGen()->GetMapChannel());

              textureTableString *texString = new textureTableString;

              texString->textureName = textureName(url, 0);
              Tab<const TCHAR *> subTextureTable = mTextureTable.Find(texString);
              if (subTextureTable.Count() != 0)
              {
                  for (int i = 0; i < subTextureTable.Count(); i++)
                      if (_tcscmp(texString->textureName, subTextureTable[i]) == 0)
                      {
                          useTexture = true;
                          break;
                      }
              }
              if (!useTexture)
                  mTextureTable.Add(texString);

              TCHAR *movieName = isMovie(url);
              if (useTexture) MSTREAMPRINTF  _T("USE %s\n"), texString->textureName);
              else
              {
                  if (movieName)
                  {
                   MSTREAMPRINTF  _T("DEF %s MovieTexture {\n"), texString->textureName);
                  }

                  else
                  {
                   MSTREAMPRINTF  _T("DEF %s ImageTexture {\n"), texString->textureName);
                  }
                  Indent(level + 1);
                MSTREAMPRINTF  _T("url \"%s\"\n"), url);
                Indent(level);
                MSTREAMPRINTF  _T("}\n"));
              }
          }
          Indent(level);
          MSTREAMPRINTF  _T("]\n"));
          level--;
          Indent(level);
          MSTREAMPRINTF  _T("}\n"));
       }
       else
       {
           for (int texNum = 0; texNum < numShaderTextures; texNum++)
           {
               if (texNum == 0)
               {
                   Indent(level);
                MSTREAMPRINTF  _T("texture "));
               }
               else
               {
                   Indent(level);
                MSTREAMPRINTF  _T("texture%d "),texNum+1);
               }
               shaderTextureChannel[texNum] = l_pIDxMaterial3->GetBitmapMappingChannel(texNum);
               if (shaderTextureChannel[texNum] > MAX_TEXTURES)
                   shaderTextureChannel[texNum] = -1;

               IDxMaterial2::BitmapTypes bt = l_pIDxMaterial3->GetBitmapUsage(texNum);
               PBBitmap *bitmap = l_pIDxMaterial3->GetEffectBitmap(texNum);
               if (!bitmap)
                   continue;
               const MaxSDK::AssetManagement::AssetUser &bmFileAsset = bitmap->bi.GetAsset();
               MSTR fN = bmFileAsset.GetFileName();
               bool useTexture = false;

               TSTR url, fileName2;
               if (!processTexture(fN.data(), fileName2, url))
                   continue;
               //TextureDesc* td = new TextureDesc(bm, fileName2, url, bm->GetTheUVGen()->GetMapChannel());

               textureTableString *texString = new textureTableString;

               texString->textureName = textureName(url, 0);
               Tab<const TCHAR *> subTextureTable = mTextureTable.Find(texString);
               if (subTextureTable.Count() != 0)
               {
                   for (int i = 0; i < subTextureTable.Count(); i++)
                       if (_tcscmp(texString->textureName, subTextureTable[i]) == 0)
                       {
                           useTexture = true;
                           break;
                       }
               }
               if (!useTexture)
                   mTextureTable.Add(texString);

               TCHAR *movieName = isMovie(url);
               if (useTexture) MSTREAMPRINTF  _T("USE %s\n"), texString->textureName);
               else
               {
                   if (movieName)
                   {
                   MSTREAMPRINTF  _T("DEF %s MovieTexture {\n"), texString->textureName);
                   /* Indent(level+1);
                   MSTREAMPRINTF  _T("speed %s\n"),floatVal(textureDescs[texNum]->tex->GetPlaybackRate()));
                   Indent(level+1);
                   MSTREAMPRINTF  _T("startTime %d\n"),textureDescs[texNum]->tex->GetStartTime()/160.0);
                   Indent(level+1);
                   MSTREAMPRINTF  _T("stopTime -1\n"));
                   if(textureDescs[texNum]->tex->GetEndCondition()==END_LOOP)
                   {
                   Indent(level+1);
                   MSTREAMPRINTF  _T("loop TRUE\n"));
                   }
                   else
                   {
                   Indent(level+1);
                   MSTREAMPRINTF  _T("loop FALSE\n"));
                   }*/
                   }

                   else
                   {
                   MSTREAMPRINTF  _T("DEF %s ImageTexture {\n"), texString->textureName);
                   }
                   Indent(level + 1);
                MSTREAMPRINTF  _T("url \"%s\"\n"), url);

                Indent(level);
                MSTREAMPRINTF  _T("}\n"));
               }
           }
       }
   }
   else
   {
       for (texNum = 0; texNum < numTextureDescs; texNum++)
       {
           if (texNum == 0)
           {
               Indent(level);
            MSTREAMPRINTF  _T("texture "));
           }
           else
           {
               Indent(level);
            MSTREAMPRINTF  _T("texture%d "),texNum+1);
           }

           bool useTexture = false;
           textureTableString *texString = new textureTableString;
           if (textureDescs[texNum]->tex)
           {
               texString->textureName = textureName(textureDescs[texNum]->url, textureDescs[texNum]->blendMode);
               Tab<const TCHAR *> subTextureTable = mTextureTable.Find(texString);
               if (subTextureTable.Count() != 0)
               {
                   for (int i = 0; i < subTextureTable.Count(); i++)
                       if (_tcscmp(texString->textureName, subTextureTable[i]) == 0)
                       {
                           useTexture = true;
                           break;
                       }
               }
               if (!useTexture)
                   mTextureTable.Add(texString);
           }
           else if (textureDescs[texNum]->cm)
           {
               texString->textureName = textureName(textureDescs[texNum]->urls[0], textureDescs[texNum]->blendMode);
               Tab<const TCHAR *> subTextureTable = mTextureTable.Find(texString);
               if (subTextureTable.Count() != 0)
               {
                   for (int i = 0; i < subTextureTable.Count(); i++)
                       if (_tcscmp(texString->textureName, subTextureTable[i]) == 0)
                       {
                           useTexture = true;
                           break;
                       }
               }
               if (!useTexture)
                   mTextureTable.Add(texString);
           }

           TCHAR *movieName = isMovie(textureDescs[texNum]->url);
           if (useTexture) MSTREAMPRINTF  _T("USE %s\n"), texString->textureName);
           else
           {
               if (textureDescs[texNum]->tex && (movieName || textureDescs[texNum]->tex->GetStartTime() > 0))
               {
                   if (movieName) MSTREAMPRINTF  _T("DEF %s MovieTexture {\n"), texString->textureName);
                   else MSTREAMPRINTF  _T(" MovieTexture{\n"));
                   Indent(level + 1);
               MSTREAMPRINTF  _T("speed %s\n"),floatVal(textureDescs[texNum]->tex->GetPlaybackRate()));
               Indent(level + 1);
               MSTREAMPRINTF  _T("startTime %d\n"),textureDescs[texNum]->tex->GetStartTime()/160.0);
               Indent(level + 1);
               MSTREAMPRINTF  _T("stopTime -1\n"));
               if (textureDescs[texNum]->tex->GetEndCondition() == END_LOOP)
               {
                   Indent(level + 1);
                  MSTREAMPRINTF  _T("loop TRUE\n"));
               }
               else
               {
                   Indent(level + 1);
                  MSTREAMPRINTF  _T("loop FALSE\n"));
               }
               }
               else if (textureDescs[texNum]->cm)
               {
               MSTREAMPRINTF  _T("DEF %s CubeTexture {\n"), texString->textureName);
               }
               else
               {
               MSTREAMPRINTF  _T("DEF %s ImageTexture {\n"), texString->textureName);
               }
               if (textureDescs[texNum]->tex)
               {
                   Indent(level + 1);
               MSTREAMPRINTF  _T("url \"%s\"\n"), textureDescs[texNum]->url);
               if ((textureDescs[texNum]->textureID == ID_RL) && (mType == Export_VRML_2_0_COVER))
               {
                   Indent(level + 1);
                  MSTREAMPRINTF  _T("environment TRUE\n"));
                  /*if(effect == BUMP_MAPPING)
                  {
                  effect=BUMP_MAPPING_ENV;
                  }*/
               }
               if (textureDescs[texNum]->repeatS == false)
               {
                  MSTREAMPRINTF  _T("repeatS FALSE\n"));
               }
               if (textureDescs[texNum]->repeatT == false)
               {
                  MSTREAMPRINTF  _T("repeatT FALSE\n"));
               }
               if ((textureDescs[texNum]->blendMode > 0) && (mType == Export_VRML_2_0_COVER))
               {
                   Indent(level + 1);
                  MSTREAMPRINTF  _T("blendMode %d\n"),textureDescs[texNum]->blendMode);
               }
               if ((textureDescs[texNum]->textureID == ID_BU) && (mType == Export_VRML_2_0_COVER))
               {
                   // name it coBump instead
                   //effect=BUMP_MAPPING;
                   //Indent(level+1);
                   //MSTREAMPRINTF  _T("bump TRUE\n"));
               }
               }
               else if (textureDescs[texNum]->cm) // cubeMap
               {
                   if ((textureDescs[texNum]->blendMode > 0) && (mType == Export_VRML_2_0_COVER))
                   {
                       Indent(level + 1);
                  MSTREAMPRINTF  _T("blendMode %d\n"),textureDescs[texNum]->blendMode);
                   }
                   else
                   {
                       Indent(level + 1);
                  MSTREAMPRINTF  _T("blendMode 5\n"));
                   }
                   Indent(level + 1);
               MSTREAMPRINTF  _T("urlXP \"%s\"\n"), textureDescs[texNum]->urls[3]);
               Indent(level + 1);
               MSTREAMPRINTF  _T("urlXN \"%s\"\n"), textureDescs[texNum]->urls[2]);
               Indent(level + 1);
               MSTREAMPRINTF  _T("urlYP \"%s\"\n"), textureDescs[texNum]->urls[5]);
               Indent(level + 1);
               MSTREAMPRINTF  _T("urlYN \"%s\"\n"), textureDescs[texNum]->urls[4]);
               Indent(level + 1);
               MSTREAMPRINTF  _T("urlZP \"%s\"\n"), textureDescs[texNum]->urls[0]);
               Indent(level + 1);
               MSTREAMPRINTF  _T("urlZN \"%s\"\n"), textureDescs[texNum]->urls[1]);
               }
               Indent(level);
            MSTREAMPRINTF  _T("}\n"));
           }
           if (textureDescs[texNum]->tex) // no need for texture transforms for cube maps
           {
               BitmapTex *bm = textureDescs[texNum]->tex;

               StdUVGen *uvGen = bm->GetUVGen();
               if (!uvGen)
                   return FALSE;

               float uOff = uvGen->GetUOffs(mStart);
               float vOff = uvGen->GetVOffs(mStart);
               float uScl = uvGen->GetUScl(mStart);
               float vScl = uvGen->GetVScl(mStart);
               float ang = uvGen->GetAng(mStart);

               if (uOff != 0.0f || vOff != 0.0f || uScl != 1.0f || vScl != 1.0f || ang != 0.0f)
               {

                   Indent(level);
                   if (texNum == 0)
                   {
                     MSTREAMPRINTF  _T("textureTransform "));
                   }
                   else
                   {
                     MSTREAMPRINTF  _T("textureTransform%d "),texNum+1);
                   }
                  MSTREAMPRINTF  _T("TextureTransform {\n"));
                  Indent(level + 1);
                  MSTREAMPRINTF  _T("center 0.5 0.5\n"));
                  if (uOff != 0.0f || vOff != 0.0f)
                  {
                      Indent(level + 1);
                      UVVert uv = UVVert(uOff + 0.5f, vOff + 0.5f, 0.0f);
                     MSTREAMPRINTF  _T("translation %s\n"), texture(uv));
                  }
                  if (ang != 0.0f)
                  {
                      Indent(level + 1);
                     MSTREAMPRINTF  _T("rotation %s\n"), floatVal(ang));
                  }
                  if (uScl != 1.0f || vScl != 1.0f)
                  {
                      Indent(level + 1);
                      UVVert uv = UVVert(uScl, vScl, 0.0f);
                     MSTREAMPRINTF  _T("scale %s\n"), texture(uv));
                  }
                  Indent(level);
                  MSTREAMPRINTF  _T("}\n"));
               }
           }
       }
   }

   for (texNum = 0; texNum < numTextureDescs; texNum++)
       delete textureDescs[texNum];

   Indent(--level);
   MSTREAMPRINTF  _T("}\n"));
   return FALSE;
}

BOOL
VRML2Export::VrmlOutSphereTest(INode *node, Object *obj)
{
    SimpleObject *so = (SimpleObject *)obj;
    float hemi;
    int basePivot, genUV, smooth;
    BOOL isWire = FALSE;
    BOOL td = HasTexture(node, isWire);

    if (isWire)
        return FALSE;

    // Reject "base pivot" mapped, non-smoothed and hemisphere spheres
    so->pblock->GetValue(SPHERE_RECENTER, mStart, basePivot, FOREVER);
    so->pblock->GetValue(SPHERE_GENUVS, mStart, genUV, FOREVER);
    so->pblock->GetValue(SPHERE_HEMI, mStart, hemi, FOREVER);
    so->pblock->GetValue(SPHERE_SMOOTH, mStart, smooth, FOREVER);
    if (!smooth || basePivot || (genUV && td) || hemi > 0.0f)
        return FALSE;
    return TRUE;
}

BOOL
VRML2Export::VrmlOutSphere(INode *node, Object *obj, int level)
{
    SimpleObject *so = (SimpleObject *)obj;
    float radius, hemi;
    int basePivot, genUV, smooth;
    BOOL isWire = FALSE;
    BOOL td = HasTexture(node, isWire);

    if (isWire)
        return FALSE;

    // Reject "base pivot" mapped, non-smoothed and hemisphere spheres
    so->pblock->GetValue(SPHERE_RECENTER, mStart, basePivot, FOREVER);
    so->pblock->GetValue(SPHERE_GENUVS, mStart, genUV, FOREVER);
    so->pblock->GetValue(SPHERE_HEMI, mStart, hemi, FOREVER);
    so->pblock->GetValue(SPHERE_SMOOTH, mStart, smooth, FOREVER);
    if (!smooth || basePivot || (genUV && td) || hemi > 0.0f)
        return FALSE;

    so->pblock->GetValue(SPHERE_RADIUS, mStart, radius, FOREVER);

    Indent(level);

   MSTREAMPRINTF  _T("geometry "));

   MSTREAMPRINTF  _T("Sphere { radius %s }\n"), floatVal(radius));

   return TRUE;
}

BOOL
VRML2Export::VrmlOutCylinderTest(INode *node, Object *obj)
{
    SimpleObject *so = (SimpleObject *)obj;
    int sliceOn, genUV, smooth;
    BOOL isWire = FALSE;
    BOOL td = HasTexture(node, isWire);

    if (isWire)
        return FALSE;

    // Reject sliced, non-smooth and mapped cylinders
    so->pblock->GetValue(CYLINDER_GENUVS, mStart, genUV, FOREVER);
    so->pblock->GetValue(CYLINDER_SLICEON, mStart, sliceOn, FOREVER);
    so->pblock->GetValue(CYLINDER_SMOOTH, mStart, smooth, FOREVER);
    if (sliceOn || (genUV && td) || !smooth)
        return FALSE;
    return TRUE;
}

BOOL
VRML2Export::VrmlOutCylinderTform(INode *node, Object *obj, int level,
                                  BOOL mirrored)
{
    if (!VrmlOutCylinderTest(node, obj))
        return FALSE;

    float height;
    SimpleObject *so = (SimpleObject *)obj;
    so->pblock->GetValue(CYLINDER_HEIGHT, mStart, height, FOREVER);
#ifdef MIRROR_BY_VERTICES
    if (mirrored)
        height = -height;
#endif

    Indent(level);
   MSTREAMPRINTF  _T("Transform {\n"));
   if (mZUp)
   {
       Indent(level + 1);
      MSTREAMPRINTF  _T("rotation 1 0 0 %s\n"),
         floatVal(float(PI/2.0)));
      Indent(level + 1);
      MSTREAMPRINTF  _T("translation 0 0 %s\n"),
         floatVal(float(height/2.0)));
   }
   else
   {
       Point3 p = Point3(0.0f, 0.0f, height / 2.0f);
       Indent(level + 1);
      MSTREAMPRINTF  _T("translation %s\n"), point(p));
   }
   Indent(level + 1);
   MSTREAMPRINTF  _T("children [\n"));
   return TRUE;
}

BOOL
VRML2Export::VrmlOutCylinder(INode *node, Object *obj, int level)
{
    SimpleObject *so = (SimpleObject *)obj;
    float radius, height;
    int sliceOn, genUV, smooth;
    BOOL isWire = FALSE;
    BOOL td = HasTexture(node, isWire);

    if (isWire)
        return FALSE;

    // Reject sliced, non-smooth and mapped cylinders
    so->pblock->GetValue(CYLINDER_GENUVS, mStart, genUV, FOREVER);
    so->pblock->GetValue(CYLINDER_SLICEON, mStart, sliceOn, FOREVER);
    so->pblock->GetValue(CYLINDER_SMOOTH, mStart, smooth, FOREVER);
    if (sliceOn || (genUV && td) || !smooth)
        return FALSE;

    so->pblock->GetValue(CYLINDER_RADIUS, mStart, radius, FOREVER);
    so->pblock->GetValue(CYLINDER_HEIGHT, mStart, height, FOREVER);
    Indent(level);
   MSTREAMPRINTF  _T("geometry "));
   MSTREAMPRINTF  _T("Cylinder { radius %s "), floatVal(radius));
   MSTREAMPRINTF  _T("height %s }\n"), floatVal(float(fabs(height))));

   return TRUE;
}

BOOL
VRML2Export::VrmlOutConeTest(INode *node, Object *obj)
{
    SimpleObject *so = (SimpleObject *)obj;
    float radius2;
    int sliceOn, genUV, smooth;
    BOOL isWire = FALSE;
    BOOL td = HasTexture(node, isWire);

    if (isWire)
        return FALSE;

    // Reject sliced, non-smooth and mappeded cylinders
    so->pblock->GetValue(CONE_GENUVS, mStart, genUV, FOREVER);
    so->pblock->GetValue(CONE_SLICEON, mStart, sliceOn, FOREVER);
    so->pblock->GetValue(CONE_SMOOTH, mStart, smooth, FOREVER);
    so->pblock->GetValue(CONE_RADIUS2, mStart, radius2, FOREVER);
    if (sliceOn || (genUV && td) || !smooth || radius2 > 0.0f)
        return FALSE;
    return TRUE;
}

BOOL
VRML2Export::VrmlOutConeTform(INode *node, Object *obj, int level,
                              BOOL mirrored)
{
    if (!VrmlOutConeTest(node, obj))
        return FALSE;
    Indent(level);
   MSTREAMPRINTF  _T("Transform {\n"));

   float height;
   SimpleObject *so = (SimpleObject *)obj;
   so->pblock->GetValue(CONE_HEIGHT, mStart, height, FOREVER);
#ifdef MIRROR_BY_VERTICES
   if (mirrored)
       height = -height;
#endif

   if (mZUp)
   {
       Indent(level + 1);
       if (height > 0.0f)
         MSTREAMPRINTF  _T("rotation 1 0 0 %s\n"),
         floatVal(float(PI/2.0)));
       else
         MSTREAMPRINTF  _T("rotation 1 0 0 %s\n"),
         floatVal(float(-PI/2.0)));
       Indent(level + 1);
      MSTREAMPRINTF  _T("translation 0 0 %s\n"),
         floatVal(float(fabs(height)/2.0)));
   }
   else
   {
       Point3 p = Point3(0.0f, 0.0f, (float)height / 2.0f);
       if (height < 0.0f)
       {
           Indent(level + 1);
         MSTREAMPRINTF  _T("rotation 1 0 0 %s\n"),
            floatVal(float(PI)));
       }
       Indent(level + 1);
      MSTREAMPRINTF  _T("translation %s\n"), point(p));
   }

   Indent(level + 1);
   MSTREAMPRINTF  _T("children [\n"));
   return TRUE;
}

BOOL
VRML2Export::VrmlOutCone(INode *node, Object *obj, int level)
{
    SimpleObject *so = (SimpleObject *)obj;
    float radius1, radius2, height;
    int sliceOn, genUV, smooth;
    BOOL isWire = FALSE;
    BOOL td = HasTexture(node, isWire);

    if (isWire)
        return FALSE;

    // Reject sliced, non-smooth and mappeded cylinders
    so->pblock->GetValue(CONE_GENUVS, mStart, genUV, FOREVER);
    so->pblock->GetValue(CONE_SLICEON, mStart, sliceOn, FOREVER);
    so->pblock->GetValue(CONE_SMOOTH, mStart, smooth, FOREVER);
    so->pblock->GetValue(CONE_RADIUS2, mStart, radius2, FOREVER);
    if (sliceOn || (genUV && td) || !smooth || radius2 > 0.0f)
        return FALSE;

    so->pblock->GetValue(CONE_RADIUS1, mStart, radius1, FOREVER);
    so->pblock->GetValue(CONE_HEIGHT, mStart, height, FOREVER);
    Indent(level);

   MSTREAMPRINTF  _T("geometry "));

   MSTREAMPRINTF  _T("Cone { bottomRadius %s "), floatVal(radius1));
   MSTREAMPRINTF  _T("height %s }\n"), floatVal(float(fabs(height))));

   return TRUE;
}

BOOL
VRML2Export::VrmlOutCubeTest(INode *node, Object *obj)
{
    Mtl *mtl = node->GetMtl();
    // Multi materials need meshes
    if (mtl && mtl->ClassID() == Class_ID(BAKE_SHELL_CLASS_ID, 0))
    {
        mtl = mtl->GetSubMtl(1);
    }
    if (mtl && mtl->IsMultiMtl())
        return FALSE;

    SimpleObject *so = (SimpleObject *)obj;
    BOOL isWire = FALSE;
    BOOL td = HasTexture(node, isWire);

    if (isWire)
        return FALSE;

    int genUV, lsegs, wsegs, hsegs;
    so->pblock->GetValue(BOXOBJ_GENUVS, mStart, genUV, FOREVER);
    so->pblock->GetValue(BOXOBJ_LSEGS, mStart, lsegs, FOREVER);
    so->pblock->GetValue(BOXOBJ_WSEGS, mStart, hsegs, FOREVER);
    so->pblock->GetValue(BOXOBJ_HSEGS, mStart, wsegs, FOREVER);
    if ((genUV && td) || lsegs > 1 || hsegs > 1 || wsegs > 1)
        return FALSE;

    return TRUE;
}

BOOL
VRML2Export::VrmlOutCubeTform(INode *node, Object *obj, int level,
                              BOOL mirrored)
{
    if (!VrmlOutCubeTest(node, obj))
        return FALSE;
    Indent(level);
   MSTREAMPRINTF  _T("Transform {\n"));

   float height;
   SimpleObject *so = (SimpleObject *)obj;
   so->pblock->GetValue(BOXOBJ_HEIGHT, mStart, height, FOREVER);
#ifdef MIRROR_BY_VERTICES
   if (mirrored)
       height = -height;
#endif

   Point3 p = Point3(0.0f, 0.0f, height / 2.0f);
   // VRML cubes grow from the middle, MAX grows from z=0
   Indent(level + 1);
   MSTREAMPRINTF  _T("translation %s\n"), point(p));

   Indent(level + 1);
   MSTREAMPRINTF  _T("children [\n"));
   return TRUE;
}

BOOL
VRML2Export::VrmlOutCube(INode *node, Object *obj, int level)
{
    Mtl *mtl = node->GetMtl();
    // Multi materials need meshes
    if (mtl && mtl->ClassID() == Class_ID(BAKE_SHELL_CLASS_ID, 0))
    {
        mtl = mtl->GetSubMtl(1);
    }
    if (mtl && mtl->IsMultiMtl())
        return FALSE;

    SimpleObject *so = (SimpleObject *)obj;
    float length, width, height;
    BOOL isWire = FALSE;
    BOOL td = HasTexture(node, isWire);

    if (isWire)
        return FALSE;

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
   MSTREAMPRINTF  _T("geometry "));
   if (mZUp)
   {
      MSTREAMPRINTF  _T("Box { size %s "),
         floatVal(float(fabs(width))));
      MSTREAMPRINTF  _T("%s "),
         floatVal(float(fabs(length))));
      MSTREAMPRINTF  _T("%s }\n"),
         floatVal(float(fabs(height))));
   }
   else
   {
      MSTREAMPRINTF  _T("Box { size %s "),
         floatVal(float(fabs(width))));
      MSTREAMPRINTF  _T("%s "),
         floatVal(float(fabs(height))));
      MSTREAMPRINTF  _T("%s }\n"),
         floatVal(float(fabs(length))));
   }

   return TRUE;
}

#define INTENDED_ASPECT_RATIO 1.3333

BOOL
VRML2Export::VrmlOutCamera(INode *node, Object *obj, int level)
{

    if (node == mCamera)
        return false;
    if (!doExport(node))
        return false;

    CameraObject *cam = (CameraObject *)node->EvalWorldState(mStart).obj;
    Matrix3 tm = GetLocalTM(node, mStart);
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
        AngAxisFromQa(q / qRot, &ang, axis);
    }
    else
        AngAxisFromQa(q, &ang, axis);

    // compute camera transform
    ViewParams vp;
    CameraState cs;
    Interval iv;
    cam->EvalCameraState(0, iv, &cs);
    vp.fov = (float)(2.0 * atan(tan(cs.fov / 2.0) / INTENDED_ASPECT_RATIO));

    Indent(level);
   MSTREAMPRINTF  _T("DEF %s Viewpoint {\n"), mNodes.GetNodeName(node));
   Indent(level + 1);
   MSTREAMPRINTF  _T("position %s\n"), point(p));
   Indent(level + 1);
   MSTREAMPRINTF  _T("orientation %s\n"), axisPoint(axis, -ang));
   Indent(level + 1);
   MSTREAMPRINTF  _T("fieldOfView %s\n"), floatVal(vp.fov));
   Indent(level + 1);
   MSTREAMPRINTF  _T("description \"%s\"\n"), mNodes.GetNodeName(node));

   if (cam->IsOrtho() && (mType == Export_VRML_2_0_COVER))
   {
       Indent(level + 1);
      MSTREAMPRINTF  _T("type \"free\"\n"));
   }
   Indent(level);
   MSTREAMPRINTF  _T("}\n"));

   return TRUE;
}

#define FORDER(A, B)   \
    if (B < A)         \
    {                  \
        float fOO = A; \
        A = B;         \
        B = fOO;       \
    }

BOOL
VRML2Export::VrmlOutSound(INode *node, SoundObject *obj, int level)
{
    float intensity, priority, minBack, maxBack, minFront, maxFront;
    int spatialize;

    obj->pblock->GetValue(PB_SND_INTENSITY, mStart, intensity, FOREVER);
    obj->pblock->GetValue(PB_SND_PRIORITY, mStart, priority, FOREVER);
    obj->pblock->GetValue(PB_SND_SPATIALIZE, mStart, spatialize, FOREVER);
    obj->pblock->GetValue(PB_SND_MIN_BACK, mStart, minBack, FOREVER);
    obj->pblock->GetValue(PB_SND_MAX_BACK, mStart, maxBack, FOREVER);
    obj->pblock->GetValue(PB_SND_MIN_FRONT, mStart, minFront, FOREVER);
    obj->pblock->GetValue(PB_SND_MAX_FRONT, mStart, maxFront, FOREVER);

    Point3 dir(0, -1, 0);

    FORDER(minBack, maxBack);
    FORDER(minFront, maxFront);
    if (minFront < minBack)
    {
        float temp = minFront;
        minFront = minBack;
        minBack = temp;
        temp = maxFront;
        maxFront = maxBack;
        maxBack = temp;
        dir = -dir;
    }

    Indent(level);
   MSTREAMPRINTF  _T("DEF %s Sound {\n"), mNodes.GetNodeName(node));
   Indent(level + 1);
   MSTREAMPRINTF  _T("direction %s\n"), point(dir));
   Indent(level + 1);
   MSTREAMPRINTF  _T("intensity %s\n"), floatVal(intensity));
   Indent(level + 1);
   MSTREAMPRINTF  _T("location 0 0 0\n"));
   Indent(level + 1);
   MSTREAMPRINTF  _T("maxBack %s\n"), floatVal(maxBack));
   Indent(level + 1);
   MSTREAMPRINTF  _T("maxFront %s\n"), floatVal(maxFront));
   Indent(level + 1);
   MSTREAMPRINTF  _T("minBack %s\n"), floatVal(minBack));
   Indent(level + 1);
   MSTREAMPRINTF  _T("minFront %s\n"), floatVal(minFront));
   Indent(level + 1);
   MSTREAMPRINTF  _T("priority %s\n"), floatVal(priority));
   Indent(level + 1);
   MSTREAMPRINTF  _T("spatialize %s\n"),
      spatialize ? _T("TRUE") : _T("FALSE"));
   if (obj->audioClip)
   {
       Indent(level + 1);
       //        MSTREAMPRINTF  _T("source USE %s\n"), VRMLName(obj->audioClip->GetName()));
      MSTREAMPRINTF  _T("source\n"));
      VrmlOutAudioClip(level + 2, obj->audioClip);
   }
   Indent(level);
   MSTREAMPRINTF  _T("}\n"));

   return TRUE;
}

static INode *
GetTopLevelParent(INode *node)
{
    while (!node->GetParentNode()->IsRootNode())
        node = node->GetParentNode();
    return node;
}

void VRML2Export::SensorBindScript(const TCHAR *objName, const TCHAR *name, int level, INode *node, INode *obj, int type)
{
   MSTREAMPRINTF  _T("\n"));
   Indent(level);
   MSTREAMPRINTF  _T("DEF %s%s-SCRIPT Script {\n"), name, objName);
   Indent(level + 1);
   if (type == KEY_SWITCH_BIND) MSTREAMPRINTF  _T("eventIn SFInt32 active\n"));
   else MSTREAMPRINTF  _T("eventIn SFTime active\n"));
   Indent(level + 1);
   MSTREAMPRINTF  _T("eventOut SFBool state\n"));
   Indent(level + 1);
   MSTREAMPRINTF  _T("url \"javascript:\n"));
   Indent(level + 2);
   MSTREAMPRINTF  _T("function active(t) {\n"));
   Indent(level + 3);
   if (type == KEY_SWITCH_BIND)
   {
       SwitchObject *swObj = (SwitchObject *)node->EvalWorldState(mStart).obj;
       int k = 0;
       while ((k < swObj->objects.Count()) && (swObj->objects[k]->node != obj))
           k++;
      MSTREAMPRINTF  _T("if (t == %d) state = TRUE;\n"), k);
   }
   else MSTREAMPRINTF  _T("state = TRUE;\n"));
   Indent(level + 2);
   MSTREAMPRINTF  _T("}\"\n"));
   Indent(level + 1);
   MSTREAMPRINTF  _T("}\n\n"));
}

void VRML2Export::TouchSensorMovieScript(TCHAR *objName, int level)
{
   MSTREAMPRINTF  _T("\n"));
   Indent(level + 1);
   MSTREAMPRINTF  _T("DEF %sStartStop Script {\n"), objName);
   Indent(level + 1);
   MSTREAMPRINTF  _T("eventIn SFTime clickTime\n"));
   Indent(level + 1);
   MSTREAMPRINTF  _T("eventOut SFTime startTime\n"));
   Indent(level + 1);
   MSTREAMPRINTF  _T("eventOut SFTime stopTime\n"));
   Indent(level + 1);
   MSTREAMPRINTF  _T("field SFBool  running TRUE\n"));
   Indent(level + 1);
   MSTREAMPRINTF  _T("url \"javascript:\n"));
   Indent(level + 2);
   MSTREAMPRINTF  _T("function clickTime(t) {\n"));
   Indent(level + 3);
   MSTREAMPRINTF  _T("if(running)\n"));
   Indent(level + 3);
   MSTREAMPRINTF  _T("{\n"));
   Indent(level + 4);
   MSTREAMPRINTF  _T("stopTime = t;\n"));
   Indent(level + 4);
   MSTREAMPRINTF  _T("running = false;\n"));
   Indent(level + 3);
   MSTREAMPRINTF  _T("}\n"));
   Indent(level + 3);
   MSTREAMPRINTF  _T("else\n"));
   Indent(level + 3);
   MSTREAMPRINTF  _T("{\n"));
   Indent(level + 4);
   MSTREAMPRINTF  _T("startTime = t;\n"));
   Indent(level + 4);
   MSTREAMPRINTF  _T("running = true;\n"));
   Indent(level + 3);
   MSTREAMPRINTF  _T("}\n"));
   Indent(level + 2);
   MSTREAMPRINTF  _T("}\"\n"));
   Indent(level + 1);
   MSTREAMPRINTF  _T("}\n"));
}

BOOL
VRML2Export::VrmlOutTouchSensor(INode *node, int level)
{
    TouchSensorObject *obj = (TouchSensorObject *)
                                 node->EvalWorldState(mStart).obj;
    int enabled;
    obj->pblock->GetValue(PB_TS_ENABLED, mStart, enabled, FOREVER);
    Indent(level);
   MSTREAMPRINTF  _T("DEF %s-SENSOR TouchSensor { enabled %s }\n"),mNodes.GetNodeName(node),
      enabled ? _T("TRUE") : _T("FALSE"));

   TCHAR *vrmlObjName = NULL;
   vrmlObjName = VrmlParent(node);
   INode *otop = NULL;
   Class_ID temp[4] = { TimeSensorClassID, SwitchClassID, OnOffSwitchClassID, NavInfoClassID };
   Tab<Class_ID> childClass;
   childClass.Append(4, temp);
   int size = obj->objects.Count();
   for (int i = 0; i < size; i++)
       if (!AddChildObjRoutes(node, obj->objects[i]->node, childClass, otop, vrmlObjName, KEY_TOUCHSENSOR_BIND, 0, level, true))
           break;

   return TRUE;
}

void VRML2Export::VrmlOutSwitchCamera(INode *sw, INode *node, int level)
{

    Indent(level + 1);
   MSTREAMPRINTF  _T("USE %s Transform {\n"), node->GetName());
   Indent(level + 1);
   if (mType == Export_X3D_V)
   {
      MSTREAMPRINTF  _T("children [\n"));
   }
   else
   {
      MSTREAMPRINTF  _T("children [\n"));
   }
   TCHAR *vrmlObjName = NULL;
   vrmlObjName = VrmlParent(sw);
   INode *otop = NULL;

   Tab<Class_ID> childClass;
   Class_ID temp = TimeSensorClassID;
   childClass.Append(1, &temp);
   AddChildObjRoutes(sw, node, childClass, otop, vrmlObjName, KEY_SWITCH_BIND, 0, level + 2, false);
}

int
VRML2Export::VrmlOutSwitch(INode *node, int level)
{
    SwitchObject *obj = (SwitchObject *)
                            node->EvalWorldState(mStart).obj;
    int defaultValue = -1;
    obj->pblock->GetValue(PB_S_DEFAULT, mStart, defaultValue, FOREVER);

    Indent(level);
   MSTREAMPRINTF  _T("DEF %s Switch { \n"),mNodes.GetNodeName(node));
   Indent(level + 1);
   MSTREAMPRINTF  _T("whichChoice %d\n"),defaultValue);
   Indent(level + 1);
   MSTREAMPRINTF  _T("choice[\n"));

   /*   TCHAR* vrmlObjName = NULL;
   vrmlObjName = VrmlParent(node);
   INode *otop = NULL;

   Tab<Class_ID> childClass;
   Class_ID temp = TimeSensorClassID;
   childClass.Append(1, &temp);
   int size = obj->objects.Count();
   for(int i=0; i < size; i++) 
      if (!AddChildObjRoutes(node, obj->objects[i]->node, childClass, otop, vrmlObjName, KEY_SWITCH_BIND, 0, level, false)) break;
   TouchSensorObj* animObj = obj->objects[i];
   Object *o = animObj->node->EvalWorldState(mStart).obj;
   if (!o)
   break;
   assert(vrmlObjName);
   if (IsAimTarget(animObj->node))
   break;
   INode* top;
   if (o->ClassID() == TimeSensorClassID)
   top = animObj->node;
   else
   top = GetTopLevelParent(animObj->node);
   ObjectBucket* ob =
   mObjTable.AddObject(top->EvalWorldState(mStart).obj);
   if (top != otop) {
   AddAnimRoute(vrmlObjName, ob->name.data(), node, top);
   AddCameraAnimRoutes(vrmlObjName, node, top);
   otop = top;
   }
   }*/
   return obj->objects.Count();
}
BOOL
VRML2Export::VrmlOutARSensor(INode *node, ARSensorObject *obj, int level)
{
    int enabled, freeze, headingOnly, currentCamera;
    obj->pblock->GetValue(PB_AR_ENABLED, mStart, enabled, FOREVER);
    obj->pblock->GetValue(PB_AR_FREEZE, mStart, freeze, FOREVER);
    obj->pblock->GetValue(PB_AR_HEADING_ONLY, mStart, headingOnly, FOREVER);
    obj->pblock->GetValue(PB_AR_CURRENT_CAMERA, mStart, currentCamera, FOREVER);
    float minx, miny, minz;
    obj->pblock->GetValue(PB_AR_MINX, mStart, minx, FOREVER);
    obj->pblock->GetValue(PB_AR_MINY, mStart, miny, FOREVER);
    obj->pblock->GetValue(PB_AR_MINZ, mStart, minz, FOREVER);
    float maxx, maxy, maxz;
    obj->pblock->GetValue(PB_AR_MAXX, mStart, maxx, FOREVER);
    obj->pblock->GetValue(PB_AR_MAXY, mStart, maxy, FOREVER);
    obj->pblock->GetValue(PB_AR_MAXZ, mStart, maxz, FOREVER);
    float ipx, ipy, ipz;
    obj->pblock->GetValue(PB_AR_IPX, mStart, ipx, FOREVER);
    obj->pblock->GetValue(PB_AR_IPY, mStart, ipy, FOREVER);
    obj->pblock->GetValue(PB_AR_IPZ, mStart, ipz, FOREVER);
    float pos, ori;
    obj->pblock->GetValue(PB_AR_POS, mStart, pos, FOREVER);
    obj->pblock->GetValue(PB_AR_ORI, mStart, ori, FOREVER);
    const TCHAR *markerName = obj->MarkerName.data();
    Indent(level);
   MSTREAMPRINTF  _T("DEF %s-SENSOR ARSensor { \n"),mNodes.GetNodeName(node));
   Indent(level);
   MSTREAMPRINTF  _T("enabled %s\n"),enabled ? _T("TRUE") : _T("FALSE"));
   Indent(level);
   MSTREAMPRINTF  _T("freeze %s\n"),freeze ? _T("TRUE") : _T("FALSE"));
   Indent(level);
   MSTREAMPRINTF  _T("currentCamera %s\n"),currentCamera ? _T("TRUE") : _T("FALSE"));
   Indent(level);
   MSTREAMPRINTF  _T("headingOnly %s\n"),headingOnly ? _T("TRUE") : _T("FALSE"));
   Indent(level);
   MSTREAMPRINTF  _T("minPosition %s %s %s\n"),floatVal(minx),floatVal(miny),floatVal(minz));
   Indent(level);
   MSTREAMPRINTF  _T("maxPosition %s %s %s\n"),floatVal(maxx),floatVal(maxy),floatVal(maxz));
   Indent(level);
   MSTREAMPRINTF  _T("invisiblePosition %s %s %s\n"),floatVal(ipx),floatVal(ipy),floatVal(ipz));
   Indent(level);
   MSTREAMPRINTF  _T("orientationThreshold %s\n"),floatVal(ori));
   Indent(level);
   MSTREAMPRINTF  _T("positionThreshold %s\n"),floatVal(pos));
   Indent(level);
   MSTREAMPRINTF  _T("markerName \"%s\"\n"),markerName);

   Indent(level);
   MSTREAMPRINTF  _T("}\n"));

   TCHAR *vrmlObjName = NULL;
   vrmlObjName = VrmlParent(node);

   INode *top = obj->triggerObject;
   if (top)
   {
       ObjectBucket *ob = mObjTable.AddObject(top->EvalWorldState(mStart).obj);
       AddAnimRoute(vrmlObjName, ob->name.data(), node, top, ENTER_FIELD);
   }

   return TRUE;
}

BOOL
VRML2Export::VrmlOutMTSensor(INode *node, MultiTouchSensorObject *obj, int level)
{
    int enabled, freeze;
    obj->pblock->GetValue(PB_MT_ENABLED, mStart, enabled, FOREVER);
    obj->pblock->GetValue(PB_MT_FREEZE, mStart, freeze, FOREVER);
    float minx, miny, minz;
    obj->pblock->GetValue(PB_MT_MINX, mStart, minx, FOREVER);
    obj->pblock->GetValue(PB_MT_MINY, mStart, miny, FOREVER);
    obj->pblock->GetValue(PB_MT_MINZ, mStart, minz, FOREVER);
    float sizex, sizey, sizez;
    obj->pblock->GetValue(PB_MT_SIZEX, mStart, sizex, FOREVER);
    obj->pblock->GetValue(PB_MT_SIZEY, mStart, sizey, FOREVER);
    obj->pblock->GetValue(PB_MT_SIZEZ, mStart, sizez, FOREVER);
    float h, p, r;
    obj->pblock->GetValue(PB_MT_ORIH, mStart, h, FOREVER);
    obj->pblock->GetValue(PB_MT_ORIP, mStart, p, FOREVER);
    obj->pblock->GetValue(PB_MT_ORIR, mStart, r, FOREVER);
    float ipx, ipy, ipz;
    obj->pblock->GetValue(PB_MT_IPX, mStart, ipx, FOREVER);
    obj->pblock->GetValue(PB_MT_IPY, mStart, ipy, FOREVER);
    obj->pblock->GetValue(PB_MT_IPZ, mStart, ipz, FOREVER);
    float pos, ori;
    obj->pblock->GetValue(PB_MT_POS, mStart, pos, FOREVER);
    obj->pblock->GetValue(PB_MT_ORI, mStart, ori, FOREVER);
    const TCHAR *markerName = obj->MarkerName.data();
    Indent(level);
   MSTREAMPRINTF  _T("DEF %s-SENSOR MultiTouchSensor { \n"),mNodes.GetNodeName(node));
   level++;
   Indent(level);
   MSTREAMPRINTF  _T("markerName \"%s\"\n"),markerName);
   Indent(level);
   MSTREAMPRINTF  _T("enabled %s\n"),enabled ? _T("TRUE") : _T("FALSE"));
   Indent(level);
   MSTREAMPRINTF  _T("freeze %s\n"),freeze ? _T("TRUE") : _T("FALSE"));
   Indent(level);
   MSTREAMPRINTF  _T("minPosition %s %s %s\n"),floatVal(minx),floatVal(miny),floatVal(minz));
   Indent(level);
   MSTREAMPRINTF  _T("size %s %s\n"),floatVal(sizex),floatVal(sizey));
   Indent(level);

   Quat surfaceRot;
   Quat vrmlRot;
   vrmlRot.SetEuler(PI / 2.0, 0, 0);
   surfaceRot.SetEuler((float)((h / 180.0) * PI), (float)((p / 180.0) * PI), (float)((r / 180.0) * PI));
   surfaceRot *= vrmlRot;
   float wsq = sqrt(1 - (surfaceRot.w * surfaceRot.w));
   float angle = 2 * acos(surfaceRot.w);
   float x = surfaceRot.x / wsq;
   float y = surfaceRot.y / wsq;
   float z = surfaceRot.z / wsq;

   MSTREAMPRINTF  _T("orientation %s %s %s %s\n"),floatVal(x),floatVal(y),floatVal(z),floatVal(angle));
   Indent(level);
   MSTREAMPRINTF  _T("invisiblePosition %s %s %s\n"),floatVal(ipx),floatVal(ipy),floatVal(ipz));
   Indent(level);
   MSTREAMPRINTF  _T("orientationThreshold %s\n"),floatVal(ori));
   Indent(level);
   MSTREAMPRINTF  _T("positionThreshold %s\n"),floatVal(pos));
   level--;
   Indent(level);
   MSTREAMPRINTF  _T("}\n"));

   TCHAR *vrmlObjName = NULL;
   vrmlObjName = VrmlParent(node);

   INode *top = obj->triggerObject;
   if (top)
   {
       ObjectBucket *ob = mObjTable.AddObject(top->EvalWorldState(mStart).obj);
       AddAnimRoute(vrmlObjName, ob->name.data(), node, top, ENTER_FIELD);
   }

   return TRUE;
}

BOOL
VRML2Export::VrmlOutARSensor(INode *node, int level)
{
    ARSensorObject *obj = (ARSensorObject *)
                              node->EvalWorldState(mStart).obj;

    return VrmlOutARSensor(node, obj, level);
}

BOOL
VRML2Export::VrmlOutMTSensor(INode *node, int level)
{
    MultiTouchSensorObject *obj = (MultiTouchSensorObject *)
                                      node->EvalWorldState(mStart).obj;

    return VrmlOutMTSensor(node, obj, level);
}

BOOL
VRML2Export::VrmlOutCOVER(INode *node, COVERObject *obj, int level)
{
    int i;
    if (mType == Export_VRML_2_0_COVER)
    {
        Indent(level);

      MSTREAMPRINTF  _T("DEF %s-SENSOR COVER { \n"),mNodes.GetNodeName(node));

      Indent(level);
      MSTREAMPRINTF  _T("}\n"));

      int numObjs = obj->objects.Count();

      MSTREAMPRINTF  _T("DEF %s-SCRIPT Script {\n"),mNodes.GetNodeName(node));
      Indent(1);
      MSTREAMPRINTF  _T("eventIn SFString key\n"));
      for (i = 0; i < numObjs; i++)
      {
          Indent(1);
         MSTREAMPRINTF  _T("eventOut SFTime key%s\n"),obj->objects[i]->keyStr);
      }
      Indent(1);
      MSTREAMPRINTF  _T("url \"javascript:\n"));
      Indent(2);
      MSTREAMPRINTF  _T("function key(k,t) {\n"));

      for (i = 0; i < numObjs; i++)
      {
          Indent(2);
         MSTREAMPRINTF  _T("if(k == '%s') { key%s = t; }\n"),obj->objects[i]->keyStr,obj->objects[i]->keyStr);
      }
      Indent(1);
      MSTREAMPRINTF  _T("}\"\n"));
      MSTREAMPRINTF  _T("}\n"));

      Indent(level);
      MSTREAMPRINTF  _T("ROUTE %s-SENSOR.keyPressed TO %s-SCRIPT.key\n"),mNodes.GetNodeName(node),mNodes.GetNodeName(node));

      TCHAR *vrmlObjName = NULL;
      vrmlObjName = VrmlParent(node);

      INode *top = obj->triggerObject;
      if (top)
      {
          ObjectBucket *ob = mObjTable.AddObject(top->EvalWorldState(mStart).obj);
          AddAnimRoute(vrmlObjName, ob->name.data(), node, top, -1);
      }

      INode *otop = NULL;
      for (i = 0; i < numObjs; i++)
      {
          COVERObj *animObj = obj->objects[i];
          if (animObj->node)
          {
              Object *o = animObj->node->EvalWorldState(mStart).obj;
              if (!o)
                  break;
              assert(vrmlObjName);
              if (IsAimTarget(animObj->node))
                  break;
              INode *top;
              if (o->ClassID() == TimeSensorClassID)
                  top = animObj->node;
              else if (o->ClassID() == OnOffSwitchClassID)
                  top = animObj->node;
              else if (o->ClassID() == SwitchClassID)
                  top = animObj->node;
              else
                  top = GetTopLevelParent(animObj->node);
              ObjectBucket *ob = mObjTable.AddObject(top->EvalWorldState(mStart).obj);
              if (top != otop)
              {
                  if (top)
                  {
                      top->SetNodeLong(top->GetNodeLong() | RUN_BY_COVER_SENSOR);
                      AddAnimRoute(vrmlObjName, ob->name.data(), node, top, i);
                      AddCameraAnimRoutes(vrmlObjName, node, top, i);
                      otop = top;
                  }
              }
          }
      }
    }
    return TRUE;
}

BOOL
VRML2Export::VrmlOutCOVER(INode *node, int level)
{
    COVERObject *obj = (COVERObject *)
                           node->EvalWorldState(mStart).obj;

    return VrmlOutCOVER(node, obj, level);
}

BOOL
VRML2Export::VrmlOutTUIButton(TabletUIElement *el, INode *node, int level)
{
    TabletUIObject *obj = (TabletUIObject *)
                              node->EvalWorldState(mStart).obj;

    TCHAR *vrmlObjName = NULL;
    vrmlObjName = VrmlParent(node);
    INode *otop = NULL;

    for (int i = 0; i < el->objects.Count(); i++)
    {
        TabletUIObj *animObj = el->objects[i];
        Object *o = animObj->node->EvalWorldState(mStart).obj;
        if (!o)
            break;
        assert(vrmlObjName);
        if (IsAimTarget(animObj->node))
            break;
        INode *top;
        if (o->ClassID() == TimeSensorClassID)
            top = animObj->node;
        else if (o->ClassID() == SwitchClassID)
            top = animObj->node;
        else if (o->ClassID() == OnOffSwitchClassID)
            top = animObj->node;
        else
            top = GetTopLevelParent(animObj->node);
        ObjectBucket *ob = mObjTable.AddObject(top->EvalWorldState(mStart).obj);
        if (top != otop)
        {
            AddAnimRoute(el->name.data(), animObj->listStr, node, node);
            //        AddCameraAnimRoutes(vrmlObjName, node, top);
            otop = top;
        }
    }
    return TRUE;
}

BOOL
VRML2Export::VrmlOutTUIElement(TabletUIElement *el, INode *node, int level)
{
    int index = 0;

    if (!doExport(node))
        return true;

    if (el->type >= 0)
    {
        el->Print(mStream);

        for (int j = 0; j < el->objects.Count(); j++)
        {
            TabletUIObj *tuiobj = (TabletUIObj *)el->objects[j];
            Object *o = tuiobj->node->EvalWorldState(mStart).obj;

            switch (el->type)
            {
            case TUIButton:
            {
                //             VrmlOutTUIButton(el, node, level);
                TSTR fromName = VRMLName(el->name.data());
                fromName += _T(".touchTime");
                TSTR toName = tuiobj->listStr;
                if (o->ClassID() == SwitchClassID)
                {

                    AddAnimRoute(fromName, toName, node, tuiobj->node, 0, TUIButton);
                }
                else if (o->ClassID() == OnOffSwitchClassID)
                {
                    toName += _T("-SCRIPT.trigger");

                    AddAnimRoute(fromName, toName, node, node, 0, TUIButton);
                }
                else if (o->ClassID() == AudioClipClassID)
                {
                    toName += _T(".startTime");

                    AddAnimRoute(fromName, toName, node, node, 0, TUIButton);
                }
                else if (o->ClassID() != NavInfoClassID)
                {
                    toName += _T("-TIMER.startTime");

                    AddAnimRoute(fromName, toName, node, node, 0, TUIButton);
                }
            }
            break;
            case TUIComboBox:
            {
                TSTR fromName = VRMLName(el->name.data());
                AddAnimRoute(fromName, tuiobj->listStr, node, tuiobj->node, 0, TUIComboBox);
            }
            break;
            case TUIFloatSlider:
                if (!mTabletUIList->NodeInList(node))
                    mTabletUIList = mTabletUIList->AddNode(node);
                break;
            case TUIToggleButton:
            {
                if (!mTabletUIList->NodeInList(node))
                    mTabletUIList = mTabletUIList->AddNode(node);

                if (o->ClassID() == SwitchClassID)
                {
                    TSTR fromName = VRMLName(el->name.data());
                    fromName += _T("-SCRIPT");

                    AddAnimRoute(fromName, tuiobj->listStr, node, tuiobj->node, 0, TUIToggleButton);
                }
                else if (o->ClassID() == TimeSensorClassID)
                {
                    TimeSensorObject *obj = (TimeSensorObject *)o;
                    if (!obj->vrmlWritten)
                        VrmlOutTimeSensor(el->objects[j]->node, obj, 0);
                    static_cast<TUIParamToggleButton *>(el->paramRollout)->PrintObjects(mStream, el->objects[j]);
                }
            }
            break;
            default:
                break;
            }

            /*         TabletUIObj *tuiobj = (TabletUIObj *)el->objects[j];
         if ((el->type == 10) || (el->type == 15))
         {
            float cycleInterval = (mIp->GetAnimRange().End() - mStart) /
               ((float) GetTicksPerFrame()* GetFrameRate());
            el->paramRollout->PrintScript(mStream, tuiobj->listStr.data(), cycleInterval);


            if (tuiobj->node)
            {
               ObjectBucket* obucket =
                  mObjTable.AddObject(tuiobj->node->EvalWorldState(mStart).obj);
               tuiobj->node->SetNodeLong(tuiobj->node->GetNodeLong() | RUN_BY_COVER_SENSOR);

               AddAnimRoute(tuiobj->listStr, tuiobj->listStr, node, node, index++);
               //            TimeSensorObject* tso = (TimeSensorObject*)tuiobj->node->EvalWorldState(mStart).obj;


               // find the timesensor closest to the node in the hierarchy
               /*               for(int j = 0; j < tso->TimeSensorObjects.Count(); j++) {
               INode* anim = tso->TimeSensorObjects[j]->node;
               AddAnimRoute(obucket->name.data(), anim->GetName(), node, anim, index++);
               }*/
            /*           }
         }
         else if (el->type == 0)
         {
            Indent(level);
            MSTREAMPRINTF  _T("ROUTE %s.touchTime TO %s-TIMER.startTime\n\n"),el->name.data(),tuiobj->listStr.data());
         }*/
        }
    }

    return TRUE;
}

void
VRML2Export::VrmlOutTUI(INode *node, TabletUIElement *el, int level)
{
    VrmlOutTUIElement(el, node, level);
    for (int j = 0; j < el->children.Count(); j++)
        VrmlOutTUI(node, el->children[j], level);
}

void VRML2Export::BindCamera(INode *node, INode *child, TCHAR *vrmlObjName, int type, int level)
{
    const TCHAR *childName = child->GetName();
    SensorBindScript(vrmlObjName, childName, level, node, child, type);
    if (type != KEY_SWITCH_BIND)
        AddInterpolator(vrmlObjName, type, childName, child);
}

BOOL VRML2Export::AddChildObjRoutes(INode *node, INode *animNode, Tab<Class_ID> childClass, INode *otop, TCHAR *vrmlObjName, int type1, int type2, int level, bool movie)
{
    if (animNode)
    {
        Object *o = animNode->EvalWorldState(mStart).obj;
        if (!o)
            return false;
        assert(vrmlObjName);
        if (IsAimTarget(animNode))
            return false;
        INode *top;
        if (o->SuperClassID() == CAMERA_CLASS_ID)
        {
            top = animNode;
            BindCamera(node, top, vrmlObjName, type1, level);
        }
        else
        {
            int k = 0;
            while ((k < childClass.Count()) && (o->ClassID() != childClass[k]))
                k++;
            if (k >= childClass.Count())
                top = GetTopLevelParent(animNode);
            else if (childClass[k] == TimeSensorClassID)
            {
                top = animNode;
                Tab<Class_ID> grandChildClass;
                TimeSensorObject *to = (TimeSensorObject *)o;
                int numChildren = to->TimeSensorObjects.Count();
                for (int j = 0; j < numChildren; j++)
                    if (!AddChildObjRoutes(node, to->TimeSensorObjects[j]->node, grandChildClass, otop, vrmlObjName, type1, type2, level, false))
                        break;
            }
            else if (childClass[k] == OnOffSwitchClassID)
                top = animNode;
            else if (childClass[k] == SwitchClassID)
                top = animNode;
            else if (childClass[k] == NavInfoClassID)
            {
                top = animNode;
                BindCamera(node, top, vrmlObjName, type1, level);
            }

            if (movie)
            {
                Mtl *mtl = animNode->GetMtl();
                BOOL dummy;
                int numTextureDescs = 0;
                TextureDesc *textureDescs[MAX_TEXTURES];
                GetTextures(mtl, dummy, numTextureDescs, textureDescs);

                for (int texNum = 0; texNum < numTextureDescs; texNum++)
                {
                    TCHAR *movieName = isMovie(textureDescs[texNum]->url);
                    if (textureDescs[texNum]->tex && (movieName || textureDescs[texNum]->tex->GetStartTime() > 0))
                    {
                        TouchSensorMovieScript(vrmlObjName, level);
                        TCHAR *name = new TCHAR[_tcslen(movieName) + _tcslen(animNode->GetName())];
                        _tcscpy(name, animNode->GetName());
                        _tcscat(name, movieName);
                        AddAnimRoute(vrmlObjName, name, node, node);
                    }
                }
            }
        }

        ObjectBucket *ob = mObjTable.AddObject(top->EvalWorldState(mStart).obj);
        if (top != otop)
        {
            AddAnimRoute(vrmlObjName, ob->name.data(), node, top, type2);
            AddCameraAnimRoutes(vrmlObjName, node, top, type2);
            otop = top;
        }
    }
    return TRUE;
}

BOOL
VRML2Export::VrmlOutProxSensor(INode *node, ProxSensorObject *obj,
                               int level)
{
    int enabled;
    float length, width, height;

    obj->pblock->GetValue(PB_PS_ENABLED, mStart, enabled, FOREVER);
    obj->pblock->GetValue(PB_PS_LENGTH, mStart, length, FOREVER);
    obj->pblock->GetValue(PB_PS_WIDTH, mStart, width, FOREVER);
    obj->pblock->GetValue(PB_PS_HEIGHT, mStart, height, FOREVER);
    Indent(level);
   MSTREAMPRINTF  _T("DEF %s ProximitySensor {\n"),mNodes.GetNodeName(node));
   Indent(level + 1);
   MSTREAMPRINTF  _T("enabled %s\n"),
      enabled ? _T("TRUE") : _T("FALSE"));
   Indent(level + 1);
   Point3 center(0.0f, 0.0f, height / 2.0f);
   MSTREAMPRINTF  _T("center %s\n"), point(center));
   Indent(level + 1);
   Point3 size(width, length, height);
   MSTREAMPRINTF  _T("size %s\n"), scalePoint(size));
   Indent(level);
   MSTREAMPRINTF  _T("}\n"));

   TCHAR *vrmlObjName = NULL;
   vrmlObjName = VrmlParent(node);
   INode *otop = NULL;

   Class_ID tmp[3] = { TimeSensorClassID, OnOffSwitchClassID, NavInfoClassID };
   Tab<Class_ID> childClass;
   childClass.Append(3, tmp);

   int numObjs = obj->objects.Count();
   for (int i = 0; i < numObjs; i++)
       if (!AddChildObjRoutes(node, obj->objects[i]->node, childClass, otop, vrmlObjName, KEY_PROXSENSOR_ENTER_BIND, ENTER_FIELD, level, false))
           break;

   numObjs = obj->objectsExit.Count();
   for (int i = 0; i < numObjs; i++)
       if (!AddChildObjRoutes(node, obj->objectsExit[i]->node, childClass, otop, vrmlObjName, KEY_PROXSENSOR_EXIT_BIND, EXIT_FIELD, level, false))
           break;

   return TRUE;
}

BOOL
VRML2Export::VrmlOutBillboard(INode *node, Object *obj, int level)
{
    BillboardObject *bb = (BillboardObject *)obj;
    int screenAlign;
    bb->pblock->GetValue(PB_BB_SCREEN_ALIGN, mStart, screenAlign, FOREVER);
    Indent(level);
   MSTREAMPRINTF  _T("DEF %sBB Billboard {\n"), mNodes.GetNodeName(node));
   if (screenAlign)
   {
       Indent(level + 1);
      MSTREAMPRINTF  _T("axisOfRotation 0 0 0\n"));
   }
   else
   {
       Point3 axis(0, 0, 1);
       Indent(level + 1);
      MSTREAMPRINTF  _T("axisOfRotation %s\n"), point(axis));
   }
   Indent(level + 1);
   MSTREAMPRINTF  _T("children [\n"));

   return TRUE;
}

void
VRML2Export::VrmlOutTimeSensor(INode *node, TimeSensorObject *obj, int level)
{

    if (!doExport(node))
        return;
    int start, end, loop, startOnLoad;
    float cycleInterval;
    int animEnd = mIp->GetAnimRange().End();
    obj->pblock->GetValue(PB_START_TIME, mStart, start, FOREVER);
    obj->pblock->GetValue(PB_STOP_TIME, mStart, end, FOREVER);
    obj->pblock->GetValue(PB_LOOP, mStart, loop, FOREVER);
    obj->pblock->GetValue(PB_START_ON_LOAD, mStart, startOnLoad, FOREVER);
    obj->pblock->GetValue(PB_CYCLEINTERVAL, mStart, cycleInterval, FOREVER);
    obj->needsScript = start != mStart || end != animEnd;

    Indent(level);
   MSTREAMPRINTF  _T("DEF %s-TIMER TimeSensor {\n"),mNodes.GetNodeName(node));
   Indent(level + 1);
   MSTREAMPRINTF  _T("cycleInterval %s\n"), floatVal(cycleInterval));
   Indent(level + 1);
   MSTREAMPRINTF  _T("loop %s\n"), loop ? _T("TRUE") : _T("FALSE"));
   Indent(level + 1);
   if (startOnLoad)
      MSTREAMPRINTF  _T("startTime 1\n"));
   else
      MSTREAMPRINTF  _T("stopTime 1\n"));
   MSTREAMPRINTF  _T("}\n"));
   if (obj->needsScript)
   {
      MSTREAMPRINTF  _T("DEF %s-SCRIPT Script {\n"),mNodes.GetNodeName(node));
      Indent(1);
      MSTREAMPRINTF  _T("eventIn SFFloat fractionIn\n"));
      Indent(1);
      MSTREAMPRINTF  _T("eventOut SFFloat fractionOut\n"));
      Indent(1);
      MSTREAMPRINTF  _T("url \"javascript:\n"));
      Indent(2);
      MSTREAMPRINTF  _T("function fractionIn(i) {\n"));
      Indent(2);
      float fract = (float(end) - float(start)) / (float(animEnd) - float(mStart));
      float offset = (start - mStart) / (float(animEnd) - float(mStart));
      MSTREAMPRINTF  _T("fractionOut = %s * i"), floatVal(fract));
      if (offset != 0.0f)
         MSTREAMPRINTF  _T(" + %s;\n"), floatVal(offset));
      else
         MSTREAMPRINTF  _T(";\n"));
      Indent(1);
      MSTREAMPRINTF  _T("}\"\n"));
      MSTREAMPRINTF  _T("}\n"));
   }

   obj->vrmlWritten = TRUE;
}

BOOL
VRML2Export::VrmlOutPointLight(INode *node, LightObject *light, int level)
{
    LightState ls;
    Interval iv = FOREVER;
    if (!mExpLights)
        return FALSE;
    light->EvalLightState(mStart, iv, &ls);

    Indent(level);
   MSTREAMPRINTF  _T("DEF %s-LIGHT PointLight {\n"), mNodes.GetNodeName(node));
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
   Indent(level + 1);
   MSTREAMPRINTF  _T("radius %s\n"), floatVal(ls.attenEnd));
   if (ls.useAtten)
   {
       Indent(level + 1);
      MSTREAMPRINTF  _T("attenuation 0 1 0\n"));
   }
   Indent(level);
   MSTREAMPRINTF  _T("}\n"));
   return TRUE;
}

BOOL
VRML2Export::VrmlOutDirectLight(INode *node, LightObject *light, int level)
{
    Point3 dir(0, 0, -1);

    if (!mExpLights)
        return FALSE;
    LightState ls;
    Interval iv = FOREVER;

    light->EvalLightState(mStart, iv, &ls);

    Indent(level);
   MSTREAMPRINTF  _T("DEF %s-LIGHT DirectionalLight {\n"), mNodes.GetNodeName(node));
   Indent(level + 1);
   MSTREAMPRINTF  _T("intensity %s\n"),
      floatVal(light->GetIntensity(mStart, FOREVER)));
   Indent(level + 1);
   MSTREAMPRINTF  _T("direction %s\n"), normPoint(dir));
   Indent(level + 1);
   Point3 col = light->GetRGBColor(mStart, FOREVER);

   MSTREAMPRINTF  _T("color %s\n"), color(col));

   Indent(level + 1);
   MSTREAMPRINTF  _T("on %s\n"), ls.on ? _T("TRUE") : _T("FALSE"));
   Indent(level);
   MSTREAMPRINTF  _T("}\n"));
   return TRUE;
}

BOOL
VRML2Export::VrmlOutSpotLight(INode *node, LightObject *light, int level)
{
    LightState ls;
    Interval iv = FOREVER;

    if (!mExpLights)
        return FALSE;
    Point3 dir(0, 0, -1);

    light->EvalLightState(mStart, iv, &ls);
    Indent(level);
   MSTREAMPRINTF  _T("DEF %s-LIGHT SpotLight {\n"), mNodes.GetNodeName(node));
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
   MSTREAMPRINTF  _T("beamWidth %s\n"), floatVal(DegToRad(ls.hotsize)));
   Indent(level + 1);
   MSTREAMPRINTF  _T("on %s\n"), ls.on ? _T("TRUE") : _T("FALSE"));
   Indent(level + 1);
   MSTREAMPRINTF  _T("radius %s\n"), floatVal(ls.attenEnd));
   if (ls.useAtten)
   {
       Indent(level + 1);
      MSTREAMPRINTF  _T("attenuation 0 1 0\n"));
   }
   Indent(level);
   MSTREAMPRINTF  _T("}\n"));
   return TRUE;
}

BOOL
VRML2Export::VrmlOutTopPointLight(INode *node, LightObject *light)
{
    LightState ls;
    Interval iv = FOREVER;

    if (!mExpLights)
        return FALSE;
    light->EvalLightState(mStart, iv, &ls);

   MSTREAMPRINTF  _T("DEF %s PointLight {\n"), mNodes.GetNodeName(node));
   Indent(1);
   MSTREAMPRINTF  _T("intensity %s\n"),
      floatVal(light->GetIntensity(mStart, FOREVER)));
   Indent(1);
   Point3 col = light->GetRGBColor(mStart, FOREVER);
   MSTREAMPRINTF  _T("color %s\n"), color(col));
   Indent(1);
   Point3 p = node->GetObjTMAfterWSM(mStart).GetTrans();
   MSTREAMPRINTF  _T("location %s\n"), point(p));

   Indent(1);
   MSTREAMPRINTF  _T("on %s\n"), ls.on ? _T("TRUE") : _T("FALSE"));
   Indent(1);
   float radius;
   if (!ls.useAtten || ls.attenEnd == 0.0f)
       radius = Length(mBoundBox.Width());
   else
       radius = ls.attenEnd;
   MSTREAMPRINTF  _T("radius %s\n"), floatVal(radius));
   if (ls.useAtten)
   {
       Indent(1);
      MSTREAMPRINTF  _T("attenuation 0 1 0\n"));
   }
   MSTREAMPRINTF  _T("}\n"));
   return TRUE;
}

BOOL
VRML2Export::VrmlOutTopDirectLight(INode *node, LightObject *light)
{
    LightState ls;
    Interval iv = FOREVER;

    if (!mExpLights)
        return FALSE;
    light->EvalLightState(mStart, iv, &ls);

   MSTREAMPRINTF  _T("DEF %s DirectionalLight {\n"), mNodes.GetNodeName(node));
   Indent(1);
   MSTREAMPRINTF  _T("intensity %s\n"),
      floatVal(light->GetIntensity(mStart, FOREVER)));
   Indent(1);
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

   Indent(1);
   MSTREAMPRINTF  _T("direction %s\n"), point(p));
   Indent(1);
   MSTREAMPRINTF  _T("on %s\n"), ls.on ? _T("TRUE") : _T("FALSE"));
   MSTREAMPRINTF  _T("}\n"));
   return TRUE;
}

BOOL
VRML2Export::VrmlOutTopSpotLight(INode *node, LightObject *light)
{
    LightState ls;
    Interval iv = FOREVER;

    if (!mExpLights)
        return FALSE;
    light->EvalLightState(mStart, iv, &ls);
   MSTREAMPRINTF  _T("DEF %s SpotLight {\n"), mNodes.GetNodeName(node));
   Indent(1);
   MSTREAMPRINTF  _T("intensity %s\n"),
      floatVal(light->GetIntensity(mStart,FOREVER)));
   Indent(1);
   Point3 col = light->GetRGBColor(mStart, FOREVER);
   MSTREAMPRINTF  _T("color %s\n"), color(col));
   Indent(1);
   Point3 p = node->GetObjTMAfterWSM(mStart).GetTrans();
   MSTREAMPRINTF  _T("location %s\n"), point(p));

   Matrix3 tm = node->GetObjTMAfterWSM(mStart);
   p = Point3(0, 0, -1);
   Point3 trans, s;
   Quat q;
   Matrix3 rot;
   AffineParts parts;
   decomp_affine(tm, &parts);
   q = parts.q;
   q.MakeMatrix(rot);
   p = p * rot;

   Indent(1);
   MSTREAMPRINTF  _T("direction %s\n"), normPoint(p));
   Indent(1);
   MSTREAMPRINTF  _T("cutOffAngle %s\n"),
      floatVal(DegToRad(ls.fallsize)));
   Indent(1);
   MSTREAMPRINTF  _T("beamWidth %s\n"), floatVal(DegToRad(ls.hotsize)));
   Indent(1);
   MSTREAMPRINTF  _T("on %s\n"), ls.on ? _T("TRUE") : _T("FALSE"));
   Indent(1);
   float radius;
   if (!ls.useAtten || ls.attenEnd == 0.0f)
       radius = Length(mBoundBox.Width());
   else
       radius = ls.attenEnd;
   MSTREAMPRINTF  _T("radius %s\n"), floatVal(radius));
   if (ls.useAtten)
   {
       float attn;
       attn = (ls.attenStart <= 1.0f) ? 1.0f : 1.0f / ls.attenStart;
       Indent(1);
      MSTREAMPRINTF  _T("attenuation 0 %s 0\n"), floatVal(attn));
   }
   MSTREAMPRINTF  _T("}\n"));
   return TRUE;
}

void
VRML2Export::OutputTopLevelLight(INode *node, LightObject *light)
{
    Class_ID id = light->ClassID();
    if (id == Class_ID(OMNI_LIGHT_CLASS_ID, 0))
        VrmlOutTopPointLight(node, light);
    else if (id == Class_ID(DIR_LIGHT_CLASS_ID, 0) || id == Class_ID(TDIR_LIGHT_CLASS_ID, 0))
        VrmlOutTopDirectLight(node, light);
    else if (id == Class_ID(SPOT_LIGHT_CLASS_ID, 0) || id == Class_ID(FSPOT_LIGHT_CLASS_ID, 0))
        VrmlOutTopSpotLight(node, light);
    else
        return;

    // Write out any animation data
    InitInterpolators(node);
    VrmlOutControllers(node, 0);
    WriteInterpolatorRoutes(0);
}

// Output a VRML Inline node.
BOOL
VRML2Export::VrmlOutInline(VRMLInsObject *obj, int level)
{
    const TCHAR *url = obj->GetUrl().data();
    Indent(level);
   MSTREAMPRINTF  _T("Inline {\n"));
   Indent(level + 1);
   if (url && url[0] == '"')
      MSTREAMPRINTF  _T("url %s\n"), url);
   else
      MSTREAMPRINTF  _T("url \"%s\"\n"), url);
   if (obj->GetUseSize())
   {
       float size = obj->GetSize() * 2.0f;
       Indent(level + 1);
      MSTREAMPRINTF  _T("bboxSize %s\n"),
         scalePoint(Point3(size, size, size)));
   }
   Indent(level);
   MSTREAMPRINTF  _T("}\n"));
   return TRUE;
}

// Output a VRML COVISEObjecz node.
BOOL
VRML2Export::VrmlOutCOVISEObject(VRMLCOVISEObjectObject *obj, int level)
{
    const TCHAR *url = obj->GetUrl().data();
    Indent(level);
   MSTREAMPRINTF  _T("COVISEObject {\n"));
   Indent(level + 1);
   if (url && url[0] == '"')
      MSTREAMPRINTF  _T("objectName %s\n"), url);
   else
      MSTREAMPRINTF  _T("objectName \"%s\"\n"), url);
   Indent(level);
   MSTREAMPRINTF  _T("}\n"));
   return TRUE;
}

// Output a VRML Cal3d node.
BOOL
VRML2Export::VrmlOutCal3D(Cal3DObject *obj, int level)
{
#ifndef NO_CAL3D
    const TCHAR *url = obj->GetUrl().data();
    Indent(level);
   MSTREAMPRINTF  _T("Cal3DNode {\n"));
   if (Cal3DCoreHelper *core = obj->getCoreHelper())
   {
       if (core->wasWritten())
       {
           Indent(level + 1);
         MSTREAMPRINTF  _T("core USE %s\n"), core->getVRMLName().c_str());
       }
       else
       {
           Indent(level + 1);
         MSTREAMPRINTF  _T("core DEF %s Cal3DCore \n"), core->getVRMLName().c_str());
         Indent(level + 1);
         MSTREAMPRINTF  _T("{\n"));
         Indent(level + 2);
         MSTREAMPRINTF  _T("modelName \"%s\"\n"), core->getName().c_str());
         Indent(level + 2);
         MSTREAMPRINTF  _T("scale %s\n"), floatVal(obj->GetSize()));
         Indent(level + 1);
         MSTREAMPRINTF  _T("}\n"));
         core->setWritten();
       }
   }
   Indent(level + 1);
   MSTREAMPRINTF  _T("animationId 0\n"));

   Indent(level);
   MSTREAMPRINTF  _T("}\n"));
#endif
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
VRML2Export::OutputMaxLOD(INode *node, Object *obj, int level, int numLevels, float *distances, INode **children, int numChildren, BOOL mirrored)
{
    int i, n, m;

    Indent(level);
   MSTREAMPRINTF  _T("DEF %s_LOD LOD {\n"), mNodes.GetNodeName(node));
   Indent(level + 1);

   float curDist = 0;
   INode *lodChildren[1000];
   float lodDistances[1000];
   int num = 0;
   // sort the distances and nodes
   for (i = 0; i < numChildren; i++)
   {
       LODCtrl *visibility = (LODCtrl *)children[i]->GetVisController();
       if (visibility)
       {
           for (n = 0; n < num; n++)
           {
               if (lodDistances[n] > visibility->max)
               {
                   // insert it here
                   for (m = num; m >= n; m--)
                   {
                       lodDistances[m + 1] = lodDistances[m];
                       lodChildren[m + 1] = lodChildren[m];
                   }
                   lodDistances[n] = visibility->max;
                   lodChildren[n] = children[i];
                   break;
               }
           }
           if (n == num)
           {
               lodDistances[n] = visibility->max;
               lodChildren[n] = children[i];
           }
           num++;
       }
   }

   MSTREAMPRINTF  _T("range [ "));
   for (i = 0; i < numLevels - 1; i++)
   {
       if (i < numLevels - 2)
         MSTREAMPRINTF  _T("%s, "), floatVal(distances[i]));
       else
         MSTREAMPRINTF  _T("%s "), floatVal(distances[i]));
   }
   MSTREAMPRINTF  _T("]\n"));

   Indent(level + 1);
   MSTREAMPRINTF  _T("center 0 0 0\n"));

   MSTREAMPRINTF  _T("level [\n"));
   for (i = 0; i < num; i++)
   {
       INode *node = lodChildren[i];
       INode *parent = node->GetParentNode();
       VrmlOutNode(node, parent, level + 1, TRUE, FALSE, mirrored);
       if (i != num - 1)
       {
           Indent(level);
         MSTREAMPRINTF  _T(",\n"));
       }
   }

   //    if (numLod > 1) {
   Indent(level);
   MSTREAMPRINTF  _T("]\n"));
   Indent(level);
   MSTREAMPRINTF  _T("}\n"));
   //    }

   return TRUE;
}

// Create a level-of-detail object.
BOOL
VRML2Export::VrmlOutLOD(INode *node, LODObject *obj, int level, BOOL mirrored)
{
    int numLod = obj->NumRefs();
    Tab<LODObj *> lodObjects = obj->GetLODObjects();
    int i;

    if (numLod == 0)
        return TRUE;

    lodObjects.Sort((CompareFnc)DistComp);

    //    if (numLod > 1) {
    Indent(level);
   MSTREAMPRINTF  _T("DEF %s_LOD LOD {\n"), mNodes.GetNodeName(node));
   Indent(level + 1);
   //Point3 p = node->GetObjTMAfterWSM(mStart).GetTrans();
   // check this but 0 0 0 is better than this MSTREAMPRINTF  _T("center %s\n"), point(p));
   //MSTREAMPRINTF  _T("center 0 0 0\n"), point(p));
   //MSTREAMPRINTF  _T("center %s\n"), point(p));

   Matrix3 tm = GetLocalTM(node, mStart);

   BOOL isIdentity = TRUE;
   for (int i = 0; i < 3; i++)
   {
       for (int j = 0; j < 3; j++)
       {
           if (i == j)
           {
               if (tm.GetRow(i)[j] != 1.0)
                   isIdentity = FALSE;
           }
           else if (fabs(tm.GetRow(i)[j]) > 0.00001)
               isIdentity = FALSE;
       }
   }
   Point3 p;
   if (isIdentity)
   {
       p = tm.GetTrans();
   }
   else
   {
       AffineParts parts;
#ifdef DDECOMP
       d_decomp_affine(tm, &parts);
#else
       decomp_affine(tm, &parts); // parts is parts
#endif
       p = parts.t;
   }
   MSTREAMPRINTF  _T("center %s\n"), point(p));

   Indent(level + 1);
   MSTREAMPRINTF  _T("range [ "));
   for (i = 0; i < numLod - 1; i++)
   {
       if (i < numLod - 2)
         MSTREAMPRINTF  _T("%s, "), floatVal(lodObjects[i]->dist));
       else
           //                MSTREAMPRINTF  _T("%s ]\n"), floatVal(lodObjects[i]->dist));
         MSTREAMPRINTF  _T("%s "), floatVal(lodObjects[i]->dist));
   }
   MSTREAMPRINTF  _T("]\n"));
   //    }

   Indent(level + 1);
   if (mType == Export_X3D_V)
   {
      MSTREAMPRINTF  _T("children [\n"));
   }
   else
   {
      MSTREAMPRINTF  _T("level [\n"));
   }
   for (i = 0; i < numLod; i++)
   {
       INode *node = lodObjects[i]->node;
       if (node)
       {
           INode *parent = node->GetParentNode();
           VrmlOutNode(node, parent, level + 1, TRUE, FALSE, mirrored);
           if (i != numLod - 1)
           {
               Indent(level);
            MSTREAMPRINTF  _T(",\n"));
           }
       }
   }

   //    if (numLod > 1) {
   Indent(level);
   MSTREAMPRINTF  _T("]\n"));
   Indent(level);
   MSTREAMPRINTF  _T("}\n"));
   //    }

   return TRUE;
}

BOOL
VRML2Export::VrmlOutSpecialTform(INode *node, Object *obj, int level,
                                 BOOL mirrored)
{
    if (!mPrimitives)
        return FALSE;

    Class_ID id = obj->ClassID();

    // Otherwise look for the primitives we know about
    if (id == Class_ID(CYLINDER_CLASS_ID, 0))
        return VrmlOutCylinderTform(node, obj, level + 1, mirrored);

    if (id == Class_ID(CONE_CLASS_ID, 0))
        return VrmlOutConeTform(node, obj, level + 1, mirrored);

    if (id == Class_ID(BOXOBJ_CLASS_ID, 0))
        return VrmlOutCubeTform(node, obj, level + 1, mirrored);

    return FALSE;
}

BOOL
VRML2Export::ObjIsPrim(INode *node, Object *obj)
{
    Class_ID id = obj->ClassID();
    if (id == Class_ID(SPHERE_CLASS_ID, 0))
        return VrmlOutSphereTest(node, obj);

    if (id == Class_ID(CYLINDER_CLASS_ID, 0))
        return VrmlOutCylinderTest(node, obj);

    if (id == Class_ID(CONE_CLASS_ID, 0))
        return VrmlOutConeTest(node, obj);

    if (id == Class_ID(BOXOBJ_CLASS_ID, 0))
        return VrmlOutCubeTest(node, obj);

    return FALSE;
}

// Write out the VRML for node we know about, including Opus nodes,
// lights, cameras and VRML primitives
BOOL
VRML2Export::VrmlOutSpecial(INode *node, INode *parent,
                            Object *obj, int level, BOOL mirrored)
{
    Class_ID id = obj->ClassID();

    /* test
   if (id == Class_ID(MR_BLUE_CLASS_ID1, MR_BLUE_CLASS_ID2)) {
   level++;
   VrmlOutMrBlue(node, parent, (MrBlueObject*) obj,
   &level, FALSE);
   return TRUE;
   }
   */

    if (id == Class_ID(OMNI_LIGHT_CLASS_ID, 0))
        return VrmlOutPointLight(node, (LightObject *)obj, level + 1);

    if (id == Class_ID(DIR_LIGHT_CLASS_ID, 0) || id == Class_ID(TDIR_LIGHT_CLASS_ID, 0))
        return VrmlOutDirectLight(node, (LightObject *)obj, level + 1);

    if (id == Class_ID(SPOT_LIGHT_CLASS_ID, 0) || id == Class_ID(FSPOT_LIGHT_CLASS_ID, 0))
        return VrmlOutSpotLight(node, (LightObject *)obj, level + 1);

    if (id == Class_ID(VRML_INS_CLASS_ID1, VRML_INS_CLASS_ID2))
        return VrmlOutInline((VRMLInsObject *)obj, level + 1);

    if (id == Class_ID(VRML_COVISEOOBJECT_CLASS_ID1, VRML_COVISEOOBJECT_CLASS_ID2))
        return VrmlOutCOVISEObject((VRMLCOVISEObjectObject *)obj, level + 1);
#ifndef NO_CAL3D
    if (id == Class_ID(CAL3D_CLASS_ID1, CAL3D_CLASS_ID2))
        return VrmlOutCal3D((Cal3DObject *)obj, level + 1);
#endif

    if (id == Class_ID(VRML_SCRIPT_CLASS_ID1, VRML_SCRIPT_CLASS_ID2))
    {
        mScriptsList = mScriptsList->AddNode(node);
        return TRUE;
    }

    if (id == Class_ID(LOD_CLASS_ID1, LOD_CLASS_ID2))
        return VrmlOutLOD(node, (LODObject *)obj, level + 1, mirrored);

    if (id == Class_ID(SIMPLE_CAM_CLASS_ID, 0) || id == Class_ID(LOOKAT_CAM_CLASS_ID, 0))
        return VrmlOutCamera(node, obj, level + 1);

    if (id == SoundClassID)
        return VrmlOutSound(node, (SoundObject *)obj, level + 1);

    if (id == ProxSensorClassID)
        return VrmlOutProxSensor(node, (ProxSensorObject *)obj, level + 1);

    if (id == BillboardClassID)
        return VrmlOutBillboard(node, obj, level + 1);

    if (id == SwitchClassID)
        return VrmlOutSwitch(node, level + 1);

    // If object has modifiers or WSMs attached, do not output as
    // a primitive
    SClass_ID sid = node->GetObjectRef()->SuperClassID();
    if (sid == WSM_DERIVOB_CLASS_ID || sid == DERIVOB_CLASS_ID)
        return FALSE;

    if (!mPrimitives)
        return FALSE;

    // Otherwise look for the primitives we know about
    if (id == Class_ID(SPHERE_CLASS_ID, 0))
        return VrmlOutSphere(node, obj, level + 1);

    if (id == Class_ID(CYLINDER_CLASS_ID, 0))
        return VrmlOutCylinder(node, obj, level + 1);

    if (id == Class_ID(CONE_CLASS_ID, 0))
        return VrmlOutCone(node, obj, level + 1);

    if (id == Class_ID(BOXOBJ_CLASS_ID, 0))
        return VrmlOutCube(node, obj, level + 1);

    return FALSE;
}

static BOOL
IsLODObject(Object *obj)
{
    return obj->ClassID() == Class_ID(LOD_CLASS_ID1, LOD_CLASS_ID2);
}

static BOOL
IsEverAnimated(INode *node)
{
    if (!node)
        return FALSE;
    for (; !node->IsRootNode(); node = node->GetParentNode())
        if (node->IsAnimated())
            return TRUE;
    return FALSE;
}

BOOL
VRML2Export::ChildIsAnimated(INode *node)
{
    if (node->IsAnimated())
        return TRUE;

    Object *obj = node->EvalWorldState(mStart).obj;

    if (ObjIsAnimated(obj))
        return TRUE;

    Class_ID id = node->GetTMController()->ClassID();

    if (id != Class_ID(PRS_CONTROL_CLASS_ID, 0))
        return TRUE;

    for (int i = 0; i < node->NumberOfChildren(); i++)
        if (ChildIsAnimated(node->GetChildNode(i)))
            return TRUE;
    return FALSE;
}

static BOOL
IsAnimTrigger(Object *obj)
{

    if (!obj)
        return FALSE;

    Class_ID id = obj->ClassID();
    /* test
   // Mr Blue nodes only 1st class if stand-alone
   if (id == Class_ID(MR_BLUE_CLASS_ID1, MR_BLUE_CLASS_ID2)) {
   MrBlueObject* mbo = (MrBlueObject*) obj;
   return mbo->GetMouseEnabled() && mbo->GetAction() == Animate;
   }
   */
    return FALSE;
}

BOOL
VRML2Export::isVrmlObject(INode *node, Object *obj, INode *parent, bool hastVisController)
{
    if (!obj)
        return FALSE;

    Class_ID id = obj->ClassID();

    /* test

   if (id == Class_ID(OMNI_LIGHT_CLASS_ID, 0))
   // Mr Blue nodes only 1st class if stand-alone
   if (id == Class_ID(MR_BLUE_CLASS_ID1, MR_BLUE_CLASS_ID2)) {
   MrBlueObject* mbo = (MrBlueObject*) obj;
   if ((mbo->GetAction() == HyperLinkJump ||
   mbo->GetAction() == SetViewpoint) &&
   mbo->GetMouseEnabled())
   return parent->IsRootNode();
   else
   return FALSE;
   }
   */

    if (id == Class_ID(VRML_INS_CLASS_ID1, VRML_INS_CLASS_ID2) || id == Class_ID(VRML_COVISEOOBJECT_CLASS_ID1, VRML_COVISEOOBJECT_CLASS_ID2) || id == Class_ID(VRML_SCRIPT_CLASS_ID1, VRML_SCRIPT_CLASS_ID2) ||
#ifndef NO_CAL3D
        id == Class_ID(CAL3D_CLASS_ID1, CAL3D_CLASS_ID2) ||
#endif
        id == SoundClassID || id == ProxSensorClassID)
        return TRUE;

    // only animated lights come out in scene graph
    if (IsLight(node))
        return (IsEverAnimated(node) || IsEverAnimated(node->GetTarget()));
    if (IsCamera(node))
        return TRUE;

    SClass_ID sid = obj->SuperClassID();
    if (sid == SHAPE_CLASS_ID)
    {
        return TRUE;
    }

    if (node->NumberOfChildren() > 0)
        return TRUE;

#ifdef _LEC_
    // LEC uses dummies as place holders and need dummy leaves written.
    if (id == Class_ID(DUMMY_CLASS_ID, 0))
        return TRUE;
#endif

    return (obj->IsRenderable() || id == Class_ID(LOD_CLASS_ID1, LOD_CLASS_ID2)) && (doExport(node) || hastVisController);
}

static int
NodeIsChildOf(INode *child, INode *parent, int level)
{
    level++;
    if (child == parent)
        return level;
    // skip invalid nodes (ex. user create the list then delete the node from the scene.)
    if (!parent)
        return -1;
    int num = parent->NumberOfChildren();
    int i;
    for (i = 0; i < num; i++)
    {
        int l = NodeIsChildOf(child, parent->GetChildNode(i), level);
        if (l > 0)
            return l;
    }
    return -1;
}

// For objects that change shape, output a CoodinateInterpolator
void
VRML2Export::VrmlOutCoordinateInterpolator(INode *node, Object *obj,
                                           int level, BOOL pMirror)
{
    int sampleRate;
    int t, i, j;
    size_t width = mIndent ? level * 2 : 0;
    TCHAR name[MAX_PATH];

    if (mCoordSample)
        sampleRate = GetTicksPerFrame();
    else
        sampleRate = TIME_TICKSPERSEC / mCoordSampleRate;

    int end = mIp->GetAnimRange().End();
    int realEnd = end;
    int frames = (end - mStart) / sampleRate + 1;

    if (((end - mStart) % sampleRate) != 0)
    {
        end += sampleRate;
        frames++;
    }
    t = mStart;
    if (t > realEnd)
        t = realEnd;
    Object *o = node->EvalWorldState(t).obj;
    TriObject *tri = (TriObject *)o->ConvertToType(t, triObjectClassID);
    Mesh &mesh = tri->GetMesh();
    int numverts = mesh.getNumVerts();
    Point3 *oldp = new Point3[numverts];
    for (j = 0; j < numverts; j++)
    {
        oldp[j] = mesh.verts[j];
    }
    // check, if it coordinates are actually ánimated
    for (i = 0, t = mStart; i < frames; i++, t += sampleRate)
    {
        if (t > realEnd)
            t = realEnd;
        Object *o = node->EvalWorldState(t).obj;
        TriObject *tri = (TriObject *)o->ConvertToType(t, triObjectClassID);
        Mesh &mesh = tri->GetMesh();

        int numverts = mesh.getNumVerts();
        for (j = 0; j < numverts; j++)
        {
            if (mesh.verts[j] != oldp[j])
                break;
        }
        if (j < numverts)
            break;
        if (i == frames - 1 && j == numverts)
            return;

        if (o != (Object *)tri)
            tri->DeleteThis();
    }
    delete[] oldp;
    Indent(level);
    _stprintf(name, _T("%s-COORD-INTERP"), mNodes.GetNodeName(node));
   MSTREAMPRINTF  _T("DEF %s CoordinateInterpolator {\n"), name);
   bool foundTimeSensor = false;
   // Now check to see if a TimeSensor references this node
   int mindistance = 1000000;
   int minTS = -1;
   INodeList *l;
   INodeList *minl;

   for (l = mTimerList; l; l = l->GetNext())
   {
       TimeSensorObject *tso = (TimeSensorObject *)
                                   l->GetNode()->EvalWorldState(mStart)
                                       .obj;

       // find the timesensor closest to the node in the hierarchy
       for (int j = 0; j < tso->TimeSensorObjects.Count(); j++)
       {
           INode *anim = tso->TimeSensorObjects[j]->node;
           if (anim)
           {
               int dist = NodeIsChildOf(node, anim, 0);
               if (dist >= 0) // we have a timesensor
               {
                   if (dist < mindistance) // it animates a group closer to the node we want to animate than the last one
                   {
                       minTS = j;
                       minl = l;
                       mindistance = dist;
                   }
               }
           }
       }
   }
   if (minTS >= 0) // now add all Timesensors with same distance
   {
       for (l = mTimerList; l; l = l->GetNext())
       {
           TimeSensorObject *tso = (TimeSensorObject *)
                                       l->GetNode()->EvalWorldState(mStart)
                                           .obj;

           // find the timesensor closest to the node in the hierarchy
           for (int j = 0; j < tso->TimeSensorObjects.Count(); j++)
           {
               INode *anim = tso->TimeSensorObjects[j]->node;
               if (anim)
               {
                   int dist = NodeIsChildOf(node, anim, 0);
                   if (dist >= 0) // we have a timesensor
                   {
                       if (dist == mindistance) // it animates a group closer to the node we want to animate than the last one
                       {

                           TSTR oTimer = mTimer;
                           TCHAR timer[MAX_PATH];
                           _stprintf(timer, _T("%s"), mNodes.GetNodeName(l->GetNode()));
                           foundTimeSensor = true;
                           if (tso->needsScript)
                               AddInterpolator(name, KEY_TIMER_SCRIPT, timer, l->GetNode());
                           else
                               AddInterpolator(name, KEY_TIMER, timer, l->GetNode());
                       }
                   }
               }
           }
       }
   }
   //if(!foundTimeSensor)
   AddInterpolator(name, KEY_COORD, mNodes.GetNodeName(node), node);

   Indent(level + 1);
   MSTREAMPRINTF  _T("key ["));
   mCycleInterval = (mIp->GetAnimRange().End() - mStart) / ((float)GetTicksPerFrame() * GetFrameRate());

   for (i = 0, t = mStart; i < frames; i++, t += sampleRate)
   {
       if (t > realEnd)
           t = realEnd;
      width += MSTREAMPRINTF  _T("%s, "),
         floatVal(t / ((float) GetTicksPerFrame()
         * GetFrameRate() * mCycleInterval)));
      if (width > 60)
      {
         MSTREAMPRINTF  _T("\n"));
         Indent(level + 3);
         width = mIndent ? level * 2 : 0;
      }
   }
   MSTREAMPRINTF  _T("]\n"));

   Indent(level + 1);
   MSTREAMPRINTF  _T("keyValue ["));

   // Now output the values for the interpolator
   for (i = 0, t = mStart; i < frames; i++, t += sampleRate)
   {
       if (t > realEnd)
           t = realEnd;
       Object *o = node->EvalWorldState(t).obj;
       TriObject *tri = (TriObject *)o->ConvertToType(t, triObjectClassID);
       Mesh &mesh = tri->GetMesh();

       int numverts = mesh.getNumVerts();
       for (j = 0; j < numverts; j++)
       {
           Point3 p = mesh.verts[j];
#ifdef MIRROR_BY_VERTICES
           if (pMirror)
               p = -p;
#endif
         width += MSTREAMPRINTF  _T("%s, "), point(p));
         if (width > 60)
         {
            MSTREAMPRINTF  _T("\n"));
            Indent(level + 3);
            width = mIndent ? level * 2 : 0;
         }
       }

       if (o != (Object *)tri)
           tri->DeleteThis();
   }
   MSTREAMPRINTF  _T("]\n"));
   Indent(level + 1);
   MSTREAMPRINTF  _T("}\n"));

   // get valid mStart object
   obj = node->EvalWorldState(mStart).obj;
}

BOOL
VRML2Export::ObjIsAnimated(Object *obj)
{
    if (!obj)
        return FALSE;
    Interval iv = obj->ObjectValidity(mStart);
    return !(iv == FOREVER);
}

static BOOL
MtlHasTexture(Mtl *mtl)
{
    if (mtl && mtl->ClassID() == Class_ID(BAKE_SHELL_CLASS_ID, 0))
    {
        mtl = mtl->GetSubMtl(1);
    }
    if (mtl->ClassID() != Class_ID(DMTL_CLASS_ID, 0))
        return FALSE;

    StdMat *sm = (StdMat *)mtl;
    // Check for texture mapint id;
    int id;
    Texmap *tm;
    for (id = ID_DI; id <= ID_DP; id++)
    {
        if (id != ID_OP)
        {
            tm = (BitmapTex *)sm->GetSubTexmap(id);
        }
        if (tm)
            break;
    }
    if (!tm)
        return FALSE;

    if (tm->ClassID() == Class_ID(ACUBIC_CLASS_ID, 0))
        return TRUE;
    if (tm->ClassID() != Class_ID(BMTEX_CLASS_ID, 0))
        return FALSE;
    BitmapTex *bm = (BitmapTex *)tm;

    TSTR bitmapFile;

    bitmapFile = bm->GetMapName();
    if (bitmapFile.data() == NULL)
        return FALSE;
    ////int l = strlen(bitmapFile)-1;
    int l = bitmapFile.Length() - 1;
    if (l < 0)
        return FALSE;

    return TRUE;
}
static bool
hasSubMaterial(INode *node, int i)
{
    if (i < 0)
        return true;
    //float firstxpar;
    Mtl /* *sub,*/ *mtl = node->GetMtl();
    if (!mtl)
        return false;
    Class_ID cid = mtl->ClassID();
    if (mtl && mtl->ClassID() == Class_ID(BAKE_SHELL_CLASS_ID, 0))
    {
        mtl = mtl->GetSubMtl(1);
    }
    if (!mtl->IsMultiMtl())
        return false;
    int num = mtl->NumSubMtls();
    if (i <= num)
        return (mtl->GetSubMtl(i) != NULL);
    return false;
    /* this is for multi sub materials, they should work also for non textured materials
   bool firstTime=true;
   for(int i = 0; i < num; i++) {
      sub = mtl->GetSubMtl(i);
      if(sub && sub->ClassID() == Class_ID(BAKE_SHELL_CLASS_ID,0))
      {
         sub = sub->GetSubMtl(1);
      }
      if (!sub)
         continue;
      if (firstTime)
	  {
		  firstxpar = sub->GetXParency();
		  firstTime=false;
	  }
      if (MtlHasTexture(sub))
         return num;
      else if (sub->GetXParency() != firstxpar)
         return num;
   }
   return 0;*/
}

static int
NumMaterials(INode *node)
{
    //float firstxpar;
    Mtl /* *sub,*/ *mtl = node->GetMtl();
    if (!mtl)
        return 0;
    Class_ID cid = mtl->ClassID();
    if (mtl && mtl->ClassID() == Class_ID(BAKE_SHELL_CLASS_ID, 0))
    {
        mtl = mtl->GetSubMtl(1);
    }
    if (!mtl->IsMultiMtl())
        return 0;
    int num = mtl->NumSubMtls();
    return num;
    /* this is for multi sub materials, they should work also for non textured materials
   bool firstTime=true;
   for(int i = 0; i < num; i++) {
      sub = mtl->GetSubMtl(i);
      if(sub && sub->ClassID() == Class_ID(BAKE_SHELL_CLASS_ID,0))
      {
         sub = sub->GetSubMtl(1);
      }
      if (!sub)
         continue;
      if (firstTime)
	  {
		  firstxpar = sub->GetXParency();
		  firstTime=false;
	  }
      if (MtlHasTexture(sub))
         return num;
      else if (sub->GetXParency() != firstxpar)
         return num;
   }
   return 0;*/
}

// Write the data for a single object.
// This function also takes care of identifying VRML primitive objects
void
VRML2Export::VrmlOutObject(INode *node, INode *parent, Object *obj, int level,
                           BOOL mirrored)
{
    // need to get a valid obj ptr
    obj = node->EvalWorldState(mStart).obj;
    BOOL isTriMesh = obj->CanConvertToType(triObjectClassID);
    BOOL instance;
    BOOL special = FALSE;
    int numTextures = NumMaterials(node);
    int start, end;

    if (numTextures == 0)
    {
        start = -1;
        end = 0;
    }
    else
    {
        start = 0;
        end = numTextures;
    }

    bool ShapeWritten = false;

    ObjectBucket *ob = mObjTable.AddObject(obj);
    if ((numTextures > 1 && ob->numInstances > 0) && (ob->objectUsed != TRUE || mirrored != ob->instMirrored))
    {
        Indent(level);

        ob->instName.printf(_T("%s%d"), mNodes.GetNodeName(node), uniqueNumber++);
          MSTREAMPRINTF  _T("DEF %s-ShapeGroup Group { children [\n"),
		  ob->instName);
          level += 3;
    }

    instance = FALSE;
    // we should check, if we have an instance but another material, then we have to write the node anyway and can´t use it
    TriObject *tri = NULL;
    if (isTriMesh && obj->IsRenderable())
    {
        tri = (TriObject *)obj->ConvertToType(mStart, triObjectClassID);
    }

    if (!(numTextures > 1 && ob->numInstances > 0) || (ob->objectUsed != TRUE || mirrored != ob->instMirrored))
    {
        ShapeWritten = true;
        int old_level = level;
        CoordsWritten = false;
        int i;
        for (i = 0; i < 20; i++)
            TexCoordsWritten[i] = false;
        for (int i = start; i < end; i++)
        {
            if (!hasSubMaterial(node, i))
            { // skip non existant materials
                continue;
            }

            if (isTriMesh && obj->IsRenderable())
            {

                if (!hasMaterial(tri, i))
                    continue;

                special = VrmlOutSpecialTform(node, obj, level, mirrored);
                if (special)
                    level += 3;
                Indent(level);
                if (numTextures > 1 && ob->numInstances > 0)
                {

                                   MSTREAMPRINTF  _T("DEF Shape-%s-%d Shape{\n"),mNodes.GetNodeName(node),i);
                }
                else
                {
                    if (!ob->objectUsed || (mirrored != ob->instMirrored))
                    {
                        if (ob->numInstances < 1)
                        {
                            if (numTextures >= 1)
                            {
                                                           MSTREAMPRINTF  _T("DEF %s-%d-SHAPE Shape {\n"),mNodes.GetNodeName(node),i);
                            }
                            else
                            {MSTREAMPRINTF  _T("Shape {\n"));
                            }
                        }
                        else
                        {
                            if (i >= 1)
                            {
                                                           MSTREAMPRINTF  _T("DEF %s-%d-SHAPE Shape {\n"),mNodes.GetNodeName(node),i);
                            }
                            else
                            {
                                ob->instName.printf(_T("%s%d"), mNodes.GetNodeName(node), uniqueNumber++);
                                                           MSTREAMPRINTF  _T("DEF %s-SHAPE Shape {\n"),ob->instName);
                            }
                        }
                        // fprintf (mStream, _T("Shape {\n"));
                    }
                }
            }

            BOOL multiMat = FALSE;
            BOOL isWire = FALSE, twoSided = FALSE;

            // Output the material
            if (isTriMesh && obj->IsRenderable()) // if not trimesh, needs no matl
            {
                if (!ob->objectUsed || (mirrored != ob->instMirrored))
                {
                    multiMat = OutputMaterial(node, isWire, twoSided, level + 1, i);
                }
            }

            // First check for VRML primitives and other special objects
            if ((special || ObjIsPrim(node, obj)) && (ob->objectUsed && mirrored == ob->instMirrored))
            {
                Indent(level);
                           MSTREAMPRINTF  _T("USE %s-SHAPE \n"),
				   ob->instName);
                           instance = TRUE;
                           if (special)
                           {
                               level = old_level;
                               Indent(level);
                                   MSTREAMPRINTF  _T("] }\n"));
                           }
                           continue;
            }
            if (VrmlOutSpecial(node, parent, obj, level, mirrored))
            {
                if (isTriMesh && obj->IsRenderable())
                {
                    if (!ob->objectUsed || (mirrored != ob->instMirrored))
                    {
                        Indent(level);
                                           MSTREAMPRINTF  _T("}\n"));
                                           if (special)
                                           {
                                               level = old_level;
                                               Indent(level);
                                                   MSTREAMPRINTF  _T("] }\n"));
                                           }
                                           ob->objectUsed = TRUE;
                                           ob->instMirrored = mirrored;
                    }
                    else
                    {
                        Indent(level);
                                           MSTREAMPRINTF  _T("USE %s-SHAPE \n"),
						   ob->instName);
                                           instance = TRUE;
                                           if (special)
                                           {
                                               level = old_level;
                                               Indent(level);
                                                   MSTREAMPRINTF  _T("] }\n"));
                                           }
                    }
                }
                continue;
            }

            // Otherwise output as a triangle mesh
            if (isTriMesh && obj->IsRenderable())
            {
                if (!ob->objectUsed || (mirrored != ob->instMirrored))
                {
#ifdef GEOMETRY_REUSE_HAST_TO_BE_REIMPLEMENTED
                    if (ob->objectUsed && i == -1)
                    {
                        instance = TRUE;
                        // We have an instance
                        Indent(level);
                                           MSTREAMPRINTF  _T("geometry USE %s-FACES\n"),
						   ob->instName.data());
                    }
                    else
                    {
                        if (!(numTextures > 1 && ob->numInstances > 0))
                        {
                            ob->objectUsed = TRUE;
                            ob->instName = mNodes.GetNodeName(node);
                            ob->instMirrored = mirrored
                        }
                        instance = FALSE;
#endif
                        if (tri == NULL)
                            tri = (TriObject *)obj->ConvertToType(mStart, triObjectClassID);

                        if (mPolygonType && !ObjIsAnimated(obj))
                            OutputPolygonObject(node, tri, multiMat, isWire,
                                                twoSided, level + 1, i, mirrored);
                        else
                            OutputTriObject(node, tri, multiMat, isWire, twoSided,
                                            level + 1, i, mirrored);

#ifndef FUNNY_TEST
                        if (obj != (Object *)tri)
                            tri->DeleteThis();
#endif
#ifdef GEOMETRY_REUSE_HAST_TO_BE_REIMPLEMENTED
                    }
#endif
                    Indent(level);
                                   MSTREAMPRINTF  _T("}\n"));

                                   if (numTextures == 0)
                                   {
                                       ob->objectUsed = TRUE;
                                       ob->instMirrored = mirrored;
                                   }
                }
                else
                {
                    Indent(level);
                                   MSTREAMPRINTF  _T("USE %s-SHAPE \n"),
					   ob->instName);
                                   instance = TRUE;
                }
            }
            else
            {
                SClass_ID sid = obj->SuperClassID();
                if (sid == SHAPE_CLASS_ID)
                {
                    PolyShape shape;
                    ((ShapeObject *)obj)->MakePolyShape(mStart, shape);
                    OutputPolyShapeObject(node, shape, level + 1);
                }
            }
        }
    }

    if (numTextures > 1 && ob->numInstances > 0)
    {
        if (ob->objectUsed && mirrored == ob->instMirrored)
        {
            if (!ShapeWritten) // it might have been written because the material was different, then don´t use it.
            {
                Indent(level);
                           MSTREAMPRINTF  _T("USE %s-ShapeGroup\n"),ob->instName);
                           instance = TRUE;
            }
        }
        else
        {
            level -= 3;
            Indent(level);
                   MSTREAMPRINTF  _T("] }\n"),
			   mNodes.GetNodeName(node));
                   ob->objectUsed = TRUE;
                   ob->instMirrored = mirrored;
        }
    }
    // Check for animated object, and generate CordinateInterpolator
    if (mCoordInterp && isTriMesh && ObjIsAnimated(obj) && !instance)
        VrmlOutCoordinateInterpolator(node, obj, level, mirrored);
}

TCHAR *
VRML2Export::VrmlParent(INode *node)
{
    static TCHAR buf[256];
    /* test
   Object *obj = node->EvalWorldState(mStart).obj;
   Class_ID id = obj->ClassID();
   while (id == Class_ID(MR_BLUE_CLASS_ID1, MR_BLUE_CLASS_ID2)) {
   node = node->GetParentNode();
   obj = node->EvalWorldState(mStart).obj;
   if (!obj)
   return NULL;  // Unattached
   id = obj->ClassID();
   }
   */
    assert(node);
    _tcscpy(buf, mNodes.GetNodeName(node));
    return buf;
}

BOOL
VRML2Export::IsAimTarget(INode *node)
{
    INode *lookAt = node->GetLookatNode();
    if (!lookAt)
        return FALSE;
    Object *lookAtObj = lookAt->EvalWorldState(mStart).obj;
    Class_ID id = lookAtObj->ClassID();
    // Only generate aim targets for targetted spot lights and cameras
    if (id != Class_ID(SPOT_LIGHT_CLASS_ID, 0) && id != Class_ID(LOOKAT_CAM_CLASS_ID, 0))
        return FALSE;
    return TRUE;
}
// Write out the node header for a Mr. Blue object
/* test
void
VRML2Export::VrmlAnchorHeader(INode* node, MrBlueObject* obj,
VRBL_TriggerType trigType, BOOL fromParent,
int level)
{
TSTR desc;
VRBL_Action action = obj->GetAction();
TCHAR* vrmlObjName = NULL;
vrmlObjName = VrmlParent(node);
if (!vrmlObjName)
return;

if (action == HyperLinkJump || action == MrBlueMessage ||
action == SetViewpoint) {
switch (trigType) {
case MouseClick:
MSTREAMPRINTF  _T("Anchor {\n"));
break;
case DistProximity:
break;
case BoundingBox:
break;
case LineOfSight:
break;
default:
assert(FALSE);
}

Indent(level+1);
TSTR camera;
TCHAR *name = _T("url");

switch (action) {
case MrBlueMessage:
MSTREAMPRINTF  _T("%s \"signal:\"\n"), name);
break;
case HyperLinkJump:
camera = obj->GetCamera();
if (camera.Length() == 0)
MSTREAMPRINTF  _T("%s \"%s\"\n"), name, obj->GetURL());
else
MSTREAMPRINTF  _T("%s \"%s#%s\"\n"), name, obj->GetURL(),
VRMLName(camera.data()));
if (trigType == MouseClick) {
desc = obj->GetDesc();
if (desc.Length() > 0) {
Indent(level+1);
MSTREAMPRINTF 
_T("description \"%s\"\n"), obj->GetDesc());
}
}
break;
case SetViewpoint:
if (obj->GetVptCamera())
camera = obj->GetVptCamera()->GetName();
else
camera = _T("");
MSTREAMPRINTF  _T("%s \"#%s\"\n"), name,
VRMLName(camera.data()));
if (trigType == MouseClick) {
desc = obj->GetVptDesc();
if (desc.Length() > 0) {
Indent(level+1);
MSTREAMPRINTF  _T("description \"%s\"\n"), desc);
}
}
break;
default:
assert(FALSE);
}
switch (trigType) {
case MouseClick:
MSTREAMPRINTF  _T("children [\n"));
break;
}
} else {
switch (trigType) {
case MouseClick:
MSTREAMPRINTF  _T("DEF %s-SENSOR TouchSensor {}\n"), vrmlObjName);
break;
case DistProximity:
break;
case BoundingBox:
break;
case LineOfSight:
break;
default:
assert(FALSE);
}
Indent(level+1);
TSTR camera;
TCHAR *name = _T("url");

switch (action) {
case MrBlueMessage:
MSTREAMPRINTF  _T("%s \"signal:\"\n"), name);
break;
case HyperLinkJump:
camera = obj->GetCamera();
if (camera.Length() == 0)
MSTREAMPRINTF  _T("%s \"%s\"\n"), name, obj->GetURL());
else
MSTREAMPRINTF  _T("%s \"%s#%s\"\n"), name, obj->GetURL(),
camera.data());
if (trigType == MouseClick) {
desc = obj->GetDesc();
if (desc.Length() > 0) {
Indent(level+1);
MSTREAMPRINTF 
_T("description \"%s\"\n"), obj->GetDesc());
}
}
break;
case SetViewpoint:
camera = obj->GetVptCamera()->GetName();
MSTREAMPRINTF  _T("%s \"#%s\"\n"), name, camera.data());
if (trigType == MouseClick) {
desc = obj->GetVptDesc();
if (desc.Length() > 0) {
Indent(level+1);
MSTREAMPRINTF  _T("description \"%s\"\n"), desc);
}
}
break;
case Animate: {
// Output the objects to animate
int size = obj->GetAnimObjects()->Count();
for(int i=0; i < size; i++) {
MrBlueAnimObj* animObj = (*obj->GetAnimObjects())[i];
Object *o = animObj->node->EvalWorldState(mStart).obj;
if (!o)
break;
assert(vrmlObjName);
if (IsAimTarget(animObj->node))
break;
INode* top;
if (o->ClassID() == TimeSensorClassID)
top = animObj->node;
else
top = GetTopLevelParent(animObj->node);
ObjectBucket* ob =
mObjTable.AddObject(top->EvalWorldState(mStart).obj);
AddAnimRoute(vrmlObjName, ob->name.data(), node, top);
AddCameraAnimRoutes(vrmlObjName, node, top);
}
break; }
default:
assert(FALSE);
}
}
}
*/

void
VRML2Export::AddCameraAnimRoutes(TCHAR *vrmlObjName, INode *fromNode,
                                 INode *top, int field)
{
    for (int i = 0; i < top->NumberOfChildren(); i++)
    {
        INode *child = top->GetChildNode(i);
        Object *obj = child->EvalWorldState(mStart).obj;
        if (!obj)
            continue;
        SClass_ID sid = obj->SuperClassID();
        if (sid == CAMERA_CLASS_ID)
            AddAnimRoute(vrmlObjName, mNodes.GetNodeName(child),
                         fromNode, child, field);
        AddCameraAnimRoutes(vrmlObjName, fromNode, child, field);
    }
}

void
VRML2Export::AddAnimRoute(const TCHAR *from, const TCHAR *to, INode *fromNode,
                          INode *toNode, int field, int tuiElementType)
{
    TCHAR fromStr[MAX_PATH];
    if (!from || *from == '\0')
    {
        _tcscpy(fromStr, mNodes.GetNodeName(fromNode));
        from = fromStr;
    }
    if (!to || *to == '\0')
        to = mNodes.GetNodeName(toNode);
    AnimRoute *ar = new AnimRoute(from, to, fromNode, toNode, field, tuiElementType);
    mAnimRoutes.Append(1, ar);
}

int
VRML2Export::NodeNeedsTimeSensor(INode *node)
{
    BOOL isCamera = IsCamera(node);
    BOOL isAudio = IsAudio(node);
    BOOL isAnim = (isAudio || (!isCamera && node->GetParentNode()->IsRootNode() && ChildIsAnimated(node)) || (isCamera && (IsEverAnimated(node) || IsEverAnimated(node->GetTarget()))));
    if (!isAnim)
        return 0;
    if (node->GetNodeLong() & (RUN_BY_PROX_SENSOR | RUN_BY_TOUCH_SENSOR | RUN_BY_ONOFF_SENSOR | RUN_BY_COVER_SENSOR))
        return 1;
    if (node->GetNodeLong() & (RUN_BY_TIME_SENSOR | RUN_BY_TABLETUI_SENSOR))
        return 0;
    return 0;
    //was , groups had a timesensor they did not need because -1 != 0 return -1;
}

TCHAR *VRML2Export::isMovie(const TCHAR *url)
{
    TCHAR *name = new TCHAR[_tcslen(url) + 1];
    const TCHAR *suffix = _tcsrchr(url, '.');
    if (suffix == NULL)
    {
        _tcscpy(name, url);
        return (name);
    }
    if ((_tcsicmp(suffix, _T(".mpg")) == 0) || (_tcsicmp(suffix, _T(".avi")) == 0) || (_tcsicmp(suffix, _T(".mov")) == 0))
    {
        suffix = _tcsrchr(url, '.');
        _tcsncpy(name, url, suffix - url);
        name[suffix - url] = '\0';
    }
    else
        return NULL;

    const TCHAR *dir = _tcsrchr(name, '/');
    if (dir != NULL)
    {
        _tcscpy(name, _T("_"));
        _tcscat(name, dir + 1);
    }
    else
    {
        TCHAR *tmpName = new TCHAR[_tcslen(name) + 2];
        _tcscpy(tmpName, _T("_"));
        _tcscat(tmpName, name);
        name = tmpName;
    }
    return name;
}

void
VRML2Export::WriteScripts()
{
    INodeList *l;
    for (l = mScriptsList; l; l = l->GetNext())
    {
        VRMLScriptObject *so = (VRMLScriptObject *)
                                   l->GetNode()->EvalWorldState(mStart)
                                       .obj;
        if (so)
        {
            // remove unwanted \rs
            const TCHAR *textdata = so->GetUrl().data();
            TCHAR *buf = new TCHAR[_tcslen(textdata) + 1];
            TCHAR *b = buf;
            while (*textdata != '\0')
            {
                if (*textdata != '\r')
                {
                    *b = *textdata;
                    b++;
                }
                textdata++;
            }
            *b = '\0';

         MSTREAMPRINTF  _T("#Script %s\n%s\n"),l->GetNode()->GetName(), buf);
         delete[] buf;
        }
    }
}

void
VRML2Export::WriteAnimRoutes()
{
    int i;
    int ts;
    TCHAR from[MAX_PATH], to[MAX_PATH];
    for (i = 0; i < mAnimRoutes.Count(); i++)
    {
        INode *toNode = mAnimRoutes[i].mToNode;
        TCHAR *toName = mNodes.GetNodeName(toNode);
        Object *toObj = toNode->EvalWorldState(mStart).obj;
        Object *fromObj = mAnimRoutes[i].mFromNode->EvalWorldState(mStart).obj;
        int tuiElementType = mAnimRoutes[i].mTuiElementType;

        if (fromObj->ClassID() == ProxSensorClassID)
        {
            if (mAnimRoutes[i].mField == ENTER_FIELD)
                _stprintf(from, _T("%s.enterTime"), mAnimRoutes[i].mFromName);
            else
                _stprintf(from, _T("%s.exitTime"), mAnimRoutes[i].mFromName);
        }
        else if (fromObj->ClassID() == OnOffSwitchClassID)
        {
            if (((OnOffSwitchObject *)fromObj)->onObject == toNode)
                _stprintf(from, _T("%s-SCRIPT.onTime"), mAnimRoutes[i].mFromName);
            else
                _stprintf(from, _T("%s-SCRIPT.offTime"), mAnimRoutes[i].mFromName);
        }
        else if (fromObj->ClassID() == COVERClassID)
        {
            if (mAnimRoutes[i].mField >= 0)
                _stprintf(from, _T("%s-SCRIPT.key%s"), mAnimRoutes[i].mFromName, ((COVERObject *)fromObj)->objects[mAnimRoutes[i].mField]->keyStr);
        }
        else if ((fromObj->ClassID() == TabletUIClassID) || (fromObj->ClassID() == SwitchClassID))
            _stprintf(from, _T("%s"), mAnimRoutes[i].mFromName);
        else
            _stprintf(from, _T("%s-SENSOR.touchTime"), mAnimRoutes[i].mFromName);
        BOOL isCamera = IsCamera(toNode);
        ts = NodeNeedsTimeSensor(toNode);
        if (ts != 0 || toObj->ClassID() == TimeSensorClassID || (toObj->ClassID() == OnOffSwitchClassID))
        {
            if (toObj->ClassID() == AudioClipClassID)
                _stprintf(to, _T("%s.startTime"), toName);
            else if (toObj->ClassID() == OnOffSwitchClassID)
                _stprintf(to, _T("%s-SCRIPT.trigger"), toName);
            else if (toObj->ClassID() == TouchSensorClassID)
            {
                _stprintf(to, _T("%sStartStop"), mAnimRoutes[i].mFromName);
            MSTREAMPRINTF  _T("ROUTE %s TO %s.clickTime\n"), from, to);
            MSTREAMPRINTF  _T("ROUTE %s.startTime TO %s.startTime\n"), to, mAnimRoutes[i].mToName);
            MSTREAMPRINTF  _T("ROUTE %s.stopTime TO %s.stopTime\n"), to, mAnimRoutes[i].mToName);
            }
            else
                _stprintf(to, _T("%s-TIMER.startTime"), toName);

         MSTREAMPRINTF  _T("ROUTE %s TO %s\n"), from, to);
        }
        else if (toObj->ClassID() == SwitchClassID)
        {
            VrmlOutSwitchScript(toNode);
            if (tuiElementType != TUIComboBox)
            {
                _stprintf(to, _T("%s-SCRIPT.trigger"), toName);
                if (tuiElementType != TUIButton)
                {
               MSTREAMPRINTF  _T("ROUTE %s.startTime_changed  TO %s\n"), from, to);
               MSTREAMPRINTF  _T("ROUTE %s.stopTime_changed  TO %s\n"), from, to);
                }
                else
               MSTREAMPRINTF  _T("ROUTE %s TO %s\n"), from, to);
                _stprintf(from, _T("%s-SCRIPT.choice"), toName);
            }
            else
                _stprintf(from, _T("%s.choice"), from);
            _stprintf(to, _T("Choice%s-SCRIPT.userChoice"), toName);
         MSTREAMPRINTF  _T("ROUTE %s TO %s\n"), from, to);
         _stprintf(from, _T("Choice%s-SCRIPT.switchChoice"), toName);
         _stprintf(to, _T("%s.whichChoice"), toName);
         MSTREAMPRINTF  _T("ROUTE %s TO %s\n"), from, to);
        }
        else if (fromObj->ClassID() == MultiTouchSensorClassID)
        {
            _stprintf(to, _T("%s.scale"), toName);
            _stprintf(from, _T("%s-SENSOR.scale_changed"), mAnimRoutes[i].mFromName);
         MSTREAMPRINTF  _T("ROUTE %s TO %s\n"), from, to);
         _stprintf(to, _T("%s.translation"), toName);
         _stprintf(from, _T("%s-SENSOR.translation_changed"), mAnimRoutes[i].mFromName);
         MSTREAMPRINTF  _T("ROUTE %s TO %s\n"), from, to);
         _stprintf(to, _T("%s.rotation"), toName);
         _stprintf(from, _T("%s-SENSOR.rotation_changed"), mAnimRoutes[i].mFromName);
         MSTREAMPRINTF  _T("ROUTE %s TO %s\n"), from, to);
        }
        else if (fromObj->ClassID() == ARSensorClassID)
        {
            _stprintf(to, _T("%s.scale"), toName);
            _stprintf(from, _T("%s-SENSOR.scale_changed"), mAnimRoutes[i].mFromName);
         MSTREAMPRINTF  _T("ROUTE %s TO %s\n"), from, to);
         _stprintf(to, _T("%s.translation"), toName);
         _stprintf(from, _T("%s-SENSOR.translation_changed"), mAnimRoutes[i].mFromName);
         MSTREAMPRINTF  _T("ROUTE %s TO %s\n"), from, to);
         _stprintf(to, _T("%s.rotation"), toName);
         _stprintf(from, _T("%s-SENSOR.rotation_changed"), mAnimRoutes[i].mFromName);
         MSTREAMPRINTF  _T("ROUTE %s TO %s\n"), from, to);
        }
        else if (fromObj->ClassID() == COVERClassID)
        {
            if (mAnimRoutes[i].mField < 0)
            {
                _stprintf(to, _T("%s.translation"), toName);
                _stprintf(from, _T("%s-SENSOR.avatar1Position"), mAnimRoutes[i].mFromName);
            MSTREAMPRINTF  _T("ROUTE %s TO %s\n"), from, to);
            _stprintf(to, _T("%s.rotation"), toName);
            _stprintf(from, _T("%s-SENSOR.avatar1Orientation"), mAnimRoutes[i].mFromName);
            MSTREAMPRINTF  _T("ROUTE %s TO %s\n"), from, to);
            }
        }
        else if (fromObj->ClassID() == TabletUIClassID)
        {
            _stprintf(to, _T("%s"), mAnimRoutes[i].mToName);
         MSTREAMPRINTF  _T("ROUTE %s TO %s\n"), from, to);
        }
        else if (toObj->ClassID() == TouchSensorClassID)
        {
            _stprintf(to, _T("%sStartStop"), mAnimRoutes[i].mFromName);
         MSTREAMPRINTF  _T("ROUTE %s TO %s.clickTime\n"), from, to);
         MSTREAMPRINTF  _T("ROUTE %s.startTime TO %s.startTime\n"), to, mAnimRoutes[i].mToName);
         MSTREAMPRINTF  _T("ROUTE %s.stopTime TO %s.stopTime\n"), to, mAnimRoutes[i].mToName);
        }
        else if ((fromObj->ClassID() == SwitchClassID) && (toObj->SuperClassID() == CAMERA_CLASS_ID))
        {
            _stprintf(to, _T("%s"), mAnimRoutes[i].mToName);
         MSTREAMPRINTF  _T("ROUTE Choice%s-SCRIPT.switchChoice TO %s%s-SCRIPT.active\n"), from, to, from);
         MSTREAMPRINTF  _T("ROUTE %s%s-SCRIPT.state TO %s.set_bind\n"), to, from, to);
        }
    }
}

// Write out the header for a single Mr. Blue node
/* test
BOOL
VRML2Export::VrmlOutMrBlue(INode* node, INode* parent, MrBlueObject* obj,
int* level, BOOL fromParent)
{
BOOL hadHeader = FALSE;
TCHAR* name;
if (fromParent)
name = mNodes.GetNodeName(parent);
else
name = mNodes.GetNodeName(node);

if (obj->GetMouseEnabled()) {
MrBlueObject* mbo = (MrBlueObject*) obj;
Indent(*level);
VrmlAnchorHeader(node, obj, MouseClick, fromParent, *level);
(*level)++;
hadHeader = TRUE;
}

if (mType != Export_VRBL)
goto end;

if (obj->GetProxDistEnabled()) {
Indent(*level);
if (!hadHeader)
MSTREAMPRINTF  _T("DEF %s "), name);
VrmlAnchorHeader(node, obj, DistProximity, fromParent, *level);
Indent(*level+1);
MSTREAMPRINTF  _T("distance %s\n"), floatVal(obj->GetProxDist()));
if (!fromParent) {
// Generate proximity point for top-level objects.
Indent(*level+1);
MSTREAMPRINTF  _T("point 0 0 0 \n"));
}
(*level)++;
hadHeader = TRUE;
}

if (obj->GetBBoxEnabled() && !fromParent) {
if (!fromParent)
Indent(*level);
if (!hadHeader)
MSTREAMPRINTF  _T("DEF %s "), name);
VrmlAnchorHeader(node, obj, BoundingBox, fromParent, *level);
Indent(*level+1);

float x = obj->GetBBoxX()/2.0f,
y = obj->GetBBoxY()/2.0f,
z = obj->GetBBoxZ()/2.0f;
Point3 p0 = Point3(-x, -y, -z), p1 = Point3(x, y, z);
MSTREAMPRINTF  _T("point [ %s, "), point(p0));
MSTREAMPRINTF  _T(" %s ]\n"), point(p1));
(*level)++;
hadHeader = TRUE;
}

if (obj->GetLosEnabled()) {
if (obj->GetLosType() == CanSee) {
Indent(*level);
if (!hadHeader)
MSTREAMPRINTF  _T("DEF %s "), name);
VrmlAnchorHeader(node, obj, LineOfSight, fromParent, *level);
Indent(*level+1);
MSTREAMPRINTF  _T("distance %s\n"),
floatVal(GetLosProxDist(node, mStart)));
Indent(*level+1);
MSTREAMPRINTF  _T("angle %s\n"),
floatVal(DegToRad(obj->GetLosVptAngle())));
}
else {
Indent(*level);
if (!hadHeader)
MSTREAMPRINTF  _T("DEF %s "), name);
VrmlAnchorHeader(node, obj, LineOfSight, fromParent, *level);
Indent(*level+1);
MSTREAMPRINTF  _T("distance %s\n"),
floatVal(GetLosProxDist(node, mStart)));
Indent(*level+1);
MSTREAMPRINTF  _T("sightAngle %s\n"),
floatVal(DegToRad(obj->GetLosVptAngle())));
Point3 p = GetLosVector(node, mStart);
Indent(*level+1);
MSTREAMPRINTF  _T("vector %s\n"), normPoint(p));
Indent(*level+1);
MSTREAMPRINTF  _T("vectorAngle %s\n"),
floatVal(DegToRad(obj->GetLosObjAngle())));
}
(*level)++;
}

end:
// Close off the nodes if this is a stand-alone helper
if (!fromParent)
EndMrBlueNode(node, *level, FALSE);

return TRUE;
}
*/

/*
// Start the headers for Mr. Blue nodes attached to the given node,
// returning the new indentation level
int
VRML2Export::StartMrBlueHelpers(INode* node, int level)
{
// Check for Mr Blue helper at child nodes
for(int i=0; i<node->NumberOfChildren(); i++) {
INode* childNode = node->GetChildNode(i);
Object *obj = childNode->EvalWorldState(mStart).obj;
Class_ID id = obj->ClassID();
if (id == Class_ID(MR_BLUE_CLASS_ID1, MR_BLUE_CLASS_ID2)) {
MrBlueObject *mbo = (MrBlueObject*) obj;
if ((mbo->GetAction() == HyperLinkJump ||
mbo->GetAction() == SetViewpoint) &&
mbo->GetMouseEnabled())
VrmlOutMrBlue(childNode, node, mbo, &level, TRUE);
}
}
return level;
}
*/

// Write out the node closer for a Mr. Blue node
/* test
void
VRML2Export::EndMrBlueNode(INode* childNode, int& level, BOOL fromParent)
{
Object *obj = childNode->EvalWorldState(mStart).obj;
Class_ID id = obj->ClassID();
if (id == Class_ID(MR_BLUE_CLASS_ID1, MR_BLUE_CLASS_ID2)) {
MrBlueObject* mbo = (MrBlueObject*) obj;
if (mbo->GetMouseEnabled()) {
if (mbo->GetAction() == HyperLinkJump ||
mbo->GetAction() == SetViewpoint) {
if (!fromParent)
return;
Indent(level);
MSTREAMPRINTF  _T("]\n"));
Indent(--level);
MSTREAMPRINTF  _T("}\n"));
}
}
// FIXME take care of these
if (mbo->GetProxDistEnabled()) {
}
if (mbo->GetBBoxEnabled()) {
}
if (mbo->GetLosEnabled()) {
}
}
}
*/

// Write out the node closers for all the Mr. Blue headers
/* test
void
VRML2Export::EndMrBlueHelpers(INode* node, int level)
{
// Check for Mr Blue helper at child nodes
for(int i=0; i<node->NumberOfChildren(); i++) {
EndMrBlueNode(node->GetChildNode(i), level, TRUE);
}
}
*/

void
VRML2Export::InitInterpolators(INode *node)
{
#ifdef _LEC_
    if (mFlipBook)
        return;
#endif
    mInterpRoutes.SetCount(0);
    LONG_PTR sensors = node->GetNodeLong() & RUN_BY_ANY_SENSOR;
    if (sensors & (RUN_BY_TIME_SENSOR | RUN_BY_TABLETUI_SENSOR))

#if MAX_PRODUCT_VERSION_MAJOR > 14
        mTimer = NULL;
#else
        mTimer = (const char *)NULL;
#endif
    else if (sensors)
        mTimer = TSTR(mNodes.GetNodeName(node)) + TSTR(_T("-TIMER"));
    else
        mTimer = TSTR(_T("Global-TIMER"));
}

void
VRML2Export::AddInterpolator(const TCHAR *interp, int type, const TCHAR *name, INode *node)
{
    // ROUTE Modification by Uwe,
    //check and see if we already have an interpolator Route for this node, if so skipp all others
    /*TSTR inte=interp;
   TSTR globalNode = "Global-TIMER";
   int i;
   for(i = 0; i < mInterpRoutes.Count(); i++)
   {
   if (mInterpRoutes[i].mInterp == inte)
   {
   if(mInterpRoutes[i].mNode == globalNode)
   {

   MSTREAMPRINTF _T("#Duplicate Interpolator, tell Uwe %s %s\n"),interp,name);
   }
   else if(mTimer == globalNode)
   {
   return;
   }
   }
   }*/
    InterpRoute *r = new InterpRoute(interp, type, name, node);
    mInterpRoutes.Append(1, r);
}

void
VRML2Export::WriteInterpolatorRoutes(int level)
{
    BOOL isCamera;
#ifdef _LEC_
    if (mFlipBook)
        return;
#endif
    int i, n;
    for (i = 0; i < mInterpRoutes.Count(); i++)
    {
        Indent(level);
        if (mInterpRoutes[i].mType == KEY_TIMER)
         MSTREAMPRINTF 
         _T("ROUTE %s-TIMER.fraction_changed TO %s.set_fraction\n"),
         (char *)mInterpRoutes[i].mNodeName.data(), (char *)mInterpRoutes[i].mInterp.data());
        else if ((mInterpRoutes[i].mType == (KEY_TIMER | KEY_TABLETUI_TOGGLE)) || (mInterpRoutes[i].mType == (KEY_TIMER_SCRIPT | KEY_TABLETUI_TOGGLE)))
        {
         MSTREAMPRINTF 
         _T("ROUTE %s-SCRIPT.newFraction TO %s.set_fraction\n"),
         (char *)mInterpRoutes[i].mNodeName.data(), (char *)mInterpRoutes[i].mInterp.data());
        }
        else if (mInterpRoutes[i].mType == KEY_TIMER_SCRIPT)
        {
                 MSTREAMPRINTF _T("ROUTE %s-TIMER.fraction_changed TO %s-SCRIPT.fractionIn\n"),
            (char *)mInterpRoutes[i].mNodeName.data(), (char *)mInterpRoutes[i].mNodeName.data());
         MSTREAMPRINTF 
            _T("ROUTE %s-SCRIPT.fractionOut TO %s.set_fraction\n"),
            (char *)mInterpRoutes[i].mNodeName.data(), (char *)mInterpRoutes[i].mInterp.data());
        }
        else if (mInterpRoutes[i].mType == KEY_TABLETUI)
         MSTREAMPRINTF 
         _T("ROUTE %s.value_changed TO %s.set_fraction\n"),
         (char *)mInterpRoutes[i].mNodeName.data(), (char *)mInterpRoutes[i].mInterp.data());
        else if (mInterpRoutes[i].mType == KEY_TABLETUI_SLIDER)
                 MSTREAMPRINTF _T("ROUTE %s-SCRIPT.value_changed TO %s.set_fraction\n"),
				(char *)mInterpRoutes[i].mNodeName.data(), (char *)mInterpRoutes[i].mInterp.data());
        else if (mInterpRoutes[i].mType == KEY_TABLETUI_SLIDER_SCRIPT)
                  MSTREAMPRINTF _T("ROUTE %s-SCRIPT.value_changed TO %s-SCRIPT.fractionIn\n"),
				(char *)mInterpRoutes[i].mInterp.data(), (char *)mInterpRoutes[i].mNodeName.data());
        else if (mInterpRoutes[i].mType == KEY_TABLETUI_SCRIPT)
        {
         MSTREAMPRINTF 
            _T("ROUTE %s-SCRIPT.startTime_changed TO %s-TIMER.startTime\n"),
            (char *)mInterpRoutes[i].mInterp.data(), (char *)mInterpRoutes[i].mNodeName.data());
         MSTREAMPRINTF 
            _T("ROUTE %s-SCRIPT.stopTime_changed TO %s-TIMER.stopTime\n"),
            (char *)mInterpRoutes[i].mInterp.data(), (char *)mInterpRoutes[i].mNodeName.data());
        }
        else if (mInterpRoutes[i].mType == KEY_TABLETUI_TOGGLE)
        {
         MSTREAMPRINTF  _T("ROUTE %s-TIMER.fraction_changed TO %s-SCRIPT.fractionChanged\n"),
         (char *)mInterpRoutes[i].mNodeName.data(), (char *)mInterpRoutes[i].mNodeName.data());
         MSTREAMPRINTF  _T("ROUTE %s-SCRIPT.timerStop TO %s-TIMER.stopTime\n"),
         (char *)mInterpRoutes[i].mNodeName.data(), (char *)mInterpRoutes[i].mNodeName.data());
         MSTREAMPRINTF 
            _T("ROUTE %s-SCRIPT.startTime_changed TO %s-TIMER.startTime\n"),
            (char *)mInterpRoutes[i].mInterp.data(), (char *)mInterpRoutes[i].mNodeName.data());
         MSTREAMPRINTF 
            _T("ROUTE %s-SCRIPT.stopTime_changed TO %s-TIMER.stopTime\n"),
            (char *)mInterpRoutes[i].mInterp.data(), (char *)mInterpRoutes[i].mNodeName.data());
         MSTREAMPRINTF 
            _T("ROUTE %s-SCRIPT.stopTime_changed TO %s-SCRIPT.stopTime\n"),
            (char *)mInterpRoutes[i].mInterp.data(), (char *)mInterpRoutes[i].mNodeName.data());
         MSTREAMPRINTF 
            _T("ROUTE %s-SCRIPT.toggleOn TO %s.state\n"),
            (char *)mInterpRoutes[i].mNodeName.data(), (char *)mInterpRoutes[i].mInterp.data());
        }
        else if (mInterpRoutes[i].mType == KEY_TABLETUI_BUTTON)
         MSTREAMPRINTF  _T("ROUTE %s-TIMER.fraction_changed TO %s.set_fraction\n"),
         (char *)mInterpRoutes[i].mNodeName.data(), (char *)mInterpRoutes[i].mInterp.data());
        else if (mInterpRoutes[i].mType == KEY_TOUCHSENSOR_BIND)
        {
         MSTREAMPRINTF  _T("ROUTE %s-SENSOR.touchTime TO %s%s-SCRIPT.active\n"), (char *)mInterpRoutes[i].mInterp.data(), (char *)mInterpRoutes[i].mNodeName.data(), (char *)mInterpRoutes[i].mInterp.data());
         MSTREAMPRINTF  _T("ROUTE %s%s-SCRIPT.state TO %s.set_bind\n"), (char *)mInterpRoutes[i].mNodeName.data(), (char *)mInterpRoutes[i].mInterp.data(), (char *)mInterpRoutes[i].mNodeName.data());
        }
        else if (mInterpRoutes[i].mType == KEY_PROXSENSOR_ENTER_BIND)
        {
         MSTREAMPRINTF  _T("ROUTE %s.enterTime TO %s%s-SCRIPT.active\n"), (char *)mInterpRoutes[i].mInterp.data(), (char *)mInterpRoutes[i].mNodeName.data(), (char *)mInterpRoutes[i].mInterp.data());
         MSTREAMPRINTF  _T("ROUTE %s%s-SCRIPT.state TO %s.set_bind\n"), (char *)mInterpRoutes[i].mNodeName.data(), (char *)mInterpRoutes[i].mInterp.data(), (char *)mInterpRoutes[i].mNodeName.data());
        }
        else if (mInterpRoutes[i].mType == KEY_PROXSENSOR_EXIT_BIND)
        {
         MSTREAMPRINTF  _T("ROUTE %s.exitTime TO %s%s-SCRIPT.active\n"), (char *)mInterpRoutes[i].mInterp.data(), (char *)mInterpRoutes[i].mNodeName.data(), (char *)mInterpRoutes[i].mInterp.data());
         MSTREAMPRINTF  _T("ROUTE %s%s-SCRIPT.state TO %s.set_bind\n"), (char *)mInterpRoutes[i].mNodeName.data(), (char *)mInterpRoutes[i].mInterp.data(), (char *)mInterpRoutes[i].mNodeName.data());
        }
        else if (mInterpRoutes[i].mType == (KEY_TOUCHSENSOR_BIND | KEY_TABLETUI_BUTTON))
        {
         MSTREAMPRINTF  _T("ROUTE %s.touchTime TO %s%s-SCRIPT.active\n"), (char *)mInterpRoutes[i].mInterp.data(), (char *)mInterpRoutes[i].mNodeName.data(), (char *)mInterpRoutes[i].mInterp.data());
         MSTREAMPRINTF  _T("ROUTE %s%s-SCRIPT.state TO %s.set_bind\n"), (char *)mInterpRoutes[i].mNodeName.data(), (char *)mInterpRoutes[i].mInterp.data(), (char *)mInterpRoutes[i].mNodeName.data());
        }
        else if (!mTimer.isNull())
        {
            // test if this interpolator has already been served by a timesensor, if so, don´t use a global timer

            TSTR globalNode = _T("Global-TIMER");
            bool foundNode = false;
            if (mTimer == globalNode)
            {
                for (n = 0; n < mInterpRoutes.Count(); n++)
                {
                    if (n != i)
                    {
                        if (((mInterpRoutes[n].mType == KEY_TIMER) || (mInterpRoutes[n].mType == KEY_TIMER_SCRIPT)) && (mInterpRoutes[i].mInterp == mInterpRoutes[n].mInterp))
                        {
                            foundNode = true;
                            break;
                        }
                    }
                }
            }
            if (!foundNode)
            {
            MSTREAMPRINTF  _T("ROUTE %s.fraction_changed TO %s.set_fraction\n"),
               (char *)mTimer.data(), (char *)mInterpRoutes[i].mInterp.data());
            }
        }
        Indent(level);
        TCHAR *setType = NULL;
        isCamera = IsCamera(mInterpRoutes[i].mNode);
        switch (mInterpRoutes[i].mType)
        {
        case KEY_POS:
            if (isCamera)
                setType = _T("set_position");
            else
                setType = _T("set_translation");
            break;
        case KEY_ROT:
            if (isCamera)
                setType = _T("set_orientation");
            else
                setType = _T("set_rotation");
            break;
        case KEY_SCL:
            setType = _T("set_scale");
            break;
        case KEY_SCL_ORI:
            setType = _T("set_scaleOrientation");
            break;
        case KEY_COORD:
            setType = _T("set_point");
            break;
        case KEY_COLOR:
            setType = _T("setColor");
            break;
        }
        if (mInterpRoutes[i].mType & KEY_INTERPOL)
        {
            assert(setType);
            if (isCamera)
                  MSTREAMPRINTF  _T("ROUTE %s.value_changed TO %s.%s\n"),
                     (char *)mInterpRoutes[i].mInterp.data(), (char *)mInterpRoutes[i].mNodeName.data(),
                     setType);
            else if (mInterpRoutes[i].mType == KEY_COLOR)
               MSTREAMPRINTF  _T("ROUTE %s.value_changed TO %s-LIGHT.%s\n"),
               (char *)mInterpRoutes[i].mInterp.data(), (char *)mInterpRoutes[i].mNodeName.data(),
               setType);
            else if (mInterpRoutes[i].mType == KEY_COORD)
               MSTREAMPRINTF _T("ROUTE %s.value_changed TO %s-COORD.%s\n"),
               (char *)mInterpRoutes[i].mInterp.data(), (char *)mInterpRoutes[i].mNodeName.data(),
               setType);
            else
               MSTREAMPRINTF  _T("ROUTE %s.value_changed TO %s.%s\n"),
               (char *)mInterpRoutes[i].mInterp.data(), (char *)mInterpRoutes[i].mNodeName.data(),
               setType);
        }
    }
}

inline BOOL
ApproxEqual(float a, float b, float eps)
{
    float d = (float)fabs(a - b);
    return d < eps;
}

int
reducePoint3Keys(Tab<TimeValue> &times, Tab<Point3> &points, float eps)
{
    if (times.Count() < 3)
        return times.Count();

    BOOL *used = new BOOL[times.Count()];
    int i;
    for (i = 0; i < times.Count(); i++)
        used[i] = TRUE;

    // The two lines are represented as p0 + v * s and q0 + w * t.
    Point3 p0, q0;
    for (i = 1; i < times.Count(); i++)
    {
        p0 = points[i];
        q0 = points[i - 1];
        if (ApproxEqual(p0.x, q0.x, eps) && ApproxEqual(p0.y, q0.y, eps) && ApproxEqual(p0.z, q0.z, eps))
            used[i] = FALSE;
        else
        {
            used[i - 1] = TRUE;
        }
    }

    int j = 0;
    for (i = 0; i < times.Count(); i++)
        if (used[i])
            j++;
    if (j == 1)
    {
        delete[] used;
        return 0;
    }
    j = 0;
    for (i = 0; i < times.Count(); i++)
    {
        if (used[i])
        {
            times[j] = times[i];
            points[j] = points[i];
            j++;
        }
    }
    times.SetCount(j);
    points.SetCount(j);
    delete[] used;
    if (j == 1)
        return 0;
    if (j == 2)
    {
        p0 = points[0];
        q0 = points[1];
        if (ApproxEqual(p0.x, q0.x, eps) && ApproxEqual(p0.y, q0.y, eps) && ApproxEqual(p0.z, q0.z, eps))
            return 0;
    }
    return j;
}

int
reduceAngAxisKeys(Tab<TimeValue> &times, Tab<AngAxis> &points, float eps)
{
    if (times.Count() < 3)
        return times.Count();

    BOOL *used = new BOOL[times.Count()];
    int i;
    for (i = 0; i < times.Count(); i++)
        used[i] = TRUE;

    // The two lines are represented as p0 + v * s and q0 + w * t.
    AngAxis p0, q0;
    for (i = 1; i < times.Count(); i++)
    {
        p0 = points[i];
        q0 = points[i - 1];
        if (ApproxEqual(p0.axis.x, q0.axis.x, eps) && ApproxEqual(p0.axis.y, q0.axis.y, eps) && ApproxEqual(p0.axis.z, q0.axis.z, eps) && ApproxEqual(p0.angle, q0.angle, eps))
            used[i] = FALSE;
        else
        {
            used[i - 1] = TRUE;
        }
    }

    int j = 0;
    for (i = 0; i < times.Count(); i++)
        if (used[i])
            j++;
    if (j == 1)
    {
        delete[] used;
        return 0;
    }
    j = 0;
    for (i = 0; i < times.Count(); i++)
    {
        if (used[i])
        {
            times[j] = times[i];
            points[j] = points[i];
            j++;
        }
    }
    times.SetCount(j);
    points.SetCount(j);
    delete[] used;
    if (j == 1)
        return 0;
    if (j == 2)
    {
        p0 = points[0];
        q0 = points[1];
        if (ApproxEqual(p0.axis.x, q0.axis.x, eps) && ApproxEqual(p0.axis.y, q0.axis.y, eps) && ApproxEqual(p0.axis.z, q0.axis.z, eps) && ApproxEqual(p0.angle, q0.angle, eps))
            return 0;
    }
    return j;
}

int
reduceScaleValueKeys(Tab<TimeValue> &times, Tab<ScaleValue> &svs, float eps)
{
    if (times.Count() < 3)
        return times.Count();

    BOOL *used = new BOOL[times.Count()];
    BOOL alliso = (ApproxEqual(svs[0].s.x, svs[0].s.y, eps) && ApproxEqual(svs[0].s.x, svs[0].s.z, eps));
    int i;
    for (i = 0; i < times.Count(); i++)
        used[i] = TRUE;

    Point3 s0, t0;
    AngAxis p0, q0;
    for (i = 1; i < times.Count(); i++)
    {
        s0 = svs[i].s;
        t0 = svs[i - 1].s;
        if (ApproxEqual(s0.x, t0.x, eps) && ApproxEqual(s0.y, t0.y, eps) && ApproxEqual(s0.z, t0.z, eps))
        {
            AngAxisFromQa(svs[i].q, &p0.angle, p0.axis);
            AngAxisFromQa(svs[i - 1].q, &q0.angle, q0.axis);
            if (ApproxEqual(p0.axis.x, q0.axis.x, eps) && ApproxEqual(p0.axis.y, q0.axis.y, eps) && ApproxEqual(p0.axis.z, q0.axis.z, eps) && ApproxEqual(p0.angle, q0.angle, eps))
                used[i] = FALSE;
            else
                used[i - 1] = TRUE;
        }
        else
        {
            used[i - 1] = TRUE;
            alliso = FALSE;
        }
    }

    if (alliso)
    { // scale always isotropic and constant
        delete[] used;
        return 0;
    }

    int j = 0;
    for (i = 0; i < times.Count(); i++)
        if (used[i])
            j++;
    if (j == 1)
    {
        delete[] used;
        return 0;
    }
    j = 0;
    for (i = 0; i < times.Count(); i++)
    {
        if (used[i])
        {
            times[j] = times[i];
            svs[j] = svs[i];
            j++;
        }
    }
    times.SetCount(j);
    svs.SetCount(j);
    delete[] used;
    if (j == 1)
        return 0;
    if (j == 2)
    {
        s0 = svs[0].s;
        t0 = svs[1].s;
        AngAxisFromQa(svs[0].q, &p0.angle, p0.axis);
        AngAxisFromQa(svs[1].q, &q0.angle, q0.axis);
        if (ApproxEqual(s0.x, t0.x, eps) && ApproxEqual(s0.y, t0.y, eps) && ApproxEqual(s0.z, t0.z, eps) && ApproxEqual(p0.axis.x, q0.axis.x, eps) && ApproxEqual(p0.axis.y, q0.axis.y, eps) && ApproxEqual(p0.axis.z, q0.axis.z, eps) && ApproxEqual(p0.angle, q0.angle, eps))
            return 0;
    }
    return j;
}

// Write out all the keyframe data for the given controller
void
VRML2Export::WriteControllerData(INode *node,
                                 Tab<TimeValue> &posTimes,
                                 Tab<Point3> &posKeys,
                                 Tab<TimeValue> &rotTimes,
                                 Tab<AngAxis> &rotKeys,
                                 Tab<TimeValue> &sclTimes,
                                 Tab<ScaleValue> &sclKeys,
                                 int type, int level)
{
    AngAxis rval;
    Point3 p, s;
    Quat q;
    size_t i, width;
    TimeValue t;
    TCHAR name[128];
    Tab<TimeValue> &timeVals = posTimes;
    int newKeys;
    float eps;

    while (type)
    {

        // Set up
        switch (type)
        {
        case KEY_POS:
            eps = float(1.0e-5);
            newKeys = reducePoint3Keys(posTimes, posKeys, eps);
            if (newKeys == 0)
                return;
            timeVals = posTimes;
            _stprintf(name, _T("%s-POS-INTERP"), mNodes.GetNodeName(node));
            Indent(level);
         MSTREAMPRINTF  _T("DEF %s PositionInterpolator {\n"), name);
         break;
        case KEY_ROT:
            eps = float(1.0e-5);
            newKeys = reduceAngAxisKeys(rotTimes, rotKeys, eps);
            if (newKeys == 0)
                return;
            timeVals = rotTimes;
            _stprintf(name, _T("%s-ROT-INTERP"), mNodes.GetNodeName(node));
            Indent(level);
         MSTREAMPRINTF  _T("DEF %s OrientationInterpolator {\n"), name);
         break;
        case KEY_SCL:
            eps = float(1.0e-5);
            newKeys = reduceScaleValueKeys(sclTimes, sclKeys, eps);
            if (newKeys == 0)
                return;
            timeVals = sclTimes;
            _stprintf(name, _T("%s-SCALE-INTERP"), mNodes.GetNodeName(node));
            Indent(level);
         MSTREAMPRINTF  _T("DEF %s PositionInterpolator {\n"), name);
         break;
        case KEY_SCL_ORI:
            timeVals = sclTimes;
            _stprintf(name, _T("%s-SCALE-ORI-INTERP"), mNodes.GetNodeName(node));
            Indent(level);
         MSTREAMPRINTF  _T("DEF %s OrientationInterpolator {\n"), name);
         break;
        case KEY_COLOR:
            eps = float(1.0e-5);
            newKeys = reducePoint3Keys(posTimes, posKeys, eps);
            if (newKeys == 0)
                return;
            timeVals = posTimes;
            _stprintf(name, _T("%s-COLOR-INTERP"), mNodes.GetNodeName(node));
            Indent(level);
         MSTREAMPRINTF  _T("DEF %s ColorInterpolator {\n"), name);
         break;
        default:
            return;
        }
        //DebugBreak();
        bool foundTimeSensor = false;
        // Now check to see if a TimeSensor references this node
        int mindistance = 1000000;
        int minTS = -1;
        INodeList *l;
        INodeList *minl;

        for (l = mTimerList; l; l = l->GetNext())
        {
            TimeSensorObject *tso = (TimeSensorObject *)
                                        l->GetNode()->EvalWorldState(mStart)
                                            .obj;

            // find the timesensor closest to the node in the hierarchy
            for (int j = 0; j < tso->TimeSensorObjects.Count(); j++)
            {
                INode *anim = tso->TimeSensorObjects[j]->node;
                if (anim)
                {
                    int dist = NodeIsChildOf(node, anim, 0);
                    if (dist >= 0) // we have a timesensor
                    {
                        if (dist < mindistance) // it animates a group closer to the node we want to animate than the last one
                        {
                            minTS = j;
                            minl = l;
                            mindistance = dist;
                        }
                    }
                }
            }
        }
        if (minTS >= 0) // now add all Timesensors with same distance
        {
            for (l = mTimerList; l; l = l->GetNext())
            {
                TimeSensorObject *tso = (TimeSensorObject *)
                                            l->GetNode()->EvalWorldState(mStart)
                                                .obj;

                // find the timesensor closest to the node in the hierarchy
                for (int j = 0; j < tso->TimeSensorObjects.Count(); j++)
                {
                    INode *anim = tso->TimeSensorObjects[j]->node;
                    if (anim)
                    {
                        int dist = NodeIsChildOf(node, anim, 0);
                        if (dist >= 0) // we have a timesensor
                        {
                            if (dist == mindistance) // it animates a group closer to the node we want to animate than the last one
                            {

                                TSTR oTimer = mTimer;
                                TCHAR timer[MAX_PATH];
                                _stprintf(timer, _T("%s"), mNodes.GetNodeName(l->GetNode()));
                                foundTimeSensor = true;
                                if (tso->needsScript)
                                    AddInterpolator(name, KEY_TIMER_SCRIPT, timer, l->GetNode());
                                else
                                    AddInterpolator(name, KEY_TIMER, timer, l->GetNode());
                            }
                        }
                    }
                }
            }
        }

        //Check if a TabletUIElement references this node
        for (l = mTabletUIList; l; l = l->GetNext())
        {
            TabletUIObject *th = (TabletUIObject *)
                                     l->GetNode()->EvalWorldState(mStart)
                                         .obj;

            const TCHAR *nodename = node->GetName();
            for (int i = 0; i < th->elements.Count(); i++)
                for (int j = 0; j < th->elements[i]->objects.Count(); j++)
                {
                    INode *nd = (INode *)th->elements[i]->objects[j]->node;
                    Object *o = nd->EvalWorldState(mStart).obj;

                    if (nd == node)
                    {
                        if (th->elements[i]->type == TUIButton)
                            AddInterpolator(name, KEY_TABLETUI_BUTTON, nd->NodeName().data(), nd);
                        else if (th->elements[i]->type == TUIFloatSlider)
                            AddInterpolator(name, KEY_TABLETUI_SLIDER, th->elements[i]->name.data(), nd);
                        else
                            AddInterpolator(th->elements[i]->name.data(), KEY_TABLETUI_SCRIPT, nd->NodeName(), nd);
                    }
                    if (th->elements[i]->type == TUIFloatSlider)
                    {
                        if (o->ClassID() == TimeSensorClassID)
                        {
                            TimeSensorObject *ts = static_cast<TimeSensorObject *>(o);
                            int l = 0;
                            for (; l < ts->TimeSensorObjects.Count(); l++)
                                if (ts->TimeSensorObjects[l]->node == node)
                                {
                                    int routes = mInterpRoutes.Count();
                                    for (int l = 0; l < routes; l++)
                                    {
                                        if (mInterpRoutes[l].mNodeName == nd->NodeName())
                                            if (mInterpRoutes[l].mType == KEY_TIMER)
                                            {
                                                AddInterpolator(mInterpRoutes[l].mInterp.data(), KEY_TABLETUI_SLIDER, th->elements[i]->name.data(), nd);
                                            }
                                            else if (mInterpRoutes[l].mType == KEY_TIMER_SCRIPT)
                                            {
                                                AddInterpolator(th->elements[i]->name.data(), KEY_TABLETUI_SLIDER_SCRIPT, nd->NodeName().data(), nd);
                                            }
                                            else if ((mInterpRoutes[l].mType == (KEY_TIMER | KEY_TABLETUI_TOGGLE)) || (mInterpRoutes[l].mType == (KEY_TIMER_SCRIPT | KEY_TABLETUI_TOGGLE)))
                                                AddInterpolator(mInterpRoutes[l].mInterp.data(), KEY_TABLETUI, th->elements[i]->name.data(), nd);
                                    }
                                }
                        }
                    }
                    else if ((th->elements[i]->type == TUIToggleButton) && (o->ClassID() == TimeSensorClassID))
                    {
                        TimeSensorObject *ts = static_cast<TimeSensorObject *>(o);
                        int l = 0;
                        for (; l < ts->TimeSensorObjects.Count(); l++)
                            if (ts->TimeSensorObjects[l]->node == node)
                            {
                                int k = 0;
                                for (; k < mInterpRoutes.Count(); k++)
                                    if ((mInterpRoutes[k].mInterp == th->elements[i]->name) && (mInterpRoutes[k].mNodeName == nd->NodeName()))
                                        break;
                                if (k == mInterpRoutes.Count())
                                    AddInterpolator(th->elements[i]->name.data(), KEY_TABLETUI_TOGGLE, nd->NodeName().data(), nd);
                            }
                        int routes = mInterpRoutes.Count();
                        for (l = 0; l < routes; l++)
                            if (mInterpRoutes[l].mNodeName == nd->NodeName())
                                if ((mInterpRoutes[l].mType == KEY_TIMER) || (mInterpRoutes[l].mType == KEY_TIMER_SCRIPT))
                                    mInterpRoutes[l].mType = mInterpRoutes[l].mType | KEY_TABLETUI_TOGGLE;
                                else if (mInterpRoutes[l].mType == KEY_TABLETUI)
                                    AddInterpolator(mInterpRoutes[l].mInterp.data(), KEY_TIMER | KEY_TABLETUI_TOGGLE, mInterpRoutes[l].mNodeName.data(), nd);
                    }
                }
        }

        //if(!foundTimeSensor)
        AddInterpolator(name, type, mNodes.GetNodeName(node), node);

        // Output the key times
        mCycleInterval = (mIp->GetAnimRange().End() - mStart) / ((float)GetTicksPerFrame() * GetFrameRate());
        Indent(level + 1);
      MSTREAMPRINTF  _T("key ["));
      width = mIndent ? level * 2 : 0;
      for (i = 0; i < timeVals.Count(); i++)
      {
          t = timeVals[i] - mStart;
          if (t < 0)
              continue;
         width += MSTREAMPRINTF  _T("%s, "),
            floatVal(t / ((float) GetTicksPerFrame()
            * GetFrameRate() * mCycleInterval)));
         if (width > 60)
         {
            MSTREAMPRINTF  _T("\n"));
            Indent(level + 3);
            width = mIndent ? level * 2 : 0;
         }
      }
      MSTREAMPRINTF  _T("]\n"));
      Indent(level + 1);
      MSTREAMPRINTF  _T("keyValue ["));

      width = mIndent ? level * 2 : 0;
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
            width += MSTREAMPRINTF  _T("%s, "), point(p));
            break;

          case KEY_COLOR:
              mHadAnim = TRUE;
              p = posKeys[i];
            width += MSTREAMPRINTF  _T("%s, "), color(p));
            break;

          case KEY_ROT:
              mHadAnim = TRUE;
              rval = rotKeys[i];
            width += MSTREAMPRINTF  _T("%s, "),
               axisPoint(rval.axis, -rval.angle));
            break;
          case KEY_SCL:
              mHadAnim = TRUE;
              s = sclKeys[i].s;
            width += MSTREAMPRINTF  _T("%s, "), scalePoint(s));
            break;
          case KEY_SCL_ORI:
              mHadAnim = TRUE;
              q = sclKeys[i].q;
              AngAxisFromQa(q, &rval.angle, rval.axis);
            width += MSTREAMPRINTF  _T("%s, "),
               axisPoint(rval.axis, -rval.angle));
            break;
          }
          if (width > 50)
          {
            MSTREAMPRINTF  _T("\n"));
            Indent(level + 2);
            width = mIndent ? level * 2 : 0;
          }
      }

      //    Indent(level);
      MSTREAMPRINTF  _T("] },\n"));

      type = (type == KEY_SCL ? KEY_SCL_ORI : 0);
    } // while (type)
    return;
}

void
VRML2Export::WriteAllControllerData(INode *node, int flags, int level,
                                    Control *lc)
{
    //TCHAR *name = node->GetName();    // for debugging
    float eps = float(1.0e-5);
    int i;
    int scalinc = 0;
    TimeValue t, prevT;
    TimeValue end = mIp->GetAnimRange().End();
    int frames;
    Point3 p, axis;
    ScaleValue s;
    Quat q;
    Quat oldu(0.0, 0.0, 0.0, 1.0);
    Matrix3 tm, ip;
    float ang;
    BOOL isCamera = IsCamera(node);
    int sampleRate;

    if (mTformSample)
        sampleRate = GetTicksPerFrame();
    else
        sampleRate = TIME_TICKSPERSEC / mTformSampleRate;
    frames = (end - mStart) / sampleRate + 1;

    int realEnd = end;
    if (((end - mStart) % sampleRate) != 0)
    {
        end += sampleRate;
        frames++;
    }

    // Tables of keyframe values
    Tab<Point3> posKeys;
    Tab<TimeValue> posTimes;
    Tab<ScaleValue> scaleKeys;
    Tab<TimeValue> scaleTimes;
    Tab<AngAxis> rotKeys;
    Tab<TimeValue> rotTimes;

    // Set up 'k' to point at the right derived class
    if (flags & KEY_POS)
    {
        posKeys.SetCount(frames);
        posTimes.SetCount(frames);
    }
    if (flags & KEY_ROT)
    {
        rotKeys.SetCount(frames);
        rotTimes.SetCount(frames);
    }
    if (flags & KEY_SCL)
    {
        scaleKeys.SetCount(frames);
        scaleTimes.SetCount(frames);
    }
    if (flags & KEY_COLOR)
    {
        posKeys.SetCount(frames);
        posTimes.SetCount(frames);
    }

    for (i = 0, t = mStart; i < frames; i++, t += sampleRate)
    {
        if (t > realEnd)
            t = realEnd;
        if (flags & KEY_COLOR)
        {
            lc->GetValue(t, &posKeys[i], FOREVER);
            posTimes[i] = t;
            continue;
        }
        // otherwise we are sampling tform controller data

        AffineParts parts;
        if (!isCamera)
        {
            tm = GetLocalTM(node, t);
        }
        else
        {
            // We have a camera
            tm = node->GetObjTMAfterWSM(t);
        }
#ifdef DDECOMP
        d_decomp_affine(tm, &parts);
#else
        decomp_affine(tm, &parts); // parts is parts
#endif

        if (flags & KEY_SCL)
        {
            s = ScaleValue(parts.k, parts.u);
            if (parts.f < 0.0f)
                s.s = -s.s;
#define AVOID_NEG_SCALE
#ifdef AVOID_NEG_SCALE
            if (s.s.x <= 0.0f && s.s.y <= 0.0f && s.s.z <= 0.0f)
            {
                s.s = -s.s;
                s.q = Conjugate(s.q);
            }
#endif
            // The following unholy kludge deals with the surprising fact
            // that, as a TM changes gradually, decomp_affine() may introduce
            // a sudden flip of sign in U at the same time as a jump in the
            // scale orientation axis.
            if (parts.u.x * oldu.x < -eps || parts.u.y * oldu.y < -eps || parts.u.z * oldu.z < -eps)
            {
                AffineParts pts;
                Matrix3 mat;
                TimeValue lowt, hight, midt;
                ScaleValue sv;
                int ct = scaleTimes.Count();

                for (hight = t, lowt = prevT;
                     hight - lowt > 1; // 1/4800 sec.
                     )
                {
                    midt = (hight + lowt) / 2;
                    if (!isCamera)
                        mat = GetLocalTM(node, midt);
                    else
                        mat = node->GetObjTMAfterWSM(midt);
#ifdef DDECOMP
                    d_decomp_affine(mat, &pts);
#else
                    decomp_affine(mat, &pts);
#endif
                    if (pts.u.x * oldu.x < -eps || pts.u.y * oldu.y < -eps || pts.u.z * oldu.z < -eps)
                        hight = midt;
                    else
                        lowt = midt;
                }
                if (lowt > prevT)
                {
                    if (!isCamera)
                        mat = GetLocalTM(node, lowt);
                    else
                        mat = node->GetObjTMAfterWSM(lowt);
#ifdef DDECOMP
                    d_decomp_affine(mat, &pts);
#else
                    decomp_affine(mat, &pts);
#endif
                    sv = ScaleValue(pts.k, pts.u);
                    if (pts.f < 0.0f)
                        sv.s = -sv.s;
#ifdef AVOID_NEG_SCALE
                    if (sv.s.x <= 0.0f && sv.s.y <= 0.0f && sv.s.z <= 0.0f)
                    {
                        sv.s = -sv.s;
                        sv.q = Conjugate(sv.q);
                    }
#endif
                    ct++;
                    scaleTimes.SetCount(ct);
                    scaleKeys.SetCount(ct);
                    scaleTimes[i + scalinc] = midt;
                    scaleKeys[i + scalinc] = sv;
                    scalinc++;
                }
                if (hight < t)
                {
                    if (!isCamera)
                        mat = GetLocalTM(node, hight);
                    else
                        mat = node->GetObjTMAfterWSM(hight);
#ifdef DDECOMP
                    d_decomp_affine(mat, &pts);
#else
                    decomp_affine(mat, &pts);
#endif
                    sv = ScaleValue(pts.k, pts.u);
                    if (pts.f < 0.0f)
                        sv.s = -sv.s;
#ifdef AVOID_NEG_SCALE
                    if (sv.s.x <= 0.0f && sv.s.y <= 0.0f && sv.s.z <= 0.0f)
                    {
                        sv.s = -sv.s;
                        sv.q = Conjugate(sv.q);
                    }
#endif
                    ct++;
                    scaleTimes.SetCount(ct);
                    scaleKeys.SetCount(ct);
                    scaleTimes[i + scalinc] = midt;
                    scaleKeys[i + scalinc] = sv;
                    scalinc++;
                }
            }
            if (parts.u.x != 0.0f)
                oldu.x = parts.u.x;
            if (parts.u.y != 0.0f)
                oldu.y = parts.u.y;
            if (parts.u.z != 0.0f)
                oldu.z = parts.u.z;

            scaleTimes[i + scalinc] = t;
            scaleKeys[i + scalinc] = s;
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
            if (isCamera && !mZUp)
            {
                // Now rotate around the X Axis PI/2
                Matrix3 rot = RotateXMatrix(PI / 2);
                Quat qRot(rot);
                AngAxisFromQa(q / qRot, &ang, axis);
            }
            else
                AngAxisFromQa(q, &ang, axis);
            rotTimes[i] = t;
            rotKeys[i] = AngAxis(axis, ang);
        }
        prevT = t;
    }
    if (flags & KEY_POS)
    {
        WriteControllerData(node,
                            posTimes, posKeys,
                            rotTimes, rotKeys,
                            scaleTimes, scaleKeys,
                            KEY_POS, level);
    }
    if (flags & KEY_ROT)
    {
        WriteControllerData(node,
                            posTimes, posKeys,
                            rotTimes, rotKeys,
                            scaleTimes, scaleKeys,
                            KEY_ROT, level);
    }
    if (flags & KEY_SCL && !isCamera)
    {
        WriteControllerData(node,
                            posTimes, posKeys,
                            rotTimes, rotKeys,
                            scaleTimes, scaleKeys,
                            KEY_SCL, level);
    }
    if (flags & KEY_COLOR)
    {
        WriteControllerData(node,
                            posTimes, posKeys,
                            rotTimes, rotKeys,
                            scaleTimes, scaleKeys,
                            KEY_COLOR, level);
    }
}

void
VRML2Export::WriteVisibilityData(INode *node, int level)
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
VRML2Export::IsLight(INode *node)
{
    Object *obj = node->EvalWorldState(mStart).obj;
    if (!obj)
        return FALSE;

    SClass_ID sid = obj->SuperClassID();
    return sid == LIGHT_CLASS_ID;
}

BOOL
VRML2Export::IsCamera(INode *node)
{
    Object *obj = node->EvalWorldState(mStart).obj;
    if (!obj)
        return FALSE;

    SClass_ID sid = obj->SuperClassID();
    return sid == CAMERA_CLASS_ID;
}

BOOL
VRML2Export::IsAudio(INode *node)
{
    Object *obj = node->EvalWorldState(mStart).obj;
    if (!obj)
        return FALSE;

    Class_ID cid = obj->ClassID();
    return cid == AudioClipClassID;
}

Control *
VRML2Export::GetLightColorControl(INode *node)
{
    if (!IsLight(node))
        return NULL;
    Object *obj = node->EvalWorldState(mStart).obj;
    IParamBlock *pblock = (IParamBlock *)obj->SubAnim(0);
    Control *cont = pblock->GetController(0); // I know color is index 0!
    return cont;
}

#define NeedsKeys(nkeys) ((nkeys) > 1 || (nkeys) == NOT_KEYFRAMEABLE)

// Write out keyframe data, if it exists
void
VRML2Export::VrmlOutControllers(INode *node, int level)
{

#ifdef _LEC_
    if (mFlipBook)
        return;
#endif
    Control *pc, *rc, *sc, *lc;
    int npk = 0, nrk = 0, nsk = 0, nvk = 0, nlk = 0;

    int flags = 0;
    BOOL isCamera = IsCamera(node);
    Object *obj = node->EvalWorldState(mStart).obj;
    int ts = NodeNeedsTimeSensor(node);

    if (ts != 0)
    {
        mCycleInterval = (mIp->GetAnimRange().End() - mStart) / ((float)GetTicksPerFrame() * GetFrameRate());
        Indent(level);
      MSTREAMPRINTF 
         _T("DEF %s-TIMER TimeSensor { loop %s cycleInterval %s },\n"),
         mNodes.GetNodeName(node),
         (ts < 0) ? _T("TRUE") : _T("FALSE"),
         floatVal(mCycleInterval));
    }

    lc = GetLightColorControl(node);
    if (lc)
        nlk = lc->NumKeys();
    if (NeedsKeys(nlk))
        WriteAllControllerData(node, KEY_COLOR, level, lc);

    Class_ID id = node->GetTMController()->ClassID();
    Class_ID pid;
    Class_ID rid;
    Class_ID sid;
    if (id == Class_ID(PRS_CONTROL_CLASS_ID, 0))
    {
        pc = node->GetTMController()->GetPositionController();
        if (pc)
            pid = pc->ClassID();
        rc = node->GetTMController()->GetRotationController();
        if (rc)
            rid = rc->ClassID();
        sc = node->GetTMController()->GetScaleController();
        if (sc)
            sid = sc->ClassID();
    }

    if (!node->IsAnimated() && id == Class_ID(PRS_CONTROL_CLASS_ID, 0) && rid == Class_ID(HYBRIDINTERP_POINT4_CLASS_ID, 0) && !isCamera && !IsLight(node))
        return;

#ifdef _DEBUG
    int inhf = node->GetTMController()->GetInheritanceFlags();
    int inhb = node->GetTMController()->InheritsParentTransform();
#endif

    if (!isCamera && id != Class_ID(PRS_CONTROL_CLASS_ID, 0))
        flags = KEY_POS | KEY_ROT | KEY_SCL;
    else if (isCamera && (node->IsAnimated() || IsEverAnimated(node->GetTarget())))
        flags = KEY_POS | KEY_ROT;
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
        if (NeedsKeys(npk) || NeedsKeys(nrk) || NeedsKeys(nsk))
            flags = KEY_POS | KEY_ROT | KEY_SCL;
    }
    if (flags)
        WriteAllControllerData(node, flags, level, NULL);
#if 0
   Control* vc = node->GetVisController();
   if (vc) nvk = vc->NumKeys();
   if (NeedsKeys(nvk))
      WriteVisibilityData(node, level);
#endif
}

void
VRML2Export::VrmlOutTopLevelCamera(int level, INode *node, BOOL topLevel)
{
    if (node != mCamera)
        return;
    if (!doExport(node))
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
        AngAxisFromQa(q / qRot, &ang, axis);
    }
    else
        AngAxisFromQa(q, &ang, axis);

    // compute camera transform
    ViewParams vp;
    CameraState cs;
    Interval iv;
    cam->EvalCameraState(0, iv, &cs);
    vp.fov = (float)(2.0 * atan(tan(cs.fov / 2.0) / INTENDED_ASPECT_RATIO));

    Indent(level);
   MSTREAMPRINTF  _T("DEF %s Viewpoint {\n"), mNodes.GetNodeName(node));
   Indent(level + 1);
   MSTREAMPRINTF  _T("position %s\n"), point(p));
   Indent(level + 1);
   MSTREAMPRINTF  _T("orientation %s\n"), axisPoint(axis, -ang));
   Indent(level + 1);
   MSTREAMPRINTF  _T("fieldOfView %s\n"), floatVal(vp.fov));
   Indent(level + 1);
   MSTREAMPRINTF  _T("description \"%s\"\n"), mNodes.GetNodeName(node));
   Indent(level);
   MSTREAMPRINTF  _T("}\n"));

   InitInterpolators(node);
   VrmlOutControllers(node, 0);
   WriteInterpolatorRoutes(level);
}

// In navinfo.cpp
extern TCHAR *navTypes[];

void
VRML2Export::VrmlOutTopLevelNavInfo(int level, INode *node, BOOL topLevel)
{
    if (!topLevel && node == mNavInfo)
        return;
    if (!doExport(node))
        return;

    NavInfoObject *ni = (NavInfoObject *)node->EvalWorldState(mStart).obj;
    int type, headlight;
    float visLimit, speed, collision, terrain, step, scale, nearC, farC;
    ni->pblock->GetValue(PB_TYPE, mIp->GetTime(), type, FOREVER);
    ni->pblock->GetValue(PB_HEADLIGHT, mIp->GetTime(), headlight, FOREVER);
    ni->pblock->GetValue(PB_VIS_LIMIT, mIp->GetTime(), visLimit, FOREVER);
    ni->pblock->GetValue(PB_SPEED, mIp->GetTime(), speed, FOREVER);
    ni->pblock->GetValue(PB_COLLISION, mIp->GetTime(), collision, FOREVER);
    ni->pblock->GetValue(PB_TERRAIN, mIp->GetTime(), terrain, FOREVER);
    ni->pblock->GetValue(PB_STEP, mIp->GetTime(), step, FOREVER);
    ni->pblock->GetValue(PB_NI_SCALE, mIp->GetTime(), scale, FOREVER);
    ni->pblock->GetValue(PB_NI_NEAR, mIp->GetTime(), nearC, FOREVER);
    ni->pblock->GetValue(PB_NI_FAR, mIp->GetTime(), farC, FOREVER);
    Indent(level);
   MSTREAMPRINTF  _T("DEF %s NavigationInfo {\n"),mNodes.GetNodeName(node));
   Indent(level + 1);
   MSTREAMPRINTF  _T("avatarSize [%s, "), floatVal(collision));
   MSTREAMPRINTF  _T("%s, "), floatVal(terrain));
   MSTREAMPRINTF  _T("%s]\n"), floatVal(step));
   Indent(level + 1);
   MSTREAMPRINTF  _T("headlight %s\n"),
      headlight ? _T("TRUE") : _T("FALSE"));
   Indent(level + 1);
   if ((floatVal(scale >= 1.0)) && (mType == Export_VRML_2_0_COVER))
   {
      MSTREAMPRINTF  _T("scale %s\n"), floatVal(scale));
      Indent(level + 1);
   }
   MSTREAMPRINTF  _T("speed %s\n"), floatVal(speed));
   Indent(level + 1);
   if (farC > 0)
   {
      MSTREAMPRINTF  _T("near %s\n"), floatVal(nearC));
      Indent(level + 1);
      MSTREAMPRINTF  _T("far %s\n"), floatVal(farC));
      Indent(level + 1);
   }
   if (type < 0 || type > 3)
       type = 0;
   MSTREAMPRINTF  _T("type \"%s\"\n"), navTypes[type]);
   Indent(level + 1);
   MSTREAMPRINTF  _T("visibilityLimit %s\n"), floatVal(visLimit));
   Indent(level);
   MSTREAMPRINTF  _T("}\n"));
}

void
VRML2Export::VrmlOutTopLevelBackground(int level, INode *node, BOOL topLevel)
{
    if (!topLevel && node == mBackground)
        return;
    if (!doExport(node))
        return;

    BackgroundObject *bg = (BackgroundObject *)
                               node->EvalWorldState(mStart).obj;
    int numColors, i;
    Point3 col[3];
    float angle2, angle3;

    bg->pblock->GetValue(PB_SKY_NUM_COLORS, mIp->GetTime(), numColors,
                         FOREVER);
    bg->pblock->GetValue(PB_SKY_COLOR1, mIp->GetTime(), col[0], FOREVER);
    bg->pblock->GetValue(PB_SKY_COLOR2, mIp->GetTime(), col[1], FOREVER);
    bg->pblock->GetValue(PB_SKY_COLOR3, mIp->GetTime(), col[2], FOREVER);
    bg->pblock->GetValue(PB_SKY_COLOR2_ANGLE, mIp->GetTime(), angle2,
                         FOREVER);
    bg->pblock->GetValue(PB_SKY_COLOR3_ANGLE, mIp->GetTime(), angle3,
                         FOREVER);

    Indent(level);
   MSTREAMPRINTF  _T("DEF %s Background {\n"),mNodes.GetNodeName(node));
   Indent(level + 1);
   MSTREAMPRINTF  _T("skyColor ["));
   for (i = 0; i < numColors + 1; i++)
      MSTREAMPRINTF  _T("%s, "), color(col[i]));
   MSTREAMPRINTF  _T("]\n"));

   if (numColors > 0)
   {
       Indent(level + 1);
      MSTREAMPRINTF  _T("skyAngle ["));
      MSTREAMPRINTF  _T("%s, "), floatVal(angle2));
      if (numColors > 1)
         MSTREAMPRINTF  _T("%s, "), floatVal(angle3));
      MSTREAMPRINTF  _T("]\n"));
   }

   bg->pblock->GetValue(PB_GROUND_NUM_COLORS, mIp->GetTime(), numColors,
                        FOREVER);
   bg->pblock->GetValue(PB_GROUND_COLOR1, mIp->GetTime(), col[0], FOREVER);
   bg->pblock->GetValue(PB_GROUND_COLOR2, mIp->GetTime(), col[1], FOREVER);
   bg->pblock->GetValue(PB_GROUND_COLOR3, mIp->GetTime(), col[2], FOREVER);
   bg->pblock->GetValue(PB_GROUND_COLOR2_ANGLE, mIp->GetTime(), angle2,
                        FOREVER);
   bg->pblock->GetValue(PB_GROUND_COLOR3_ANGLE, mIp->GetTime(), angle3,
                        FOREVER);

   Indent(level + 1);
   MSTREAMPRINTF  _T("groundColor ["));
   for (i = 0; i < numColors + 1; i++)
      MSTREAMPRINTF  _T("%s, "), color(col[i]));
   MSTREAMPRINTF  _T("]\n"));

   if (numColors > 0)
   {
       Indent(level + 1);
      MSTREAMPRINTF  _T("groundAngle ["));
      MSTREAMPRINTF  _T("%s, "), floatVal(angle2));
      if (numColors > 1)
         MSTREAMPRINTF  _T("%s, "), floatVal(angle3));
      MSTREAMPRINTF  _T("]\n"));
   }

   TSTR url;
   if (bg->back.Length() > 0)
   {
       Indent(level + 1);
       url = PrefixUrl(bg->back);
      MSTREAMPRINTF  _T("backUrl \"%s\"\n"), url.data());
   }
   if (bg->bottom.Length() > 0)
   {
       Indent(level + 1);
       url = PrefixUrl(bg->bottom);
      MSTREAMPRINTF  _T("bottomUrl \"%s\"\n"), url.data());
   }
   if (bg->front.Length() > 0)
   {
       Indent(level + 1);
       url = PrefixUrl(bg->front);
      MSTREAMPRINTF  _T("frontUrl \"%s\"\n"), url.data());
   }
   if (bg->left.Length() > 0)
   {
       Indent(level + 1);
       url = PrefixUrl(bg->left);
      MSTREAMPRINTF  _T("leftUrl \"%s\"\n"), url.data());
   }
   if (bg->right.Length() > 0)
   {
       Indent(level + 1);
       url = PrefixUrl(bg->right);
      MSTREAMPRINTF  _T("rightUrl \"%s\"\n"), url.data());
   }
   if (bg->top.Length() > 0)
   {
       Indent(level + 1);
       url = PrefixUrl(bg->top);
      MSTREAMPRINTF  _T("topUrl \"%s\"\n"), url.data());
   }
   Indent(level);
   MSTREAMPRINTF  _T("}\n"));
}

void
VRML2Export::VrmlOutTopLevelFog(int level, INode *node, BOOL topLevel)
{
    if (!topLevel && node == mFog)
        return;
    if (!doExport(node))
        return;

    FogObject *fog = (FogObject *)node->EvalWorldState(mStart).obj;
    Point3 p;
    float visLimit;
    int type;
    fog->pblock->GetValue(PB_COLOR, mIp->GetTime(), p, FOREVER);
    fog->pblock->GetValue(PB_TYPE, mIp->GetTime(), type, FOREVER);
    fog->pblock->GetValue(PB_VIS_LIMIT, mIp->GetTime(), visLimit, FOREVER);
    Indent(level);
   MSTREAMPRINTF  _T("DEF %s Fog {\n"), mNodes.GetNodeName(node));
   Indent(level + 1);
   MSTREAMPRINTF  _T("color %s\n"), color(p));
   Indent(level + 1);
   MSTREAMPRINTF  _T("fogType \"%s\"\n"), type == 0 ? _T("LINEAR") :
      _T("EXPONENTIAL"));
   Indent(level + 1);
   MSTREAMPRINTF  _T("visibilityRange %s\n"), floatVal(visLimit));
   Indent(level);
   MSTREAMPRINTF  _T("}\n"));
}

void
VRML2Export::VrmlOutTopLevelSky(int level, INode *node, BOOL topLevel)
{
    if (!topLevel && node == mSky)
        return;
    if (!doExport(node))
        return;

    SkyObject *sky = (SkyObject *)node->EvalWorldState(mStart).obj;
    int enabled;
    int currentTime;
    int timeLapse;
    int year;
    int month;
    int day;
    int hour;
    int minute;
    float latitude;
    float longitude;
    float altitude;
    float radius;
    sky->pblock->GetValue(PB_SKY_ENABLED, mIp->GetTime(), enabled, FOREVER);
    sky->pblock->GetValue(PB_SKY_CURRENTTIME, mIp->GetTime(), currentTime, FOREVER);
    sky->pblock->GetValue(PB_SKY_TIMELAPSE, mIp->GetTime(), timeLapse, FOREVER);
    sky->pblock->GetValue(PB_SKY_YEAR, mIp->GetTime(), year, FOREVER);
    sky->pblock->GetValue(PB_SKY_MONTH, mIp->GetTime(), month, FOREVER);
    sky->pblock->GetValue(PB_SKY_DAY, mIp->GetTime(), day, FOREVER);
    sky->pblock->GetValue(PB_SKY_HOUR, mIp->GetTime(), hour, FOREVER);
    sky->pblock->GetValue(PB_SKY_MINUTE, mIp->GetTime(), minute, FOREVER);
    sky->pblock->GetValue(PB_SKY_LATITUDE, mIp->GetTime(), latitude, FOREVER);
    sky->pblock->GetValue(PB_SKY_LONGITUDE, mIp->GetTime(), longitude, FOREVER);
    sky->pblock->GetValue(PB_SKY_ALTITUDE, mIp->GetTime(), altitude, FOREVER);
    sky->pblock->GetValue(PB_SKY_RADIUS, mIp->GetTime(), radius, FOREVER);
    Indent(level);
   MSTREAMPRINTF  _T("DEF %s Sky {\n"), mNodes.GetNodeName(node));
   Indent(level + 1);
   MSTREAMPRINTF  _T("enabled %s\n"), enabled!=0 ? _T("TRUE") : _T("FALSE"));
   Indent(level + 1);
   MSTREAMPRINTF  _T("currentTime %s\n"), currentTime!=0 ? _T("TRUE") : _T("FALSE"));
   Indent(level + 1);
   MSTREAMPRINTF  _T("timeLapse %s\n"), timeLapse!=0 ? _T("TRUE") : _T("FALSE"));
   Indent(level + 1);
   MSTREAMPRINTF  _T("radius %s\n"), floatVal(radius));
   Indent(level + 1);
   MSTREAMPRINTF  _T("year %d\n"), year);
   Indent(level + 1);
   MSTREAMPRINTF  _T("month %d\n"), month);
   Indent(level + 1);
   MSTREAMPRINTF  _T("day %d\n"), day);
   Indent(level + 1);
   MSTREAMPRINTF  _T("hour %d\n"), hour);
   Indent(level + 1);
   MSTREAMPRINTF  _T("minute %d\n"), minute);
   Indent(level + 1);
   MSTREAMPRINTF  _T("latitude %s\n"), floatVal(latitude));
   Indent(level + 1);
   MSTREAMPRINTF  _T("longitude %s\n"), floatVal(longitude));
   Indent(level + 1);
   MSTREAMPRINTF  _T("altitude %s\n"), floatVal(altitude));
   Indent(level);
   MSTREAMPRINTF  _T("}\n"));
}

void
VRML2Export::VrmlOutInitializeAudioClip(int level, INode *node)
{
    AudioClipObject *ac = (AudioClipObject *)node->EvalWorldState(mStart).obj;
    if (ac)
        ac->written = 0;
}

void
VRML2Export::VrmlOutAudioClip(int level, INode *node)
{
    float pitch;
    int loop, start;
    AudioClipObject *ac = (AudioClipObject *)node->EvalWorldState(mStart).obj;
    if (ac->written)
    {
        Indent(level);
      MSTREAMPRINTF  _T("USE %s\n"), mNodes.GetNodeName(node));
      return;
    }
    ac->pblock->GetValue(PB_AC_PITCH, mIp->GetTime(), pitch, FOREVER);
    ac->pblock->GetValue(PB_AC_LOOP, mIp->GetTime(), loop, FOREVER);
    ac->pblock->GetValue(PB_AC_START, mIp->GetTime(), start, FOREVER);

    Indent(level);
   MSTREAMPRINTF  _T("DEF %s AudioClip {\n"), mNodes.GetNodeName(node));
   Indent(level + 1);
   MSTREAMPRINTF  _T("description \"%s\"\n"), ac->desc.data());
   Indent(level + 1);
   MSTREAMPRINTF  _T("url \"%s\"\n"), ac->url.data());
   Indent(level + 1);
   MSTREAMPRINTF  _T("pitch %s\n"), floatVal(pitch));
   Indent(level + 1);
   MSTREAMPRINTF  _T("loop %s\n"), loop ? _T("TRUE") : _T("FALSE"));
   Indent(level + 1);
   if (start)
      MSTREAMPRINTF  _T("startTime 1\n"));
   else
      MSTREAMPRINTF  _T("stopTime 1\n"));
   Indent(level);
   MSTREAMPRINTF  _T("}\n"));
   ac->written = TRUE;
}

// From dllmain.cpp
extern HINSTANCE hInstance;

void
VRML2Export::VrmlOutFileInfo()
{
    TCHAR filename[MAX_PATH];
    DWORD size, dummy;
    float vernum = 2.0f;
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
    if (mType == Export_X3D_V)
    {
      MSTREAMPRINTF  _T("META \"generator\" \"Uwes VRML Plugin, Version 8, Revision 06, .NET\"\n"));
    }
    else
    {
      MSTREAMPRINTF  _T("# Produced by Uwes VRML Plugin, Version 8, Revision 06, .NET\n"),vernum, betanum);
    }

    time_t ltime;
    time(&ltime);
    TCHAR *time = _tctime(&ltime);
    // strip the CR
    time[_tcslen(time) - 1] = _T('\0');
    const TCHAR *fn = mIp->GetCurFileName().data();
    if (fn && _tcslen(fn) > 0)
    {
        if (mType == Export_X3D_V)
        {
         MSTREAMPRINTF  _T("META \"reference\" \"MAX File: %s, Date: %s\"\n\n"), fn, time);
        }
        else
        {
         MSTREAMPRINTF  _T("# MAX File: %s, Date: %s\n\n"), fn, time);
        }
    }
    else
    {
        if (mType == Export_X3D_V)
        {
         MSTREAMPRINTF  _T("META \"modified\" \"Date: %s\"\n\n"), time);
        }
        else
        {
         MSTREAMPRINTF  _T("# Date: %s\n\n"), time);
        }
    }
}

void
VRML2Export::VrmlOutWorldInfo()
{
    if (mTitle.Length() == 0 && mInfo.Length() == 0)
        return;

   MSTREAMPRINTF  _T("WorldInfo {\n"));
   if (mTitle.Length() != 0)
   {
       Indent(1);
      MSTREAMPRINTF  _T("title \"%s\"\n"), mTitle.data());
   }
   if (mInfo.Length() != 0)
   {
       Indent(1);
      MSTREAMPRINTF  _T("info \"%s\"\n"), mInfo.data());
   }
   MSTREAMPRINTF  _T("}\n"));
}

int
VRML2Export::StartAnchor(INode *node, int &level)
{
    SensorBucket *sb = mSensorTable.FindSensor(node);
    if (!sb)
        return 0;
    INode *sensor;
    INodeList *l;
    int numAnchors = 0;
    for (l = sb->mSensors; l; l = l->GetNext())
    {
        sensor = l->GetNode();
        Object *obj = sensor->EvalWorldState(mStart).obj;
        assert(obj);
        if (obj->ClassID() == AnchorClassID)
        {
            numAnchors++;
            AnchorObject *ao = (AnchorObject *)
                                   sensor->EvalWorldState(mStart).obj;
            Indent(level);
         MSTREAMPRINTF  _T("Anchor {\n"));
         level++;
         Indent(level);
         MSTREAMPRINTF  _T("description \"%s\"\n"), ao->description.data());
         int type;
         ao->pblock->GetValue(PB_AN_TYPE, mStart, type, FOREVER);
         if (type == 0)
         {
             Indent(level);
            MSTREAMPRINTF  _T("parameter \"%s\"\n"), ao->parameter.data());
            Indent(level);
            MSTREAMPRINTF  _T("url \"%s\"\n"), ao->URL.data());
         }
         else
         {
             if (ao->cameraObject)
             {
                 Indent(level);
               MSTREAMPRINTF  _T("url \"#%s\"\n"),
                  VRMLName(ao->cameraObject->GetName()));
             }
         }
         Indent(level);
         MSTREAMPRINTF  _T("children [\n"));
         level++;
        }
    }
    return numAnchors;
}

// Recursively count a node and all its children
static int
CountNodes(INode *node)
{
    int total, kids, i;

    if (node == NULL)
        return 0;
    total = 1;
    kids = node->NumberOfChildren();
    for (i = 0; i < kids; i++)
        total += CountNodes(node->GetChildNode(i));
    return total;
}

// Output a single node as VRML and recursively output the children of
// the node.
void
VRML2Export::VrmlOutNode(INode *node, INode *parent, int level, BOOL isLOD,
                         BOOL lastChild, BOOL mirrored)
{
    const TCHAR *nodeName = node->GetName();
    switchObjects.Append(1, &node);

    bool hasVisController = false;

    if (node->GetVisController())
        hasVisController = true;
    // Don't gen code for LOD references, only LOD nodes
    if (!isLOD && ObjectIsLODRef(node))
        return;

    if (mEnableProgressBar)
        SendMessage(hWndPDlg, 666, 0,
                    (LPARAM)mNodes.GetNodeName(node));

    BOOL outputName = TRUE;
    int numChildren = node->NumberOfChildren();
    Object *obj = node->EvalWorldState(mStart).obj;

    //ObjectBucket* ob = mObjTable.AddObject(obj,true); // check for instances
    BOOL isVrml = isVrmlObject(node, obj, parent, hasVisController);
    BOOL isCamera = IsCamera(node);
    BOOL numAnchors = 0;
    BOOL written = FALSE;
    BOOL mirror = FALSE;
    int cnt;

    if (node->IsRootNode())
    {
        VrmlOutWorldInfo();
        // Compute the world bounding box and a list of timesensors
        ScanSceneGraph1();

        if (mCamera && doExport(mCamera))
            VrmlOutTopLevelCamera(level + 2, mCamera, TRUE);
        if (mNavInfo && doExport(mNavInfo))
            VrmlOutTopLevelNavInfo(level + 2, mNavInfo, TRUE);
        if (mBackground && doExport(mBackground))
            VrmlOutTopLevelBackground(level + 2, mBackground, TRUE);
        if (mFog && doExport(mFog))
            VrmlOutTopLevelFog(level + 2, mFog, TRUE);
        if (mSky && doExport(mSky))
            VrmlOutTopLevelSky(level + 2, mSky, TRUE);

        // Make a list of al the LOD objects and texture maps in the scene.
        // Also output top-level objects
        ScanSceneGraph2();
        if (!mHasNavInfo)
        {
            if (mHasLights)
            {
            MSTREAMPRINTF  _T("NavigationInfo { headlight FALSE }\n"));
            }
            else
            {
            MSTREAMPRINTF  _T("NavigationInfo { headlight TRUE }\n"));
            }
        }
    }

    // give third party dlls a chance to write the node
    if (!node->IsRootNode())
    {
        written = FALSE;
        for (cnt = 0; cnt < mCallbacks->GetPreNodeCount(); cnt++)
        {
            DllPreNode preNode = mCallbacks->GetPreNode(cnt);
            PreNodeParam params;
            params.version = 0;
            params.indent = level;
            params.fName = mFilename;
            params.i = mIp;
            params.node = node;
#if MAX_PRODUCT_VERSION_MAJOR > 14
            if (mStream.IsFileOpen())
                mStream.Close();
#else
            if (mStream)
                fclose(mStream);
            mStream = NULL;
#endif
            written = (*(preNode))(&params);
            if (written)
                break; // only the first one gets to write the node
        }
#if MAX_PRODUCT_VERSION_MAJOR > 14
        if (!mStream.IsFileOpen())
            mStream.Open(mFilename, false, CP_UTF8);
#else
        if (!mStream)
            mStream = _tfopen(mFilename, _T("a"));
#endif
    }

    // Anchors need to come first, even though they are child nodes
    if (!node->IsRootNode() && !written)
    {
        /* test
      int newLevel = StartMrBlueHelpers(node, level);
      level = newLevel;
      */
        numAnchors = StartAnchor(node, level);
        // Initialize set of timers/interpolator per top-level node
        if (node->GetParentNode()->IsRootNode())
            InitInterpolators(node);
    }
    bool wroteSwitch = false;

    if (isVrml && (doExport(node) || hasVisController) && !written)
    {
        if (isCamera)
        {
            INode *sw = isSwitched(node);
            if (sw)
            {
                VrmlOutSwitchCamera(sw, node, level);
                VrmlOutControllers(node, level);
                Indent(level + 1);
            MSTREAMPRINTF  _T("]\n"));
            }
            else
                VrmlOutControllers(node, level);
        }
        else
        {
            StartNode(node, level, outputName, obj);
            if (!IsLODObject(obj))
                mirror = OutputNodeTransform(node, level + 1, mirrored);

            // Output the data for the object at this node
            Indent(level + 1);
         MSTREAMPRINTF  _T("children [\n"));
         if (!IsLODObject(obj))
         {
             // If the node has a controller, output the data
             VrmlOutControllers(node, level + 1);
         }
        }
    }
    bool LODWritten = false;

    if ((/*!isCamera &&*/ isVrml && ((mExportSelected && isChildSelected(node) && (!node->IsHidden() && !mExportHidden) || (!mExportSelected && (!node->IsHidden() && !mExportHidden))) || hasVisController) && !written) || IsAnimTrigger(obj))
    {
        VrmlOutObject(node, parent, obj, level + 2, mirrored ^ mirror);
    }

    if (mEnableProgressBar)
        SendMessage(hWndPB, PBM_STEPIT, 0, 0);

    // Now output the children
    if (!(written & WroteNodeChildren))
    {

        int todo = numChildren;
        int i;
        INode **children;
        if (numChildren)
        {
            children = new INode *[numChildren];
        }
        else
        {
            children = NULL;
        }
        int numLevels = 0;
        float distances[1000];
        for (i = 0; i < numChildren; i++)
        {
            children[i] = node->GetChildNode(i);
            //check to see if this is a Max LOD
            LODCtrl *visibility = dynamic_cast<LODCtrl *>(children[i]->GetVisController());
            if (visibility)
            {
                distances[numLevels] = visibility->max;
                numLevels++;
            }
        }
        if (numLevels > 1)
        {
            OutputMaxLOD(node, obj, level, numLevels, distances, children, numChildren, mirrored ^ mirror);
            level++;
            LODWritten = true;
        }
        INode *switchNode = NULL;
        for (i = 0; i < todo; i++)
        {
            //don´t write children with LOD controller, they should have already been written
            if (dynamic_cast<LODCtrl *>(children[i]->GetVisController()))
            {
                continue;
            }
            wroteSwitch = false;
            switchNode = isSwitched(children[i]);
            if (switchNode)
            {
                if ((doExport(node)) && !written)
                {
                    wroteSwitch = true;
                    level++;
                    numSwitchObjects = VrmlOutSwitch(switchNode, level) + 1;
                }
                int n = i, m;
                while (n < todo)
                {
                    if (isSwitched(children[n], switchNode))
                    {
                        VrmlOutNode(children[n], node, level + 2, FALSE,
                                    i == todo - 1, mirrored ^ mirror);
                        m = n + 1;
                        while (m < todo)
                        {
                            children[m - 1] = children[m];
                            m++;
                        }
                        todo--;
                    }
                    else
                    {
                        n++;
                    }
                }
                i--;

                if (wroteSwitch)
                {
                    Indent(level + 1);
               MSTREAMPRINTF  _T("] }\n"));
               level--;
                }
            }
            else
            {
                VrmlOutNode(children[i], node, level + 2, FALSE,
                            i == todo - 1, mirrored ^ mirror);
            }
        }
        delete[] children;

        /* already done in outputLOD if(LODWritten)
      {
      Indent(level+1);
      MSTREAMPRINTF  _T("] }\n"));
      level--;
      }*/
    }

    // need to get a valid obj ptr VrmlOutNode (VrmlOutCoordinateInterpolator)
    // causes the obj ptr (cache) to be invalid
    obj = node->EvalWorldState(mStart).obj;
    if (obj && (obj->ClassID() == BillboardClassID) && (numChildren > 0) && (doExport(node) || hasVisController) && !written)
    {
        Indent(level + 1);
      MSTREAMPRINTF  _T("] }\n"));
    }

    if (!node->IsRootNode() && !isCamera && isVrml && (doExport(node) || hasVisController) && !written)
    {
        OutputTouchSensors(node, level);
        OutputARSensors(node, level);
        OutputMTSensors(node, level);
        Indent(level + 1);
      MSTREAMPRINTF  _T("]\n"));
    }

    if (!node->IsRootNode() && !written)
    {
        OutputTabletUIScripts(node, level);
        if (node->GetParentNode()->IsRootNode())
        {
            /*  if (isCamera) WriteInterpolatorRoutes(level, TRUE);
         else*/ WriteInterpolatorRoutes(level); // must be in place of field
        }
    }
    if (isVrml && !node->IsRootNode() && (doExport(node) || hasVisController) && !written)
        EndNode(node, obj, level, lastChild);

    // give third party dlls a chance to finish up the node
    if (!node->IsRootNode())
    {
        for (cnt = 0; cnt < mCallbacks->GetPostNodeCount(); cnt++)
        {
            DllPostNode postNode = mCallbacks->GetPostNode(cnt);
            PostNodeParam params;
            params.version = 0;
            params.indent = level;
            params.fName = mFilename;
            params.i = mIp;
            params.node = node;
#if MAX_PRODUCT_VERSION_MAJOR > 14
            if (mStream.IsFileOpen())
                mStream.Close();
#else
            if (mStream)
                fclose(mStream);
            mStream = NULL;
#endif

            (*(postNode))(&params);
        }
#if MAX_PRODUCT_VERSION_MAJOR > 14
        if (!mStream.IsFileOpen())
            mStream.Open(mFilename, false, CP_UTF8);
#else
        if (!mStream)
            mStream = _tfopen(mFilename, _T("a"));
#endif
    }

    // End the anchors if needed
    if (!node->IsRootNode() && !written)
    {
        /* test
      EndMrBlueHelpers(node, level);
      */
        for (; numAnchors > 0; numAnchors--)
        {
            Indent(level);
         MSTREAMPRINTF  _T("] }\n"));
         level--;
        }
        //   if (node->GetParentNode()->IsRootNode())
        //       WriteInterpolatorRoutes(level, FALSE);
    }

    // DEFUSE objectsob->objectUsed = TRUE;
}

bool
VRML2Export::OutputSwitches(INode *node, int level)
{
    SensorBucket *sb = mSensorTable.FindSensor(node);
    if (!sb)
        return false;
    INode *sensor;
    INodeList *l;
    for (l = sb->mSensors; l; l = l->GetNext())
    {
        sensor = l->GetNode();
        Object *obj = sensor->EvalWorldState(mStart).obj;
        assert(obj);
        if (obj->ClassID() == SwitchClassID)
        {
            VrmlOutSwitch(sensor, level);
            return true;
        }
    }
    return false;
}
INode *
VRML2Export::isSwitched(INode *node, INode *firstNode)
{
    SensorBucket *sb = mSensorTable.FindSensor(node);
    if (!sb)
        return NULL;
    INode *sensor;
    INodeList *l;
    for (l = sb->mSensors; l; l = l->GetNext())
    {
        sensor = l->GetNode();
        if (firstNode)
        {
            if (firstNode == sensor)
                return sensor;
        }
        else
        {
            Object *obj = sensor->EvalWorldState(mStart).obj;
            assert(obj);
            if (obj->ClassID() == SwitchClassID)
            {
                return sensor;
            }
        }
    }
    return NULL;
}

void
VRML2Export::OutputTouchSensors(INode *node, int level)
{
    SensorBucket *sb = mSensorTable.FindSensor(node);
    if (!sb)
        return;
    INode *sensor;
    INodeList *l;
    for (l = sb->mSensors; l; l = l->GetNext())
    {
        sensor = l->GetNode();
        Object *obj = sensor->EvalWorldState(mStart).obj;
        assert(obj);
        if (obj->ClassID() == TouchSensorClassID)
            VrmlOutTouchSensor(sensor, level);
    }
}

void
VRML2Export::OutputARSensors(INode *node, int level)
{
    SensorBucket *sb = mSensorTable.FindSensor(node);
    if (!sb)
        return;
    INode *sensor;
    INodeList *l;
    for (l = sb->mSensors; l; l = l->GetNext())
    {
        sensor = l->GetNode();
        Object *obj = sensor->EvalWorldState(mStart).obj;
        assert(obj);
        if (obj->ClassID() == ARSensorClassID)
            VrmlOutARSensor(sensor, level);
        //	else if (obj->ClassID() == COVERClassID)
        //		VrmlOutCOVER(sensor, level+1);
    }
}
void
VRML2Export::OutputMTSensors(INode *node, int level)
{
    SensorBucket *sb = mSensorTable.FindSensor(node);
    if (!sb)
        return;
    INode *sensor;
    INodeList *l;
    for (l = sb->mSensors; l; l = l->GetNext())
    {
        sensor = l->GetNode();
        Object *obj = sensor->EvalWorldState(mStart).obj;
        assert(obj);
        if (obj->ClassID() == MultiTouchSensorClassID)
            VrmlOutMTSensor(sensor, level);
        //	else if (obj->ClassID() == COVERClassID)
        //		VrmlOutCOVER(sensor, level+1);
    }
}

bool
VRML2Export::OutputTabletUIScripts(INode *node, int level)
{
    SensorBucket *sb = mSensorTable.FindSensor(node);
    if (!sb)
        return false;
    TabletUIElement *tuielem;
    TabletUIElementList *l;
    for (l = sb->mTUIElems; l; l = l->GetNext())
    {
        tuielem = l->GetElem();
        if (tuielem->type == TUIButton)
        {
            Object *o = node->EvalWorldState(mStart).obj;
            if (!o)
                return false;
            if (o->ClassID() == TimeSensorClassID)
            {
                TimeSensorObject *to = (TimeSensorObject *)o;
                int numChildren = to->TimeSensorObjects.Count();
                for (int j = 0; j < numChildren; j++)
                {
                    if (to->TimeSensorObjects[j]->node->EvalWorldState(mStart).obj->SuperClassID() == CAMERA_CLASS_ID)
                        BindCamera(NULL, to->TimeSensorObjects[j]->node, VRMLName(tuielem->name), KEY_TOUCHSENSOR_BIND | KEY_TABLETUI_BUTTON, level);
                }
            }
            else
                BindCamera(NULL, node, VRMLName(tuielem->name), KEY_TOUCHSENSOR_BIND | KEY_TABLETUI_BUTTON, level);
        }
    }
    return true;
}

// Traverse the scene graph looking for LOD nodes and texture maps.
// Mark nodes affected by sensors (time, touch, proximity).
void
VRML2Export::TraverseNode(INode *node)
{
    if (!node)
        return;
    Object *obj = node->EvalWorldState(mStart).obj;
    Class_ID id;

    ObjectBucket *ob = mObjTable.AddObject(obj, true); // check for instances
    if (!doExport(node))
        return;
    if (obj)
    {
        id = obj->ClassID();
        if (id == Class_ID(LOD_CLASS_ID1, LOD_CLASS_ID2))
            mLodList = mLodList->AddNode(node);

        if (IsLight(node))
        {
            mHasLights = TRUE;
            if (doExport(node) && !IsEverAnimated(node) && !IsEverAnimated(node->GetTarget()))
            {
                OutputTopLevelLight(node, (LightObject *)obj);
            }
        }
        /*  if ((id == Class_ID(SIMPLE_CAM_CLASS_ID, 0) ||
         id == Class_ID(LOOKAT_CAM_CLASS_ID, 0)))
         VrmlOutTopLevelCamera(0, node, FALSE);*/

        if (id == NavInfoClassID)
        {
            mHasNavInfo = TRUE;
            VrmlOutTopLevelNavInfo(0, node, FALSE);
        }

        if (id == BackgroundClassID)
            VrmlOutTopLevelBackground(0, node, FALSE);

        if (id == FogClassID)
            VrmlOutTopLevelFog(0, node, FALSE);

        if (id == SkyClassID)
            VrmlOutTopLevelSky(0, node, FALSE);

        if (id == AudioClipClassID)
            VrmlOutInitializeAudioClip(0, node);

        if (id == TouchSensorClassID)
        {
            TouchSensorObject *ts = (TouchSensorObject *)obj;
            if (ts->triggerObject)
            {
                mSensorTable.AddSensor(ts->triggerObject, node, NULL);
            }
            int ct;
            INode *nd;
            for (ct = ts->objects.Count() - 1; ct >= 0; ct--)
            {
                nd = ts->objects[ct]->node;
                if (nd)
                {
                    nd->SetNodeLong(nd->GetNodeLong() | RUN_BY_TOUCH_SENSOR);
                }
            }
        }

        if (id == ARSensorClassID)
            VrmlOutARSensor(node, (ARSensorObject *)obj, 0);
        if (id == MultiTouchSensorClassID)
            VrmlOutMTSensor(node, (MultiTouchSensorObject *)obj, 0);

        if (id == COVERClassID)
            VrmlOutCOVER(node, (COVERObject *)obj, 0);

        if (id == OnOffSwitchClassID)
        {
            OnOffSwitchObject *so = (OnOffSwitchObject *)obj;
            if (so->onObject || so->offObject)
            {
                TCHAR *vrmlObjName = NULL;
                vrmlObjName = VrmlParent(node);
                if (so->onObject)
                {
                    mSensorTable.AddSensor(so->onObject, node, NULL);
                    so->onObject->SetNodeLong(so->onObject->GetNodeLong() | RUN_BY_ONOFF_SENSOR);

                    /*Object *o = so->onObject->EvalWorldState(mStart).obj;
               if (!o)
               break;
               assert(vrmlObjName);
               if (IsAimTarget(so->onObject))
               break;
               INode* top;
               if (o->ClassID() == TimeSensorClassID)
               top = so->onObject;
               else
               top = GetTopLevelParent(so->onObject);
               ObjectBucket* ob =
               mObjTable.AddObject(top->EvalWorldState(mStart).obj);
               if (top != otop) {
               AddAnimRoute(vrmlObjName, ob->name.data(), node, top);
               AddCameraAnimRoutes(vrmlObjName, node, top);
               otop = top;
               }*/

                    Object *toObj = so->onObject->EvalWorldState(mStart).obj;
                    ObjectBucket *ob = mObjTable.AddObject(so->onObject->EvalWorldState(mStart).obj);
                    AddAnimRoute(vrmlObjName, ob->name.data(), node, so->onObject);
                }
                if (so->offObject)
                {
                    mSensorTable.AddSensor(so->offObject, node, NULL);
                    so->offObject->SetNodeLong(so->offObject->GetNodeLong() | RUN_BY_ONOFF_SENSOR);
                    ObjectBucket *ob = mObjTable.AddObject(so->offObject->EvalWorldState(mStart).obj);
                    AddAnimRoute(vrmlObjName, ob->name.data(), node, so->offObject);
                }
                //add a switch script
                int defaultState = 0;
            MSTREAMPRINTF  _T("DEF %s-SCRIPT Script {\n"),mNodes.GetNodeName(node));
            Indent(1);
            MSTREAMPRINTF  _T("eventIn SFTime trigger\n"));
            Indent(1);
            MSTREAMPRINTF  _T("eventOut SFTime onTime\n"));
            Indent(1);
            MSTREAMPRINTF  _T("eventOut SFTime offTime\n"));
            Indent(1);
            MSTREAMPRINTF  _T("field SFInt32 state %d\n"),defaultState);
            Indent(1);
            MSTREAMPRINTF  _T("url \"javascript:\n"));
            Indent(2);
            MSTREAMPRINTF  _T("function trigger(t) {\n"));
            Indent(2);
            MSTREAMPRINTF  _T("if(state)\n"));
            Indent(2);
            MSTREAMPRINTF  _T("{\n"));
            Indent(4);
            MSTREAMPRINTF  _T("state = 0;\n"));
            Indent(4);
            MSTREAMPRINTF  _T("offTime = t;\n"));
            Indent(2);
            MSTREAMPRINTF  _T("}\n"));
            Indent(2);
            MSTREAMPRINTF  _T("else\n"));
            Indent(2);
            MSTREAMPRINTF  _T("{\n"));
            Indent(4);
            MSTREAMPRINTF  _T("state = 1;\n"));
            Indent(4);
            MSTREAMPRINTF  _T("onTime = t;\n"));
            Indent(2);
            MSTREAMPRINTF  _T("}\n"));
            Indent(1);
            MSTREAMPRINTF  _T("}\"\n"));
            MSTREAMPRINTF  _T("}\n"));
            }
        }
        if (id == SwitchClassID)
        {
            SwitchObject *so = (SwitchObject *)obj;
            if (so->objects.Count() > 0)
            {
                if (so->objects[0]->node)
                {
                    int i;
                    for (i = 0; i < so->objects.Count(); i++)
                    {
                        mSensorTable.AddSensor(so->objects[i]->node, node, NULL);
                    }
                    //add a switch script
                    int numObjs = so->objects.Count();
                    int defaultValue = -1;
                    int enableNoChoice = 1;
                    so->pblock->GetValue(PB_S_DEFAULT, mStart, defaultValue, FOREVER);
                    so->pblock->GetValue(PB_S_ALLOW_NONE, mStart, enableNoChoice, FOREVER);
               MSTREAMPRINTF  _T("DEF %s-SCRIPT Script {\n"),mNodes.GetNodeName(node));
               Indent(1);
               MSTREAMPRINTF  _T("eventIn SFTime trigger\n"));
               Indent(1);
               MSTREAMPRINTF  _T("eventOut SFInt32 choice\n"));
               Indent(1);
               MSTREAMPRINTF  _T("field SFInt32 numChoices %d\n"),numObjs);
               Indent(1);
               MSTREAMPRINTF  _T("field SFInt32 currentChoice %d\n"),defaultValue);
               Indent(1);
               MSTREAMPRINTF  _T("url \"javascript:\n"));
               Indent(2);
               MSTREAMPRINTF  _T("function trigger(i) {\n"));
               Indent(2);
               MSTREAMPRINTF  _T("currentChoice++;\n"));
               Indent(2);
               if (enableNoChoice)
                  MSTREAMPRINTF  _T("if(currentChoice >= numChoices) currentChoice = -1;\n"));
               else
                  MSTREAMPRINTF  _T("if(currentChoice >= numChoices) currentChoice = 0;\n"));
               Indent(2);
               MSTREAMPRINTF  _T("choice = currentChoice;\n"));
               Indent(2);
               MSTREAMPRINTF  _T("}\"\n"));
               MSTREAMPRINTF  _T("}\n"));
                }
            }
            int ct;
            INode *nd;
            for (ct = so->objects.Count() - 1; ct >= 0; ct--)
            {
                nd = so->objects[ct]->node;
                if (nd) // node might have been deleted
                {
                    nd->SetNodeLong(nd->GetNodeLong() | RUN_BY_SWITCH_SENSOR);
                }
            }
        }
        if (id == ProxSensorClassID)
        {
            ProxSensorObject *ps = (ProxSensorObject *)obj;
            int ct;
            INode *nd;
            for (ct = ps->objects.Count() - 1; ct >= 0; ct--)
            {
                nd = ps->objects[ct]->node;
                if (nd)
                {
                    nd->SetNodeLong(nd->GetNodeLong() | RUN_BY_PROX_SENSOR);
                }
            }
        }
        if (id == TimeSensorClassID)
        {
            TimeSensorObject *ts = (TimeSensorObject *)obj;
            int ct;
            INode *nd;
            for (ct = ts->TimeSensorObjects.Count() - 1; ct >= 0; ct--)
            {
                nd = ts->TimeSensorObjects[ct]->node;
                if (nd)
                {
                    nd->SetNodeLong(nd->GetNodeLong() | RUN_BY_TIME_SENSOR);
                }
            }
        }
        if (id == TabletUIClassID)
        {
            TabletUIObject *th = (TabletUIObject *)obj;
            int ct;
            INode *nd;
            for (int i = 0; i < th->elements.Count(); i++)
                for (ct = th->elements[i]->objects.Count() - 1; ct >= 0; ct--)
                {
                    nd = th->elements[i]->objects[ct]->node;
                    if (nd)
                    {
                        Object *o = nd->EvalWorldState(mStart).obj;
                        Class_ID o_id = o->ClassID();
                        if (th->elements[i]->type == TUIFloatSlider)
                            nd->SetNodeLong(nd->GetNodeLong() | RUN_BY_TABLETUI_SENSOR);
                        else if ((th->elements[i]->type == TUIButton) && (o_id == Class_ID(SIMPLE_CAM_CLASS_ID, 0) || o_id == Class_ID(LOOKAT_CAM_CLASS_ID, 0) || (o_id == TimeSensorClassID) || o_id == NavInfoClassID))
                        {
                            mSensorTable.AddSensor(nd, NULL, th->elements[i]);
                            nd->SetNodeLong(nd->GetNodeLong() | RUN_BY_TOUCH_SENSOR);
                        }
                        else if (o_id != TimeSensorClassID)
                            nd->SetNodeLong(nd->GetNodeLong() | RUN_BY_TOUCH_SENSOR);
                    }
                }
        }
        if (id == AnchorClassID)
        {
            AnchorObject *ao = (AnchorObject *)obj;
            if (ao->triggerObject)
            {
                mSensorTable.AddSensor(ao->triggerObject, node, NULL);
            }
        }
        ObjectBucket *ob = mObjTable.AddObject(obj);
        if (!ob->hasName)
        {
            ob->name = mNodes.GetNodeName(node);
            ob->hasName = TRUE;
        }
        if (!ob->hasInstName && !ObjIsPrim(node, obj))
        {
            ob->instName = mNodes.GetNodeName(node);
            ob->instMirrored = false;
            ob->hasInstName = TRUE;
        }
    }

    int n = node->NumberOfChildren();
    for (int i = 0; i < n; i++)
        TraverseNode(node->GetChildNode(i));
}

void
VRML2Export::ComputeWorldBoundBox(INode *node, ViewExp *vpt)
{
    if (!node)
        return;
    Object *obj = node->EvalWorldState(mStart).obj;
    Class_ID id;

    node->SetNodeLong(0);
    if (obj)
    {
        id = obj->ClassID();
        if (id == TimeSensorClassID)
        {
            TimeSensorObject *tso = (TimeSensorObject *)obj;
            if (!tso->vrmlWritten)
                VrmlOutTimeSensor(node, tso, 0);
            mTimerList = mTimerList->AddNode(node);
        }
        else if (doExport(node) && (id == TabletUIClassID))
        {
            TabletUIObject *tabObj = (TabletUIObject *)obj;
            for (int i = 0; i < tabObj->elements.Count(); i++)
                if (tabObj->elements[i]->parent == NULL)
                {
                    TabletUIElement *root = tabObj->elements[i];
                    VrmlOutTUI(node, root, 0);
                }
        }
        Box3 bb;
        obj->GetWorldBoundBox(mStart, node, vpt, bb);
        mBoundBox += bb;
    }

    int n = node->NumberOfChildren();
    for (int i = 0; i < n; i++)
        ComputeWorldBoundBox(node->GetChildNode(i), vpt);
}

// Compute the world bounding box and a list of timesensors;
// also initialize each INode's nodeLong data
void
VRML2Export::ScanSceneGraph1()
{
#if MAX_PRODUCT_VERSION_MAJOR > 14
    ViewExp &vpt = mIp->GetViewExp(NULL);
    INode *node = mIp->GetRootNode();
    ComputeWorldBoundBox(node, &vpt);
#else
    ViewExp *vpt = mIp->GetViewport(NULL);
    INode *node = mIp->GetRootNode();
    ComputeWorldBoundBox(node, vpt);
#endif
}

// Make a list of al the LOD objects and texture maps in the scene.
// Also output top-level objects
void
VRML2Export::ScanSceneGraph2()
{
    INode *node = mIp->GetRootNode();
    TraverseNode(node);
}

// Return TRUE iff the node is referenced by the LOD node.
BOOL VRML2Export::ObjectIsReferenced(INode *lodNode, INode *node)
{

    LODObject *obj = (LODObject *)lodNode->EvalWorldState(mStart).obj;
    Tab<LODObj *> lodObjects = obj->GetLODObjects();

    Object *refObj = node->EvalWorldState(mStart).obj;
    int numRefs = obj->NumRefs();

    for (int i = 0; i < numRefs; i++)
        if (lodObjects[i]->node != NULL)
        {
            if (refObj == ((INode *)lodObjects[i]->node)->EvalWorldState(mStart).obj)
                return TRUE;
        }

    return FALSE;
}

// Return TRUE iff the node is referenced by ANY LOD node.
BOOL
VRML2Export::ObjectIsLODRef(INode *node)
{
    INodeList *l = mLodList;

    for (; l; l = l->GetNext())
        if (ObjectIsReferenced(l->GetNode(), node))
            return TRUE;

    return FALSE;
}

static INT_PTR CALLBACK
    ConfirmDlgProc(HWND hDlg, UINT msg, WPARAM wParam, LPARAM lParam)
{
    VRML2Export *exporter;
    if (msg == WM_INITDIALOG)
    {
        SetWindowLongPtr(hDlg, GWLP_USERDATA, lParam);
    }
    exporter = (VRML2Export *)GetWindowLongPtr(hDlg, GWLP_USERDATA);
    switch (msg)
    {
    case WM_INITDIALOG:
        CenterWindow(hDlg, GetParent(hDlg));
        Static_SetText(GetDlgItem(hDlg, IDC_SOURCE_NAME), exporter->sourceFile);
        Static_SetText(GetDlgItem(hDlg, IDC_DESTINATION_NAME), exporter->destFile);
        return TRUE;
    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        case IDSKIP:
            exporter->mReplace = false;
            EndDialog(hDlg, TRUE);
            return TRUE;
        case IDSKIPALL:
            exporter->mReplace = false;
            exporter->mSkipAll = true;
            EndDialog(hDlg, TRUE);
            return TRUE;
        case IDREPLACEALL:
            exporter->mReplace = true;
            exporter->mReplaceAll = true;
            EndDialog(hDlg, TRUE);
            return TRUE;
        case IDOK:
            exporter->mReplace = true;
            EndDialog(hDlg, TRUE);
            return TRUE;
        }
        return FALSE;
    }
    return FALSE;
}

void VRML2Export::askForConfirmation()
{
    DialogBoxParam(hInstance, MAKEINTRESOURCE(IDD_CONFIRM),
                   GetActiveWindow(), ConfirmDlgProc,
                   (LPARAM) this);
}

extern HINSTANCE hInstance;

static INT_PTR CALLBACK
    ProgressDlgProc(HWND hDlg, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_INITDIALOG:
        CenterWindow(hDlg, GetParent(hDlg));
        Static_SetText(GetDlgItem(hDlg, IDC_PROGRESS_NNAME), _T(" "));
        return TRUE;
    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        case IDCANCEL:
            DestroyWindow(hDlg);
            hDlg = NULL;
            return TRUE;
        case IDOK:
            DestroyWindow(hDlg);
            hDlg = NULL;
            return TRUE;
        }
        return FALSE;
    case 666:
        Static_SetText(GetDlgItem(hDlg, IDC_PROGRESS_NNAME), (TCHAR *)lParam);
        return TRUE;
    }
    return FALSE;
}

// Export the current scene as VRML
#ifdef _LEC_
int
VRML2Export::DoFBExport(const TCHAR *filename, Interface *i, VRBLExport *exp,
                        int frame, TimeValue time)
{
    mIp = i;
    mStart = time;
    mCoordInterp = FALSE;

    mGenNormals = exp->GetGenNormals();
    mExpLights = exp->GetExpLights();
    mCopyTextures = exp->GetCopyTextures();
    mForceWhite = exp->GetForceWhite();
    mExportSelected = exp->GetExportSelected();
    mExportOccluders = exp->GetExportOccluders();
    mIndent = exp->GetIndent();
    mType = exp->GetExportType();
    mUsePrefix = exp->GetUsePrefix();
    mUrlPrefix = exp->GetUrlPrefix();
    mCamera = exp->GetCamera();
    mZUp = exp->GetZUp();
    mDigits = exp->GetDigits();
    mFlipBook = exp->GetFlipBook();
    mCoordSample = exp->GetCoordSample();
    mCoordSampleRate = exp->GetCoordSampleRate();
    mNavInfo = exp->GetNavInfo();
    mBackground = exp->GetBackground();
    mFog = exp->GetFog();
    mSky = exp->GetSky();
    mTitle = exp->GetTitle();
    mInfo = exp->GetInfo();
    mExportHidden = exp->GetExportHidden();
    mPrimitives = exp->GetPrimitives();
    mPolygonType = exp->GetPolygonType();
    mEnableProgressBar = exp->GetEnableProgressBar();
    mPreLight = exp->GetPreLight();
    mCPVSource = exp->GetCPVSource();
    mCallbacks = exp->GetCallbacks(); //FIXME add callback support

    TCHAR buf[16];
    _stprintf(buf, _T("%d"), frame);
    TSTR wName(filename);
    int extLoc;
    extLoc = wName.last('.');
    if (extLoc != -1)
        wName.remove(extLoc);
    wName.Append(buf);
    if (mType == Export_X3D_V)
    {
        wName.Append(_T(".x3dv"));
    }
    else
    {
        wName.Append(_T(".wrl"));
    }

#if MAX_PRODUCT_VERSION_MAJOR > 14
    mStream.Open(wName.data(), false, CP_UTF8);
#else
    if (!mStream)
        mStream = _tfopen(wName, _T("a"));
#endif
// write out the file
//WorkFile theFile(wName.data(), _T("w"));
//mStream = theFile.MStream();
#if MAX_PRODUCT_VERSION_MAJOR > 14
    if (!mStream.IsFileOpen())
    {
#else
    if (mStream != NULL)
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

    // Write out the VRML header
    if (mType == Export_X3D_V)
    {
      MSTREAMPRINTF  _T("#X3D V3.2 utf8 \nPROFILE Interchange\n"));
    }
    else
    {
      MSTREAMPRINTF  _T("#VRML V2.0 utf8\n\n"));
    }

    VrmlOutFileInfo();
    // Write out global TimeSensor
    bool doAnim = true; // get this from a checkbox in the VRML output dialog later
    float CycleInterval = (mIp->GetAnimRange().End() - mStart) / ((float)GetTicksPerFrame() * GetFrameRate());
   MSTREAMPRINTF 
      _T("DEF Global-TIMER TimeSensor { loop %s cycleInterval %s },\n"),
      (doAnim) ? _T("TRUE") : _T("FALSE"),
      floatVal(CycleInterval));

   // give third party dlls a chance to write scene
   int cnt;
   for (cnt = 0; cnt < mCallbacks->GetPreSceneCount(); cnt++)
   {
       DllPreScene preScene = mCallbacks->GetPreScene(cnt);
       PreSceneParam params;
       params.version = 0;
       params.fName = mFilename;
       params.i = mIp;
#if MAX_PRODUCT_VERSION_MAJOR > 14
       if (mStream.IsFileOpen())
           mStream.Close();
#else
       if (mStream)
           fclose(mStream);
       mStream = NULL;
#endif
       if ((*(preScene))(&params))
       { //third party wrote the scene
           return TRUE;
       }
   }

#if MAX_PRODUCT_VERSION_MAJOR > 14
   if (!mStream.IsFileOpen())
       mStream.Open(mFilename, false, CP_UTF8);
#else
   if (!mStream)
       mStream = _tfopen(mFilename, _T("a"));
#endif

   // generate the hash table of unique node names
   GenerateUniqueNodeNames(mIp->GetRootNode());

   if (mEnableProgressBar)
   {
       DisableProcessWindowsGhosting(); // prevents windows from freezing the progressbar
       RECT rcClient; // client area of parent window
       int cyVScroll; // height of a scroll bar arrow
       hWndPDlg = CreateDialog(hInstance, MAKEINTRESOURCE(IDD_PROGRESSDLG),
                               GetActiveWindow(), ProgressDlgProc);
       GetClientRect(hWndPDlg, &rcClient);
       cyVScroll = GetSystemMetrics(SM_CYVSCROLL);
       ShowWindow(hWndPDlg, SW_SHOW);
       hWndPB = CreateWindow(PROGRESS_CLASS, NULL,
                             WS_CHILD | WS_VISIBLE, rcClient.left,
                             rcClient.bottom - cyVScroll,
                             rcClient.right, cyVScroll,
                             hWndPDlg, (HMENU)0, hInstance, NULL);
       // Set the range and increment of the progress bar.
       SendMessage(hWndPB, PBM_SETRANGE, 0, MAKELPARAM(0,
                                                       CountNodes(mIp->GetRootNode()) + 1));
       SendMessage(hWndPB, PBM_SETSTEP, (WPARAM)1, 0);
   }

   // Write out the scene graph
   VrmlOutNode(mIp->GetRootNode(), NULL, -2, FALSE, TRUE, FALSE);

   WriteAnimRoutes();
   WriteScripts();
   delete mLodList;
   delete mTabletUIList;
   delete mScriptsList;
   SetCursor(normal);
   if (hWndPB)
   {
       DestroyWindow(hWndPB);
       hWndPB = NULL;
   }
   if (hWndPDlg)
   {
       DestroyWindow(hWndPDlg);
       hWndPDlg = NULL;
   }
#if MAX_PRODUCT_VERSION_MAJOR > 14
   if (mStream.IsFileOpen())
       mStream.Close();
#else
   if (mStream)
       fclose(mStream);
   mStream = NULL;
#endif

   return 1;
}
#endif

// Export the current scene as VRML
int
VRML2Export::DoExport(const TCHAR *filename, Interface *i, VRBLExport *exp)
{
    mIp = i;
    mStart = mIp->GetAnimRange().Start();

    mGenNormals = exp->GetGenNormals();
    mExpLights = exp->GetExpLights();
    mCopyTextures = exp->GetCopyTextures();
    mForceWhite = exp->GetForceWhite();
    mExportSelected = exp->GetExportSelected();
    mExportOccluders = exp->GetExportOccluders();
    mIndent = exp->GetIndent();
    mType = exp->GetExportType();
    mUsePrefix = exp->GetUsePrefix();
    mUrlPrefix = exp->GetUrlPrefix();
    mCamera = exp->GetCamera();
    mZUp = exp->GetZUp();
    mDigits = exp->GetDigits();
    mCoordInterp = exp->GetCoordInterp();
    mTformSample = exp->GetTformSample();
    mTformSampleRate = exp->GetTformSampleRate();
    mCoordSample = exp->GetCoordSample();
    mCoordSampleRate = exp->GetCoordSampleRate();
    mNavInfo = exp->GetNavInfo();
    mBackground = exp->GetBackground();
    mFog = exp->GetFog();
    mSky = exp->GetSky();
    mTitle = exp->GetTitle();
    mInfo = exp->GetInfo();
    mExportHidden = exp->GetExportHidden();
    mPrimitives = exp->GetPrimitives();
    mPolygonType = exp->GetPolygonType();
    mEnableProgressBar = exp->GetEnableProgressBar();
    mPreLight = exp->GetPreLight();
    mCPVSource = exp->GetCPVSource();
    mCallbacks = exp->GetCallbacks();
    mFilename = (TCHAR *)filename;

#if MAX_PRODUCT_VERSION_MAJOR > 14
    mStream.Open(mFilename, false, CP_UTF8);
#else
    mStream = _tfopen(mFilename, _T("a"));
#endif
#if MAX_PRODUCT_VERSION_MAJOR > 14
    if (!mStream.IsFileOpen())
    {
#else
    if (mStream == NULL)
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
    if (mType == Export_X3D_V)
    {
      MSTREAMPRINTF  _T("#X3D V3.2 utf8 \nPROFILE Interchange\n"));
    }
    else
    {
      MSTREAMPRINTF  _T("#VRML V2.0 utf8\n\n"));
    }
    VrmlOutFileInfo();

    // Write out global TimeSensor
    bool doAnim = true; // get this from a checkbox in the VRML output dialog later
    float CycleInterval = (mIp->GetAnimRange().End() - mStart) / ((float)GetTicksPerFrame() * GetFrameRate());
   MSTREAMPRINTF 
      _T("DEF Global-TIMER TimeSensor { loop %s cycleInterval %s },\n"),
      (doAnim) ? _T("TRUE") : _T("FALSE"),
      floatVal(CycleInterval));

   // generate the hash table of unique node names
   GenerateUniqueNodeNames(mIp->GetRootNode());

   if (mEnableProgressBar)
   {
       RECT rcClient; // client area of parent window
       int cyVScroll; // height of a scroll bar arrow
       hWndPDlg = CreateDialog(hInstance, MAKEINTRESOURCE(IDD_PROGRESSDLG),
                               GetActiveWindow(), ProgressDlgProc);
       GetClientRect(hWndPDlg, &rcClient);
       cyVScroll = GetSystemMetrics(SM_CYVSCROLL);
       ShowWindow(hWndPDlg, SW_SHOW);
       // InitCommonControls();
       hWndPB = CreateWindow(PROGRESS_CLASS, NULL,
                             WS_CHILD | WS_VISIBLE, rcClient.left,
                             rcClient.bottom - cyVScroll,
                             rcClient.right, cyVScroll,
                             hWndPDlg, (HMENU)0, hInstance, NULL);
       // Set the range and increment of the progress bar.
       SendMessage(hWndPB, PBM_SETRANGE, 0, MAKELPARAM(0,
                                                       CountNodes(mIp->GetRootNode()) + 1));
       SendMessage(hWndPB, PBM_SETSTEP, (WPARAM)1, 0);
   }

   // give third party dlls a chance to write before the scene was written
   BOOL written = FALSE;
   int cnt;
   for (cnt = 0; cnt < mCallbacks->GetPreSceneCount(); cnt++)
   {
       DllPreScene preScene = mCallbacks->GetPreScene(cnt);
       PreSceneParam params;
       params.version = 0;
       params.fName = mFilename;
       params.i = mIp;
#if MAX_PRODUCT_VERSION_MAJOR > 14
       if (mStream.IsFileOpen())
           mStream.Close();
#else
       if (mStream)
           fclose(mStream);
       mStream = NULL;
#endif
       written = (*(preScene))(&params); //third party wrote the scene
       if (written)
           break; // first come first served
   }

#if MAX_PRODUCT_VERSION_MAJOR > 14
   if (!mStream.IsFileOpen())
       mStream.Open(mFilename, false, CP_UTF8);
#else
   if (!mStream)
       mStream = _tfopen(mFilename, _T("a"));
#endif

#ifndef NO_CAL3D
   if (Cal3DObject::cores)
       Cal3DObject::cores->clearWritten();
#endif

   // Write out the scene graph
   if (!written)
   {
       VrmlOutNode(mIp->GetRootNode(), NULL, -2, FALSE, TRUE, FALSE);
       WriteAnimRoutes();
       WriteScripts();
       delete mLodList;
       delete mScriptsList;
   }

   // give third party dlls a chance to write after the scene was written
   for (cnt = 0; cnt < mCallbacks->GetPostSceneCount(); cnt++)
   {
       DllPostScene postScene = mCallbacks->GetPostScene(cnt);
       PostSceneParam params;
       params.version = 0;
       params.fName = mFilename;
       params.i = mIp;
#if MAX_PRODUCT_VERSION_MAJOR > 14
       if (mStream.IsFileOpen())
           mStream.Close();
#else
       if (mStream)
           fclose(mStream);
       mStream = NULL;
#endif
       (*(postScene))(&params);
   }

#if MAX_PRODUCT_VERSION_MAJOR > 14
   if (!mStream.IsFileOpen())
       mStream.Open(mFilename, false, CP_UTF8);
#else
   if (!mStream)
       mStream = _tfopen(mFilename, _T("a"));
#endif

   SetCursor(normal);
   if (hWndPB)
   {
       DestroyWindow(hWndPB);
       hWndPB = NULL;
   }
   if (hWndPDlg)
   {
       DestroyWindow(hWndPDlg);
       hWndPDlg = NULL;
   }

#if MAX_PRODUCT_VERSION_MAJOR > 14
   if (mStream.IsFileOpen())
       mStream.Close();
#else
   if (mStream)
       fclose(mStream);
   mStream = NULL;
#endif

   return 1;
}

VRML2Export::VRML2Export()
{
    mGenNormals = FALSE;
    mExpLights = FALSE;
    mHadAnim = FALSE;
    mLodList = NULL;
    mTimerList = NULL;
    mTabletUIList = NULL;
    mScriptsList = NULL;
    mTformSample = TRUE;
    mTformSampleRate = 10;
    mCoordSample = FALSE;
    mCoordSampleRate = 3;
    mHasLights = FALSE;
    mHasNavInfo = FALSE;
    mFlipBook = FALSE;
    effect = NO_EFFECT;
    mReplace = false;
    mReplaceAll = false;
    mSkipAll = false;
    numSwitchObjects = 0;

#if MAX_PRODUCT_VERSION_MAJOR > 14
#else
    mStream = NULL;
#endif

    shaderEffects.push_back(ShaderEffect(_T("")));
    shaderEffects.push_back(ShaderEffect(_T("coVRShaderBump_")));
    shaderEffects.push_back(ShaderEffect(_T("coVRShaderBumpEnv")));
    shaderEffects.push_back(ShaderEffect(_T("coVRShaderBumpCube")));
}

VRML2Export::~VRML2Export()
{
    for (INodeList *l = mTimerList; l; l = l->GetNext())
    {
        TimeSensorObject *tso = (TimeSensorObject *)
                                    l->GetNode()->EvalWorldState(mStart)
                                        .obj;
        tso->vrmlWritten = false;
    }
    if (mTimerList != NULL)
        delete mTimerList;
}

// Traverse the scene graph generating Unique Node Names
void
VRML2Export::GenerateUniqueNodeNames(INode *node)
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

static DWORD HashCode(DWORD o, int size)
{
    DWORD code = (DWORD)o;
    return (code >> 2) % size;
}

// Object Hash table stuff

ObjectBucket *
ObjectHashTable::AddObject(Object *o, bool countInstances)
{
    DWORD hashCode = HashCode((DWORD)o, OBJECT_HASH_TABLE_SIZE);
    ObjectBucket *ob;

    for (ob = mTable[hashCode]; ob; ob = ob->next)
    {
        if (ob->obj == o)
        {
            if (countInstances)
            {
                ob->numInstances++;
            }
            return ob;
        }
    }
    ob = new ObjectBucket(o);
    ob->next = mTable[hashCode];
    mTable[hashCode] = ob;
    return ob;
}

void
SensorHashTable::AddSensor(INode *node, INode *sensor, TabletUIElement *tuielem)
{
    DWORD hashCode = HashCode((DWORD)node, SENSOR_HASH_TABLE_SIZE);
    SensorBucket *sb;

    for (sb = mTable[hashCode]; sb; sb = sb->mNext)
    {
        if (sb->mNode == node)
        {
            if (sensor != NULL)
                sb->mSensors = sb->mSensors->AddNode(sensor);
            if (tuielem != NULL)
                sb->mTUIElems = sb->mTUIElems->AddElem(tuielem);
            return;
        }
    }
    sb = new SensorBucket(node);
    if (sensor != NULL)
        sb->mSensors = sb->mSensors->AddNode(sensor);
    if (tuielem != NULL)
        sb->mTUIElems = sb->mTUIElems->AddElem(tuielem);
    sb->mNext = mTable[hashCode];
    mTable[hashCode] = sb;
}

SensorBucket *
SensorHashTable::FindSensor(INode *node)
{
    DWORD hashCode = HashCode((DWORD)node, SENSOR_HASH_TABLE_SIZE);
    SensorBucket *sb;

    for (sb = mTable[hashCode]; sb; sb = sb->mNext)
    {
        if (sb->mNode == node)
        {
            return sb;
        }
    }
    return NULL;
}

void VRML2Export::VrmlOutSwitchScript(INode *node)
{
    SwitchObject *swObj = (SwitchObject *)node->EvalWorldState(mStart).obj;

    //if (swObj->needsScript)
    //{

      MSTREAMPRINTF  _T("DEF Choice%s-SCRIPT Script {\n"), node->GetName());
      Indent(1);
      MSTREAMPRINTF  _T("eventIn SFInt32 userChoice\n"));
      Indent(1);
      MSTREAMPRINTF  _T("eventOut SFInt32 switchChoice\n"));
      Indent(1);
      MSTREAMPRINTF  _T("url \"javascript:\n"));
      Indent(2);
      MSTREAMPRINTF  _T("function userChoice(k) {\n"));
      int m;
      int k = 0;
      Tab<int> index;
      index.SetCount(switchObjects.Count());
      for (int i = 0; i < switchObjects.Count(); i++)
      {
          m = 0;
          while ((m < swObj->objects.Count()) && (switchObjects[i] != swObj->objects[m]->node))
              m++;
          if (m < swObj->objects.Count())
              index[m] = k++;
      }

      for (int i = 0; i < swObj->objects.Count(); i++)
      {
          Indent(3);
          if (i == 0) MSTREAMPRINTF  _T("if (k == %d) switchChoice = %d;\n"), i, index[i]);
          else MSTREAMPRINTF  _T("else if (k == %d) switchChoice = %d;\n"), i, index[i]);
      }

      int defaultValue = -1;
      int enableNoChoice = 1;
      swObj->pblock->GetValue(PB_S_DEFAULT, mStart, defaultValue, FOREVER);
      swObj->pblock->GetValue(PB_S_ALLOW_NONE, mStart, enableNoChoice, FOREVER);

      if (enableNoChoice != 0)
      {
          Indent(3);
         MSTREAMPRINTF  _T("else switchChoice = -1;\n"));
      }
      Indent(2);
      MSTREAMPRINTF  _T("}\"\n}\n"));

      //swObj->needsScript = false;
      //}
}

// Output any grid helpers
/*
void
VRML2Export::VrmlOutGridHelpers(INode* node)
{
if (!node) return;

Object*  obj = node->EvalWorldState(mStart).obj;
Class_ID id;

if (obj) {
id = obj->ClassID();
if (id == Class_ID(GRIDHELP_CLASS_ID, 0)) {
float len;
float width;
float grid;
Matrix3 tm;

len   = ((GridHelpObject*)obj)->GetLength(mStart);
width = ((GridHelpObject*)obj)->GetWidth(mStart);
grid  = ((GridHelpObject*)obj)->GetGrid(mStart);
// tm    = ((GridHelpObject*)obj)->myTM; //private member

int pCnt;
pCnt = obj->NumPoints();
for (int i = 0; i < pCnt; i++) {
Point3 pt;
pt = obj->GetPoint(i);
}
}
}

int n = node->NumberOfChildren();
for (int i = 0; i < n; i++) {
VrmlOutGridHelpers(node->GetChildNode(i));
}
}
*/
