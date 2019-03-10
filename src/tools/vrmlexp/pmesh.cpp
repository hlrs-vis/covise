/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
	FILE: pmesh.cpp

	DESCRIPTION:  simple Polygon class module

	CREATED BY: greg finch

	HISTORY: created 1 december, 1996

 *>	Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#include "vrml.h"
#include "pmesh.h"

#define EDGEFLAGS(i) (i == 0 ? EDGE_A : i == 1 ? EDGE_B : EDGE_C)

PMesh::PMesh(Mesh &om, PType type, BOOL *gTV)
{
    int i;
    mOrgMesh = om;
    mType = type;
    mIsTextured = gTV[0];
    mIsTextured2 = gTV[1];
    mIsTextured3 = gTV[2];
    mIsTextured4 = gTV[3];
    mIsTextured5 = gTV[4];
    mIsTextured6 = gTV[5];
    mCosEps = cos(COPLANAR_NORMAL_EPSILON * PI / 180);
    mAdjEdges = new AdjEdgeList(mOrgMesh);
    mAdjFaces = new AdjFaceList(mOrgMesh, *mAdjEdges);

    mSuppressEdges = new int[om.getNumFaces()];
    for (i = 0; i < om.getNumFaces(); i++)
        mSuppressEdges[i] = 0;

    mOrgMesh.buildRenderNormals();
}

PMesh::~PMesh()
{
    if (mPolygons.Count())
        mPolygons.Delete(0, mPolygons.Count());
    if (mVertices.Count())
        mVertices.Delete(0, mVertices.Count());
    if (mTVertices.Count())
        mTVertices.Delete(0, mTVertices.Count());
    if (mTVertices2.Count())
        mTVertices2.Delete(0, mTVertices2.Count());
    if (mTVertices3.Count())
        mTVertices3.Delete(0, mTVertices3.Count());
    if (mTVertices4.Count())
        mTVertices4.Delete(0, mTVertices4.Count());
    if (mTVertices5.Count())
        mTVertices5.Delete(0, mTVertices5.Count());
    if (mTVertices6.Count())
        mTVertices6.Delete(0, mTVertices6.Count());

    delete[] mSuppressEdges;
    delete mAdjEdges;
    delete mAdjFaces;
}

BOOL
PMesh::CheckIfMatIDsMatch(int faceNum, int adjFaceNum)
{
    int faceMatID = mOrgMesh.faces[faceNum].getMatID();
    int adjFaceNumID = mOrgMesh.faces[adjFaceNum].getMatID();
    if (faceMatID == adjFaceNumID)
        return TRUE; // everything is OK to collapse
    else
        return FALSE;
}

BOOL
PMesh::CheckIfUVVertMatch(AEdge edge, int faceNum, int adjFaceNum)
{
    UVVert e0_v0;
    UVVert e0_v1;
    UVVert e1_v0;
    UVVert e1_v1;

    return TRUE; //FIXME
    // don't have TVerts so we don't care
    if (mOrgMesh.getNumTVerts() <= 0)
        return TRUE;

    e0_v0 = mOrgMesh.getTVert(mOrgMesh.tvFace[faceNum].t[edge.e0]);
    e0_v1 = mOrgMesh.getTVert(mOrgMesh.tvFace[faceNum].t[(edge.e0 + 1) % 3]);
    e1_v0 = mOrgMesh.getTVert(mOrgMesh.tvFace[adjFaceNum].t[edge.e1]);
    e1_v1 = mOrgMesh.getTVert(mOrgMesh.tvFace[adjFaceNum].t[(edge.e1 + 1) % 3]);

    if ((e0_v0 == e1_v1) && (e0_v1 == e1_v0))
        return TRUE; // everything is OK to collapse
    else
        return FALSE;
}

BOOL
PMesh::CheckIfCoplanar(int faceNum, int adjFaceNum)
{
    Point3 faceNormal;
    Point3 adjFaceNormal;

    faceNormal = mOrgMesh.getFaceNormal(faceNum);
    adjFaceNormal = mOrgMesh.getFaceNormal(adjFaceNum);
    // check for degenerate face
    if (Length(faceNormal) && Length(adjFaceNormal))
        return (DotProd(faceNormal, adjFaceNormal) > mCosEps);
    else
        return FALSE;
}

void
PMesh::CheckAdjacentFaces(int faceNum, BitArray &uFaces)
{
    Point3 faceNormal;
    Point3 adjFaceNormal;
    int adjFaceNum;
    int visEdge; // visable edge for face 0

    DWORD face0[3];
    DWORD face1[3];

    if (mType == OutputTriangles || mType == OutputQuads)
    {
        int hFaceNum = -1;
        AEdge hEdge;
        for (int i = 0; i < 3; i++)
        {
            adjFaceNum = mAdjFaces->list[faceNum].f[i];
            // teapot generate -1 for some faces !?
            if (adjFaceNum == -1)
                continue;
            // face was already written
            if (uFaces[adjFaceNum])
                continue;
            // check for degenerate polygons like those in cones.
            if (!IsValidFace(adjFaceNum))
                continue;
            if (mOrgMesh.faces[adjFaceNum].flags & FACE_HIDDEN)
                continue;
            if (!CheckIfCoplanar(faceNum, adjFaceNum))
                continue;
            if (!CheckIfMatIDsMatch(faceNum, adjFaceNum))
                continue;

            hFaceNum = adjFaceNum;

            face0[0] = mOrgMesh.faces[faceNum].v[0];
            face0[1] = mOrgMesh.faces[faceNum].v[1];
            face0[2] = mOrgMesh.faces[faceNum].v[2];
            face1[0] = mOrgMesh.faces[adjFaceNum].v[0];
            face1[1] = mOrgMesh.faces[adjFaceNum].v[1];
            face1[2] = mOrgMesh.faces[adjFaceNum].v[2];
            // find the shared coplanar edge
            AEdge edge = AdjoiningEdge(face0, face1);
            hEdge = edge;

            if (mIsTextured)
                if (!CheckIfUVVertMatch(edge, faceNum, adjFaceNum))
                    continue;

            visEdge = mOrgMesh.faces[faceNum].flags;
            // see if the coplanar edge is a visible edge
            /*
            #define EDGE_A		(1<<0)
            #define EDGE_B		(1<<1)
            #define EDGE_C		(1<<2)
            */
            if (visEdge & (1 << edge.e0))
                continue;
        }
        if (hFaceNum != -1)
        {
            /*
            mOrgMesh.faces[faceNum].setEdgeVis(0, EDGE_VIS);
            mOrgMesh.faces[faceNum].setEdgeVis(1, EDGE_VIS);
            mOrgMesh.faces[faceNum].setEdgeVis(2, EDGE_VIS);
            mOrgMesh.faces[faceNum].setEdgeVis(hEdge.e0, EDGE_INVIS);
            mOrgMesh.faces[hFaceNum].setEdgeVis(0, EDGE_VIS);
            mOrgMesh.faces[hFaceNum].setEdgeVis(1, EDGE_VIS);
            mOrgMesh.faces[hFaceNum].setEdgeVis(2, EDGE_VIS);
            mOrgMesh.faces[hFaceNum].setEdgeVis(hEdge.e1, EDGE_INVIS);
*/
            mSuppressEdges[faceNum] = EDGEFLAGS(hEdge.e0);
            mSuppressEdges[hFaceNum] = EDGEFLAGS(hEdge.e1);
            uFaces.Set(hFaceNum);
        }
        else
        { // single face
            /*
            mOrgMesh.faces[faceNum].setEdgeVis(0, EDGE_VIS);
            mOrgMesh.faces[faceNum].setEdgeVis(1, EDGE_VIS);
            mOrgMesh.faces[faceNum].setEdgeVis(2, EDGE_VIS);
*/
            mSuppressEdges[faceNum] = 0;
        }
        return;
    }
    else
    {
        for (int i = 0; i < 3; i++)
        {
            adjFaceNum = mAdjFaces->list[faceNum].f[i];
            // teapot generate -1 for some faces !?
            if (adjFaceNum == -1)
                continue;
            // check for degenerate polygons like those in cones.
            if (!IsValidFace(adjFaceNum))
                continue;
            if (mOrgMesh.faces[adjFaceNum].flags & FACE_HIDDEN)
                continue;
            if (!CheckIfCoplanar(faceNum, adjFaceNum))
                continue;
            if (!CheckIfMatIDsMatch(faceNum, adjFaceNum))
                continue;

            face0[0] = mOrgMesh.faces[faceNum].v[0];
            face0[1] = mOrgMesh.faces[faceNum].v[1];
            face0[2] = mOrgMesh.faces[faceNum].v[2];
            face1[0] = mOrgMesh.faces[adjFaceNum].v[0];
            face1[1] = mOrgMesh.faces[adjFaceNum].v[1];
            face1[2] = mOrgMesh.faces[adjFaceNum].v[2];
            // find the shared coplanar edge
            AEdge edge = AdjoiningEdge(face0, face1);

            if (mIsTextured)
                if (!CheckIfUVVertMatch(edge, faceNum, adjFaceNum))
                    continue;

            // if respecting visible lines.
            visEdge = mOrgMesh.faces[adjFaceNum].flags;
            /*
            #define EDGE_A		(1<<0)
            #define EDGE_B		(1<<1)
            #define EDGE_C		(1<<2)
            */
            // see if the coplanar edge is a visible edge
            if ((visEdge & (1 << edge.e1)) && (mType == OutputVisibleEdges) && !uFaces[adjFaceNum])
                continue;

            /*
            mOrgMesh.faces[faceNum].setEdgeVis(edge.e0, EDGE_INVIS);
            mOrgMesh.faces[adjFaceNum].setEdgeVis(edge.e1, EDGE_INVIS);
*/
            // face was already written
            if (uFaces[adjFaceNum] && mType == OutputVisibleEdges)
                continue;

            mSuppressEdges[faceNum] |= EDGEFLAGS(edge.e0);
            mSuppressEdges[adjFaceNum] |= EDGEFLAGS(edge.e1);
            if (uFaces[adjFaceNum])
                continue;

            uFaces.Set(adjFaceNum);
            CheckAdjacentFaces(adjFaceNum, uFaces);
        }
    }
}

// this is for things like four sided cones which have 8 verts and 12 faces !?
BOOL
PMesh::IsValidFace(int face)
{
    Point3 pts[3];

    pts[0] = mOrgMesh.getVert(mOrgMesh.faces[face].v[0]);
    pts[1] = mOrgMesh.getVert(mOrgMesh.faces[face].v[1]);
    pts[2] = mOrgMesh.getVert(mOrgMesh.faces[face].v[2]);

    if (pts[0] == pts[1])
        return FALSE;
    if (pts[0] == pts[2])
        return FALSE;
    if (pts[1] == pts[2])
        return FALSE;

    return TRUE;
}

// generate the coplanar faces
void
PMesh::GenCPFaces()
{
    int i;
    // used to store the faces used
    BitArray uFaces;
    uFaces.SetSize(mOrgMesh.getNumFaces());

    // used to store the faces written
    BitArray wFaces;
    wFaces.SetSize(mOrgMesh.getNumFaces());

    PMPoly newPolygon;
    int polyCnt;

    // generate a list of coplanar faces
    uFaces.ClearAll();
    wFaces.ClearAll();
    polyCnt = 0;
    for (i = 0; i < mOrgMesh.getNumFaces(); i++)
    {
        // skip faces that have been written
        if (!uFaces[i])
        {
            if (mOrgMesh.faces[i].flags & FACE_HIDDEN)
                continue;
            if (!IsValidFace(i))
                continue;
            uFaces.Set(i);
            if (mType != OutputTriangles)
                CheckAdjacentFaces(i, uFaces);

            newPolygon.SetFNormal(mOrgMesh.getFaceNormal(i));
            AddPolygon(&newPolygon);
            //if (mIsTextured) AddTPolygon(&newPolygon);

            // loop thur CP faces an add tri faces to list
            wFaces ^= uFaces;
            AddCPTriFaces(i, polyCnt, uFaces, wFaces);
            wFaces = uFaces;
            polyCnt++;
        }
    }
}

void
PMesh::AddCPTriFaces(int faceNum, int polyCnt, BitArray &uFaces, BitArray &wFaces)
{
    int adjFaceNum;

    mPolygons[polyCnt].AddTriFace(&faceNum);
    wFaces.Clear(faceNum);
    if (mType == OutputTriangles)
        return;

    for (int i = 0; i < 3; i++)
    {
        adjFaceNum = mAdjFaces->list[faceNum].f[i];
        // teapot generate -1 for some faces!?
        if (adjFaceNum == -1)
            continue;

        if (wFaces[adjFaceNum])
        {
            wFaces.Clear(adjFaceNum);
            if (mType == OutputQuads)
            {
                mPolygons[polyCnt].AddTriFace(&adjFaceNum);
                return;
            }
            AddCPTriFaces(adjFaceNum, polyCnt, uFaces, wFaces);
        }
    }
}

void
PMesh::GenEdges()
{
    int i, j;
    PMEdge newEdge;
    int numEdges = 0;
    int visEdge = 0;
    int face[3];
    int tFace;

    for (i = 0; i < GetPolygonCnt(); i++)
    {
        for (j = 0; j < mPolygons[i].GetTriFaceCnt(); j++)
        {
            /*
            if (mIsTextured) {
                tFace[0] = mOrgMesh.tvFace[mPolygons[i].GetTriFace(j)].t[0];
                tFace[1] = mOrgMesh.tvFace[mPolygons[i].GetTriFace(j)].t[1];
                tFace[2] = mOrgMesh.tvFace[mPolygons[i].GetTriFace(j)].t[2];
		    }
            */
            tFace = mPolygons[i].GetTriFace(j);
            face[0] = mOrgMesh.faces[tFace].v[0];
            face[1] = mOrgMesh.faces[tFace].v[1];
            face[2] = mOrgMesh.faces[tFace].v[2];
            //            visEdge = mOrgMesh.faces[tFace].flags | ~mSuppressEdges[tFace]; // ??
            visEdge = ~mSuppressEdges[tFace]; // ??

            if (visEdge & EDGE_A)
            {
                numEdges++;
                newEdge.SetEdgeVisiblity(VisibleEdge);
                newEdge.SetVIndex(0, face[0]);
                newEdge.SetVIndex(1, face[1]);
                newEdge.SetFace(0, tFace, EDGE_A);
                mPolygons[i].AddEdge(&newEdge);
                /*
                if (mIsTextured) {
                    newEdge.SetVIndex(0, tFace[0]);
                    newEdge.SetVIndex(1, tFace[1]);
                    mPolygons[i].AddTEdge(&newEdge);
                }
                */
            }
            if (visEdge & EDGE_B)
            {
                numEdges++;
                newEdge.SetEdgeVisiblity(VisibleEdge);
                newEdge.SetVIndex(0, face[1]);
                newEdge.SetVIndex(1, face[2]);
                newEdge.SetFace(0, tFace, EDGE_B);
                mPolygons[i].AddEdge(&newEdge);
                /*
                if (mIsTextured) {
                    newEdge.SetVIndex(0, tFace[1]);
                    newEdge.SetVIndex(1, tFace[2]);
                    mPolygons[i].AddTEdge(&newEdge);
                }
                */
            }
            if (visEdge & EDGE_C)
            {
                numEdges++;
                newEdge.SetEdgeVisiblity(VisibleEdge);
                newEdge.SetVIndex(0, face[2]);
                newEdge.SetVIndex(1, face[0]);
                newEdge.SetFace(0, tFace, EDGE_C);
                mPolygons[i].AddEdge(&newEdge);
                /*
                if (mIsTextured) {
                    newEdge.SetVIndex(0, tFace[2]);
                    newEdge.SetVIndex(1, tFace[0]);
                    mPolygons[i].AddTEdge(&newEdge);
                }
                */
            }
        }
    }
}

BOOL
PMesh::GenPolygons()
{
    int i;

    int numVerts = mOrgMesh.getNumVerts();
    mVMapping.SetSize(numVerts);
    mVMapping.ClearAll();

    int numTVerts = mOrgMesh.getNumTVerts();
    mTVMapping.SetSize(numTVerts);
    mTVMapping.ClearAll();

    int numTVerts2 = mOrgMesh.getNumMapVerts(2);
    mTVMapping2.SetSize(numTVerts2);
    mTVMapping2.ClearAll();
    int numTVerts3 = mOrgMesh.getNumMapVerts(3);
    mTVMapping3.SetSize(numTVerts3);
    mTVMapping3.ClearAll();
    int numTVerts4 = mOrgMesh.getNumMapVerts(4);
    mTVMapping4.SetSize(numTVerts4);
    mTVMapping4.ClearAll();
    int numTVerts5 = mOrgMesh.getNumMapVerts(5);
    mTVMapping5.SetSize(numTVerts5);
    mTVMapping5.ClearAll();
    int numTVerts6 = mOrgMesh.getNumMapVerts(6);
    mTVMapping6.SetSize(numTVerts6);
    mTVMapping6.ClearAll();

    // keep track of which vertices have been added
    BitArray aVerts;
    aVerts.SetSize(numVerts);
    aVerts.ClearAll();

    BitArray aTVerts;
    aTVerts.SetSize(numTVerts);
    aTVerts.ClearAll();
    BitArray aTVerts2;
    aTVerts2.SetSize(numTVerts2);
    aTVerts2.ClearAll();
    BitArray aTVerts3;
    aTVerts3.SetSize(numTVerts3);
    aTVerts3.ClearAll();
    BitArray aTVerts4;
    aTVerts4.SetSize(numTVerts4);
    aTVerts4.ClearAll();
    BitArray aTVerts5;
    aTVerts5.SetSize(numTVerts5);
    aTVerts5.ClearAll();
    BitArray aTVerts6;
    aTVerts6.SetSize(numTVerts6);
    aTVerts6.ClearAll();

    GenCPFaces(); // generate the polygons' triangle face lists
    GenEdges(); // generate the polygons' visible edge lists
    GenVertices(); // generate the list of verts that make up the polygons

    for (i = 0; i < numVerts; i++)
    {
        if (mVMapping[i] && !aVerts[i])
        {
            AddVertex(&mOrgMesh.getVert(i));
            aVerts.Set(i);
        }
    }
    if (mIsTextured)
    {
        for (i = 0; i < numTVerts; i++)
        {
            if (mTVMapping[i] && !aTVerts[i])
            {
                AddTVertex(&mOrgMesh.getTVert(i));
                aTVerts.Set(i);
            }
        }
    }
    if (mIsTextured2)
    {
        for (i = 0; i < numTVerts2; i++)
        {
            if (mTVMapping2[i] && !aTVerts2[i])
            {
                AddTVertex2(&mOrgMesh.mapVerts(2)[i]);
                aTVerts2.Set(i);
            }
        }
    }
    if (mIsTextured3)
    {
        for (i = 0; i < numTVerts3; i++)
        {
            if (mTVMapping3[i] && !aTVerts3[i])
            {
                AddTVertex3(&mOrgMesh.mapVerts(3)[i]);
                aTVerts3.Set(i);
            }
        }
    }
    if (mIsTextured4)
    {
        for (i = 0; i < numTVerts4; i++)
        {
            if (mTVMapping4[i] && !aTVerts4[i])
            {
                AddTVertex4(&mOrgMesh.mapVerts(4)[i]);
                aTVerts4.Set(i);
            }
        }
    }
    if (mIsTextured5)
    {
        for (i = 0; i < numTVerts5; i++)
        {
            if (mTVMapping5[i] && !aTVerts5[i])
            {
                AddTVertex5(&mOrgMesh.mapVerts(5)[i]);
                aTVerts5.Set(i);
            }
        }
    }
    if (mIsTextured6)
    {
        for (i = 0; i < numTVerts6; i++)
        {
            if (mTVMapping6[i] && !aTVerts6[i])
            {
                AddTVertex6(&mOrgMesh.mapVerts(6)[i]);
                aTVerts6.Set(i);
            }
        }
    }

    mPolygons.Shrink();
    for (i = 0; i < mPolygons.Count(); i++)
    {
        mPolygons[i].Shrink();
    }
    mVertices.Shrink();
    mTVertices.Shrink();
    mTVertices2.Shrink();
    mTVertices3.Shrink();
    mTVertices4.Shrink();
    mTVertices5.Shrink();
    mTVertices6.Shrink();

    // test for concavity

    for (i = 0; i < GetPolygonCnt(); i++)
    {
        Point3 o, p, q, r, s;
        int j;
        int u = mPolygons[i].GetVIndexCnt();
        int vex = 0;

        if (u > 3)
        {
            for (j = 0; j <= u + 1; j++)
            {
                int w = LookUpVert(mPolygons[i].GetVIndex(j % u));
                p = GetVertex(w);
                if (vex > 1)
                {
                    r = Normalize((q - o) ^ (p - q));
                    if (vex == 2)
                        s = r;
                    else if (DotProd(s, r) < 0.0)
                        return TRUE; // concave
                }
                o = q;
                q = p;
                vex++;
            }
        }
    }
    return FALSE; // convex
}

BOOL
PMesh::CheckIfColinear(Point3 first, Point3 second, Point3 third)
{
    Point3 distance0; // distance from vertLast to edge's v0
    float length0;
    Point3 distance1; // distance from edge's v0 to v1
    float length1;

    distance0 = second - first;
    length0 = Length(distance0);
    distance1 = third - second;
    length1 = Length(distance1);

    if (length0 && length1)
        return (DotProd(distance0, distance1) / (length0 * length1) > mCosEps);
    else
        return TRUE;
}

void
PMesh::GetNextEdge(int polyNum, int v0Prev, int vertNum,
                   BitArray &uVerts, BitArray &uEdges /*, BOOL tFace*/)
{
    int i;
    int v0, v1; // face vertex index
    int tV0 = 0; // texture face vertex index
    int tV1 = 0; // texture face vertex index
    int t2V0 = 0; // texture face vertex index
    int t2V1 = 0; // texture face vertex index
    int t3V0 = 0; // texture face vertex index
    int t3V1 = 0; // texture face vertex index
    int t4V0 = 0; // texture face vertex index
    int t4V1 = 0; // texture face vertex index
    int t5V0 = 0; // texture face vertex index
    int t5V1 = 0; // texture face vertex index
    int t6V0 = 0; // texture face vertex index
    int t6V1 = 0; // texture face vertex index
    Point3 vert;
    UVVert tVert;
    int eFace; // this the face0 of the edge
    int numEdges;
    //BitArray*   mapping;
    PMEdge edge;
    Point3 clv0, clv1, clv2; // colinear verts

    Point3 norm;
    RVertex *rv;
    int norCnt;
    int smGroup;

    //if (tFace) {
    //numEdges = mPolygons[polyNum].GetTEdgeCnt();
    //mapping  = &mTVMapping;
    //} else {
    numEdges = mPolygons[polyNum].GetEdgeCnt();
    //mapping  = &mVMapping;
    //}

    for (i = 0; i < numEdges; i++)
    {
        //if (tFace) {
        //edge = mPolygons[polyNum].GetTEdge(i);
        //} else {
        edge = mPolygons[polyNum].GetEdge(i);
        //}

        v0 = edge.GetVIndex(0);
        v1 = edge.GetVIndex(1);

        eFace = edge.GetFace(0);

        if (mIsTextured)
        {
            if (edge.GetFaceFlags(0) & EDGE_A)
            {
                tV0 = mOrgMesh.tvFace[eFace].t[0];
                tV1 = mOrgMesh.tvFace[eFace].t[1];
            }
            else if (edge.GetFaceFlags(0) & EDGE_B)
            {
                tV0 = mOrgMesh.tvFace[eFace].t[1];
                tV1 = mOrgMesh.tvFace[eFace].t[2];
            }
            else if (edge.GetFaceFlags(0) & EDGE_C)
            {
                tV0 = mOrgMesh.tvFace[eFace].t[2];
                tV1 = mOrgMesh.tvFace[eFace].t[0];
            }
        }

        if (mIsTextured2 && mOrgMesh.mapFaces(2)!=NULL)
        {
            if (edge.GetFaceFlags(0) & EDGE_A)
            {
                t2V0 = mOrgMesh.mapFaces(2)[eFace].t[0];
                t2V1 = mOrgMesh.mapFaces(2)[eFace].t[1];
            }
            else if (edge.GetFaceFlags(0) & EDGE_B)
            {
                t2V0 = mOrgMesh.mapFaces(2)[eFace].t[1];
                t2V1 = mOrgMesh.mapFaces(2)[eFace].t[2];
            }
            else if (edge.GetFaceFlags(0) & EDGE_C)
            {
                t2V0 = mOrgMesh.mapFaces(2)[eFace].t[2];
                t2V1 = mOrgMesh.mapFaces(2)[eFace].t[0];
            }
        }
        if (mIsTextured3 && mOrgMesh.mapFaces(3) != NULL)
        {
            if (edge.GetFaceFlags(0) & EDGE_A)
            {
                t3V0 = mOrgMesh.mapFaces(3)[eFace].t[0];
                t3V1 = mOrgMesh.mapFaces(3)[eFace].t[1];
            }
            else if (edge.GetFaceFlags(0) & EDGE_B)
            {
                t3V0 = mOrgMesh.mapFaces(3)[eFace].t[1];
                t3V1 = mOrgMesh.mapFaces(3)[eFace].t[2];
            }
            else if (edge.GetFaceFlags(0) & EDGE_C)
            {
                t3V0 = mOrgMesh.mapFaces(3)[eFace].t[2];
                t3V1 = mOrgMesh.mapFaces(3)[eFace].t[0];
            }
        }
        if (mIsTextured4 && mOrgMesh.mapFaces(4) != NULL)
        {
            if (edge.GetFaceFlags(0) & EDGE_A)
            {
                t4V0 = mOrgMesh.mapFaces(4)[eFace].t[0];
                t4V1 = mOrgMesh.mapFaces(4)[eFace].t[1];
            }
            else if (edge.GetFaceFlags(0) & EDGE_B)
            {
                t4V0 = mOrgMesh.mapFaces(4)[eFace].t[1];
                t4V1 = mOrgMesh.mapFaces(4)[eFace].t[2];
            }
            else if (edge.GetFaceFlags(0) & EDGE_C)
            {
                t4V0 = mOrgMesh.mapFaces(4)[eFace].t[2];
                t4V1 = mOrgMesh.mapFaces(4)[eFace].t[0];
            }
        }
        if (mIsTextured5 && mOrgMesh.mapFaces(5) != NULL)
        {
            if (edge.GetFaceFlags(0) & EDGE_A)
            {
                t5V0 = mOrgMesh.mapFaces(5)[eFace].t[0];
                t5V1 = mOrgMesh.mapFaces(5)[eFace].t[1];
            }
            else if (edge.GetFaceFlags(0) & EDGE_B)
            {
                t5V0 = mOrgMesh.mapFaces(5)[eFace].t[1];
                t5V1 = mOrgMesh.mapFaces(5)[eFace].t[2];
            }
            else if (edge.GetFaceFlags(0) & EDGE_C)
            {
                t5V0 = mOrgMesh.mapFaces(5)[eFace].t[2];
                t5V1 = mOrgMesh.mapFaces(5)[eFace].t[0];
            }
        }
        if (mIsTextured6 && mOrgMesh.mapFaces(6) != NULL)
        {
            if (edge.GetFaceFlags(0) & EDGE_A)
            {
                t6V0 = mOrgMesh.mapFaces(6)[eFace].t[0];
                t6V1 = mOrgMesh.mapFaces(6)[eFace].t[1];
            }
            else if (edge.GetFaceFlags(0) & EDGE_B)
            {
                t6V0 = mOrgMesh.mapFaces(6)[eFace].t[1];
                t6V1 = mOrgMesh.mapFaces(6)[eFace].t[2];
            }
            else if (edge.GetFaceFlags(0) & EDGE_C)
            {
                t6V0 = mOrgMesh.mapFaces(6)[eFace].t[2];
                t6V1 = mOrgMesh.mapFaces(6)[eFace].t[0];
            }
        }
        if (v0 == vertNum)
        {
            if (uEdges[i]) // must be a T check other edges
                continue;
            uEdges.Set(i);
            if (!uVerts[v1])
            {
                if (edge.GetEdgeVisiblity() == VisibleEdge)
                {
                    if (!uVerts[v0])
                    {
                        uVerts.Set(v0);
                        //if (!(*mapping)[v0])
                        //mapping->Set(v0);
                        if (!mVMapping[v0])
                            mVMapping.Set(v0);

                        vert = mOrgMesh.getVert(v0);
                        mPolygons[polyNum].AddVert(&vert);
                        mPolygons[polyNum].AddToPolygon(v0);

                        // add the vertex normal
                        norm = mPolygons[polyNum].GetFNormal();
                        rv = mOrgMesh.getRVertPtr(v0);
                        norCnt = (int)(rv->rFlags & NORCT_MASK);
                        smGroup = mOrgMesh.faces[eFace].getSmGroup();

                        if (rv->rFlags & SPECIFIED_NORMAL)
                        {
                            norm = rv->rn.getNormal();
                        }
                        else if (norCnt && smGroup)
                        {
                            if (norCnt == 1)
                                norm = rv->rn.getNormal();
                            else
                                for (int k = 0; k < norCnt; k++)
                                {
                                    if (rv->ern[k].getSmGroup() & smGroup)
                                    {
                                        norm = rv->ern[k].getNormal();
                                        break;
                                    }
                                }
                        }
                        mPolygons[polyNum].AddVNormal(&norm);

                        if (mIsTextured)
                        {
                            if (!mTVMapping[tV0])
                                mTVMapping.Set(tV0);
                            tVert = mOrgMesh.getTVert(tV0);
                            mPolygons[polyNum].AddTVert(&tVert);
                            mPolygons[polyNum].AddToTPolygon(tV0);
                        }
                        if (mIsTextured2 && mOrgMesh.mapFaces(2) != NULL)
                        {
                            if (!mTVMapping2[t2V0])
                                mTVMapping2.Set(t2V0);
                            tVert = mOrgMesh.mapVerts(2)[t2V0];
                            mPolygons[polyNum].AddTVert2(&tVert);
                            mPolygons[polyNum].AddToTPolygon2(t2V0);
                        }
                        if (mIsTextured3 && mOrgMesh.mapFaces(3) != NULL)
                        {
                            if (!mTVMapping3[t3V0])
                                mTVMapping3.Set(t3V0);
                            tVert = mOrgMesh.mapVerts(3)[t3V0];
                            mPolygons[polyNum].AddTVert3(&tVert);
                            mPolygons[polyNum].AddToTPolygon3(t3V0);
                        }
                        if (mIsTextured4 && mOrgMesh.mapFaces(4) != NULL)
                        {
                            if (!mTVMapping4[t4V0])
                                mTVMapping4.Set(t4V0);
                            tVert = mOrgMesh.mapVerts(4)[t4V0];
                            mPolygons[polyNum].AddTVert4(&tVert);
                            mPolygons[polyNum].AddToTPolygon4(t4V0);
                        }
                        if (mIsTextured5 && mOrgMesh.mapFaces(5) != NULL)
                        {
                            if (!mTVMapping5[t5V0])
                                mTVMapping5.Set(t5V0);
                            tVert = mOrgMesh.mapVerts(5)[t5V0];
                            mPolygons[polyNum].AddTVert5(&tVert);
                            mPolygons[polyNum].AddToTPolygon5(t5V0);
                        }
                        if (mIsTextured6 && mOrgMesh.mapFaces(6) != NULL)
                        {
                            if (!mTVMapping6[t6V0])
                                mTVMapping6.Set(t6V0);
                            tVert = mOrgMesh.mapVerts(6)[t6V0];
                            mPolygons[polyNum].AddTVert6(&tVert);
                            mPolygons[polyNum].AddToTPolygon6(t6V0);
                        }
                    }
                    else
                    { // see if it is colinear
                        if (v0Prev != -1)
                        { // if this is not the first edge
                            clv0 = mOrgMesh.getVert(v0Prev);
                            clv1 = mOrgMesh.getVert(v0);
                            clv2 = mOrgMesh.getVert(v1);
                            if (CheckIfColinear(clv0, clv1, clv2))
                            {
                                // remove the last vert and vert normal added
                                mPolygons[polyNum].RemoveLastVert();
                                mPolygons[polyNum].RemoveLastVNormal();
                                mPolygons[polyNum].RemoveLastFromPolygon();
                                uVerts.Clear(v0);
                                mVMapping.Clear(v0);
                                if (mIsTextured)
                                {
                                    mPolygons[polyNum].RemoveLastTVert();
                                    mPolygons[polyNum].RemoveLastFromTPolygon();
                                    mTVMapping.Clear(tV0);
                                }
                                if (mIsTextured2)
                                {
                                    mPolygons[polyNum].RemoveLastTVert2();
                                    mPolygons[polyNum].RemoveLastFromTPolygon2();
                                    mTVMapping2.Clear(t2V0);
                                }
                                if (mIsTextured3)
                                {
                                    mPolygons[polyNum].RemoveLastTVert3();
                                    mPolygons[polyNum].RemoveLastFromTPolygon3();
                                    mTVMapping3.Clear(t3V0);
                                }
                                if (mIsTextured4)
                                {
                                    mPolygons[polyNum].RemoveLastTVert4();
                                    mPolygons[polyNum].RemoveLastFromTPolygon4();
                                    mTVMapping4.Clear(t4V0);
                                }
                                if (mIsTextured5)
                                {
                                    mPolygons[polyNum].RemoveLastTVert5();
                                    mPolygons[polyNum].RemoveLastFromTPolygon5();
                                    mTVMapping5.Clear(t5V0);
                                }
                                if (mIsTextured6)
                                {
                                    mPolygons[polyNum].RemoveLastTVert6();
                                    mPolygons[polyNum].RemoveLastFromTPolygon6();
                                    mTVMapping6.Clear(t6V0);
                                }
                            }
                        }
                        else
                            assert(FALSE);
                    }
                    if (!uVerts[v1])
                    {
                        uVerts.Set(v1);
                        //if (!(*mapping)[v1])
                        //mapping->Set(v1);
                        if (!mVMapping[v1])
                            mVMapping.Set(v1);
                        vert = mOrgMesh.getVert(v1);
                        mPolygons[polyNum].AddVert(&vert);
                        mPolygons[polyNum].AddToPolygon(v1);

                        // add the vertex normal
                        norm = mPolygons[polyNum].GetFNormal();
                        rv = mOrgMesh.getRVertPtr(v1);
                        norCnt = (int)(rv->rFlags & NORCT_MASK);
                        smGroup = mOrgMesh.faces[eFace].getSmGroup();

                        if (rv->rFlags & SPECIFIED_NORMAL)
                        {
                            norm = rv->rn.getNormal();
                        }
                        else if (norCnt && smGroup)
                        {
                            if (norCnt == 1)
                                norm = rv->rn.getNormal();
                            else
                                for (int k = 0; k < norCnt; k++)
                                {
                                    if (rv->ern[k].getSmGroup() & smGroup)
                                    {
                                        norm = rv->ern[k].getNormal();
                                        break;
                                    }
                                }
                        }
                        mPolygons[polyNum].AddVNormal(&norm);

                        if (mIsTextured)
                        {
                            if (!mTVMapping[tV1])
                                mTVMapping.Set(tV1);
                            tVert = mOrgMesh.getTVert(tV1);
                            mPolygons[polyNum].AddTVert(&tVert);
                            mPolygons[polyNum].AddToTPolygon(tV1);
                        }
                        if (mIsTextured2 && mOrgMesh.mapFaces(2) != NULL)
                        {
                            if (!mTVMapping2[t2V1])
                                mTVMapping2.Set(t2V1);
                            tVert = mOrgMesh.mapVerts(2)[t2V1];
                            mPolygons[polyNum].AddTVert2(&tVert);
                            mPolygons[polyNum].AddToTPolygon2(t2V1);
                        }
                        if (mIsTextured3 && mOrgMesh.mapFaces(3) != NULL)
                        {
                            if (!mTVMapping3[t3V1])
                                mTVMapping3.Set(t3V1);
                            tVert = mOrgMesh.mapVerts(3)[t3V1];
                            mPolygons[polyNum].AddTVert3(&tVert);
                            mPolygons[polyNum].AddToTPolygon3(t3V1);
                        }
                        if (mIsTextured4 && mOrgMesh.mapFaces(4) != NULL)
                        {
                            if (!mTVMapping4[t4V1])
                                mTVMapping4.Set(t4V1);
                            tVert = mOrgMesh.mapVerts(4)[t4V1];
                            mPolygons[polyNum].AddTVert4(&tVert);
                            mPolygons[polyNum].AddToTPolygon4(t4V1);
                        }
                        if (mIsTextured5 && mOrgMesh.mapFaces(5) != NULL)
                        {
                            if (!mTVMapping5[t5V1])
                                mTVMapping5.Set(t5V1);
                            tVert = mOrgMesh.mapVerts(5)[t5V1];
                            mPolygons[polyNum].AddTVert5(&tVert);
                            mPolygons[polyNum].AddToTPolygon5(t5V1);
                        }
                        if (mIsTextured6 && mOrgMesh.mapFaces(6) != NULL)
                        {
                            if (!mTVMapping6[t6V1])
                                mTVMapping6.Set(t6V1);
                            tVert = mOrgMesh.mapVerts(6)[t6V1];
                            mPolygons[polyNum].AddTVert6(&tVert);
                            mPolygons[polyNum].AddToTPolygon6(t6V1);
                        }
                    }
                }
                GetNextEdge(polyNum, v0, v1, uVerts, uEdges);
                return;
            }
            else
            {
                // last edge; check if polygon has a hole or last/first vert is colinear
                PMEdge firstEdge;
                PMEdge tstEdge;

                // must have a hole in the face or a non contiguous edge list
                if (numEdges > uEdges.NumberSet())
                {
                    float shortDist; // shortest distance between verts of outside/inside edges
                    float tstDist; // current distance
                    Point3 vert0; // first vert0 of edge0
                    Point3 vert1; // vert0 of current edge

                    int tVert0 = 0; // index of the texture vert0 of the edge

                    int first = TRUE;

                    firstEdge = mPolygons[polyNum].GetEdge(0);
                    vert0 = mOrgMesh.getVert(firstEdge.GetVIndex(0));

                    for (int j = 0; j < numEdges; j++)
                    {
                        if (!uEdges[j])
                        {
                            // find the closest vert
                            tstEdge = mPolygons[polyNum].GetEdge(j);
                            vert1 = mOrgMesh.getVert(tstEdge.GetVIndex(0));
                            if (!first)
                            {
                                tstDist = Length(vert1 - vert0);
                                if (fabs(tstDist) < fabs(shortDist))
                                { // inside or outside
                                    shortDist = tstDist;
                                    edge = tstEdge;
                                }
                            }
                            else
                            {
                                shortDist = Length(vert1 - vert0);
                                edge = tstEdge;
                                first = FALSE;
                            }
                        }
                    }
                    if (!first)
                    { // this should always be the case, but ...
                        // insert first vert into the polylist
                        mPolygons[polyNum].AddToPolygon(firstEdge.GetVIndex(0));
                        if (mIsTextured)
                        {
                            if (firstEdge.GetFaceFlags(0) & EDGE_A)
                            {
                                tVert0 = mOrgMesh.tvFace[firstEdge.GetFace(0)].t[0];
                            }
                            else if (firstEdge.GetFaceFlags(0) & EDGE_B)
                            {
                                tVert0 = mOrgMesh.tvFace[firstEdge.GetFace(0)].t[1];
                            }
                            else if (firstEdge.GetFaceFlags(0) & EDGE_C)
                            {
                                tVert0 = mOrgMesh.tvFace[firstEdge.GetFace(0)].t[2];
                            }
                            mPolygons[polyNum].AddToTPolygon(tVert0);
                        }
                        if (mIsTextured2 && mOrgMesh.mapFaces(2) != NULL)
                        {
                            if (firstEdge.GetFaceFlags(0) & EDGE_A)
                            {
                                tVert0 = mOrgMesh.mapFaces(2)[firstEdge.GetFace(0)].t[0];
                            }
                            else if (firstEdge.GetFaceFlags(0) & EDGE_B)
                            {
                                tVert0 = mOrgMesh.mapFaces(2)[firstEdge.GetFace(0)].t[1];
                            }
                            else if (firstEdge.GetFaceFlags(0) & EDGE_C)
                            {
                                tVert0 = mOrgMesh.mapFaces(2)[firstEdge.GetFace(0)].t[2];
                            }
                            mPolygons[polyNum].AddToTPolygon2(tVert0);
                        }
                        if (mIsTextured3 && mOrgMesh.mapFaces(3) != NULL)
                        {
                            if (firstEdge.GetFaceFlags(0) & EDGE_A)
                            {
                                tVert0 = mOrgMesh.mapFaces(3)[firstEdge.GetFace(0)].t[0];
                            }
                            else if (firstEdge.GetFaceFlags(0) & EDGE_B)
                            {
                                tVert0 = mOrgMesh.mapFaces(3)[firstEdge.GetFace(0)].t[1];
                            }
                            else if (firstEdge.GetFaceFlags(0) & EDGE_C)
                            {
                                tVert0 = mOrgMesh.mapFaces(3)[firstEdge.GetFace(0)].t[2];
                            }
                            mPolygons[polyNum].AddToTPolygon3(tVert0);
                        }
                        if (mIsTextured4 && mOrgMesh.mapFaces(4) != NULL)
                        {
                            if (firstEdge.GetFaceFlags(0) & EDGE_A)
                            {
                                tVert0 = mOrgMesh.mapFaces(4)[firstEdge.GetFace(0)].t[0];
                            }
                            else if (firstEdge.GetFaceFlags(0) & EDGE_B)
                            {
                                tVert0 = mOrgMesh.mapFaces(4)[firstEdge.GetFace(0)].t[1];
                            }
                            else if (firstEdge.GetFaceFlags(0) & EDGE_C)
                            {
                                tVert0 = mOrgMesh.mapFaces(4)[firstEdge.GetFace(0)].t[2];
                            }
                            mPolygons[polyNum].AddToTPolygon4(tVert0);
                        }
                        if (mIsTextured5 && mOrgMesh.mapFaces(5) != NULL)
                        {
                            if (firstEdge.GetFaceFlags(0) & EDGE_A)
                            {
                                tVert0 = mOrgMesh.mapFaces(5)[firstEdge.GetFace(0)].t[0];
                            }
                            else if (firstEdge.GetFaceFlags(0) & EDGE_B)
                            {
                                tVert0 = mOrgMesh.mapFaces(5)[firstEdge.GetFace(0)].t[1];
                            }
                            else if (firstEdge.GetFaceFlags(0) & EDGE_C)
                            {
                                tVert0 = mOrgMesh.mapFaces(5)[firstEdge.GetFace(0)].t[2];
                            }
                            mPolygons[polyNum].AddToTPolygon5(tVert0);
                        }
                        if (mIsTextured6 && mOrgMesh.mapFaces(6) != NULL)
                        {
                            if (firstEdge.GetFaceFlags(0) & EDGE_A)
                            {
                                tVert0 = mOrgMesh.mapFaces(6)[firstEdge.GetFace(0)].t[0];
                            }
                            else if (firstEdge.GetFaceFlags(0) & EDGE_B)
                            {
                                tVert0 = mOrgMesh.mapFaces(6)[firstEdge.GetFace(0)].t[1];
                            }
                            else if (firstEdge.GetFaceFlags(0) & EDGE_C)
                            {
                                tVert0 = mOrgMesh.mapFaces(6)[firstEdge.GetFace(0)].t[2];
                            }
                            mPolygons[polyNum].AddToTPolygon6(tVert0);
                        }

                        // walk thru the inside edges
                        GetNextEdge(polyNum, -1, edge.GetVIndex(0), uVerts, uEdges);
                        // insert last vert into the polylist
                        mPolygons[polyNum].AddToPolygon(edge.GetVIndex(0));
                        if (mIsTextured)
                        {
                            if (edge.GetFaceFlags(0) & EDGE_A)
                            {
                                tVert0 = mOrgMesh.tvFace[edge.GetFace(0)].t[0];
                            }
                            else if (edge.GetFaceFlags(0) & EDGE_B)
                            {
                                tVert0 = mOrgMesh.tvFace[edge.GetFace(0)].t[1];
                            }
                            else if (edge.GetFaceFlags(0) & EDGE_C)
                            {
                                tVert0 = mOrgMesh.tvFace[edge.GetFace(0)].t[2];
                            }
                            mPolygons[polyNum].AddToTPolygon(tVert0);
                        }
                        if (mIsTextured2 && mOrgMesh.mapFaces(2) != NULL)
                        {
                            if (edge.GetFaceFlags(0) & EDGE_A)
                            {
                                tVert0 = mOrgMesh.mapFaces(2)[edge.GetFace(0)].t[0];
                            }
                            else if (edge.GetFaceFlags(0) & EDGE_B)
                            {
                                tVert0 = mOrgMesh.mapFaces(2)[edge.GetFace(0)].t[1];
                            }
                            else if (edge.GetFaceFlags(0) & EDGE_C)
                            {
                                tVert0 = mOrgMesh.mapFaces(2)[edge.GetFace(0)].t[2];
                            }
                            mPolygons[polyNum].AddToTPolygon2(tVert0);
                        }
                        if (mIsTextured3 && mOrgMesh.mapFaces(3) != NULL)
                        {
                            if (edge.GetFaceFlags(0) & EDGE_A)
                            {
                                tVert0 = mOrgMesh.mapFaces(3)[edge.GetFace(0)].t[0];
                            }
                            else if (edge.GetFaceFlags(0) & EDGE_B)
                            {
                                tVert0 = mOrgMesh.mapFaces(3)[edge.GetFace(0)].t[1];
                            }
                            else if (edge.GetFaceFlags(0) & EDGE_C)
                            {
                                tVert0 = mOrgMesh.mapFaces(3)[edge.GetFace(0)].t[2];
                            }
                            mPolygons[polyNum].AddToTPolygon3(tVert0);
                        }
                        if (mIsTextured4 && mOrgMesh.mapFaces(4) != NULL)
                        {
                            if (edge.GetFaceFlags(0) & EDGE_A)
                            {
                                tVert0 = mOrgMesh.mapFaces(4)[edge.GetFace(0)].t[0];
                            }
                            else if (edge.GetFaceFlags(0) & EDGE_B)
                            {
                                tVert0 = mOrgMesh.mapFaces(4)[edge.GetFace(0)].t[1];
                            }
                            else if (edge.GetFaceFlags(0) & EDGE_C)
                            {
                                tVert0 = mOrgMesh.mapFaces(4)[edge.GetFace(0)].t[2];
                            }
                            mPolygons[polyNum].AddToTPolygon4(tVert0);
                        }
                        if (mIsTextured5 && mOrgMesh.mapFaces(5) != NULL)
                        {
                            if (edge.GetFaceFlags(0) & EDGE_A)
                            {
                                tVert0 = mOrgMesh.mapFaces(5)[edge.GetFace(0)].t[0];
                            }
                            else if (edge.GetFaceFlags(0) & EDGE_B)
                            {
                                tVert0 = mOrgMesh.mapFaces(5)[edge.GetFace(0)].t[1];
                            }
                            else if (edge.GetFaceFlags(0) & EDGE_C)
                            {
                                tVert0 = mOrgMesh.mapFaces(5)[edge.GetFace(0)].t[2];
                            }
                            mPolygons[polyNum].AddToTPolygon5(tVert0);
                        }
                        if (mIsTextured6 && mOrgMesh.mapFaces(6) != NULL)
                        {
                            if (edge.GetFaceFlags(0) & EDGE_A)
                            {
                                tVert0 = mOrgMesh.mapFaces(6)[edge.GetFace(0)].t[0];
                            }
                            else if (edge.GetFaceFlags(0) & EDGE_B)
                            {
                                tVert0 = mOrgMesh.mapFaces(6)[edge.GetFace(0)].t[1];
                            }
                            else if (edge.GetFaceFlags(0) & EDGE_C)
                            {
                                tVert0 = mOrgMesh.mapFaces(6)[edge.GetFace(0)].t[2];
                            }
                            mPolygons[polyNum].AddToTPolygon6(tVert0);
                        }

                        // done
                        return;
                    }
                    else
                        assert(FALSE);
                }

                // check last vertex
                firstEdge = mPolygons[polyNum].GetEdge(0);
                if (v0Prev == -1)
                    assert(FALSE);
                clv0 = mOrgMesh.getVert(v0Prev);
                clv1 = mOrgMesh.getVert(v0);
                clv2 = mOrgMesh.getVert(v1);

                if (CheckIfColinear(clv0, clv1, clv2))
                {
                    // remove the last vert and vert normal added
                    mPolygons[polyNum].RemoveLastVert();
                    mPolygons[polyNum].RemoveLastVNormal();
                    mPolygons[polyNum].RemoveLastFromPolygon();
                    uVerts.Clear(v0);
                    mVMapping.Clear(v0);
                    if (mIsTextured)
                    {
                        mPolygons[polyNum].RemoveLastTVert();
                        mPolygons[polyNum].RemoveLastFromTPolygon();
                        mTVMapping.Clear(tV0);
                    }
                    if (mIsTextured2 && mOrgMesh.mapFaces(2) != NULL)
                    {
                        mPolygons[polyNum].RemoveLastTVert2();
                        mPolygons[polyNum].RemoveLastFromTPolygon2();
                        mTVMapping2.Clear(t2V0);
                    }
                    if (mIsTextured3 && mOrgMesh.mapFaces(3) != NULL)
                    {
                        mPolygons[polyNum].RemoveLastTVert3();
                        mPolygons[polyNum].RemoveLastFromTPolygon3();
                        mTVMapping3.Clear(t3V0);
                    }
                    if (mIsTextured4 && mOrgMesh.mapFaces(4) != NULL)
                    {
                        mPolygons[polyNum].RemoveLastTVert4();
                        mPolygons[polyNum].RemoveLastFromTPolygon4();
                        mTVMapping4.Clear(t4V0);
                    }
                    if (mIsTextured5 && mOrgMesh.mapFaces(5) != NULL)
                    {
                        mPolygons[polyNum].RemoveLastTVert5();
                        mPolygons[polyNum].RemoveLastFromTPolygon5();
                        mTVMapping5.Clear(t5V0);
                    }
                    if (mIsTextured6 && mOrgMesh.mapFaces(6) != NULL)
                    {
                        mPolygons[polyNum].RemoveLastTVert6();
                        mPolygons[polyNum].RemoveLastFromTPolygon6();
                        mTVMapping6.Clear(t6V0);
                    }
                }

                // check if last/first edges are colinear
                clv0 = mOrgMesh.getVert(v0);
                clv1 = mOrgMesh.getVert(firstEdge.GetVIndex(0));
                clv2 = mOrgMesh.getVert(firstEdge.GetVIndex(1));

                if (CheckIfColinear(clv0, clv1, clv2))
                {
                    // remove the first vert and normal added
                    mPolygons[polyNum].RemoveFirstVert();
                    mPolygons[polyNum].RemoveFirstVNormal();
                    mPolygons[polyNum].RemoveFirstFromPolygon();
                    uVerts.Clear(v1);
                    mVMapping.Clear(v1);
                    if (mIsTextured)
                    {
                        mPolygons[polyNum].RemoveFirstTVert();
                        mPolygons[polyNum].RemoveFirstFromTPolygon();
                        mTVMapping.Clear(tV1);
                    }
                    if (mIsTextured2 && mOrgMesh.mapFaces(2) != NULL)
                    {
                        mPolygons[polyNum].RemoveFirstTVert2();
                        mPolygons[polyNum].RemoveFirstFromTPolygon2();
                        mTVMapping2.Clear(t2V1);
                    }
                    if (mIsTextured3 && mOrgMesh.mapFaces(3) != NULL)
                    {
                        mPolygons[polyNum].RemoveFirstTVert3();
                        mPolygons[polyNum].RemoveFirstFromTPolygon3();
                        mTVMapping3.Clear(t3V1);
                    }
                    if (mIsTextured4 && mOrgMesh.mapFaces(4) != NULL)
                    {
                        mPolygons[polyNum].RemoveFirstTVert4();
                        mPolygons[polyNum].RemoveFirstFromTPolygon4();
                        mTVMapping4.Clear(t4V1);
                    }
                    if (mIsTextured5 && mOrgMesh.mapFaces(5) != NULL)
                    {
                        mPolygons[polyNum].RemoveFirstTVert5();
                        mPolygons[polyNum].RemoveFirstFromTPolygon5();
                        mTVMapping5.Clear(t5V1);
                    }
                    if (mIsTextured6 && mOrgMesh.mapFaces(6) != NULL)
                    {
                        mPolygons[polyNum].RemoveFirstTVert6();
                        mPolygons[polyNum].RemoveFirstFromTPolygon6();
                        mTVMapping6.Clear(t6V1);
                    }
                }
            }
        }
    }
}

void
PMesh::GenVertices()
{
    // used to store the verts used
    BitArray uVerts;
    // used to store the edges used
    BitArray uEdges;
    int i;
    int numVerts = mOrgMesh.getNumVerts();
    //int numTVerts   = mOrgMesh.getNumTVerts();

    for (i = 0; i < GetPolygonCnt(); i++)
    {
        uEdges.SetSize(mPolygons[i].GetEdgeCnt());
        uEdges.ClearAll();
        uVerts.SetSize(numVerts);
        uVerts.ClearAll();
        GetNextEdge(i, -1, mPolygons[i].GetEdge(0).GetVIndex(0), uVerts, uEdges);
    }

    /*
    if (mIsTextured) {
        for (i = 0; i < GetPolygonCnt(); i++) {
            uEdges.SetSize(mPolygons[i].GetTEdgeCnt());
            uEdges.ClearAll();
            uVerts.SetSize(numTVerts);
            uVerts.ClearAll();
            GetNextEdge(i, -1, mPolygons[i].GetTEdge(0).GetVIndex(0), uVerts, uEdges, TRUE);
        }
    }
    */
}

// cheeze function to find the adjoining edge
AEdge
AdjoiningEdge(DWORD *face0, DWORD *face1)
{
    AEdge edge;
    if (face0[0] == face1[1] && face0[1] == face1[0])
    {
        edge.e0 = 0;
        edge.e1 = 0;
        return edge;
    }
    if (face0[0] == face1[2] && face0[1] == face1[1])
    {
        edge.e0 = 0;
        edge.e1 = 1;
        return edge;
    }
    if (face0[0] == face1[0] && face0[1] == face1[2])
    {
        edge.e0 = 0;
        edge.e1 = 2;
        return edge;
    }
    if (face0[1] == face1[1] && face0[2] == face1[0])
    {
        edge.e0 = 1;
        edge.e1 = 0;
        return edge;
    }
    if (face0[1] == face1[2] && face0[2] == face1[1])
    {
        edge.e0 = 1;
        edge.e1 = 1;
        return edge;
    }
    if (face0[1] == face1[0] && face0[2] == face1[2])
    {
        edge.e0 = 1;
        edge.e1 = 2;
        return edge;
    }
    if (face0[2] == face1[1] && face0[0] == face1[0])
    {
        edge.e0 = 2;
        edge.e1 = 0;
        return edge;
    }
    if (face0[2] == face1[2] && face0[0] == face1[1])
    {
        edge.e0 = 2;
        edge.e1 = 1;
        return edge;
    }
    if (face0[2] == face1[0] && face0[0] == face1[2])
    {
        edge.e0 = 2;
        edge.e1 = 2;
        return edge;
    }
    // when all else fails
    edge.e0 = 0;
    edge.e1 = 0;
    return edge;
}

int
PMesh::LookUpTVert(int num)
{
    int cnt = 0;
    for (int i = 0; i < num; i++)
    {
        if (mTVMapping[i])
            cnt++;
    }
    return cnt;
}

int
PMesh::LookUpTVert2(int num)
{
    int cnt = 0;
    for (int i = 0; i < num; i++)
    {
        if (mTVMapping2[i])
            cnt++;
    }
    return cnt;
}
int
PMesh::LookUpTVert3(int num)
{
    int cnt = 0;
    for (int i = 0; i < num; i++)
    {
        if (mTVMapping3[i])
            cnt++;
    }
    return cnt;
}
int
PMesh::LookUpTVert4(int num)
{
    int cnt = 0;
    for (int i = 0; i < num; i++)
    {
        if (mTVMapping4[i])
            cnt++;
    }
    return cnt;
}
int
PMesh::LookUpTVert5(int num)
{
    int cnt = 0;
    for (int i = 0; i < num; i++)
    {
        if (mTVMapping5[i])
            cnt++;
    }
    return cnt;
}
int
PMesh::LookUpTVert6(int num)
{
    int cnt = 0;
    for (int i = 0; i < num; i++)
    {
        if (mTVMapping6[i])
            cnt++;
    }
    return cnt;
}

int
PMesh::LookUpVert(int num)
{
    int cnt = 0;
    for (int i = 0; i < num; i++)
    {
        if (mVMapping[i])
            cnt++;
    }
    return cnt;
}
