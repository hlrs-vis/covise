/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
	FILE: pmesh.h

	DESCRIPTION:  simple Polygon class module

	CREATED BY: greg finch

	HISTORY: created 1 december, 1996

 *>	Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#include "meshadj.h"

#define COPLANAR_NORMAL_EPSILON 0.1 // in degrees
#define LineEPSILON 1e-03f

struct AEdge
{
    int e0;
    int e1;
};

AEdge AdjoiningEdge(DWORD *, DWORD *);

enum PType
{
    OutputTriangles,
    OutputQuads,
    OutputNGons,
    OutputVisibleEdges
};
enum EType
{
    HiddenEdge,
    CoplanarEdge,
    VisibleEdge
};

/*
#define     Edge_A     (1<<0)
#define     Edge_B     (1<<1)
#define     Edge_C     (1<<2)
*/
class PMEdge
{
    EType mType; // type of edge (hidden, etc...)
    int mVertex0; // index of the first vertex
    int mVertex1; // index of the ccw vertex
    int mFace0; // the face on the left
    int mFace0Flags; // this is the the ccw index numbers ei. AB, BC, CA
    int mFace1; // the face on the right
    int mFace1Flags; // this is the the ccw index numbers ei. AB, BC, CA
public:
    PMEdge()
    {
        mType = HiddenEdge;
        mVertex0 = -1;
        mVertex1 = -1;
        mFace0 = -1;
        mFace1 = -1;
    }
    int GetVIndex(int i)
    {
        return (i ? mVertex1 : mVertex0);
    }
    void SetVIndex(int i, int v)
    {
        (i ? mVertex1 = v : mVertex0 = v);
    }
    int GetFaceFlags(int i)
    {
        return (i ? mFace1Flags : mFace0Flags);
    }
    int GetFace(int i)
    {
        return (i ? mFace1 : mFace0);
    }
    void SetFace(int i, int face, int flags)
    {
        (i ? mFace1 = face : mFace0 = face);
        (i ? mFace1Flags = flags : mFace0Flags = flags);
    }
    void SetEdgeVisiblity(EType type)
    {
        mType = type;
    }
    EType GetEdgeVisiblity()
    {
        return mType;
    }
};

class PMPoly
{
    PType mType; // type of polygon tri, quad, ngon, etc...
    Point3 mFNormal; // the face normal
    Tab<PMEdge> mEdges; // the polygon edges
    //Tab<PMEdge> mTEdges;    // the polygon texture edges
    Tab<int> mPolygon; // the the ccw vert indices
    Tab<int> mTPolygon; // the the ccw texture vert indices
    Tab<int> mTPolygon2; // the the ccw texture vert indices
    Tab<int> mTPolygon3; // the the ccw texture vert indices
    Tab<int> mTPolygon4; // the the ccw texture vert indices
    Tab<int> mTPolygon5; // the the ccw texture vert indices
    Tab<int> mTPolygon6; // the the ccw texture vert indices
    Tab<int> mTriFaces; // the coplanar tri faces indices
    Tab<UVVert> mTVerts; // the polygon texture verts
    Tab<UVVert> mTVerts2; // the polygon texture verts
    Tab<UVVert> mTVerts3; // the polygon texture verts
    Tab<UVVert> mTVerts4; // the polygon texture verts
    Tab<UVVert> mTVerts5; // the polygon texture verts
    Tab<UVVert> mTVerts6; // the polygon texture verts
    Tab<Point3> mVNormals; // the vertex normals
    Tab<Point3> mVerts; // the polygon verts
public:
    PMPoly()
    {
        mType = OutputTriangles;
    }
    ~PMPoly()
    {
        if (mEdges.Count())
            mEdges.Delete(0, mEdges.Count());
        //if (mTEdges.Count())
        //    mTEdges.Delete(0,   mTEdges.Count());
        if (mPolygon.Count())
            mPolygon.Delete(0, mPolygon.Count());
        if (mTPolygon.Count())
            mTPolygon.Delete(0, mTPolygon.Count());
        if (mTriFaces.Count())
            mTriFaces.Delete(0, mTriFaces.Count());
        if (mTVerts.Count())
            mTVerts.Delete(0, mTVerts.Count());
        if (mTVerts2.Count())
            mTVerts2.Delete(0, mTVerts2.Count());
        if (mTVerts3.Count())
            mTVerts3.Delete(0, mTVerts3.Count());
        if (mTVerts4.Count())
            mTVerts4.Delete(0, mTVerts4.Count());
        if (mTVerts5.Count())
            mTVerts5.Delete(0, mTVerts5.Count());
        if (mTVerts6.Count())
            mTVerts6.Delete(0, mTVerts6.Count());
        if (mVNormals.Count())
            mVNormals.Delete(0, mVNormals.Count());
        if (mVerts.Count())
            mVerts.Delete(0, mVerts.Count());
    }

    void Shrink()
    {
        mEdges.Shrink();
        //mTEdges.Shrink();
        mPolygon.Shrink();
        mTPolygon.Shrink();
        mTriFaces.Shrink();
        //mNormals.Shrink();
        mTVerts.Shrink();
        mTVerts2.Shrink();
        mTVerts3.Shrink();
        mTVerts4.Shrink();
        mTVerts5.Shrink();
        mTVerts6.Shrink();
        mVerts.Shrink();
    }
    PType GetType()
    {
        return mType;
    }
    Point3 GetFNormal()
    {
        return mFNormal;
    }
    void SetFNormal(Point3 fNormal)
    {
        mFNormal = fNormal;
    }
    int GetEdgeCnt()
    {
        return mEdges.Count();
    }
    PMEdge GetEdge(int i)
    {
        return mEdges[i];
    }
    int AddEdge(PMEdge *newEdge)
    // return the location inserted
    {
        return mEdges.Insert(mEdges.Count(), 1, newEdge);
    }
    /*
    int     GetTEdgeCnt()
                { return mTEdges.Count(); }
    PMEdge  GetTEdge(int i)
                { return mTEdges[i]; }
    int     AddTEdge(PMEdge* newEdge)
             // return the location inserted
                { return mTEdges.Insert(mTEdges.Count(), 1, newEdge);}
    */
    int GetTriFaceCnt()
    {
        return mTriFaces.Count();
    }
    int GetTriFace(int i)
    {
        return mTriFaces[i];
    }
    int AddTriFace(int *index)
    {
        return mTriFaces.Insert(mTriFaces.Count(), 1, index);
    }
    int GetVertCnt()
    {
        return mVerts.Count();
    }
    Point3 GetVert(int i)
    {
        return mVerts[i];
    }
    int AddVert(Point3 *newVert)
    {
        return mVerts.Insert(mVerts.Count(), 1, newVert);
    }
    int RemoveLastVert()
    {
        return mVerts.Delete(mVerts.Count() - 1, 1);
    }
    int RemoveFirstVert()
    {
        return mVerts.Delete(0, 1);
    }
    int GetTVertCnt()
    {
        return mTVerts.Count();
    }
    UVVert GetTVert(int i)
    {
        return mTVerts[i];
    }
    int AddTVert(UVVert *newTVert)
    {
        return mTVerts.Insert(mTVerts.Count(), 1, newTVert);
    }
    int RemoveLastTVert()
    {
        return mTVerts.Delete(mTVerts.Count() - 1, 1);
    }
    int RemoveFirstTVert()
    {
        return mTVerts.Delete(0, 1);
    }
    int GetTVert2Cnt()
    {
        return mTVerts2.Count();
    }
    UVVert GetTVert2(int i)
    {
        return mTVerts2[i];
    }
    int AddTVert2(UVVert *newTVert)
    {
        return mTVerts2.Insert(mTVerts2.Count(), 1, newTVert);
    }
    int RemoveLastTVert2()
    {
        return mTVerts2.Delete(mTVerts2.Count() - 1, 1);
    }
    int RemoveFirstTVert2()
    {
        return mTVerts2.Delete(0, 1);
    }
    int GetTVert3Cnt()
    {
        return mTVerts3.Count();
    }
    UVVert GetTVert3(int i)
    {
        return mTVerts3[i];
    }
    int AddTVert3(UVVert *newTVert)
    {
        return mTVerts3.Insert(mTVerts3.Count(), 1, newTVert);
    }
    int RemoveLastTVert3()
    {
        return mTVerts3.Delete(mTVerts3.Count() - 1, 1);
    }
    int RemoveFirstTVert3()
    {
        return mTVerts3.Delete(0, 1);
    }
    int GetTVert4Cnt()
    {
        return mTVerts4.Count();
    }
    UVVert GetTVert4(int i)
    {
        return mTVerts4[i];
    }
    int AddTVert4(UVVert *newTVert)
    {
        return mTVerts4.Insert(mTVerts4.Count(), 1, newTVert);
    }
    int RemoveLastTVert4()
    {
        return mTVerts4.Delete(mTVerts4.Count() - 1, 1);
    }
    int RemoveFirstTVert4()
    {
        return mTVerts4.Delete(0, 1);
    }
    int GetTVert5Cnt()
    {
        return mTVerts5.Count();
    }
    UVVert GetTVert5(int i)
    {
        return mTVerts5[i];
    }
    int AddTVert5(UVVert *newTVert)
    {
        return mTVerts5.Insert(mTVerts5.Count(), 1, newTVert);
    }
    int RemoveLastTVert5()
    {
        return mTVerts5.Delete(mTVerts5.Count() - 1, 1);
    }
    int RemoveFirstTVert5()
    {
        return mTVerts5.Delete(0, 1);
    }
    int GetTVert6Cnt()
    {
        return mTVerts6.Count();
    }
    UVVert GetTVert6(int i)
    {
        return mTVerts6[i];
    }
    int AddTVert6(UVVert *newTVert)
    {
        return mTVerts6.Insert(mTVerts6.Count(), 1, newTVert);
    }
    int RemoveLastTVert6()
    {
        return mTVerts6.Delete(mTVerts6.Count() - 1, 1);
    }
    int RemoveFirstTVert6()
    {
        return mTVerts6.Delete(0, 1);
    }
    int GetVNormalCnt()
    {
        return mVNormals.Count();
    }
    Point3 GetVNormal(int i)
    {
        return mVNormals[i];
    }
    int AddVNormal(Point3 *newVNormal)
    {
        return mVNormals.Insert(mVNormals.Count(), 1, newVNormal);
    }
    int RemoveFirstVNormal()
    {
        return mVNormals.Delete(0, 1);
    }
    int RemoveLastVNormal()
    {
        return mVNormals.Delete(mVNormals.Count() - 1, 1);
    }
    int GetVIndexCnt()
    {
        return mPolygon.Count();
    }
    int GetVIndex(int i)
    {
        return mPolygon[i];
    }
    int AddToPolygon(int index)
    {
        return mPolygon.Insert(mPolygon.Count(), 1, &index);
    }
    int RemoveLastFromPolygon()
    {
        return mPolygon.Delete(mPolygon.Count() - 1, 1);
    }
    int RemoveFirstFromPolygon()
    {
        return mPolygon.Delete(0, 1);
    }
    int GetTVIndexCnt()
    {
        return mTPolygon.Count();
    }
    int GetTVIndex(int i)
    {
        return mTPolygon[i];
    }
    int AddToTPolygon(int index)
    {
        return mTPolygon.Insert(mTPolygon.Count(), 1, &index);
    }
    int RemoveLastFromTPolygon()
    {
        return mTPolygon.Delete(mTPolygon.Count() - 1, 1);
    }
    int RemoveFirstFromTPolygon()
    {
        return mTPolygon.Delete(0, 1);
    }
    int GetTVIndexCnt2()
    {
        return mTPolygon2.Count();
    }
    int GetTVIndex2(int i)
    {
        return mTPolygon2[i];
    }
    int AddToTPolygon2(int index)
    {
        return mTPolygon2.Insert(mTPolygon2.Count(), 1, &index);
    }
    int RemoveLastFromTPolygon2()
    {
        return mTPolygon2.Delete(mTPolygon2.Count() - 1, 1);
    }
    int RemoveFirstFromTPolygon2()
    {
        return mTPolygon2.Delete(0, 1);
    }
    int GetTVIndexCnt3()
    {
        return mTPolygon3.Count();
    }
    int GetTVIndex3(int i)
    {
        return mTPolygon3[i];
    }
    int AddToTPolygon3(int index)
    {
        return mTPolygon3.Insert(mTPolygon3.Count(), 1, &index);
    }
    int RemoveLastFromTPolygon3()
    {
        return mTPolygon3.Delete(mTPolygon3.Count() - 1, 1);
    }
    int RemoveFirstFromTPolygon3()
    {
        return mTPolygon3.Delete(0, 1);
    }
    int GetTVIndexCnt4()
    {
        return mTPolygon4.Count();
    }
    int GetTVIndex4(int i)
    {
        return mTPolygon4[i];
    }
    int AddToTPolygon4(int index)
    {
        return mTPolygon4.Insert(mTPolygon4.Count(), 1, &index);
    }
    int RemoveLastFromTPolygon4()
    {
        return mTPolygon4.Delete(mTPolygon4.Count() - 1, 1);
    }
    int RemoveFirstFromTPolygon4()
    {
        return mTPolygon4.Delete(0, 1);
    }
    int GetTVIndexCnt5()
    {
        return mTPolygon5.Count();
    }
    int GetTVIndex5(int i)
    {
        return mTPolygon5[i];
    }
    int AddToTPolygon5(int index)
    {
        return mTPolygon5.Insert(mTPolygon5.Count(), 1, &index);
    }
    int RemoveLastFromTPolygon5()
    {
        return mTPolygon5.Delete(mTPolygon5.Count() - 1, 1);
    }
    int RemoveFirstFromTPolygon5()
    {
        return mTPolygon5.Delete(0, 1);
    }
    int GetTVIndexCnt6()
    {
        return mTPolygon6.Count();
    }
    int GetTVIndex6(int i)
    {
        return mTPolygon6[i];
    }
    int AddToTPolygon6(int index)
    {
        return mTPolygon6.Insert(mTPolygon6.Count(), 1, &index);
    }
    int RemoveLastFromTPolygon6()
    {
        return mTPolygon6.Delete(mTPolygon6.Count() - 1, 1);
    }
    int RemoveFirstFromTPolygon6()
    {
        return mTPolygon6.Delete(0, 1);
    }
};

class PMesh
{
public:
    PMesh(Mesh &, PType, BOOL *);
    ~PMesh();

    BOOL GenPolygons(); // generate the polygons:
    //    returns TRUE if concave
    int GetPolygonCnt() // get the number of polygons
    {
        return mPolygons.Count();
    };
    PMPoly *GetPolygon(int num) // get a polygon
    {
        return &mPolygons[num];
    };
    int GetVertexCnt() // get the number of vertices
    {
        return mVertices.Count();
    };
    Point3 GetVertex(int num) // get a vertex
    {
        return mVertices[num];
    };
    int GetTVertex2Cnt() // get the number of vertices
    {
        return mTVertices2.Count();
    };
    Point3 GetTVertex2(int num) // get a vertex
    {
        return mTVertices2[num];
    };
    int GetTVertex3Cnt() // get the number of vertices
    {
        return mTVertices3.Count();
    };
    Point3 GetTVertex3(int num) // get a vertex
    {
        return mTVertices3[num];
    };
    int GetTVertex4Cnt() // get the number of vertices
    {
        return mTVertices4.Count();
    };
    Point3 GetTVertex4(int num) // get a vertex
    {
        return mTVertices4[num];
    };
    int GetTVertex5Cnt() // get the number of vertices
    {
        return mTVertices5.Count();
    };
    Point3 GetTVertex5(int num) // get a vertex
    {
        return mTVertices5[num];
    };
    int GetTVertex6Cnt() // get the number of vertices
    {
        return mTVertices6.Count();
    };
    Point3 GetTVertex6(int num) // get a vertex
    {
        return mTVertices6[num];
    };
    int GetTVertexCnt() // get the number of vertices
    {
        return mTVertices.Count();
    };
    Point3 GetTVertex(int num) // get a vertex
    {
        return mTVertices[num];
    };
    int LookUpVert(int num); // get the pmesh to trimesh mapping
    int LookUpTVert(int num); // get the pmesh to trimesh mapping
    int LookUpTVert2(int num); // get the pmesh to trimesh mapping
    int LookUpTVert3(int num); // get the pmesh to trimesh mapping
    int LookUpTVert4(int num); // get the pmesh to trimesh mapping
    int LookUpTVert5(int num); // get the pmesh to trimesh mapping
    int LookUpTVert6(int num); // get the pmesh to trimesh mapping

private:
    Mesh mOrgMesh; // the orginal mesh which PMesh is generated from
    PType mType; // enum { OutputTriangles, OutputQuads, OutputNGons, OutputVisibleEdges };
    BOOL mIsTextured; // has UV coords
    BOOL mIsTextured2; // has UV coords2
    BOOL mIsTextured3; // has UV coords3
    BOOL mIsTextured4; // has UV coords4
    BOOL mIsTextured5; // has UV coords5
    BOOL mIsTextured6; // has UV coords6
    double mCosEps; // allowable normals angular delta
    AdjEdgeList *mAdjEdges; // used to check for coplanar faces
    AdjFaceList *mAdjFaces; // used to check for coplanar faces
    int *mSuppressEdges; // internal, suppressed edges

    Tab<PMPoly> mPolygons; // used to store the polygons
    int AddPolygon(PMPoly *newPoly) // return the location inserted
    {
        return mPolygons.Insert(mPolygons.Count(), 1, newPoly);
    }

    Tab<Point3> mVertices; // used to store the meshes verticies
    BitArray mVMapping; // used to store the vertices used
    int AddVertex(Point3 *newVert)
    {
        return mVertices.Insert(mVertices.Count(), 1, newVert);
    }

    Tab<Point3> mTVertices2; // used to store the meshes verticies
    BitArray mTVMapping2; // used to store the texture vertices used
    int AddTVertex2(Point3 *newVert)
    {
        return mTVertices2.Insert(mTVertices2.Count(), 1, newVert);
    }
    Tab<Point3> mTVertices3; // used to store the meshes verticies
    BitArray mTVMapping3; // used to store the texture vertices used
    int AddTVertex3(Point3 *newVert)
    {
        return mTVertices3.Insert(mTVertices3.Count(), 1, newVert);
    }
    Tab<Point3> mTVertices4; // used to store the meshes verticies
    BitArray mTVMapping4; // used to store the texture vertices used
    int AddTVertex4(Point3 *newVert)
    {
        return mTVertices4.Insert(mTVertices4.Count(), 1, newVert);
    }
    Tab<Point3> mTVertices5; // used to store the meshes verticies
    BitArray mTVMapping5; // used to store the texture vertices used
    int AddTVertex5(Point3 *newVert)
    {
        return mTVertices5.Insert(mTVertices5.Count(), 1, newVert);
    }
    Tab<Point3> mTVertices6; // used to store the meshes verticies
    BitArray mTVMapping6; // used to store the texture vertices used
    int AddTVertex6(Point3 *newVert)
    {
        return mTVertices6.Insert(mTVertices6.Count(), 1, newVert);
    }
    Tab<Point3> mTVertices; // used to store the meshes verticies
    BitArray mTVMapping; // used to store the texture vertices used
    int AddTVertex(Point3 *newVert)
    {
        return mTVertices.Insert(mTVertices.Count(), 1, newVert);
    }

    void GenCPFaces(); // generate the coplanar faces
    void AddCPTriFaces(int faceNum, int polyCnt, BitArray &uFaces, BitArray &wFaces); //add tri face to polygon
    void GenEdges(); // generate the visible edges
    void GenVertices(); // generate the vertices
    void CheckAdjacentFaces(int, BitArray &);
    BOOL CheckIfMatIDsMatch(int faceNum, int adjFaceNum); // make sure matID is the same
    BOOL CheckIfUVVertMatch(AEdge edge, int faceNum, int adjFaceNum); // make sure the UVVert is the same
    BOOL CheckIfCoplanar(int, int); // check if a face and an adj face are coplanar
    BOOL CheckIfColinear(Point3 first, Point3 second, Point3 third); // check if three verts are colinear
    void GetNextEdge(int polyNum, int v0Prev, int edgeNum, BitArray &uVerts, BitArray &uEdges); // traverse the edges
    BOOL IsValidFace(int); // check for bad faces
};
