/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
	FILE:			EvalCpolor.cpp
	DESCRIPTION:	Vertex Color Renderer
	CREATED BY:		Christer Janson
	HISTORY:		Created Monday, December 12, 1996

 *>	Copyright (c) 1997 Kinetix, All Rights Reserved.
 **********************************************************************/
//
// Description:
// These functions calculates the diffuse or ambient color at each vertex
// or face of an INode.
//
// Exports:
// BOOL calcMixedVertexColors(INode*, TimeValue, int, ColorTab&, EvalColProgressCallback* callb = NULL);
//      This function calculates the interpolated diffuse or ambient
//      color at each vertex of an INode.
//      Usage: Pass in a node pointer and the TimeValue to generate
//      a list of Colors corresponding to each vertex in the mesh
//      Use the int flag to specify if you want to have diffuse or
//      ambient colors, or if you want to use the lights in the scene.
//      Note:
//        You are responsible for deleting the Color objects in the table.
//      Additional note:
//        Since materials are assigned by face, this function renders each
//        face connected to the specific vertex (at the point of the vertex)
//        and then mixes the colors.
//
//***************************************************************************

#define LIGHT_AMBIENT 0x00
#define LIGHT_DIFFUSE 0x01
#define LIGHT_SCENELIGHT 0x02

typedef Tab<Color *> ColorTab;

class SingleVertexColor
{
public:
    ~SingleVertexColor();
    ColorTab vertexColors;
};

typedef Tab<SingleVertexColor *> VertexColorTab;
typedef BOOL (*EVALCOL_PROGRESS)(float);

class EvalColProgressCallback
{
public:
    virtual BOOL progress(float prog) = 0;
};

BOOL calcMixedVertexColors(INode *node, TimeValue t, int lightModel, ColorTab &vxColTab, EvalColProgressCallback *callb = NULL);
BOOL calcVertexColors(INode *node, TimeValue t, int lightModel, VertexColorTab &vxColTab, EvalColProgressCallback *callb = NULL);
