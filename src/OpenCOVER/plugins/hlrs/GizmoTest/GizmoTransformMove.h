///////////////////////////////////////////////////////////////////////////////////////////////////
// LibGizmo
// File Name : 
// Creation : 10/01/2012
// Author : Cedric Guillemet
// Description : LibGizmo
//
///Copyright (C) 2012 Cedric Guillemet
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
//of the Software, and to permit persons to whom the Software is furnished to do
///so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
///FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// 



#ifndef GIZMOTRANSFORMMOVE_H__
#define GIZMOTRANSFORMMOVE_H__


#include "GizmoTransform.h"

class CGizmoTransformMove : public CGizmoTransform  
{

public:
	CGizmoTransformMove();
	virtual ~CGizmoTransformMove();

	// return true if gizmo transform capture mouse
	virtual bool OnMouseDown(unsigned int x, unsigned int y);
	virtual void OnMouseMove(unsigned int x, unsigned int y);
	virtual void OnMouseUp(unsigned int x, unsigned int y);

	virtual void Draw();
	// snap

	virtual void SetSnap(float snapx, float snapy, float snapz)
	{
		m_MoveSnap = tvector3(snapx, snapy, snapz);
	}
    virtual void SetSnap(const float snap) {}

	tvector3 GetMoveSnap()
	{
		return m_MoveSnap;
	}

	virtual void ApplyTransform(tvector3& trans, bool bAbsolute);

protected:
	enum MOVETYPE
	{
		MOVE_NONE,
		MOVE_X,
		MOVE_Y,
		MOVE_Z,
		MOVE_XY,
		MOVE_XZ,
		MOVE_YZ,
		MOVE_XYZ
	};
	MOVETYPE m_MoveType,m_MoveTypePredict;
	//tplane m_plan;
	//tvector3 m_LockVertex;
	tvector3 m_MoveSnap;

	bool GetOpType(MOVETYPE &type, unsigned int x, unsigned int y);
	tvector3 RayTrace(tvector3& rayOrigin, tvector3& rayDir, tvector3& norm);
};

#endif // !defined(AFX_GIZMOTRANSFORMMOVE_H__8276C568_C663_463C_AE7F_B913E2A712A4__INCLUDED_)
