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


#ifndef GIZMOTRANSFORMROTATE_H__
#define GIZMOTRANSFORMROTATE_H__

#include "GizmoTransform.h"

class CGizmoTransformRotate : public CGizmoTransform  
{

public:
	CGizmoTransformRotate();
	virtual ~CGizmoTransformRotate();

	// return true if gizmo transform capture mouse
	virtual bool OnMouseDown(unsigned int x, unsigned int y);
	virtual void OnMouseMove(unsigned int x, unsigned int y);
	virtual void OnMouseUp(unsigned int x, unsigned int y);

	virtual void Draw();

    virtual void SetSnap(const float snap) {m_AngleSnap = snap; }
    virtual void SetSnap(float snapx, float snapy, float snapz) {}
    /*
	void SetAngleSnap(float snap)
	{
		m_AngleSnap = snap;
	}
    */

	float GetAngleSnap()
	{
		return m_AngleSnap;
	}

	virtual void ApplyTransform(tvector3& trans, bool bAbsolute);

protected:
	enum ROTATETYPE
	{
		ROTATE_NONE,
		ROTATE_X,
		ROTATE_Y,
		ROTATE_Z,
		ROTATE_SCREEN,
		ROTATE_TWIN
	};
	ROTATETYPE m_RotateType,m_RotateTypePredict;
	tplane m_plan;
	tvector3 m_LockVertex,m_LockVertex2;
	float m_Ng2;
	tvector3 m_Vtx,m_Vty,m_Vtz;
	tvector3 m_Axis,m_Axis2;
	tmatrix m_OrigScale,m_InvOrigScale;
	float m_AngleSnap;


	bool GetOpType(ROTATETYPE &type, unsigned int x, unsigned int y);
	bool CheckRotatePlan(tvector3 &vNorm, float factor, const tvector3 &rayOrig,const tvector3 &rayDir,int id);
	void Rotate1Axe(const tvector3& rayOrigin,const tvector3& rayDir);

};

#endif // !defined(AFX_GIZMOTRANSFORMROTATE_H__F92EF632_4BAE_49CE_B7B8_213704C82589__INCLUDED_)
