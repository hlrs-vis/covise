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


#ifndef GIZMOTRANSFORMSCALE_H__
#define GIZMOTRANSFORMSCALE_H__

#include "GizmoTransform.h"

class CGizmoTransformScale : public CGizmoTransform  
{
public:
	CGizmoTransformScale();
	virtual ~CGizmoTransformScale();

	// return true if gizmo transform capture mouse
	virtual bool OnMouseDown(unsigned int x, unsigned int y);
	virtual void OnMouseMove(unsigned int x, unsigned int y);
	virtual void OnMouseUp(unsigned int x, unsigned int y);

	virtual void Draw();

    /*
	void SetScaleSnap(float snap)
	{
		m_ScaleSnap = snap;
	}
    */
    virtual void SetSnap(const float snap) {m_ScaleSnap = snap; }
    virtual void SetSnap(float snapx, float snapy, float snapz) {}

	float GetScaleSnap()
	{
		return m_ScaleSnap;
	}

	virtual void ApplyTransform(tvector3& trans, bool bAbsolute);

protected:
	enum SCALETYPE
	{
		SCALE_NONE,
		SCALE_X,
		SCALE_Y,
		SCALE_Z,
		SCALE_XY,
		SCALE_XZ,
		SCALE_YZ,
		SCALE_XYZ
	};
	SCALETYPE m_ScaleType,m_ScaleTypePredict;


	unsigned int m_LockX, m_LockY;
	float m_ScaleSnap;

	bool GetOpType(SCALETYPE &type, unsigned int x, unsigned int y);
	//tvector3 RayTrace(tvector3& rayOrigin, tvector3& rayDir, tvector3& norm, tmatrix& mt, tvector3 trss);
	void SnapScale(float &val);

};

#endif // !defined(AFX_GIZMOTRANSFORMSCALE_H__85E46839_15B0_4CE4_A85A_547015AF853A__INCLUDED_)
