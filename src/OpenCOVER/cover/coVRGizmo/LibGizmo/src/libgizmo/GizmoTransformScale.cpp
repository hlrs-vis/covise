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


#include "stdafx.h"
#include "GizmoTransformScale.h"
#ifdef MAC_OS
#import <OpenGL/OpenGL.h>
#else
#include <GL/gl.h>
#endif

extern tvector3 ptd;


IGizmo *CreateScaleGizmo()
{
    return new CGizmoTransformScale;
}


//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CGizmoTransformScale::CGizmoTransformScale() : CGizmoTransform()
{
	m_ScaleType = SCALE_NONE;
}

CGizmoTransformScale::~CGizmoTransformScale()
{

}



bool CGizmoTransformScale::GetOpType(SCALETYPE &type, unsigned int x, unsigned int y)
{
	// init
	tvector3 trss(GetTransformedVector(0).Length(),
		GetTransformedVector(1).Length(),
		GetTransformedVector(2).Length());

	m_LockX = x;
    m_LockY = y;
	m_svgMatrix = *m_pMatrix;

	tmatrix mt;
	mt = *m_pMatrix;
	mt.NoTrans();
	mt.Inverse();


	//tmatrix mt;
    if (mLocation == LOCATE_LOCAL)
    {
	    mt = *m_pMatrix;
	    mt.Inverse();
    }
    else
    {
        // world
        mt.Translation( -m_pMatrix->V4.position);
    }

	// ray casting
	tvector3 rayOrigin,rayDir,df2;
	BuildRay(x, y, rayOrigin, rayDir);

	// plan 1 : X/Z
	df2 = RayTrace2(rayOrigin, rayDir, GetTransformedVector(1), mt, trss);


	if ( (df2.x<0.2f) && (df2.z<0.2f) && (df2.x>0) && (df2.z>0)) { type = SCALE_XYZ; return true; }
	else if ( ( df2.x >= 0 ) && (df2.x <= 1) && ( fabs(df2.z) < 0.1f ) ) { type = SCALE_X; return true;	}
	else if ( ( df2.z >= 0 ) && (df2.z <= 1) && ( fabs(df2.x) < 0.1f ) ) { type = SCALE_Z; return true;	}
	else if ( (df2.x<0.5f) && (df2.z<0.5f) && (df2.x>0) && (df2.z>0)) {	type = SCALE_XZ; return true; }
	else
	{
		//plan 2 : X/Y
		df2 = RayTrace2(rayOrigin, rayDir, GetTransformedVector(2), mt, trss);

		if ( (df2.x<0.2f) && (df2.y<0.2f) && (df2.x>0) && (df2.y>0)) { type = SCALE_XYZ; return true; }
		else if ( ( df2.x >= 0 ) && (df2.x <= 1) && ( fabs(df2.y) < 0.1f ) ) { type = SCALE_X; return true;	}
		else if ( ( df2.y >= 0 ) && (df2.y <= 1) && ( fabs(df2.x) < 0.1f ) ) { type = SCALE_Y; return true; }
		else if ( (df2.x<0.5f) && (df2.y<0.5f) && (df2.x>0) && (df2.y>0)) { type = SCALE_XY; return true; }
		else
		{
			//plan 3: Y/Z
			df2 = RayTrace2(rayOrigin, rayDir, GetTransformedVector(0), mt, trss);

			if ( (df2.y<0.2f) && (df2.z<0.2f) && (df2.y>0) && (df2.z>0)) { type = SCALE_XYZ; return true; }
			else if ( ( df2.y >= 0 ) && (df2.y <= 1) && ( fabs(df2.z) < 0.1f ) ) { type = SCALE_Y; return true;	}
			else if ( ( df2.z >= 0 ) && (df2.z <= 1) && ( fabs(df2.y) < 0.1f ) ) { type = SCALE_Z; return true;	}
			else if ( (df2.y<0.5f) && (df2.z<0.5f) && (df2.y>0) && (df2.z>0)) { type = SCALE_YZ; return true; }
		}
	}

	type = SCALE_NONE;
	return false;
}


bool CGizmoTransformScale::OnMouseDown(unsigned int x, unsigned int y)
{
	if (m_pMatrix)
	{
		return GetOpType(m_ScaleType, x, y);
	}

	m_ScaleType = SCALE_NONE;
	return false;
}

void CGizmoTransformScale::SnapScale(float &val)
{
	if (m_bUseSnap)
	{
		val*=(100.0f);
		SnapIt(val,m_ScaleSnap);
		val/=(100.0f);
	}
}

void CGizmoTransformScale::OnMouseMove(unsigned int x, unsigned int y)
{
	if (m_ScaleType != SCALE_NONE)
	{
		tvector3 rayOrigin,rayDir,df, inters, machin;
		tvector3 scVect,scVect2;

		BuildRay(x, y, rayOrigin, rayDir);
		m_plan.RayInter(inters,rayOrigin,rayDir);

		switch (m_ScaleType)
		{
		case SCALE_XZ: scVect = tvector3(1,0,1); break;
		case SCALE_X:  scVect = tvector3(1,0,0); break;
		case SCALE_Z:  scVect = tvector3(0,0,1); break;
		case SCALE_XY: scVect = tvector3(1,1,0); break;
		case SCALE_YZ: scVect = tvector3(0,1,1); break;
		case SCALE_Y:  scVect = tvector3(0,1,0); break;
		case SCALE_XYZ:scVect = tvector3(1,1,1); break;
		}

		df = inters-m_pMatrix->GetTranslation();
		df/=GetScreenFactor();
		scVect2 = tvector3(1,1,1) - scVect;

		if (m_ScaleType == SCALE_XYZ)
		{
			int difx = x - m_LockX;
			float lng2 = 1.0f + ( float(difx) / 200.0f);
			SnapScale(lng2);
			scVect *=lng2;
		}
		else
		{
            int difx = x - m_LockX;
            int dify = y - m_LockY;

            float len = sqrtf( (float)(difx*difx) + (float)(dify*dify) );

            float lng2 = len /100.f;
            /*
			float lng2 = ( df.Dot(m_LockVertex));
            char tmps[512];
            sprintf(tmps, "%5.4f\n", lng2 );
            OutputDebugStringA( tmps );


			if (lng2 < 1.f)
			{
				if ( lng2<= 0.001f )
					lng2 = 0.001f;
				else
				{
					//lng2+=4.f;
					lng2/=5.f;
				}
			}
            else
            {
                int a = 1;
            }
            */
			SnapScale(lng2);
			scVect *= lng2;
			scVect += scVect2;
		}


		tmatrix mt,mt2;



		mt.Scaling(scVect);

        mt2.Identity();
		mt2.SetLine(0,GetTransformedVector(0));
		mt2.SetLine(1,GetTransformedVector(1));
		mt2.SetLine(2,GetTransformedVector(2));

		//mt2.Translation(0,0,0);
		//mt.Multiply(mt2);

        if (mLocation == LOCATE_WORLD)
        {
            mt2 = mt * m_svgMatrix;
        }
        else
        {
		    mt2 = mt * m_svgMatrix;//.Multiply(m_svgMatrix);
        }
		*m_pMatrix = mt2;
        //if (mTransform) mTransform->Update();
	}
	else
	{
		// predict move
		if (m_pMatrix)
		{
			GetOpType(m_ScaleTypePredict, x, y);
		}
	}

}

void CGizmoTransformScale::OnMouseUp(unsigned int x, unsigned int y)
{
	m_ScaleType = SCALE_NONE;
}

void CGizmoTransformScale::Draw()
{
	if (m_pMatrix)
	{
		ComputeScreenFactor();

		//glDisable(GL_DEPTH_TEST);
		tvector3 orig(m_pMatrix->m16[12],m_pMatrix->m16[13],m_pMatrix->m16[14]);


		// axis
		tvector3 axeX(1,0,0),axeY(0,1,0),axeZ(0,0,1);
        if (mLocation == LOCATE_LOCAL)
        {
            axeX.TransformVector(*m_pMatrix);
		    axeY.TransformVector(*m_pMatrix);
		    axeZ.TransformVector(*m_pMatrix);
		    axeX.Normalize();
		    axeY.Normalize();
		    axeZ.Normalize();
        }

		DrawTri(orig, 0.5f*GetScreenFactor(),((m_ScaleTypePredict==SCALE_XZ)||(m_ScaleTypePredict==SCALE_XYZ)), axeX, axeZ);
		DrawTri(orig, 0.5f*GetScreenFactor(),((m_ScaleTypePredict==SCALE_XY)||(m_ScaleTypePredict==SCALE_XYZ)), axeX, axeY);
		DrawTri(orig, 0.5f*GetScreenFactor(),((m_ScaleTypePredict==SCALE_YZ)||(m_ScaleTypePredict==SCALE_XYZ)), axeY, axeZ);

		axeX*=GetScreenFactor();
		axeY*=GetScreenFactor();
		axeZ*=GetScreenFactor();


		// plan1
		if (m_ScaleTypePredict != SCALE_X)
			DrawAxis(orig,axeX,axeY,axeZ,0.05f,0.83f,vector4(1,0,0,1));
		else
			DrawAxis(orig,axeX,axeY,axeZ,0.05f,0.83f,vector4(1,1,1,1));

		//plan2
		if (m_ScaleTypePredict != SCALE_Y)
			DrawAxis(orig,axeY,axeX,axeZ,0.05f,0.83f,vector4(0,1,0,1));
		else
			DrawAxis(orig,axeY,axeX,axeZ,0.05f,0.83f,vector4(1,1,1,1));

		//plan3
		if (m_ScaleTypePredict != SCALE_Z)
			DrawAxis(orig,axeZ,axeX,axeY,0.05f,0.83f,vector4(0,0,1,1));
		else
			DrawAxis(orig,axeZ,axeX,axeY,0.05f,0.83f,vector4(1,1,1,1));

/*
		// debug
		glPointSize(20);
		glBegin(GL_POINTS);
		glVertex3fv(&ptd.x);
		glEnd();

		glEnable(GL_DEPTH_TEST);
        */
#if 0
#ifdef WIN32
    GDD->GetD3D9Device()->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
    GDD->GetD3D9Device()->SetRenderState(D3DRS_CULLMODE , D3DCULL_NONE );
    GDD->GetD3D9Device()->SetRenderState(D3DRS_ZENABLE , D3DZB_TRUE);
	GDD->GetD3D9Device()->SetRenderState(D3DRS_ALPHATESTENABLE , FALSE);
	GDD->GetD3D9Device()->SetRenderState(D3DRS_ZWRITEENABLE , TRUE);
#endif
	extern RenderingState_t GRenderingState;
	GRenderingState.mAlphaTestEnable = 0;
	GRenderingState.mZWriteEnable = 1;
	GRenderingState.mBlending = 0;
	GRenderingState.mCulling = 0;
	GRenderingState.mZTestType = 1;
#endif
	}


}

void CGizmoTransformScale::ApplyTransform(tvector3& trans, bool bAbsolute)
{
	if (bAbsolute)
	{
		tmatrix m_InvOrigScale,m_OrigScale;

		m_OrigScale.Scaling(GetTransformedVector(0).Length(),
		GetTransformedVector(1).Length(),
		GetTransformedVector(2).Length());

		m_InvOrigScale.Inverse(m_OrigScale);
		m_svgMatrix = *m_pMatrix;

		tmatrix mt;
		mt.Scaling(trans.x/100.0f,trans.y/100.0f,trans.z/100.0f);
		mt.Multiply(m_InvOrigScale);
		mt.Multiply(m_svgMatrix);
		*m_pMatrix=mt;
	}
	else
	{
		tmatrix mt,mt2;
		m_svgMatrix = *m_pMatrix;
		mt.Scaling(trans.x/100.0f,trans.y/100.0f,trans.z/100.0f);

		mt2.SetLine(0,GetTransformedVector(0));
		mt2.SetLine(1,GetTransformedVector(1));
		mt2.SetLine(2,GetTransformedVector(2));
		mt2.Translation(0,0,0);
		mt.Multiply(mt2);
		mt.Multiply(m_svgMatrix);
		*m_pMatrix = mt;
	}

}
