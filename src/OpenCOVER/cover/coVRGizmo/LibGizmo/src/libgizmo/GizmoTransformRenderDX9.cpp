///////////////////////////////////////////////////////////////////////////////////////////////////
// Zenith Engine
// File Name : GizmoTransform.h
// Creation : 12/07/2007
// Author : Cedric Guillemet
// Description : 
//
///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; version 2 of the License.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "GizmoTransformRender.h"
#include <D3D9.h>

#ifdef WIN32
typedef struct {
    float x;
    float y;
    float z;

    tulong diffuse;
} PSM_LVERTEX;
IDirect3DVertexDeclaration9* GGizmoVertDecl = NULL;
#endif
void InitDecl()
{
#ifdef WIN32
    if (!GGizmoVertDecl)
    {
        D3DVERTEXELEMENT9 decl1[] = 
        {
            { 0, 0,  D3DDECLTYPE_FLOAT3,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
            { 0, 12, D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR,  0 },
            D3DDECL_END()
        };

        GDD->GetD3D9Device()->CreateVertexDeclaration( decl1, &GGizmoVertDecl );
    }
    GDD->GetD3D9Device()->SetRenderState(D3DRS_ZENABLE , D3DZB_FALSE);
    GDD->GetD3D9Device()->SetTexture(0,NULL);
    GDD->GetD3D9Device()->SetTexture(1,NULL);
    GDD->GetD3D9Device()->SetTexture(2,NULL);
    GDD->GetD3D9Device()->SetTexture(3,NULL);
    GDD->GetD3D9Device()->SetTexture(4,NULL);
    GDD->GetD3D9Device()->SetTexture(5,NULL);
    GDD->GetD3D9Device()->SetTexture(6,NULL);
    GDD->GetD3D9Device()->SetTexture(7,NULL);
    GDD->GetD3D9Device()->SetRenderState(D3DRS_COLORVERTEX , true);
    //GDD->GetD3D9Device()->SetRenderState(D3DRS_LIGHTING , true);
    GDD->GetD3D9Device()->SetRenderState(D3DRS_AMBIENT  , 0xFFFFFFFF);
    GDD->GetD3D9Device()->SetVertexShader(NULL);
    GDD->GetD3D9Device()->SetPixelShader(NULL);

    
#endif
}
void CGizmoTransformRender::DrawCircle(const tvector3 &orig,float r,float g,float b, const tvector3 &vtx, const tvector3 &vty)
{
#ifdef WIN32
    InitDecl();
    PSM_LVERTEX svVts[51];
	for (int i = 0; i <= 50 ; i++)
	{
		tvector3 vt;
		vt = vtx * cos((2*ZPI/50)*i);
		vt += vty * sin((2*ZPI/50)*i);
		vt += orig;
        svVts[i].diffuse = 0xFF000000 + (int(r*255.0f) << 16) + (int(g*255.0f) << 8) + (int(b*255.0f));
        svVts[i].x = vt.x;
        svVts[i].y = vt.y;
        svVts[i].z = vt.z;
	}
    IDirect3DDevice9 *pDev = GDD->GetD3D9Device();
    GDD->GetD3D9Device()->SetVertexDeclaration(GGizmoVertDecl);
	GDD->GetD3D9Device()->DrawPrimitiveUP(D3DPT_LINESTRIP , 50, svVts, sizeof(PSM_LVERTEX));
    GDD->GetD3D9Device()->SetRenderState(D3DRS_ZENABLE , D3DZB_TRUE);
#endif
}


void CGizmoTransformRender::DrawCircleHalf(const tvector3 &orig,float r,float g,float b,const tvector3 &vtx, const tvector3 &vty, tplane &camPlan)
{
#ifdef WIN32
    InitDecl();
    int inc = 0;
    PSM_LVERTEX svVts[51];
	for (int i = 0; i < 30 ; i++)
	{
		tvector3 vt;
		vt = vtx * cos((ZPI/30)*i);
		vt += vty * sin((ZPI/30)*i);
		vt +=orig;
		if (camPlan.DotNormal(vt))
        {
            svVts[inc].diffuse = 0xFF000000 + (int(r*255.0f) << 16) + (int(g*255.0f) << 8) + (int(b*255.0f));
            svVts[inc].x = vt.x;
            svVts[inc].y = vt.y;
            svVts[inc].z = vt.z;
            inc ++;
        }
	}
    IDirect3DDevice9 *pDev = GDD->GetD3D9Device();
    GDD->GetD3D9Device()->SetVertexDeclaration(GGizmoVertDecl);
	GDD->GetD3D9Device()->DrawPrimitiveUP(D3DPT_LINESTRIP , inc-1, svVts, sizeof(PSM_LVERTEX));
    GDD->GetD3D9Device()->SetRenderState(D3DRS_ZENABLE , D3DZB_TRUE);
#endif   
}

void CGizmoTransformRender::DrawAxis(const tvector3 &orig, const tvector3 &axis, const tvector3 &vtx, const tvector3 &vty, float fct,float fct2, const tvector4 &col)
{
#ifdef WIN32
    InitDecl();
    PSM_LVERTEX svVts[100];
    int inc = 0;
    IDirect3DDevice9 *pDev = GDD->GetD3D9Device();

    svVts[0].x = orig.x;
    svVts[0].y = orig.y;
    svVts[0].z = orig.z;
    svVts[0].diffuse = 0xFF000000 + (int(col.x*255.0f) << 16) + (int(col.y*255.0f) << 8) + (int(col.z*255.0f));

    svVts[1].x = orig.x + axis.x;
    svVts[1].y = orig.y + axis.y;
    svVts[1].z = orig.z + axis.z;
    svVts[1].diffuse = svVts[0].diffuse;

    GDD->GetD3D9Device()->SetVertexDeclaration(GGizmoVertDecl);
	GDD->GetD3D9Device()->DrawPrimitiveUP(D3DPT_LINELIST , 1, svVts, sizeof(PSM_LVERTEX));

	//glBegin(GL_TRIANGLE_FAN);
	for (int i=0;i<=30;i++)
	{
		tvector3 pt;
		pt = vtx * cos(((2*ZPI)/30.0f)*i)*fct;
		pt+= vty * sin(((2*ZPI)/30.0f)*i)*fct;
		pt+=axis*fct2;
		pt+=orig;
		//glVertex3fv(&pt.x);

        svVts[inc].x = pt.x;
        svVts[inc].y = pt.y;
        svVts[inc].z = pt.z;
        svVts[inc++].diffuse = svVts[0].diffuse;

		pt = vtx * cos(((2*ZPI)/30.0f)*(i+1))*fct;
		pt+= vty * sin(((2*ZPI)/30.0f)*(i+1))*fct;
		pt+=axis*fct2;
		pt+=orig;
		//glVertex3fv(&pt.x);

        svVts[inc].x = pt.x;
        svVts[inc].y = pt.y;
        svVts[inc].z = pt.z;
        svVts[inc++].diffuse = svVts[0].diffuse;

		//glVertex3f(orig.x+axis.x,orig.y+axis.y,orig.z+axis.z);

        svVts[inc].x = orig.x+axis.x;
        svVts[inc].y = orig.y+axis.y;
        svVts[inc].z = orig.z+axis.z;
        svVts[inc++].diffuse = svVts[0].diffuse;

	}
	//glEnd();
    GDD->GetD3D9Device()->SetVertexDeclaration(GGizmoVertDecl);
    GDD->GetD3D9Device()->DrawPrimitiveUP(D3DPT_TRIANGLEFAN , 91, svVts, sizeof(PSM_LVERTEX));
    GDD->GetD3D9Device()->SetRenderState(D3DRS_ZENABLE , D3DZB_TRUE);
#endif
}

void CGizmoTransformRender::DrawCamem(const tvector3& orig,const tvector3& vtx,const tvector3& vty,float ng)
{
#ifdef WIN32
    InitDecl();
    IDirect3DDevice9 *pDev = GDD->GetD3D9Device();
    GDD->GetD3D9Device()->SetRenderState(D3DRS_SRCBLEND , D3DBLEND_SRCALPHA);
    GDD->GetD3D9Device()->SetRenderState(D3DRS_DESTBLEND , D3DBLEND_INVSRCALPHA);
    GDD->GetD3D9Device()->SetRenderState(D3DRS_ALPHABLENDENABLE, TRUE);
    GDD->GetD3D9Device()->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE );

    PSM_LVERTEX svVts[52];

    svVts[0].x = orig.x;
    svVts[0].y = orig.y;
    svVts[0].z = orig.z;
    svVts[0].diffuse = 0x80FFFF00;

	for (int i = 0 ; i <= 50 ; i++)
	{
		tvector3 vt;
		vt = vtx * cos(((ng)/50)*i);
		vt += vty * sin(((ng)/50)*i);
		vt+=orig;
		//glVertex3f(vt.x,vt.y,vt.z);

        svVts[i+1].x = vt.x;
        svVts[i+1].y = vt.y;
        svVts[i+1].z = vt.z;
        svVts[i+1].diffuse = svVts[0].diffuse;
	}

    GDD->GetD3D9Device()->SetVertexDeclaration(GGizmoVertDecl);
	GDD->GetD3D9Device()->DrawPrimitiveUP(D3DPT_TRIANGLEFAN , 50, svVts, sizeof(PSM_LVERTEX));
        GDD->GetD3D9Device()->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
	
	for (int  i = 0 ; i < 52 ; i++)
        svVts[i].diffuse = 0xFFFFFF33;

    GDD->GetD3D9Device()->SetVertexDeclaration(GGizmoVertDecl);
    GDD->GetD3D9Device()->DrawPrimitiveUP(D3DPT_LINESTRIP , 50, svVts, sizeof(PSM_LVERTEX));

    GDD->GetD3D9Device()->SetRenderState(D3DRS_CULLMODE , D3DCULL_CCW );
    GDD->GetD3D9Device()->SetRenderState(D3DRS_ZENABLE , D3DZB_TRUE);
#endif
}

void CGizmoTransformRender::DrawQuad(const tvector3& orig, float size, bool bSelected, const tvector3& axisU, const tvector3 &axisV)
{
#ifdef WIN32
    InitDecl();
    IDirect3DDevice9 *pDev = GDD->GetD3D9Device();
    GDD->GetD3D9Device()->SetRenderState(D3DRS_SRCBLEND , D3DBLEND_SRCALPHA);
    GDD->GetD3D9Device()->SetRenderState(D3DRS_DESTBLEND , D3DBLEND_INVSRCALPHA);
    GDD->GetD3D9Device()->SetRenderState(D3DRS_ALPHABLENDENABLE, TRUE);
    GDD->GetD3D9Device()->SetRenderState(D3DRS_CULLMODE , D3DCULL_NONE );

	PSM_LVERTEX svVts[5];
	svVts[0].x = orig.x;
    svVts[0].y = orig.y;
    svVts[0].z = orig.z;
	svVts[1].x = orig.x + (axisU.x * size);
    svVts[1].y = orig.y + (axisU.y * size);
    svVts[1].z = orig.z + (axisU.z * size);

	svVts[2].x = orig.x + (axisV.x * size);
    svVts[2].y = orig.y + (axisV.y * size);
    svVts[2].z = orig.z + (axisV.z * size);
	svVts[3].x = orig.x + (axisU.x + axisV.x)*size;
    svVts[3].y = orig.y + (axisU.y + axisV.y)*size;
    svVts[3].z = orig.z + (axisU.z + axisV.z)*size;
    svVts[4].x = orig.x;
    svVts[4].y = orig.y;
    svVts[4].z = orig.z;


	if (!bSelected)
        svVts[0].diffuse = svVts[1].diffuse = svVts[2].diffuse = svVts[3].diffuse = svVts[4].diffuse = 0x80FFFF00;
	else
        svVts[0].diffuse = svVts[1].diffuse = svVts[2].diffuse = svVts[3].diffuse = svVts[4].diffuse = 0xA0FFFFFF;
		
    GDD->GetD3D9Device()->SetVertexDeclaration(GGizmoVertDecl);
    GDD->GetD3D9Device()->DrawPrimitiveUP(D3DPT_TRIANGLESTRIP , 2, svVts, sizeof(PSM_LVERTEX));

	if (!bSelected)
        svVts[0].diffuse = svVts[1].diffuse = svVts[2].diffuse = svVts[3].diffuse = svVts[4].diffuse = 0xFFFFFF30;
	else
        svVts[0].diffuse = svVts[1].diffuse = svVts[2].diffuse = svVts[3].diffuse = svVts[4].diffuse = 0xA0FFFF90;


	svVts[3].x = orig.x + (axisV.x * size);
    svVts[3].y = orig.y + (axisV.y * size);
    svVts[3].z = orig.z + (axisV.z * size);
	svVts[2].x = orig.x + (axisU.x + axisV.x)*size;
    svVts[2].y = orig.y + (axisU.y + axisV.y)*size;
    svVts[2].z = orig.z + (axisU.z + axisV.z)*size;

    GDD->GetD3D9Device()->SetVertexDeclaration(GGizmoVertDecl);
    GDD->GetD3D9Device()->DrawPrimitiveUP(D3DPT_LINESTRIP , 4, svVts, sizeof(PSM_LVERTEX));


	GDD->GetD3D9Device()->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
    GDD->GetD3D9Device()->SetRenderState(D3DRS_CULLMODE , D3DCULL_CCW );
    GDD->GetD3D9Device()->SetRenderState(D3DRS_ZENABLE , D3DZB_TRUE);
#endif
}


void CGizmoTransformRender::DrawTri(const tvector3& orig, float size, bool bSelected, const tvector3& axisU, const tvector3& axisV)
{
#ifdef WIN32
    InitDecl();
    IDirect3DDevice9 *pDev = GDD->GetD3D9Device();
    GDD->GetD3D9Device()->SetRenderState(D3DRS_SRCBLEND , D3DBLEND_SRCALPHA);
    GDD->GetD3D9Device()->SetRenderState(D3DRS_DESTBLEND , D3DBLEND_INVSRCALPHA);
    GDD->GetD3D9Device()->SetRenderState(D3DRS_ALPHABLENDENABLE, TRUE);
    GDD->GetD3D9Device()->SetRenderState(D3DRS_CULLMODE , D3DCULL_NONE );


	tvector3 pts[3];
	pts[0] = orig;

	pts[1] = (axisU );
	pts[2] = (axisV );

	pts[1]*=size;
	pts[2]*=size;
	pts[1]+=orig;
	pts[2]+=orig;


	PSM_LVERTEX svVts[4];
	svVts[0].x = pts[0].x;
    svVts[0].y = pts[0].y;
    svVts[0].z = pts[0].z;
	svVts[1].x = pts[1].x;
    svVts[1].y = pts[1].y;
    svVts[1].z = pts[1].z;
	svVts[2].x = pts[2].x;
    svVts[2].y = pts[2].y;
    svVts[2].z = pts[2].z;
	svVts[3].x = pts[0].x;
    svVts[3].y = pts[0].y;
    svVts[3].z = pts[0].z;



	if (!bSelected)
        svVts[0].diffuse = svVts[1].diffuse = svVts[2].diffuse = svVts[3].diffuse = svVts[4].diffuse = 0x80FFFF00;
	else
        svVts[0].diffuse = svVts[1].diffuse = svVts[2].diffuse = svVts[3].diffuse = svVts[4].diffuse = 0xA0FFFFFF;
		
    GDD->GetD3D9Device()->SetVertexDeclaration(GGizmoVertDecl);
    GDD->GetD3D9Device()->DrawPrimitiveUP(D3DPT_TRIANGLELIST , 1, svVts, sizeof(PSM_LVERTEX));

	if (!bSelected)
        svVts[0].diffuse = svVts[1].diffuse = svVts[2].diffuse = svVts[3].diffuse = svVts[4].diffuse = 0xFFFFFF30;
	else
        svVts[0].diffuse = svVts[1].diffuse = svVts[2].diffuse = svVts[3].diffuse = svVts[4].diffuse = 0xA0FFFF90;

    GDD->GetD3D9Device()->SetVertexDeclaration(GGizmoVertDecl);
    GDD->GetD3D9Device()->DrawPrimitiveUP(D3DPT_LINESTRIP , 3, svVts, sizeof(PSM_LVERTEX));

	GDD->GetD3D9Device()->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
    GDD->GetD3D9Device()->SetRenderState(D3DRS_CULLMODE , D3DCULL_CCW );
    GDD->GetD3D9Device()->SetRenderState(D3DRS_ZENABLE , D3DZB_TRUE);
#endif
}