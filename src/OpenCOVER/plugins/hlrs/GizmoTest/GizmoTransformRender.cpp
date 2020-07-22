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
#include "GizmoTransformRender.h"
#ifdef MAC_OS
#import <OpenGL/OpenGL.h>
#else
#include <GL/gl.h>
#endif

void CGizmoTransformRender::DrawCircle(const tvector3 &orig,float r,float g,float b,const tvector3 &vtx,const tvector3 &vty)
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
	glColor4f(r,g,b,1);

	glBegin(GL_LINE_LOOP);
	for (int i = 0; i < 50 ; i++)
	{
		tvector3 vt;
		vt = vtx * cos((2*ZPI/50)*i);
		vt += vty * sin((2*ZPI/50)*i);
		vt += orig;
		glVertex3f(vt.x,vt.y,vt.z);
	}
	glEnd();
}


void CGizmoTransformRender::DrawCircleHalf(const tvector3 &orig,float r,float g,float b,const tvector3 &vtx,const tvector3 &vty,tplane &camPlan)
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
	glColor4f(r,g,b,1);

	glBegin(GL_LINE_STRIP);
	for (int i = 0; i < 30 ; i++)
	{
		tvector3 vt;
		vt = vtx * cos((ZPI/30)*i);
		vt += vty * sin((ZPI/30)*i);
		vt +=orig;
		if (camPlan.DotNormal(vt))
			glVertex3f(vt.x,vt.y,vt.z);
	}
	glEnd();
}

void CGizmoTransformRender::DrawAxis(const tvector3 &orig, const tvector3 &axis, const tvector3 &vtx,const tvector3 &vty, float fct,float fct2,const tvector4 &col)
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
	glColor4fv(&col.x);
	glBegin(GL_LINES);
	glVertex3fv(&orig.x);
	glVertex3f(orig.x+axis.x,orig.y+axis.y,orig.z+axis.z);
	glEnd();

	glBegin(GL_TRIANGLE_FAN);
	for (int i=0;i<=30;i++)
	{
		tvector3 pt;
		pt = vtx * cos(((2*ZPI)/30.0f)*i)*fct;
		pt+= vty * sin(((2*ZPI)/30.0f)*i)*fct;
		pt+=axis*fct2;
		pt+=orig;
		glVertex3fv(&pt.x);
		pt = vtx * cos(((2*ZPI)/30.0f)*(i+1))*fct;
		pt+= vty * sin(((2*ZPI)/30.0f)*(i+1))*fct;
		pt+=axis*fct2;
		pt+=orig;
		glVertex3fv(&pt.x);
		glVertex3f(orig.x+axis.x,orig.y+axis.y,orig.z+axis.z);

	}
	glEnd();

}

void CGizmoTransformRender::DrawCamem(const tvector3& orig,const tvector3& vtx,const tvector3& vty,float ng)
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
	int i = 0 ;
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glDisable(GL_CULL_FACE);


	glColor4f(1,1,0,0.5f);
	glBegin(GL_TRIANGLE_FAN);
	glVertex3fv(&orig.x);
	for (i = 0 ; i <= 50 ; i++)
	{
		tvector3 vt;
		vt = vtx * cos(((ng)/50)*i);
		vt += vty * sin(((ng)/50)*i);
		vt+=orig;
		glVertex3f(vt.x,vt.y,vt.z);
	}
	glEnd();

	glDisable(GL_BLEND);


	glColor4f(1,1,0.2f,1);
	tvector3 vt;
	glBegin(GL_LINE_LOOP);

	glVertex3fv(&orig.x);
	for ( i = 0 ; i <= 50 ; i++)
	{
		tvector3 vt;
		vt = vtx * cos(((ng)/50)*i);
		vt += vty * sin(((ng)/50)*i);
		vt+=orig;
		glVertex3f(vt.x,vt.y,vt.z);
	}

	glEnd();
}

void CGizmoTransformRender::DrawQuad(const tvector3& orig, float size, bool bSelected, const tvector3& axisU, const tvector3 &axisV)
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glDisable(GL_CULL_FACE);

	tvector3 pts[4];
	pts[0] = orig;
	pts[1] = orig + (axisU * size);
	pts[2] = orig + (axisU + axisV)*size;
	pts[3] = orig + (axisV * size);

	if (!bSelected)
		glColor4f(1,1,0,0.5f);
	else
		glColor4f(1,1,1,0.6f);

	glBegin(GL_QUADS);
	glVertex3fv(&pts[0].x);
	glVertex3fv(&pts[1].x);
	glVertex3fv(&pts[2].x);
	glVertex3fv(&pts[3].x);
	glEnd();

	if (!bSelected)
		glColor4f(1,1,0.2f,1);
	else
		glColor4f(1,1,1,0.6f);

	glBegin(GL_LINE_STRIP);
	glVertex3fv(&pts[0].x);
	glVertex3fv(&pts[1].x);
	glVertex3fv(&pts[2].x);
	glVertex3fv(&pts[3].x);
	glEnd();

	glDisable(GL_BLEND);
}


void CGizmoTransformRender::DrawTri(const tvector3& orig, float size, bool bSelected, const tvector3& axisU, const tvector3& axisV)
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glDisable(GL_CULL_FACE);

	tvector3 pts[3];
	pts[0] = orig;

	pts[1] = (axisU );
	pts[2] = (axisV );

	pts[1]*=size;
	pts[2]*=size;
	pts[1]+=orig;
	pts[2]+=orig;

	if (!bSelected)
		glColor4f(1,1,0,0.5f);
	else
		glColor4f(1,1,1,0.6f);

	glBegin(GL_TRIANGLES);
	glVertex3fv(&pts[0].x);
	glVertex3fv(&pts[1].x);
	glVertex3fv(&pts[2].x);
	glVertex3fv(&pts[3].x);
	glEnd();

	if (!bSelected)
		glColor4f(1,1,0.2f,1);
	else
		glColor4f(1,1,1,0.6f);

	glBegin(GL_LINE_STRIP);
	glVertex3fv(&pts[0].x);
	glVertex3fv(&pts[1].x);
	glVertex3fv(&pts[2].x);
	glEnd();

	glDisable(GL_BLEND);
}
