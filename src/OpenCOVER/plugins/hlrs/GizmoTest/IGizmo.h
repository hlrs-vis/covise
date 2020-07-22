///////////////////////////////////////////////////////////////////////////////////////////////////
// LibGizmo
// File Name : IGizmo.h
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

#ifndef IGIZMO_H__
#define IGIZMO_H__


class IGizmo
{
public:
    enum LOCATION
    {
        LOCATE_VIEW,
        LOCATE_WORLD,
        LOCATE_LOCAL,
    };

	enum ROTATE_AXIS
	{
		AXIS_X = 1,
		AXIS_Y = 2,
		AXIS_Z = 4,
		AXIS_TRACKBALL = 8,
		AXIS_SCREEN = 16,
		AXIS_ALL = 31

	};


	virtual void SetEditMatrix(float *pMatrix) = 0;

	virtual void SetCameraMatrix(const float *Model, const float *Proj) = 0;
    virtual void SetScreenDimension( int screenWidth, int screenHeight) = 0;
    virtual void SetDisplayScale( float aScale ) = 0;

    // return true if gizmo transform capture mouse
	virtual bool OnMouseDown(unsigned int x, unsigned int y) = 0;
	virtual void OnMouseMove(unsigned int x, unsigned int y) = 0;
	virtual void OnMouseUp(unsigned int x, unsigned int y) = 0;

    // snaping
    virtual void UseSnap(bool bUseSnap) = 0;
	virtual bool IsUsingSnap() = 0;
    virtual void SetSnap(float snapx, float snapy, float snapz) = 0;
    virtual void SetSnap(const float snap) = 0;


    virtual void SetLocation(LOCATION aLocation)  = 0;
    virtual LOCATION GetLocation() = 0;
	virtual void SetAxisMask(unsigned int mask) = 0;

    // rendering
	virtual void Draw() = 0;
};

// Create new gizmo
IGizmo *CreateMoveGizmo();
IGizmo *CreateRotateGizmo();
IGizmo *CreateScaleGizmo();


#endif
