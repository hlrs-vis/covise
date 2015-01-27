/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(AS_OPENGL_H__)
#define AS_OPENGL_H__

#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <GL/gl.h>
#include <GL/glu.h>
#include <gl/glaux.h>

extern GLuint glList;

class Window // output window related routines
{
private:
public:
    static int width, height; // window size

    static void reshapeCallback(int, int);
    static void displayCallback(void);
};

class as_OpenGL
{
public:
    as_OpenGL(HWND hWnd);
    virtual ~as_OpenGL();

    void mouseMove(float x, float y);

    HWND hWnd;
    HGLRC hRC;
    HDC hDC;
    HANDLE hThread;
};

DWORD WINAPI InitGL(HWND hDlg);
void RenderScene(void);

extern as_OpenGL *OpenGL;
#endif
