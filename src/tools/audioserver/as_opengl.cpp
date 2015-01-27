/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <windows.h>
#include <stdio.h>

#include "as_opengl.h"
#include "resource.h"
#include "common.h"
#include "as_sound.h"
#include "as_control.h"

as_OpenGL *OpenGL = NULL;

HANDLE hOpenGLThread;
unsigned long lpOpenGLThreadId;
void renderThread(void);
bool render = true;

int Window::height;
int Window::width;

GLuint glList;

// ============================================================================
as_OpenGL::as_OpenGL(HWND hWndParent)
{
    RECT rct;
    GetClientRect(hWnd3DView, &rct);
    Window::width = rct.right - rct.left;
    Window::height = rct.bottom - rct.top;
    hOpenGLThread = CreateThread(
        NULL,
        NULL,
        (LPTHREAD_START_ROUTINE)InitGL,
        hWndParent,
        0,
        &lpOpenGLThreadId);
}

as_OpenGL::~as_OpenGL()
{
    render = false;
    Sleep(1000);
}

// Here, we do the regular initialization of OpenGL
DWORD WINAPI InitGL(HWND hDlg)
{
    HDC hDC1;
    HGLRC hglrc1;

    PIXELFORMATDESCRIPTOR pfd = {
        sizeof(PIXELFORMATDESCRIPTOR), // size of this pfd
        1, // version number
        PFD_DRAW_TO_WINDOW | // support window
        PFD_SUPPORT_OPENGL | // support OpenGL
        PFD_DOUBLEBUFFER, // double buffered
        PFD_TYPE_RGBA, // RGBA type
        16, // 24-bit color depth
        0,
        0, 0, 0, 0, 0, // color bits ignored
        0, // no alpha buffer
        0, // shift bit ignored
        0, // no accumulation buffer
        0, 0, 0, 0, // accum bits ignored
        8, // 32-bit z-buffer
        0, // no stencil buffer
        0, // no auxiliary buffer
        PFD_MAIN_PLANE, // main layer
        0, // reserved
        0, 0, 0 // layer masks ignored
    };

    int iPixelFormat;
    float specular[] = { 1.0, 1.0, 1.0, 1.0 };
    float shininess[] = { 100.0 };
    float position[] = { -10.0, 10.0, 10.0, 0.0 };

    if (NULL == hDlg)
    {
        char msg[128];
        sprintf(msg, "Could not get Dlg item, error %d", GetLastError());
        MessageBox(NULL, msg, "", MB_OK);
        return FALSE;
    }

    //  get the HDC
    hDC1 = GetDC(hDlg);
    if (NULL == hDC1)
    {
        char msg[128];
        sprintf(msg, "Could not get drawing context, error %d", GetLastError());
        MessageBox(NULL, msg, "", MB_OK);
        return FALSE;
    }

    // get the device context's best, available pixel format match
    iPixelFormat = ChoosePixelFormat(hDC1, &pfd);
    if (0 == iPixelFormat)
    {
        MessageBox(NULL, "ChoosePixelFormat Failed", NULL, MB_OK);
        return FALSE;
    }

    // make that match the device context's current pixel format
    if (FALSE == SetPixelFormat(hDC1, iPixelFormat, &pfd))
    {
        MessageBox(NULL, "SetPixelFormat Failed", NULL, MB_OK);
        return FALSE;
    }

    //  get the HDC
    hDC1 = GetDC(hWnd3DView);
    if (NULL == hDC1)
    {
        char msg[128];
        sprintf(msg, "Could not get drawing context, error %d", GetLastError());
        MessageBox(NULL, msg, "", MB_OK);
        return FALSE;
    }

    hglrc1 = wglCreateContext(hDC1);
    if (NULL == hglrc1)
    {
        char msg[128];
        sprintf(msg, "Could not get OpenGL context, error %d", GetLastError());
        MessageBox(NULL, msg, "", MB_OK);
        return FALSE;
    }

    AddLogMsg("Starting OpenGL rendering...");

    // Select rendering context and draw to it
    if (false == wglMakeCurrent(hDC1, hglrc1))
    {
        char msg[128];
        sprintf(msg, "Could not get make context current, error %d", GetLastError());
        AddLogMsg(msg);
        return FALSE;
    }

    // OpenGL Initialization
    glShadeModel(GL_SMOOTH); // Enable Smooth Shading
    glClearDepth(1.0f); // Depth Buffer Setup
    glEnable(GL_DEPTH_TEST); // Enables Depth Testing
    glDepthFunc(GL_LEQUAL); // The Type Of Depth Testing To Do
    // Really Nice Perspective Calculations
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glClear(GL_DEPTH_BUFFER_BIT); // clear depth buffer
    glClearColor(0.0, 0.0, 0.0, 0.0); // set clear color to black
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); // set polygon drawing mode to fill front and back of each polygon

    // Generate material properties:
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shininess);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);

    // Generate light source:
    glLightfv(GL_LIGHT0, GL_POSITION, position);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    Window::reshapeCallback(Window::width, Window::height);
    gluLookAt(0.1f, 0.25f, -2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);

    //render here
    do
    {

        // render
        Window::displayCallback();

        // swap buffers
        if (false == SwapBuffers(hDC1))
        {
            char msg[128];
            sprintf(msg, "Could not swap buffers, error %d", GetLastError());
            AddLogMsg(msg);
            return FALSE;
        }

        // Throttle the loop. If we don't put a sleep here, the thread will use 100% cpu
        Sleep(10);
    } while (true == render);

    AddLogMsg("Stopped OpenGL rendering");

    wglMakeCurrent(NULL, NULL);

    // Cleanup
    ReleaseDC(hWnd3DView, hDC1);

    wglDeleteContext(hglrc1);

    ExitThread(0);
    return FALSE;
}

// Resize And Initialize The GL Window
GLvoid Window::reshapeCallback(GLsizei width, GLsizei height)
{
    if (height == 0) // Prevent A Divide By Zero By
        height = 1; // Making Height Equal One

    // Reset The Current Viewport
    glViewport(0, 0, (GLsizei)(width), (GLsizei)(height));
    glMatrixMode(GL_PROJECTION); // Select The Projection Matrix
    glLoadIdentity(); // Reset The Projection Matrix
    // Calculate The Aspect Ratio Of The Window
    gluPerspective(60.0f, (GLfloat)(width) / (GLfloat)(height),
                   0.0f, 50.0f);
    glMatrixMode(GL_MODELVIEW); // Select The Modelview Matrix
    glLoadIdentity(); // Reset The Modelview Matrix
}

//----------------------------------------------------------------------------
// Callback method called when window readraw is necessary or
// when glutPostRedisplay() was called.
void Window::displayCallback(void)
{
    int i;
    int maxSounds = 32;
    long status;
    D3DVECTOR direction;
    D3DVECTOR position;
    long color;
    float colorR, colorG, colorB;
    as_Sound *pSound;
    GLUquadricObj *qobj[32];

    // clear color and depth buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW); // Select The Modelview Matrix

    glDisable(GL_LIGHTING);

    glBegin(GL_LINES);
    // red line, x-axis
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(-50.0f, 0.0f, 0.0f);
    glVertex3f(50.0f, 0.0f, 0.0f);

    // green line, y-axis
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, -50.0f, 0.0f);
    glVertex3f(0.0f, 50.0f, 0.0f);

    // blue line, z-axis
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, -50.0f);
    glVertex3f(0.0f, 0.0f, 50.0f);
    glEnd();

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    for (i = 0; i < maxSounds; i++)
    {
        pSound = AS_Control->getSoundByHandle(i);
        if (NULL != pSound)
        {

            status = pSound->GetStatus();
            switch (status)
            {
            case STATUS_INITIAL:
                break;
            case STATUS_PLAYING:
                break;
            case STATUS_LOOPING:
                break;
            case STATUS_STOPPED:
                break;
            case STATUS_NOTUSED:
                break;
            default:
                break;
            }

            pSound->GetDirection(&direction);
            pSound->GetPosition(&position);

            color = AS_Control->getHandleColor(i);
            colorR = ((color & 0x00FF0000) >> 16) / 255.0f;
            colorG = ((color & 0x0000ff00) >> 8) / 255.0f;
            colorB = (color & 0x000000ff) / 255.0f;

            glColor3f(colorR, colorG, colorB);

            glPushMatrix();
            glTranslatef(position.x, position.y, position.z);

            qobj[i] = gluNewQuadric();
            gluSphere(
                qobj[i],
                0.1f,
                10,
                10);
            gluDeleteQuadric(qobj[i]);

            glPopMatrix();
        }
        Sleep(1);
    }
    glFlush();
}
