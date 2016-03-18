/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// This is the main DLL file.


#include "ManagedOpenCOVER.h"
#include "cover/coVRPluginList.h"
#include "cover/coVRConfig.h"
#include <config/CoviseConfig.h>
#include <cover/coCommandLine.h>
#include <config/coConfigConstants.h>

#include <osgViewer/api/Win32/GraphicsWindowWin32>

using namespace ManagedOpenCOVER;

using namespace System;
using namespace System::Threading;

#include <stdio.h>
void coOpenCOVERWindow::ErrorExit(LPTSTR lpszFunction)
{
    // Retrieve the system error message for the last-error code
    LPVOID lpMsgBuf;
    LPVOID lpDisplayBuf;
    DWORD dw = GetLastError();

    FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        dw,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR)&lpMsgBuf,
        0, NULL);

    // Display the error message and exit the process

    lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT,
                                      (lstrlen((LPCTSTR)lpMsgBuf) + lstrlen((LPCTSTR)lpszFunction) + 40) * sizeof(TCHAR));

    StringCchPrintf((LPTSTR)lpDisplayBuf,
                    LocalSize(lpDisplayBuf) / sizeof(TCHAR),
                    TEXT("%s failed with error %d: %s"),
                    lpszFunction, dw, lpMsgBuf);
    ::MessageBox(NULL, (LPCTSTR)lpDisplayBuf, TEXT("Error"), MB_OK);

    LocalFree(lpMsgBuf);
    LocalFree(lpDisplayBuf);
    ExitProcess(dw);
}

coOpenCOVERWindow::coOpenCOVERWindow()
    : m_hWnd(NULL)
{
	
    covise::coConfigConstants::setRank(0);
    openCOVER = NULL;
}

void coOpenCOVERWindow::testit(float)
{
}
#ifdef OLDSURFACE
void coOpenCOVERWindow::addedContact(Contact ^ c)
{
    long Identity = 0;
    if (c->Tag.Type == TagType::Identity)
        Identity = c->Tag.Identity.Value;
    if (c->Tag.Type == TagType::Byte)
        Identity = c->Tag.Byte.Value;
    SurfaceContact cont(c->CenterX * m_dScaleX, c->CenterY * m_dScaleY, c->Orientation, c->PhysicalArea, Identity, c->IsFingerRecognized, c->IsTagRecognized, c->Id);
    surfacePlugin->addedContact(cont);
}
void coOpenCOVERWindow::removedContact(Contact ^ c)
{
    long Identity = 0;
    if (c->Tag.Type == TagType::Identity)
        Identity = c->Tag.Identity.Value;
    if (c->Tag.Type == TagType::Byte)
        Identity = c->Tag.Byte.Value;
    SurfaceContact cont(c->CenterX * m_dScaleX, c->CenterY * m_dScaleY, c->Orientation, c->PhysicalArea, Identity, c->IsFingerRecognized, c->IsTagRecognized, c->Id);
    surfacePlugin->removedContact(cont);
}
void coOpenCOVERWindow::changedContact(Contact ^ c)
{
    long Identity = 0;
    if (c->Tag.Type == TagType::Identity)
        Identity = c->Tag.Identity.Value;
    if (c->Tag.Type == TagType::Byte)
        Identity = c->Tag.Byte.Value;
    SurfaceContact cont(c->CenterX * m_dScaleX, c->CenterY * m_dScaleY, c->Orientation, c->PhysicalArea, Identity, c->IsFingerRecognized, c->IsTagRecognized, c->Id);
    surfacePlugin->changedContact(cont);
}
#else
void coOpenCOVERWindow::addedContact(TouchPoint^ c)
{
	long Identity = 0;
	if(c->IsTagRecognized)
		Identity = c->Tag.Value;
	SurfaceContact cont(c->CenterX*m_dScaleX,c->CenterY*m_dScaleY,c->Orientation,c->PhysicalArea,Identity,c->IsFingerRecognized,c->IsTagRecognized,c->Id);
	surfacePlugin->addedContact(cont);
}
void coOpenCOVERWindow::removedContact(TouchPoint^ c)
{
	long Identity = 0;
	if(c->IsTagRecognized)
		Identity = c->Tag.Value;
	SurfaceContact cont(c->CenterX*m_dScaleX,c->CenterY*m_dScaleY,c->Orientation,c->PhysicalArea,Identity,c->IsFingerRecognized,c->IsTagRecognized,c->Id);
	surfacePlugin->removedContact(cont);
}
void coOpenCOVERWindow::changedContact(TouchPoint^ c)
{
	long Identity = 0;
	if(c->IsTagRecognized)
		Identity = c->Tag.Value;
	SurfaceContact cont(c->CenterX*m_dScaleX,c->CenterY*m_dScaleY,c->Orientation,c->PhysicalArea,Identity,c->IsFingerRecognized,c->IsTagRecognized,c->Id);
	surfacePlugin->changedContact(cont);
}
#endif

	#ifdef OLDSURFACE
void coOpenCOVERWindow::manipulation(Affine2DOperationDeltaEventArgs ^ e)
{
    MotionEvent me(e->AngularVelocity, e->CumulativeExpansion, e->CumulativeRotation, e->CumulativeScale, e->CumulativeTranslationX, e->CumulativeTranslationY, e->DeltaX, e->DeltaY, e->ExpansionDelta, e->ExpansionVelocity, e->ManipulationOriginX, e->ManipulationOriginY, e->RotationDelta, e->ScaleDelta, e->VelocityX, e->VelocityY);
    surfacePlugin->manipulation(me);
}
	#endif
//
// This is the key method to override
//
void coOpenCOVERWindow::init(IntPtr window, array<System::String ^> ^ args)
{
#ifdef _WIN32
    // disable "debug dialog": it prevents the application from exiting,
    // but still all sockets remain open
    DWORD dwMode = SetErrorMode(SEM_NOGPFAULTERRORBOX);
    SetErrorMode(dwMode | SEM_NOGPFAULTERRORBOX);
#endif

    //now create the actual OpenCOVER

    int argc = args->GetLength(0) + 1;
    char **argv;
    argv = new char *[argc];
    argv[0] = new char[100];
	#ifdef OLDSURFACE
    strcpy(argv[0], "SurfaceCOVER");
	#else
    strcpy(argv[0], "Sur40COVER");
	#endif
    for (int i = 1; i < argc; i++)
    {
        char *p = (char *)System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(args[i - 1]).ToPointer();
        argv[i] = new char[strlen(p) + 1];
        strcpy(argv[i], p);
    }

    coCommandLine(argc, argv);

#ifdef _WIN32
    if (coCoviseConfig::isOn("COVER.Console", true))
    {
        std::string filebase = coCoviseConfig::getEntry("file", "COVER.Console");
        if (!filebase.empty())
        {
            char *filename = new char[strlen(filebase.c_str()) + 100];
            sprintf(filename, "%s%d.err.txt", filebase.c_str(), 0);
            freopen(filename, "w", stderr);
            sprintf(filename, "%s%d.out.txt", filebase.c_str(), 0);
            freopen("conout$", "w", stdout);
            delete[] filename;
        }
        else
        {

            AllocConsole();

            freopen("conin$", "r", stdin);
            freopen("conout$", "w", stdout);
            freopen("conout$", "w", stderr);
        }
    }
#endif //_WIN32


    m_hWnd = (HWND)window.ToPointer();

    HDC m_hDC;
    m_hDC = GetDC(m_hWnd);
    m_dScaleX = GetDeviceCaps(m_hDC, LOGPIXELSX) / 96.0;
    m_dScaleY = GetDeviceCaps(m_hDC, LOGPIXELSY) / 96.0;
    openCOVER = new OpenCOVER(m_hWnd);
    openCOVER->init();
    openCOVER->setIgnoreMouseEvents(true);

    surfacePlugin = (SurfacePlugin *)coVRPluginList::instance()->addPlugin("Surface");
}
void coOpenCOVERWindow::shutdown()
{
    openCOVER->setExitFlag(1);
    delete openCOVER;
}

void coOpenCOVERWindow::frame()
{
    //now create the actual OpenCOVER

    if (openCOVER->getExitFlag() != 0)
    {
        openCOVER->doneRendering();
        ExitProcess(0);
    }
    openCOVER->frame();
}
