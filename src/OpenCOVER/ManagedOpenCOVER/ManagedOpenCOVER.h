/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ManagedOpenCOVER.h
#define SURFACE
#include <cover/coVRPlugin.h>
#include "plugins/hlrs/Surface/SurfacePlugin.h"
#pragma once
#include <cover/OpenCOVER.h>
#pragma unmanaged
#include <strsafe.h>
#pragma managed

// Link lib
#pragma comment(lib, "strsafe.lib")

// To use these, we must add some references...
//	o PresentationFramework (for HwndHost)
//		* PresentationCo
//		* WindowsBase
using namespace System;
using namespace System::Windows;
using namespace System::Windows::Interop;
using namespace System::Windows::Input;
using namespace System::Windows::Media;
using namespace System::Runtime::InteropServices;
using namespace Microsoft::Surface::Core;
using namespace Microsoft::Surface::Core::Manipulations;

#define UNREF(x) x;

namespace ManagedOpenCOVER
{

public
ref class coOpenCOVERWindow
{
public:
    coOpenCOVERWindow();

    static void ErrorExit(LPTSTR lpszFunction);

    void frame();
    void testit(float con);
    void shutdown();
    void addedContact(Contact ^ c);
    void removedContact(Contact ^ c);
    void changedContact(Contact ^ c);
    void manipulation(Affine2DOperationDeltaEventArgs ^ e);
    void init(IntPtr hwnd, array<System::String ^> ^ args);

private:
    HWND m_hWnd;
    OpenCOVER *openCOVER;
    SurfacePlugin *surfacePlugin;
    float m_dScaleX;
    float m_dScaleY;

protected:
};
}
