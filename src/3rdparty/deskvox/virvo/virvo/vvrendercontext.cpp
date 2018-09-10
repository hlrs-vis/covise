// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#include "vvplatform.h"

#ifdef _WIN32
#include <GL/glew.h>
#include <GL/wglew.h>
#endif

#ifdef HAVE_OPENGL
#include "vvopengl.h"
#endif

#include "gl/util.h"
#include "math/math.h"
#include "vvcocoaglcontext.h"
#include "vvdebugmsg.h"
#include "vvrendercontext.h"

#include <cassert>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

namespace gl = virvo::gl;


#if defined(HAVE_X11) && defined(USE_X11)
#include <GL/glx.h>
#include <X11/Xlib.h>
#endif

#ifdef _WIN32

namespace
{

  class Attribs
  {
    std::vector<int> attribs;

  public:
    void add(int key, int value)
    {
      attribs.push_back(key);
      attribs.push_back(value);
    }

    int const* get() const
    {
      // check if the array is "null-terminated"
      assert( attribs.size() >= 2 );
      assert( attribs[attribs.size() - 2] == 0 && attribs[attribs.size() - 1] == 0 );

      return &attribs[0];
    }
  };

  // Creates a dummy window and a default OpenGL rendering context.
  // This is necessary since to use wglChoosePixelFormatARB we must have a valid context
  // which can only be created if one sets the pixel format for the DC. After a pixel format
  // is returned from wglChoosePixelFormatARB we would have to set the pixel format a
  // second time, which is not allowed.
  class DefaultContext
  {
    HWND hwnd;
    HDC hdc;
    HGLRC hglrc;

  public:
    DefaultContext();
    ~DefaultContext();
  };

  DefaultContext::DefaultContext()
    : hwnd(0)
    , hdc(0)
    , hglrc(0)
  {
    HINSTANCE hInstance = GetModuleHandle(NULL);

    WNDCLASSEX wc;

    memset(&wc, 0, sizeof(wc));

    wc.cbSize           = sizeof(wc);
    wc.hInstance        = hInstance;
    wc.lpfnWndProc      = DefWindowProc;
    wc.lpszClassName    = TEXT("VVDefaultContext");

    if (!RegisterClassEx(&wc))
    {
    }

    // Create the dummy window
    hwnd = CreateWindowEx(
      0,
      wc.lpszClassName,
      TEXT(""),
      WS_OVERLAPPEDWINDOW,
      0, 0, 0, 0,
      NULL,
      NULL,
      hInstance,
      NULL);

    if (hwnd == NULL)
      throw std::runtime_error("not supported");

    // Get the window context
    hdc = GetDC(hwnd);
    if (hdc == NULL)
      throw std::runtime_error("not supported");

    // Create a minimal GL pixel format
    PIXELFORMATDESCRIPTOR pfd = {
      sizeof(pfd),
      1,
      PFD_DRAW_TO_WINDOW |
      PFD_SUPPORT_OPENGL |
      PFD_DEPTH_DONTCARE |
      PFD_DOUBLEBUFFER_DONTCARE |
      PFD_STEREO_DONTCARE,
      PFD_TYPE_RGBA,
      24,             // color bits
      0,0,0,0,0,0,
      0,              // alpha bits
      0,0,0,0,0,0,
      0,              // depth bits
      0,              // stencil bits
      0,0,0,0,0,0,
    };

    // Choose a pixel format
    int pixelformat = ChoosePixelFormat(hdc, &pfd);
    if (pixelformat < 0)
      throw std::runtime_error("not supported");

    // Set the pixel format
    if (!SetPixelFormat(hdc, pixelformat, NULL/*&pfd*/))
      throw std::runtime_error("not supported");

    // Create the context
    hglrc = wglCreateContext(hdc);
    if (hglrc == NULL)
      throw std::runtime_error("not supported");

    // and make it current
    if (!wglMakeCurrent(hdc, hglrc))
      throw std::runtime_error("not supported");
  }

  DefaultContext::~DefaultContext()
  {
    // Delete the rendering context
    wglMakeCurrent(NULL, NULL);
    wglDeleteContext(hglrc);

    // Release the window DC
    ReleaseDC(hwnd, hdc);

    // Destroy the window
    DestroyWindow(hwnd);
  }

  std::vector<int> GetPixelFormats(vvContextOptions const& co, HDC hdc)
  {
    if (!wglChoosePixelFormatARB)
      return std::vector<int>();

    Attribs attribs;

    // Since we are trying to create a pbuffer, the pixel format we
    // request (and subsequently use) must be "p-buffer capable".
    attribs.add(WGL_DRAW_TO_PBUFFER_ARB, 1);

    // We require a minimum of 24-bit depth.
    attribs.add(WGL_DEPTH_BITS_ARB, 24);

    // We require a minimum of 8-bits for each R, G, B, and A.
    attribs.add(WGL_RED_BITS_ARB, 8);
    attribs.add(WGL_GREEN_BITS_ARB, 8);
    attribs.add(WGL_BLUE_BITS_ARB, 8);
    attribs.add(WGL_ALPHA_BITS_ARB, 8);

    if (co.doubleBuffering)
      attribs.add(WGL_DOUBLE_BUFFER_ARB, 1);

    attribs.add(0,0);

    // Now obtain a list of pixel formats that meet these minimum requirements.
    std::vector<int> buffer(512);

    unsigned count = 0;

    if (!wglChoosePixelFormatARB(hdc, attribs.get(), 0, buffer.size(), &buffer[0], &count))
      return std::vector<int>();

    return std::vector<int>(buffer.begin(), buffer.begin() + count);
  }

} // namespace

#endif

struct ContextArchData
{
#ifdef USE_COCOA
  vvCocoaGLContext* cocoaContext;
#endif

#if defined(HAVE_X11) && defined(USE_X11)
  GLXContext glxContext;
  Display* display;
  Drawable drawable;
  std::vector<int> attributes;
  GLXFBConfig* fbConfigs;
#endif

#ifdef _WIN32
  ContextArchData()
    : hbuffer(NULL)
    , hdc(NULL)
    , hglrc(NULL)
    , format(-1)
  {
  }

  HPBUFFERARB hbuffer;
  HDC hdc;
  HGLRC hglrc;
  int format;
#endif
};

vvRenderContext::vvRenderContext(const vvContextOptions& co)
  : _options(co)
{
  vvDebugMsg::msg(1, "vvRenderContext::vvRenderContext()");

  _archData = new ContextArchData;
  _initialized = false;
  init();
}

vvRenderContext::~vvRenderContext()
{
  vvDebugMsg::msg(1, "vvRenderContext::~vvRenderContext");

  if (_initialized)
  {
#ifdef USE_COCOA
    delete _archData->cocoaContext;
#endif

#if defined(HAVE_X11) && defined(USE_X11)
    glXDestroyContext(_archData->display, _archData->glxContext);
    switch (_options.type)
    {
    case vvContextOptions::VV_PBUFFER:
      glXDestroyPbuffer(_archData->display, _archData->drawable);
      break;
    case vvContextOptions::VV_WINDOW:
      // fall through
    default:
      XDestroyWindow(_archData->display, _archData->drawable);
      break;
    }
#endif

#ifdef _WIN32
    wglMakeCurrent(NULL, NULL);

    if (_archData->hglrc)
      wglDeleteContext(_archData->hglrc);

    if (wglReleasePbufferDCARB)
      wglReleasePbufferDCARB(_archData->hbuffer, _archData->hdc);

    if (wglDestroyPbufferARB)
      wglDestroyPbufferARB(_archData->hbuffer);
#endif
  }
  delete _archData;
}

bool vvRenderContext::makeCurrent() const
{
  vvDebugMsg::msg(3, "vvRenderContext::makeCurrent()");

  if (_initialized)
  {
#ifdef USE_COCOA
    return _archData->cocoaContext->makeCurrent();
#endif

#if defined(HAVE_X11) && defined(USE_X11)
    return glXMakeCurrent(_archData->display, _archData->drawable, _archData->glxContext);
#endif

#ifdef _WIN32
    return 0 != wglMakeCurrent(_archData->hdc, _archData->hglrc);
#endif
  }
  return false;
}

void vvRenderContext::swapBuffers() const
{
  vvDebugMsg::msg(3, "vvRenderContext::swapBuffers()");

  if (_initialized)
  {
#ifdef USE_COCOA
    _archData->cocoaContext->swapBuffers();
#endif

#if defined(HAVE_X11) && defined(USE_X11)
    glXSwapBuffers(_archData->display, _archData->drawable);
#endif

#ifdef _WIN32
    SwapBuffers(_archData->hdc);
#endif
  }
}

void vvRenderContext::resize(int w, int h)
{
  vvDebugMsg::msg(3, "vvRenderContext::resize()");

  if ((_options.width != w) || (_options.height != h))
  {
    _options.width = w;
    _options.height = h;
    if (_initialized)
    {
#ifdef USE_COCOA
      _archData->cocoaContext->resize(w, h);
#endif

#if defined(HAVE_X11) && defined(USE_X11)
      switch (_options.type)
      {
      case vvContextOptions::VV_PBUFFER:
      {
        glXDestroyPbuffer(_archData->display, _archData->drawable);
        initPbuffer();
        makeCurrent();
        break;
      }
      case vvContextOptions::VV_WINDOW:
        // fall through
      default:
        XResizeWindow(_archData->display, _archData->drawable,
                      static_cast<uint>(w),
                      static_cast<uint>(h));
        XSync(_archData->display, False);
        break;
      }
#endif

#ifdef HAVE_OPENGL
      virvo::recti vp = gl::getViewport();
      glViewport(vp[0], vp[1], (GLsizei)w, (GLsizei)h);
#endif

#ifdef _WIN32
      vvDebugMsg::msg(0, "vvRenderContext::resize() not implemented yet");
#endif
    }
  }
}

bool vvRenderContext::matchesCurrent(const vvContextOptions& co)
{
  vvDebugMsg::msg(3, "vvRenderContext::matchesCurrent()");

#ifdef USE_COCOA
  (void)co;
  vvDebugMsg::msg(0, "vvRenderContext::matchesCurrent() not implemented yet");
  return false;
#endif

#if defined(HAVE_X11) && defined(USE_X11)
  Display* dpy = glXGetCurrentDisplay();

  if (dpy == NULL)
  {
    return false;
  }

  Display* other = XOpenDisplay(co.displayName.c_str());

  if (other == NULL)
  {
    return false;
  }

  if (DefaultScreen(dpy) != DefaultScreen(other))
  {
    XCloseDisplay(other);
    return false;
  }

  if (RootWindow(dpy, XDefaultScreen(dpy)) != RootWindow(other, XDefaultScreen(other)))
  {
    XCloseDisplay(other);
    return false;
  }

  XCloseDisplay(other);

  uint w;
  glXQueryDrawable(dpy, glXGetCurrentDrawable(), GLX_WIDTH, &w);

  uint h;
  glXQueryDrawable(dpy, glXGetCurrentDrawable(), GLX_HEIGHT, &h);

  if (w < (uint)co.width || h < (uint)co.height)
  {
    return false;
  }

  return true;
#endif

#ifdef _WIN32
  (void)co;
  vvDebugMsg::msg(0, "vvRenderContext::matchesCurrent() not implemented yet");
  return false;
#endif
}

void vvRenderContext::init()
{
  vvDebugMsg::msg(3, "vvRenderContext::init()");

#ifdef USE_COCOA
  _archData->cocoaContext = new vvCocoaGLContext(_options);
  _initialized = true;
#endif

#if defined(HAVE_X11) && defined(USE_X11)
  _archData->display = XOpenDisplay(_options.displayName.c_str());

  _archData->attributes.push_back(GLX_RGBA);
  _archData->attributes.push_back(GLX_RED_SIZE);
  _archData->attributes.push_back(8);
  _archData->attributes.push_back(GLX_GREEN_SIZE);
  _archData->attributes.push_back(8);
  _archData->attributes.push_back(GLX_BLUE_SIZE);
  _archData->attributes.push_back(8);
  _archData->attributes.push_back(GLX_ALPHA_SIZE);
  _archData->attributes.push_back(8);
  _archData->attributes.push_back(GLX_DEPTH_SIZE);
  _archData->attributes.push_back(24);
  _archData->attributes.push_back(None);

  if(_archData->display != NULL)
  {
    switch(_options.type)
    {
    case vvContextOptions::VV_PBUFFER:
      if (initPbuffer())
      {
         _archData->glxContext = glXCreateNewContext(_archData->display, _archData->fbConfigs[0], GLX_RGBA_TYPE, 0, True);
        _initialized = true;
        return;
      }
      else
      {
        _options.type = vvContextOptions::VV_WINDOW;
      }
      // no pbuffer created - fall through
      // fall through
    case vvContextOptions::VV_WINDOW:
      // fall through
    default:
      {
        const Drawable parent = RootWindow(_archData->display, DefaultScreen(_archData->display));

        XVisualInfo* vi = glXChooseVisual(_archData->display,
                                          DefaultScreen(_archData->display),
                                          &(_archData->attributes)[0]);

        XSetWindowAttributes wa = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        wa.colormap = XCreateColormap(_archData->display, parent, vi->visual, AllocNone);
        wa.background_pixmap = None;
        wa.border_pixel = 0;

        if (vvDebugMsg::getDebugLevel() == 0)
        {
          wa.override_redirect = true;
        }
        else
        {
          wa.override_redirect = false;
        }

        _archData->glxContext = glXCreateContext(_archData->display, vi, NULL, True);

        if (_archData->glxContext != 0)
        {

          int windowWidth = _options.width;
          int windowHeight = _options.height;
          if (vvDebugMsg::getDebugLevel() > 0)
          {
            windowWidth = 512;
            windowHeight = 512;
          }

          _archData->drawable = XCreateWindow(_archData->display, parent, 0, 0, windowWidth, windowHeight, 0,
                                              vi->depth, InputOutput, vi->visual,
                                              CWBackPixmap|CWBorderPixel|CWColormap|CWOverrideRedirect, &wa);
          XMapWindow(_archData->display, _archData->drawable);
          XFlush(_archData->display);
          _initialized = true;
        }
        else
        {
          vvDebugMsg::msg( 0, "Couldn't create OpenGL context");
          _initialized = false;
          return;
        }
        _initialized = true;
        delete vi;
        break;
      }
    }
  }
  else
  {
    _initialized = false;
    std::ostringstream errmsg;
    errmsg << "vvRenderContext::init() error: Could not open display " << _options.displayName;
    vvDebugMsg::msg(0, errmsg.str().c_str());
  }
#endif

#ifdef _WIN32
  _initialized = false;

  // Create a default OpenGL rendering context
  DefaultContext context;

  // Initialize OpenGL extensions
  glewInit();

  // Get a valid device context
  HDC hdc = wglGetCurrentDC();

  // Now obtain a list of pixel formats that meet these minimum requirements...
  std::vector<int> pformats = GetPixelFormats(_options, hdc);

  //
  // After determining a list of pixel formats, the next step is to create
  // a pbuffer of the chosen format
  //

  if (!wglCreatePbufferARB)
    throw std::runtime_error("not supported");

  _archData->hbuffer = NULL;
  _archData->format = -1;

  for (size_t i = 0; i < pformats.size(); ++i)
  {
    _archData->format = pformats[i];
    _archData->hbuffer = wglCreatePbufferARB(hdc,
                                             _archData->format,
                                             _options.width,
                                             _options.height,
                                             NULL);

    if (_archData->hbuffer != NULL)
      break;
  }

  if (_archData->hbuffer == NULL)
    throw std::runtime_error("not supported");

  //
  // The next step is to create a device context for the newly created pbuffer
  //

  if (!wglGetPbufferDCARB)
    throw std::runtime_error("not supported");

  _archData->hdc = wglGetPbufferDCARB(_archData->hbuffer);

  if (!_archData->hdc)
    throw std::runtime_error("not supported");

  //
  // The final step of pbuffer creation is to create an OpenGL rendering context and
  // associate it with the handle for the pbuffer’s device context
  //

  _archData->hglrc = wglCreateContext(_archData->hdc);

  if (!_archData->hglrc)
    throw std::runtime_error("not supported");

  _initialized = true;
#endif
}

bool vvRenderContext::initPbuffer()
{
  vvDebugMsg::msg(3, "vvRenderContext::initPbuffer()");

#if defined(HAVE_X11) && defined(USE_X11)
  int nelements;
  _archData->fbConfigs = glXChooseFBConfig(_archData->display, DefaultScreen(_archData->display),
                                           &(_archData->attributes)[1], &nelements); // first entry (GLX_RGBA) in attributes list confuses pbuffers
  if ((_archData->fbConfigs != NULL) && (nelements > 0))
  {
    // TODO: find the nicest fbconfig.
    int pbAttrList[] = { GLX_PBUFFER_WIDTH, _options.width, GLX_PBUFFER_HEIGHT, _options.height, None };
    _archData->drawable = glXCreatePbuffer(_archData->display, _archData->fbConfigs[0], pbAttrList);
    if (!_archData->drawable)
    {
      std::cerr << "No pbuffer created" << std::endl;
      return false;
    }
    return true;
  }
#endif
  return false;
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
