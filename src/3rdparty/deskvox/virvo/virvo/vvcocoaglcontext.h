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

#ifndef VV_COCOAGLCONTEXT_H
#define VV_COCOAGLCONTEXT_H

#ifdef __APPLE__

#ifdef __OBJC__
class vvContextOptions;
#else
class vvContextOptions;
class NSAutoreleasePool;
class NSOpenGLContext;
class NSOpenGLPixelFormat;
class NSView;
class NSWindow;
#endif

class vvCocoaGLContext
{
public:
  vvCocoaGLContext(const vvContextOptions& options);
  ~vvCocoaGLContext();

  bool makeCurrent() const;
  void swapBuffers() const;
  void resize(int w, int h);
private:
  const vvContextOptions& _options;

  NSAutoreleasePool* _autoreleasePool;
  NSOpenGLContext* _context;
  NSOpenGLPixelFormat* _pixelFormat;
  NSWindow* _win;
  NSView* _glView;

  void init();
  void destroy();
  void createGLContext();
};

#endif

#endif

