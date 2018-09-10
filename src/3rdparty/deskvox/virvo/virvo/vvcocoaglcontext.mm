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

#ifdef __APPLE__
#pragma clang diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#pragma GCC diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma GCC diagnostic ignored "-Wfour-char-constants"
#pragma GCC diagnostic ignored "-Wzero-length-array"
#pragma GCC diagnostic ignored "-Wc++11-extensions"
#pragma GCC diagnostic ignored "-Wc99-extensions"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wmissing-method-return-type"
#pragma GCC diagnostic ignored "-Wextra-semi"
#import <Cocoa/Cocoa.h>
#pragma clang diagnostic pop
#endif

#import <iostream>

#import "vvcocoaglcontext.h"
#import "vvrendercontext.h"

vvCocoaGLContext::vvCocoaGLContext(const vvContextOptions& options)
  : _options(options)
{
  init();
}

vvCocoaGLContext::~vvCocoaGLContext()
{
  destroy();
}

bool vvCocoaGLContext::makeCurrent() const
{
  if (_context != NULL)
  {
    [_context makeCurrentContext];
    return true;
  }
  else
  {
    return false;
  }
}

void vvCocoaGLContext::swapBuffers() const
{
  [_context flushBuffer];
}

void vvCocoaGLContext::resize(const int w, const int h)
{
  if (_win != NULL)
  {
    NSRect rect = NSMakeRect(0.0f, 0.0f,
                             static_cast<float>(w),
                             static_cast<float>(h));
    [_win setFrame: rect display: YES];
    [_glView setFrame: rect];
    [_context update];
  }
}

void vvCocoaGLContext::init()
{
  _autoreleasePool = [[NSAutoreleasePool alloc] init];

  (void)[NSApplication sharedApplication];
  NSRect rect = NSMakeRect(0.0f, 0.0f,
                           static_cast<float>(_options.width),
                           static_cast<float>(_options.height));
  _win = [[NSWindow alloc]
    initWithContentRect:rect
    styleMask: NSWindowStyleMaskTitled
    backing: NSBackingStoreBuffered
    defer:NO];

  if (!_win)
  {
    std::cerr << "Couldn't open NSWindow" << std::endl;
  }
  [_win makeKeyAndOrderFront:nil];

  NSRect glRect = NSMakeRect(0.0f, 0.0f,
                             static_cast<float>(_options.width),
                             static_cast<float>(_options.height));

  _glView = [[NSView alloc] initWithFrame:glRect];
  [_win setContentView:_glView];
  createGLContext();
  [_context setView:_glView];
  [_context update];
  makeCurrent();
}

void vvCocoaGLContext::destroy()
{
  [_context release];
  _context = 0;

  [_glView release];
  _glView = 0;

  [_win close];
  [_win release];
  _win = 0;

  [_autoreleasePool release];
  _autoreleasePool = 0;
}

void vvCocoaGLContext::createGLContext()
{
  NSOpenGLPixelFormatAttribute attr[] = { 
    NSOpenGLPFAAccelerated,
    NSOpenGLPFADepthSize,
    (NSOpenGLPixelFormatAttribute)32,
    _options.doubleBuffering ?  (NSOpenGLPixelFormatAttribute)NSOpenGLPFADoubleBuffer : (NSOpenGLPixelFormatAttribute)0,
    (NSOpenGLPixelFormatAttribute)0
  };

  _pixelFormat = (NSOpenGLPixelFormat*)[[NSOpenGLPixelFormat alloc]
    initWithAttributes: attr];

  _context = (NSOpenGLContext*)[[NSOpenGLContext alloc]
    initWithFormat: _pixelFormat
    shareContext: nil];
}

