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

#include <iostream>
using std::cerr;
using std::endl;

#include <GL/glew.h>
#include "vvplatform.h"

#include <string.h>
#ifndef VV_REMOTE_RENDERING
#include "vvopengl.h"
#endif
#include <assert.h>
#include "math/math.h"
#include "private/vvlog.h"
#include "vvdebugmsg.h"
#include "vvsoftimg.h"
#include "vvtoolshed.h"

#include "private/vvgltools.h"

using virvo::mat4;
using virvo::vec3;

const int vvSoftImg::PIXEL_SIZE = 4;

struct vvSoftImg::Impl
{
  GLuint    texName;                          ///< name of warp texture
  GLuint    pboName;                          ///< name for PBO for CUDA/OpenGL interop
  GLboolean glsTexture2D;                     ///< state buffer for GL_TEXTURE_2D
  GLboolean glsBlend;                         ///< stores GL_BLEND
  GLboolean glsLighting;                      ///< stores GL_LIGHTING
  GLboolean glsCulling;                       ///< stores GL_CULL_FACE
  GLint     glsBlendSrc;                      ///< stores glBlendFunc(source,...)
  GLint     glsBlendDst;                      ///< stores glBlendFunc(...,destination)
  GLint     glsUnpackAlignment;               ///< stores glPixelStore(GL_UNPACK_ALIGNMENT,...)
  GLfloat   glsRasterPos[4];                  ///< current raster position (glRasterPos)
};

//----------------------------------------------------------------------------
/** Constructor
  @param w,h initial image size (default: w=h=0)
*/
vvSoftImg::vvSoftImg(int w, int h)
  : impl(new Impl)
{
   vvDebugMsg::msg(1, "vvSoftImg::vvSoftImg(): ", w, h);

   glewInit();

   usePbo = false;
   width  = w;
   height = h;
   warpInterpolation = true;                      // set default interpolation type for warp
   deleteData = true;
   if (width * height > 0)
      data = new uchar[width * height * PIXEL_SIZE];
   else
      data = NULL;
   impl->pboName = 0;

   canUsePbo = glGenBuffers && glDeleteBuffers && glBufferData && glBindBuffer;

#ifndef VV_REMOTE_RENDERING
   // Generate texture name:
   glGenTextures(1, &impl->texName);
#endif
   reinitTex = true;
}


//----------------------------------------------------------------------------
/// Destructor
vvSoftImg::~vvSoftImg()
{
   vvDebugMsg::msg(1, "vvSoftImg::~vvSoftImg()");
#ifndef VV_REMOTE_RENDERING
   glDeleteTextures(1, &impl->texName);
   if (canUsePbo && impl->pboName != 0)
      glDeleteBuffers(1, &impl->pboName);
#endif
   if (deleteData) delete[] data;
}


//----------------------------------------------------------------------------
/** Set data array from buffer.
  @param buf  data to render
*/
void vvSoftImg::setBuffer(uchar* buf)
{
  vvDebugMsg::msg(3, "vvSoftImg::setBuffer()");

  if (deleteData) delete[] data;
  data = buf;
  deleteData = false;
}


//----------------------------------------------------------------------------
/** Resize image.
  @param w new image width in pixels
  @param h new image height in pixels
*/
void vvSoftImg::setSize(int w, int h)
{
   vvDebugMsg::msg(3, "vvSoftImg::setSize() ", w, h);

   if (width!=w || height!=h || (usePbo!=(impl->pboName!=0))) // recreate image buffer only if needed
   {
      width  = w;
      height = h;
      if (deleteData)
      {
        delete[] data;
        data = new uchar[width * height * PIXEL_SIZE];
      }
#ifndef VV_REMOTE_RENDERING
      if (canUsePbo)
      {
          if (usePbo)
          {
              if (impl->pboName == 0)
                  glGenBuffers(1, &impl->pboName);
              glBindBuffer(GL_PIXEL_UNPACK_BUFFER, impl->pboName);
              glBufferData(GL_PIXEL_UNPACK_BUFFER, width*height*PIXEL_SIZE, NULL, GL_STREAM_COPY);
              glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
          }
          else if (impl->pboName)
          {
              glDeleteBuffers(1, &impl->pboName);
              impl->pboName = 0;
          }
      }

      reinitTex = true;
#endif
      vvDebugMsg::msg(1, "New image size is: ", width, height);
   }
}


//----------------------------------------------------------------------------
/** Initialize texture.
*/
void vvSoftImg::initTexture(GLuint format)
{
   vvDebugMsg::msg(3, "vvSoftImg::initTexture()");

   int texWidth  = vvToolshed::getTextureSize(width);
   int texHeight = vvToolshed::getTextureSize(height);

   glBindTexture(GL_TEXTURE_2D, impl->texName);

   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (warpInterpolation) ? GL_LINEAR : GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (warpInterpolation) ? GL_LINEAR : GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
   // try GL_REPLACE and GL_MODULATE
   glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

   // Only load texture if it can be accomodated:
   glTexImage2D(GL_PROXY_TEXTURE_2D, 0, format, texWidth, texHeight, 0,
      format, GL_UNSIGNED_BYTE, NULL);

   GLint glWidth;                                 // return value from OpenGL call
   glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &glWidth);
   if (glWidth==0)
   {
      cerr << "Unsupported intermediate image texture size (" <<
         texWidth << " x " << texHeight << ")." << endl;
   }
   else
   {
       glTexImage2D(GL_TEXTURE_2D, 0, format, texWidth, texHeight, 0,
               format, GL_UNSIGNED_BYTE, NULL);
   }

   reinitTex = false;

   vvGLTools::printGLError("vvSoftImg::initTexture()");
}


//----------------------------------------------------------------------------
/** Copy source image to current image with zoom.
  @param src source image
*/
void vvSoftImg::zoom(vvSoftImg* src)
{
   int x, y, xsrc, ysrc;

   for (y=0; y<height; ++y)
      for (x=0; x<width; ++x)
   {
      xsrc = (int)((float)(x * src->width)  / (float)width);
      ysrc = (int)((float)(y * src->height) / (float)height);
      memcpy(data + PIXEL_SIZE * (x + y * width),
         src->data + PIXEL_SIZE * (xsrc + ysrc * src->width), PIXEL_SIZE);
   }
}


//----------------------------------------------------------------------------
/** Copy source image to top left position in current image.
  If source image is smaller than current image, the remaining canvas is black.<BR>
  @param align  alignmemt:<BR>
                destination < source: which part of the source image to copy
                source < destination: where to put the source image in destination image
  @param src    source image
*/
void vvSoftImg::copy(AlignType align, vvSoftImg* src)
{
   int x, y, minWidth, minHeight;

   if (width > src->width || height > src->height)// if destination image is larger than source:
                                                  // clear destination image
         memset(data, 0, PIXEL_SIZE * width * height);

   switch (align)
   {
      case BOTTOM_LEFT:
         minWidth  = ts_min(width,  src->width);
         minHeight = ts_min(height, src->height);
         for (y=0; y<minHeight; ++y)
            for (x=0; x<minWidth; ++x)
               memcpy(data + PIXEL_SIZE * (x + y * width),
                  src->data + PIXEL_SIZE * (x + y * src->width), PIXEL_SIZE);
         break;
      default: break;
   }
}


//----------------------------------------------------------------------------
/** Overlay source image at top left position into current image.
  The background color is assumed to be black.
  If source image is larger than current image, display top left corner.
  @param src source image
*/
void vvSoftImg::overlay(vvSoftImg* src)
{
   int x, y, minWidth, minHeight;
   uchar* srcPixel;                               // first byte of source pixel

   minWidth  = ts_min(width,  src->width);
   minHeight = ts_min(height, src->height);

   for (y=0; y<minHeight; ++y)
      for (x=0; x<minWidth; ++x)
   {
      srcPixel = src->data + PIXEL_SIZE * (x + y * src->width);
      if (*srcPixel != 0 || *(srcPixel + 1) != 0 ||
         *(srcPixel + 2) != 0)
         memcpy(data + PIXEL_SIZE * (x + y * width), srcPixel, PIXEL_SIZE);
   }
}


//----------------------------------------------------------------------------
/// Clear image (black is clear color)
void vvSoftImg::clear()
{
   vvDebugMsg::msg(3, "vvSoftImg::clear()");

   memset(data, 0, width * height * PIXEL_SIZE);
}


//----------------------------------------------------------------------------
/** Fill image with one color.
  @param r,g,b,a  RGBA components of fill color [0..255]
*/
void vvSoftImg::fill(int r, int g, int b, int a)
{
   int i;
   uchar* ptr;

   vvDebugMsg::msg(1, "vvSoftImg::fill()");

   assert(PIXEL_SIZE==4);                         // this algorithm only works with RGBA pixels
   ptr = data;
   for (i=0; i<width * height; ++i)
   {
      *ptr = (uchar)r; ++ptr;
      *ptr = (uchar)g; ++ptr;
      *ptr = (uchar)b; ++ptr;
      *ptr = (uchar)a; ++ptr;
   }
}


//----------------------------------------------------------------------------
/// Draw image into current OpenGL context.
void vvSoftImg::draw()
{
#ifndef VV_REMOTE_RENDERING
   GLenum format;                                 // pixel format

   vvDebugMsg::msg(3, "vvSoftImg::draw()");

   saveGLState();

   // Set identity matrix for projection:
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();

   glEnable(GL_BLEND);                            // enable alpha blending
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

   // Draw image:
   glRasterPos2f(-1.0f,-1.0f);                    // pixmap origin is bottom left corner of output window
   switch (PIXEL_SIZE)
   {
      case 1:  format = GL_LUMINANCE; break;
      case 2:  format = GL_LUMINANCE_ALPHA; break;
      case 3:  format = GL_RGB; break;
      case 4:
      default: format = GL_RGBA; break;
   }

   if (canUsePbo && impl->pboName)
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, impl->pboName);
   glDrawPixels(width, height, format, GL_UNSIGNED_BYTE, impl->pboName ? 0 : data);
   if (canUsePbo && impl->pboName)
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

   restoreGLState();
#endif
}


//----------------------------------------------------------------------------
/** Draws a border around the image.
  @param r,g,b  RGB components of border color
*/
void vvSoftImg::drawBorder(int r, int g, int b)
{
   int x, y;

   vvDebugMsg::msg(3, "vvSoftImg::drawBorder()");

   // Draw horizontal lines:
   for (y=0; y<height; y+=height-1)
      for (x=0; x<width*PIXEL_SIZE; x+=PIXEL_SIZE)
   {
      data[x + y*width*PIXEL_SIZE]     = (uchar)r;
      data[x + y*width*PIXEL_SIZE + 1] = (uchar)g;
      data[x + y*width*PIXEL_SIZE + 2] = (uchar)b;
      data[x + y*width*PIXEL_SIZE + 3] = (uchar)255;
   }

   // Draw vertical lines:
   for (x=0; x<width; x+=width-1)
      for (y=0; y<height; ++y)
   {
      data[x*PIXEL_SIZE + y*width*PIXEL_SIZE]     = (uchar)r;
      data[x*PIXEL_SIZE + y*width*PIXEL_SIZE + 1] = (uchar)g;
      data[x*PIXEL_SIZE + y*width*PIXEL_SIZE + 2] = (uchar)b;
      data[x*PIXEL_SIZE + y*width*PIXEL_SIZE + 3] = (uchar)255;
   }
}


//----------------------------------------------------------------------------
/** Warp the source image to the current image.
  @param w        4x4 warp matrix, only 2D components are used
  @param srcImg   source image which is to be warped
*/
void vvSoftImg::warp(mat4 const& w, vvSoftImg* srcImg)
{
   int xs, ys;                                    // source image coordinates
   int i, j;                                      // counters
   float xd, yd;                                  // precomputed destination values
   float pc;                                      // perspective correction
   float inv00, inv01, inv03;                     // elements of 1st row of inverted warp matrix
   float inv10, inv11, inv13;                     // elements of 2nd row of inverted warp matrix
   float inv30, inv31, inv33;                     // elements of 3rd row of inverted warp matrix

   vvDebugMsg::msg(3, "vvSoftImg::warp()");

   mat4 inv = inverse(w);                         // inverted warp matrix
                                                  // invert to compute source coords from destination coords

   inv00 = inv(0, 0);
   inv01 = inv(0, 1);
   inv03 = inv(0, 3);
   inv10 = inv(1, 0);
   inv11 = inv(1, 1);
   inv13 = inv(1, 3);
   inv30 = inv(3, 0);
   inv31 = inv(3, 1);
   inv33 = inv(3, 3);

   for (j=0; j<height; ++j)                       // loop thru destination pixels
   {
      yd = (float)j;
      for (i=0; i<width; ++i)
      {
         // Compute source coordinates:
         // pixel' = inv_warp x pixel
         xd = (float)i;
         pc = xd * inv30 + yd * inv31 + inv33;
         xs = (int)((xd * inv00 + yd * inv01 + inv03) / pc);
         ys = (int)((xd * inv10 + yd * inv11 + inv13) / pc);

         // Check if source pixel is inside source image. If not,
         // assume a black background.
         if (xs>srcImg->width-1 || ys>srcImg->height-1 || xs<0 || ys<0)
            memset(data + PIXEL_SIZE * (i + j * width), '\0', PIXEL_SIZE);
         else
            memcpy(data + PIXEL_SIZE * (i + j * width),
               srcImg->data + PIXEL_SIZE * (xs + ys * srcImg->width), PIXEL_SIZE);
      }
   }
}


//----------------------------------------------------------------------------
/** Warp the current image to the OpenGL viewport using 2D texture mapping.
  @param w 4x4 warp matrix, only 2D components are used
*/
void vvSoftImg::warpTex(mat4 const& w)
{
#ifndef VV_REMOTE_RENDERING
   const float ZPOS = 0.0f;                       // texture z position. TODO: make zPos changeable and adjust warp matrix accordingly
   static vvSoftImg* texImg = new vvSoftImg(1,1); // texture image
   GLenum format;                                 // pixel format
   int texWidth, texHeight;                       // texture compatible image size
   bool useOriginalImg;                           // true if no additional image needs to be generated because the original image size is already a power of 2
   vvSoftImg* imgUsed;                            // points to the image to use for the warp

   vvDebugMsg::msg(3, "vvSoftImg::warpTex()");
   VV_LOG(3) << "warp matrix: " << w;

   /* Save intermediate image to file:
     {
        FILE* fp;
        fp = fopen("image.rgba", "wb");
        fwrite(imgUsed->data, imgUsed->height * imgUsed->width * 4, 1, fp);
        fclose(fp);
     }
   */

   saveGLState();

   texWidth  = vvToolshed::getTextureSize(width);
   texHeight = vvToolshed::getTextureSize(height);

   if (texWidth==width && texHeight==height)
   {
      useOriginalImg = true;
      imgUsed = this;
   }
   else
   {
      useOriginalImg = false;
      imgUsed = texImg;
   }

   if (!useOriginalImg)
   {
      texImg->setSize(texWidth, texHeight);
      texImg->copy(BOTTOM_LEFT, this);
   }

   switch (PIXEL_SIZE)
   {
      case 1:  format = GL_LUMINANCE; break;
      case 2:  format = GL_LUMINANCE_ALPHA; break;
      case 3:  format = GL_RGB; break;
      case 4:
      default: format = GL_RGBA; break;
   }

   glEnable(GL_BLEND);                            // enable alpha blending
   glDisable(GL_LIGHTING);                        // disable lighting: must be done in compositing step
   glDisable(GL_CULL_FACE);                       // disable culling: couldn't be correct
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   glEnable(GL_TEXTURE_2D);                       // enable 2D texturing
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

   // Generate texture:
   if (reinitTex)
       initTexture(format);

   if (canUsePbo && impl->pboName)
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, impl->pboName);

   glBindTexture(GL_TEXTURE_2D, impl->texName);
   // Now texture can be loaded:
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imgUsed->width, imgUsed->height,
           format, GL_UNSIGNED_BYTE, impl->pboName ? 0 : imgUsed->data);

   // Modelview matrix becomes warp matrix:
   glMatrixMode(GL_MODELVIEW);
   glLoadMatrixf(w.data());

   // Projection matrix is identity matrix:
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();

   // Draw texture image:
   glBegin(GL_QUADS);
   glColor4f(1.0, 1.0, 1.0, 1.0);
   glNormal3f(0.0, 0.0, 1.0);
   // bottom left
   glTexCoord2f(0.0f, 0.0f); glVertex3f( 0.0f,                 0.0f,                   ZPOS);
   // bottom right
   glTexCoord2f(1.0f, 0.0f); glVertex3f((float)imgUsed->width, 0.0f,                   ZPOS);
   // top right
   glTexCoord2f(1.0f, 1.0f); glVertex3f((float)imgUsed->width, (float)imgUsed->height, ZPOS);
   // top left
   glTexCoord2f(0.0f, 1.0f); glVertex3f( 0.0f,                 (float)imgUsed->height, ZPOS);
   glEnd();

   if (canUsePbo && impl->pboName)
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

   restoreGLState();

   vvGLTools::printGLError("vvSoftImg::warpTex");
#else
   w = w;                                         // suppress compiler warning
#endif
}


//----------------------------------------------------------------------------
/** Draw a pixel.
  @param x,y    position
  @param color  pixel color, 32 bit value: bits 0..7=first color,
                8..15=second color etc.
*/
void vvSoftImg::putPixel(int x, int y, uint color)
{
   int pos, i;

   x = ts_clamp(x, 0, width-1);
   y = ts_clamp(y, 0, height-1);
   pos = PIXEL_SIZE * (x + y * width);
   for (i=0; i<PIXEL_SIZE; ++i)
   {
      switch (i)
      {
         case 0: data[pos]   = (uchar)(color >> 24); break;
         case 1: data[pos+1] = (uchar)(color >> 16); break;
         case 2: data[pos+2] = (uchar)(color >> 8);  break;
         case 3: data[pos+3] = (uchar)(color);       break;
         default: break;
      }
   }
}


//----------------------------------------------------------------------------
/** Draw a line.
  @param x0,y0  line start position
  @param x1,y1  line end position
  @param r,g,b  color [0..255]
*/
void vvSoftImg::drawLine(int x0, int y0, int x1, int y1, int r, int g, int b)
{
   vvDebugMsg::msg(3, "vvSoftImg::drawLine()");
   vvToolshed::draw2DLine(x0, y0, x1, y1, (uint)((r << 24) | (g << 16) | (b << 8) | 0xff),
      data, PIXEL_SIZE, width, height);
}


//----------------------------------------------------------------------------
/// Save OpenGL state information before drawing operations.
void vvSoftImg::saveGLState()
{
#ifndef VV_REMOTE_RENDERING
   vvDebugMsg::msg(3, "vvSoftImg::saveGLState()");

   // Store modelview matrix:
   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();

   // Store projection matrix:
   glMatrixMode(GL_PROJECTION);
   glPushMatrix();

   // Store blending function:
   glGetIntegerv(GL_BLEND_SRC, &impl->glsBlendSrc);
   glGetIntegerv(GL_BLEND_DST, &impl->glsBlendDst);

   // Store unpack alignment:
   glGetIntegerv(GL_UNPACK_ALIGNMENT, &impl->glsUnpackAlignment);

   // Store raster position:
                                                  // memorize raster position
   glGetFloatv(GL_CURRENT_RASTER_POSITION, impl->glsRasterPos);

   // Store boolean OpenGL states:
   glGetBooleanv(GL_TEXTURE_2D, &impl->glsTexture2D);
   glGetBooleanv(GL_BLEND, &impl->glsBlend);
   glGetBooleanv(GL_LIGHTING, &impl->glsLighting);
   glGetBooleanv(GL_CULL_FACE, &impl->glsCulling);
#endif
}


//----------------------------------------------------------------------------
/// Restore OpenGL state information after drawing operations.
void vvSoftImg::restoreGLState()
{
#ifndef VV_REMOTE_RENDERING
   vvDebugMsg::msg(3, "vvSoftImg::restoreGLState()");

   // Restore state of GL_TEXTURE_2D:
   if (impl->glsTexture2D) glEnable(GL_TEXTURE_2D);
   else glDisable(GL_TEXTURE_2D);

   // Restore state of GL_BLEND:
   if (impl->glsBlend==(uchar)true) glEnable(GL_BLEND);
   else glDisable(GL_BLEND);

   // Restore state of GL_LIGHTING:
   if (impl->glsLighting==(uchar)true) glEnable(GL_LIGHTING);
   else glDisable(GL_LIGHTING);

   // Restore state of GL_CULL_FACE:
   if (impl->glsCulling==(uchar)true) glEnable(GL_CULL_FACE);
   else glDisable(GL_CULL_FACE);

   // Restore blending function:
   glBlendFunc(impl->glsBlendSrc, impl->glsBlendDst);

   // Restore unpack alignment:
   glPixelStorei(GL_UNPACK_ALIGNMENT, impl->glsUnpackAlignment);

   // Restore raster position:
   glRasterPos4fv(impl->glsRasterPos);

   // Restore projection matrix:
   glMatrixMode(GL_PROJECTION);
   glPopMatrix();

   // Restore modelview matrix:
   glMatrixMode(GL_MODELVIEW);
   glPopMatrix();
#endif
}


//----------------------------------------------------------------------------
/// Set warp interpolation mode.
void vvSoftImg::setWarpInterpolation(bool newMode)
{
   vvDebugMsg::msg(3, "vvSoftImg::setWarpInterpolation()");
   warpInterpolation = newMode;
   reinitTex = true;
}


//----------------------------------------------------------------------------
/** Set new image data.
  @param w,h  image width and height [pixels]
  @param data image data, w*h*PIXEL_SIZE bytes expected. The data
              must _not_ be deleted by the caller!
*/
void vvSoftImg::setImageData(int w, int h, uchar* d)
{
   if (data!=d)
   {
      if (deleteData) delete[] data;
      data = d;
      deleteData = false;
   }
   width  = w;
   height = h;
}


//----------------------------------------------------------------------------
/** Print information about image.
  @param title some text to print before the image data
*/
void vvSoftImg::print(const char* title)
{
   if (title) cerr << title << " ";
   cerr << "width  = " << width << ", height = " << height << endl;
}


//----------------------------------------------------------------------------
/** Return name for PBO used for CUDA/OpenGL interoperability
*/
uint vvSoftImg::getPboName() const
{
   return impl->pboName;
}


//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
