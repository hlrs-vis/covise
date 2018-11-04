#include <GL/glew.h>
#include "ReadBackCuda.h"

#include <sysdep/opengl.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <cstdio>

#define USE_BGRA

static const int TileSizeX = 16;
static const int TileSizeY = 16;

static bool cucheck(const char *msg, cudaError_t err)
{
#if 0
   if(err == cudaSuccess)
   {
      cudaThreadSynchronize();
      err = cudaGetLastError();
   }
#endif

   if(err == cudaSuccess)
      return true;

   std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;

   return false;
}

ReadBackCuda::ReadBackCuda()
: pboName(0)
, imgRes(NULL)
, outImg(NULL)
, imgSize(0)
{
   glewInit();

   bool canUsePbo = glGenBuffers && glDeleteBuffers && glBufferData && glBindBuffer;
   if(!canUsePbo)
   {
      throw(ReadBackError("no PBO support"));
   }
}

ReadBackCuda::~ReadBackCuda()
{
   if(imgRes)
      cucheck("unreg buf", cudaGraphicsUnregisterResource(imgRes));
   cucheck("cufree", cudaFree(outImg));
   if(pboName != 0)
      glDeleteBuffers(1, &pboName);
}

bool ReadBackCuda::initPbo(size_t size, size_t subsize)
{
   if(imgSize != size || pboName == 0)
   {
      if(pboName == 0)
      {
         glGenBuffers(1, &pboName);
      }
      else
      {
         cucheck("unreg buf", cudaGraphicsUnregisterResource(imgRes));
         cucheck("cufree", cudaFree(outImg));
      }

      glBindBuffer(GL_PIXEL_PACK_BUFFER, pboName);
      glBufferData(GL_PIXEL_PACK_BUFFER, size, NULL, GL_STREAM_COPY);
      glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

      imgSize = size;

      cucheck("reg buf", cudaGraphicsGLRegisterBuffer(&imgRes, pboName, cudaGraphicsMapFlagsNone));
      cucheck("malloc", cudaMalloc(&outImg, subsize));
   }

   return pboName != 0;
}

__device__ void colorconvert(uchar r, uchar g, uchar b, uchar *y, uchar *u, uchar *v)
{
   *y = 0.299f*r + 0.587f*g + 0.114f*b;
   *u = 128.f + 0.5f*r - 0.418688f*g - 0.081312f*b;
   *v = 128.f - 0.168736f*r - 0.331265f*g + 0.5f*b;
}

__device__ void colorconvert(uchar r, uchar g, uchar b, uchar *y, float *u, float *v)
{
   *y = 0.299f*r + 0.587f*g + 0.114f*b;
   *u += 128.f + 0.5f*r - 0.418688f*g - 0.081312f*b;
   *v += 128.f - 0.168736f*r - 0.331265f*g + 0.5f*b;
}


__global__ void rgb2yuv(const uchar *inimg, uchar *outimg, int w, int h, int subx, int suby)
{
   if(subx<1) subx=1;
   if(suby<1) suby=1;
#ifdef USE_BGRA
   const int bpp = 4;
#else
   const int bpp = 3;
#endif

   const int wsub = (w+subx-1)/subx;
   const int hsub = (h+suby-1)/suby;

   uchar *ybase = outimg;
   uchar *ubase = &ybase[w*h];

   int x=threadIdx.x+blockIdx.x*TileSizeX;
   int y=threadIdx.y+blockIdx.y*TileSizeY;

   uchar *vbase = &ubase[wsub*hsub];

   float u=0.f, v=0.f;

   uchar *uu = &ubase[x+y*wsub];
   uchar *vv = &vbase[x+y*wsub];

   x *= subx;
   y *= suby;

   if(x>=w || y>=h)
      return;

   for(int iy=0; iy<suby; ++iy)
   {
      const int yy=y+iy;
      for(int ix=0; ix<subx; ++ix)
      {
         const int xx=x+ix;
         uchar *cy = &ybase[xx+yy*w];
         const uchar r = inimg[(xx+yy*w)*bpp+0];
         const uchar g = inimg[(xx+yy*w)*bpp+1];
         const uchar b = inimg[(xx+yy*w)*bpp+2];

         colorconvert(r,g,b, cy,&u,&v);
      }
   }
   *uu = u/(subx*suby);
   *vv = v/(subx*suby);
}

bool ReadBackCuda::readpixelsyuv(GLint x, GLint y, GLint w, GLint pitch, GLint h,
      GLenum format, int ps, GLubyte *bits, GLint buf, int subx, int suby)
{
   //initGlInterop();

   GLint readbuf=GL_BACK;
   glGetIntegerv(GL_READ_BUFFER, &readbuf);

   //tempctx tc(_localdpy, EXISTING_DRAWABLE, GetCurrentDrawable());

   glReadBuffer(buf);
   glPushClientAttrib(GL_CLIENT_PIXEL_STORE_BIT);

   if(pitch%8==0) glPixelStorei(GL_PACK_ALIGNMENT, 8);
   else if(pitch%4==0) glPixelStorei(GL_PACK_ALIGNMENT, 4);
   else if(pitch%2==0) glPixelStorei(GL_PACK_ALIGNMENT, 2);
   else if(pitch%1==0) glPixelStorei(GL_PACK_ALIGNMENT, 1);

   int e=glGetError();
   while(e!=GL_NO_ERROR) e=glGetError();  // Clear previous error
   //_prof_rb.startframe();

   size_t subsize = w*h+2*((w+subx-1)/subx)*((h+suby-1)/suby);
   if(!initPbo(w*h*4, subsize))
      return false;

   glBindBuffer(GL_PIXEL_PACK_BUFFER, pboName);
   glBufferData(GL_PIXEL_PACK_BUFFER, w*h*4, NULL, GL_STREAM_COPY);
#ifdef USE_BGRA
   glReadPixels(x, y, w, h, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, NULL);
#else
   glReadPixels(x, y, w, h, GL_BGR, GL_UNSIGNED_BYTE, NULL);
#endif
   glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

   glPopClientAttrib();
   //tc.restore();
   glReadBuffer(readbuf);

   cucheck("map res", cudaGraphicsMapResources(1, &imgRes, NULL));
   size_t sz;
   uchar *img;
   cucheck("get map ptr", cudaGraphicsResourceGetMappedPointer((void **)&img, &sz, imgRes));

   dim3 grid(((w+TileSizeX-1)/TileSizeX+subx-1)/subx, ((h+TileSizeY-1)/TileSizeY+suby-1)/suby);
   dim3 block(TileSizeX, TileSizeY);

   //fprintf(stderr, "rgb2yuv: x=%d, y=%d, w=%d, h=%d, sub=(%d %d)\n", x, y, w, h, subx, suby);

   rgb2yuv<<<grid, block>>>(img, outImg, w, h, subx, suby);

   cucheck("memcpy", cudaMemcpy(bits, outImg, subsize, cudaMemcpyDeviceToHost));
   cucheck("unmap res", cudaGraphicsUnmapResources(1, &imgRes, NULL));

   //_prof_rb.endframe(w*h, 0, stereo? 0.5 : 1);
   //checkgl("Read Pixels");

   return true;
}

bool ReadBackCuda::readpixels(GLint x, GLint y, GLint w, GLint pitch, GLint h,
      GLenum format, int ps, GLubyte *bits, GLint buf)
{
   //initGlInterop();

   GLint readbuf=GL_BACK;
   glGetIntegerv(GL_READ_BUFFER, &readbuf);

   //tempctx tc(_localdpy, EXISTING_DRAWABLE, GetCurrentDrawable());

   glReadBuffer(buf);
   glPushClientAttrib(GL_CLIENT_PIXEL_STORE_BIT);

   if(pitch%8==0) glPixelStorei(GL_PACK_ALIGNMENT, 8);
   else if(pitch%4==0) glPixelStorei(GL_PACK_ALIGNMENT, 4);
   else if(pitch%2==0) glPixelStorei(GL_PACK_ALIGNMENT, 2);
   else if(pitch%1==0) glPixelStorei(GL_PACK_ALIGNMENT, 1);

   int e=glGetError();
   while(e!=GL_NO_ERROR) e=glGetError();  // Clear previous error
   //_prof_rb.startframe();

   if(!initPbo(w*h*ps, 0))
      return false;

   glBindBuffer(GL_PIXEL_PACK_BUFFER, pboName);
   glBufferData(GL_PIXEL_PACK_BUFFER, w*h*ps, NULL, GL_STREAM_COPY);
   glReadPixels(x, y, w, h, format, GL_UNSIGNED_BYTE, NULL);
   glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

   glPopClientAttrib();
   //tc.restore();
   glReadBuffer(readbuf);

   cucheck("map res", cudaGraphicsMapResources(1, &imgRes, NULL));
   size_t sz;
   char *img;
   cucheck("get map ptr", cudaGraphicsResourceGetMappedPointer((void **)&img, &sz, imgRes));
   cucheck("memcpy", cudaMemcpy(bits, img, w*h*ps, cudaMemcpyDeviceToHost));
   cucheck("unmap res", cudaGraphicsUnmapResources(1, &imgRes, NULL));

   //_prof_rb.endframe(w*h, 0, stereo? 0.5 : 1);
   //checkgl("Read Pixels");

   return true;
}


