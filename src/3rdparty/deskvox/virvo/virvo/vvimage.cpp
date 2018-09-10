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

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include <iostream>
#include <cstring>
#include <cassert>

#include "vvimage.h"
#include "vvdebugmsg.h"
#include "vvvideo.h"
#include "vvtoolshed.h"

#ifdef HAVE_CONFIG_H
#include <vvconfig.h>
#endif

#ifdef HAVE_SNAPPY
#include <snappy.h>
#endif

using namespace std;

//----------------------------------------------------------------------------
/** Constructor for initialization with an image
    @param h   picture height
    @param w   picture width
    @param image   pointer to the image
*/
vvImage::vvImage(short h, short w, uchar* image)
: imageptr(image)
, videoEncoder(NULL)
, videoDecoder(NULL)
, height(h)
, width(w)
{
  vvDebugMsg::msg(3, "vvImage::vvImage(): ", w, h);
  videosize = 0;
  size = height*width*4;
  codetype = VV_RAW;
  codedimage = new uchar[size*2];
  videoimageptr = new uchar[width*height*6];
  videocodedimage = new uchar[width*height*6+8];
  tmpimage = new uchar[width*height];
  t = VV_SERVER;
  videostyle = 0;
  videoquant = 1;
}

//----------------------------------------------------------------------------
/** Constructor for an empty image
 */
vvImage::vvImage()
: imageptr(0)
, videoEncoder(NULL)
, videoDecoder(NULL)
, height(0)
, width(0)
{
  height = 0;
  width = 0;
  size = 0;
  videosize = 0;
  imageptr = 0;
  codedimage = 0;
  videoimageptr = 0;
  videocodedimage = 0;
  tmpimage = 0;
  codetype = VV_RAW;
  t = VV_CLIENT;
  videostyle = 0;
  videoquant = 1;
}

//----------------------------------------------------------------------------
/** Destructor
 */
vvImage::~vvImage()
{
  if (t == VV_CLIENT)
  {
    if (imageptr != codedimage)
    {
      delete[] imageptr;
    }
  }
  delete[] codedimage;

  delete[] videoimageptr;
  delete[] videocodedimage;
  delete[] tmpimage;
  delete videoEncoder;
  delete videoDecoder;
}

//----------------------------------------------------------------------------
/**Reinitializes an image object with new width and height
@param h   picture height
@param w   picture width
@param image   pointer to the image
*/
void vvImage::setNewImage(short h, short w, uchar* image)
{
  vvDebugMsg::msg(3, "vvImage::setNewImage(): ", w, h);

  if (width != w || height != h)
  {
    destroyCodecs();
  }

  height =h;
  width = w;
  imageptr = image;
  size = height*width*4;
  codetype = VV_RAW;
  delete[] codedimage;
  codedimage = new uchar[size*2];
  delete [] videoimageptr;
  videoimageptr = new uchar[width*height*6];
  delete [] videocodedimage;
  videocodedimage = new uchar[width*height*6+8];
  delete [] tmpimage;
  tmpimage = new uchar[width*height];
}

//----------------------------------------------------------------------------
/**Sets the image pointer to a new image which has the same height and
width as the old one.
@param image   pointer to the image
*/
void vvImage::setNewImagePtr(uchar* image)
{
  imageptr = image;
  size = height*width*4;
  codetype = VV_RAW;
}

static void checksum(const uchar *img, short height, short width)
{
  unsigned sum=0;
  for(int i=0; i<height; i++)
  {
    for(int j=0; j<width; j++)
    {
      sum += img[(i*width+j)*4] + img[(i*width+j)*4+1] + img[(i*width+j)*4+2] + img[(i*width+j)*4+3];
    }
  }
  char buf[100];
  sprintf(buf, "check sum: 0x%08x, last: 0x%08x", sum, *(uint32_t*)&img[width*height-4]);
  vvDebugMsg::msg(0, buf);
}

//----------------------------------------------------------------------------
/**Encodes an image
@param ct   codetype to use (see detailed description of the class and CodeType,
            values starting from VV_VIDEO are used for the different video codecs)
@param sw   start pixel relating to width
@param ew   end pixel relating to width
@param sh   start pixel relating to height
@param eh   end pixel relating to height
@return size of encoded image in bytes, or -1 on error
*/
int vvImage::encode(short ct, short sw, short ew, short sh, short eh)
{
  short realheight, realwidth;
  int start;
  float cr;

  if (size <= 0)
  {
    vvDebugMsg::msg(1, "Illegal image parameters ");
    return -1;
  }
  if(vvDebugMsg::isActive(3))
  {
    checksum(imageptr, height, width);
  }
  switch(ct)
  {
    case VV_RAW:cr=1;break;
    case VV_RLE:
    {
      if (spec_RLC_encode(0, height, width))
      {
        vvDebugMsg::msg(1, "No compression possible");
        codetype = VV_RAW;
      }
      else
        codetype = VV_RLE;
      cr = (float)size / (height*width*4);
    }break;
    case VV_RLE_RECT:
    {
      codetype = VV_RLE_RECT;
      if(sh<0 || eh<0  || sw<0 || ew<0 ||
        (realheight=short(eh-sh+1))<=0 || (realwidth=short(ew-sw+1))<=0 ||
        eh > height-1 || ew > width-1)
      {
        vvDebugMsg::msg(1,"Wrong usage vvImage::encode()");
        return -1;
      }
      start = (sh)*width*4 + (sw)*4;
      virvo::serialization::write32(&codedimage[0],(ulong)start);
      virvo::serialization::write16(&codedimage[4],realwidth);
      if (spec_RLC_encode(start, realheight, realwidth, 6))
      {
        vvDebugMsg::msg(1,"No compression possible");
        codetype = VV_RAW;
      }
      cr = (float)size / (height*width*4);
    }break;
    case VV_SNAPPY:
    {
      if ( (size = snappyEncode(imageptr, codedimage, width*height*4, width*height*4*2, 4)) < 0)
      {
        vvDebugMsg::msg(0,"Error: snappyEncode()");
        return -1;
      }
      codetype = VV_SNAPPY;
      imageptr = codedimage;
      cr = (float)size / (height*width*4);
    }break;
    default:
    {
      int codec = ct - VV_VIDEO;
      if(!videoEncoder)
        createCodecs();
      if(!videoEncoder)
        return -1;
      if(videoEncoder->getCodec() != codec)
      {
        videoEncoder->setCodec((vvVideo::Codec)codec);
      }
      int i;
      for (i=0; i<width*height; ++i)
        memcpy(&videoimageptr[i * 3], &imageptr[i * 4], 3);
      for (i=0; i<width*height; ++i)
        memcpy(&tmpimage[i], &imageptr[i * 4 +3], 1);
      //memset(imageptr, 0xff, width*height*1);
      if (videoEncode())
      {
        vvDebugMsg::msg(1,"Error: videoEncode()");
        return -1;
      }
      if ( (size = gen_RLC_encode(tmpimage, codedimage, width*height, width*height*4)) < 0)
      {
        vvDebugMsg::msg(1,"Error: gen_RLC_encode()");
        return -1;
      }
      codetype = VV_VIDEO;
      imageptr = codedimage;
      cr = (float)(size+videosize) / (height*width*4);
    }break;
  }
  vvDebugMsg::msg(2, "compression rate: ", cr);
  vvDebugMsg::msg(3, "image encoding succeeded");
  return size;
}

//----------------------------------------------------------------------------
/** Decodes an image
 */
int vvImage::decode()
{
  short  realwidth;
  int start;

  switch(codetype)
  {
    case VV_RAW: imageptr = codedimage;break;
    case VV_RLE:
    {
      spec_RLC_decode(0, width);
    }break;
    case VV_RLE_RECT:
    {
      memset(imageptr, 0, height*width*4);
      start = (int)virvo::serialization::read32(&codedimage[0]);
      realwidth = virvo::serialization::read16(&codedimage[4]);
      spec_RLC_decode(start, realwidth, 6);
    }break;
    case VV_SNAPPY:
    {
      if (snappyDecode(codedimage, imageptr, size, width*height*4, 4))
      {
        vvDebugMsg::msg(1,"Error: snappyDecode()");
        return -1;
      }
    }break;
    case VV_VIDEO:
    {
      int i;
      if (videoDecode())
      {
        vvDebugMsg::msg(1,"Error: videoDecode()");
        return -1;
      }
      for (i=0; i<width*height; ++i)
        memcpy(&imageptr[i * 4], &videoimageptr[i * 3], 3);
      if (gen_RLC_decode(codedimage, tmpimage, size, width*height))
      {
        vvDebugMsg::msg(1,"Error: gen_RLC_decode()");
        return -1;
      }
      for (i=0; i<width*height; ++i)
        memcpy(&imageptr[i * 4 + 3], &tmpimage[i], 1);
      size = width*height*4;
    }break;
    default:
      vvDebugMsg::msg(1,"No encoding type with that identifier");
      return -1;
  }
  codetype = VV_RAW;
  if(vvDebugMsg::isActive(3))
  {
    checksum(imageptr, height, width);
  }
  vvDebugMsg::msg(3, "image decoding succeeded");
  return 0;
}

//----------------------------------------------------------------------------
/** Sets the image height.
 */
void vvImage::setHeight(short h)
{
  if(h != height)
  {
    height = h;
    destroyCodecs();
  }
}

//----------------------------------------------------------------------------
/** Sets the image width.
 */
void vvImage::setWidth(short w)
{
  if(w != width)
  {
    width = w;
    destroyCodecs();
  }
}

//----------------------------------------------------------------------------
/** Sets the code type.
 */
void vvImage::setCodeType(CodeType ct)
{
  codetype = ct;
}

//----------------------------------------------------------------------------
/** Sets the image size.
 */
void vvImage::setSize(int s)
{
  size = s;
}

//----------------------------------------------------------------------------
/** Sets the video image size.
 */
void vvImage::setVideoSize(int s)
{
  if (vvDebugMsg::isActive(3))
    fprintf(stderr, "setVideoSize: s=%d\n", s);
  videosize = s;
}

//----------------------------------------------------------------------------
/** Sets the style of video encoding
 */
void vvImage::setVideoStyle(int s)
{
  if ( (s<0) || (s>6) )
  {
    vvDebugMsg::msg(1, "videoStyle hast to be between 0 and 6, using 0 now");
    videostyle = 0;
  }
  else
    videostyle = s;
}

//----------------------------------------------------------------------------
/** Sets the value for the video quantizer
 */
void vvImage::setVideoQuant(int q)
{
  if ( (q<1) || (q>31) )
  {
    vvDebugMsg::msg(1,"videoQuant has to be between 1 and 31, using 1 now");
    videoquant = 1;
  }
  else
    videoquant = q;
}

//----------------------------------------------------------------------------
/**Returns the image height
 */
short vvImage::getHeight() const
{
  return height;
}

//----------------------------------------------------------------------------
/** Returns the image width
 */
short vvImage::getWidth() const
{
  return width;
}

//----------------------------------------------------------------------------
/** Returns the code type
 */
vvImage::CodeType vvImage::getCodeType() const
{
  return codetype;
}

//----------------------------------------------------------------------------
/** Returns the image size in bytes
 */
int vvImage::getSize() const
{
  return size;
}

//----------------------------------------------------------------------------
/** Returns the video image size in bytes
 */
int vvImage::getVideoSize() const
{
  return videosize;
}

//----------------------------------------------------------------------------
/** Returns the pointer to the image
 */
uchar* vvImage::getImagePtr() const
{
  return imageptr;
}

//----------------------------------------------------------------------------
/** Returns the pointer to the encoded image
 */
uchar* vvImage::getCodedImage() const
{
  return codedimage;
}

//----------------------------------------------------------------------------
/** Returns the pointer to the encoded video image
 */
uchar* vvImage::getVideoCodedImage() const
{
  return videocodedimage;
}

//----------------------------------------------------------------------------
/** Writes a RLE encoded set of same pixels.
@param sP   number of same pixels
@param d   destination in coded image where to write
*/
void vvImage::put_same(short& sP, int& d)
{
  codedimage[d] = (uchar)(126+sP);
  d += 5;
  sP = 1;
}

//----------------------------------------------------------------------------
/** Writes a RLE encoded set of different pixels.
@param dP   number of different pixels
@param d   destination in coded image where to write
*/
void vvImage::put_diff(short& dP, int& d)
{
  codedimage[d] = (uchar)(dP-1);
  d += 1+4*dP;
  dP=0;
}

//----------------------------------------------------------------------------
/**Does the Run Length Encoding for a defined cutout of an image.
@param start   start pixel for RLE encoding
@param h   height of pixel square to encode
@param w   width of pixel square to encode
@param dest   start writing in coded image at position dest
 * runs of pixels are introduced by a single byte l,
 * l < 128 introduces a run of l+1 pixels of differing colors, (l+1)*4 color bytes follow,
 * l >= 128 introduces a run of l-127 pixels of the same color, 4 color bytes follow
*/
int vvImage::spec_RLC_encode(int start, short h, short w, int dest)
{
  short samePixel=1; // we include the leading pixel in the count
  short diffPixel=0;

  for ( int i=0; i < h; i++)
  {
    int src = start + i*width*4;
    for ( int j=0; j < w; j++)
    {
      int next = src + 4;
      if(j == w-1)
      {
        // skip to next line
        next = src + (width-w)*4 + 4;
        if(i == h-1)
          // but not if we are on the last line
          next = -1;
      }
      if (next>0 && memcmp(&imageptr[src], &imageptr[next], 4)==0)
      {
        if(samePixel == 129)
          put_same(samePixel, dest);
        else
        {
          if(diffPixel > 0 )
            put_diff(diffPixel, dest);
          samePixel++;
          if(samePixel == 2)
          {
            if ((dest+5) > size)
              return -1;
            memcpy(&codedimage[dest+1], &imageptr[src], 4);
          }
        }
      }
      else
      {
        if (samePixel > 1)
          put_same(samePixel, dest);
        else
        {
          if ((dest+5+4*diffPixel) > size)
            return -1;
          memcpy(&codedimage[dest+1+diffPixel*4], &imageptr[src], 4);
          diffPixel++;
          if(diffPixel == 128)
            put_diff(diffPixel, dest);
        }
      }
      src += 4;
    }
  }
  // if we have a final run, finish it
  if (samePixel > 1)
  {
    samePixel--;
    put_same(samePixel, dest);
  }
  else if (diffPixel > 0)
    put_diff(diffPixel, dest);

  imageptr = codedimage;
  size = dest;
  return 0;
}

//----------------------------------------------------------------------------
/** Does the Run Length Decoding for a cutout of an image
@param start   start pixel where the decoded pixel square is
written
@param w   width of pixel square to decode
@param src   start position of encoded pixels in coded image
*/
int vvImage::spec_RLC_decode(int start, short w, int src)
{
  int dest = start;
  while (src < size)
  {
    int length = (int)codedimage[src];
    if (length > 127)
    {
      for(int i=0; i<(length - 126); i++)
      {
        if (((dest-start-4*w)% (4*width)) == 0 && dest != start)
          dest += (width-w)*4;
		assert(dest <= (height*width*4)-4);
		assert(src+1 <= size-4);
        memcpy(&imageptr[dest], &codedimage[src+1], 4);
        dest += 4;
      }
      src += 5;
    }
    else
    {
      length++;
      for(int i=0; i<(length); i++)
      {
        if (((dest-start-4*w)% (4*width)) == 0 && dest != start)
          dest += (width-w)*4;
		assert(dest <= (height*width*4)-4);
		assert(src+1+i*4 <= size-4);
        memcpy(&imageptr[dest], &codedimage[src+1+i*4], 4);
        dest +=4;
      }
      src += 1+4*length;
    }
  }
  size = height*width*4;
  return 0;
}

//----------------------------------------------------------------------------
/** Allocates momory for a new image
 */
int vvImage::alloc_mem()
{
  vvDebugMsg::msg(3, "vvImage::alloc_mem(): ", width, height);

  if (codedimage != imageptr)
  {
    delete[] codedimage;
    codedimage = NULL;
  }
  delete[] imageptr;
  imageptr = NULL;
  delete [] videoimageptr;
  videoimageptr = NULL;
  delete [] videocodedimage;
  videocodedimage = NULL;
  delete [] tmpimage;
  tmpimage = NULL;

  if (codetype != VV_RAW)
  {
    imageptr = new uchar[height*width*4];
    if (!imageptr)
      return -1;
  }
  if (codetype == VV_VIDEO)
  {
    videoimageptr = new uchar[height*width*6];
    if (!videoimageptr)
      return -1;
    videocodedimage = new uchar[height*width*6+8];
    if (!videocodedimage)
      return -1;
    tmpimage = new uchar[height*width];
    if (!tmpimage)
      return -1;
  }
  codedimage = new uchar[height*width*4*2];
  if (!codedimage)
    return -1;

  return 0;
}

//----------------------------------------------------------------------------
/** Does the video encoding
 */
int vvImage::videoEncode()
{
  if(!videoEncoder)
    createCodecs();
  if(!videoEncoder)
    return -1;

  videosize = width * height * 6 + 8;
  if (videoEncoder->encodeFrame(videoimageptr, videocodedimage, &videosize))
  {
    vvDebugMsg::msg(1,"Error videoEncode()");
    return -1;
  }
  vvDebugMsg::msg(3, "encoded video image size: ", videosize);

  return 0;
}

//----------------------------------------------------------------------------
/** Does the video decoding
 */
int vvImage::videoDecode()
{
  int newsize = width * height * 6;

  if(!videoDecoder)
    createCodecs();
  if(!videoDecoder)
    return -1;

  if (videoDecoder->decodeFrame(videocodedimage, videoimageptr, videosize, &newsize))
  {
    vvDebugMsg::msg(1,"Error videoDecode()");
    return -1;
  }

  return 0;
}

//----------------------------------------------------------------------------
/** general function for the RLC encoding
 */
                                                  // size=total size in byte
int vvImage::gen_RLC_encode(const uchar* in, uchar* out, int size, int space, int symbol_size)
{
  int same_symbol=1;
  int diff_symbol=0;
  int src=0;
  int dest=0;
  bool same;
  int i;

  if ((size % symbol_size) != 0)
  {
    vvDebugMsg::msg(1,"No RLC encoding possible with this parameters");
    return -1;
  }

  while (src < (size - symbol_size))
  {
    same = true;
    for (i=0; i<symbol_size; i++)
    {
      if (in[src+i] != in[src+symbol_size+i])
      {
        same = false;
        break;
      }
    }
    if (same)
    {
      if (same_symbol == 129)
      {
        out[dest] = (uchar)(126+same_symbol);
        dest += symbol_size+1;
        same_symbol = 1;
      }
      else
      {
        same_symbol++;
        if (diff_symbol > 0)
        {
          out[dest] = (uchar)(diff_symbol-1);
          dest += 1+symbol_size*diff_symbol;
          diff_symbol=0;
        }
        if (same_symbol == 2)
        {
          if ((dest+1+symbol_size) > space)
          {
            vvDebugMsg::msg(1,"Not enough memory to encode");
            return -1;
          }
          memcpy(&out[dest+1], &in[src], symbol_size);
        }
      }
    }
    else
    {
      if (same_symbol > 1)
      {
        out[dest] = (uchar)(126+same_symbol);
        dest += symbol_size+1;
        same_symbol = 1;
      }
      else
      {
        if ((dest+1+diff_symbol*symbol_size+symbol_size) > space)
        {
          vvDebugMsg::msg(1,"Not enough memory to encode");
          return -1;
        }
        memcpy(&out[dest+1+diff_symbol*symbol_size], &in[src], symbol_size);
        diff_symbol++;
        if (diff_symbol == 128)
        {
          out[dest] = (uchar)(diff_symbol-1);
          dest += 1+symbol_size*diff_symbol;
          diff_symbol=0;
        }
      }
    }
    src += symbol_size;
  }
  if (same_symbol > 1)
  {
    out[dest] = (uchar)(126+same_symbol);
    dest += symbol_size+1;
  }
  else
  {
    if ((dest+1+diff_symbol*symbol_size+symbol_size) > space)
    {
      vvDebugMsg::msg(1,"Not enough memory to encode");
      return -1;
    }
    memcpy(&out[dest+1+diff_symbol*symbol_size], &in[src], symbol_size);
    diff_symbol++;
    out[dest] = (uchar)(diff_symbol-1);
    dest += 1+symbol_size*diff_symbol;
  }
  if (dest > size)
  {
    vvDebugMsg::msg(1,"No compression possible with RLC !!!");
  }
  return dest;
}

//----------------------------------------------------------------------------
/** general function for the RLC decoding
 */
int vvImage::gen_RLC_decode(const uchar* in, uchar* out, int size, int space, int symbol_size)
{
  int src=0;
  int dest=0;
  int i, length;

  while (src < size)
  {
    length = (int)in[src];
    if (length > 127)
    {
      for(i=0; i<(length - 126); i++)
      {
        if ((dest + symbol_size) > space)
        {
          vvDebugMsg::msg(1,"Not enough memory to decode");
          return -1;
        }
        memcpy(&out[dest], &in[src+1], symbol_size);
        dest += symbol_size;
      }
      src += 1+symbol_size;
    }
    else
    {
      length++;
      if ((dest + length*symbol_size) > space)
      {
        vvDebugMsg::msg(1,"Not enough memory to decode");
        return -1;
      }
      memcpy(&out[dest], &in[src+1], symbol_size*length);
      dest += length*symbol_size;
      src += 1+symbol_size*length;
    }
  }
  return 0;
}

int vvImage::destroyCodecs()
{
  delete videoEncoder;
  videoEncoder = NULL;
  delete videoDecoder;
  videoDecoder = NULL;

  return 0;
}

int vvImage::createCodecs()
{
  if(width <= 0 || height <= 0)
    return -1;

  if(!videoEncoder)
    videoEncoder = new vvVideo();
  if(!videoDecoder)
    videoDecoder = new vvVideo();
  if(videoEncoder->createEncoder(width, height) == -1)
  {
    cerr << "vvImage::vvImage(): failed to create video encoder" << endl;
    delete videoEncoder;
    videoEncoder = NULL;
    delete videoDecoder;
    videoDecoder = NULL;
    return -1;
  }
  if(videoDecoder->createDecoder(width, height) == -1)
  {
    cerr << "vvImage::vvImage(): failed to create video decoder" << endl;
    delete videoDecoder;
    videoDecoder = NULL;
    return -1;
  }
  return 0;
}

//----------------------------------------------------------------------------
/** general function for encoding with snappy
 */
                                                  // size=total size in byte
int vvImage::snappyEncode(const uchar* in, uchar* out, int size, int space, int symbol_size)
{
#ifdef HAVE_SNAPPY
  (void)symbol_size;
  if(snappy::MaxCompressedLength(size) > uint(space))
    return -1;
  size_t compressed = 0;
  snappy::RawCompress((const char *)in, size, (char *)out, &compressed);
  return compressed;
#else
  (void)in;
  (void)out;
  (void)size;
  (void)space;
  (void)symbol_size;
  return -1;
#endif
}

//----------------------------------------------------------------------------
/** general function for decoding with snappy
 */
int vvImage::snappyDecode(const uchar* in, uchar* out, int size, int space, int symbol_size)
{
#ifdef HAVE_SNAPPY
  (void)space;
  (void)symbol_size;
  if(!snappy::RawUncompress((const char *)in, size, (char *)out))
    return -1;
  return 0;
#else
  (void)in;
  (void)out;
  (void)size;
  (void)space;
  (void)symbol_size;
  return -1;
#endif
}
//----------------------------------------------------------------------------
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
