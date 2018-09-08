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

#ifndef VV_IMAGE_H
#define VV_IMAGE_H

#include "vvexport.h"
#include "vvinttypes.h"

class vvVideo;

//----------------------------------------------------------------------------
/**This class provides different encoding and decoding types for RGB images. <BR>

Supported code types:
- no encoding (code type VV_RAW)
- Run Length Encoding over the whole image (code type VV_RLE)
- Run Length Encoding over a quadratic part of the image (code type VV_RLE_RECT).
  Therefore start and end pixels for width an height must be specified.
  The rest of the image is interpreted as background and the pixels get the
  value 0,0,0,0. (picture width from 0 - width-1, picture height from 0 - height-1)
- Video Encoding (code type VV_VIDEO). For this type the VV_FFMPEG/VV_XVID Flag must be set.<BR>

Here is an example code fragment for encoding and decoding an image with
800 x 600 pixels :<BR>
<PRE>

//Create a new image class instance
vvImage* im = new vvImage(600, 800, (char *)imagepointer);

//Encode with normal RLE
if(im->encode(1) < 0)
{
delete im;
return -1;
}

//Or encode with RLE but only the lower half of the image
if(im->encode(2, 0, 799, 300, 599 ) < 0)
{
delete im;
return -1;
}
//Decode the image
if(im->decode())
return -1;

delete im;
</PRE>
*/
class VIRVOEXPORT vvImage
{
  public:
  enum CodeType
  {
    VV_RAW,
    VV_RLE,
    VV_RLE_RECT,
    VV_SNAPPY,
    VV_VIDEO // keep last, actual video codec value will be added to this
  };

    vvImage(short height, short width, uchar *pixels);
    vvImage();
    virtual ~vvImage();
    virtual int encode(short codetype, short sh=-1, short eh=-1, short sw=-1, short ew=-1);
    virtual int decode();
    void setNewImage(short, short, uchar*);
    void setHeight(short);
    void setWidth(short);
    void setCodeType(CodeType ct);
    void setSize(int);
    void setVideoSize(int);
    void setNewImagePtr(uchar*);
    void setVideoStyle(int);
    void setVideoQuant(int);
    CodeType getCodeType() const;
    short getHeight() const;
    short getWidth() const;
    int getSize() const;
    int getVideoSize() const;
    uchar* getImagePtr() const;
    uchar* getCodedImage() const;
    uchar* getVideoCodedImage() const;
    int alloc_mem();

  private:

    CodeType codetype;
    int size;
    int videosize;
    uchar* imageptr;
    uchar* codedimage;
    uchar* videoimageptr;
    uchar* videocodedimage;
    uchar* tmpimage;
    int videostyle;
    int videoquant;
    vvVideo* videoEncoder;
    vvVideo* videoDecoder;

    int spec_RLC_encode(int, short, short, int dest=0);
    int spec_RLC_decode(int, short, int src=0);
    void put_diff(short&, int&);
    void put_same(short&, int&);
    int videoEncode();
    int videoDecode();

protected:
    enum Type
    {
      VV_SERVER,
      VV_CLIENT
    };

    Type t;
    short height;
    short width;
    int createCodecs();
    int destroyCodecs();

    typedef int (*CodecFunc)(const uchar *, uchar *, int, int, int);
    static int gen_RLC_encode(const uchar *in, uchar *out, int size, int space, int symbol_size=1);
    static int gen_RLC_decode(const uchar *in, uchar *out, int size, int space, int symbol_size=1);
    static int snappyEncode(const uchar *in, uchar *out, int size, int space, int symbol_size=1);
    static int snappyDecode(const uchar *in, uchar *out, int size, int space, int symbol_size=1);
};

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
