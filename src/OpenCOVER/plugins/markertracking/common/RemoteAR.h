/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _REMOTEAR_H_
#define _REMOTEAR_H_
/************************************************************************
 *									*
 *          								*
 *                            (C) 2002					*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 *	File			RemoteAR.h 				*
 *									*
 *	Description		RemoteAR optical tracking system interface c4lass				*
 *									*
 *	Author			Uwe Woessner				*
 *									*
 *	Date			July 2002				*
 *									*
 *	Status			in dev					*
 *									*
 * This Plugin is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This plugin is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library (see license.txt); if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 ************************************************************************/

#include <cover/MarkerTracking.h>

#include <osg/Matrix>
#include <osg/Vec3>
#include <osg/Vec2>

#include <util/DLinkList.h>
#include <osg/Vec3>
#include <osg/Vec2>
#include <osg/Array>
#include <net/covise_host.h>
#include <net/covise_connect.h>

#if defined(VV_FFMPEG) || defined(VV_XVID)
extern "C" {
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}
#endif

using namespace covise;
using namespace opencover;
namespace osg
{
class Geode;
class MatrixTransform;
class Texture2D;
class Geometry;
}
#pragma pack(push, 8)
struct VideoParameters
{
    int xsize, ysize; // dimensions of video
    int encSize; // Size of encoded video frame
    int ID; // ID of sending OpenCOVER instance
    int format; // Encoder format (FFMPEG, XVID, CUDA)
    int codec; // compression codec used by sending OpenCover
    int pixelformat; // pixel format used for previously given codec
    float viewerPosition[3]; // Vector defining viewer position in world coordinates
    float screenWidth; // x-dimension of VR-screen
    float screenHeight; // y-dimension of VR-screen
    float matrix[4][4]; // Matrix containing orientation, scale and position
    // of tracked object in video
};
#pragma pack(pop)

class RemoteVideo
{
public:
    RemoteVideo(int id, std::string arvariant = "MarkerTracking");
    ~RemoteVideo();

    void update(const char *image, osg::Matrix &mat);
    inline int getRemoteID()
    {
        return remoteID;
    }

    bool isDesktopMode();
    void setDesktopMode(bool mode);

private:
    int remoteID;
    osg::MatrixTransform *video;
    osg::Texture2D *videoTex;
    int xsize, ysize;
    int texXsize, texYsize;
    osg::Vec2Array *texcoord;
    osg::Geode *geometryNode;
    void createGeode();
    bool bDesktopMode;
};
class RemoteAR : public RemoteARInterface
{
public:
    RemoteAR();
    virtual ~RemoteAR();
    static RemoteAR *remoteAR;
    VideoParameters vp;
    inline int getAllocXSize()
    {
        return allocXSize;
    }
    inline void setAllocXSize(int xsize)
    {
        allocXSize = xsize;
    }
    inline int getAllocYSize()
    {
        return allocYSize;
    }
    inline void setAllocYSize(int ysize)
    {
        allocYSize = ysize;
    }
    virtual void receiveImage(const char *data);
    virtual void update();
    virtual void updateBitrate(const int bitrate);
    virtual bool usesIRMOS() const;
    virtual bool isReceiver() const;
    virtual ClientConnection *getIRMOSClient() const;
    void debug();
    DLinkList<RemoteVideo *> *getRemoteVideoList();

protected:
    int sendBinARMessage(covise::TokenBuffer &tb);

private:
#if defined(VV_FFMPEG) || defined(VV_XVID)
    CodecID codecID;
    PixelFormat pix_fmt;
    bool ffmpeg_startup;
    int gop_size;
    int max_b_frames;

    // for ffmpeg encoding
    AVCodec *encoder;
    AVCodecContext *encContext;
    AVFrame *encPicture, *yuvPicture;
    int outbuf_size;
    uint8_t *outbuf, *yuvBuffer;

    // for ffmpeg decoding
    AVCodec *decoder;
    AVCodecContext *decContext;
    AVFrame *decPicture, *rgbPicture;
    int got_picture;
    uint8_t *inbuf, *rgbBuffer;
    SwsContext *xformContext;

#endif

    bool m_send_raw;

    Message *m_send_message;

    int allocXSize, allocYSize;
    unsigned char *imageBuffer;
    unsigned char *textureBuffer;
    unsigned char *sendBuffer;
    unsigned char *sendVideoBuffer;
    bool sendVideo;
    int encSize;
    int key;
    int style;
    int quant;
    VideoParameters *sendVP;
    osg::Matrix viewerMat;
    DLinkList<RemoteVideo *> remoteVideos;
    static int nextPT(int n);
    bool initEncoder();
    bool initDecoder();
    bool useIRMOS;
    bool irmosReceiver;
    ClientConnection *irmos_client;
    Host *irmos_host;
};
#endif
