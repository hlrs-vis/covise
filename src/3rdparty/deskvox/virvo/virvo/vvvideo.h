//****************************************************************************
// Filename:          vvvideo.h
// Author:            Michael Poehnl
// Institution:       University of Stuttgart, Supercomputing Center
// History:           08-01-2002  Creation date
//****************************************************************************

#ifndef VV_VIDEO_H
#define VV_VIDEO_H

/**This class is the interface to the XviD library (xvidcore-0.9.0)
   It is used by the vvImage class for the XviD encoding of RGB frames. <BR>

   @author Michael Poehnl
*/
#include "vvexport.h"

struct AVCodec;
struct AVCodecContext;
struct AVFrame;
struct AVPicture;
struct SwsContext;

class VIRVOEXPORT vvVideo
{
public:
  enum Codec
  {
    VV_RAW = 0,
    VV_FFV1,
    VV_JPEGLS,
    VV_MJPEG,
    VV_MPEG2,
    VV_MPEG4,
    VV_FLV1,
    VV_NUM_CODECS
  };
  enum ColorFormat
  {
    VV_RGB24 = 0,
    VV_YUV420P,
    VV_YUVJ420P,
    VV_NUM_COLOR_FORMATS
  };
  vvVideo();
  ~vvVideo();
  void setSize(short w, short h);
  void setCodec(Codec codec);
  Codec getCodec() const;
  void setColorFormat(ColorFormat format);
  ColorFormat getColorFormat() const;
  void setFramerate(float fr);
  void setQuantizer(int min_q, int max_q);
  void setBitrate(int br);
  void setMaxKeyInterval(int max_k);

  int createEncoder(int w, int h);
  int createDecoder(int w, int h);
  int encodeFrame(const unsigned char* src, unsigned char* dst, int* enc_size);
  int decodeFrame(const unsigned char* src, unsigned char* dst, int src_size, int* dst_size);
  int destroyEncoder();
  int destroyDecoder();

private:
  float framerate;
  int min_quantizer;
  int max_quantizer;
  int bitrate;
  int max_key_interval;

  // ffmpeg
  AVCodecContext *enc_ctx;
  AVCodecContext *dec_ctx;
  AVFrame *enc_picture;
  AVFrame *dec_picture;
  SwsContext *enc_imgconv_ctx;
  SwsContext *dec_imgconv_ctx;
  int codec_id;
  int pixel_fmt;

  static bool global_init_done;
};
#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
