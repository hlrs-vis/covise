//****************************************************************************
// Filename:          vvvideo.cpp
// Author:            Michael Poehnl
// Institution:       University of Stuttgart, Supercomputing Center
// History:           19-12-2002  Creation date
//****************************************************************************
#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#ifdef HAVE_FFMPEG
#define VV_FFMPEG
#else
#undef VV_FFMPEG
#endif

#include "vvvideo.h"
#include "vvdebugmsg.h"
#ifndef NULL
#ifdef __GNUG__
#define NULL (__null)
#else
#define NULL (0)
#endif
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cassert>

#ifdef VV_FFMPEG
#define __STDC_CONSTANT_MACROS
#include <stdint.h>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(56, 60, 100)
typedef PixelFormat AVPixelFormat;

#define AV_PIX_FMT_RGB24 PIX_FMT_RGB24
#define AV_PIX_FMT_YUV420P PIX_FMT_YUV420P
#define AV_PIX_FMT_YUVJ420P PIX_FMT_YUVJ420P

#define AV_CODEC_ID_RAWVIDEO CODEC_ID_RAWVIDEO
#define AV_CODEC_ID_FFV1 CODEC_ID_FFV1
#define AV_CODEC_ID_JPEGLS CODEC_ID_JPEGLS
#define AV_CODEC_ID_MJPEG CODEC_ID_MJPEG
#define AV_CODEC_ID_MPEG1VIDEO CODEC_ID_MPEG1VIDEO
#define AV_CODEC_ID_MPEG2VIDEO CODEC_ID_MPEG2VIDEO
#define AV_CODEC_ID_MPEG4 CODEC_ID_MPEG4
#define AV_CODEC_ID_FLV1 CODEC_ID_FLV1
#define AV_CODEC_ID_RAWVIDEO CODEC_ID_RAWVIDEO

#define AV_CODEC_FLAG_QSCALE CODEC_FLAG_QSCALE
#define AV_CODEC_FLAG_LOW_DELAY CODEC_FLAG_LOW_DELAY

#define av_frame_alloc avcodec_alloc_frame
#endif

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(55, 39, 101)
typedef CodecID AVCodecID;
#endif

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(53, 35, 0)
static int avcodec_open2(AVCodecContext *avctx, AVCodec *codec, void * /*AVDictionary **options*/)
{
  return avcodec_open(avctx, codec);
}
#endif
#endif

bool vvVideo::global_init_done = false;

//----------------------------------------------------------------------------
/** Constructor
@param fr  framerate
@param min_q  lower bound for quantizer
@param max_q  upper bound for quantizer
@param br  target bitrate
@param max_k  maximum key interval
*/
vvVideo::vvVideo()
: framerate(25.f)
, min_quantizer(-1)
, max_quantizer(-1)
, bitrate(-1)
, max_key_interval(25)
, enc_ctx(NULL)
, dec_ctx(NULL)
, enc_picture(NULL)
, dec_picture(NULL)
, enc_imgconv_ctx(NULL)
, dec_imgconv_ctx(NULL)
, codec_id(-1)
, pixel_fmt(-1)
{
  if(!global_init_done)
  {
#if defined(VV_FFMPEG)
#if LIBAVCODEC_VERSION_MAJOR < 54
    avcodec_init();
#endif
    avcodec_register_all();
#endif
    global_init_done = true;
  }

  setCodec(VV_FLV1);
}

//----------------------------------------------------------------------------
/** Destructor
 */
vvVideo::~vvVideo()
{
  destroyEncoder();
  destroyDecoder();
}

void vvVideo::setCodec(vvVideo::Codec codec)
{
#if defined(VV_FFMPEG)
  pixel_fmt = AV_PIX_FMT_RGB24;
  switch(codec)
  {
  case VV_RAW:
    codec_id = AV_CODEC_ID_RAWVIDEO;
    break;
  case VV_FFV1:
    codec_id = AV_CODEC_ID_FFV1;
    break;
  case VV_JPEGLS:
    codec_id = AV_CODEC_ID_JPEGLS;
    break;
  case VV_MJPEG:
    codec_id = AV_CODEC_ID_MJPEG;
    break;
  case VV_MPEG2:
    codec_id = AV_CODEC_ID_MPEG2VIDEO;
    break;
  case VV_MPEG4:
    codec_id = AV_CODEC_ID_MPEG4;
    break;
  case VV_FLV1:
    codec_id = AV_CODEC_ID_FLV1;
    break;
  default:
    codec_id = AV_CODEC_ID_RAWVIDEO;
    break;
  }

  /* these did not work:
     codec_id = AV_CODEC_ID_THEORA;
     codec_id = AV_CODEC_ID_HUFFYUV;
     codec_id = AV_CODEC_ID_LJPEG;
     codec_id = AV_CODEC_ID_PNG;
     codec_id = AV_CODEC_ID_SVQ3;
     codec_id = AV_CODEC_ID_VP5;
     codec_id = AV_CODEC_ID_VP8;
     codec_id = AV_CODEC_ID_FFH264;
     codec_id = AV_CODEC_ID_H264;
     codec_id = AV_CODEC_ID_ROQ;
   */
#else
  (void)codec;
#endif
}

vvVideo::Codec vvVideo::getCodec() const
{
#if defined(VV_FFMPEG)
  switch(codec_id)
  {
  case AV_CODEC_ID_RAWVIDEO:
    return VV_RAW;
  case AV_CODEC_ID_FFV1:
    return VV_FFV1;
  case AV_CODEC_ID_JPEGLS:
    return VV_JPEGLS;
  case AV_CODEC_ID_MJPEG:
    return VV_MJPEG;
  case AV_CODEC_ID_MPEG2VIDEO:
    return VV_MPEG2;
  case AV_CODEC_ID_MPEG4:
    return VV_MPEG4;
  case AV_CODEC_ID_FLV1:
    return VV_FLV1;
  }
#endif

  return VV_RAW;
}

void vvVideo::setColorFormat(vvVideo::ColorFormat fmt)
{
#if defined(VV_FFMPEG)
  switch(fmt)
  {
  case VV_RGB24:
    pixel_fmt = AV_PIX_FMT_RGB24;
    break;
  case VV_YUV420P:
    pixel_fmt = AV_PIX_FMT_YUV420P;
    break;
  case VV_YUVJ420P:
    pixel_fmt = AV_PIX_FMT_YUVJ420P;
    break;
  default:
    vvDebugMsg::msg(0, "vvVideo::setColorFormat: unknown vvVideo color format ", fmt);
    assert("Missing support for vvVideo color format" == NULL);
    break;
  }
#else
  (void)fmt;
#endif
}

vvVideo::ColorFormat vvVideo::getColorFormat() const
{
#if defined(VV_FFMPEG)
  switch(pixel_fmt)
  {
  case AV_PIX_FMT_RGB24:
    return VV_RGB24;
  case AV_PIX_FMT_YUV420P:
    return VV_YUV420P;
  case AV_PIX_FMT_YUVJ420P:
    return VV_YUVJ420P;
  default:
    vvDebugMsg::msg(0, "vvVideo::getColorFormat: unknown avcodec pixel format ", pixel_fmt);
    assert("Missing support for avcodec pixel format" == NULL);
    break;
  }
#endif

  return VV_RGB24;
}

//----------------------------------------------------------------------------
/** Creates an XviD encoder
@param w  width of frames
@param h  height of frames
@return   0 for success, != 0 for error
*/
int vvVideo::createEncoder(int w, int h)
{
  vvDebugMsg::msg(3, "vvVideo::createEncoder, codec id is ", codec_id);
  destroyEncoder();

#if defined(VV_FFMPEG)
  AVCodec *encoder = avcodec_find_encoder((AVCodecID)codec_id);
  if(!encoder)
  {
    vvDebugMsg::msg(1, "Error: failed to find encoder for codec id ", codec_id);
    return -1;
  }

  AVPixelFormat fmt = AV_PIX_FMT_YUV420P;
  const AVPixelFormat *pix = encoder->pix_fmts;
  if(pix)
    fmt = *pix;
  while(pix && *pix != -1)
  {
    if(*pix == AV_PIX_FMT_RGB24)
    {
      fmt = *pix;
      break;
    }
    ++pix;
  }

  enc_ctx = avcodec_alloc_context3(NULL);
  if(!enc_ctx)
  {
    vvDebugMsg::msg(1, "Error: failed to allocate encoding context");
    return -1;
  }
  enc_ctx->width = w;
  enc_ctx->height = h;
  AVRational avr = {1, static_cast<int>(framerate) };
  enc_ctx->time_base= avr;
  enc_ctx->max_b_frames = 0;
  enc_ctx->pix_fmt = fmt;
  enc_ctx->flags |= AV_CODEC_FLAG_QSCALE;
  enc_ctx->gop_size = max_key_interval;

  if(bitrate > 0)
    enc_ctx->bit_rate = bitrate;
  if(min_quantizer > 0 && max_quantizer > 0)
  {
    enc_ctx->qmin = min_quantizer;
    enc_ctx->qmax = max_quantizer;
  }

  switch(codec_id)
  {
  case AV_CODEC_ID_MJPEG:
  case AV_CODEC_ID_MPEG1VIDEO:
  case AV_CODEC_ID_MPEG4:
  case AV_CODEC_ID_FLV1:
    break;
  case AV_CODEC_ID_MPEG2VIDEO:
  default:
    enc_ctx->flags |= AV_CODEC_FLAG_LOW_DELAY;
    break;
  }

  if(avcodec_open2(enc_ctx, encoder, NULL) < 0)
  {
    vvDebugMsg::msg(0, "Error: failed to open encoder");
    return -1;
  }

  enc_picture = av_frame_alloc();
  if(!enc_picture)
  {
    vvDebugMsg::msg(0, "Error: failed to allocate encoding picture");
    return -1;
  }
  avpicture_alloc((AVPicture *)enc_picture, enc_ctx->pix_fmt, w, h);

  pixel_fmt = enc_ctx->pix_fmt;

  enc_imgconv_ctx = sws_getContext(w, h,
      AV_PIX_FMT_RGB24,
      w, h, enc_ctx->pix_fmt, SWS_POINT,
      NULL, NULL, NULL);
  if(enc_imgconv_ctx == NULL) {
    vvDebugMsg::msg(1, "Error: cannot initialize the conversion context\n");
    return -1;
  }

  return 0;
#else
  (void)w;
  (void)h;
  return -1;
#endif
}

//----------------------------------------------------------------------------
/** Deletes the encoder
@return   0 for success, != 0 for error
*/
int vvVideo::destroyEncoder()
{
#if defined(VV_FFMPEG)
  if(enc_picture)
  {
    avpicture_free((AVPicture *)enc_picture);
    av_free(enc_picture);
    enc_picture = NULL;
  }

  if(enc_ctx)
  {
    avcodec_close(enc_ctx);
    av_free(enc_ctx);
    enc_ctx = NULL;
  }

  if(enc_imgconv_ctx)
  {
    sws_freeContext(enc_imgconv_ctx);
    enc_imgconv_ctx = NULL;
  }

  return 0;
#else
  return -1;
#endif
}

//----------------------------------------------------------------------------
/** Creates an XviD decoder
@param w  width of frames
@param h  height of frames
@return   0 for success, != 0 for error
*/
int vvVideo::createDecoder(int w, int h)
{
  vvDebugMsg::msg(3, "vvVideo::createDecoder, codec id is ", codec_id);
  destroyDecoder();

#if defined(VV_FFMPEG)
  AVCodec *decoder = avcodec_find_decoder((AVCodecID)codec_id);
  if(!decoder)
  {
    vvDebugMsg::msg(1, "error: failed to find decoder");
    return -1;
  }

  dec_ctx = avcodec_alloc_context3(NULL);
  if(!dec_ctx)
  {
    vvDebugMsg::msg(1, "error: failed to allocate decoding context");
    return -1;
  }
  dec_ctx->width = w;
  dec_ctx->height = h;
  dec_ctx->pix_fmt = (AVPixelFormat)pixel_fmt;

  if(avcodec_open2(dec_ctx, decoder, NULL) < 0)
  {
    vvDebugMsg::msg(0, "error: failed to open decoder");
    return -1;
  }

  dec_picture = av_frame_alloc();
  if(!dec_picture)
  {
    vvDebugMsg::msg(1, "error: failed to allocate decoding picture");
    return -1;
  }

  dec_imgconv_ctx = sws_getContext(w, h,
      dec_ctx->pix_fmt, 
      w, h, AV_PIX_FMT_RGB24, SWS_POINT,
      NULL, NULL, NULL);
  if(dec_imgconv_ctx == NULL) {
    vvDebugMsg::msg(1, "Error: cannot initialize the conversion context\n");
    return -1;
  }
  return 0;
#else
  (void)w;
  (void)h;
  return -1;
#endif
}

//----------------------------------------------------------------------------
/** Deletes the decoder
@return   0 for success, != 0 for error
*/
int vvVideo::destroyDecoder()
{
#if defined(VV_FFMPEG)
  if(dec_picture)
  {
    av_free(dec_picture);
    dec_picture = NULL;
  }

  if(dec_ctx)
  {
    avcodec_close(dec_ctx);
    free(dec_ctx);
    dec_ctx = NULL;
  }

  if(dec_imgconv_ctx)
  {
    sws_freeContext(dec_imgconv_ctx);
    dec_imgconv_ctx = NULL;
  }

  return 0;
#else
  return -1;
#endif
}

//----------------------------------------------------------------------------
/** Encodes a frame
@param src  pointer to frame to encode
@param dst  pointer to destination
@param enc_size  IN, available space for encoded frame
@param enc_size  OUT, size of encoded frame
@param style  style of encoding [0 .. 6]
@param quant  quantizer value to use for that frame [1 .. 31]
@return   0 for success, != 0 for error
*/
int vvVideo::encodeFrame(const unsigned char* src, unsigned char* dst, int* enc_size)
{
  vvDebugMsg::msg(3, "vvVideo::encodeFrame");
#if defined(VV_FFMPEG)
  if(codec_id != enc_ctx->codec_id)
  {
    vvDebugMsg::msg(2, "vvVideo::encodeFrame: changed codec id to ", codec_id);
    createEncoder(enc_ctx->width, enc_ctx->height);
  }
  *(uint8_t *)dst = getCodec();
  dst++;
  *(uint8_t *)dst = getColorFormat();
  dst++;
  AVPicture src_picture;
  avpicture_fill(&src_picture, (uint8_t *)src, AV_PIX_FMT_RGB24, enc_ctx->width, enc_ctx->height);

  sws_scale(enc_imgconv_ctx, src_picture.data, src_picture.linesize, 0,
      enc_ctx->height, enc_picture->data, enc_picture->linesize);

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 24, 102)
  *enc_size = avcodec_encode_video(enc_ctx, dst, *enc_size, enc_picture);
#else
  AVPacket pkt;
  av_init_packet(&pkt);
  pkt.data = (uint8_t*)dst;
  pkt.size = *enc_size;
  int got_output=0;
  int ret = avcodec_encode_video2(enc_ctx, &pkt, enc_picture, &got_output);
  if (ret < 0)
  {
    vvDebugMsg::msg(1, "vvVideo::encodeFrame: encoding failed");
    return -1;
  }
  *enc_size = pkt.size;
#endif
  if(*enc_size == -1)
  {
    vvDebugMsg::msg(1, "vvVideo::encodeFrame: encoding failed");
    return -1;
  }
  *enc_size += 2; // for codec & color format

  vvDebugMsg::msg(3, "vvVideo::encodeFrame, encoded size is ", *enc_size);

  return 0;
#else
  (void)src;
  (void)dst;
  (void)enc_size;
  return -1;
#endif
}

//----------------------------------------------------------------------------
/** Decodes a frame
@param src  pointer to encoded frame
@param dst  pointer to destination
@param src_size  size of encoded frame
@param dst_size  IN, available space for decoded frame
@param dst_size  OUT, size of decoded frame
@return   0 for success, != 0 for error
*/
int vvVideo::decodeFrame(const unsigned char* src, unsigned char* dst, int src_size, int* dst_size)
{
  vvDebugMsg::msg(3, "vvVideo::decodeFrame, encoded size is ", src_size);
#if defined(VV_FFMPEG)
  int got_picture = 0;
  setCodec((Codec)*(uint8_t *)src);
  src++;
  src_size--;
  setColorFormat((ColorFormat)*(uint8_t *)src);
  src++;
  src_size--;
  if(codec_id != dec_ctx->codec_id || pixel_fmt != dec_ctx->pix_fmt)
  {
    vvDebugMsg::msg(2, "vvVideo::decodeFrame: changed codec id to ", codec_id);
    createDecoder(dec_ctx->width, dec_ctx->height);
  }
#if LIBAVCODEC_VERSION_MAJOR > 52 || (LIBAVCODEC_VERSION_MAJOR==52 && LIBAVCODEC_VERSION_MINOR>120)
  AVPacket pkt;
  av_init_packet(&pkt);
  pkt.data = (uint8_t*)src;
  pkt.size = src_size;
  *dst_size = avcodec_decode_video2(dec_ctx, dec_picture, &got_picture, &pkt);
#else
  *dst_size = avcodec_decode_video(dec_ctx, dec_picture, &got_picture, src, src_size);
#endif

  if(!got_picture)
  {
    vvDebugMsg::msg(1, "vvVideo::decodeFrame: no picture, encoded size was ", src_size);
    return -1;
  }

  AVPicture dst_picture;
  avpicture_fill(&dst_picture, dst, AV_PIX_FMT_RGB24, dec_ctx->width, dec_ctx->height);
  int ret = sws_scale(dec_imgconv_ctx, dec_picture->data, dec_picture->linesize, 0, 
      dec_ctx->height, dst_picture.data, dst_picture.linesize);
  (void)ret;

  *dst_size = dec_ctx->width * dec_ctx->height * 3;

  return 0;
#else
  (void)src;
  (void)dst;
  (void)src_size;
  (void)dst_size;
  return -1;
#endif
}

//----------------------------------------------------------------------------
/** Sets the target framerate
@param fr  framerate
*/
void vvVideo::setFramerate(float fr)
{
  framerate = fr;
}

//----------------------------------------------------------------------------
/** Sets the quantizer bounds
@param min_q  lower quantizer bound
@param max_q  upper quantizer bound
*/
void vvVideo::setQuantizer(int min_q, int max_q)
{
  min_quantizer = min_q;
  max_quantizer = max_q;
}

//----------------------------------------------------------------------------
/** Sets the target bitrate
@param br  bitrate
*/
void vvVideo::setBitrate(int br)
{
  bitrate = br;
}

//----------------------------------------------------------------------------
/** Sets the maximum interval for key frames
@param max_k maximum interval in frames
*/
void vvVideo::setMaxKeyInterval(int max_k)
{
  max_key_interval = max_k;
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
