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
 *	File			RemoteAR.cpp 				*
 *									*
 *	Description		RemoteAR optical tracking system interface class				*
 *									*
 *	Author			Uwe Woessner				*
 *									*
 *	Date			July 2002				*
 *									*
 *	Status			in dev					*
 *									*
 *****************/

#include <util/common.h>
#include <string>
#include <iostream>
using std::string;

#ifdef _WIN32
#include <windows.h>
#endif
#include "RemoteARP.h"
#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
//#include <util/coTimer.h>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/Texture2D>
#include <osg/Geometry>
#include <osg/TexEnv>
#include <osg/Texture2D>
#include <cover/coVRCommunication.h>
#include <cover/coVRMSController.h>
#include <net/message_types.h>
#include "dxtc2.h"

//#undef HAVE_CUDA

#if defined(VV_FFMPEG) || defined(VV_XVID)
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
#endif

#include <config/CoviseConfig.h>

#ifdef __linux__
#include <asm/ioctls.h>
#define sigset signal

#endif
#ifndef _WIN32
#include <sys/ipc.h>
#endif

RemoteAR *RemoteAR::remoteAR = 0;
RemoteVideo::RemoteVideo(int id)
{
    remoteID = id;
    videoTex = new osg::Texture2D;
    videoTex->setImage(new osg::Image());
    xsize = -1;
    ysize = -1;
    createGeode();
    video = new osg::MatrixTransform();
    video->addChild(geometryNode);
    char buf[100];
    sprintf(buf, "RemoteVideo_%d", id);
    video->setName(buf);
    cover->getObjectsRoot()->addChild(video);
}

RemoteVideo::~RemoteVideo()
{
}

void RemoteVideo::update(const char *image, osg::Matrix &mat)
{

    video->setMatrix(mat);
    float xt, yt;
    xt = ((float)RemoteAR::remoteAR->vp.xsize) / ((float)RemoteAR::remoteAR->getAllocXSize());
    yt = ((float)RemoteAR::remoteAR->vp.ysize) / ((float)RemoteAR::remoteAR->getAllocYSize());
    (*texcoord)[0].set(0.0, yt);
    (*texcoord)[1].set(xt, yt);
    (*texcoord)[2].set(xt, 0.0);
    (*texcoord)[3].set(0.0, 0.0);

    /*cerr << "vp.x:" << RemoteAR::remoteAR->vp.xsize << " vp.y " << RemoteAR::remoteAR->vp.ysize << endl;
     cerr << "xt = " << xt<< "yt =" << yt << endl;
     texcoord[0].set(0.0,0.0);
     texcoord[1].set(1.0,0.0);
     texcoord[2].set(1.0,1.0);
     texcoord[3].set(0.0,1.0);
     */
    videoTex->getImage()->setImage((*RemoteAR::remoteAR).getAllocXSize(), (*RemoteAR::remoteAR).getAllocYSize(), 1,
                                   GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, (unsigned char *)image,
                                   osg::Image::NO_DELETE);
    videoTex->getImage()->dirty();
}

RemoteAR::RemoteAR()
{
    remoteAR = this;

    sendVideo = false;
    style = 0;
    quant = 0;
    encSize = 0;
    key = 0;
    textureBuffer = 0;
    imageBuffer = 0;
    irmos_client = NULL;
    irmos_host = NULL;

//! CUDA COMPRESSION //
#if defined(HAVE_CUDA)
    if (coCoviseConfig::isOn("COVER.Plugin.ARToolKit.RemoteAR.Transmit", false) && (MarkerTracking::instance()->isRunning()))
    {

        cout << "Init CUDA!" << endl;
        init_cuda();

        style = coCoviseConfig::getInt("COVER.Plugin.ARToolKit.RemoteAR.EncodingStyle", 0);
        quant = coCoviseConfig::getInt("COVER.Plugin.ARToolKit.RemoteAR.EncodingQuant", 1);

        sendVideo = true;
        sendBuffer = new unsigned char[MarkerTracking::instance()->videoWidth * MarkerTracking::instance()->videoHeight * 3 + sizeof(VideoParameters)];
        sendVideoBuffer = sendBuffer + sizeof(VideoParameters);
        sendVP = (VideoParameters *)sendBuffer;
    }
#endif
//! CUDA COMPRESSION //

#if defined(VV_FFMPEG) || defined(VV_XVID)
#define INBUF_SIZE 4096
    encContext = NULL;

    // init required ffmpeg stuff
    avcodec_init();
    avcodec_register_all();

    // we send an mpeg video
    codecID = CODEC_ID_RAWVIDEO;

    // set internal pixel format for encoding/decoding
    pix_fmt = PIX_FMT_YUV420P;

    // set group of picture size in video
    gop_size = 10;

    // set max number of B-frames in video
    max_b_frames = 1;
    ffmpeg_startup = true;

    fprintf(stderr, "isRunning %d width %d height %d\n", MarkerTracking::instance()->isRunning(), MarkerTracking::instance()->videoWidth, MarkerTracking::instance()->videoHeight);
    sendVideo = coCoviseConfig::isOn("COVER.Plugin.ARToolKitPlus.RemoteAR.Transmit", false);
    irmosReceiver = coCoviseConfig::isOn("COVER.Plugin.ARToolKitPlus.RemoteAR.irmosReceiver", false);
    useIRMOS = coCoviseConfig::isOn("COVER.Plugin.ARToolKitPlus.RemoteAR.UseIRMOS", false);
    std::cerr << "SendVideo: " << sendVideo << " UseIRMOS. " << useIRMOS << " IrmosReceiver: " << irmosReceiver << std::endl;
    bool instRuns = (MarkerTracking::instance()->isRunning());
    if ((sendVideo || (useIRMOS && irmosReceiver)) && instRuns)
    {
        //Configure RemoteAR sending client

        style = coCoviseConfig::getInt("COVER.Plugin.ARToolKitPlus.RemoteAR.EncodingStyle", 0);
        quant = coCoviseConfig::getInt("COVER.Plugin.ARToolKitPlus.RemoteAR.EncodingQuant", 1);
        std::cerr << "RemoteAR::RemoteAR(): Sending RemoteAR triggered!" << std::endl;
        sendVideo = true;
        sendBuffer = new unsigned char[MarkerTracking::instance()->videoWidth * MarkerTracking::instance()->videoHeight * 3 + sizeof(VideoParameters)];
        sendVideoBuffer = sendBuffer + sizeof(VideoParameters);
        sendVP = (VideoParameters *)sendBuffer;
    }

    if (useIRMOS)
    {
        //Configure client for IRMOS usage
        //Valid for sending and receiving
        //Otherwise(NoIRMOS) everything is sent through VRB
        std::cerr << "RemoteAR::RemoteAR(): Using IRMOS!" << std::endl;
        int recPort = 0;
        std::string recServer = "";

        recPort = coCoviseConfig::getInt("COVER.Plugin.ARToolKitPlus.RemoteAR.irmosServerPort", 31332);
        recServer = coCoviseConfig::getEntry("value", "COVER.Plugin.ARToolKitPlus.RemoteAR.irmosServer", "127.0.0.1");

        if (irmosReceiver)
        {
            std::cerr << "RemoteAR::RemoteAR(): Configuring receiver!" << std::endl;
            sendVideo = false;
            sendVideo = false;
            recPort = coCoviseConfig::getInt("COVER.Plugin.ARToolKitPlus.RemoteAR.irmosReceiverPort", 31666);
            recServer = coCoviseConfig::getEntry("value", "COVER.Plugin.ARToolKitPlus.RemoteAR.irmosReceiverServer", "141.58.8.10");
        }
        irmos_host = new Host(recServer.c_str(), true);
        irmos_client = new ClientConnection(irmos_host, recPort, 0, (covise::sender_type)0);
        //irmos_client->get_dataformat();
        cerr << "RemoteAR::RemoteAR(): Status of Client: " << irmos_client->is_connected() << endl;
    }

#endif
    setAllocXSize(-1);
    setAllocYSize(-1);
    imageBuffer = NULL;
}

RemoteAR::~RemoteAR()
{

//! CUDA COMPRESSION //
#if defined(HAVE_CUDA)
    if (coCoviseConfig::isOn("COVER.Plugin.ARToolKit.RemoteAR.Transmit", false) && (MarkerTracking::instance()->isRunning()))
    {
        close_cuda();
    }
#endif
    //!##################//

    if (useIRMOS)
    {
        delete irmos_client;
        delete irmos_host;
        irmos_client = NULL;
        irmos_host = NULL;
    }

#if defined(VV_FFMPEG) || defined(VV_XVID)
    if (encContext != NULL)
    {
        avcodec_close(encContext);
        av_free(encoder);
        av_free(encPicture);
        av_free(yuvPicture);
        free(outbuf);
        outbuf = 0;
        free(yuvBuffer);
        yuvBuffer = 0;

        avcodec_close(decContext);
        av_free(decoder);
        av_free(decPicture);
        av_free(rgbPicture);
        free(inbuf);
        inbuf = 0;
        free(rgbBuffer);
        rgbBuffer = 0;
    }
#endif

    delete textureBuffer;
    textureBuffer = NULL;
    delete imageBuffer;
    imageBuffer = NULL;
}

int RemoteAR::nextPT(int n)
{
    int p = 1;
    for (int i = 0; i < 32; ++i)
    {
        if (n < p)
            return p;
        p = p << 1;
    }
    return 0;
}

void RemoteAR::receiveImage(const char *data)
{
    std::cerr << "Receiving image ... !" << std::endl;
#if !defined(VV_FFMPEG) && !defined(VV_XVID) && !defined(HAVE_CUDA)
    (void)data;

#else
    memcpy(&vp, data, sizeof(VideoParameters));
    unsigned char *compressedVideoData = (unsigned char *)data + sizeof(VideoParameters);

    if (vp.xsize * vp.ysize > getAllocXSize() * getAllocYSize())
    {
        setAllocXSize(nextPT(vp.xsize));
        setAllocYSize(nextPT(vp.ysize));
        delete[] textureBuffer;
        if (imageBuffer != textureBuffer)
            delete[] imageBuffer;
        textureBuffer = new unsigned char[getAllocXSize() * getAllocYSize() * 3];
        if (vp.xsize != getAllocXSize())
            imageBuffer = new unsigned char[vp.xsize * vp.ysize * 3];
        else
            imageBuffer = textureBuffer;
    }

#if defined(HAVE_CUDA)
    //! CUDA DECOMPRESSION //
    cudaDecompression(vp.xsize, vp.ysize, compressedVideoData, imageBuffer, MarkerTracking::instance()->videoMode);
//!##################//

#else

    // init decoder on startup
    if (ffmpeg_startup)
    {
        initDecoder();
    }

    // decode the image
    std::cerr << "Receiving image ... decoding !" << std::endl;
    avcodec_decode_video(decContext, decPicture, &got_picture, compressedVideoData, vp.encSize);

    if (!got_picture)
    {
        cerr << "RemoteAR::receiveImage -> could not decode image" << endl;
        return;
    }

    // convert the image from global pixel format to RGB24
    SwsContext *xformContext = sws_getContext(decContext->width, decContext->height, pix_fmt, decContext->width, decContext->height, PIX_FMT_RGB24, SWS_BICUBIC, NULL, NULL, NULL);
    sws_scale(xformContext, decPicture->data, decPicture->linesize, 0, decContext->height, rgbPicture->data, rgbPicture->linesize);
    //  avcodec_img_convert ((AVPicture *)rgbPicture, PIX_FMT_RGB24, (AVPicture *) decPicture, pix_fmt, decContext->width, decContext->height);

    // copy decoded image to imageBuffer
    //memcpy (imageBuffer, rgbPicture->data[0], vp.xsize * vp.ysize * 3);
    imageBuffer = rgbPicture->data[0];
#endif

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
        {
            viewerMat(i, j) = vp.matrix[i][j];
        }

    if (vp.xsize != getAllocXSize())
    {
        //align image to powers of two
        if (vp.format == GL_RGB)
        {
            for (int i = 0; i < vp.ysize; i++)
            {
                memcpy(textureBuffer + (getAllocXSize() * i * 3), imageBuffer + (vp.xsize * i * 3), vp.xsize * 3);
            }
        }
        else
        {
            for (int i = 0; i < vp.ysize; i++)
            {
                for (int j = 0; j < vp.xsize; j++)
                {
                    int indexTexture = getAllocXSize() * i * 3 + j * 3;
                    int indexImage = vp.xsize * i * 3 + j * 3;
                    textureBuffer[indexTexture + 2] = imageBuffer[indexImage];
                    textureBuffer[indexTexture + 1] = imageBuffer[indexImage + 1];
                    textureBuffer[indexTexture] = imageBuffer[indexImage + 2];
                }
            }
        }
    }
    else
    {
    }

    remoteVideos.reset();
    while (remoteVideos.current())
    {
        if (remoteVideos.current()->getRemoteID() == vp.ID)
        {
            remoteVideos.current()->update((const char *)textureBuffer, viewerMat);
            return;
        }
        remoteVideos.next();
    }
    RemoteVideo *rv = NULL;
    cerr << "new RemoteVideo " << vp.ID << endl;
    rv = new RemoteVideo(vp.ID);
    if (rv)
    {
        rv->update((const char *)textureBuffer, viewerMat);
        remoteVideos.append(rv);
    }
#endif
    std::cerr << "Receiving image ... done !" << std::endl;
}

/** create Performer geometry and texture objects
 */
void RemoteVideo::createGeode()
{

    osg::Vec3Array *coord = new osg::Vec3Array(4);
    (*coord)[0].set(-0.5, 0, -0.5);
    (*coord)[1].set(0.5, 0, -0.5);
    (*coord)[2].set(0.5, 0, 0.5);
    (*coord)[3].set(-0.5, 0, 0.5);

    texcoord = new osg::Vec2Array(4);

    (*texcoord)[0].set(0.0, 0.0);
    (*texcoord)[1].set(1.0, 0.0);
    (*texcoord)[2].set(1.0, 1.0);
    (*texcoord)[3].set(0.0, 1.0);

    osg::Vec4Array *color = new osg::Vec4Array(1);
    osg::Vec3Array *normal = new osg::Vec3Array(1);
    (*color)[0].set(1, 1, 1, 1.0f);
    (*normal)[0].set(0.0f, -1.0f, 0.0f);

    osg::Material *material = new osg::Material();

    material->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 1.0f));
    material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    material->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    material->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    material->setAlpha(osg::Material::FRONT_AND_BACK, 1.0f);

    osg::TexEnv *texEnv = new osg::TexEnv();
    texEnv->setMode(osg::TexEnv::MODULATE);

    geometryNode = new osg::Geode();
    osg::Geometry *geometry = new osg::Geometry();

    ushort *vertices = new ushort[4];
    vertices[0] = 0;
    vertices[1] = 1;
    vertices[2] = 2;
    vertices[3] = 3;
    osg::DrawElementsUShort *plane = new osg::DrawElementsUShort(osg::PrimitiveSet::QUADS, 4, vertices);

    geometry->setVertexArray(coord);
    geometry->addPrimitiveSet(plane);
    geometry->setColorArray(color);
    geometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    geometry->setNormalArray(normal);
    geometry->setNormalBinding(osg::Geometry::BIND_OVERALL);
    geometry->setTexCoordArray(0, texcoord);

    osg::StateSet *stateSet = new osg::StateSet();

    stateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    stateSet->setAttributeAndModes(material, osg::StateAttribute::ON);
    stateSet->setTextureAttributeAndModes(0, videoTex, osg::StateAttribute::ON);
    stateSet->setTextureAttribute(0, texEnv);
    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);

    geometryNode->setStateSet(stateSet);

    geometryNode->addDrawable(geometry);
}

void RemoteAR::update()
{
#if defined(VV_FFMPEG) || defined(VV_XVID) || defined(HAVE_CUDA)

    if (MarkerTracking::instance()->isRunning() && sendVideo)
    {

        sendVP->format = MarkerTracking::instance()->videoMode;

        if (MarkerTracking::instance()->videoData == 0)
        {
            cerr << "RemoteAR:update -- no videoData from MarkerTracking::instance()" << endl;
            return;
        }

#if defined(HAVE_CUDA)
        //! CUDA COMPRESSION //
        encSize = MarkerTracking::instance()->videoWidth * MarkerTracking::instance()->videoHeight / 2;
        cudaCompression(MarkerTracking::instance()->videoWidth, MarkerTracking::instance()->videoHeight, MarkerTracking::instance()->videoData, sendVideoBuffer, MarkerTracking::instance()->videoMode);
//!##################//
#else
        // init encoder on startup
        if (ffmpeg_startup)
        {
            initEncoder();
        }

        encPicture->linesize[0] = encContext->width * 3;
        encPicture->data[0] = (uint8_t *)MarkerTracking::instance()->videoData;

        // convert the image from RGB24 to global pixel format
        //img_convert ((AVPicture *)yuvPicture, pix_fmt, (AVPicture *) encPicture, PIX_FMT_RGB24, encContext->width, encContext->height);
        SwsContext *xformContext = sws_getContext(encContext->width, encContext->height, PIX_FMT_RGB24, encContext->width, encContext->height, pix_fmt, SWS_BICUBIC, NULL, NULL, NULL);
        sws_scale(xformContext, encPicture->data, encPicture->linesize, 0, encContext->height, yuvPicture->data, yuvPicture->linesize);

        // encode the image
        encSize = avcodec_encode_video(encContext, sendVideoBuffer, outbuf_size, yuvPicture);
//sendVideoBuffer = yuvPicture->data;
//memcpy(sendVideoBuffer, yuvPicture->data, outbuf_size);

#endif

        sendVP->xsize = MarkerTracking::instance()->videoWidth;
        sendVP->ysize = MarkerTracking::instance()->videoHeight;
        osg::Matrix worldPlane; // the initial video plane has y normal and size 1x1 and is in the origin
        osg::Matrix objectPlane; // the same but now in object coords
        int i, j;
        // we need the Transformation of the picture plane in object coordinates
        // we put the plane behind the object bounding sphere
        osg::BoundingSphere bsphere = cover->getObjectsRoot()->getBound();

        float radiusInWorld = bsphere.radius() * cover->getScale();
        osg::Vec3 positionInWorld = bsphere.center() * cover->getBaseMat();
        float planeDist = positionInWorld[1] + radiusInWorld;
        osg::Vec3 viewerPos = cover->getViewerMat().getTrans();
        float scalePlane = ((-viewerPos[1]) + planeDist) / (-viewerPos[1]);
        worldPlane = osg::Matrix::scale(coVRConfig::instance()->screens[0].hsize * scalePlane, 1, coVRConfig::instance()->screens[0].vsize * scalePlane);
        worldPlane *= osg::Matrix::translate(0, planeDist, 0);
        objectPlane = worldPlane * cover->getInvBaseMat();
        for (i = 0; i < 4; i++)
            for (j = 0; j < 4; j++)
            {
                sendVP->matrix[i][j] = objectPlane(i, j);
            }
        sendVP->screenWidth = coVRConfig::instance()->screens[0].hsize;
        sendVP->screenHeight = coVRConfig::instance()->screens[0].vsize;
        sendVP->encSize = encSize;
        sendVP->ID = coVRCommunication::instance()->getID();

        // transmit encoded image
        this->sendBinARMessage("AR_VIDEO_FRAME", (char *)sendBuffer, encSize + sizeof(VideoParameters));
    }
#endif
}

// initialization of the ffmpeg encoder
bool RemoteAR::initEncoder()
{
#if defined(VV_FFMPEG) || defined(VV_XVID)
    outbuf_size = MarkerTracking::instance()->videoWidth * MarkerTracking::instance()->videoHeight * 3;
    outbuf = new uint8_t[outbuf_size];
    encoder = avcodec_find_encoder(codecID);
    if (!encoder)
    {
        cerr << "RemoteAR::init_encoder -> error initialising encoder " << codecID << endl;
        return false;
    }
    encContext = avcodec_alloc_context();
    if (!encContext)
    {
        cerr << "RemoteAR::init_encoder -> could not allocate encContext" << endl;
        return false;
    }
    encContext->width = MarkerTracking::instance()->videoWidth;
    encContext->height = MarkerTracking::instance()->videoHeight;
    encContext->time_base.num = 1;
    encContext->time_base.den = 25;
    encContext->gop_size = gop_size;
    encContext->max_b_frames = max_b_frames;
    encContext->pix_fmt = pix_fmt;
    encContext->bit_rate = 2000;

    if (avcodec_open(encContext, encoder) < 0)
    {
        cerr << "RemoteAR::init_encoder -> could not open encoder" << endl;
        return false;
    }
    encPicture = avcodec_alloc_frame();
    if (!encPicture)
    {
        cerr << "RemoteAR::init_encoder -> could not allocate encPicture" << endl;
        return false;
    }

    yuvPicture = avcodec_alloc_frame();
    if (yuvPicture == 0)
    {
        cerr << "RemoteAR::init_encoder -> could not allocate yuvPicture" << endl;
        return false;
    }
    // Determine required buffer size and allocate buffer
    int numBytes = avpicture_get_size(PIX_FMT_YUV420P, encContext->width, encContext->height);
    yuvBuffer = new uint8_t[numBytes];

    // prepare 'encPicture'. 'yuvPicture'
    avpicture_fill((AVPicture *)encPicture, outbuf, PIX_FMT_YUV420P, encContext->width, encContext->height);
    avpicture_fill((AVPicture *)yuvPicture, yuvBuffer, PIX_FMT_YUV420P, encContext->width, encContext->height);

    ffmpeg_startup = false;
    return true;
}

// initialization of the ffmpeg decoder
bool RemoteAR::initDecoder()
{
    inbuf = new uint8_t[INBUF_SIZE + FF_INPUT_BUFFER_PADDING_SIZE];

    /* set end of buffer to 0 (this ensures that no overreading happens for damaged mpeg streams) */
    memset(inbuf + INBUF_SIZE, 0, FF_INPUT_BUFFER_PADDING_SIZE);

    decoder = avcodec_find_decoder(codecID);
    if (!decoder)
    {
        cerr << "RemoteAR::init_decoder -> error initialising decoder" << endl;
        return false;
    }
    decContext = avcodec_alloc_context();
    if (!decContext)
    {
        cerr << "RemoteAR::init_decoder -> could not allocate decContext" << endl;
        return false;
    }
    decContext->width = vp.xsize;
    decContext->height = vp.ysize;
    cerr << "RemoteAR::init_decoder(): x= " << vp.xsize << " y= " << vp.ysize << std::endl;
    decContext->time_base.num = 1;
    decContext->time_base.den = 25;
    decContext->gop_size = gop_size; /* emit one intra frame every ten frames */
    decContext->max_b_frames = max_b_frames;
    decContext->pix_fmt = pix_fmt;
    int av_err = 0;
    if ((av_err = avcodec_open(decContext, decoder)) < 0)
    {
        cerr << "RemoteAR::init_decoder -> could not open decoder! Error:" << av_err << endl;
        return false;
    }
    decPicture = avcodec_alloc_frame();
    if (!decPicture)
    {
        cerr << "RemoteAR::init_decoder -> could not allocate decPicture" << endl;
        return false;
    }

    rgbPicture = avcodec_alloc_frame();
    if (rgbPicture == 0)
    {
        cerr << "RemoteAR::init_decoder -> could not allocate rgbPicture" << endl;
        return false;
    }

    // Determine required buffer size and allocate buffer
    int numBytes = avpicture_get_size(PIX_FMT_RGB24, decContext->width, decContext->height);
    rgbBuffer = new uint8_t[numBytes];

    // prepare 'rgbPicture'
    avpicture_fill((AVPicture *)rgbPicture, rgbBuffer, PIX_FMT_RGB24, decContext->width, decContext->height);

    ffmpeg_startup = false;
    return true;
#else
    return false;
#endif
}

int RemoteAR::sendBinARMessage(const char *keyword, const char *data, int len)
{
    Message *message = NULL;
    int size = 0;

    if (useIRMOS) // && (debugOneTimeSend !=0))
    {
        //Conatct IRMOS ASC here
        //Check if we are master first, cause we only send from master
        if (!coVRMSController::instance()->isSlave())
        {
            //Create Covise message
            message = new Message();

            size = strlen(keyword) + 2;
            size += len;

            message->data = new char[size];
            message->data[0] = 0;
            strcpy(&message->data[1], keyword);
            memcpy(&message->data[strlen(keyword) + 2], data, len);

            message->type = COVISE_MESSAGE_RENDER;
            message->length = size;

            //Send to ASC here
            //Create client connection
            //Use sendMessage
            //vrbc->sendMessage(message);
            if (irmos_client->is_connected())
            {
                irmos_client->sendMessage(message);
            }
            //irmos_client->send("OK\0",3);
            //#ifdef   _DEBUG
            //         std::cerr << "Send COVISE AR-message" << std::endl;
            //         std::cerr << "Size: " << message->length << std::endl;
            //         std::cerr << "Type: " << message->data[1] << std::endl;
            //         std::cerr << "Connection status: " << irmos_client->is_connected() << std::endl;
            //#endif
            delete[] message -> data;
            message->data = NULL;
            delete message;
        }
    }
    else
    {
        //Call cover->sendBinMessage here (using VRB && Controller)
        //#ifdef _DEBUG
        //      std::cerr << "Send COVISE AR-message" << std::endl;
        //      std::cerr << "Size: " << len << std::endl;
        //      std::cerr << "Type: " << data << std::endl;
        //#endif
        if (cover->isVRBconnected())
        {
            return cover->sendBinMessage(keyword, data, len);
        }
    }
    return 1;
}

void RemoteAR::updateBitrate(const int
#if defined(VV_FFMPEG) || defined(VV_XVID)
                                 bitrate
#endif
                             )
{
#if defined(VV_FFMPEG) || defined(VV_XVID)
    encContext->bit_rate = bitrate;
    encContext->bit_rate_tolerance = (int)bitrate * 0.1f;
#endif
}

bool RemoteAR::usesIRMOS() const
{
    return useIRMOS;
}

bool RemoteAR::isReceiver() const
{
    return irmosReceiver;
}

ClientConnection *RemoteAR::getIRMOSClient() const
{
    return irmos_client;
}
