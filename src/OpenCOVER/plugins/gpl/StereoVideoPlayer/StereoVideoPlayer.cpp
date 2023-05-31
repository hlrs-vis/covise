/****************************************************************************\
**                                                            (C)2001 HLRS  **
**                                                                          **
** Description: PLMXML Plugin (loads PLMXML documents)                      **
**                                                                          **
**                                                                          **
** Author: U.Woessner                                                       **
**                                                                          **
** History:  		         		                            **
** Nov-01  v1	    				       		            **
**                                                                          **
\****************************************************************************/
#define __STDC_CONSTANT_MACROS

#include "StereoVideoPlayer.h"

#include <cover/coVRFileManager.h>
#include <cover/coVRMSController.h>
#include <cover/VRSceneGraph.h>
#include <cover/VRViewer.h>
#include <config/CoviseConfig.h>
#include <cover/coVRTui.h>

#include <osg/Group>
#include <osg/Geometry>
#include <osg/Texture>
#include <osg/StateSet>
#include <osg/Image>
#include <osgDB/ReadFile>

using namespace osg;

StereoVideoPlayerPlugin *StereoVideoPlayerPlugin::plugin = NULL;

static FileHandler handlers[] = {
    { NULL,
      StereoVideoPlayerPlugin::loadMovie,
      StereoVideoPlayerPlugin::unloadMovie,
      "mp4" },
    { NULL,
      StereoVideoPlayerPlugin::loadMovie,
      StereoVideoPlayerPlugin::unloadMovie,
      "mpg" },
    { NULL,
      StereoVideoPlayerPlugin::loadMovie,
      StereoVideoPlayerPlugin::unloadMovie,
      "wmv" },
    { NULL,
      StereoVideoPlayerPlugin::loadMovie,
      StereoVideoPlayerPlugin::unloadMovie,
      "avi" },
    { NULL,
      StereoVideoPlayerPlugin::loadMovie,
      StereoVideoPlayerPlugin::unloadMovie,
      "mov" }
};

bool StereoVideoPlayerPlugin::parseFilename(string *name)
{
    int hostIndex;
    string hostName, filename;

    if (coVRMSController::instance()->isMaster())
        hostIndex = 0;
    else
        hostIndex = coVRMSController::instance()->getID();
    char buffer[3];
    sprintf(buffer, "%d", hostIndex);
    hostName = buffer;

    string stereoEye = "";
    if (coVRConfig::instance()->channels[0].stereoMode == osg::DisplaySettings::RIGHT_EYE)
        stereoEye = "RIGHT";
    else if (coVRConfig::instance()->channels[0].stereoMode == osg::DisplaySettings::LEFT_EYE)
        stereoEye = "LEFT";

    string suffix = "";
    size_t found = name->find(".");
    if (found != string::npos)
    {
        filename = name->substr(0, found);
        suffix = name->substr(found);
    }
    else
        filename = *name;

    filename = filename + "-" + hostName + stereoEye + suffix;
    ifstream ifs(filename.c_str(), ifstream::in);
    if (!ifs.fail())
    {
        ifs.close();
        *name = filename;
        errorLabel->setLabel("");

        return true;
    }

    string label = filename + " not found on host " + hostName + ". \n\n If [name].[extension] (e.g. myvideo.avi) is used as a parameter, \n the files on the master and the slaves have to use the following filename convention:\n\n [name]-[hostID][stereoMode].[extension]\n\n (e.g. myvideo-0RIGHT.avi).\n The master has the hostID 0, the stereoMode is defined in the config file.\n";
    errorLabel->setLabel(label);

    return false;
}

bool StereoVideoPlayerPlugin::openMovie(const string filename, MovieData *movieDat)
{

    VideoStream *vStream = movieDat->getStream();
    AVPixelFormat pixFormat = AV_PIX_FMT_RGB24;
    if (!vStream->openMovieCodec(filename, &pixFormat))
        return false;
    movieDat->setGLFormat(pixFormat);

#ifdef HAVE_SDL
    if (vStream->getAudioPlayback() && (vStream->getAudioStreamID() != -1))
    {
        fprintf(stderr, "openSDL\n");

        if (!videoPlayer->openSDL(vStream))
            vStream->setAudioPlayback(false);
    }
#endif

    if (!vStream->allocateFrame())
        return false;
    if (!vStream->allocateRGBFrame(pixFormat))
        return false;

    return true;
}

int StereoVideoPlayerPlugin::loadMovie(const char *filename, osg::Group *parent, const char *)
{
    plugin->addMovieEnvironment(filename);

    return 0;
}

bool StereoVideoPlayerPlugin::addMovieEnvironment(string filename)
{

    if (plugin->parseFilename(&filename))
    {
        movie = new MovieData(this);

        if (openMovie(filename, movie))
        {
            if ((imgPerFrame = videoPlayer->getFrame(movie->getStream(), movie->getImage(), true, movie->getGLFormat())) < 0)
            {
                fprintf(stderr, "Can not read frame\n");
                return false;
            }

            oldCamera = VRViewer::instance()->getCamera();
            createMovieScene(movie->getImage());
            startLoadTime = cover->frameTime();
            return true;
        }
        else
        {
            delete movie;
            movie = NULL;
        }
    }

    return false;
}

int StereoVideoPlayerPlugin::unloadMovie(const char *filename, const char *)
{
    if (cover->debugLevel(5))
        cerr << "StereoVideoPlayer::unloadMovies\n";

    VRViewer::instance()->setCamera(plugin->oldCamera);
    cover->getScene()->removeChild(plugin->planeCam);

    plugin->videoPlayer->quit = true;
    delete plugin->movie;
    plugin->movie = NULL;

    if (cover->debugLevel(5))
        cerr << "END StereoVideoPlayer::unloadMovies\n";

    return 0;
}

StereoVideoPlayerPlugin::StereoVideoPlayerPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    coVRConfig::instance()->windows[0].window->getWindowRectangle(xpos, ypos, screenWidth, screenHeight);

    int w = coCoviseConfig::getInt("width", "COVER.ChannelConfig.Channel:0", screenWidth);
    int h = coCoviseConfig::getInt("height", "COVER.ChannelConfig.Channel:0", screenHeight);
    if (w > 1)
        screenWidth = w;
    if (h > 1)
        screenHeight = h;

    plugin = this;
    videoPlayer = new FFMPEGVideoPlayer();
    videoPlayer->setStatus(FFMPEGVideoPlayer::Pause);
    movie = NULL;
    oldCamera = NULL;
    xscale = yscale = 1;
    switchLeftRight = false;
}

bool StereoVideoPlayerPlugin::init()
{
    fprintf(stderr, "StereoVideoPlayerPlugin::StereoVideoPlayerPlugin\n");

    coVRFileManager::instance()->registerFileHandler(&handlers[0]);
    coVRFileManager::instance()->registerFileHandler(&handlers[1]);
    coVRFileManager::instance()->registerFileHandler(&handlers[2]);
    coVRFileManager::instance()->registerFileHandler(&handlers[3]);

    StereoVideoPlayerTab = new coTUITab("StereoVideoPlayer", coVRTui::instance()->mainFolder->getID());
    StereoVideoPlayerTab->setPos(0, 0);

    fileType = new coTUIComboBox("File Type", StereoVideoPlayerTab->getID());
    fileType->addEntry("Separate files");
    fileType->addEntry("Side-by-Side");
    fileType->addEntry("Above/Below");
    fileType->setEventListener(this);
    fileType->setPos(0, 1);
    fileType->setSelectedEntry(0);

    switchButton = new coTUIToggleButton("Switch Left/Right", StereoVideoPlayerTab->getID(), false);
    switchButton->setEventListener(this);
    switchButton->setState(false);
    switchButton->setPos(1, 1);

    playButton = new coTUIToggleBitmapButton("play.png", "pause.png", StereoVideoPlayerTab->getID(), false);
    playButton->setEventListener(this);
    playButton->setState(false);
    playButton->setPos(0, 2);
    playButton->setSize(50, 50);

    stopButton = new coTUIButton("stop.png", StereoVideoPlayerTab->getID());
    stopButton->setEventListener(this);
    stopButton->setPos(0, 3);
    stopButton->setSize(50, 50);

    loopButton = new coTUIToggleBitmapButton("loop.png", "loopActive.png", StereoVideoPlayerTab->getID(), false);
    loopButton->setEventListener(this);
    loopButton->setState(false);
    loopButton->setPos(0, 4);
    loopButton->setSize(50, 50);

    speedLabel = new coTUILabel("Speed in percent:", StereoVideoPlayerTab->getID());
    speedLabel->setPos(0, 5);
    speedEdit = new coTUIEditFloatField("speed", StereoVideoPlayerTab->getID());
    speedEdit->setValue(100.0);
    speedEdit->setPos(1, 5);
    speedEdit->setEventListener(this);

    errorLabel = new coTUILabel("", StereoVideoPlayerTab->getID());
    errorLabel->setPos(0, 6);
    errorLabel->setColor(Qt::red);

    return true;
}

void StereoVideoPlayerPlugin::tabletEvent(coTUIElement *tUIItem)
{

    if (movie)
    {
        if (tUIItem == fileType)
        {
            movie->getStream()->setFileTypeParams(fileType->getSelectedEntry(), switchLeftRight);
            switch (fileType->getSelectedEntry())
            {
            case 1:
                xscale = 2;
                yscale = 1;
                break;

            case 2:
                xscale = 1;
                yscale = 2;
                break;

            default:
                xscale = yscale = 1;
            }

            reloadScene();
        }
        else if (tUIItem == switchButton)
        {
            switchLeftRight = !switchLeftRight;
            movie->getStream()->setFileTypeParams(fileType->getSelectedEntry(), switchLeftRight);
            reloadScene();
        }
        else if (tUIItem == playButton)
        {
            if (videoPlayer->getStatus(FFMPEGVideoPlayer::Pause))
                videoPlayer->setStatus(FFMPEGVideoPlayer::Play);
            else
                videoPlayer->setStatus(FFMPEGVideoPlayer::Pause);
        }
        else if (tUIItem == stopButton)
        {
            videoPlayer->setStatus(FFMPEGVideoPlayer::Stop);
            playButton->setState(false);
        }
        else if (tUIItem == loopButton)
            videoPlayer->setStatus(FFMPEGVideoPlayer::Loop);
        else if (tUIItem == speedEdit)
            videoPlayer->setSpeed(speedEdit->getValue());
    }
}

void StereoVideoPlayerPlugin::reloadScene()
{

    VRViewer::instance()->setCamera(plugin->oldCamera);
    cover->getScene()->removeChild(plugin->planeCam);
    if (videoPlayer->getStatus(FFMPEGVideoPlayer::Play))
    {
        videoPlayer->setStatus(FFMPEGVideoPlayer::Stop);
        playButton->setState(false);
    }

    movie->rewind();
    imgPerFrame = videoPlayer->getFrame(movie->getStream(), movie->getImage(), true, movie->getGLFormat());

    createMovieScene(movie->getImage());
}

void StereoVideoPlayerPlugin::createMovieScene(osg::Image *image)
{
    osg::Geometry *plane = createPlane(image);

    osg::Geode *planeGeode = new osg::Geode();
    planeGeode->addDrawable(plane);
    planeGeode->setStateSet(createMovieStateSet(image));

    planeCam = new osg::Camera();
    setCameraProperties(planeCam);
    planeCam->addChild(planeGeode);

    cover->getScene()->addChild(planeCam);
}

bool checkPower2(int v)
{
    int v1 = 1;
    while ((v1 <<= 1) < v)
    {
    }

    if (v1 == v)
        return (true);
    else
        return (false);
}

osg::StateSet *StereoVideoPlayerPlugin::createMovieStateSet(osg::Image *image)
{
    if (cover->debugLevel(5))
        cerr << "StereoVideoPlayerPlugin::createMovieStateSet" << endl;

    osg::StateSet *movieStateSet = new osg::StateSet();
    movieStateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    movieStateSet->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);

    {
        Texture2D *texture = new Texture2D();
        texture->setDataVariance(osg::Object::DYNAMIC);

        //          if(cover->debugLevel(2)) cerr << "StereoVideoPlayerPlugin::createMovieStateSet(" << filename << ")" << endl;

        texture->setImage(image);
        texture = new Texture2D(image);

        texture->setWrap(Texture::WRAP_S, Texture::CLAMP_TO_BORDER);
        texture->setWrap(Texture::WRAP_T, Texture::CLAMP_TO_BORDER);

        texture->setSourceFormat(GL_RGB);
        texture->setInternalFormat(GL_RGB);

        movieStateSet->setTextureAttributeAndModes(0, texture, osg::StateAttribute::ON);
    }

    if (cover->debugLevel(1))
        cerr << "o";

    cerr.flush();
    if (cover->debugLevel(5))
        cerr << "END StereoVideoPlayerPlugin::createMovieStateSet" << endl;

    return (movieStateSet);
}

osg::Geometry *StereoVideoPlayerPlugin::createPlane(osg::Image *image)
{

    osg::Vec3Array *planeVertices = new osg::Vec3Array();
    osg::Vec3Array *planeNormals = new osg::Vec3Array();
    osg::Vec4Array *planeColors = new osg::Vec4Array();
    osg::Vec2Array *planeTexCoords = new osg::Vec2Array();

    planeVertices->push_back(osg::Vec3(-screenWidth / 2.0f, 0, -screenHeight / 2.0f));
    planeVertices->push_back(osg::Vec3(-screenWidth / 2.0f, 0, screenHeight / 2.0f));
    planeVertices->push_back(osg::Vec3(screenWidth / 2.0f, 0, screenHeight / 2.0f));
    planeVertices->push_back(osg::Vec3(screenWidth / 2.0f, 0, -screenHeight / 2.0f));

    for (int i = 0; i < 4; i++)
    {
        planeNormals->push_back(osg::Vec3(0, 1, 0));
        planeColors->push_back(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    }

    float delta[2];
    deltaTex((float)image->s() * xscale, (float)image->t() * yscale, (float)screenWidth, (float)screenHeight, &delta[0]);
    planeTexCoords->push_back(osg::Vec2(-delta[0], 1 + delta[1]));
    planeTexCoords->push_back(osg::Vec2(-delta[0], -delta[1]));
    planeTexCoords->push_back(osg::Vec2(1 + delta[0], -delta[1]));
    planeTexCoords->push_back(osg::Vec2(1 + delta[0], 1 + delta[1]));

    osg::DrawElementsUInt *planePrimitive = new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS, 0);
    for (int i = 0; i < 4; i++)
        planePrimitive->push_back(i);

    osg::Geometry *planeGeom = new osg::Geometry();
    planeGeom->setVertexArray(planeVertices);
    planeGeom->setNormalArray(planeNormals);
    planeGeom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    planeGeom->setColorArray(planeColors);
    planeGeom->setColorBinding(osg::Geometry::BIND_OVERALL);
    planeGeom->setTexCoordArray(0, planeTexCoords);
    planeGeom->addPrimitiveSet(planePrimitive);

    return planeGeom;
}

void StereoVideoPlayerPlugin::deltaTexRectangle(float texWidth, float texHeight, float screenWidth, float screenHeight, float delta[])
{
    float p = screenWidth / screenHeight;

    if (texWidth / texHeight <= p)
    {
        delta[0] = 0.5 * (p * texHeight - texWidth);
        delta[1] = 0.0;
    }
    else
    {
        delta[0] = 0.0;
        delta[1] = 0.5 * (texWidth / p - texHeight);
    }
}

void StereoVideoPlayerPlugin::deltaTex(float width, float height, float screenWidth, float screenHeight, float delta[])
{
    float q = width / height;
    float p = screenWidth / screenHeight;

    if (q <= p)
    {
        delta[0] = 0.5 * (p / q - 1.0);
        delta[1] = 0.0;
    }
    else
    {
        delta[0] = 0.0;
        delta[1] = 0.5 * (q / p - 1.0);
    }
}

void StereoVideoPlayerPlugin::setCameraProperties(osg::Camera *planeCam)
{
    // just inherit the main cameras view
    planeCam->setReferenceFrame(osg::Transform::ABSOLUTE_RF);

    planeCam->setViewMatrix(osg::Matrix::rotate(-osg::PI / 2, osg::X_AXIS)); //Kamera schaut in z-Richtung(default), deshalb 90° drehen z->y-Achse
    planeCam->setProjectionMatrixAsOrtho2D(-((screenWidth) / 2.0), //Orthogonalprojektion
                                           ((screenWidth) / 2.0),
                                           -((screenHeight) / 2.0),
                                           ((screenHeight) / 2.0));

    // set the camera to render nach der Projektor-Kamera.
    planeCam->setRenderOrder(osg::Camera::POST_RENDER);

    // only clear the depth buffer
    planeCam->setClearMask(0);
}

// this is called if the plugin is removed at runtime
StereoVideoPlayerPlugin::~StereoVideoPlayerPlugin()
{

    delete videoPlayer;
    if (movie)
        unloadMovie("", "");
    delete fileType;
    delete playButton;
    delete stopButton;
    delete loopButton;
    delete speedLabel;
    delete speedEdit;
    delete errorLabel;
    delete StereoVideoPlayerTab;
    coVRFileManager::instance()->unregisterFileHandler(&handlers[0]);
    coVRFileManager::instance()->unregisterFileHandler(&handlers[1]);
    coVRFileManager::instance()->unregisterFileHandler(&handlers[2]);
    coVRFileManager::instance()->unregisterFileHandler(&handlers[3]);
}

double StereoVideoPlayerPlugin::getMasterTime(bool first, double offset)
{
    double masterTime = 0.0;

    double coverTime = cover->frameTime();

    if (first)
        startLoadTime = coverTime - offset;
    masterTime = coverTime - startLoadTime;

    return masterTime * videoPlayer->getSpeed();
}

void StereoVideoPlayerPlugin::preFrame()
{
    static unsigned int maxbuffersize = 0;
    static bool first = true;
    static bool playing = false;
    double masterTime;
    static double lastPts = 0.0;

    if (cover->debugLevel(5))
        cerr << "StereoVideoPlayerPlugin::update" << endl;

    if (!playing)
    {
        if (videoPlayer->getStatus(FFMPEGVideoPlayer::Play))
        {
            masterTime = getMasterTime(true, lastPts);

#ifdef HAVE_SDL
            if (movie->getStream()->getAudioPlayback())
                SDL_PauseAudio(0);
#endif
            playing = true;
        }
    }
    else
    {

        if (videoPlayer->getStatus(FFMPEGVideoPlayer::Stop))
        {
#ifdef HAVE_SDL
            if (movie->getStream()->getAudioPlayback())
                SDL_PauseAudio(1);
#endif

            playing = false;
            lastPts = 0.0;

            movie->rewind();
            imgPerFrame = videoPlayer->getFrame(movie->getStream(), movie->getImage(), true, movie->getGLFormat());
        }
        else if (videoPlayer->getStatus(FFMPEGVideoPlayer::Pause))
        {
#ifdef HAVE_SDL
            if (movie->getStream()->getAudioPlayback())
                SDL_PauseAudio(1);
#endif
            playing = false;
        }
        else
            masterTime = getMasterTime(false);
    }

    if (playing)
    {
        double diffPts = 0.0;
        if (movie->getStream()->vq->getSize() > 0)
        {
            DisplayImage *newImg = movie->getStream()->vq->getImage();
            diffPts = newImg->getPts() - lastPts;

            if (movie->getStream()->vq->getSize() > maxbuffersize)
                maxbuffersize = movie->getStream()->vq->getSize();
            //			fprintf(stderr, "Video: buffersize %d Videobuffersize %d diffPts %f Time %f VideoClock %f AudioClock %f\n", maxbuffersize, movie->getStream()->vq->getSize(), diffPts, masterTime, newImg->getPts(), movie->getStream()->getAudioTime());

            if (newImg->getPts() <= masterTime)
            {
                while (newImg->getPts() < masterTime - diffPts * 2)
                {
                    lastPts = newImg->getPts();
                    movie->getStream()->vq->removeImage();
                    delete newImg;

                    if (!(newImg = movie->getStream()->vq->getImage()))
                        break;
                }

                if (newImg)
                {
                    movie->getStream()->setRGBBuffer(newImg->getRGBImage());
                    videoPlayer->setImage(movie->getStream(), movie->getImage(), movie->getStream()->getRGBFrame(), movie->getGLFormat());

                    lastPts = newImg->getPts();
                    movie->getStream()->vq->removeImage();
                    delete newImg;
                }
            }
        }

        int loaded = 1;
        while ((movie->getStream()->vq->getSize() < movie->getStream()->getMaxVideoBufferSize()) && (loaded++ < imgPerFrame))
        {

            //					fprintf(stderr, "Get Image: Videobuffersize %d Time %f lastPts %f\n", movie->getStream()->vq->getSize(), masterTime, lastPts);
            if ((movie->getStream()->readFrame() < 0) && (movie->getStream()->vq->getSize() == 0))
            {
                playing = false;
                lastPts = 0.0;
                masterTime = getMasterTime(true, lastPts);
                if (!videoPlayer->getStatus(FFMPEGVideoPlayer::Loop))
                {
                    videoPlayer->setStatus(FFMPEGVideoPlayer::Stop);
                    movie->rewind();
                    imgPerFrame = videoPlayer->getFrame(movie->getStream(), movie->getImage(), false, movie->getGLFormat());
                    playButton->setState(false);
                }
                else
                {
                    movie->rewind();
                    imgPerFrame = videoPlayer->getFrame(movie->getStream(), movie->getImage(), true, movie->getGLFormat());
                }
            }
        }
    }
}

MovieData::MovieData(StereoVideoPlayerPlugin *stereoPlugin)
{
    vStream = new VideoStream();
    vStream->myPlugin = stereoPlugin;
    image = new osg::Image();
    glFormat = GL_RGB;
}

MovieData::~MovieData()
{
    delete vStream;
}

void MovieData::setGLFormat(AVPixelFormat pixFormat)
{
    if (pixFormat == AV_PIX_FMT_BGR24)
        glFormat = GL_BGR;
    else
        glFormat = GL_RGB;
}

void MovieData::rewind()
{

    vStream->setVideoClock(0.0);
    vStream->vq->clearImgList();

#ifdef HAVE_SDL
    if (vStream->getAudioPlayback())
    {
        vStream->pq->clearPktList();
        vStream->initAudio();
    }
#endif

#if LIBAVFORMAT_VERSION_INT < (53 << 16)
    int i = av_seek_frame(vStream->getFormatContext(), vStream->getVideoStreamID(), 0, AVSEEK_FLAG_BACKWARD);
#else
    int i = avformat_seek_file(vStream->getFormatContext(), vStream->getVideoStreamID(), 0, 0, 5, AVSEEK_FLAG_BACKWARD);
#endif
}

COVERPLUGIN(StereoVideoPlayerPlugin)
