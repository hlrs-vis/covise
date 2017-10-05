#ifndef _STEREOVIDEO_PLUGIN_H
#define _STEREOVIDEO_PLUGIN_H
/****************************************************************************\
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: PLMXML Plugin (load video files)                       **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                             **
 **                                                                          **
 ** History:  					                             **
 ** Nov-01  v1	    				                             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include "FFMPEGVideoPlayer.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coTabletUI.h>
using namespace covise;
using namespace opencover;

#include <osg/Group>
#include <osg/ImageStream>
#include <osg/Camera>

#include <string>

class StereoVideoPlayerPlugin;

class MovieData
{
public:
    MovieData(StereoVideoPlayerPlugin *);
    ~MovieData();
    VideoStream *getStream()
    {
        return vStream;
    };
    osg::Image *getImage()
    {
        return image;
    };
    void setGLFormat(AVPixelFormat);
    GLenum getGLFormat()
    {
        return glFormat;
    };
    void rewind();

private:
    VideoStream *vStream;
    osg::Image *image;
    GLenum glFormat;
};

class PLUGINEXPORT StereoVideoPlayerPlugin : public coVRPlugin, public coTUIListener
{
public:
    friend class VideoStream;

    static StereoVideoPlayerPlugin *plugin;

    StereoVideoPlayerPlugin();
    ~StereoVideoPlayerPlugin();

    bool init();

    static int loadMovie(const char *, osg::Group *, const char *covise_key);
    static int unloadMovie(const char *, const char *covise_key);
    void preFrame();
    double getMasterTime(bool, double offset = 0.0);
    void tabletEvent(coTUIElement *);

private:
    int xpos, ypos, screenWidth, screenHeight;
    int xscale, yscale;
    double startLoadTime;
    int imgPerFrame;
    MovieData *movie;
    osg::Camera *planeCam, *oldCamera;
    FFMPEGVideoPlayer *videoPlayer;
    bool switchLeftRight;

    bool parseFilename(std::string *);
    void createMovieScene(osg::Image *);
    void reloadScene();
    void setCameraProperties(osg::Camera *);
    bool addMovieEnvironment(std::string);
    osg::StateSet *createMovieStateSet(osg::Image *);
    bool openMovie(std::string, MovieData *);
    osg::Geometry *createPlane(osg::Image *);
    void deltaTex(float, float, float, float, float[]);
    void deltaTexRectangle(float, float, float, float, float[]);

    coTUITab *StereoVideoPlayerTab;
    coTUIToggleBitmapButton *playButton;
    coTUIButton *stopButton;
    coTUIToggleBitmapButton *loopButton;
    coTUIToggleButton *switchButton;
    coTUILabel *speedLabel;
    coTUIEditFloatField *speedEdit;
    coTUILabel *errorLabel;
    coTUIComboBox *fileType;
};

#endif
