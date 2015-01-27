/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VIC_H
#define _VIC_H
/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Vic, Video conferencing                                     **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-03  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coMenu.h>

namespace vrui
{
class coRowMenu;
class coSubMenuItem;
class coCheckboxMenuItem;
class coButtonMenuItem;
class coPotiMenuItem;
class coTrackerButtonInteraction;
class coTexturedBackground;
class coPopupHandle;
}

namespace opencover
{
class coVRLabel;
}

#define NUMPORTS 4
#define SHM_RESOLUTION 256
#define COLORDEPTH 4

using namespace vrui;
using namespace opencover;

class imageHeader
{
public:
    int xsize;
    int ysize;
    int readBuf;
    int writeBuf;
};

class VideoWindow
{
public:
    VideoWindow(int i);
    virtual ~VideoWindow();

    void update();
    void show();
    void hide();

private:
    static u_char *sharedMemoryAttach(int portNumber);

    int portNumber;
    int displayedBuf;
    u_char *sharedMemory;
    imageHeader *header;
    u_char *buf1;
    u_char *buf2;
    coPopupHandle *popupHandle;
    coTexturedBackground *videoTexture;
};

class Vic : public coVRPlugin, public coMenuListener
{
public:
    Vic();
    virtual ~Vic();
    bool init();

    // this will be called in PreFrame
    void preFrame();
    virtual void menuEvent(coMenuItem *menuItem);

    coSubMenuItem *pinboardEntry;
    coRowMenu *vicMenu;
    coCheckboxMenuItem *videoEntry[NUMPORTS];

private:
    VideoWindow *videos[NUMPORTS];
};
#endif
