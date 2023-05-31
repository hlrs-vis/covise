/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Template Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "Vic.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>

#include <OpenVRUI/coToolboxMenu.h>
#include <OpenVRUI/coRowMenu.h>

#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coPopupHandle.h>
#include <OpenVRUI/coTexturedBackground.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <cover/coVRMSController.h>
#include <sys/types.h>
#ifndef WIN32
#include <sys/ioctl.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#endif

VideoWindow::VideoWindow(int num)
{

    videoTexture = 0;
    popupHandle = 0;

    portNumber = num;

    fprintf(stderr, "Vic::VideoWindow::<init> info: Creating %d\n", portNumber);

    fprintf(stderr, "Determine shared memory!\n");
    ;
    sharedMemory = sharedMemoryAttach(portNumber);

    fprintf(stderr, "Create buffers!\n");
    ;
    header = (imageHeader *)sharedMemory;
    buf1 = sharedMemory + sizeof(imageHeader);
    buf2 = sharedMemory + sizeof(imageHeader) + SHM_RESOLUTION * SHM_RESOLUTION * COLORDEPTH;

    fprintf(stderr, "Start writing buffer!\n");
    displayedBuf = header->writeBuf;
    char buf[1000];
    sprintf(buf, "Video%c", 'A' + portNumber);

    fprintf(stderr, "Create video texture!\n");
    fprintf(stderr, "Buffer1 is: %s\n", buf1);
    if (buf1 == NULL)
    {
        fprintf(stderr, "Buffer to be used is NULL!");
        exit(-1);
    }
    videoTexture = new coTexturedBackground((uint *)buf2, NULL, NULL, COLORDEPTH, SHM_RESOLUTION, SHM_RESOLUTION, 0);

    fprintf(stderr, "Set size!\n");
    videoTexture->setSize(500, 500, 0);

    fprintf(stderr, "Set Textsize!\n");
    videoTexture->setTexSize(500, -500);

    fprintf(stderr, "Min width!\n");
    videoTexture->setMinWidth(500);

    fprintf(stderr, "Minheight!\n");
    videoTexture->setMinHeight(500);
    fprintf(stderr, "Texture creation  finished!\n");

    popupHandle = new coPopupHandle(buf);
    popupHandle->setScale(2 * cover->getSceneSize() / 2500);
    popupHandle->setPos(-500 * cover->getSceneSize() / 2500, 0, -500 * cover->getSceneSize() / 2500);
    popupHandle->addElement(videoTexture);
}

VideoWindow::~VideoWindow()
{
}

void VideoWindow::update()
{
    if (videoTexture && popupHandle)
    {
        char c = 1;
        popupHandle->update();
        if (coVRMSController::instance()->isCluster())
        {
            if (coVRMSController::instance()->isMaster())
            {
                if (header->xsize && header->ysize && header->writeBuf != displayedBuf)
                {
                    coVRMSController::instance()->sendSlaves(&c, 1);
                    coVRMSController::instance()->sendSlaves((char *)buf1, SHM_RESOLUTION * SHM_RESOLUTION * COLORDEPTH);
                }
                else
                {
                    c = 0;
                    coVRMSController::instance()->sendSlaves(&c, 1);
                }
            }
            else
            {
                coVRMSController::instance()->readMaster(&c, 1);
                if (c)
                {
                    videoTexture->setUpdated(true);
                    coVRMSController::instance()->readMaster((char *)buf1, SHM_RESOLUTION * SHM_RESOLUTION * COLORDEPTH);
                    videoTexture->setImage((uint *)buf1, NULL, NULL, COLORDEPTH, SHM_RESOLUTION, SHM_RESOLUTION, 0);
                }
            }
        }
        if (header->xsize && header->ysize && header->writeBuf != displayedBuf)
        {
            videoTexture->setUpdated(true);
            videoTexture->setImage((uint *)buf1, NULL, NULL, COLORDEPTH, SHM_RESOLUTION, SHM_RESOLUTION, 0);
            displayedBuf = header->writeBuf;
        }
    }
}

void VideoWindow::show()
{
    if (popupHandle)
        popupHandle->setVisible(true);
}

void VideoWindow::hide()
{
    if (popupHandle)
        popupHandle->setVisible(false);
}

u_char *VideoWindow::sharedMemoryAttach(int portNumber)
{
    fprintf(stderr, "sharedMemoryAttach Begin\n");
    u_char *sharedMemory = 0;

#ifndef WIN32
    int key = 4341 + portNumber;
#define PERMS 0666
    fprintf(stderr, "SharedMemoryAttach:Key used for SharedMemory: %d\n", key);
    int shmid = shmget(key, sizeof(imageHeader) + 2 * SHM_RESOLUTION * SHM_RESOLUTION * COLORDEPTH, 0);
    fprintf(stderr, "sharedMemoryAttach: returned ID: %d\n", shmid);
    if (shmid < 0)
    {
        fprintf(stderr, "creating shared memory segment  key %x = %d for device %d\n", key, key, portNumber);
        shmid = shmget(key, sizeof(imageHeader) + 2 * SHM_RESOLUTION * SHM_RESOLUTION * COLORDEPTH, PERMS | IPC_CREAT);
        if (shmid < 0)
        {
            fprintf(stderr, "Could access shared memory key %x = %d for device %d\n", key, key, portNumber);
            return 0;
        }
    }
    else
    {
    }
    sharedMemory = (u_char *)shmat(shmid, NULL, 0);
#else
    sharedMemory = NULL; // TODO: Memory mapped files
#endif
    if (sharedMemory == NULL)
    {
        fprintf(stderr, "Null-Pointer memory!\n");
    }
    else
    {
        fprintf(stderr, "Valid memory pointer!\n");
    }

    if (sharedMemory == (u_char *)-1)
    {
        sharedMemory = NULL;
    }
    //shmctl(shmid,IPC_RMID, NULL);

    return sharedMemory;
}

Vic::Vic()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

bool Vic::init()
{
    fprintf(stderr, "Vic::Vic\n");

    pinboardEntry = new coSubMenuItem("Vic");
    cover->getMenu()->add(pinboardEntry);
    vicMenu = new coRowMenu("Videos");
    pinboardEntry->setMenu(vicMenu);

    for (int i = 0; i < NUMPORTS; i++)
    {
        char buf[1000];
        sprintf(buf, "Video%c", 'A' + i);
        videoEntry[i] = new coCheckboxMenuItem(buf, false);
        vicMenu->add(videoEntry[i]);
        videoEntry[i]->setMenuListener(this);
        videos[i] = new VideoWindow(i);
    }

    return true;
}

// this is called if the plugin is removed at runtime
Vic::~Vic()
{
    fprintf(stderr, "Vic::~Vic\n");
}

void
Vic::preFrame()
{
    for (int i = 0; i < NUMPORTS; i++)
    {
        videos[i]->update();
    }
}

void Vic::menuEvent(coMenuItem *menuItem)
{
    for (int i = 0; i < NUMPORTS; i++)
    {
        if (menuItem == videoEntry[i])
        {
            if (videoEntry[i]->getState())
            {
                videos[i]->show();
            }
            else
            {
                videos[i]->hide();
            }
        }
    }
}

COVERPLUGIN(Vic)
