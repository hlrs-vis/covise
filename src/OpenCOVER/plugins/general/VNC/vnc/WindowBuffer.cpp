/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                          (C)2009 HLRS  **
 **                                                                        **
 ** Description: VNC Window Buffer                                         **
 **                                                                        **
 **                                                                        **
 ** Author: Lukas Pinkowski                                                **
 **                                                                        **
 **                                                                        **
 ** License: GPL v2 or later                                               **
 **                                                                        **
\****************************************************************************/

#include "WindowBuffer.h"

#include <cover/coVRMSController.h>

using opencover::coVRMSController;

VNCWindowBuffer::VNCWindowBuffer(int width, int height, int id, int parentId)
    : id(id)
    , parentId(parentId)
    , framebuffer(0)
    , transferbuffer(0)
    , closed(false)
    , pixelDataChanged(false)
    , dimensionChanged(false)
{
    resize(width, height);
}

void VNCWindowBuffer::resize(int width, int height)
{
#if 0
   int oldWidth = this->width;
   int oldHeight = this->height;

   (void) oldWidth;
   (void) oldHeight;
#endif

    this->width = width;
    this->height = height;

    // size of one RGB triplett is 3 bytes; we need double block size
    // so we can take half for the transfer buffer
    uint8_t *newFB = new uint8_t[2 * width * height * 3];

    // copy old framebuffer into the new one?

    // delete old Framebuffer, set to new
    if (this->framebuffer)
        delete[] this->framebuffer;
    this->framebuffer = newFB;

    // take second half of the buffer
    this->transferbuffer = &newFB[width * height * 3];

    // set to special color...
    for (int i = 0; i < 2 * width * height * 3; i += 3)
    {
        this->framebuffer[i] = 255;
        this->framebuffer[i] = 0;
        this->framebuffer[i] = 255;
    }
}

void VNCWindowBuffer::close()
{
    closed = true;
}

bool VNCWindowBuffer::hasPixelDataChanged()
{
    bool tmp = pixelDataChanged;
    pixelDataChanged = false;
    return tmp;
}

bool VNCWindowBuffer::hasDimensionChanged()
{
    bool tmp = dimensionChanged;
    dimensionChanged = false;
    return tmp;
}

bool VNCWindowBuffer::synchronize()
{
    if (coVRMSController::instance()->isCluster())
    {
        if (coVRMSController::instance()->isMaster())
        {
            return sendUpdateRectangles();
        }
        else
        {
            return receiveUpdateRectangles();
        }
    }
    else
    {
        bool tmp = !updateQueue.empty();

        while (!updateQueue.empty())
        {
            pixelDataChanged = true;
            updateQueue.pop();
        }

        return tmp;
    }
}

bool VNCWindowBuffer::sendUpdateRectangles()
{
    int count = updateQueue.size();
    bool tmp = count > 0;

    coVRMSController::instance()->sendSlaves((char *)&count, sizeof(count));

    while (!updateQueue.empty())
    {
        rfbRectangle rect = updateQueue.front();
        updateQueue.pop();

        coVRMSController::instance()->sendSlaves((char *)&rect.x, sizeof(rect.x));
        coVRMSController::instance()->sendSlaves((char *)&rect.y, sizeof(rect.y));
        coVRMSController::instance()->sendSlaves((char *)&rect.w, sizeof(rect.w));
        coVRMSController::instance()->sendSlaves((char *)&rect.h, sizeof(rect.h));

        sendPixelData(rect);
    }

    return tmp;
}

bool VNCWindowBuffer::receiveUpdateRectangles()
{
    int count = 0;

    coVRMSController::instance()->readMaster((char *)&count, sizeof(count));
    bool tmp = count > 0;

    for (int i = 0; i < count; ++i)
    {
        rfbRectangle rect;
        coVRMSController::instance()->readMaster((char *)&rect.x, sizeof(rect.x));
        coVRMSController::instance()->readMaster((char *)&rect.y, sizeof(rect.y));
        coVRMSController::instance()->readMaster((char *)&rect.w, sizeof(rect.w));
        coVRMSController::instance()->readMaster((char *)&rect.h, sizeof(rect.h));

        receivePixelData(rect);
    }

    return tmp;
}

void VNCWindowBuffer::sendPixelData(const rfbRectangle &rect)
{
    // normally only a small portion of the framebuffer needs to
    // be updated, but in case the whole framebuffer was updated
    // we use the framebuffer directly instead of copying it to the
    // transferbuffer
    if (rect.w == width && rect.h == height)
    {
        coVRMSController::instance()->sendSlaves(framebuffer, 3 * width * height);
    }
    else
    {
        // copy pixel data to transfer buffer and send it to slaves
        char *it = (char *)transferbuffer;

        for (int iy = rect.y; iy < rect.y + rect.h; ++iy)
        {
            for (int ix = rect.x; ix < rect.x + rect.w; ++ix)
            {
                *it++ = ((char *)framebuffer)[3 * iy * width + 3 * ix];
                *it++ = ((char *)framebuffer)[3 * iy * width + 3 * ix + 1];
                *it++ = ((char *)framebuffer)[3 * iy * width + 3 * ix + 2];
            }
        }

        coVRMSController::instance()->sendSlaves(transferbuffer, 3 * int(rect.w) * int(rect.h));
    }
}

void VNCWindowBuffer::receivePixelData(const rfbRectangle &rect)
{
    char *it = (char *)transferbuffer;

    coVRMSController::instance()->readMaster(transferbuffer, 3 * int(rect.w) * int(rect.h));

    for (int iy = rect.y; iy < rect.y + rect.h; ++iy)
    {
        for (int ix = rect.x; ix < rect.x + rect.w; ++ix)
        {
            ((char *)framebuffer)[3 * iy * width + 3 * ix] = *it++;
            ((char *)framebuffer)[3 * iy * width + 3 * ix + 1] = *it++;
            ((char *)framebuffer)[3 * iy * width + 3 * ix + 2] = *it++;
        }
    }
}
