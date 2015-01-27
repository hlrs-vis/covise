/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VNCWINDOWBUFFER_H
#define VNCWINDOWBUFFER_H

#ifndef WIN32
#include <stdint.h>
#endif

#include <queue>

#include "rfbproto.h"

class VNCWindowBuffer
{
public:
    VNCWindowBuffer(int width, int height, int id, int parentId = 0);

    typedef std::queue<rfbRectangle> UpdateQueue;

    int getId() const
    {
        return id;
    }
    int getParentId() const
    {
        return parentId;
    }
    int getWidth() const
    {
        return width;
    }
    int getHeight() const
    {
        return height;
    }
    UpdateQueue *getUpdateQueue()
    {
        return &updateQueue;
    }
    bool isClosed() const
    {
        return closed;
    }
    uint8_t *getFramebuffer()
    {
        return framebuffer;
    }
    uint8_t *getTransferbuffer()
    {
        return transferbuffer;
    }

    void resize(int width, int height);
    void close();
    bool hasPixelDataChanged();
    bool hasDimensionChanged();

    /// synchronizes the pixel data and window size in a cluster
    bool synchronize();

private:
    /// sends coordinates and data of updated rectangles
    bool sendUpdateRectangles();

    /// receives coordinates and data of updated rectangles
    bool receiveUpdateRectangles();

    void sendPixelData(const rfbRectangle &rect);
    void receivePixelData(const rfbRectangle &rect);

    int width; ///< width of the framebuffer
    int height; ///< height of the framebuffer
    int id; ///< window ID for SharedAppVnc
    int parentId; ///< parent ID in case this is a dialog

    uint8_t *framebuffer; ///< the framebuffer
    uint8_t *transferbuffer; ///< the transfer buffer for the cluster

    UpdateQueue updateQueue; ///< queue for the cluster

    bool closed; ///< whether this window was closed
    bool pixelDataChanged; ///< whether the pixel data has changed
    bool dimensionChanged; ///< whether the window size has changed
}; // class VNCWindowBuffer

#endif // VNCWINDOWBUFFER_H
