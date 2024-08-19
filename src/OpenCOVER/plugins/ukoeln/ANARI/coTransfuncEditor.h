/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-

#pragma once

#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coPopupHandle.h>

class Canvas;

typedef void (*coColorUpdateFunc)(const float *rgb, unsigned len, void *userData);
typedef void (*coOpacityUpdateFunc)(const float *opacity, unsigned len, void *userData);

class coTransfuncEditor
{
public:
    coTransfuncEditor();
   ~coTransfuncEditor();

    void show();
    void hide();
    void update();

    void setColorUpdateFunc(coOpacityUpdateFunc func, void *userData);
    void setOpacityUpdateFunc(coOpacityUpdateFunc func, void *userData);
private:
    vrui::coPopupHandle *handle{nullptr};
    vrui::coFrame *frame{nullptr};
    vrui::coPanel *panel{nullptr};
    Canvas *canvas{nullptr};
};
