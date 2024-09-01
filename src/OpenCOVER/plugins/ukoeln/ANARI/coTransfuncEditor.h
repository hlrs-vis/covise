/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-

#pragma once

#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coPopupHandle.h>

class Canvas;

typedef void (*coColorUpdateFunc)(const float *rgb, unsigned numRGB, void *userData);
typedef void (*coOpacityUpdateFunc)(const float *opacity, unsigned numOpacities, void *userData);
typedef void (*coTransfuncOnSaveFunc)(const float *rgb, unsigned numRGB,
                                      const float *opacity, unsigned numOpacities,
                                      float absRangeLo, float absRangeHi,
                                      float relRangeLo, float relRangeHi,
                                      float opacityScale, void *userData);

class coTransfuncEditor : public vrui::coButtonActor
{
public:
    coTransfuncEditor();
   ~coTransfuncEditor();

    void setTransfunc(const float *rgb, unsigned numRGB,
                      const float *opacity, unsigned numOpacities,
                      float absRangeLo, float absRangeHi,
                      float relRangeLo, float relRangeHi,
                      float opacityScale);
    void setColor(const float *rgb, unsigned numRGB);
    void setOpacity(const float *opacity, unsigned numOpacities);

    void show();
    void hide();
    void update();

    void setColorUpdateFunc(coOpacityUpdateFunc func, void *userData);
    void setOpacityUpdateFunc(coOpacityUpdateFunc func, void *userData);
    void setOnSaveFunc(coTransfuncOnSaveFunc func, void *userData);
private:
    vrui::coPopupHandle *handle{nullptr};
    vrui::coFrame *frame{nullptr};
    vrui::coPanel *panel{nullptr};
    vrui::coPushButton *save{nullptr};
    Canvas *canvas{nullptr};

    coTransfuncOnSaveFunc onSaveFunc{nullptr};
    void *onSaveUserData{nullptr};

    void buttonEvent(vrui::coButton *);
};
