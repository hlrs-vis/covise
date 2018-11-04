/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef CO_FUNCTION_EDITOR_H
#define CO_FUNCTION_EDITOR_H

#include <OpenVRUI/coValuePoti.h>
#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coLabel.h>
#include <cover/coCollabInterface.h>

class coHSVSelector;
class vvTransFunc;
class coPin;
class coPinEditor;
class coPreviewCube;

namespace vrui
{
class coPanel;
class coToggleButton;
class coPushButton;
class coPopupHandle;
}

class coUndoValuePoti;
class coUndoSlopePoti;

/// Function editor base class.
class coFunctionEditor : public vrui::coButtonActor,
                         public vrui::coValuePotiActor,
                         public vrui::vruiCollabInterface
{
protected:
    bool instantClassification;
    int numChannels;
    int activeChannel;

    virtual void remoteOngoing(const char *message);

public:
    vrui::coPanel *panel;
    vrui::coPopupHandle *dropHandle;

    coFunctionEditor(const char *collaborativeUIName);
    ~coFunctionEditor();
    virtual void update() = 0;
    virtual void show();
    virtual void hide();
    virtual bool isVisible();
    virtual void setDiscreteColors(int);
    virtual void updateColorBar();
    virtual void putUndoBuffer();
    virtual void updateBackground(unsigned char *backgroundTextureData) = 0;
    virtual void updatePinList(); // update my pinList to reflect the transferFunction

    virtual void updateVolume();
    virtual void setInstantMode(bool);
    virtual void setMin(float m);
    virtual void setMax(float m);
    virtual void setColor(float h, float s, float v, int context = -1) = 0;
    virtual float getMin();
    virtual float getMax();
    virtual coPin *getCurrentPin();

    virtual int getNumChannels() const;
    virtual void setNumChannels(int chan);
    virtual int getActiveChannel() const;
    virtual void setActiveChannel(int chan);
};

class coUndoValuePoti : public vrui::coValuePoti
{
public:
    coUndoValuePoti(const char *, vrui::coValuePotiActor *, const char *, opencover::coCOIM *, const char *);
    int hit(vrui::vruiHit *);

private:
    coFunctionEditor *editor;
};

class coUndoSlopePoti : public vrui::coSlopePoti
{
public:
    coUndoSlopePoti(const char *, vrui::coValuePotiActor *, const char *, opencover::coCOIM *, const char *);
    int hit(vrui::vruiHit *);

private:
    coFunctionEditor *editor;
};
#endif
