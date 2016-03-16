/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef CO_DEFAULT_FUNCTION_EDITOR_H
#define CO_DEFAULT_FUNCTION_EDITOR_H

#include "coFunctionEditor.h"
#include <OpenVRUI/coFrame.h>
#include <virvo/vvtransfunc.h>

#include <queue>

/// Function editor base class.
class coDefaultFunctionEditor : public coFunctionEditor
{
public:
    enum
    {
        MaxChannels = 4
    };

private:
    void enqueueTransfuncFilenames(const char *dirName);
    void updateLabels();
    std::queue<const char *> _transfuncFilenames;

protected:
    int defaultColors;
    int defaultAlpha;
    vrui::coPushButton *savedFunctions; ///< saved functions from earlier sessions
    vrui::coPushButton *defColor; ///< preset color
    vrui::coPushButton *defAlpha; ///< preset alpha
    vrui::coPushButton *undo; ///< undo button
    vrui::coToggleButton *hist; ///< histogram button
    vrui::coPushButton *apply; ///< apply current transfer function (for RGBA voxels)
    vrui::coPushButton *load; ///< load tf from volume
    vrui::coPushButton *save; ///< store current transfer function
    vrui::coLabel *scalarMin; ///< minimum scalar value
    vrui::coLabel *scalarMax; ///< maximum scalar value
    vrui::coToggleButton *store;
    void buttonEvent(vrui::coButton *);
    vrui::coToggleButton *colorButton;
    vrui::coToggleButton *alphaPeakButton;
    vrui::coToggleButton *alphaBlankButton;
    vrui::coPushButton *deleteButton;
    vrui::coPushButton *channelButton;
    vrui::coLabel *channelLabel;
    vrui::coButton *recentlyPressedButton;
    float minValue[MaxChannels];
    float maxValue[MaxChannels];
    void (*applyFunction)(void *userData);
    void (*loadFunction)(void *userData);
    void (*saveFunction)(void *userData);
    bool instantClassification;
    virtual void remoteOngoing(const char *message);
    std::vector<vvTransFunc> theTransferFunc;
    void *userData;

public:
    virtual void updateColorBar();
    virtual void updatePinList();
    virtual void setDiscreteColors(int); // update my pinList to reflect the transferFunction
    virtual void setNumChannels(int num);
    virtual void setActiveChannel(int num);
    void setTransferFunc(const vvTransFunc &func, int chan = 0);
    const vvTransFunc &getTransferFunc(int chan = 0) const;
    const std::vector<vvTransFunc> &getTransferFuncs() const;
    void setTransferFuncs(const std::vector<vvTransFunc> &func);
    bool getUseChannelWeights() const;
    std::vector<float> getChannelWeights() const;
    coHSVSelector *hsvSel;
    coPreviewCube *cube;
    coUndoValuePoti *topWidth;
    coUndoValuePoti *botWidth;
    coUndoValuePoti *max;
    coUndoValuePoti *brightness;
    vrui::coToggleButton *mixChannelsButton;
    coUndoValuePoti *mixChannels01;
    coPinEditor *pinedit;
    vrui::coFrame *panelFrame;
    void setColor(float h, float s, float v, int context = -1);
    void setMin(float m);
    void setMax(float m);
    float getMin();
    float getMax();

    coDefaultFunctionEditor(void (*)(void *), void *userData);
    virtual ~coDefaultFunctionEditor();
    void setSaveFunc(void (*)(void *));
    void setLoadFunc(void (*)(void *));
    void update();
    void updateVolume();
    void updateBackground(unsigned char *backgroundTextureData);

    void endPinCreation();
    void potiValueChanged(float, float, vrui::coValuePoti *, int context = -1);
    void potiPressed(vrui::coValuePoti *poti, int context = -1);
    void potiReleased(vrui::coValuePoti *poti, int context = -1);
    int getContext();
    void setInstantMode(bool);
    void putUndoBuffer();
    coPin *getCurrentPin();
};
#endif
