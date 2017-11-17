/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <algorithm>

#include <util/common.h>
#include <util/coFileUtil.h>
#include <cover/coVRPluginSupport.h>
#include <config/CoviseConfig.h>
#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/coPopupHandle.h>
#include <OpenVRUI/coFlatPanelGeometry.h>
#include <OpenVRUI/coSquareButtonGeometry.h>
#include <OpenVRUI/coRectButtonGeometry.h>
#include <OpenVRUI/coToggleButtonGeometry.h>
#include <OpenVRUI/coDefaultButtonGeometry.h>
#include "coHSVSelector.h"
#include <OpenVRUI/coValuePoti.h>
#include "coPreviewCube.h"
#include "coPinEditor.h"
#include "coDefaultFunctionEditor.h"
#include "VolumePlugin.h"
#include <virvo/vvvoldesc.h>
#include <virvo/vvfileio.h>
#include <virvo/vvtransfunc.h>

using namespace vrui;
using namespace opencover;

/** Constructor.
  The name passed in the constructor call to coFunctionEditor will be used
  as the section name in covise config, where its location etc. can be set.
  <pre>
  <TFE>
    <MenuPosition value="0.0 0.0 0.0"/>
    <MenuOrientation value="0.0 0.0 0.0"/>
    <MenuSize value="1.0"/>
  </TFE>
  </pre>
*/
coDefaultFunctionEditor::coDefaultFunctionEditor(void (*applyFunc)(void *),
                                                 void *cbUserData)
    : coFunctionEditor("TFE")
    , loadFunction(NULL)
    , saveFunction(NULL)
{
    userData = cbUserData;
    applyFunction = applyFunc;
    for (int i=0; i<MaxChannels; ++i)
    {
        minValue[i] = 0.;
        maxValue[i] = 1.;
    }
    instantClassification = false;
    defaultColors = defaultAlpha = 0;
    theTransferFunc.resize(1);
    theTransferFunc[0].setDefaultColors(defaultColors, minValue[0], maxValue[0]);
    theTransferFunc[0].setDefaultAlpha(defaultAlpha, minValue[0], maxValue[0]);
    colorButton = new coToggleButton(new coSquareButtonGeometry("Volume/color-menu"), this);
    colorButton->setPos(0, -10.9);
    alphaPeakButton = new coToggleButton(new coSquareButtonGeometry("Volume/peak-menu"), this);
    alphaPeakButton->setPos(0, -22.9);
    alphaBlankButton = new coToggleButton(new coSquareButtonGeometry("Volume/blank-menu"), this);
    alphaBlankButton->setPos(0, -34.9);
    deleteButton = new coPushButton(new coSquareButtonGeometry("Volume/remove-menu"), this);
    deleteButton->setPos(0, -70.9);
    savedFunctions = new coPushButton(new coSquareButtonGeometry("Volume/fromfile"), this);
    savedFunctions->setPos(15, -70.8);
    defColor = new coPushButton(new coSquareButtonGeometry("Volume/defcolor"), this);
    defColor->setPos(28, -70.8);
    defAlpha = new coPushButton(new coSquareButtonGeometry("Volume/defalpha"), this);
    defAlpha->setPos(41, -70.8);
    undo = new coPushButton(new coSquareButtonGeometry("Volume/undo"), this);
    undo->setPos(54, -70.8);
    channelButton = new coPushButton(new coSquareButtonGeometry("Volume/channel"), this);
    channelButton->setPos(67, -70.8);
    channelLabel = new coLabel();
    channelLabel->setPos(80, -68, 1);
    channelLabel->setFontSize(8);

    bool ignore;
    const bool useHistogram = covise::coCoviseConfig::isOn("value", "COVER.Plugin.Volume.UseHistogram", false, &ignore);
    if (useHistogram)
    {
        hist = new coToggleButton(new coSquareButtonGeometry("Volume/histogram"), this);
        hist->setPos(67, -70.8);
    }
    else
    {
        hist = NULL;
    }
    apply = new coPushButton(new coSquareButtonGeometry("Volume/apply"), this);
    apply->setPos(77, -70.8);
    load = new coPushButton(new coSquareButtonGeometry("Volume/load"), this);
    load->setPos(100, -70.8);
    save = new coPushButton(new coSquareButtonGeometry("Volume/save"), this);
    save->setPos(123, -70.8);
    recentlyPressedButton = NULL;
    pinedit = new coPinEditor(&theTransferFunc[activeChannel], this);
    pinedit->setPos(15, -55.6);
    cube = new coPreviewCube();
    cube->setHSVA(0.0, 1.0, 1.0, 0.8);
    cube->setSize(5);
    scalarMin = new coLabel();
    scalarMax = new coLabel();
    scalarMin->setFontSize(5);
    scalarMax->setFontSize(5);
    setMin(0.);
    setMax(1.);
    scalarMin->setPos(15, -59, 1);
    scalarMax->setPos(92, -59, 1);

    topWidth = new coUndoValuePoti("Top", this, "Volume/valuepoti-bg", VolumeCoim.get(), "TOP");
    botWidth = new coUndoValuePoti("Bottom", this, "Volume/valuepoti-bg", VolumeCoim.get(), "BOTTOM");
    topWidth->setMax(2);
    botWidth->setMax(2);
    max = new coUndoValuePoti("Max", this, "Volume/valuepoti-bg", VolumeCoim.get(), "MAXIMUM");
    brightness = new coUndoValuePoti("Bright", this, "Volume/valuepoti-bg", VolumeCoim.get(), "BRIGHTNESS");
    hsvSel = new coHSVSelector(cube, this);
    topWidth->setSize(0.4);
    botWidth->setSize(0.4);
    max->setSize(0.4);
    brightness->setSize(0.4);
    mixChannelsButton = new coToggleButton(new coSquareButtonGeometry("Volume/mix-channels"), this);
    mixChannelsButton->setPos(105, -20);
    mixChannels01 = new coUndoValuePoti("Mix", this, "Volume/valuepoti-bg", VolumeCoim.get(), "MIX_CHANNELS01");
    mixChannels01->setMin(-1);
    mixChannels01->setMax( 1);

    cube->setPos(114.5, -9.5);
    hsvSel->setPos(103, -44.9);
    topWidth->setPos(100, -50);
    botWidth->setPos(103, -50);
    max->setPos(103, -50);
    brightness->setPos(108, -70);
    mixChannels01->setPos(100, -50);

    // Add interaction elements:
    panel->addElement(savedFunctions);
    panel->addElement(defColor);
    panel->addElement(defAlpha);
    panel->addElement(undo);
    if (useHistogram)
    {
        panel->addElement(hist);
    }
    panel->addElement(apply);
    panel->addElement(load);
    panel->addElement(save);
    panel->addElement(channelButton);
    panel->addElement(channelLabel);
    panel->addElement(colorButton);
    panel->addElement(alphaPeakButton);
    panel->addElement(alphaBlankButton);
    panel->addElement(mixChannelsButton);
    panel->addElement(deleteButton);
    panel->addElement(hsvSel);
    panel->addElement(cube);
    panel->addElement(topWidth);
    panel->addElement(botWidth);
    panel->addElement(max);
    panel->addElement(brightness);
    panel->addElement(mixChannels01);
    panel->addElement(scalarMin);
    panel->addElement(scalarMax);
    panel->addElement(pinedit);
    panel->setScale(5.0);
    panel->resize();
    panel->hide(apply);
    panel->hide(load);
    panel->hide(save);

    panelFrame = new coFrame("UI/Frame");
    panelFrame->addElement(panel);
    dropHandle->addElement(panelFrame);

    pinedit->init();
    updatePinList();

    enqueueTransfuncFilenames(".");
}

/// Destructor
coDefaultFunctionEditor::~coDefaultFunctionEditor()
{
    delete topWidth;
    delete botWidth;
    delete max;
    delete brightness;
    delete mixChannels01;
    delete colorButton;
    delete alphaPeakButton;
    delete alphaBlankButton;
    delete deleteButton;
    delete channelButton;
    delete channelLabel;
    delete defColor;
    delete defAlpha;
    delete undo;
    delete hist;
    delete apply;
    delete load;
    delete save;
    delete pinedit;
    delete cube;
    delete scalarMin;
    delete scalarMax;
    delete hsvSel;
    delete panelFrame;
}

void coDefaultFunctionEditor::setTransferFuncs(const std::vector<vvTransFunc> &func)
{
    theTransferFunc = func;
    setNumChannels(theTransferFunc.size());
    if (activeChannel > numChannels)
        setActiveChannel(0);
    else
        setActiveChannel(activeChannel);
}

void coDefaultFunctionEditor::setTransferFunc(const vvTransFunc &func, int channel)
{
    assert(channel >= 0);
    assert(theTransferFunc.size() > channel);

    theTransferFunc[channel] = func;
    if (channel == activeChannel)
        updatePinList();
}

const std::vector<vvTransFunc> &coDefaultFunctionEditor::getTransferFuncs() const
{
    return theTransferFunc;
}

const vvTransFunc &coDefaultFunctionEditor::getTransferFunc(int chan) const
{
    assert(chan >= 0);
    assert(theTransferFunc.size() > chan);

    return theTransferFunc[chan];
}

bool coDefaultFunctionEditor::getUseChannelWeights() const
{
    return mixChannelsButton->getState();
}

std::vector<float> coDefaultFunctionEditor::getChannelWeights() const
{
    size_t num_channels = theTransferFunc.size();
    std::vector<float> channelWeights(num_channels);
    std::fill(channelWeights.begin(), channelWeights.end(), 1.0f);

    if (num_channels == 2 && true)
    {
        float s = (mixChannels01->getValue() + 1.0f) / 2.0f;
        channelWeights[0] = s;
        channelWeights[1] = 1.0f - s;
    }

    return channelWeights;
}

void coDefaultFunctionEditor::enqueueTransfuncFilenames(const char *dirName)
{
    // Clear the queue with file names.
    std::queue<const char *> clear;
    std::swap(_transfuncFilenames, clear);

    covise::coDirectory *dir = covise::coDirectory::open(dirName);

    if (dir != NULL)
    {
        // Start with the most recently saved transfer-function.
        // Thus store the file names in reverse order of occurrence
        // in the directory listing.
        std::vector<const char *> reverseFilenames;
        for (int i = 0; i < dir->count(); ++i)
        {
            const char *filename = dir->name(i);
            const char *fullname = dir->full_name(i);

            if (strlen(filename) >= 26)
            {
                char prefix[23];
                strncpy(prefix, filename, 22);
                prefix[22] = '\0';
                if (strcmp(prefix, "cover-transferfunction") == 0)
                {
                    reverseFilenames.push_back(fullname);
                }
            }
        }

        // Reverse the order.
        while (!reverseFilenames.empty())
        {
            _transfuncFilenames.push(reverseFilenames.back());
            reverseFilenames.pop_back();
        }
    }
}

void coDefaultFunctionEditor::updateLabels()
{
    std::stringstream s;
    s << activeChannel;
    channelLabel->setString(s.str());


    char num[50];

    float m = minValue[activeChannel];
    if (((float)(int)m) != m)
        sprintf(num, "%.2f", m);
    else
        sprintf(num, "%d", (int)m);
    scalarMin->setString(num);

    m = maxValue[activeChannel];
    if (((float)(int)m) != m)
        sprintf(num, "%.2f", m);
    else
        sprintf(num, "%d", (int)m);
    scalarMax->setString(num);
}

void coDefaultFunctionEditor::setLoadFunc(void (*f)(void *))
{
    loadFunction = f;
    if (loadFunction)
        panel->show(load);
    else
        panel->hide(load);
}

void coDefaultFunctionEditor::setSaveFunc(void (*f)(void *))
{
    saveFunction = f;
    if (saveFunction)
        panel->show(save);
    else
        panel->hide(save);
}

coPin *coDefaultFunctionEditor::getCurrentPin()
{
    return pinedit->currentPin;
}

void coDefaultFunctionEditor::updateBackground(unsigned char *texture)
{
    pinedit->updateBackground(texture);
}

void coDefaultFunctionEditor::updatePinList()
{
    pinedit->setTransFuncPtr(&theTransferFunc[activeChannel]);
    pinedit->updatePinList();
}

void coDefaultFunctionEditor::updateColorBar()
{
    pinedit->updateColorBar();
}

void coDefaultFunctionEditor::setDiscreteColors(int nc)
{
    theTransferFunc[activeChannel].setDiscreteColors(nc);
}

void coDefaultFunctionEditor::setColor(float h, float s, float v, int context)
{
    pinedit->setColor(h, s, v, context);
    brightness->setValue(v);
}

void coDefaultFunctionEditor::setMin(float m)
{
    minValue[activeChannel] = m;
    updateLabels();
    updatePinList();
}

void coDefaultFunctionEditor::setMax(float m)
{
    maxValue[activeChannel] = m;
    updateLabels();
    updatePinList();
}

float coDefaultFunctionEditor::getMin()
{
    return minValue[activeChannel];
}

float coDefaultFunctionEditor::getMax()
{
    return maxValue[activeChannel];
}

void coDefaultFunctionEditor::endPinCreation()
{
    colorButton->setState(false);
    alphaPeakButton->setState(false);
    alphaBlankButton->setState(false);
}

// Update function editor panel
void coDefaultFunctionEditor::update()
{
    coFunctionEditor::update();

    hsvSel->update();
    cube->update();
    pinedit->update();

    float range = getMax() - getMin();
    topWidth->setMax(2 * range);
    botWidth->setMax(2 * range);
}

int coDefaultFunctionEditor::getContext()
{
    if (pinedit->currentPin)
        return pinedit->currentPin->getID();
    else
        return -1;
}

void coDefaultFunctionEditor::potiValueChanged(float oldVal, float newVal, coValuePoti *poti, int context)
{
    (void)oldVal;
    if (poti == topWidth)
    {
        pinedit->setTopWidth(newVal, context);
    }
    else if (poti == botWidth)
    {
        pinedit->setBotWidth(newVal, context);
    }
    else if (poti == brightness)
    {
        hsvSel->setBrightness(brightness->getValue());
        pinedit->setBrightness(brightness->getValue());
    }
    else if (poti == mixChannels01)
    {
        pinedit->updateColorBar();
    }
    else if (poti == max)
    {
        pinedit->setMax(newVal, context);
    }
}

void coDefaultFunctionEditor::potiPressed(coValuePoti *poti, int context)
{
    (void)context;
    if (poti == mixChannels01)
        pinedit->setMixChannelsActive(true);
}

void coDefaultFunctionEditor::potiReleased(coValuePoti *poti, int context)
{
    (void)context;
    if (poti == mixChannels01)
        pinedit->setMixChannelsActive(false);
}

void coDefaultFunctionEditor::updateVolume()
{
    if (instantClassification)
    {
        (*applyFunction)(userData);
    }
}

void coDefaultFunctionEditor::remoteOngoing(const char *message)
{

    switch (message[0])
    {
    case 'C':
    {
        theTransferFunc[activeChannel].setDefaultColors(defaultColors++, getMin(), getMax());
        if (defaultColors >= theTransferFunc[activeChannel].getNumDefaultColors())
            defaultColors = 0;
        pinedit->updatePinList();
    }
    break;
    case 'A':
    {
        theTransferFunc[activeChannel].setDefaultAlpha(defaultAlpha++, getMin(), getMax());
        if (defaultAlpha >= theTransferFunc[activeChannel].getNumDefaultAlpha())
            defaultAlpha = 0;
        pinedit->updatePinList();
    }
    break;
    case 'U':
    {
        theTransferFunc[activeChannel].getUndoBuffer();
        pinedit->updatePinList();
        cerr << "getUndoFromRemote" << endl;
    }
    break;
    case 'P':
    {
        theTransferFunc[activeChannel].putUndoBuffer();
        cerr << "putUndoFromRemote" << endl;
    }
    break;
    case 'S':
    {
        int id;
        sscanf(message, "S%d", &id);
    }
    break;
    case 'R':
    {
        int id;
        sscanf(message, "R%d", &id);
    }
    break;
    case 'F':
    {
    }
    break;
    default:
    {
        cerr << "coDefaultFunctionEditor: Unknown remote command" << endl;
    }
    break;
    }
}

void coDefaultFunctionEditor::putUndoBuffer()
{
    theTransferFunc[activeChannel].putUndoBuffer();
    sendOngoingMessage("P");
    //cerr << "putUndoLocal" << endl;
}

void coDefaultFunctionEditor::buttonEvent(coButton *button)
{

    const bool released = !button->isPressed();
    if (!released)
    {
        recentlyPressedButton = button;
    }

    if (released && (button == savedFunctions)
        && (recentlyPressedButton == savedFunctions))
    {
        vvFileIO *fio = new vvFileIO();
        vvVolDesc *vd = new vvVolDesc();
        if (fio->importTF(vd, _transfuncFilenames.front()) == vvFileIO::OK)
        {
            cerr << "Loaded transfer-function file: " << _transfuncFilenames.front() << endl;
            putUndoBuffer();
            for (int i = 0; i < vd->tf.size() && i < theTransferFunc.size(); ++i)
            {
                theTransferFunc[i] = vd->tf[i];
            }
            updatePinList();
            const char *tmp = _transfuncFilenames.front();
            _transfuncFilenames.pop();
            _transfuncFilenames.push(tmp);
            sendOngoingMessage("F");
        }
        else
        {
            cerr << "Couldn't load standard transfer-function" << endl;
        }
        delete vd;
        delete fio;
    }
    else if (released && (button == defColor)
             && (recentlyPressedButton == defColor))
    {
        putUndoBuffer();
        theTransferFunc[activeChannel].setDefaultColors(defaultColors++, getMin(), getMax());
        if (defaultColors >= theTransferFunc[activeChannel].getNumDefaultColors())
            defaultColors = 0;
        updatePinList();
        sendOngoingMessage("C");
    }
    else if (released && (button == defAlpha)
             && (recentlyPressedButton == defAlpha)) // default alpha
    {
        putUndoBuffer();
        theTransferFunc[activeChannel].setDefaultAlpha(defaultAlpha++, getMin(), getMax());
        if (defaultAlpha >= theTransferFunc[activeChannel].getNumDefaultAlpha())
            defaultAlpha = 0;
        updatePinList();
        sendOngoingMessage("A");
    }
    else if (released && (button == undo)
             && (recentlyPressedButton == undo))
    {
        theTransferFunc[activeChannel].getUndoBuffer();
        updatePinList();
        sendOngoingMessage("U");
        cerr << "getUndoLocal" << endl;
    }
    else if (button == hist)
    {
        pinedit->setBackgroundType(!button->getState());
    }
    else if (button == colorButton)
    {
        if (!released)
        {
            putUndoBuffer();
            pinedit->addPin(coPinEditor::COLOR);
        }
        else
            pinedit->undoAddPin();
        alphaPeakButton->setState(false);
        alphaBlankButton->setState(false);
    }
    else if (button == alphaPeakButton)
    {
        if (!released)
        {
            putUndoBuffer();
            pinedit->addPin(coPinEditor::ALPHA_HAT);
        }
        else
            pinedit->undoAddPin();
        colorButton->setState(false);
        alphaBlankButton->setState(false);
    }
    else if (button == alphaBlankButton)
    {
        if (!released)
        {
            putUndoBuffer();
            pinedit->addPin(coPinEditor::ALPHA_BLANK);
        }
        else
            pinedit->undoAddPin();
        colorButton->setState(false);
        alphaPeakButton->setState(false);
    }
    else if (button == mixChannelsButton)
    {
        pinedit->updateColorBar();
    }
    else if (button == deleteButton)
    {
        putUndoBuffer();
        pinedit->deleteCurrentPin();
    }
    else if (button == channelButton)
    {
        if (released)
        {
            int cur = activeChannel + 1;
            if (cur >= numChannels)
                cur = 0;
            setActiveChannel(cur);
        }
    }
    else if (button == apply)
    {
        (*applyFunction)(userData);
    }
    else if (button == save)
    {
        if (saveFunction)
            (*saveFunction)(userData);
    }
    else if (button == load)
    {
        if (loadFunction)
            (*loadFunction)(userData);
    }

    if (released)
    {
        recentlyPressedButton = NULL;
    }
}

void coDefaultFunctionEditor::setInstantMode(bool inst)
{
    instantClassification = inst;
    if (instantClassification)
        panel->hide(apply);
    else
        panel->show(apply);
}

void coDefaultFunctionEditor::setNumChannels(int num)
{
    coFunctionEditor::setNumChannels(num);
    theTransferFunc.resize(num);
    if (numChannels > 1)
    {
        panel->show(channelButton);
        panel->show(channelLabel);

        if (numChannels == 2)
        {
            panel->show(mixChannelsButton);
            panel->show(mixChannels01);
        }
    }
    else
    {
        panel->hide(channelButton);
        panel->hide(channelLabel);

        panel->hide(mixChannelsButton);
        panel->hide(mixChannels01);
    }
}

void coDefaultFunctionEditor::setActiveChannel(int num)
{
    coFunctionEditor::setActiveChannel(num);
    updatePinList();
    updateLabels();
}
