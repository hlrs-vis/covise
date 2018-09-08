// DeskVOX - Volume Exploration Utility for the Desktop
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
// 
// This file is part of DeskVOX.
//
// DeskVOX is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the 
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#ifndef VV_DIALOGS_H
#define VV_DIALOGS_H

#include <vvplatform.h>

#include <fx.h>

#include <iostream>
#include <string.h>

// Local:
#include "vvshell.h"
#include "vvcanvas.h"
#include "vvmovie.h"
#include "vvclient.h"

/**************************************************/
class VVVolumeDialog : public FXDialogBox
{
  FXDECLARE(VVVolumeDialog)

  protected:
    VVVolumeDialog(){}
    FXTextField* _fileTField;
    FXCanvas* _iconCanvas;
    FXLabel* _widthLabel;
    FXLabel* _heightLabel;
    FXLabel* _slicesLabel;
    FXLabel* _framesLabel;
    FXLabel* _bpcLabel;
    FXLabel* _chanLabel;
    FXLabel* _voxelsLabel;
    FXLabel* _bytesLabel;
    FXLabel* _dxLabel;
    FXLabel* _dyLabel;
    FXLabel* _dzLabel;
    FXLabel* _minLabel;
    FXLabel* _maxLabel;
    FXLabel* _realMinLabel;
    FXLabel* _realMaxLabel;
    vox::vvCanvas* _canvas;

  public:
    enum
    {
      ID_ICON_CANVAS = FXDialogBox::ID_LAST,
      ID_MAKE_ICON,
      ID_CHANNELS,
      ID_LAST
    };
    VVVolumeDialog(FXWindow*, vox::vvCanvas*);
    long onPaint(FXObject*,FXSelector,void*);
    long onMakeIcon(FXObject*,FXSelector,void*);
    long onChannels(FXObject*,FXSelector,void*);
    void updateValues();
    void drawIcon();
};

/**************************************************/
/*
class VVSystemInfo : public FXDialogBox
{
  FXDECLARE(VVSystemInfo)

  protected:
    VVSystemInfo(){}

  public:
    VVSystemInfo(FXWindow*, vvCanvas*);
    void updateValues();
};
*/
/***************************************************/

class VVCameraSetDialog : public FXDialogBox
{
  FXDECLARE(VVCameraSetDialog)

  private:
    VVCameraSetDialog(){}

  public:
    enum
    {
      ID_FOV = FXDialogBox::ID_LAST,
      ID_WIDTH,
      ID_CLIP,
      ID_PERSPECTIVE,
      ID_INTERACTIVE,
      ID_DEF,
      ID_LEFT,
      ID_RIGHT,
      ID_TOP,
      ID_BOTTOM,
      ID_FRONT,
      ID_BACK,
      ID_LAST
    };
    vox::vvCanvas* _canvas;
    VVShell* _shell;
    FXCheckButton* _perspectiveCheck;
    FXRealSlider* _fovSlider;
    FXRealSlider* _widthSlider;
    FXRealSlider* _nearSlider;
    FXRealSlider* _farSlider;
    FXLabel* _fovLabel;
    FXLabel* _widthLabel;
    FXLabel* _nearLabel;
    FXLabel* _farLabel;

    VVCameraSetDialog(FXWindow*, vox::vvCanvas*);
    virtual ~VVCameraSetDialog();
    long onPerspectiveChange(FXObject*,FXSelector,void*);
    long onFOVChange(FXObject*,FXSelector,void*);
    long onWidthChange(FXObject*,FXSelector,void*);
    long onClipChange(FXObject*,FXSelector,void*);
    long onDefaultSelect(FXObject*,FXSelector,void*);
    long onLeft(FXObject*,FXSelector,void*);
    long onRight(FXObject*,FXSelector,void*);
    long onTop(FXObject*,FXSelector,void*);
    long onBottom(FXObject*,FXSelector,void*);
    long onFront(FXObject*,FXSelector,void*);
    long onBack(FXObject*,FXSelector,void*);
    void reset();
    void updateValues();
};

/********************************************************/

class VVClippingDialog : public FXDialogBox
{
  FXDECLARE(VVClippingDialog)

  private:
    VVClippingDialog(){}

  public:
    enum
    {
      ID_ENABLE=FXDialogBox::ID_LAST,
      ID_SINGLE,
      ID_OPAQUE,
      ID_PERIMETER,
      ID_X,
      ID_Y,
      ID_Z,
      ID_ORIGIN,
      ID_LAST
    };
    FXCheckButton* _enableCheck;
    FXCheckButton* _singleCheck;
    FXCheckButton* _opaqueCheck;
    FXCheckButton* _perimeterCheck;

    FXRealSlider* _xSlider;
    FXRealSlider* _ySlider;
    FXRealSlider* _zSlider;
    FXRealSlider* _originSlider;

    vox::vvCanvas* _canvas;
    VVShell* _shell;

    VVClippingDialog(FXWindow*, vox::vvCanvas*);
    virtual ~VVClippingDialog();
    long onEnableChange(FXObject*, FXSelector, void*);
    long onSingleChange(FXObject*, FXSelector, void*);
    long onOpaqueChange(FXObject*, FXSelector, void*);
    long onPerimeterChange(FXObject*, FXSelector, void*);
    long onSliderChange(FXObject*, FXSelector, void*);
    void updateClipParameters();
    void updateValues();
};

/********************************************************/

class VVROIDialog : public FXDialogBox
{
  FXDECLARE(VVROIDialog)

  private:
    VVROIDialog(){}

  public:
    enum
    {
      ID_ENABLE=FXDialogBox::ID_LAST,
      ID_X,
      ID_Y,
      ID_Z,
      ID_SIZE_X,
      ID_SIZE_Y,
      ID_SIZE_Z,
      ID_LAST
    };
    FXCheckButton* _enableCheck;

    FXRealSlider* _xSlider;
    FXRealSlider* _ySlider;
    FXRealSlider* _zSlider;
    FXRealSlider* _sizeSliderX;
    FXRealSlider* _sizeSliderY;
    FXRealSlider* _sizeSliderZ;

    vox::vvCanvas* _canvas;
    VVShell* _shell;

    VVROIDialog(FXWindow*, vox::vvCanvas*);
    virtual ~VVROIDialog();
    long onEnableChange(FXObject*, FXSelector, void*);
    long onSliderChange(FXObject*, FXSelector, void*);
    void updateROIParameters();
    void updateValues();
};

/*********************************************************/

class VVDimensionDialog : public FXDialogBox
{
  FXDECLARE(VVDimensionDialog)

  private:
    VVDimensionDialog(){}

  public:
    enum
    {
      ID_X=FXDialogBox::ID_LAST,
      ID_Y,
      ID_Z,
      ID_TIME,
      ID_RESET,
      ID_APPLY,
      ID_LAST
    };
    FXRealSpinner* _xSpinner;
    FXRealSpinner* _ySpinner;
    FXRealSpinner* _zSpinner;
    float defaultDist[3];

    vox::vvCanvas* _canvas;
    VVShell* _shell;

    VVDimensionDialog(FXWindow*, vox::vvCanvas*);
    virtual ~VVDimensionDialog();

    long onApplySelect(FXObject*,FXSelector,void*);
    long onResetSelect(FXObject*,FXSelector,void*);
    void updateValues();
    void scaleZ(float);
    void initDefaultDistances();
};

/*********************************************************/

class VVDrawDialog : public FXDialogBox
{
  FXDECLARE(VVDrawDialog)

  private:
    bool _isInteractive;
    
    VVDrawDialog(){}

  public:
    enum
    {
      ID_APPLY=FXDialogBox::ID_LAST,
      ID_DEF,
      ID_DRAW,
      ID_LAST
    };
    vox::vvCanvas* _canvas;
    VVShell* _shell;
    FXTextField* _startXTField;
    FXTextField* _startYTField;
    FXTextField* _startZTField;
    FXTextField* _endXTField;
    FXTextField* _endYTField;
    FXTextField* _endZTField;
    FXTextField* _colorTField;
    FXButton*    _lineButton;
    FXButton*    _boxButton;

    VVDrawDialog(FXWindow*, vox::vvCanvas*);
    virtual ~VVDrawDialog();
    long onCmdDraw(FXObject*,FXSelector,void*);
    long onDefaultSelect(FXObject*,FXSelector,void*);
    void updateValues();
};

/*********************************************************/

class VVMergeDialog : public FXDialogBox
{
  FXDECLARE(VVMergeDialog)

  private:
    vox::vvCanvas* _canvas;
    VVShell* _shell;

    VVMergeDialog(){}

  public:
    enum
    {
      ID_BROWSE = FXDialogBox::ID_LAST,
      ID_HELP,
      ID_SLICES2VOL,
      ID_VOL2ANIM,
      ID_CHAN2VOL,
      ID_LIMIT,
      ID_NUMBERED,
      ID_LAST
    };
    FXRadioButton* _slices2volButton;
    FXRadioButton* _vol2animButton;
    FXRadioButton* _chan2volButton;
    FXTextField*   _fileTField;
    FXTextField*   _numberTField;
    FXTextField*   _incrementTField;
    FXCheckButton* _numFilesCB;
    FXCheckButton* _limitFilesCB;

    VVMergeDialog(FXWindow*, vox::vvCanvas*);
    virtual ~VVMergeDialog();
    void updateValues();
    long onCmdBrowse(FXObject*, FXSelector, void*);
    long onCmdHelp(FXObject*, FXSelector, void*);
    long onCmdSlices2Vol(FXObject*,FXSelector,void*);
    long onCmdVol2Anim(FXObject*,FXSelector,void*);
    long onCmdChan2Vol(FXObject*,FXSelector,void*);
    long onCBNumbered(FXObject*,FXSelector,void*);
    long onCBLimit(FXObject*,FXSelector,void*);
};

/*********************************************************/

class VVServerDialog : public FXDialogBox
{
  FXDECLARE(VVServerDialog)

  private:
    vox::vvCanvas* _canvas;
    VVShell* _shell;
    vox::vvClient* _client;

    VVServerDialog(){}

  public:
    enum
    {
      ID_REQUEST = FXDialogBox::ID_LAST,
      ID_DATASET,
      ID_LOD,
      ID_LAST
    };
    FXComboBox*  _datasetCombo;
    FXComboBox*  _lodCombo;
    FXTextField* _x0TField;
    FXTextField* _y0TField;
    FXTextField* _x1TField;
    FXTextField* _y1TField;
    FXTextField* _startSliceTField;
    FXTextField* _endSliceTField;

    VVServerDialog(FXWindow*, vox::vvCanvas*);
    virtual ~VVServerDialog();
    void updateValues();
    void getDatasetList();
    long onCmdRequest(FXObject*,FXSelector,void*);
};

/*********************************************************/

class VVScreenshotDialog : public FXDialogBox
{
  FXDECLARE(VVScreenshotDialog)

  protected:
    VVScreenshotDialog(){}

  public:
    enum
    {
      ID_PICTURE=FXDialogBox::ID_LAST,
      ID_DIM,
      ID_BROWSE,
      ID_LAST
    };
    FXLabel* _sizeLabel;
    FXTextField* _widthTField;
    FXTextField* _heightTField;
    FXTextField* _fileTField;
    FXTextField* _dirTField;
    FXCheckButton* _useScreenDim;
    vox::vvCanvas* _canvas;
    VVShell* _shell;

    VVScreenshotDialog(FXWindow*, vox::vvCanvas*);
    virtual ~VVScreenshotDialog();
    long onCmdPicture(FXObject*, FXSelector, void*);
    long onDimSelect(FXObject*, FXSelector, void*);
    long onBrowseSelect(FXObject*, FXSelector, void*);
    void updateValues();
};

/*********************************************************/

class VVMovieDialog : public FXDialogBox
{
  FXDECLARE(VVMovieDialog)

  protected:
    FXLabel* _selectedLabel;
    FXLabel* _stepLabel1;
    FXLabel* _stepLabel2;
    FXSlider* _frameSlider;
    FXTextField* _widthTField;
    FXTextField* _heightTField;
    FXTextField* _fileTField;
    FXTextField* _dirTField;
    FXCheckButton* _useScreenDim;
    vox::vvCanvas* _canvas;
    vox::vvMovie*  _movie;
    VVShell* _shell;

  public:
    enum
    {
      ID_SELECT=FXDialogBox::ID_LAST,
      ID_RELOAD,
      ID_BACK_BACK_BACK,
      ID_BACK_BACK,
      ID_BACK,
      ID_FWD_FWD_FWD,
      ID_FWD_FWD,
      ID_FWD,
      ID_WND_SIZE,
      ID_BROWSE,
      ID_WRITE,
      ID_SLIDER,
      ID_HELP,
      ID_LAST
    };
    FXLabel* _sizeLabel;

    VVMovieDialog(){}
    VVMovieDialog(FXWindow*, vox::vvCanvas*);
    virtual ~VVMovieDialog();
    long onCmdSelect(FXObject*,FXSelector,void*);
    long onCmdReload(FXObject*,FXSelector,void*);
    long onCmdBBB(FXObject*,FXSelector,void*);
    long onCmdBB(FXObject*,FXSelector,void*);
    long onCmdB(FXObject*,FXSelector,void*);
    long onCmdFFF(FXObject*,FXSelector,void*);
    long onCmdFF(FXObject*,FXSelector,void*);
    long onCmdF(FXObject*,FXSelector,void*);
    long onCmdSize(FXObject*,FXSelector,void*);
    long onBrowseSelect(FXObject*,FXSelector,void*);
    long onCmdWrite(FXObject*,FXSelector,void*);
    long onCmdHelp(FXObject*,FXSelector,void*);
    long onSliderChng(FXObject*,FXSelector,void*);
    void updateValues();
    void setMovieStep(int);
};

/*********************************************************/

class VVTimeStepDialog : public FXDialogBox
{
  FXDECLARE(VVTimeStepDialog)

  private:
    VVTimeStepDialog(){}
    void setTimeStep(int);

  public:
    enum
    {
      ID_SLIDER=FXDialogBox::ID_LAST,
      ID_BACK_BACK,
      ID_BACK,
      ID_PLAY,
      ID_FWD,
      ID_FWD_FWD,
      ID_LAST
    };
    FXLabel*  _stepLabel;
    FXSlider* _stepSlider;
    vox::vvCanvas* _canvas;
    VVShell* _shell;
    FXButton* _playButton;
    FXTextField* _speedTField;

    VVTimeStepDialog(FXWindow*, vox::vvCanvas*);
    virtual ~VVTimeStepDialog();
    long onCmdBackBack(FXObject*, FXSelector, void*);
    long onCmdBack(FXObject*, FXSelector, void*);
    long onCmdPlay(FXObject*, FXSelector, void*);
    long onCmdFwd(FXObject*, FXSelector, void*);
    long onCmdFwdFwd(FXObject*, FXSelector, void*);
    long onChngSlider(FXObject*, FXSelector, void*);
    void updateValues();
    void stepBack();
    void stepForward();
    void gotoStart();
    void gotoEnd();
    void playback();
    void scaleSpeed(float);
};

/*********************************************************/

class VVHistWindow : public FXDialogBox
{
  FXDECLARE(VVHistWindow)

  private:
    VVHistWindow(){}

  public:
    enum
    {
      ID_CLEAR=FXDialogBox::ID_LAST,
      ID_CANVAS,
      ID_LAST
    };
    std::vector< FXColor > _colorArray;
    std::vector< std::vector< float > > _valArray;
    FXCanvas* _histCanvas;
    size_t _channels;

    VVHistWindow(FXWindow* owner);
    virtual ~VVHistWindow();
    void drawHist();
    void createColorLookup();
    long onClearSelect(FXObject*,FXSelector,void*);
    long onPaint(FXObject*,FXSelector,void*);
};

/*********************************************************/

class VVDiagramDialog : public FXDialogBox
{
  FXDECLARE(VVDiagramDialog)

  private:
    bool _isInteractive;
    VVDiagramDialog(){}

  public:
    enum
    {
      ID_APPLY = FXDialogBox::ID_LAST,
      ID_DEF,
      ID_LAST
    };
    FXTextField* _startXTField;
    FXTextField* _startYTField;
    FXTextField* _startZTField;
    FXTextField* _endXTField;
    FXTextField* _endYTField;
    FXTextField* _endZTField;
    VVShell* _shell;
    VVHistWindow* _histWindow;
    vox::vvCanvas* _canvas;

    VVDiagramDialog(FXWindow*, vox::vvCanvas*);
    virtual ~VVDiagramDialog();
    long onApplySelect(FXObject*,FXSelector,void*);
    long onDefaultSelect(FXObject*,FXSelector,void*);
    void updateValues();
};

/*********************************************************/

class VVGammaDialog : public FXDialogBox
{
  FXDECLARE(VVGammaDialog)

  private:
    VVGammaDialog();
    float getDialValue(FXDial*);
    void  setDialValue(FXDial*, float);

  public:
    enum
    {
      ID_GAMMA = FXDialogBox::ID_LAST,
      ID_GRED,
      ID_GGREEN,
      ID_GBLUE,
      ID_GFOUR,
      ID_DEFAULTS,
      ID_CLOSE
    };
    FXCheckButton* _gammaCheck;
    FXDial* _gRedDial;
    FXDial* _gGreenDial;
    FXDial* _gBlueDial;
    FXDial* _gFourDial;
    FXLabel* _gRedLabel;
    FXLabel* _gGreenLabel;
    FXLabel* _gBlueLabel;
    FXLabel* _gFourLabel;
    vox::vvCanvas* _canvas;
    VVShell* _parent;

    VVGammaDialog(FXWindow*, vox::vvCanvas*);
    long onClose(FXObject*, FXSelector, void*);
    long onSetDefaults(FXObject*, FXSelector, void*);
    long onUseGammaChange(FXObject*, FXSelector, void*);
    long onGammaChange(FXObject*, FXSelector, void*);
    void updateValues();
};

/*********************************************************/

class VVChannel4Dialog : public FXDialogBox
{
  FXDECLARE(VVChannel4Dialog)

  private:
    VVChannel4Dialog(){}

  public:
    using FXDialogBox::show;
    enum
    {
      ID_HUE = FXDialogBox::ID_LAST,
      ID_SAT,
      ID_OK,
      ID_CANCEL
    };
    FXRealSlider* _hueSlider;
    FXRealSlider* _satSlider; // saturation
    FXLabel* _hueLabel;
    FXLabel* _satLabel;
    vox::vvCanvas* _canvas;
    VVShell* _parent;
    float _rBak, _gBak, _bBak;    ///< backup values for cancel

    VVChannel4Dialog(FXWindow*, vox::vvCanvas*);
    long onHueChange(FXObject*, FXSelector, void*);
    long onSatChange(FXObject*, FXSelector, void*);
    long onOK(FXObject*, FXSelector, void*);
    long onCancel(FXObject*, FXSelector, void*);
    void updateValues();
    virtual void show();
};

/*********************************************************/

class VVOpacityDialog : public FXDialogBox
{
  FXDECLARE(VVOpacityDialog)

  private:
    VVOpacityDialog(){}

  public:
    using FXDialogBox::show;
    enum
    {
      ID_RED=FXDialogBox::ID_LAST,
      ID_GREEN,
      ID_BLUE,
      ID_ALPHA,
      ID_ENABLE,
      ID_OK,
      ID_CANCEL
    };
    FXCheckButton* _enableCheck;
    FXRealSlider* _redSlider;
    FXRealSlider* _greenSlider;
    FXRealSlider* _blueSlider;
    FXRealSlider* _alphaSlider;
    FXLabel* _redLabel;
    FXLabel* _greenLabel;
    FXLabel* _blueLabel;
    FXLabel* _alphaLabel;
    vox::vvCanvas* _canvas;
    VVShell* _parent;
    float _rBak, _gBak, _bBak, _aBak;    ///< backup values for cancel

    VVOpacityDialog(FXWindow*, vox::vvCanvas*);
    long onRedChange(FXObject*, FXSelector, void*);
    long onGreenChange(FXObject*, FXSelector, void*);
    long onBlueChange(FXObject*, FXSelector, void*);
    long onAlphaChange(FXObject*, FXSelector, void*);
    long onEnableChange(FXObject*, FXSelector, void*);
    long onOK(FXObject*, FXSelector, void*);
    long onCancel(FXObject*, FXSelector, void*);
    void updateValues();
    virtual void show();
};

/*********************************************************/

class VVChannelDialog : public FXDialogBox
{
  FXDECLARE(VVChannelDialog)

  private:
    VVChannelDialog(){}

  public:
    enum
    {
      ID_RED=FXDialogBox::ID_LAST,
      ID_OK,
      ID_CANCEL,
      ID_SLIDERS_BASE   // this has to be the last ID in the list
    };
    vox::vvCanvas* _canvas;
    VVShell* _parent;
    FXVerticalFrame* _slidersFrame;
    FXRealSlider** _sliders;
    int _numSliders;

    VVChannelDialog(FXWindow*, vox::vvCanvas*);
    long onOK(FXObject*, FXSelector, void*);
    long onCancel(FXObject*, FXSelector, void*);
    void updateValues();
};

/*********************************************************/

class VVFloatRangeDialog : public FXDialogBox
{
  FXDECLARE(VVFloatRangeDialog)

  private:
    VVFloatRangeDialog(){};

  public:
    using FXDialogBox::show;
    enum
    {
      ID_APPLY = FXDialogBox::ID_LAST,
      ID_MIN_DATA,
      ID_MAX_DATA,
      ID_BOT_PERCENT,
      ID_TOP_PERCENT,
      ID_HDR,
      ID_FAST,
      ID_SKIP,
      ID_DUP,
      ID_LOCK,
      ID_OPACITY,
      ID_CLOSE
    };
    FXint _algoType;
    FXDataTarget _algoDataTarget;
    FXTextField* _minTF;
    FXTextField* _maxTF;
    FXTextField* _botClamp;
    FXTextField* _topClamp;
    FXRadioButton* _isoRadio;
    FXRadioButton* _weightRadio;
    FXCheckButton* _hdrCheck;
    FXCheckButton* _fastCheck;
    FXCheckButton* _skipCheck;
    FXCheckButton* _dupCheck;
    FXCheckButton* _lockCheck;
    FXCheckButton* _opacityCheck;
    FXTextField* _fastNumber;
    VVShell* _shell;
    vox::vvCanvas* _canvas;
    float _prevRange[2];

    VVFloatRangeDialog(FXWindow*, vox::vvCanvas*);
    long onClose(FXObject*, FXSelector, void*);
    long onApply(FXObject*, FXSelector, void*);
    long onMinData(FXObject*, FXSelector, void*);
    long onMaxData(FXObject*, FXSelector, void*);
    long onBottomPercent(FXObject*, FXSelector, void*);
    long onTopPercent(FXObject*, FXSelector, void*);
    long onFast(FXObject*, FXSelector, void*);
    long onHDRMapping(FXObject*, FXSelector, void*);
    void updateValues();
    void updateDependents();
    virtual void show();
};

/*********************************************************/

class VVDataTypeDialog : public FXDialogBox
{
  FXDECLARE(VVDataTypeDialog)

  private:
    VVDataTypeDialog(){};

  public:
    enum
    {
      ID_SWAP_ENDIAN = FXDialogBox::ID_LAST,
      ID_SWAP_CHANNELS,
      ID_CNV_8,
      ID_CNV_16,
      ID_CNV_FLOAT,
      ID_CHANNEL_1,
      ID_CHANNEL_2,
      ID_CHANNEL,
      ID_GRADIENT,
      ID_DEL_CHANNEL,
      ID_ADD_CHANNEL,
      ID_ADD_GRADIENT_MAGNITUDE,
      ID_ADD_GRADIENT_VECTOR,
      ID_ADD_VARIANCE
    };
    VVShell* _shell;
    vox::vvCanvas* _canvas;
    FXButton* _bpc8Button;
    FXButton* _bpc16Button;
    FXButton* _bpcFloatButton;
    FXButton* _swapEndianButton;
    FXButton* _swapChannelsButton;
    FXButton* _delChannelButton;
    FXButton* _addChannelButton;
    FXButton* _addGradMagButton;
    FXButton* _addGradVecButton;
    FXButton* _addVarianceButton;
    FXComboBox* _channel1Combo;
    FXComboBox* _channel2Combo;
    FXComboBox* _channelCombo;

    VVDataTypeDialog(FXWindow*, vox::vvCanvas*);
    long onSwapEndian(FXObject*, FXSelector, void*);
    long onSwapChannels(FXObject*, FXSelector, void*);
    long onDeleteChannel(FXObject*, FXSelector, void*);
    long onAddChannel(FXObject*, FXSelector, void*);
    long onAddGradMag(FXObject*, FXSelector, void*);
    long onAddGradVec(FXObject*, FXSelector, void*);
    long onAddVariance(FXObject*, FXSelector, void*);
    long onBytesPerChannel(FXObject*, FXSelector, void*);
    void updateValues();
    void updateDialogs();
};

class VVEditVoxelsDialog : public FXDialogBox
{
  FXDECLARE(VVEditVoxelsDialog)

  private:
    VVEditVoxelsDialog(){};

  public:
    enum
    {
      ID_SHIFT = FXDialogBox::ID_LAST,
      ID_INTERPOL,
      ID_FLIP_X,
      ID_FLIP_Y,
      ID_FLIP_Z,
      ID_CROP,
      ID_ORDER,
      ID_RESIZE
    };
    VVShell* _shell;
    vox::vvCanvas* _canvas;
    FXTextField* _shiftXField;
    FXTextField* _shiftYField;
    FXTextField* _shiftZField;
    FXTextField* _resizeXField;
    FXTextField* _resizeYField;
    FXTextField* _resizeZField;
    FXTextField* _cropXField;
    FXTextField* _cropYField;
    FXTextField* _cropZField;
    FXTextField* _cropWField;
    FXTextField* _cropHField;
    FXTextField* _cropSField;
    FXCheckButton* _interpolType;

    VVEditVoxelsDialog(FXWindow*, vox::vvCanvas*);
    long onShiftVoxels(FXObject*, FXSelector, void*);
    long onResize(FXObject*, FXSelector, void*);
    long onCrop(FXObject*, FXSelector, void*);
    long onFlipX(FXObject*, FXSelector, void*);
    long onFlipY(FXObject*, FXSelector, void*);
    long onFlipZ(FXObject*, FXSelector, void*);
    long onInvertOrder(FXObject*, FXSelector, void*);
    void updateValues();
    void updateDialogs();
};

class VVHeightFieldDialog : public FXDialogBox
{
  FXDECLARE(VVHeightFieldDialog)

  private:
    VVHeightFieldDialog(){};

  public:
    enum
    {
      ID_OK = FXDialogBox::ID_LAST,
      ID_MODE
    };
    VVShell* _shell;
    vox::vvCanvas* _canvas;
    FXTextField* _slicesTF;
    FXComboBox* _modeCombo;

    VVHeightFieldDialog(FXWindow*, vox::vvCanvas*);
    long onOK(FXObject*, FXSelector, void*);
    void updateDialogs();
    void updateValues();
};

class VVGLSettingsDialog : public FXDialogBox 
{
  FXDECLARE(VVGLSettingsDialog)

  private:
    VVGLSettingsDialog(){};

  public:
    VVShell* _shell;

    VVGLSettingsDialog(FXWindow*);
};

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
