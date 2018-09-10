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

// Virvo:
#include <vvopengl.h>
#include <vvdebugmsg.h>
#include <vvfileio.h>
#include <vvtoolshed.h>

// Local:
#include "vvdialogs.h"
#include "vvtranswindow.h"
#include "vvcanvas.h"
#include "vvshell.h"

#include <algorithm>

using namespace vox;

const FXColor VVTransferWindow::BLACK = FXRGB(0,0,0);
const FXColor VVTransferWindow::WHITE = FXRGB(255,255,255);
const float VVTransferWindow::CLICK_TOLERANCE = 0.03f; // [TF space]
const int VVTransferWindow::TF_WIDTH  = 768;
const int VVTransferWindow::TF_HEIGHT = 256;
const int VVTransferWindow::COLORBAR_HEIGHT = 30;
const int VVTransferWindow::BINLIMITS_HEIGHT = 10;

/*******************************************************************************/
FXDEFMAP(VVTransferWindow) VVTransferWindowMap[]=
{
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_PYRAMID,       VVTransferWindow::onCmdPyramid),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_BELL,          VVTransferWindow::onCmdBell),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_CUSTOM,        VVTransferWindow::onCmdCustom),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_SKIP,          VVTransferWindow::onCmdSkip),
  FXMAPFUNC(SEL_CHANGED,           VVTransferWindow::ID_P_TOP_X,       VVTransferWindow::onChngPyramid),
  FXMAPFUNC(SEL_CHANGED,           VVTransferWindow::ID_P_BOTTOM_X,    VVTransferWindow::onChngPyramid),
  FXMAPFUNC(SEL_CHANGED,           VVTransferWindow::ID_P_MAX,         VVTransferWindow::onChngPyramid),
  FXMAPFUNC(SEL_CHANGED,           VVTransferWindow::ID_B_WIDTH,       VVTransferWindow::onChngBell),
  FXMAPFUNC(SEL_CHANGED,           VVTransferWindow::ID_B_MAX,         VVTransferWindow::onChngBell),
  FXMAPFUNC(SEL_CHANGED,           VVTransferWindow::ID_S_WIDTH,       VVTransferWindow::onChngSkip),
  FXMAPFUNC(SEL_CHANGED,           VVTransferWindow::ID_C_WIDTH,       VVTransferWindow::onChngCustomWidth),
  FXMAPFUNC(SEL_CHANGED,           VVTransferWindow::ID_DIS_COLOR,     VVTransferWindow::onChngDisColors),
  FXMAPFUNC(SEL_PAINT,             VVTransferWindow::ID_TF_CANVAS_1D,  VVTransferWindow::onTFCanvasPaint),
  FXMAPFUNC(SEL_PAINT,             VVTransferWindow::ID_TF_CANVAS_2D,  VVTransferWindow::onTFCanvasPaint),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_DELETE,        VVTransferWindow::onCmdDelete),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_UNDO,          VVTransferWindow::onCmdUndo),
  FXMAPFUNC(SEL_LEFTBUTTONPRESS,   VVTransferWindow::ID_TF_CANVAS_1D,  VVTransferWindow::onMouseLDown1D),
  FXMAPFUNC(SEL_LEFTBUTTONRELEASE, VVTransferWindow::ID_TF_CANVAS_1D,  VVTransferWindow::onMouseLUp1D),
  FXMAPFUNC(SEL_MIDDLEBUTTONPRESS,  VVTransferWindow::ID_TF_CANVAS_1D, VVTransferWindow::onMouseMDown1D),
  FXMAPFUNC(SEL_MIDDLEBUTTONRELEASE,VVTransferWindow::ID_TF_CANVAS_1D, VVTransferWindow::onMouseMUp1D),
  FXMAPFUNC(SEL_RIGHTBUTTONPRESS,  VVTransferWindow::ID_TF_CANVAS_1D,  VVTransferWindow::onMouseRDown1D),
  FXMAPFUNC(SEL_RIGHTBUTTONRELEASE,VVTransferWindow::ID_TF_CANVAS_1D,  VVTransferWindow::onMouseRUp1D),
  FXMAPFUNC(SEL_MOTION,            VVTransferWindow::ID_TF_CANVAS_1D,  VVTransferWindow::onMouseMove1D),
  FXMAPFUNC(SEL_MOUSEWHEEL,        VVTransferWindow::ID_TF_CANVAS_1D,  VVTransferWindow::onMouseWheel1D),
  FXMAPFUNC(SEL_LEFTBUTTONPRESS,   VVTransferWindow::ID_TF_CANVAS_2D,  VVTransferWindow::onMouseLDown2D),
  FXMAPFUNC(SEL_LEFTBUTTONRELEASE, VVTransferWindow::ID_TF_CANVAS_2D,  VVTransferWindow::onMouseLUp2D),
  FXMAPFUNC(SEL_MOTION,            VVTransferWindow::ID_TF_CANVAS_2D,  VVTransferWindow::onMouseMove2D),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_COLOR_COMBO,   VVTransferWindow::onCmdColorCombo),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_ALPHA_COMBO,   VVTransferWindow::onCmdAlphaCombo),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_INSTANT,       VVTransferWindow::onCmdInstant),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_OWN_COLOR,     VVTransferWindow::onCmdOwnColor),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_APPLY,         VVTransferWindow::onCmdApply),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_IMPORT_TF,     VVTransferWindow::onCmdImportTF),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_SAVE_TF,       VVTransferWindow::onCmdSaveTF),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_SAVE_TF_BIN,   VVTransferWindow::onCmdSaveTFBin),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_LOAD_TF,       VVTransferWindow::onCmdLoadTF),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_COLOR,         VVTransferWindow::onCmdColor),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_HIST_ALL,      VVTransferWindow::onCmdHistAll),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_HIST_FIRST,    VVTransferWindow::onCmdHistFirst),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_HISTOGRAM,     VVTransferWindow::onCmdHistogram),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_OPACITY,       VVTransferWindow::onCmdOpacity),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_BINS,          VVTransferWindow::onCmdBins),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_PINS,          VVTransferWindow::onCmdPins),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_PICK_COLOR,    VVTransferWindow::onCmdPickColor),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_NORMALIZATION, VVTransferWindow::onCmdNormalization),
  FXMAPFUNC(SEL_CHANGED,           VVTransferWindow::ID_COLOR_PICKER,  VVTransferWindow::onChngPickerColor),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_COLOR_PICKER,  VVTransferWindow::onChngPickerColor),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_NEW_POINT,     VVTransferWindow::onCmdNewPoint),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_DELETE_POINT,  VVTransferWindow::onCmdDeletePoint),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_MIN,           VVTransferWindow::onCmdSetMin),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_MAX,           VVTransferWindow::onCmdSetMax),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_ZOOM_LUT,      VVTransferWindow::onCmdZoomLUT),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_CENTER,        VVTransferWindow::onCmdCenter),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_INVERT,        VVTransferWindow::onCmdInvertAlpha),
  FXMAPFUNC(SEL_COMMAND,           VVTransferWindow::ID_DEFAULT,       VVTransferWindow::onCmdDefault),
};

FXIMPLEMENT(VVTransferWindow,FXDialogBox,VVTransferWindowMap,ARRAYNUMBER(VVTransferWindowMap))

// Construct a dialog box
VVTransferWindow::VVTransferWindow(FXWindow* owner, vvCanvas* c) :
  FXDialogBox(owner, "Transfer Function", DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE, 100, 100)
{
  _canvas = c;
  _shell = (VVShell*)owner;
  _mouseButton = 0;

  FXVerticalFrame* master = new FXVerticalFrame(this,LAYOUT_FILL_X|LAYOUT_FILL_Y);
  _tfBook = new FXTabBook(master,this,ID_TF_BOOK,PACK_UNIFORM_WIDTH|PACK_UNIFORM_HEIGHT|LAYOUT_FILL_X|LAYOUT_FILL_Y|LAYOUT_RIGHT);
  
  // Tab page 1:
  FXTabItem* tab1=new FXTabItem(_tfBook,"&1D Transfer Function",NULL);
  (void)tab1;
  FXVerticalFrame* page1 = new FXVerticalFrame(_tfBook,FRAME_THICK|FRAME_RAISED|LAYOUT_FILL_X|LAYOUT_FILL_Y);

  FXVerticalFrame* glpanel = new FXVerticalFrame(page1, FRAME_SUNKEN|LAYOUT_SIDE_LEFT|LAYOUT_FIX_WIDTH|LAYOUT_FIX_HEIGHT, 0,0,TF_WIDTH,TF_HEIGHT);
  _glVisual1D = new FXGLVisual(getApp(), VISUAL_DOUBLEBUFFER);
  _glCanvas1D = new FXGLCanvas(glpanel, _glVisual1D, this, ID_TF_CANVAS_1D, LAYOUT_FILL_X|LAYOUT_FILL_Y|LAYOUT_TOP|LAYOUT_LEFT);

  FXHorizontalFrame* zoomFrame = new FXHorizontalFrame(page1, LAYOUT_FILL_X);
  _zoomMinButton = new FXButton(zoomFrame, "", NULL, this, ID_MIN, FRAME_RAISED | FRAME_THICK| LAYOUT_LEFT,0,0,0,0,20,20);
  FXHorizontalFrame* realFrame = new FXHorizontalFrame(zoomFrame, LAYOUT_CENTER_X);
  _realMinLabel = new FXLabel(realFrame, "", NULL, LABEL_NORMAL);
  new FXButton(realFrame, "Zoom to range", NULL, this, ID_ZOOM_LUT, FRAME_RAISED | FRAME_THICK);
  _realMaxLabel = new FXLabel(realFrame, "", NULL, LABEL_NORMAL);
  new FXButton(zoomFrame, "Set Defaults", NULL, this, ID_DEFAULT, FRAME_RAISED | FRAME_THICK);
  _centerButton = new FXButton(zoomFrame, "Center origin", NULL, this, ID_CENTER, FRAME_RAISED | FRAME_THICK| LAYOUT_CENTER_X,0,0,0,0,20,20);
  _zoomMaxButton = new FXButton(zoomFrame, "", NULL, this, ID_MAX, FRAME_RAISED | FRAME_THICK | LAYOUT_RIGHT,0,0,0,0,20,20);
  
  // Tab page 2:
  FXTabItem* tab2=new FXTabItem(_tfBook,"&2D Transfer Function",NULL);
  (void)tab2;
  FXVerticalFrame* page2 = new FXVerticalFrame(_tfBook,FRAME_THICK|FRAME_RAISED|LAYOUT_FILL_X|LAYOUT_FILL_Y);
  _glVisual2D = new FXGLVisual(getApp(), VISUAL_DOUBLEBUFFER);
  _glCanvas2D = new FXGLCanvas(page2, _glVisual2D, this, ID_TF_CANVAS_2D, FRAME_SUNKEN | LAYOUT_FIX_HEIGHT | LAYOUT_FIX_WIDTH, 0, 0, TF_WIDTH,TF_HEIGHT);

  // Common elements:
  FXHorizontalFrame* buttonFrame = new FXHorizontalFrame(master, LAYOUT_FILL_Y | LAYOUT_CENTER_X | PACK_UNIFORM_WIDTH);
  new FXButton(buttonFrame,"Color",      NULL,this,ID_COLOR,  FRAME_RAISED|FRAME_THICK,0,0,0,0,20,20);   // sets width for all buttons
  new FXButton(buttonFrame,"Pyramid",    NULL,this,ID_PYRAMID,FRAME_RAISED|FRAME_THICK);
  new FXButton(buttonFrame,"Gaussian",   NULL,this,ID_BELL,   FRAME_RAISED|FRAME_THICK);
  new FXButton(buttonFrame,"Custom",     NULL,this,ID_CUSTOM, FRAME_RAISED|FRAME_THICK);
  new FXButton(buttonFrame,"Skip Range", NULL,this,ID_SKIP,   FRAME_RAISED|FRAME_THICK);
  new FXButton(buttonFrame,"Delete",     NULL,this,ID_DELETE, FRAME_RAISED|FRAME_THICK);
  new FXButton(buttonFrame,"Undo",       NULL,this,ID_UNDO,   FRAME_RAISED|FRAME_THICK);

  FXHorizontalFrame* controlFrame=new FXHorizontalFrame(master,LAYOUT_FILL_X);
  FXVerticalFrame* comboFrame=new FXVerticalFrame(controlFrame,LAYOUT_FILL_Y);

  FXGroupBox *colorComboGP=new FXGroupBox(comboFrame,"Preset Colors", FRAME_GROOVE | LAYOUT_FILL_X);
  _colorCombo=new FXComboBox(colorComboGP,5,this,ID_COLOR_COMBO,COMBOBOX_INSERT_LAST|COMBOBOX_STATIC|FRAME_SUNKEN|FRAME_THICK|LAYOUT_FILL_X);
  _colorCombo->appendItem("Bright Colors");
  _colorCombo->appendItem("Hue Gradient");
  _colorCombo->appendItem("Grayscale");
  _colorCombo->appendItem("White");
  _colorCombo->appendItem("Red");
  _colorCombo->appendItem("Green");
  _colorCombo->appendItem("Blue");
  _colorCombo->setNumVisible(_colorCombo->getNumItems());
  _colorCombo->setCurrentItem(0);

  FXGroupBox *alphaComboGP=new FXGroupBox(comboFrame,"Preset Alpha", FRAME_GROOVE | LAYOUT_FILL_X);
  _alphaCombo=new FXComboBox(alphaComboGP,5,this,ID_ALPHA_COMBO,COMBOBOX_INSERT_LAST|COMBOBOX_STATIC|FRAME_SUNKEN|FRAME_THICK|LAYOUT_FILL_X);
  _alphaCombo->appendItem("Ascending");
  _alphaCombo->appendItem("Descending");
  _alphaCombo->appendItem("Opaque");
  _alphaCombo->setNumVisible(_alphaCombo->getNumItems());
  _alphaCombo->setCurrentItem(0);

  _mousePosLabel = new FXLabel(comboFrame, "", NULL, LABEL_NORMAL | LAYOUT_LEFT);
  _pinPosLabel = new FXLabel(comboFrame, "", NULL, LABEL_NORMAL | LAYOUT_LEFT);

  _pinSwitcher = new FXSwitcher(controlFrame, LAYOUT_FILL_X | LAYOUT_FILL_Y);

  // Switcher state #0:
  new FXVerticalFrame(_pinSwitcher, LAYOUT_FILL_X | LAYOUT_FILL_Y);

  // Switcher state #1:
  FXGroupBox* pinGroup1 = new FXGroupBox(_pinSwitcher,"Color settings",FRAME_GROOVE|LAYOUT_FILL_X|LAYOUT_FILL_Y);
  FXVerticalFrame* colorFrame = new FXVerticalFrame(pinGroup1, LAYOUT_FILL_X);
  new FXButton(colorFrame,"Select color",NULL,this,ID_PICK_COLOR, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X, 0, 0, 0, 0, 20, 20);

  // Switcher state #2:
  FXGroupBox* pinGroup2 = new FXGroupBox(_pinSwitcher,"Pyramid settings",FRAME_GROOVE|LAYOUT_FILL_X|LAYOUT_FILL_Y);
  FXVerticalFrame* sliderFrame1 = new FXVerticalFrame(pinGroup2, LAYOUT_FILL_X);

  FXMatrix* pTopXMat = new FXMatrix(sliderFrame1, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(pTopXMat, "Top width X: ",NULL,LABEL_NORMAL);
  _pTopXLabel = new FXLabel(pTopXMat, "0",NULL,LABEL_NORMAL);
  _pTopXSlider=new FXRealSlider(sliderFrame1,this,ID_P_TOP_X,SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|LAYOUT_FILL_X|LAYOUT_FILL_COLUMN);
  _pTopXSlider->setRange(0.0f, 2.0f);
  _pTopXSlider->setValue(0.0f);
  _pTopXSlider->setTickDelta(.01);

  FXMatrix* pBottomXMat = new FXMatrix(sliderFrame1, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(pBottomXMat, "Bottom width X: ",NULL,LABEL_NORMAL);
  _pBottomXLabel = new FXLabel(pBottomXMat, "0",NULL,LABEL_NORMAL);
  _pBottomXSlider=new FXRealSlider(sliderFrame1,this,ID_P_BOTTOM_X, SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|LAYOUT_FILL_X|LAYOUT_FILL_COLUMN);
  _pBottomXSlider->setRange(0.0f, 2.0f);
  _pBottomXSlider->setValue(0.0f);
  _pBottomXSlider->setTickDelta(.01);

  FXMatrix* pMaxMat = new FXMatrix(sliderFrame1, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(pMaxMat, "Maximum opacity: ",NULL,LABEL_NORMAL);
  _pMaxLabel = new FXLabel(pMaxMat, "0",NULL,LABEL_NORMAL);
  _pMaxSlider=new FXRealSlider(sliderFrame1,this,ID_P_MAX,SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|LAYOUT_FILL_X|LAYOUT_FILL_COLUMN);
  _pMaxSlider->setRange(0.0f, 1.0f);
  _pMaxSlider->setValue(0.0f);
  _pMaxSlider->setTickDelta(.01);

  FXHorizontalFrame* pColorFrame = new FXHorizontalFrame(sliderFrame1);
  _pColorButton = new FXCheckButton(pColorFrame,"Has own color",this,ID_OWN_COLOR,ICON_BEFORE_TEXT|LAYOUT_LEFT);
  new FXButton(pColorFrame,"Pick color",NULL,this,ID_PICK_COLOR, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X, 0, 0, 0, 0, 20, 20);

  // Switcher state #3:
  FXGroupBox* pinGroup3 = new FXGroupBox(_pinSwitcher,"Gaussian settings",FRAME_GROOVE|LAYOUT_FILL_X|LAYOUT_FILL_Y);
  FXVerticalFrame* sliderFrame2 = new FXVerticalFrame(pinGroup3, LAYOUT_FILL_X);

  FXMatrix* _bWidthMat = new FXMatrix(sliderFrame2, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(_bWidthMat, "Width: ",NULL,LABEL_NORMAL);
  _bWidthLabel = new FXLabel(_bWidthMat, "0",NULL,LABEL_NORMAL);
  _bWidthSlider=new FXRealSlider(sliderFrame2,this,ID_B_WIDTH,SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|LAYOUT_FILL_X|LAYOUT_FILL_COLUMN);
  _bWidthSlider->setRange(0.0f, 1.0f);
  _bWidthSlider->setValue(0.0f);
  _bWidthSlider->setTickDelta(.01);

  FXMatrix* _bMaxMat = new FXMatrix(sliderFrame2, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(_bMaxMat, "Maximum value: ",NULL,LABEL_NORMAL);
  _bMaxLabel = new FXLabel(_bMaxMat, "0",NULL,LABEL_NORMAL);
  _bMaxSlider=new FXRealSlider(sliderFrame2,this,ID_B_MAX,SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|LAYOUT_FILL_X|LAYOUT_FILL_COLUMN);
  _bMaxSlider->setRange(0.0f, 5.0f);
  _bMaxSlider->setValue(0.0f);
  _bMaxSlider->setTickDelta(.01);

  FXHorizontalFrame* bColorFrame = new FXHorizontalFrame(sliderFrame2);
  _bColorButton = new FXCheckButton(bColorFrame,"Has own color",this,ID_OWN_COLOR,ICON_BEFORE_TEXT|LAYOUT_LEFT);
  new FXButton(bColorFrame,"Pick color",NULL,this,ID_PICK_COLOR, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X, 0, 0, 0, 0, 20, 20);

  // Switcher state #4: skip range widget activated
  FXGroupBox* pinGroup4 = new FXGroupBox(_pinSwitcher,"Skip Range Settings",FRAME_GROOVE|LAYOUT_FILL_X|LAYOUT_FILL_Y);
  FXVerticalFrame* sliderFrame3 = new FXVerticalFrame(pinGroup4, LAYOUT_FILL_X);

  FXMatrix* _sWidthMat = new FXMatrix(sliderFrame3, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(_sWidthMat, "Width: ",NULL,LABEL_NORMAL);
  _sWidthLabel = new FXLabel(_sWidthMat, "0",NULL,LABEL_NORMAL);
  _sWidthSlider=new FXRealSlider(sliderFrame3,this,ID_S_WIDTH,SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|LAYOUT_FILL_X|LAYOUT_FILL_COLUMN);
  _sWidthSlider->setRange(0.0f, 1.0f);
  _sWidthSlider->setValue(0.0f);
  _sWidthSlider->setTickDelta(.01);

  // Switcher state #5: custom widget activated
  FXGroupBox* pinGroup5 = new FXGroupBox(_pinSwitcher,"Custom Widget Settings",FRAME_GROOVE|LAYOUT_FILL_X|LAYOUT_FILL_Y);
  FXVerticalFrame* sliderFrame5 = new FXVerticalFrame(pinGroup5, LAYOUT_FILL_X);

  FXMatrix* _cWidthMat = new FXMatrix(sliderFrame5, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(_cWidthMat, "Width: ",NULL,LABEL_NORMAL);
  _cWidthLabel = new FXLabel(_cWidthMat, "0",NULL,LABEL_NORMAL);
  _cWidthSlider=new FXRealSlider(sliderFrame5,this,ID_C_WIDTH,SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|LAYOUT_FILL_X|LAYOUT_FILL_COLUMN);
  _cWidthSlider->setRange(0.0f, 1.0f);
  _cWidthSlider->setValue(0.0f);
  _cWidthSlider->setTickDelta(.01);

  new FXButton(sliderFrame5,"New control point",NULL,this,ID_NEW_POINT, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X, 0, 0, 0, 0, 20, 20);
  new FXButton(sliderFrame5,"Delete control point",NULL,this,ID_DELETE_POINT, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X, 0, 0, 0, 0, 20, 20);

  // Continue with pin independent widgets:
  FXHorizontalFrame* checkboxFrame=new FXHorizontalFrame(master, LAYOUT_FILL_X | PACK_UNIFORM_WIDTH);
  _instantButton = new FXCheckButton(checkboxFrame,"Instant Classification",this,ID_INSTANT,ICON_BEFORE_TEXT|LAYOUT_LEFT);
  _cbNorm = new FXCheckButton(checkboxFrame, "Logarithmic histogram normalization", this, ID_NORMALIZATION, ICON_BEFORE_TEXT|LAYOUT_LEFT);
  _cbNorm->setCheck(true);

  FXGroupBox* histoGroup = new FXGroupBox(master,"Display options", FRAME_GROOVE | LAYOUT_FILL_X);
  FXHorizontalFrame* miscFrame = new FXHorizontalFrame(histoGroup);
  _opaCheck = new FXCheckButton(miscFrame,"Opacity",this,ID_OPACITY, ICON_BEFORE_TEXT);
  _opaCheck->setCheck(true);
  _binsCheck = new FXCheckButton(miscFrame, "Show bin limits", this, ID_BINS, ICON_BEFORE_TEXT);
  _binsCheck->setCheck(false);
  _pinsCheck = new FXCheckButton(miscFrame, "Show pin lines", this, ID_PINS, ICON_BEFORE_TEXT);
  _pinsCheck->setCheck(true);
  _invertCheck = new FXCheckButton(miscFrame, "Invert opacity bar", this, ID_INVERT, ICON_BEFORE_TEXT);
  int alpha = getApp()->reg().readIntEntry("Settings", "InvertAlpha", 0);
  _invertCheck->setCheck((alpha==0) ? false : true);
  FXHorizontalFrame* histoFrame = new FXHorizontalFrame(histoGroup);
  _histoCheck = new FXCheckButton(histoFrame,"Histogram",this,ID_HISTOGRAM, ICON_BEFORE_TEXT);
  _histoCheck->setCheck(true);
  _histFirst = new FXRadioButton(histoFrame,"Histogram for first time step",this,ID_HIST_FIRST, ICON_BEFORE_TEXT);
  _histFirst->setCheck(true);
  _histAll = new FXRadioButton(histoFrame,"Histogram for all time steps",this,ID_HIST_ALL, ICON_BEFORE_TEXT);

  FXHorizontalFrame* disColorFrame = new FXHorizontalFrame(master,LAYOUT_FILL_X);
  new FXLabel(disColorFrame, "Discrete Colors:",NULL,LABEL_NORMAL);
  _disColorSlider = new FXSlider(disColorFrame,this, ID_DIS_COLOR, SLIDER_HORIZONTAL | SLIDER_ARROW_DOWN | LAYOUT_FILL_X);
  _disColorSlider->setRange(0,64);
  _disColorSlider->setValue(0);
  _disColorSlider->setTickDelta(1);
  _disColorLabel = new FXLabel(disColorFrame, "",NULL,LABEL_NORMAL);

  FXGroupBox* tfGroup = new FXGroupBox(master,"Transfer Function I/O",FRAME_GROOVE|LAYOUT_FILL_X|LAYOUT_FILL_Y);
  FXHorizontalFrame* tfFrame=new FXHorizontalFrame(tfGroup, LAYOUT_FILL_X | PACK_UNIFORM_WIDTH);
  new FXButton(tfFrame,"Save TF",NULL,this,ID_SAVE_TF, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X, 0, 0, 0, 0, 20, 20);
  new FXButton(tfFrame,"Save TF Bins",NULL,this,ID_SAVE_TF_BIN, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X, 0, 0, 0, 0, 20, 20);
  new FXButton(tfFrame,"Load TF",NULL,this,ID_LOAD_TF, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X, 0, 0, 0, 0, 20, 20);
  new FXButton(tfFrame,"Import TF",NULL,this,ID_IMPORT_TF, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X, 0, 0, 0, 0, 20, 20);
  
  FXHorizontalFrame* endFrame=new FXHorizontalFrame(master, LAYOUT_FILL_X | PACK_UNIFORM_WIDTH);
  new FXButton(endFrame,"Apply",NULL,this,ID_APPLY, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X, 0, 0, 0, 0, 20, 20);
  new FXButton(endFrame,"Close",NULL,this,ID_ACCEPT, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X, 0, 0, 0, 0, 20, 20);

  // Initialize color picker:
  _colorPicker  = new FXColorDialog((FXWindow*)this, "Pin Color",DECOR_TITLE|DECOR_BORDER, 50,50);
  _colorPicker->setTarget(this);
  _colorPicker->setSelector(ID_COLOR_PICKER);
  _colorPicker->setOpaqueOnly(true);

  _currentWidget = NULL;
  _histoTexture1D = NULL;
  _histoTexture2D = NULL;
  _dataZoom[0] = 0.0f;
  _dataZoom[1] = 1.0f;
  _is1DHistogramDirty = true;
  _is2DHistogramDirty = true;
}

// Must delete the menus
VVTransferWindow::~VVTransferWindow()
{
  delete _glVisual1D;
  delete _glVisual2D;
  delete _histoTexture1D;
  delete _histoTexture2D;
}

void VVTransferWindow::initGL()
{
  glDrawBuffer(GL_BACK);         // set draw buffer to front in order to read image data
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0f, 1.0f, 0.0f, 1.0f, 1.0f, -1.0f);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

long VVTransferWindow::onTFCanvasPaint(FXObject*,FXSelector,void*)
{
  if (_glCanvas1D->makeCurrent())
  {
    initGL();
    _glCanvas1D->makeNonCurrent();
  }
  if (_glCanvas2D->makeCurrent())
  {
    initGL();
    _glCanvas2D->makeNonCurrent();
  }
  drawTF();
  return 1;
}

long VVTransferWindow::onCmdColor(FXObject*,FXSelector,void*)
{
  newWidget(COLOR);
  return 1;
}

long VVTransferWindow::onCmdPyramid(FXObject*,FXSelector,void*)
{
  newWidget(PYRAMID);
  return 1;
}

long VVTransferWindow::onCmdBell(FXObject*,FXSelector,void*)
{
  newWidget(BELL);
  return 1;
}

long VVTransferWindow::onCmdSkip(FXObject*,FXSelector,void*)
{
  newWidget(SKIP);
  return 1;
}

long VVTransferWindow::onCmdCustom(FXObject*,FXSelector,void*)
{
  newWidget(CUSTOM);
  return 1;
}

long VVTransferWindow::onCmdNewPoint(FXObject*,FXSelector,void*)
{
  vvTFCustom* cuw;
  if ((cuw = dynamic_cast<vvTFCustom*>(_currentWidget)) != NULL)
  {
    cuw->_currentPoint = cuw->addPoint(cuw->_pos[0]);  // add point in middle of widget 
    if(_instantButton->getCheck()) updateTransFunc();
    drawTF();    
  }
  return 1;
}

long VVTransferWindow::onCmdDeletePoint(FXObject*,FXSelector,void*)
{
  std::list<vvTFPoint*>::iterator iter;
  vvTFCustom* cuw;
  
  if ((cuw = dynamic_cast<vvTFCustom*>(_currentWidget)) != NULL)
  {
    cuw->removeCurrentPoint();
    if(_instantButton->getCheck()) updateTransFunc();
    drawTF();    
  }
  return 1;
}

long VVTransferWindow::onCmdPickColor(FXObject*,FXSelector,void*)
{
  float r=0.0f, g=0.0f, b=0.0f;
  vvTFColor* cw=NULL;
  vvTFPyramid* pw=NULL;
  vvTFBell* bw=NULL;

  if(!_currentWidget) return 1;
  
  // Find out widget type:
  if ((cw = dynamic_cast<vvTFColor*>(_currentWidget))   != NULL ||
      (pw = dynamic_cast<vvTFPyramid*>(_currentWidget)) != NULL ||
      (bw = dynamic_cast<vvTFBell*>(_currentWidget))    != NULL)
  {
    // Get widget color:
    if (cw) cw->_col.getRGB(r,g,b);
    else if (pw) pw->_col.getRGB(r,g,b);
    else if (bw) bw->_col.getRGB(r,g,b);
    else assert(0);

    // Set color picker color:   
    _colorPicker->setRGBA(FXRGBA(r*255,g*255,b*255,255));
    if (_colorPicker->execute() == 0)   // has picker exited with 'cancel'?
    {
      // Undo changes to color:
      if (cw) cw->_col.setRGB(r,g,b);
      else if (pw) pw->_col.setRGB(r,g,b);
      else if (bw) bw->_col.setRGB(r,g,b);
       drawTF();
      if(_instantButton->getCheck()) updateTransFunc();
    }
  }
  return 1;
}

long VVTransferWindow::onChngPickerColor(FXObject*,FXSelector,void*)
{
  vvTFColor* cw=NULL;
  vvTFPyramid* pw=NULL;
  vvTFBell* bw=NULL;

  if(!_currentWidget) return 1;
  
  // Find out widget type:
  if ((cw = dynamic_cast<vvTFColor*>(_currentWidget))   != NULL ||
      (pw = dynamic_cast<vvTFPyramid*>(_currentWidget)) != NULL ||
      (bw = dynamic_cast<vvTFBell*>(_currentWidget))    != NULL)
  {
    FXColor col = _colorPicker->getRGBA();
    float r = float(FXREDVAL(col))   / 255.0f;
    float g = float(FXGREENVAL(col)) / 255.0f;
    float b = float(FXBLUEVAL(col))  / 255.0f;
    if (cw) cw->_col.setRGB(r,g,b);
    else if (pw) pw->_col.setRGB(r,g,b);
    else if (bw) bw->_col.setRGB(r,g,b);
    drawTF();
    if(_instantButton->getCheck()) updateTransFunc();
  }  
 
  return 1;
}

long VVTransferWindow::onChngPyramid(FXObject*,FXSelector,void*)
{
  if(!_currentWidget)
  {
    _pTopXLabel->setText(FXStringFormat("%.5g", _pTopXSlider->getValue()));
    _pBottomXLabel->setText(FXStringFormat("%.5g", _pBottomXSlider->getValue()));
    _pMaxLabel->setText(FXStringFormat("%.5g",_pMaxSlider->getValue()));
    return 1;
  }
  vvTFPyramid *pw=dynamic_cast<vvTFPyramid*>(_currentWidget);
  assert(pw);
  pw->_top[0] = normd2datad(_pTopXSlider->getValue());
  pw->_bottom[0] = normd2datad(_pBottomXSlider->getValue());
  pw->_opacity = _pMaxSlider->getValue();
  _pTopXLabel->setText(FXStringFormat("%.5g", pw->_top[0]));
  _pBottomXLabel->setText(FXStringFormat("%.5g", pw->_bottom[0]));
  _pMaxLabel->setText(FXStringFormat("%.5g", pw->_opacity));
  drawTF();
  if(_instantButton->getCheck()) updateTransFunc();
  return 1;
}

long VVTransferWindow::onChngBell(FXObject*,FXSelector,void*)
{
  if(!_currentWidget)
  {
    _bWidthLabel->setText(FXStringFormat("%.5g", _bWidthSlider->getValue()));
    _bMaxLabel->setText(FXStringFormat("%.5g", _bMaxSlider->getValue()));
    return 1;
  }
  vvTFBell *bw=dynamic_cast<vvTFBell*>(_currentWidget);
  assert(bw);
  bw->_size[0] = _bWidthSlider->getValue();
  bw->_opacity = _bMaxSlider->getValue();
  _bWidthLabel->setText(FXStringFormat("%.5g", bw->_size[0]));
  _bMaxLabel->setText(FXStringFormat("%.5g", bw->_opacity));
  drawTF();
  if(_instantButton->getCheck()) updateTransFunc();
  return 1;
}

long VVTransferWindow::onChngSkip(FXObject*,FXSelector,void*)
{
  if(!_currentWidget) 
  {
    _sWidthLabel->setText(FXStringFormat("%.2g", _sWidthSlider->getValue()));
    return 1;
  }
  vvTFSkip *sw=dynamic_cast<vvTFSkip*>(_currentWidget);
  assert(sw);
  sw->_size[0] = normd2datad(_sWidthSlider->getValue());
  _sWidthLabel->setText(FXStringFormat("%.5g", sw->_size[0]));
  drawTF();
  if(_instantButton->getCheck()) updateTransFunc();
  return 1;
}

long VVTransferWindow::onChngCustomWidth(FXObject*,FXSelector,void*)
{
  if(!_currentWidget) 
  {
    _cWidthLabel->setText(FXStringFormat("%.2g", _cWidthSlider->getValue()));
    return 1;
  }
  vvTFCustom *cuw=dynamic_cast<vvTFCustom*>(_currentWidget);
  assert(cuw);
  cuw->setSize(normd2datad(_cWidthSlider->getValue()));
  _cWidthLabel->setText(FXStringFormat("%.2g", cuw->_size[0]));
  drawTF();
  if(_instantButton->getCheck()) updateTransFunc();
  return 1;
}

long VVTransferWindow::onMouseLDown1D(FXObject*,FXSelector,void* ptr)
{
  _mouseButton = 1;
  if(!_canvas || _canvas->_vd->tf[0]._widgets.size() == 0)
  {
    _currentWidget = NULL;
    return 1;
  }
  _canvas->_vd->tf[0].putUndoBuffer();
  _glCanvas1D->grab();
  FXEvent* ev = (FXEvent*)ptr;
  float normX = float(ev->win_x) / float(_glCanvas1D->getWidth());
  _currentWidget = closestWidget(norm2data(normX), 1.0f - float(ev->win_y) / float(_glCanvas1D->getHeight()), -1.0f);
  updateLabels();
  drawTF();
  return 1;
}

long VVTransferWindow::onMouseLUp1D(FXObject*,FXSelector,void*)
{
  _mouseButton = 0;
  _glCanvas1D->ungrab();
  return 1;
}

long VVTransferWindow::onMouseMDown1D(FXObject*,FXSelector,void*)
{
  _mouseButton = 2;
  _glCanvas1D->grab();
  return 1;
}

long VVTransferWindow::onMouseMUp1D(FXObject*,FXSelector,void*)
{
  _mouseButton = 0;
  _glCanvas1D->ungrab();
  return 1;
}

long VVTransferWindow::onMouseRDown1D(FXObject*,FXSelector,void* ptr)
{
  vvTFCustom* cuw;
  
  _mouseButton = 3;
  _canvas->_vd->tf[0].putUndoBuffer();
  _glCanvas1D->grab();
  FXEvent* ev = (FXEvent*)ptr;
  if ((cuw=dynamic_cast<vvTFCustom*>(_currentWidget))!=NULL)  // is current widget of custom type?
  {
    if (cuw->selectPoint(1.0f - (float(ev->win_y - COLORBAR_HEIGHT - BINLIMITS_HEIGHT) / 
                                 float(_glCanvas1D->getHeight() - COLORBAR_HEIGHT - BINLIMITS_HEIGHT)), normd2datad(CLICK_TOLERANCE), 
                         norm2data(float(ev->win_x) / float(_glCanvas1D->getWidth())), normd2datad(CLICK_TOLERANCE)) == NULL)
    {
      cuw->_currentPoint = NULL;
    }
  } 
  drawTF();
  return 1;
}

long VVTransferWindow::onMouseRUp1D(FXObject*,FXSelector,void*)
{
  _mouseButton = 0;
  _glCanvas1D->ungrab();
  return 1;
}

long VVTransferWindow::onMouseMove1D(FXObject*, FXSelector, void* ptr)
{
  vvTFCustom* cuw;
  float dx, dy;   // distance mouse moved in TF space since previous callback [0..1]
  float mousePosVal;
  
  FXEvent* ev = (FXEvent*)ptr;
  float pos = ts_clamp(float(ev->win_x) / float(_glCanvas1D->getWidth()), 0.0f, 1.0f);
  mousePosVal = norm2data(pos);
  _mousePosLabel->setText(FXStringFormat("Mouse: %.5g", mousePosVal));

  if (_glCanvas1D->grabbed())
  {
    if (_mouseButton==1)
    {
      if(!_currentWidget || !_canvas) return 1;
      if(_canvas->_vd->tf[0]._widgets.size() == 0) return 1;
      _pinPosLabel->setText(FXStringFormat("Pin: %.5g", mousePosVal));
      _currentWidget->_pos[0] = mousePosVal;
    }
    else if (_mouseButton==2)   // pan TF area left/right
    {
      dx = normd2datad((float(ev->win_x - ev->last_x) / float(_glCanvas1D->getWidth())));
      _dataZoom[0] -= dx;
      _dataZoom[1] -= dx;
      setDirtyHistogram();
      updateValues();
    }
    else if (_mouseButton==3)
    {
      if ((cuw=dynamic_cast<vvTFCustom*>(_currentWidget))!=NULL)  // is current widget of custom type?
      {
        dx =  (float(ev->win_x - ev->last_x) / float(_glCanvas1D->getWidth())) * (_dataZoom[1] - _dataZoom[0]);
        dy = - float(ev->win_y - ev->last_y) / float(_glCanvas1D->getHeight() - COLORBAR_HEIGHT - BINLIMITS_HEIGHT);
        cuw->moveCurrentPoint(dy, dx);
      }
    }
    drawTF();
    if(_instantButton->getCheck()) updateTransFunc();
  }
  return 1;
}

long VVTransferWindow::onMouseLDown2D(FXObject*,FXSelector,void* ptr)
{
  if(!_canvas || _canvas->_vd->tf[0]._widgets.size() == 0)
  {
    _currentWidget = NULL;
    return 1;
  }
  _canvas->_vd->tf[0].putUndoBuffer();
  _glCanvas2D->grab();
  FXEvent* ev = (FXEvent*)ptr;
  _currentWidget = closestWidget(float(ev->win_x) / float(_glCanvas1D->getWidth()), 
                                 1.0f - float(ev->win_y) / float(_glCanvas2D->getHeight()), 
                                 -1.0f);
  updateLabels();
  draw2DTF();
  return 1;
}

long VVTransferWindow::onMouseLUp2D(FXObject*,FXSelector,void*)
{
  _glCanvas2D->ungrab();
  return 1;
}

long VVTransferWindow::onMouseMove2D(FXObject*, FXSelector, void* ptr)
{
  if (!_glCanvas2D->grabbed()) return 1;
  if(!_currentWidget || !_canvas) return 1;
  if(_canvas->_vd->tf[0]._widgets.size() == 0) return 1;
  FXEvent* ev = (FXEvent*)ptr;
  float xPos = ts_clamp(float(ev->win_x) / float(_glCanvas2D->getWidth()),  0.0f, 1.0f);
  float yPos = ts_clamp(1.0f - float(ev->win_y) / float(_glCanvas2D->getHeight()), 0.0f, 1.0f);
  _currentWidget->_pos[0] = xPos;
  _currentWidget->_pos[1] = yPos;
  draw2DTF();
  if(_instantButton->getCheck()) updateTransFunc();
  return 1;
}

/** One mouse wheel click has a value of +120 when rotating up, -120 when rotating down.
*/
long VVTransferWindow::onMouseWheel1D(FXObject*, FXSelector sel, void* ptr)
{
  (void)sel;
  const float ZOOM_PER_CLICK = 0.1f;
  float diff;
  float mousePos;
  float factor;
  
  FXEvent* ev = (FXEvent*)ptr;
  float xPos = ts_clamp(float(ev->win_x) / float(_glCanvas2D->getWidth()),  0.0f, 1.0f);
  int rot = ev->code / 120;  // normalize to 1 unit per click
  
  // Calculate new data zoom values while keeping mouse pointer position fixed.
  factor = 1.0f - rot * ZOOM_PER_CLICK;
  mousePos = norm2data(xPos);

  // dataZoomMin:
  diff = mousePos - _dataZoom[0];
  diff *= factor;   // calculate new distance between min and mousePos
  _dataZoom[0] = mousePos - diff;
  
  // dataZoomMax:
  diff = _dataZoom[1] - mousePos;
  diff *= factor;   // calculate new distance between max and mousePos
  _dataZoom[1] = mousePos + diff;

  setDirtyHistogram();
  updateValues();
  
  return 1;
}

void VVTransferWindow::updateTransFunc()
{
  _shell->_glcanvas->makeCurrent();
  _canvas->_renderer->updateTransferFunction();
  _shell->_glcanvas->makeNonCurrent();
}

long VVTransferWindow::onCmdDelete(FXObject*,FXSelector,void*)
{
  if(_canvas->_vd->tf[0]._widgets.size() == 0 || _currentWidget == NULL) return 1;
  _canvas->_vd->tf[0].putUndoBuffer();
  _canvas->_vd->tf[0]._widgets.erase(std::find(_canvas->_vd->tf[0]._widgets.begin(), _canvas->_vd->tf[0]._widgets.end(), _currentWidget));
  _currentWidget = NULL;
  drawTF();
  updateLabels();
  if(_instantButton->getCheck()) updateTransFunc();
  return 1;
}

long VVTransferWindow::onCmdUndo(FXObject*,FXSelector,void*)
{
  _canvas->_vd->tf[0].getUndoBuffer();
  _currentWidget = NULL;
  drawTF();
  updateLabels();
  if(_instantButton->getCheck()) updateTransFunc();
  return 1;
}

long VVTransferWindow::onCmdColorCombo(FXObject*,FXSelector,void*)
{
  _canvas->_vd->tf[0].putUndoBuffer();
  _canvas->_vd->tf[0].setDefaultColors(_colorCombo->getCurrentItem(), _dataZoom[0], _dataZoom[1]);
  _currentWidget = NULL;
  drawTF();
  updateLabels();
  if(_instantButton->getCheck()) updateTransFunc();
  return 1;
}

long VVTransferWindow::onCmdAlphaCombo(FXObject*,FXSelector,void*)
{
  _canvas->_vd->tf[0].putUndoBuffer();
  _canvas->_vd->tf[0].setDefaultAlpha(_alphaCombo->getCurrentItem(), _dataZoom[0], _dataZoom[1]);
  _currentWidget = NULL;
  drawTF();
  updateLabels();
  if(_instantButton->getCheck()) updateTransFunc();
  return 1;
}

long VVTransferWindow::onCmdInstant(FXObject*,FXSelector,void* ptr)
{
  if(ptr != 0) updateTransFunc();
  return 1;
}

long VVTransferWindow::onCmdOwnColor(FXObject*,FXSelector,void* ptr)
{
  vvTFPyramid* pw;
  vvTFBell* bw;

  if ((pw = dynamic_cast<vvTFPyramid*>(_currentWidget)) != NULL)
  {
    pw->setOwnColor(ptr != NULL);
  }
  else if ((bw = dynamic_cast<vvTFBell*>(_currentWidget)) != NULL)
  {
    bw->setOwnColor(ptr != NULL);
  }
  
  drawTF();
  if(_instantButton->getCheck()) updateTransFunc();
  return 1;
}

long VVTransferWindow::onCmdApply(FXObject*,FXSelector,void*)
{
  updateTransFunc();
  return 1;
}

long VVTransferWindow::onCmdImportTF(FXObject*,FXSelector,void*)
{
  const FXchar patterns[]="XVF files (*.xvf)\nAll Files (*.*)";
  FXString filename = _shell->getOpenFilename("Import Transfer Function", patterns);
  if(filename.length() > 0)
  {
    vvFileIO* fio = new vvFileIO();
    if (fio->importTF(_canvas->_vd, filename.text())==vvFileIO::OK)
    {
      _currentWidget = NULL;
      updateTransFunc();
      drawTF();
      updateLabels();
    }
    delete fio;
  }
  return 1;
}

long VVTransferWindow::onCmdSaveTF(FXObject*,FXSelector,void*)
{
  const FXchar patterns[]="Transfer function files (*.vtf)\nAll Files (*.*)";
  FXString filename = _shell->getSaveFilename("Save Transfer Function", "transfunc.vtf", patterns);
  if(filename.length() == 0) return 1;
  if(vvToolshed::isFile(filename.text()))
  {
    int over = FXMessageBox::question((FXWindow*)this, MBOX_OK_CANCEL, "Warning", "Overwrite existing file?");
    if(over == FX::MBOX_CLICKED_CANCEL) return 1;
  }
  _canvas->_vd->tf[0].save(filename.text());
  return 1;
}

long VVTransferWindow::onCmdSaveTFBin(FXObject*, FXSelector, void*)
{
  const FXchar patterns[]="Transfer function files (*.vtf)\nAll Files (*.*)";
  FXString filename = _shell->getSaveFilename("Save Transfer Function", "transfunc.vtf", patterns);
  if(filename.length() == 0) return 1;
  if(vvToolshed::isFile(filename.text()))
  {
    int over = FXMessageBox::question((FXWindow*)this, MBOX_OK_CANCEL, "Warning", "Overwrite existing file?");
    if(over == FX::MBOX_CLICKED_CANCEL) return 1;
  }
  _canvas->_vd->tf[0].saveBinMeshviewer(filename.text());
  return 1;
}

long VVTransferWindow::onCmdLoadTF(FXObject*,FXSelector,void*)
{
  const FXchar patterns[]="Transfer function files (*.vtf)\nAll Files (*.*)";
  FXString filename = _shell->getOpenFilename("Load Transfer Function", patterns);
  if(filename.length() == 0) return 1;
  if(!vvToolshed::isFile(filename.text()))
  {
    FXMessageBox::question((FXWindow*)this, MBOX_OK, "Error", "File does not exist");
    return 1;
  }
  _canvas->_vd->tf[0].load(filename.text());
  if(_instantButton->getCheck()) updateTransFunc();
  drawTF();
  return 1;
}

long VVTransferWindow::onCmdNormalization(FXObject*,FXSelector,void*)
{
  setDirtyHistogram();
  drawTF();
  return 1;
}

void VVTransferWindow::newWidget(WidgetType wt)
{
  vvColor col;
  vvTFWidget* widget = NULL;

  if(!_canvas) return;
  _canvas->_vd->tf[0].putUndoBuffer();
  switch(wt)
  {
    case COLOR:
      widget = new vvTFColor(col, norm2data(0.5f));
      break;
    case PYRAMID:
      widget = new vvTFPyramid(col, false, 1.0f, norm2data(0.5f), normd2datad(0.4f), normd2datad(0.2f));
      break;
    case BELL:
      widget = new vvTFBell(col, false, 1.0f, norm2data(0.5f), normd2datad(0.2f));
      break;
    case CUSTOM:
      widget = new vvTFCustom(norm2data(0.5f), norm2data(0.5f));
      break;
    case SKIP:
      widget = new vvTFSkip(norm2data(0.5f), normd2datad(0.2f));
      break;
    default: return;
  }
  _canvas->_vd->tf[0]._widgets.push_back(widget);
  _currentWidget = widget;
  if(_instantButton->getCheck()) updateTransFunc();
  drawTF();
  updateLabels();
}

void VVTransferWindow::drawHistogram()
{
  computeHistogram();
  switch (_tfBook->getCurrent())
  {
    case 0:
      if (_glCanvas1D->makeCurrent())
      {
        glRasterPos2f(0.0f, 0.0f); 
        glPixelZoom(1.0f, 1.0f);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDrawPixels(_glCanvas1D->getWidth(), _glCanvas1D->getHeight() - COLORBAR_HEIGHT - BINLIMITS_HEIGHT, 
          GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)_histoTexture1D);
        glDisable(GL_BLEND);
        _glCanvas1D->makeNonCurrent();
      }
      break;
    case 1:
      if (_glCanvas2D->makeCurrent())
      {
        glRasterPos2f(0.0f, 0.0f); 
        glPixelZoom(1.0f, 1.0f);
        glDrawPixels(_glCanvas2D->getWidth(), _glCanvas2D->getHeight(), 
          GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)_histoTexture2D);
        _glCanvas2D->makeNonCurrent();
      }
      break;
    default: break;
  }
}

void VVTransferWindow::setDirtyHistogram()
{
  _is1DHistogramDirty = true;
  _is2DHistogramDirty = true;
}

void VVTransferWindow::computeHistogram()
{
  size_t size[2];

  switch (_tfBook->getCurrent())
  {
    case 0:
    {
      if (!_is1DHistogramDirty)
        break;

      std::cerr << "Calculating histogram...";
      vvColor col(0.4f, 0.4f, 0.4f);
      delete[] _histoTexture1D;
      size[0] = _glCanvas1D->getWidth();
      size[1] = _glCanvas1D->getHeight() - COLORBAR_HEIGHT - BINLIMITS_HEIGHT;
      _histoTexture1D = new uchar[size[0] * size[1] * 4];
      _canvas->_vd->makeHistogramTexture((_histAll->getCheck()) ? -1 : 0, 0, 1, size, _histoTexture1D, 
        (_cbNorm->getCheck()) ? vvVolDesc::VV_LOGARITHMIC : vvVolDesc::VV_LINEAR, &col, _dataZoom[0], _dataZoom[1]);
      _is1DHistogramDirty = false;
      std::cerr << "done" << std::endl;
      break;
    }
    case 1:
    {
      if (!_is2DHistogramDirty)
        break;

      std::cerr << "Calculating histogram...";
      vvColor col(1.0f, 1.0f, 1.0f);
      delete[] _histoTexture2D;
      size[0] = _glCanvas2D->getWidth();
      size[1] = _glCanvas2D->getHeight();
      _histoTexture2D = new uchar[size[0] * size[1] * 4];
      _canvas->_vd->makeHistogramTexture((_histAll->getCheck()) ? -1 : 0, 0, 2, size, _histoTexture2D,
        (_cbNorm->getCheck()) ? vvVolDesc::VV_LOGARITHMIC : vvVolDesc::VV_LINEAR, &col, _dataZoom[0], _dataZoom[1]);
      _is2DHistogramDirty = false;
      std::cerr << "done" << std::endl;
      break;
    }
    default: break;
  }
}

void VVTransferWindow::drawTF()
{
  switch (_tfBook->getCurrent())
  {
    case 0:
      draw1DTF();
      break;
    case 1:
      draw2DTF();
      break;
    default: break;
  }
}

void VVTransferWindow::draw1DTF()
{
  if (_glCanvas1D->makeCurrent())
  {
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);   // white background
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    _glCanvas1D->makeNonCurrent();
  }
  drawColorTexture();
  drawPinBackground();
  if (_pinsCheck->getCheck()) drawPinLines();
  drawCustomWidgets();
  if (_binsCheck->getCheck()) drawBinLimits();
  if (_glCanvas1D->makeCurrent())
  {
    if(_glVisual1D->isDoubleBuffer())
    {
      _glCanvas1D->swapBuffers();
    }
    _glCanvas1D->makeNonCurrent();
  }
}

void VVTransferWindow::draw2DTF()
{
  float r,g,b;

  if (_glCanvas2D->makeCurrent())
  {
    _canvas->getBackgroundColor(r, g, b);
    glClearColor(r, g, b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    _glCanvas2D->makeNonCurrent();
  }
  if (_histoCheck->getCheck())
  {
    drawHistogram();
  }
  draw2DTFTexture();
  draw2DTFWidgets();
  if (_glCanvas2D->makeCurrent())
  {
    if(_glVisual2D->isDoubleBuffer())
    {
      _glCanvas2D->swapBuffers();
    }
    _glCanvas2D->makeNonCurrent();
  }
}

void VVTransferWindow::draw2DTFWidgets()
{
  for (std::vector<vvTFWidget*>::const_iterator it = _canvas->_vd->tf[0]._widgets.begin();
       it != _canvas->_vd->tf[0]._widgets.end() ; ++it)
  {
    draw2DWidget(*it);
  }
}

void VVTransferWindow::draw2DWidget(vvTFWidget* w)
{
  float xHalf=0.0f, yHalf=0.0f; // half width and height
  vvTFPyramid* pw;
  vvTFBell* bw;
  bool selected;

  if ((pw = dynamic_cast<vvTFPyramid*>(w)) != NULL)
  {
    xHalf = pw->_bottom[0] / 2.0f;
    yHalf = pw->_bottom[1] / 2.0f;
  }
  else if ((bw = dynamic_cast<vvTFBell*>(w)) != NULL)
  {
    xHalf = bw->_size[0] / 2.0f;
    yHalf = bw->_size[1] / 2.0f;
  }  
  selected = (w == _currentWidget);
  if (_glCanvas2D->makeCurrent())
  {
    glColor3f(1.0f, 1.0f, 1.0f);
    if (selected) glLineWidth(4.0f);
    else glLineWidth(2.0f);

    glBegin(GL_LINE_STRIP);
      glVertex2f(w->_pos[0] - xHalf, w->_pos[1] - yHalf);   // bottom left
      glVertex2f(w->_pos[0] + xHalf, w->_pos[1] - yHalf);   // bottom right
      glVertex2f(w->_pos[0] + xHalf, w->_pos[1] + yHalf);   // top right
      glVertex2f(w->_pos[0] - xHalf, w->_pos[1] + yHalf);   // top left
      glVertex2f(w->_pos[0] - xHalf, w->_pos[1] - yHalf);   // bottom left
    glEnd();

    _glCanvas2D->makeNonCurrent();
  }
}

void VVTransferWindow::drawPinBackground()
{
  if (_histoCheck->getCheck())
  {
    drawHistogram();
  }
  if (_opaCheck->getCheck())
  {
    drawAlphaTexture();
  }
}

void VVTransferWindow::drawCustomWidgets()
{
  vvTFCustom* cuw;
  if(!_canvas) return;

  for (std::vector<vvTFWidget*>::const_iterator it = _canvas->_vd->tf[0]._widgets.begin();
       it != _canvas->_vd->tf[0]._widgets.end(); ++it)
  {
    vvTFWidget* w = *it;
    if (w==_currentWidget)    // don't render control points when widget not current
    {
      if ((cuw=dynamic_cast<vvTFCustom*>(w))!=NULL) // is widget of control point type?
      {
        drawControlPoints(cuw);
      }
    }
  }
}

void VVTransferWindow::drawBinLimits()
{
  float xmin, xmax, ymin, ymax;
   
  if (_glCanvas1D->makeCurrent())
  {
    glLineWidth(1.0f);
    
    // Calculate y coordinates:
    ymin = (float(_glCanvas1D->getHeight() - COLORBAR_HEIGHT - BINLIMITS_HEIGHT)) / _glCanvas1D->getHeight();
    ymax = (float(_glCanvas1D->getHeight() - COLORBAR_HEIGHT)) / _glCanvas1D->getHeight();

    // Draw white background for tick marks:
    glBegin(GL_QUADS);
      glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
      glVertex3f(0.0f, ymin, 0.0f);
      glVertex3f(1.0f, ymin, 0.0f);
      glVertex3f(1.0f, ymax, 0.0f);
      glVertex3f(0.0f, ymax, 0.0f);
    glEnd();

    if (_shell->_floatRangeDialog->_hdrCheck->getCheck())
    {
      for (size_t i=0; i<vvVolDesc::NUM_HDR_BINS; ++i)
      {
        xmin = data2norm(_canvas->_vd->_hdrBinLimits[i]);
        if (i==vvVolDesc::NUM_HDR_BINS-1) xmax = data2norm(_canvas->_vd->range(0)[1]);
        else xmax = data2norm(_canvas->_vd->_hdrBinLimits[i+1]);
      
        glBegin(GL_LINES);
          glColor4f(0.0f, 0.0f, 0.0f, 1.0f);
          glVertex3f(xmin, ymin, 0.0f);
          glVertex3f(xmin, ymax, 0.0f);
          if (i==vvVolDesc::NUM_HDR_BINS-1)   // draw last bin end line?
          {            
            glVertex3f(xmax, ymin, 0.0f);
            glVertex3f(xmax, ymax, 0.0f);
          }
        glEnd();
      }
    }
    else
    {
      for (size_t i=0; i<=256; ++i)
      {
        glBegin(GL_LINES);
          glColor4f(0.0f, 0.0f, 0.0f, 1.0f);
          xmin = data2norm((float(i) / 256.0f) * 
            (_canvas->_vd->range(0)[1] - _canvas->_vd->range(0)[0]) + _canvas->_vd->range(0)[0]);
          glVertex3f(xmin, ymin, 0.0f);
          glVertex3f(xmin, ymax, 0.0f);
        glEnd();
      }
    }
    _glCanvas1D->makeNonCurrent();
  }
}

void VVTransferWindow::drawControlPoints(vvTFCustom* cuw)
{
  float x,y;

  std::list<vvTFPoint*>::iterator iter;
  for(iter=cuw->_points.begin(); iter!=cuw->_points.end(); iter++) 
  {
    x = cuw->_pos[0] + (*iter)->_pos[0];
    y = (*iter)->_opacity * (_glCanvas1D->getHeight() - COLORBAR_HEIGHT - BINLIMITS_HEIGHT) / float(_glCanvas1D->getHeight());  
    if (_glCanvas1D->makeCurrent())
    {
      drawSphere(data2norm(x), y, 0.02f, (*iter)==cuw->_currentPoint);
      _glCanvas1D->makeNonCurrent();
    }
  }
}

void VVTransferWindow::drawSphere(float x, float y, float radius, bool isHighlighted)
{
  (void)isHighlighted;
  GLUquadricObj* sphereObj;

  glEnable(GL_LIGHT0);
  glEnable(GL_LIGHTING);
  float redMaterial[] = { 1, 0, 0, 1 };
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, redMaterial);
  glPushMatrix();
  glTranslatef(x, y, 0.0f);
  sphereObj = gluNewQuadric();
  gluSphere(sphereObj, radius, 8, 8);
  glPopMatrix();
  glDisable(GL_LIGHTING);
}

/** General routine to draw a circle on the current OpenGL canvas.
*/
void VVTransferWindow::drawCircle(float x, float y, float radius, bool isHighlighted)
{
  int i;
  float radians;
  
  if (isHighlighted) glColor3f(1.0f, 0.0f, 0.0f);
  else glColor3f(0.0f, 0.0f, 0.0f);
  glLineWidth(2.0f);
  glBegin(GL_LINE_LOOP);
    for (i=0; i<360; ++i)
    {
      radians = i * TS_PI / 180.0f;
      glVertex2f(x + cos(radians) * radius, y + sin(radians) * radius);
    }
  glEnd();
}

void VVTransferWindow::drawPinLines()
{
  if(!_canvas) return;

  for (std::vector<vvTFWidget*>::const_iterator it = _canvas->_vd->tf[0]._widgets.begin();
       it != _canvas->_vd->tf[0]._widgets.end(); ++it)
  {
    drawPinLine(*it);
  }
}

/** Convert canvas x coordinates to data values.
  @param canvas canvas x coordinate [0..1]
  @return data value
*/
float VVTransferWindow::norm2data(float canvas)
{
  return canvas * (_dataZoom[1] - _dataZoom[0]) + _dataZoom[0];
}

/** Convert data value to x coordinate in TF canvas.
  @param data data value
  @return canvas x coordinate [0..1]
*/
float VVTransferWindow::data2norm(float data)
{
  return (data - _dataZoom[0]) / (_dataZoom[1] - _dataZoom[0]);
}

/** Convert horizontal differences on the canvas to data differences.
*/
float VVTransferWindow::normd2datad(float canvas)
{
  return canvas * (_dataZoom[1] - _dataZoom[0]);
}

/** Convert differences in data to the canvas.
*/
float VVTransferWindow::datad2normd(float data)
{
  return data / (_dataZoom[1] - _dataZoom[0]);
}

/** Converts HDR space values to linear data values.
*/
float VVTransferWindow::hdr2data(float hdr)
{
  size_t bin = _canvas->_vd->findHDRBin(hdr);
  float binStart = _canvas->_vd->_hdrBinLimits[bin];
  float binEnd = (bin < vvVolDesc::NUM_HDR_BINS-1) ? _canvas->_vd->_hdrBinLimits[bin+1] : _canvas->_vd->range(0)[1];
  float binPos = (binEnd - binStart) / 2.0f + binStart;
  return binPos;
}

/** Converts linear data values to HDR space values.
*/
float VVTransferWindow::data2hdr(float data)
{
  size_t bin = _canvas->_vd->findHDRBin(data);
  float binStart = _canvas->_vd->_hdrBinLimits[bin];
  float binEnd = (bin < vvVolDesc::NUM_HDR_BINS-1) ? _canvas->_vd->_hdrBinLimits[bin+1] : _canvas->_vd->range(0)[1];
  float binPos = (binEnd - binStart) / 2.0f + binStart;
  return binPos;
}

/** Converts HDR space value to bin position in linear space.
*/
float VVTransferWindow::hdr2realbin(float hdr)
{    
  size_t bin = _canvas->_vd->findHDRBin(hdr);
  float realbin = float(bin) * (_canvas->_vd->range(0)[1] - _canvas->_vd->range(0)[0]) / float(vvVolDesc::NUM_HDR_BINS-1) + _canvas->_vd->range(0)[0];
  return realbin;
}

/** Converts bin position in linear space to HDR space value.
*/
float VVTransferWindow::realbin2hdr(float realbin)
{    
  size_t bin = (realbin - _canvas->_vd->range(0)[0]) / (_canvas->_vd->range(0)[1] - _canvas->_vd->range(0)[0]) * float(vvVolDesc::NUM_HDR_BINS-1);
  bin = ts_clamp(bin, size_t(0), vvVolDesc::NUM_HDR_BINS-1);
  float binStart = _canvas->_vd->_hdrBinLimits[bin];
  float binEnd = (bin < vvVolDesc::NUM_HDR_BINS-1) ? _canvas->_vd->_hdrBinLimits[bin+1] : _canvas->_vd->range(0)[1];
  float binPos = (binEnd - binStart) / 2.0f + binStart;
  return binPos;
}

void VVTransferWindow::drawPinLine(vvTFWidget* w)
{
  float xPos, yTop, height;
  bool selected;

  if (dynamic_cast<vvTFColor*>(w) != NULL)
  { 
    yTop = 1.0f;
    height = float(COLORBAR_HEIGHT) / float(_glCanvas1D->getHeight());
  }
  else if ((dynamic_cast<vvTFPyramid*>(w) != NULL) ||
           (dynamic_cast<vvTFBell*>(w) != NULL) ||
           (dynamic_cast<vvTFSkip*>(w) != NULL) ||
           (dynamic_cast<vvTFCustom*>(w) != NULL))
  {
    yTop = 1.0f - float(COLORBAR_HEIGHT + BINLIMITS_HEIGHT) / float(_glCanvas1D->getHeight());
    height = float(_glCanvas1D->getHeight() - COLORBAR_HEIGHT - BINLIMITS_HEIGHT) / float(_glCanvas1D->getHeight());
  }
  else return;

  selected = (w == _currentWidget);
  xPos = data2norm(w->_pos[0]);

  if (_glCanvas1D->makeCurrent())
  {
    glColor3f(0.0f, 0.0f, 0.0f);
    if (selected) glLineWidth(4.0f);
    else glLineWidth(2.0f);
    glBegin(GL_LINES);
      glVertex2f(xPos, yTop);
      glVertex2f(xPos, yTop - height);
    glEnd();
    _glCanvas1D->makeNonCurrent();
  }
}

/** Returns the pointer to the closest widget to a specific point in TF space.
  @param x,y,z  query position in TF space [0..1]. -1 if undefined
*/
vvTFWidget* VVTransferWindow::closestWidget(float x, float y, float z)
{
  (void)z;
  vvTFWidget* w = NULL;
  vvTFWidget* temp = NULL;
  float dist, xDist, yDist;     // [TF space]
  float minDist = FLT_MAX;      // [TF space]
  bool isColor;

  if (!_canvas) return NULL;
  for (std::vector<vvTFWidget*>::const_iterator it = _canvas->_vd->tf[0]._widgets.begin();
       it != _canvas->_vd->tf[0]._widgets.end(); ++it)
  {
    temp = *it;
    switch (_tfBook->getCurrent())
    {
      case 0: 
        isColor = (y > 1.0f - float(COLORBAR_HEIGHT + BINLIMITS_HEIGHT) / float(_glCanvas1D->getHeight())) ? true : false;
        if ((isColor && dynamic_cast<vvTFColor*>(temp)) || (!isColor && dynamic_cast<vvTFColor*>(temp)==NULL)) 
        {
          dist = fabs(x - temp->_pos[0]);
          if(dist < minDist && dist <= normd2datad(CLICK_TOLERANCE))
          {
            minDist = dist;
            w = temp;
          }
        }
        break;
      case 1:
        if (!dynamic_cast<vvTFColor*>(temp))
        {
          xDist = x - temp->_pos[0];
          yDist = y - temp->_pos[1];
          dist = float(sqrt(xDist * xDist + yDist * yDist));
          if (dist < minDist && dist <= normd2datad(CLICK_TOLERANCE))
          {
            minDist = dist;
            w = temp;
          }
        }
        break;
      default: assert(0); break;
    }
  }
  return w;
}

void VVTransferWindow::updateLabels()
{
  vvTFColor* cw;
  vvTFPyramid* pw;
  vvTFBell* bw;
  vvTFSkip* sw;
  vvTFCustom* cuw;

  if ((cw=dynamic_cast<vvTFColor*>(_currentWidget))!=NULL)
  {
    _pinSwitcher->setCurrent(1);
  }
  else if ((pw=dynamic_cast<vvTFPyramid*>(_currentWidget))!=NULL)
  {
    _pinSwitcher->setCurrent(2);
    _pTopXSlider->setValue(datad2normd(pw->_top[0]));
    _pTopXLabel->setText(FXStringFormat("%.5g", pw->_top[0]));
    _pBottomXSlider->setValue(datad2normd(pw->_bottom[0]));
    _pBottomXLabel->setText(FXStringFormat("%.5g", pw->_bottom[0]));
    _pMaxSlider->setValue(pw->_opacity);
    _pMaxLabel->setText(FXStringFormat("%.5g", pw->_opacity));
    _pColorButton->setCheck(pw->hasOwnColor());
  }
  else if ((bw=dynamic_cast<vvTFBell*>(_currentWidget))!=NULL)
  {
    _pinSwitcher->setCurrent(3);
    _bWidthSlider->setValue(datad2normd(bw->_size[0]));
    _bWidthLabel->setText(FXStringFormat("%.5g", bw->_size[0]));
    _bMaxSlider->setValue(bw->_opacity);
    _bMaxLabel->setText(FXStringFormat("%.5g", bw->_opacity));
    _bColorButton->setCheck(bw->hasOwnColor());
  }
  else if ((sw=dynamic_cast<vvTFSkip*>(_currentWidget))!=NULL)
  {
    _pinSwitcher->setCurrent(4);
    _sWidthSlider->setValue(datad2normd(sw->_size[0]));
    _sWidthLabel->setText(FXStringFormat("%.5g", sw->_size[0]));
  }
  else if ((cuw=dynamic_cast<vvTFCustom*>(_currentWidget))!=NULL)
  {
    _pinSwitcher->setCurrent(5);
    _cWidthSlider->setValue(datad2normd(cuw->_size[0]));
    _cWidthLabel->setText(FXStringFormat("%.5g", cuw->_size[0]));
  }
  else    // no current widget
  {
    _pinSwitcher->setCurrent(0);
  }
  if (_currentWidget) _pinPosLabel->setText(FXStringFormat("Pin = %.5g", _currentWidget->_pos[0]));
  else _pinPosLabel->setText("");
}

/** Create color bar for regular or HDR mode.
*/
void VVTransferWindow::makeColorBar(int width, uchar* colorBar)
{ 
  const int RGBA = 4;   // bytes per pixel in RGBA mode
  uchar* tmpBar;
  float fval;
  int i;

  if (_shell->_floatRangeDialog->_hdrCheck->getCheck())
  {
    // Shift colors according to HDR bins:
    tmpBar = new uchar[vvVolDesc::NUM_HDR_BINS * RGBA * 3];
    _canvas->_vd->tf[0].makeColorBar(vvVolDesc::NUM_HDR_BINS, tmpBar, _canvas->_vd->range(0)[0], _canvas->_vd->range(0)[1], _invertCheck->getCheck()!=0);
    for (i=0; i<width; ++i)
    {
      fval = norm2data(float(i) / float(width));  // calculates pixel position in linear data space
      size_t bin = _canvas->_vd->findHDRBin(fval);
      bin = ts_clamp(bin, size_t(0), vvVolDesc::NUM_HDR_BINS-1);
      memcpy(colorBar + i*RGBA, tmpBar + bin*RGBA, RGBA);
      memcpy(colorBar + RGBA*(i+width), tmpBar + RGBA * (bin + vvVolDesc::NUM_HDR_BINS), RGBA);
      if (_shell->_floatRangeDialog->_opacityCheck->getCheck())
      {
        memcpy(colorBar + RGBA*(i+2*width), tmpBar + RGBA * (bin + 2*vvVolDesc::NUM_HDR_BINS), RGBA);
      }
    }
    delete[] tmpBar;
    if (!_shell->_floatRangeDialog->_opacityCheck->getCheck())
    {
      tmpBar = new uchar[width * RGBA * 3];
      _canvas->_vd->tf[0].makeColorBar(width, tmpBar, _dataZoom[0], _dataZoom[1], _invertCheck->getCheck()!=0);
      memcpy(colorBar + RGBA * 2 * width, tmpBar + RGBA * 2 * width, RGBA * width); // copy linear opacity to color bar
      memcpy(colorBar + RGBA * width, colorBar, RGBA * width); // copy non-linear color to combined color/opacity bar
      for (i=0; i<width; ++i)
      {
        colorBar[RGBA * (width + i) + 3] = 255 - tmpBar[RGBA * (2 * width + i)];    // use opacity from opacity bar for combined bar
      }
      delete[] tmpBar;
    }
  }
  else  // standard iso-range TF mode
  {
    _canvas->_vd->tf[0].makeColorBar(width, colorBar, _dataZoom[0], _dataZoom[1], _invertCheck->getCheck()!=0);
  }
}

/** Create alpha texture for regular or HDR mode.
  @param alphaTex needs to be pre-allocated for width*height*4 bytes
*/
void VVTransferWindow::makeAlphaTexture(int width, int height, uchar* alphaTex)
{
  const int RGBA = 4;   // bytes per pixel in RGBA mode
  uchar* tmpTex;
  float fval;
  int i,y;

  if (_shell->_floatRangeDialog->_hdrCheck->getCheck() && _shell->_floatRangeDialog->_opacityCheck->getCheck())
  {   // histogram equalization mode?
    tmpTex = new uchar[vvVolDesc::NUM_HDR_BINS * height * RGBA];
    _canvas->_vd->tf[0].makeAlphaTexture(vvVolDesc::NUM_HDR_BINS, height, tmpTex, _canvas->_vd->range(0)[0], _canvas->_vd->range(0)[1]);
    for (i=0; i<width; ++i)
    {
      fval = norm2data(float(i) / float(width));
      size_t bin = _canvas->_vd->findHDRBin(fval);
      bin = ts_clamp(bin, size_t(0), vvVolDesc::NUM_HDR_BINS-1);
      for (y=0; y<height; ++y)
      {
        memcpy(alphaTex + RGBA * (y * width + i), tmpTex + RGBA * (y * vvVolDesc::NUM_HDR_BINS + bin), RGBA);
      }
    }
    delete[] tmpTex;
  }
  else  // standard iso-range TF mode
  {
    _canvas->_vd->tf[0].makeAlphaTexture(width, height, alphaTex, _dataZoom[0], _dataZoom[1]);
  }
}

void VVTransferWindow::drawColorTexture()
{
  uchar* colorBar;
  uchar background[4];
  float r,g,b;
  int width;
  
  width = _glCanvas1D->getWidth();
  colorBar = new uchar[width * 4 * 3];
  makeColorBar(width, colorBar);
  if (_glCanvas1D->makeCurrent())
  {
    // Draw background:
    _canvas->getBackgroundColor(r, g, b);   // use background color from volume rendering canvas
    background[0] = uchar(r * 255.0f);
    background[1] = uchar(g * 255.0f);
    background[2] = uchar(b * 255.0f);
    background[3] = 255;
    glRasterPos2f(0.0f, 1.0f);  // pixmap origin is bottom left corner of output window
    glPixelZoom(float(_glCanvas1D->getWidth()), -30.0f); // full canvas width, 20 pixels high
    glDrawPixels(1, 1, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)background);

    // Draw color bars:
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glRasterPos2f(0.0f, 1.0f);  // pixmap origin is bottom left corner of output window
    glPixelZoom(1.0f, -10.0f); // full canvas width, 10*3 pixels high, upside-down
    glDrawPixels(width, 3, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)colorBar);
    glDisable(GL_BLEND);
    
    _glCanvas1D->makeNonCurrent();
  }
  delete[] colorBar;
}

void VVTransferWindow::drawAlphaTexture()
{
  int width = _glCanvas1D->getWidth();
  int height = _glCanvas1D->getHeight() - COLORBAR_HEIGHT - BINLIMITS_HEIGHT;
  uchar* alphaTexture = new uchar[width * height * 4];
  makeAlphaTexture(width, height, alphaTexture);
  if (_glCanvas1D->makeCurrent())
  {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glRasterPos2f(0.0f, 1.0f - float(COLORBAR_HEIGHT + BINLIMITS_HEIGHT) / float(_glCanvas1D->getHeight())); 
    glPixelZoom(1.0f, -1.0f);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)alphaTexture);
    glDisable(GL_BLEND);
    _glCanvas1D->makeNonCurrent();
  }
  delete[] alphaTexture;
}

void VVTransferWindow::draw2DTFTexture()
{
  static int WIDTH  = 128;
  static int HEIGHT = 128;
  static uchar* tfTexture = new uchar[WIDTH * HEIGHT * 4];

  if (_glCanvas2D->makeCurrent())
  {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    _canvas->_vd->tf[0].make2DTFTexture(WIDTH, HEIGHT, tfTexture, 0.0f, 1.0f, 0.0f, 1.0f);
    glRasterPos2f(0.0f, 0.0f); 
    glPixelZoom(float(_glCanvas2D->getWidth()) / float(WIDTH), float(_glCanvas2D->getHeight()) / float(HEIGHT));
    glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)tfTexture);
    glDisable(GL_BLEND);
    _glCanvas2D->makeNonCurrent();
  }
}

long VVTransferWindow::onChngDisColors(FXObject*,FXSelector,void*)
{
  _disColorLabel->setText(FXStringFormat("%d",_disColorSlider->getValue()));
  _canvas->_vd->tf[0].setDiscreteColors(_disColorSlider->getValue());
  drawTF();
  if(_instantButton->getCheck()) updateTransFunc();
  return 1;
}

long VVTransferWindow::onCmdHistAll(FXObject*,FXSelector,void*)
{
  _histFirst->setCheck(false);
  setDirtyHistogram();
  drawTF();
  return 1;
}

long VVTransferWindow::onCmdHistFirst(FXObject*,FXSelector,void*)
{
  _histAll->setCheck(false);
  setDirtyHistogram();
  drawTF();
  return 1;
}

long VVTransferWindow::onCmdHistogram(FXObject*,FXSelector,void*)
{
  if (_histoCheck->getCheck()) 
  {
    setDirtyHistogram();
    _histFirst->enable();
    _histAll->enable();
  }
  else
  {
    _histFirst->disable();
    _histAll->disable();
  }
  drawTF();
  return 1;
}

long VVTransferWindow::onCmdOpacity(FXObject*,FXSelector,void*)
{
  drawTF();
  return 1;
}

long VVTransferWindow::onCmdBins(FXObject*,FXSelector,void*)
{
  drawTF();
  return 1;
}

long VVTransferWindow::onCmdPins(FXObject*,FXSelector,void*)
{
  drawTF();
  return 1;
}

long VVTransferWindow::onCmdSetMin(FXObject*,FXSelector,void*)
{
  FXInputDialog* dialog;
  dialog = new FXInputDialog((FXWindow*)this, "Set minimum", "");
  dialog->setText(FXStringFormat("%.5g", _dataZoom[0]));
  dialog->execute();
  _dataZoom[0] = FXFloatVal(dialog->getText());
  _zoomMinButton->setText(FXStringFormat("%.5g", _dataZoom[0]));
  delete dialog;
  setDirtyHistogram();
  return 1;
}

long VVTransferWindow::onCmdSetMax(FXObject*,FXSelector,void*)
{
  FXInputDialog* dialog;
  dialog = new FXInputDialog((FXWindow*)this, "Set maximum", "");
  dialog->setText(FXStringFormat("%.5g", _dataZoom[1]));
  dialog->execute();
  _dataZoom[1] = FXFloatVal(dialog->getText());
  _zoomMaxButton->setText(FXStringFormat("%.5g", _dataZoom[1]));
  delete dialog;
  setDirtyHistogram();
  return 1;
}

long VVTransferWindow::onCmdZoomLUT(FXObject*,FXSelector,void*)
{
  zoomLUT();
  return 1;
}

void VVTransferWindow::zoomLUT()
{
  _dataZoom[0] = _canvas->_vd->range(0)[0];
  _dataZoom[1] = _canvas->_vd->range(0)[1];
  setDirtyHistogram();
  updateValues();
}

long VVTransferWindow::onCmdDefault(FXObject*,FXSelector,void*)
{
  _canvas->_vd->tf[0].putUndoBuffer();
  _dataZoom[0] = _canvas->_vd->range(0)[0];
  _dataZoom[1] = _canvas->_vd->range(0)[1];
  _canvas->_vd->tf[0].setDefaultAlpha(0, _dataZoom[0], _dataZoom[1]);
  _canvas->_vd->tf[0].setDefaultColors(0, _dataZoom[0], _dataZoom[1]);
  _currentWidget = NULL;
  _canvas->_renderer->updateVolumeData();
  setDirtyHistogram();
  drawTF();
  updateValues();
  updateLabels();
  updateTransFunc();
  _shell->_floatRangeDialog->updateValues();
  return 1;
}

long VVTransferWindow::onCmdCenter(FXObject*,FXSelector,void*)
{
  float range = _dataZoom[1] - _dataZoom[0];
  _dataZoom[0] = -range/2.0f;
  _dataZoom[1] = range/2.0f;
  setDirtyHistogram();
  updateValues();
  return 1;
}

long VVTransferWindow::onCmdInvertAlpha(FXObject*,FXSelector,void*)
{
  int status = (_invertCheck->getCheck()) ? 1 : 0;
  getApp()->reg().writeIntEntry("Settings", "InvertAlpha", status);
  getApp()->reg().write();
  drawTF();
  return 1;
}

void VVTransferWindow::updateValues()
{
  _currentWidget = NULL;
  _zoomMinButton->setText(FXStringFormat("%.5g", _dataZoom[0]));
  _zoomMaxButton->setText(FXStringFormat("%.5g", _dataZoom[1]));
  _realMinLabel->setText(FXStringFormat("%.5g", _canvas->_vd->range(0)[0]));
  _realMaxLabel->setText(FXStringFormat("%.5g", _canvas->_vd->range(0)[1]));
  updateLabels();
  _disColorSlider->setValue(_canvas->_vd->tf[0].getDiscreteColors());
  _disColorLabel->setText(FXStringFormat("%d", _disColorSlider->getValue()));
  drawTF();
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
