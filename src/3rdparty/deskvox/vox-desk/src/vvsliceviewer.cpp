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
#include <vvdebugmsg.h>
#include <vvtoolshed.h>

// Local:
#include "vvsliceviewer.h"
#include "vvcanvas.h"
#include "vvshell.h"

using namespace vox;

const int VVSliceViewer::SLICE_WIDTH  = 256;
const int VVSliceViewer::SLICE_HEIGHT = 256;

/*******************************************************************************/
FXDEFMAP(VVSliceViewer) VVSliceViewerMap[]=
{
  FXMAPFUNC(SEL_COMMAND, VVSliceViewer::ID_MIRROR_X,  VVSliceViewer::onCmdMirrorX),
  FXMAPFUNC(SEL_COMMAND, VVSliceViewer::ID_MIRROR_Y,  VVSliceViewer::onCmdMirrorY),
  FXMAPFUNC(SEL_CHANGED, VVSliceViewer::ID_SLICE,     VVSliceViewer::onChngSlice),
  FXMAPFUNC(SEL_COMMAND, VVSliceViewer::ID_BEGINNING, VVSliceViewer::onCmdBeginning),
  FXMAPFUNC(SEL_COMMAND, VVSliceViewer::ID_BACK,      VVSliceViewer::onCmdBack),
  FXMAPFUNC(SEL_COMMAND, VVSliceViewer::ID_FORWARD,   VVSliceViewer::onCmdForward),
  FXMAPFUNC(SEL_COMMAND, VVSliceViewer::ID_END,       VVSliceViewer::onCmdEnd),
  FXMAPFUNC(SEL_COMMAND, VVSliceViewer::ID_X_AXIS,    VVSliceViewer::onCmdXAxis),
  FXMAPFUNC(SEL_COMMAND, VVSliceViewer::ID_Y_AXIS,    VVSliceViewer::onCmdYAxis),
  FXMAPFUNC(SEL_COMMAND, VVSliceViewer::ID_Z_AXIS,    VVSliceViewer::onCmdZAxis),
  FXMAPFUNC(SEL_PAINT,   VVSliceViewer::ID_SLICE_CANVAS, VVSliceViewer::onPaint),
};

FXIMPLEMENT(VVSliceViewer,FXDialogBox,VVSliceViewerMap,ARRAYNUMBER(VVSliceViewerMap))

// Construct a dialog box
VVSliceViewer::VVSliceViewer(FXWindow* owner, vvCanvas* c)
  : FXDialogBox(owner, "Slice Viewer", DECOR_TITLE|DECOR_BORDER, 100, 100)
  , _axis(virvo::cartesian_axis<3>::Z)
{
  vvDebugMsg::msg(1, "VVSliceViewer::VVSliceViewer()");

  _canvas = c;
  _shell = (VVShell*)owner;

  FXVerticalFrame* master = new FXVerticalFrame(this, LAYOUT_CENTER_X);

  FXHorizontalFrame* resolutionFrame = new FXHorizontalFrame(master, LAYOUT_CENTER_X);
  new FXLabel(resolutionFrame, "Original slice resolution:", NULL, LABEL_NORMAL);
  _resolutionLabel = new FXLabel(resolutionFrame, "", NULL, LABEL_NORMAL);
  
  FXHorizontalFrame* horizontalFrame = new FXHorizontalFrame(master, LAYOUT_FILL_X);
  FXVerticalFrame* leftFrame  = new FXVerticalFrame(horizontalFrame);
  FXVerticalFrame* rightFrame = new FXVerticalFrame(horizontalFrame);

  FXGroupBox* buttonGroup = new FXGroupBox(leftFrame, "Slicing axis", LAYOUT_SIDE_TOP | FRAME_GROOVE | LAYOUT_FILL_X);
  _xAxisButton = new FXRadioButton(buttonGroup, "X axis", this, ID_X_AXIS);
  _yAxisButton = new FXRadioButton(buttonGroup, "Y axis", this, ID_Y_AXIS);
  _zAxisButton = new FXRadioButton(buttonGroup, "Z axis", this, ID_Z_AXIS);
  _zAxisButton->setCheck(true);

  _mirrorX = new FXCheckButton(leftFrame, "Flip horizontal", this, ID_MIRROR_X, ICON_BEFORE_TEXT);
  _mirrorY = new FXCheckButton(leftFrame, "Flip vertical", this, ID_MIRROR_Y, ICON_BEFORE_TEXT);

  _sliceCanvas = new FXCanvas(rightFrame, this, ID_SLICE_CANVAS, FRAME_SUNKEN | LAYOUT_FIX_HEIGHT | LAYOUT_FIX_WIDTH, 0,0,SLICE_WIDTH,SLICE_HEIGHT);

  FXGroupBox* group = new FXGroupBox(master, "", GROUPBOX_NORMAL | FRAME_GROOVE | LAYOUT_FILL_X);

  FXHorizontalFrame* sliceFrame = new FXHorizontalFrame(group, LAYOUT_CENTER_X);
  new FXLabel(sliceFrame, "Current slice:", NULL, LABEL_NORMAL);
  _sliceLabel = new FXLabel(sliceFrame, "", NULL, LABEL_NORMAL);

  _sliceSlider = new FXSlider(group, this, ID_SLICE, SLIDER_HORIZONTAL | SLIDER_ARROW_DOWN | LAYOUT_FILL_X);
  _sliceSlider->setRange(1, 1);
  _sliceSlider->setValue(1);
  _sliceSlider->setTickDelta(5);

  FXHorizontalFrame* buttonFrame = new FXHorizontalFrame(master, LAYOUT_CENTER_X | PACK_UNIFORM_WIDTH);
  new FXButton(buttonFrame,"|<<",NULL,this,ID_BEGINNING, FRAME_RAISED | FRAME_THICK);
  new FXButton(buttonFrame,"|<", NULL,this,ID_BACK, FRAME_RAISED | FRAME_THICK);
  new FXButton(buttonFrame,">|", NULL,this,ID_FORWARD, FRAME_RAISED | FRAME_THICK);
  new FXButton(buttonFrame,">>|",NULL,this,ID_END, FRAME_RAISED | FRAME_THICK);

  new FXButton(master,"Close",NULL,this,ID_ACCEPT, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X,0,0,0,0,20,20);
}

// Must delete the menus
VVSliceViewer::~VVSliceViewer()
{
  vvDebugMsg::msg(1, "VVSliceViewer::~VVSliceViewer()");
}

long VVSliceViewer::onChngSlice(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVSliceViewer::onChngSlice()");
  selectSlice(_sliceSlider->getValue() - 1);
  return 1;
}

long VVSliceViewer::onCmdBeginning(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVSliceViewer::onCmdBeginning()");
  selectSlice(0);
  return 1;
}

long VVSliceViewer::onCmdMirrorX(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVSliceViewer::onCmdMirrorX()");
  showSlice(_sliceSlider->getValue() - 1);
  return 1;
}

long VVSliceViewer::onCmdMirrorY(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVSliceViewer::onCmdMirrorY()");
  showSlice(_sliceSlider->getValue() - 1);
  return 1;
}

long VVSliceViewer::onCmdBack(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVSliceViewer::onCmdBack()");
  int current = _sliceSlider->getValue() - 1;
  --current;
  current = ts_max(current, 0);
  selectSlice(current);
  return 1;
}

long VVSliceViewer::onCmdForward(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVSliceViewer::onCmdForward()");

  size_t width, height, slices;
  _canvas->_vd->getVolumeSize(_axis, width, height, slices);

  size_t current = _sliceSlider->getValue() - 1;
  ++current;
  current = ts_min(current, slices-1);
  selectSlice(current);
  return 1;
}

long VVSliceViewer::onCmdEnd(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVSliceViewer::onCmdEnd()");

  size_t width, height, slices;
  _canvas->_vd->getVolumeSize(_axis, width, height, slices);

  selectSlice(slices - 1);
  return 1;
}

long VVSliceViewer::onCmdXAxis(FXObject*,FXSelector,void*)
{
  _axis = virvo::cartesian_axis<3>::X;
  _yAxisButton->setCheck(false);
  _zAxisButton->setCheck(false);
  updateValues();
  return 1;
}

long VVSliceViewer::onCmdYAxis(FXObject*,FXSelector,void*)
{
  _axis = virvo::cartesian_axis<3>::Y;
  _xAxisButton->setCheck(false);
  _zAxisButton->setCheck(false);
  updateValues();
  return 1;
}

long VVSliceViewer::onCmdZAxis(FXObject*,FXSelector,void*)
{
  _axis = virvo::cartesian_axis<3>::Z;
  _xAxisButton->setCheck(false);
  _yAxisButton->setCheck(false);
  updateValues();
  return 1;
}

void VVSliceViewer::selectSlice(int slice)
{
  size_t width, height, slices;
  _canvas->_vd->getVolumeSize(_axis, width, height, slices);
  _sliceSlider->setValue(slice + 1);
  _sliceLabel->setText(FXStringFormat("%d of %" VV_PRIdSIZE, slice + 1, slices));
  showSlice(slice);
}

void VVSliceViewer::showSlice(int slice)
{
  static uchar* sliceTexture = NULL;
  static FXColor* sliceBuffer = NULL;
  static FXImage* sliceImage = NULL;
  int color[3], pos;

  size_t width, height, slices;
  _canvas->_vd->getVolumeSize(_axis, width, height, slices);

  FXDCWindow dc(_sliceCanvas);
  delete sliceTexture;
  sliceTexture = new uchar[width * height * 3];
  _canvas->_vd->makeSliceImage(-1, _axis, slice, sliceTexture);
  delete sliceBuffer;
  sliceBuffer = new FXColor[width * height];
  for(size_t y=0; y<height; ++y)
  {
    for(size_t x=0; x<width; ++x)
    {
      pos = y * width + x;
      for (int c=0; c<3; ++c)
      {
        color[c] = (int)(sliceTexture[pos * 3 + c]);
      }
      sliceBuffer[pos] = FXRGB(color[0], color[1], color[2]);
    }
  }
  delete sliceImage;
  sliceImage = new FXImage(_shell->getApp(), sliceBuffer, IMAGE_NEAREST | IMAGE_KEEP, width, height);
  sliceImage->create();
  sliceImage->mirror(_mirrorX->getCheck()!=0, _mirrorY->getCheck()!=0);
  sliceImage->scale(_sliceCanvas->getWidth(), _sliceCanvas->getHeight());
  dc.drawImage(sliceImage, 0, 0);
}

void VVSliceViewer::updateValues()
{
  size_t width, height, slices;
  _canvas->_vd->getVolumeSize(_axis, width, height, slices);
  _sliceSlider->setRange(1, slices);
  _resolutionLabel->setText(FXStringFormat("%" VV_PRIdSIZE " x %" VV_PRIdSIZE, width, height));
  selectSlice(0);
}

/// process paint event from GUI
long VVSliceViewer::onPaint(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVSliceViewer::onPaint()");

  showSlice(_sliceSlider->getValue() - 1);
  return 1;
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
