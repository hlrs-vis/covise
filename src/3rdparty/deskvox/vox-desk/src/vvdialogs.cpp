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

#ifdef _WIN32
#pragma warning(disable: 4244)    // disable warning about conversion from int to short
#pragma warning(disable: 4511)    // disable warning: copy constructor could not be generated
#pragma warning(disable: 4512)    // disable warning: assignment operator could not be generated
#endif

// Virvo:
#include <virvo/math/math.h>
#include <vvopengl.h>
#include <vvvirvo.h>
#include <vvdebugmsg.h>
#include <vvtoolshed.h>

// Local:
#include "vvdialogs.h"
#include "vvcanvas.h"
#include "vvshell.h"
#include "vvmovie.h"
#include "vvtranswindow.h"

using namespace virvo;
using namespace vox;
using namespace std;

#if FOX_MAJOR >= 1 || FOX_MINOR >= 6
using FXSystem::getCurrentDirectory;
#else
using FXFile::getCurrentDirectory;
#endif

FXDEFMAP(VVVolumeDialog) VVVolumeDialogMap[]=
{
  FXMAPFUNC(SEL_COMMAND, VVVolumeDialog::ID_MAKE_ICON, VVVolumeDialog::onMakeIcon),
  FXMAPFUNC(SEL_COMMAND, VVVolumeDialog::ID_CHANNELS,  VVVolumeDialog::onChannels),
  FXMAPFUNC(SEL_PAINT, VVVolumeDialog::ID_ICON_CANVAS, VVVolumeDialog::onPaint),
};

FXIMPLEMENT(VVVolumeDialog,FXDialogBox,VVVolumeDialogMap,ARRAYNUMBER(VVVolumeDialogMap))

// Construct a dialog box
VVVolumeDialog::VVVolumeDialog(FXWindow* owner, vvCanvas* c)
  : FXDialogBox(owner,"Volume Information",DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE,100,100)
{  
  _canvas = c;

  FXVerticalFrame* master = new FXVerticalFrame(this, LAYOUT_FILL_X | LAYOUT_FILL_Y);

  FXHorizontalFrame* nameFrame = new FXHorizontalFrame(master, LAYOUT_FILL_X);
  new FXLabel(nameFrame,"File name:",NULL,LABEL_NORMAL);
  _fileTField = new FXTextField(nameFrame, 50, NULL, 0, TEXTFIELD_NORMAL | TEXTFIELD_READONLY);
  _fileTField->setText("default.xvf");

  FXHorizontalFrame* horizFrame = new FXHorizontalFrame(master, LAYOUT_FILL_X);
  FXVerticalFrame* leftFrame = new FXVerticalFrame(horizFrame, LAYOUT_FILL_Y);
  FXVerticalFrame* rightFrame = new FXVerticalFrame(horizFrame, LAYOUT_FILL_Y);

  FXMatrix* mat = new FXMatrix(leftFrame,2, MATRIX_BY_COLUMNS);

  new FXLabel(mat,"Slice width:",NULL,LABEL_NORMAL);
  _widthLabel = new FXLabel(mat, "0",NULL,LABEL_NORMAL);

  new FXLabel(mat,"Slice height:",NULL,LABEL_NORMAL);
  _heightLabel = new FXLabel(mat, "0",NULL,LABEL_NORMAL);

  new FXLabel(mat,"Slices per volume:",NULL,LABEL_NORMAL);
  _slicesLabel = new FXLabel(mat, "0",NULL,LABEL_NORMAL);

  new FXLabel(mat,"Time steps:",NULL,LABEL_NORMAL);
  _framesLabel = new FXLabel(mat, "0",NULL,LABEL_NORMAL);

  new FXLabel(mat,"Bytes per channel:",NULL,LABEL_NORMAL);
  _bpcLabel = new FXLabel(mat,"1",NULL,LABEL_NORMAL);

  new FXButton(mat,"Channels:",NULL,this,ID_CHANNELS, FRAME_RAISED | FRAME_THICK | LAYOUT_LEFT);
  _chanLabel = new FXLabel(mat,"1",NULL,LABEL_NORMAL);

  new FXLabel(mat,"Voxels per time step:",NULL,LABEL_NORMAL);
  _voxelsLabel = new FXLabel(mat, "0",NULL,LABEL_NORMAL);

  new FXLabel(mat,"Bytes per time step:",NULL,LABEL_NORMAL);
  _bytesLabel = new FXLabel(mat, "0",NULL,LABEL_NORMAL);

  new FXLabel(mat,"Sample distance X:",NULL,LABEL_NORMAL);
  _dxLabel = new FXLabel(mat,"1.0",NULL,LABEL_NORMAL);

  new FXLabel(mat,"Sample distance Y:",NULL,LABEL_NORMAL);
  _dyLabel = new FXLabel(mat,"1.0",NULL,LABEL_NORMAL);

  new FXLabel(mat,"Sample distance Z:",NULL,LABEL_NORMAL);
  _dzLabel = new FXLabel(mat, "1.0",NULL,LABEL_NORMAL);

  new FXLabel(mat,"Physical minimum:",NULL,LABEL_NORMAL);
  _realMinLabel = new FXLabel(mat, "0.0",NULL,LABEL_NORMAL);

  new FXLabel(mat,"Physical maximum:",NULL,LABEL_NORMAL);
  _realMaxLabel = new FXLabel(mat, "1.0",NULL,LABEL_NORMAL);

  new FXLabel(mat,"Minimum data value channel 1:",NULL,LABEL_NORMAL);
  _minLabel = new FXLabel(mat, "0.0",NULL,LABEL_NORMAL);

  new FXLabel(mat,"Maximum data value channel 1:",NULL,LABEL_NORMAL);
  _maxLabel = new FXLabel(mat, "1.0",NULL,LABEL_NORMAL);

  new FXLabel(rightFrame,"Icon:", NULL, LABEL_NORMAL);
  _iconCanvas = new FXCanvas(rightFrame, this, ID_ICON_CANVAS, FRAME_SUNKEN | LAYOUT_FIX_HEIGHT | LAYOUT_FIX_WIDTH | LAYOUT_CENTER_X, 0, 0, 64, 64);

  new FXButton(rightFrame,"Update Icon",NULL,this,ID_MAKE_ICON, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X);

  new FXButton(master,"Close",NULL,this,ID_ACCEPT, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X, 0,0,0,0, 20,20,3,3);
}

void VVVolumeDialog::drawIcon()
{
  static uchar*   iconTexture = NULL;
  static FXColor* iconBuffer = NULL;
  static FXImage* iconImage = NULL;
  int x, y, c, color[3], pos;
  int iconBytes;
  int iconSize;

  FXDCWindow dc(_iconCanvas);
  iconSize = _canvas->_vd->iconSize;
  if (iconSize==0)
  {
    dc.setForeground(_iconCanvas->getBackColor());
    dc.fillRectangle(0, 0, _iconCanvas->getWidth(), _iconCanvas->getHeight());
  }
  else
  {
    iconBytes = iconSize * iconSize * vvVolDesc::ICON_BPP;
    delete iconTexture;
    iconTexture = new uchar[iconBytes];
    memcpy(iconTexture, _canvas->_vd->iconData, iconBytes);
    delete iconBuffer;
    iconBuffer = new FXColor[iconSize * iconSize];
    for(y=0; y<iconSize; ++y)
    {
      for(x=0; x<iconSize; ++x)
      {
        pos = y * iconSize + x;
        for (c=0; c<3; ++c)
        {
          color[c] = (int)(iconTexture[pos * vvVolDesc::ICON_BPP + c]);
        }
        iconBuffer[pos] = FXRGB(color[0], color[1], color[2]);
      }
    }
    delete iconImage;
    iconImage = new FXImage(getApp(), iconBuffer, IMAGE_NEAREST | IMAGE_KEEP, iconSize, iconSize);
    iconImage->create();
    iconImage->scale(_iconCanvas->getWidth(), _iconCanvas->getHeight());
    dc.drawImage(iconImage, 0, 0);
  }
}

long VVVolumeDialog::onMakeIcon(FXObject*, FXSelector, void*)
{
  _canvas->_vd->makeIcon(vvVolDesc::DEFAULT_ICON_SIZE);
  drawIcon();
  return 1;
}

long VVVolumeDialog::onChannels(FXObject*, FXSelector, void*)
{
  FXString info;
  for (int c=0; c<_canvas->_vd->getChan(); ++c)
  {
    info += "Channel " + FXStringFormat("%d", c) + ": ";
    std::string name = _canvas->_vd->getChannelName(c);
    if (!name.empty()) info += name.c_str();
    else info += "UNNAMED";
    info += "\n";
  }

  FXMessageBox::information((FXWindow*)this, MBOX_OK, "Channel Names", "%s", info.text());

  return 1;
}

/// process paint event from GUI
long VVVolumeDialog::onPaint(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVVolumeDialog::onPaint()");
  drawIcon();
  return 1;
}

void VVVolumeDialog::updateValues()
{
  float fMin, fMax;

  _fileTField->setText(_canvas->_vd->getFilename());
  _widthLabel->setText(FXStringFormat("%" VV_PRIdSIZE,  _canvas->_vd->vox[0]));
  _heightLabel->setText(FXStringFormat("%" VV_PRIdSIZE, _canvas->_vd->vox[1]));
  _slicesLabel->setText(FXStringFormat("%" VV_PRIdSIZE, _canvas->_vd->vox[2]));
  _framesLabel->setText(FXStringFormat("%" VV_PRIdSIZE, _canvas->_vd->frames));
  _bpcLabel->setText(FXStringFormat("%" VV_PRIdSIZE,    _canvas->_vd->bpc));
  _chanLabel->setText(FXStringFormat("%d",   _canvas->_vd->getChan()));
  _voxelsLabel->setText(FXStringFormat("%" VV_PRIdSIZE, _canvas->_vd->getFrameVoxels()));
  _bytesLabel->setText(FXStringFormat("%" VV_PRIdSIZE,  _canvas->_vd->getFrameBytes()));
  _dxLabel->setText(FXStringFormat("%.9g",   _canvas->_vd->getDist()[0]));
  _dyLabel->setText(FXStringFormat("%.9g",   _canvas->_vd->getDist()[1]));
  _dzLabel->setText(FXStringFormat("%.9g",   _canvas->_vd->getDist()[2]));
  _realMinLabel->setText(FXStringFormat("%.9g",  _canvas->_vd->mapping(0)[0]));
  _realMaxLabel->setText(FXStringFormat("%.9g",  _canvas->_vd->mapping(0)[1]));
  _canvas->_vd->findMinMax(0, fMin, fMax);
  _minLabel->setText(FXStringFormat("%.9g",  _canvas->_vd->range(0)[0]));
  _maxLabel->setText(FXStringFormat("%.9g",  _canvas->_vd->range(0)[1]));
}

/*******************************************************************************/
/*
// Construct a dialog box
VVSystemInfo::VVSystemInfo(FXWindow* owner, vvCanvas* c)
  : FXDialogBox(owner,"System Information",DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE,100,100)
{  
  char* status[3] = {"supported", "not found"};

  _canvas = c;
  FXVerticalFrame* master = new FXVerticalFrame(this, LAYOUT_FILL_X | LAYOUT_FILL_Y);

  FXHorizontalFrame* horizFrame = new FXHorizontalFrame(master, LAYOUT_FILL_X);
  FXVerticalFrame* leftFrame = new FXVerticalFrame(horizFrame, LAYOUT_FILL_Y);
  FXVerticalFrame* rightFrame = new FXVerticalFrame(horizFrame, LAYOUT_FILL_Y);

  FXMatrix* mat = new FXMatrix(leftFrame,2, MATRIX_BY_COLUMNS);

  if (_glcanvas->makeCurrent())
  {
    new FXLabel(mat,"GL_EXT_texture3D",NULL,LABEL_NORMAL);
    new FXLabel(mat,((vvGLTools::isGLextensionSupported("GL_EXT_texture3D")) ? status[0] : status[1]),NULL,LABEL_NORMAL);
  }

  new FXButton(master,"Close",NULL,this,ID_ACCEPT, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X);
}
*/
/*******************************************************************************/

FXDEFMAP(VVCameraSetDialog) VVCameraSetDialogMap[]=
{
  FXMAPFUNC(SEL_CHANGED, VVCameraSetDialog::ID_FOV,         VVCameraSetDialog::onFOVChange),
  FXMAPFUNC(SEL_CHANGED, VVCameraSetDialog::ID_WIDTH,       VVCameraSetDialog::onWidthChange),
  FXMAPFUNC(SEL_CHANGED, VVCameraSetDialog::ID_CLIP,        VVCameraSetDialog::onClipChange),
  FXMAPFUNC(SEL_COMMAND, VVCameraSetDialog::ID_PERSPECTIVE, VVCameraSetDialog::onPerspectiveChange),
  FXMAPFUNC(SEL_COMMAND, VVCameraSetDialog::ID_DEF,         VVCameraSetDialog::onDefaultSelect),
  FXMAPFUNC(SEL_COMMAND, VVCameraSetDialog::ID_LEFT,        VVCameraSetDialog::onLeft),
  FXMAPFUNC(SEL_COMMAND, VVCameraSetDialog::ID_RIGHT,       VVCameraSetDialog::onRight),
  FXMAPFUNC(SEL_COMMAND, VVCameraSetDialog::ID_TOP,         VVCameraSetDialog::onTop),
  FXMAPFUNC(SEL_COMMAND, VVCameraSetDialog::ID_BOTTOM,      VVCameraSetDialog::onBottom),
  FXMAPFUNC(SEL_COMMAND, VVCameraSetDialog::ID_FRONT,       VVCameraSetDialog::onFront),
  FXMAPFUNC(SEL_COMMAND, VVCameraSetDialog::ID_BACK,        VVCameraSetDialog::onBack),
};

FXIMPLEMENT(VVCameraSetDialog,FXDialogBox,VVCameraSetDialogMap,ARRAYNUMBER(VVCameraSetDialogMap))

// Construct a dialog box
VVCameraSetDialog::VVCameraSetDialog(FXWindow* owner, vvCanvas* c) : 
  FXDialogBox(owner,"Camera Settings",DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE,100,100)
{
  _shell = (VVShell*)owner;
  _canvas = c;
  FXVerticalFrame * master=new FXVerticalFrame(this,LAYOUT_FILL_X|LAYOUT_FILL_Y,0,0,0,0, 0,0,0,0);
  _perspectiveCheck = new FXCheckButton(master,"Perspective Mode",this,ID_PERSPECTIVE,ICON_BEFORE_TEXT|LAYOUT_SIDE_TOP|LAYOUT_LEFT);

  FXGroupBox* projectionGroup = new FXGroupBox(master, "Projection parameters", FRAME_GROOVE | LAYOUT_FILL_X);

  FXMatrix* fovLabelMat=new FXMatrix(projectionGroup,2,MATRIX_BY_COLUMNS|LAYOUT_FILL_X);
  new FXLabel(fovLabelMat,"Field of view [degrees]:");
  _fovLabel = new FXLabel(fovLabelMat, "");
  _fovSlider=new FXRealSlider(projectionGroup,this,ID_FOV,LAYOUT_TOP|LAYOUT_FIX_WIDTH|LAYOUT_FIX_HEIGHT|SLIDER_HORIZONTAL,0,0,300,30);
  _fovSlider->setRange(0,180);

  FXMatrix* widthLabelMat=new FXMatrix(projectionGroup,2,MATRIX_BY_COLUMNS|LAYOUT_FILL_X);
  new FXLabel(widthLabelMat,"Viewport width [mm]:");
  _widthLabel = new FXLabel(widthLabelMat, "");
  _widthSlider=new FXRealSlider(projectionGroup,this,ID_WIDTH,LAYOUT_TOP|LAYOUT_FIX_WIDTH|LAYOUT_FIX_HEIGHT|SLIDER_HORIZONTAL,0,0,300,30);
  _widthSlider->setRange(0,1000);

  FXMatrix* nearLabelMat=new FXMatrix(projectionGroup,2,MATRIX_BY_COLUMNS|LAYOUT_FILL_X);
  new FXLabel(nearLabelMat,"Near view frustum clipping plane:");
  _nearLabel = new FXLabel(nearLabelMat, "");
  _nearSlider = new FXRealSlider(projectionGroup,this,ID_CLIP,LAYOUT_TOP|LAYOUT_FIX_WIDTH|LAYOUT_FIX_HEIGHT|SLIDER_HORIZONTAL,0,0,300,30);
  _nearSlider->setRange(1, 1000);

  FXMatrix* farLabelMat=new FXMatrix(projectionGroup,2,MATRIX_BY_COLUMNS|LAYOUT_FILL_X);
  new FXLabel(farLabelMat,"Far view frustum clipping plane:");
  _farLabel = new FXLabel(farLabelMat, "");
  _farSlider=new FXRealSlider(projectionGroup,this,ID_CLIP,LAYOUT_TOP|LAYOUT_FIX_WIDTH|LAYOUT_FIX_HEIGHT|SLIDER_HORIZONTAL,0,0,300,30);
  _farSlider->setRange(1, 1000);

  FXGroupBox* presetGroup = new FXGroupBox(master, "Preset views", FRAME_GROOVE | LAYOUT_FILL_X);
  FXHorizontalFrame* presetFrame1 = new FXHorizontalFrame(presetGroup, LAYOUT_FILL_X | PACK_UNIFORM_WIDTH);
  new FXButton(presetFrame1,"Left",  NULL,this,ID_LEFT,  FRAME_RAISED|FRAME_THICK|LAYOUT_FILL_X);
  new FXButton(presetFrame1,"Front", NULL,this,ID_FRONT, FRAME_RAISED|FRAME_THICK|LAYOUT_FILL_X);
  new FXButton(presetFrame1,"Top",   NULL,this,ID_TOP,   FRAME_RAISED|FRAME_THICK|LAYOUT_FILL_X);
  FXHorizontalFrame* presetFrame2 = new FXHorizontalFrame(presetGroup, LAYOUT_FILL_X | PACK_UNIFORM_WIDTH);
  new FXButton(presetFrame2,"Right", NULL,this,ID_RIGHT, FRAME_RAISED|FRAME_THICK|LAYOUT_FILL_X);
  new FXButton(presetFrame2,"Back",  NULL,this,ID_BACK,  FRAME_RAISED|FRAME_THICK|LAYOUT_FILL_X);
  new FXButton(presetFrame2,"Bottom",NULL,this,ID_BOTTOM,FRAME_RAISED|FRAME_THICK|LAYOUT_FILL_X);

  FXHorizontalFrame* buttonFrame=new FXHorizontalFrame(master,LAYOUT_FILL_Y | LAYOUT_CENTER_X | PACK_UNIFORM_WIDTH);
  new FXButton(buttonFrame,"Reset",NULL,this,ID_DEF,FRAME_RAISED|FRAME_THICK|LAYOUT_CENTER_X|LAYOUT_CENTER_Y,0,0,0,0, 20,20,3,3);
  new FXButton(buttonFrame,"Close",NULL,this,ID_ACCEPT,FRAME_RAISED|FRAME_THICK|LAYOUT_CENTER_X|LAYOUT_CENTER_Y,0,0,0,0, 20,20,3,3);
}

// Must delete the menus
VVCameraSetDialog::~VVCameraSetDialog()
{
  delete _perspectiveCheck;
  delete _fovSlider;
  delete _widthSlider;
  delete _widthLabel;
  delete _nearSlider;
  delete _farSlider;
  delete _fovLabel;
  delete _nearLabel;
  delete _farLabel;
}

long VVCameraSetDialog::onPerspectiveChange(FXObject*,FXSelector,void* ptr)
{
  if (ptr)    // change to perspective mode?
  {
    _canvas->setProjectionMode(true, _fovSlider->getValue() * VV_PI / 180.0f, 
      _nearSlider->getValue(), _farSlider->getValue());
  }
  else    // change to orthogonal mode
  {
    _canvas->setProjectionMode(false, _widthSlider->getValue(), 
      _nearSlider->getValue(), _farSlider->getValue());
  }

  updateValues();
  return 1;
}

long VVCameraSetDialog::onFOVChange(FXObject*,FXSelector,void*)
{
  if (_perspectiveCheck->getCheck())    // fov only makes sense for perspective mode
  {
    _canvas->setProjectionMode(true, _fovSlider->getValue() * VV_PI / 180.0f, _nearSlider->getValue(), _farSlider->getValue());
    updateValues();
  }
  return 1;
}

long VVCameraSetDialog::onWidthChange(FXObject*,FXSelector,void*)
{
  if (!_perspectiveCheck->getCheck())    // width only makes sense for orthogonal mode
  {
    _canvas->setProjectionMode(false, _widthSlider->getValue(), _nearSlider->getValue(), _farSlider->getValue());
    updateValues();
  }
  return 1;
}

long VVCameraSetDialog::onClipChange(FXObject*,FXSelector,void*)
{
  _canvas->setProjectionMode((_perspectiveCheck->getCheck() != 0), 
    (_perspectiveCheck->getCheck()) ? (_fovSlider->getValue() * VV_PI / 180.0f) : _widthSlider->getValue(), 
    _nearSlider->getValue(), _farSlider->getValue());
  updateValues();
  return 1;
}

long VVCameraSetDialog::onLeft(FXObject*,FXSelector,void*)
{
  _canvas->_ov.setDefaultView(vvObjView::LEFT);
  updateValues();
  return 1;
}

long VVCameraSetDialog::onRight(FXObject*,FXSelector,void*)
{
  _canvas->_ov.setDefaultView(vvObjView::RIGHT);
  updateValues();
  return 1;
}

long VVCameraSetDialog::onTop(FXObject*,FXSelector,void*)
{
  _canvas->_ov.setDefaultView(vvObjView::TOP);
  updateValues();
  return 1;
}

long VVCameraSetDialog::onBottom(FXObject*,FXSelector,void*)
{
  _canvas->_ov.setDefaultView(vvObjView::BOTTOM);
  updateValues();
  return 1;
}

long VVCameraSetDialog::onFront(FXObject*,FXSelector,void*)
{
  _canvas->_ov.setDefaultView(vvObjView::FRONT);
  updateValues();
  return 1;
}

long VVCameraSetDialog::onBack(FXObject*,FXSelector,void*)
{
  _canvas->_ov.setDefaultView(vvObjView::BACK);
  updateValues();
  return 1;
}

void VVCameraSetDialog::updateValues()
{
  _perspectiveCheck->setCheck(_canvas->getPerspectiveMode());
  if (_perspectiveCheck->getCheck())    // change to perspective mode?
  {
    _fovSlider->enable();
    _widthSlider->disable();
  }
  else    // change to orthogonal mode
  {
    _fovSlider->disable();
    _widthSlider->enable();
  }

  _fovLabel->setText(FXStringFormat("%.1f", _canvas->_ov.getFOV() * 180.0f / VV_PI));
  _widthLabel->setText(FXStringFormat("%.1f", _canvas->_ov.getViewportWidth()));
  _nearLabel->setText(FXStringFormat("%.1f", _canvas->_ov.getNearPlane()));
  _farLabel->setText(FXStringFormat("%.1f", _canvas->_ov.getFarPlane()));

  _fovSlider->setValue(_canvas->_ov.getFOV() * 180.0f / VV_PI);
  _widthSlider->setValue(_canvas->_ov.getViewportWidth());
  _nearSlider->setValue(_canvas->_ov.getNearPlane());
  _farSlider->setValue(_canvas->_ov.getFarPlane());
}

long VVCameraSetDialog::onDefaultSelect(FXObject*,FXSelector,void*)
{
  reset();
  return 1;
}

void VVCameraSetDialog::reset()
{
  _canvas->setPerspectiveMode(_canvas->getPerspectiveMode());
  _canvas->_ov.resetObject();
  _canvas->_ov.resetCamera();
  updateValues();
}

/*******************************************************************************/

/*******************************************************************************/

FXDEFMAP(VVClippingDialog) VVClippingDialogMap[]=
{
  FXMAPFUNC(SEL_CHANGED,    VVClippingDialog::ID_X,         VVClippingDialog::onSliderChange),
  FXMAPFUNC(SEL_COMMAND,    VVClippingDialog::ID_X,         VVClippingDialog::onSliderChange),
  FXMAPFUNC(SEL_CHANGED,    VVClippingDialog::ID_Y,         VVClippingDialog::onSliderChange),
  FXMAPFUNC(SEL_COMMAND,    VVClippingDialog::ID_Y,         VVClippingDialog::onSliderChange),
  FXMAPFUNC(SEL_CHANGED,    VVClippingDialog::ID_Z,         VVClippingDialog::onSliderChange),
  FXMAPFUNC(SEL_COMMAND,    VVClippingDialog::ID_Z,         VVClippingDialog::onSliderChange),
  FXMAPFUNC(SEL_CHANGED,    VVClippingDialog::ID_ORIGIN,    VVClippingDialog::onSliderChange),
  FXMAPFUNC(SEL_COMMAND,    VVClippingDialog::ID_ORIGIN,    VVClippingDialog::onSliderChange),
  FXMAPFUNC(SEL_COMMAND,    VVClippingDialog::ID_ENABLE,    VVClippingDialog::onEnableChange),
  FXMAPFUNC(SEL_COMMAND,    VVClippingDialog::ID_SINGLE,    VVClippingDialog::onSingleChange),
  FXMAPFUNC(SEL_COMMAND,    VVClippingDialog::ID_OPAQUE,    VVClippingDialog::onOpaqueChange),
  FXMAPFUNC(SEL_COMMAND,    VVClippingDialog::ID_PERIMETER, VVClippingDialog::onPerimeterChange),
};

FXIMPLEMENT(VVClippingDialog,FXDialogBox,VVClippingDialogMap,ARRAYNUMBER(VVClippingDialogMap))

// Construct a dialog box
VVClippingDialog::VVClippingDialog(FXWindow* owner, vvCanvas* c) :
  FXDialogBox(owner,"Clip Plane Settings",DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE,100,100)
{
  _shell = (VVShell*)owner;
  _canvas = c;
  FXVerticalFrame* master=new FXVerticalFrame(this,LAYOUT_FILL_X|LAYOUT_FILL_Y,0,0,0,0, 0,0,0,0);

  _enableCheck = new FXCheckButton(master,"Enable Clipping Plane",this,ID_ENABLE,ICON_BEFORE_TEXT|LAYOUT_SIDE_TOP|LAYOUT_LEFT);

  FXGroupBox* clipGroup = new FXGroupBox(master,"",LAYOUT_SIDE_TOP|FRAME_GROOVE|LAYOUT_FILL_X, 0,0,0,0);

  FXMatrix* sliderMat=new FXMatrix(clipGroup,2,MATRIX_BY_COLUMNS|LAYOUT_FILL_X);
  new FXLabel(sliderMat,"Normal X Coordinate:");

  _xSlider=new FXRealSlider(sliderMat,this,ID_X,LAYOUT_TOP|LAYOUT_FIX_WIDTH|LAYOUT_FIX_HEIGHT|SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|SLIDER_TICKS_BOTTOM,0,0,300,30);
  _xSlider->setRange(-1.0, 1.0);
  _xSlider->setTickDelta(0.1);

  new FXLabel(sliderMat,"Normal Y Coordinate:");

  _ySlider=new FXRealSlider(sliderMat,this,ID_Y,LAYOUT_TOP|LAYOUT_FIX_WIDTH|LAYOUT_FIX_HEIGHT|SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|SLIDER_TICKS_BOTTOM,0,0,300,30);
  _ySlider->setRange(-1.0, 1.0);
  _ySlider->setTickDelta(0.1);

  new FXLabel(sliderMat,"Normal Z Coordinate:");

  _zSlider=new FXRealSlider(sliderMat,this,ID_Z,LAYOUT_TOP|LAYOUT_FIX_WIDTH|LAYOUT_FIX_HEIGHT|SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|SLIDER_TICKS_BOTTOM,0,0,300,30);
  _zSlider->setRange(-1.0, 1.0);
  _zSlider->setTickDelta(0.1);

  new FXLabel(sliderMat,"Origin:");

  _originSlider=new FXRealSlider(sliderMat,this,ID_ORIGIN,LAYOUT_TOP|LAYOUT_FIX_WIDTH|LAYOUT_FIX_HEIGHT|SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|SLIDER_TICKS_BOTTOM,0,0,300,30);
  _originSlider->setRange(-200.0, 200.0);
  _originSlider->setValue(0.0);
  _originSlider->setTickDelta(10.0);

  _singleCheck = new FXCheckButton(clipGroup,"Single Slice",this,ID_SINGLE,ICON_BEFORE_TEXT|LAYOUT_SIDE_TOP|LAYOUT_LEFT);
  _opaqueCheck = new FXCheckButton(clipGroup,"Opaque",this,ID_OPAQUE,ICON_BEFORE_TEXT|LAYOUT_SIDE_TOP|LAYOUT_LEFT);
  _perimeterCheck = new FXCheckButton(clipGroup,"Perimeter",this,ID_PERIMETER,ICON_BEFORE_TEXT|LAYOUT_SIDE_TOP|LAYOUT_LEFT);

  new FXButton(master,"OK",NULL,this,ID_ACCEPT,FRAME_RAISED|FRAME_THICK|LAYOUT_CENTER_X|LAYOUT_CENTER_Y,0,0,0,0, 20,20,3,3);
}

// Must delete the menus
VVClippingDialog::~VVClippingDialog()
{
  delete _enableCheck;
  delete _singleCheck;
  delete _opaqueCheck;
  delete _xSlider;
  delete _ySlider;
  delete _zSlider;
  delete _originSlider;
}

long VVClippingDialog::onEnableChange(FXObject*,FXSelector,void* ptr)
{
  _canvas->_renderer->setParameter(vvRenderState::VV_CLIP_MODE, static_cast< unsigned >(ptr != NULL));
  if(_shell->_glcanvas->makeCurrent())
  {
    _canvas->_renderer->updateTransferFunction();
    _shell->_glcanvas->makeNonCurrent();
  }
  updateValues();
  return 1;
}

long VVClippingDialog::onSingleChange(FXObject*,FXSelector,void* ptr)
{
  _canvas->_renderer->setParameter(vvRenderState::VV_CLIP_SINGLE_SLICE, (ptr != 0));
  return 1;
}

long VVClippingDialog::onPerimeterChange(FXObject*,FXSelector,void* ptr)
{
  _canvas->_renderer->setParameter(vvRenderState::VV_CLIP_PERIMETER, (ptr != 0));
  return 1;
}

long VVClippingDialog::onOpaqueChange(FXObject*,FXSelector,void* ptr)
{
  _canvas->_renderer->setParameter(vvRenderState::VV_CLIP_OPAQUE, (ptr != 0));
  if(_shell->_glcanvas->makeCurrent())
  {
    _canvas->_renderer->updateTransferFunction();
    _shell->_glcanvas->makeNonCurrent();
  }
  return 1;
}

long VVClippingDialog::onSliderChange(FXObject*,FXSelector,void*)
{
  updateClipParameters();
  return 1;
}

void VVClippingDialog::updateClipParameters()
{
  float x = (float)_xSlider->getValue();
  float y = (float)_ySlider->getValue();
  float z = (float)_zSlider->getValue();
  float d = (float)_originSlider->getValue();

  vec3 normal = normalize( vec3(x, y, z) );
  vec3 point = normal * d;
  _canvas->_renderer->setParameter(vvRenderState::VV_CLIP_PLANE_POINT, point);
  _canvas->_renderer->setParameter(vvRenderState::VV_CLIP_PLANE_NORMAL, normal);
}

void VVClippingDialog::updateValues()
{
  vec3 normal = _canvas->_renderer->getParameter(vvRenderState::VV_CLIP_PLANE_NORMAL);
  normal = normalize(normal);

  _enableCheck->setCheck(_canvas->_renderer->getParameter(vvRenderState::VV_CLIP_MODE).asUint());
  _singleCheck->setCheck(_canvas->_renderer->getParameter(vvRenderState::VV_CLIP_SINGLE_SLICE).asBool());
  _opaqueCheck->setCheck(_canvas->_renderer->getParameter(vvRenderState::VV_CLIP_OPAQUE).asBool());
  _perimeterCheck->setCheck(_canvas->_renderer->getParameter(vvRenderState::VV_CLIP_PERIMETER).asBool());

  _xSlider->setValue(normal[0]);
  _ySlider->setValue(normal[1]);
  _zSlider->setValue(normal[2]);
  if (_enableCheck->getCheck())
  {
    _xSlider->enable();
    _ySlider->enable();
    _zSlider->enable();
    _originSlider->enable();
    _singleCheck->enable();
    _opaqueCheck->enable();
    _perimeterCheck->enable();
  }
  else
  {
    _xSlider->disable();
    _ySlider->disable();
    _zSlider->disable();
    _originSlider->disable();
    _singleCheck->disable();
    _opaqueCheck->disable();
    _perimeterCheck->disable();
  }
}

/*******************************************************************************/

FXDEFMAP(VVROIDialog) VVROIDialogMap[]=
{
  FXMAPFUNC(SEL_CHANGED,    VVROIDialog::ID_X,         VVROIDialog::onSliderChange),
  FXMAPFUNC(SEL_COMMAND,    VVROIDialog::ID_X,         VVROIDialog::onSliderChange),
  FXMAPFUNC(SEL_CHANGED,    VVROIDialog::ID_Y,         VVROIDialog::onSliderChange),
  FXMAPFUNC(SEL_COMMAND,    VVROIDialog::ID_Y,         VVROIDialog::onSliderChange),
  FXMAPFUNC(SEL_CHANGED,    VVROIDialog::ID_Z,         VVROIDialog::onSliderChange),
  FXMAPFUNC(SEL_COMMAND,    VVROIDialog::ID_Z,         VVROIDialog::onSliderChange),
  FXMAPFUNC(SEL_CHANGED,    VVROIDialog::ID_SIZE_X,      VVROIDialog::onSliderChange),
  FXMAPFUNC(SEL_COMMAND,    VVROIDialog::ID_SIZE_X,      VVROIDialog::onSliderChange),
  FXMAPFUNC(SEL_CHANGED,    VVROIDialog::ID_SIZE_Y,      VVROIDialog::onSliderChange),
  FXMAPFUNC(SEL_COMMAND,    VVROIDialog::ID_SIZE_Y,      VVROIDialog::onSliderChange),
  FXMAPFUNC(SEL_CHANGED,    VVROIDialog::ID_SIZE_Z,      VVROIDialog::onSliderChange),
  FXMAPFUNC(SEL_COMMAND,    VVROIDialog::ID_SIZE_Z,      VVROIDialog::onSliderChange),
  FXMAPFUNC(SEL_COMMAND,    VVROIDialog::ID_ENABLE,    VVROIDialog::onEnableChange),
};

FXIMPLEMENT(VVROIDialog,FXDialogBox,VVROIDialogMap,ARRAYNUMBER(VVROIDialogMap))

// Construct a dialog box
VVROIDialog::VVROIDialog(FXWindow* owner, vvCanvas* c) :
  FXDialogBox(owner,"Region of Interest Settings",DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE,100,100)
{
  _shell = (VVShell*)owner;
  _canvas = c;
  FXVerticalFrame* master=new FXVerticalFrame(this,LAYOUT_FILL_X|LAYOUT_FILL_Y,0,0,0,0, 0,0,0,0);

  _enableCheck = new FXCheckButton(master,"Enable Region of Interest",this,ID_ENABLE,ICON_BEFORE_TEXT|LAYOUT_SIDE_TOP|LAYOUT_LEFT);

  FXGroupBox* clipGroup = new FXGroupBox(master,"",LAYOUT_SIDE_TOP|FRAME_GROOVE|LAYOUT_FILL_X, 0,0,0,0);

  FXMatrix* sliderMat=new FXMatrix(clipGroup,2,MATRIX_BY_COLUMNS|LAYOUT_FILL_X);
  new FXLabel(sliderMat,"Box x coordinate:");

  _xSlider=new FXRealSlider(sliderMat,this,ID_X,LAYOUT_TOP|LAYOUT_FIX_WIDTH|LAYOUT_FIX_HEIGHT|SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|SLIDER_TICKS_BOTTOM,0,0,300,30);
  _xSlider->setRange(-1.0, 1.0);
  _xSlider->setValue(0.0);
  _xSlider->setTickDelta(0.1);

  new FXLabel(sliderMat,"Box y coordinate:");

  _ySlider=new FXRealSlider(sliderMat,this,ID_Y,LAYOUT_TOP|LAYOUT_FIX_WIDTH|LAYOUT_FIX_HEIGHT|SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|SLIDER_TICKS_BOTTOM,0,0,300,30);
  _ySlider->setRange(-1.0, 1.0);
  _ySlider->setValue(0.0);
  _ySlider->setTickDelta(0.1);

  new FXLabel(sliderMat,"Box z coordinate:");

  _zSlider=new FXRealSlider(sliderMat,this,ID_Z,LAYOUT_TOP|LAYOUT_FIX_WIDTH|LAYOUT_FIX_HEIGHT|SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|SLIDER_TICKS_BOTTOM,0,0,300,30);
  _zSlider->setRange(-1.0, 1.0);
  _zSlider->setValue(0.0);
  _zSlider->setTickDelta(0.1);

  new FXLabel(sliderMat,"Box size x:");

  _sizeSliderX = new FXRealSlider(sliderMat,this,ID_SIZE_X,LAYOUT_TOP|LAYOUT_FIX_WIDTH|LAYOUT_FIX_HEIGHT|SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|SLIDER_TICKS_BOTTOM,0,0,300,30);
  _sizeSliderX->setRange(0.01, 1.0);
  _sizeSliderX->setValue(0.3);
  _sizeSliderX->setTickDelta(0.1);

  new FXLabel(sliderMat,"Box size y:");

  _sizeSliderY = new FXRealSlider(sliderMat,this,ID_SIZE_Y,LAYOUT_TOP|LAYOUT_FIX_WIDTH|LAYOUT_FIX_HEIGHT|SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|SLIDER_TICKS_BOTTOM,0,0,300,30);
  _sizeSliderY->setRange(0.01, 1.0);
  _sizeSliderY->setValue(0.3);
  _sizeSliderY->setTickDelta(0.1);

  new FXLabel(sliderMat,"Box size z:");

  _sizeSliderZ = new FXRealSlider(sliderMat,this,ID_SIZE_Z,LAYOUT_TOP|LAYOUT_FIX_WIDTH|LAYOUT_FIX_HEIGHT|SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|SLIDER_TICKS_BOTTOM,0,0,300,30);
  _sizeSliderZ->setRange(0.01, 1.0);
  _sizeSliderZ->setValue(0.3);
  _sizeSliderZ->setTickDelta(0.1);

  new FXButton(master,"OK",NULL,this,ID_ACCEPT,FRAME_RAISED|FRAME_THICK|LAYOUT_CENTER_X|LAYOUT_CENTER_Y,0,0,0,0, 20,20,3,3);
}

// Must delete the menus
VVROIDialog::~VVROIDialog()
{
  delete _enableCheck;
  delete _xSlider;
  delete _ySlider;
  delete _zSlider;
  delete _sizeSliderX;
  delete _sizeSliderY;
  delete _sizeSliderZ;
}

long VVROIDialog::onEnableChange(FXObject*,FXSelector,void* ptr)
{
  vec3 size;

  if (ptr != 0)
    {
      size = vec3(_sizeSliderX->getValue(), _sizeSliderY->getValue(), _sizeSliderZ->getValue());
      _canvas->_renderer->setProbeSize(size);
      _canvas->_renderer->setROIEnable(true);
    }
  else
    {
      _canvas->_renderer->setROIEnable(false);
    }

  if(_shell->_glcanvas->makeCurrent())
  {
//    _canvas->_renderer->updateTransferFunction();
    _shell->_glcanvas->makeNonCurrent();
  }
  updateValues();
  return 1;
}

long VVROIDialog::onSliderChange(FXObject*,FXSelector,void*)
{
  updateROIParameters();
  return 1;
}

void VVROIDialog::updateROIParameters()
{
  vec3 pos;
  vec3 size;

  pos[0] = _xSlider->getValue();
  pos[1] = _ySlider->getValue();
  pos[2] = _zSlider->getValue();
  _canvas->_renderer->setProbePosition(pos);

  size = vec3(_sizeSliderX->getValue(), _sizeSliderY->getValue(), _sizeSliderZ->getValue());
  _canvas->_renderer->setProbeSize(size);
}

void VVROIDialog::updateValues()
{
  vec3 volSize2;

  _enableCheck->setCheck(_canvas->_renderer->isROIEnabled());
  vec3 pos = _canvas->_renderer->getProbePosition();
  _xSlider->setValue(pos[0]);
  _ySlider->setValue(pos[1]);
  _zSlider->setValue(pos[2]);

  volSize2 = _canvas->_vd->getSize() * 0.5f;
  _xSlider->setRange(-volSize2[0], volSize2[0]);
  _ySlider->setRange(-volSize2[1], volSize2[1]);
  _zSlider->setRange(-volSize2[2], volSize2[2]);
  _xSlider->setTickDelta(volSize2[0] / 10.0f);
  _ySlider->setTickDelta(volSize2[1] / 10.0f);
  _zSlider->setTickDelta(volSize2[2] / 10.0f);

  vec3 roiSize = _canvas->_renderer->getProbeSize();
  _sizeSliderX->setValue(roiSize[0]);
  _sizeSliderY->setValue(roiSize[1]);
  _sizeSliderZ->setValue(roiSize[2]);  

  if (_enableCheck->getCheck())
  {
    _xSlider->enable();
    _ySlider->enable();
    _zSlider->enable();
    _sizeSliderX->enable();
    _sizeSliderY->enable();
    _sizeSliderZ->enable();
  }
  else
  {
    _xSlider->disable();
    _ySlider->disable();
    _zSlider->disable();
    _sizeSliderX->disable();
    _sizeSliderY->disable();
    _sizeSliderZ->disable();
  }
}

/*******************************************************************************/

/*******************************************************************************/

FXDEFMAP(VVDimensionDialog) VVDimensionDialogMap[]=
{
  FXMAPFUNC(SEL_COMMAND,    VVDimensionDialog::ID_RESET, VVDimensionDialog::onResetSelect),
  FXMAPFUNC(SEL_COMMAND,    VVDimensionDialog::ID_APPLY, VVDimensionDialog::onApplySelect),
};

FXIMPLEMENT(VVDimensionDialog,FXDialogBox,VVDimensionDialogMap,ARRAYNUMBER(VVDimensionDialogMap))

// Construct a dialog box
VVDimensionDialog::VVDimensionDialog(FXWindow* owner, vvCanvas* c) : 
  FXDialogBox(owner,"Sample Distances",DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE,100,100)
{
  _shell = (VVShell*)owner;
  _canvas = c;
  FXVerticalFrame* master=new FXVerticalFrame(this,LAYOUT_FILL_X|LAYOUT_FILL_Y,0,0,0,0, 0,0,0,0);

  FXMatrix* spinnerMat = new FXMatrix(master,2,MATRIX_BY_COLUMNS);

  new FXLabel(spinnerMat, "Sample distance x [mm]:");
  _xSpinner = new FXRealSpinner(spinnerMat,20,this,ID_X,FRAME_SUNKEN|FRAME_THICK|LAYOUT_SIDE_TOP);
  _xSpinner->setRange(0, DBL_MAX);
  _xSpinner->setValue(1.0);
  _xSpinner->setIncrement(0.1);

  new FXLabel(spinnerMat, "Sample distance y [mm]:");
  _ySpinner = new FXRealSpinner(spinnerMat,20,this,ID_Y,FRAME_SUNKEN|FRAME_THICK|LAYOUT_SIDE_TOP);
  _ySpinner->setRange(0, DBL_MAX);
  _ySpinner->setValue(1.0);
  _ySpinner->setIncrement(0.1);

  new FXLabel(spinnerMat, "Sample distance z [mm]:");
  _zSpinner = new FXRealSpinner(spinnerMat,20,this,ID_Z,FRAME_SUNKEN|FRAME_THICK|LAYOUT_SIDE_TOP);
  _zSpinner->setRange(0, DBL_MAX);
  _zSpinner->setValue(1.0);
  _zSpinner->setIncrement(0.1);

  FXMatrix* buttonMat=new FXMatrix(master,3,MATRIX_BY_COLUMNS | LAYOUT_CENTER_X | PACK_UNIFORM_WIDTH);
  new FXButton(buttonMat,"Reset", NULL,this,ID_RESET,FRAME_RAISED|FRAME_THICK|LAYOUT_CENTER_X|LAYOUT_CENTER_Y,0,0,0,0, 20,20,3,3);
  new FXButton(buttonMat,"Apply", NULL,this,ID_APPLY,FRAME_RAISED|FRAME_THICK|LAYOUT_CENTER_X|LAYOUT_CENTER_Y,0,0,0,0, 20,20,3,3);
  new FXButton(buttonMat,"Close", NULL,this,ID_ACCEPT,FRAME_RAISED|FRAME_THICK|LAYOUT_CENTER_X|LAYOUT_CENTER_Y,0,0,0,0, 20,20,3,3);

  defaultDist[0] = defaultDist[1] = defaultDist[2] = 1.0f;
}

// Must delete the menus
VVDimensionDialog::~VVDimensionDialog()
{
  delete _xSpinner;
  delete _ySpinner;
  delete _zSpinner;
}

long VVDimensionDialog::onApplySelect(FXObject*,FXSelector,void*)
{
  vvVector3 dist;
  dist.set((float)_xSpinner->getValue(), (float)_ySpinner->getValue(), (float)_zSpinner->getValue());
  _canvas->_vd->setDist(dist);

  _shell->_volumeDialog->updateValues();
  _shell->drawScene();
  return 1;
}

long VVDimensionDialog::onResetSelect(FXObject*,FXSelector,void*)
{
  vvVector3 dist;
  dist.set(defaultDist[0], defaultDist[1], defaultDist[2]);
  _canvas->_vd->setDist(dist);
  updateValues();
  _shell->_volumeDialog->updateValues();
  _shell->drawScene();
  return 1;
}

void VVDimensionDialog::scaleZ(float scale)
{
  vvVector3 size;
  _canvas->_vd->getDist()[2] *= scale;
  updateValues();
  _shell->_volumeDialog->updateValues();
  _shell->drawScene();
}

/// This should be called when file is loaded.
void VVDimensionDialog::initDefaultDistances()
{
  int i;
  for (i=0; i<3; ++i)
  {
    defaultDist[i] = _canvas->_vd->getDist()[i];
  }
}

void VVDimensionDialog::updateValues()
{
  _xSpinner->setValue(_canvas->_vd->getDist()[0]);
  _ySpinner->setValue(_canvas->_vd->getDist()[1]);
  _zSpinner->setValue(_canvas->_vd->getDist()[2]);
}

/*******************************************************************************/

/*******************************************************************************/

FXDEFMAP(VVDrawDialog) VVDrawDialogMap[]=
{
  FXMAPFUNC(SEL_COMMAND, VVDrawDialog::ID_DRAW, VVDrawDialog::onCmdDraw),
};

FXIMPLEMENT(VVDrawDialog,FXDialogBox,VVDrawDialogMap,ARRAYNUMBER(VVDrawDialogMap))

// Construct a dialog box
VVDrawDialog::VVDrawDialog(FXWindow* owner, vvCanvas* c) : 
  FXDialogBox(owner,"Draw Line",DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE,100,100)
{
  _shell = (VVShell*)owner;
  _canvas = c;
  FXHorizontalFrame* master = new FXHorizontalFrame(this, LAYOUT_FILL_X);
  FXMatrix* fieldMat = new FXMatrix(master,2,MATRIX_BY_COLUMNS);

  new FXLabel(fieldMat, "Start point x [voxels]:");
  _startXTField = new FXTextField(fieldMat, 10, NULL,0,TEXTFIELD_INTEGER);
  _startXTField->setText(FXStringFormat("%d",0));
  new FXLabel(fieldMat, "Start point y [voxels]:");
  _startYTField = new FXTextField(fieldMat, 10, NULL,0,TEXTFIELD_INTEGER);
  _startYTField->setText(FXStringFormat("%d",0));
  new FXLabel(fieldMat, "Start point z [voxels]:");
  _startZTField = new FXTextField(fieldMat, 10, NULL,0,TEXTFIELD_INTEGER);
  _startZTField->setText(FXStringFormat("%d",0));

  new FXLabel(fieldMat, "End point x [voxels]:");
  _endXTField = new FXTextField(fieldMat, 10, NULL,0,TEXTFIELD_INTEGER);
  _endXTField->setText(FXStringFormat("%d",31));
  new FXLabel(fieldMat, "End point y [voxels]:");
  _endYTField = new FXTextField(fieldMat, 10, NULL,0,TEXTFIELD_INTEGER);
  _endYTField->setText(FXStringFormat("%d",31));
  new FXLabel(fieldMat, "End point z [voxels]:");
  _endZTField = new FXTextField(fieldMat, 10, NULL,0,TEXTFIELD_INTEGER);
  _endZTField->setText(FXStringFormat("%d",31));

  new FXLabel(fieldMat, "Color [0..255]:");
  _colorTField = new FXTextField(fieldMat, 10, NULL,0,TEXTFIELD_INTEGER);
  _colorTField->setText(FXStringFormat("%d",255));

  FXVerticalFrame* buttons = new FXVerticalFrame(master,LAYOUT_FILL_X | PACK_UNIFORM_WIDTH);
  _lineButton = new FXButton(buttons,"Draw Line",NULL,this,ID_DRAW,FRAME_RAISED|FRAME_THICK|LAYOUT_CENTER_X|LAYOUT_CENTER_Y);
  _boxButton  = new FXButton(buttons,"Draw Box",NULL,this,ID_DRAW,FRAME_RAISED|FRAME_THICK|LAYOUT_CENTER_X|LAYOUT_CENTER_Y);
  new FXButton(buttons,"Close",NULL,this,ID_CANCEL,FRAME_RAISED|FRAME_THICK|LAYOUT_CENTER_X|LAYOUT_CENTER_Y,0,0,0,0, 20,20,3,3);
}

// Must delete the menus
VVDrawDialog::~VVDrawDialog()
{
}

long VVDrawDialog::onCmdDraw(FXObject* sender,FXSelector,void*)
{
  int x1 = FXIntVal(_startXTField->getText());
  int y1 = FXIntVal(_startYTField->getText());
  int z1 = FXIntVal(_startZTField->getText());

  int x2 = FXIntVal(_endXTField->getText());
  int y2 = FXIntVal(_endYTField->getText());
  int z2 = FXIntVal(_endZTField->getText());

  uchar* col = new uchar[_canvas->_vd->bpc];
  memset(col, FXIntVal(_colorTField->getText()), _canvas->_vd->bpc);
  
  if      (sender==_lineButton) _canvas->_vd->drawLine(x1, y1, z1, x2, y2, z2, col);
  else if (sender==_boxButton)  _canvas->_vd->drawBox(x1, y1, z1, x2, y2, z2, 0, col);
  delete[] col;

  _shell->updateRendererVolume();
  _shell->drawScene();

  return 1;
}

void VVDrawDialog::updateValues()
{
}

/*******************************************************************************/

/*******************************************************************************/

FXDEFMAP(VVMergeDialog) VVMergeDialogMap[]=
{
  FXMAPFUNC(SEL_COMMAND, VVMergeDialog::ID_BROWSE,     VVMergeDialog::onCmdBrowse),
  FXMAPFUNC(SEL_COMMAND, VVMergeDialog::ID_HELP,       VVMergeDialog::onCmdHelp),
  FXMAPFUNC(SEL_COMMAND, VVMergeDialog::ID_SLICES2VOL, VVMergeDialog::onCmdSlices2Vol),
  FXMAPFUNC(SEL_COMMAND, VVMergeDialog::ID_VOL2ANIM,   VVMergeDialog::onCmdVol2Anim),
  FXMAPFUNC(SEL_COMMAND, VVMergeDialog::ID_CHAN2VOL,   VVMergeDialog::onCmdChan2Vol),
  FXMAPFUNC(SEL_COMMAND, VVMergeDialog::ID_LIMIT,      VVMergeDialog::onCBLimit),
  FXMAPFUNC(SEL_COMMAND, VVMergeDialog::ID_NUMBERED,   VVMergeDialog::onCBNumbered),
};

FXIMPLEMENT(VVMergeDialog,FXDialogBox,VVMergeDialogMap,ARRAYNUMBER(VVMergeDialogMap))

// Construct a dialog box
VVMergeDialog::VVMergeDialog(FXWindow* owner, vvCanvas* c):FXDialogBox(owner,"Merge Files",
  DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE,100,100)
{
  _shell = (VVShell*)owner;
  _canvas = c;
  FXVerticalFrame* master = new FXVerticalFrame(this,LAYOUT_FILL_X|LAYOUT_FILL_Y);

  FXGroupBox* buttonGroup = new FXGroupBox(master, "Merge type", LAYOUT_SIDE_TOP | FRAME_GROOVE | LAYOUT_FILL_X);
  _slices2volButton = new FXRadioButton(buttonGroup, "Slices to volume",     this, ID_SLICES2VOL);
  _vol2animButton   = new FXRadioButton(buttonGroup, "Volumes to animation", this, ID_VOL2ANIM);
  _chan2volButton   = new FXRadioButton(buttonGroup, "Channels to volume",   this, ID_CHAN2VOL);
  _slices2volButton->setCheck(true);

  FXMatrix* matrix1 = new FXMatrix(master,3,MATRIX_BY_COLUMNS);

  new FXLabel(matrix1, "Name of first file:");
  _fileTField = new FXTextField(matrix1, 40, NULL,0,TEXTFIELD_NORMAL);
  new FXButton(matrix1, "Browse...", NULL,this,ID_BROWSE, FRAME_RAISED | FRAME_THICK);

  FXMatrix* matrix2 = new FXMatrix(master,3,MATRIX_BY_COLUMNS);

  _limitFilesCB = new FXCheckButton(matrix2,"Limit number of files.",this,ID_LIMIT,ICON_BEFORE_TEXT|LAYOUT_LEFT);
  _limitFilesCB->setCheck(false);
  new FXLabel(matrix2, "Number of files:");
  _numberTField = new FXTextField(matrix2, 10, NULL,0,TEXTFIELD_INTEGER|FRAME_SUNKEN|FRAME_THICK);
  _numberTField->setText("1");
  _numberTField->disable();

  _numFilesCB = new FXCheckButton(matrix2,"Files are numbered.",this,ID_NUMBERED,ICON_BEFORE_TEXT|LAYOUT_LEFT);
  _numFilesCB->setCheck(true);
  new FXLabel(matrix2, "File name increment:");
  _incrementTField = new FXTextField(matrix2, 10, NULL,0,TEXTFIELD_INTEGER|FRAME_SUNKEN|FRAME_THICK);
  _incrementTField->setText("1");
  _incrementTField->enable();

  FXHorizontalFrame* buttonFrame = new FXHorizontalFrame(master, LAYOUT_FILL_X | LAYOUT_CENTER_X | PACK_UNIFORM_WIDTH);
  new FXButton(buttonFrame,"Help...",NULL, this, ID_HELP,  FRAME_RAISED | FRAME_THICK | LAYOUT_LEFT);
  new FXButton(buttonFrame,"Merge",  NULL, this, ID_ACCEPT,FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X, 0,0,0,0,20,20);
  new FXButton(buttonFrame,"Cancel", NULL, this, ID_CANCEL,FRAME_RAISED | FRAME_THICK | LAYOUT_RIGHT);
}

// Must delete the menus
VVMergeDialog::~VVMergeDialog()
{
  delete _fileTField;
  delete _numberTField;
}

long VVMergeDialog::onCmdBrowse(FXObject*, FXSelector, void*)
{
  const FXchar patterns[] = "All VOX Files (*.xvf,*.avf,*.rvf,*.tif,*.rgb,*.ppm,*.pgm,*.dcm,*.ximg,*.hdr,*.volb)\nVolume Files (*.rvf,*.xvf,*.avf,*.tif,*.tiff,*.hdr,*.volb)\nImage Files (*.tif,*.rgb,*.ppm,*.pgm,*.dcm,*.ximg)\nAll Files (*.*)";
  FXString filename = _shell->getOpenFilename("Merge Files", patterns);
  if(filename.length() == 0) return 1;
  _fileTField->setText(filename);
  return 1;
}

long VVMergeDialog::onCmdHelp(FXObject*, FXSelector, void*)
{
  const char* info = "If numbered, files must be numbered consecutively, for instance:\n" \
    "image000.tif, image001.tif, image002.tif, ...\n" \
    "Altenatively, they can be numbered with a constant increment greater than 1.\n\n" \
    "The naming scheme '*_zXXX_chXX.*' is recognized for image\n" \
    "files at a given depth (_z) and for a given channel (_ch).\n" \
    "Example: 'image_z010_ch02.tif'";
  FXMessageBox::information((FXWindow*)this, MBOX_OK, "File Format", "%s", info);
  return 1;
}

long VVMergeDialog::onCmdSlices2Vol(FXObject*,FXSelector,void*)
{
  _vol2animButton->setCheck(false);
  _chan2volButton->setCheck(false);
  return 1;
}

long VVMergeDialog::onCmdVol2Anim(FXObject*,FXSelector,void*)
{
  _slices2volButton->setCheck(false);
  _chan2volButton->setCheck(false);
  return 1;
}

long VVMergeDialog::onCmdChan2Vol(FXObject*,FXSelector,void*)
{
  _slices2volButton->setCheck(false);
  _vol2animButton->setCheck(false);
  return 1;
}

long VVMergeDialog::onCBNumbered(FXObject*,FXSelector,void*)
{
  if (_numFilesCB->getCheck()) _incrementTField->enable();
  else _incrementTField->disable();
  return 1;
}

long VVMergeDialog::onCBLimit(FXObject*,FXSelector,void*)
{
  if (_limitFilesCB->getCheck()) _numberTField->enable();
  else _numberTField->disable();
  return 1;
}

void VVMergeDialog::updateValues()
{
  if (_fileTField->getText()=="") 
  {
    FXString path = getApp()->reg().readStringEntry("Settings", "CurrentDirectory", getCurrentDirectory().text());
    _fileTField->setText(path);
  }
}

/*******************************************************************************/

/*******************************************************************************/

FXDEFMAP(VVServerDialog) VVServerDialogMap[]=
{
  FXMAPFUNC(SEL_COMMAND, VVServerDialog::ID_REQUEST, VVServerDialog::onCmdRequest),
};

FXIMPLEMENT(VVServerDialog,FXDialogBox,VVServerDialogMap,ARRAYNUMBER(VVServerDialogMap))

// Construct a dialog box
VVServerDialog::VVServerDialog(FXWindow* owner, vvCanvas* c):FXDialogBox(owner,"Server Request",
  DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE,100,100)
{
  const int TEXT_WIDTH = 10;

  _client = new vvClient();
  _shell = (VVShell*)owner;
  _canvas = c;
  FXHorizontalFrame* master = new FXHorizontalFrame(this,LAYOUT_FILL_X|LAYOUT_FILL_Y);

  FXMatrix* fieldMat = new FXMatrix(master, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);

  new FXLabel(fieldMat, "Data set:");
  _datasetCombo = new FXComboBox(fieldMat,1,this,ID_DATASET, COMBOBOX_INSERT_LAST | COMBOBOX_STATIC | FRAME_SUNKEN | FRAME_THICK | LAYOUT_FILL_X);

  new FXLabel(fieldMat, "Level of Detail:");
  _lodCombo = new FXComboBox(fieldMat,1,this,ID_LOD, COMBOBOX_INSERT_LAST | COMBOBOX_STATIC | FRAME_SUNKEN | FRAME_THICK | LAYOUT_FILL_X);

  new FXLabel(fieldMat, "x0:");
  _x0TField = new FXTextField(fieldMat, TEXT_WIDTH, NULL,0,TEXTFIELD_NORMAL);

  new FXLabel(fieldMat, "y0:");
  _y0TField = new FXTextField(fieldMat, TEXT_WIDTH, NULL,0,TEXTFIELD_NORMAL);

  new FXLabel(fieldMat, "x1:");
  _x1TField = new FXTextField(fieldMat, TEXT_WIDTH, NULL,0,TEXTFIELD_NORMAL);

  new FXLabel(fieldMat, "y1:");
  _y1TField = new FXTextField(fieldMat, TEXT_WIDTH, NULL,0,TEXTFIELD_NORMAL);

  new FXLabel(fieldMat, "Start slice:");
  _startSliceTField = new FXTextField(fieldMat, TEXT_WIDTH, NULL,0,TEXTFIELD_NORMAL);

  new FXLabel(fieldMat, "End slice:");
  _endSliceTField = new FXTextField(fieldMat, TEXT_WIDTH, NULL,0,TEXTFIELD_NORMAL);

  FXVerticalFrame* buttonFrame = new FXVerticalFrame(master, LAYOUT_FILL_X | LAYOUT_CENTER_X | PACK_UNIFORM_WIDTH);
  new FXButton(buttonFrame,"Send request",  NULL, this, ID_REQUEST, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X, 0,0,0,0,20,20);
  new FXButton(buttonFrame,"Close", NULL, this, ID_CANCEL, FRAME_RAISED | FRAME_THICK | LAYOUT_RIGHT);
}

// Must delete the menus
VVServerDialog::~VVServerDialog()
{
  delete _datasetCombo;
  delete _lodCombo;
  delete _x0TField;
  delete _y0TField;
  delete _x1TField;
  delete _y1TField;
  delete _startSliceTField;
  delete _endSliceTField;
  delete _client;
}

void VVServerDialog::getDatasetList()
{
  _datasetCombo->clearItems();
  _datasetCombo->appendItem("Dataset #1");
  _datasetCombo->appendItem("Dataset #2");

  // FIXME: add code to request data sets from web server
}

void VVServerDialog::updateValues()
{
  getDatasetList();
  _datasetCombo->setNumVisible(ts_min(_datasetCombo->getNumItems(), 10));
  _datasetCombo->setCurrentItem(0);
  
  _lodCombo->clearItems();
  _lodCombo->appendItem("64");
  _lodCombo->appendItem("128");
  _lodCombo->appendItem("256");
  _lodCombo->appendItem("512");
  _lodCombo->appendItem("1024");
  _lodCombo->setCurrentItem(2);
  _lodCombo->setNumVisible(_lodCombo->getNumItems());
  
  _x0TField->setText(FXStringFormat("%d", 0));
  _y0TField->setText(FXStringFormat("%d", 0));
  _x1TField->setText(FXStringFormat("%" VV_PRIdSIZE, _canvas->_vd->vox[0] - 1));
  _y1TField->setText(FXStringFormat("%" VV_PRIdSIZE, _canvas->_vd->vox[1] - 1));
  _startSliceTField->setText(FXStringFormat("%d", 0));
  _endSliceTField->setText(FXStringFormat("%" VV_PRIdSIZE, _canvas->_vd->vox[2] - 1));
}

long VVServerDialog::onCmdRequest(FXObject*, FXSelector, void*)
{
  uchar* data;

  int id = _datasetCombo->getCurrentItem();
  int lod = _lodCombo->getNumItems() - 1 - _lodCombo->getCurrentItem();
  int x0 = FXIntVal(_x0TField->getText());
  int y0 = FXIntVal(_y0TField->getText());
  int x1 = FXIntVal(_x1TField->getText());
  int y1 = FXIntVal(_y1TField->getText());
  int startSlice = FXIntVal(_startSliceTField->getText());
  int endSlice = FXIntVal(_endSliceTField->getText());

  data = _client->getRegion(lod, x0, y0, x1, y1, startSlice, endSlice, id, 
    "http://sql-ct.tacc.utexas.edu/region", "http://sqlct.tacc.utexas.edu");

  if (data==NULL)
  {
    cerr << "Error: data transfer from server unsuccessful" << endl;
  }

  return 1;
}

/*******************************************************************************/

/*******************************************************************************/

FXDEFMAP(VVScreenshotDialog) VVScreenshotDialogMap[]=
{

  FXMAPFUNC(SEL_COMMAND,    VVScreenshotDialog::ID_PICTURE, VVScreenshotDialog::onCmdPicture),
  FXMAPFUNC(SEL_COMMAND,    VVScreenshotDialog::ID_DIM,     VVScreenshotDialog::onDimSelect),
  FXMAPFUNC(SEL_COMMAND,    VVScreenshotDialog::ID_BROWSE,  VVScreenshotDialog::onBrowseSelect),
};

FXIMPLEMENT(VVScreenshotDialog,FXDialogBox,VVScreenshotDialogMap,ARRAYNUMBER(VVScreenshotDialogMap))

// Construct a dialog box
VVScreenshotDialog::VVScreenshotDialog(FXWindow* owner, vvCanvas* c):FXDialogBox(owner,"Settings",
  DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE,100,100)
{
  _shell = (VVShell*)owner;
  _canvas = c;
  FXVerticalFrame* master=new FXVerticalFrame(this, LAYOUT_FILL_X | LAYOUT_FILL_Y);

  FXHorizontalFrame* sizeFrame = new FXHorizontalFrame(master);
  _useScreenDim = new FXCheckButton(sizeFrame,"Use canvas size: ",this,ID_DIM,ICON_BEFORE_TEXT|LAYOUT_LEFT);
  _sizeLabel = new FXLabel(sizeFrame, "");
  
  FXGroupBox* sizeGroup = new FXGroupBox(master,"Image size",LAYOUT_LEFT | FRAME_GROOVE);

  new FXLabel(sizeGroup, "Width:");
  _widthTField = new FXTextField(sizeGroup, 25, NULL,0,TEXTFIELD_INTEGER | TEXTFIELD_NORMAL);
  _widthTField->setText(FXStringFormat("%d",300));

  new FXLabel(sizeGroup, "Height:");
  _heightTField = new FXTextField(sizeGroup, 25, NULL,0,TEXTFIELD_INTEGER | TEXTFIELD_NORMAL);
  _heightTField->setText(FXStringFormat("%d",400));

  FXHorizontalFrame* dirFrame = new FXHorizontalFrame(master);
  new FXLabel(dirFrame, "Directory:");
  _dirTField = new FXTextField(dirFrame, 25, NULL,0,TEXTFIELD_NORMAL);
  new FXButton(dirFrame,"Browse...",NULL,this,ID_BROWSE,FRAME_RAISED | FRAME_THICK);
    
  FXHorizontalFrame* fileFrame = new FXHorizontalFrame(master, LAYOUT_FILL_X);
  new FXLabel(fileFrame, "Base file name:");
  _fileTField = new FXTextField(fileFrame, 25, NULL,0,TEXTFIELD_NORMAL | LAYOUT_FILL_X);
  _fileTField->setText("image");
  
  FXHorizontalFrame* buttonFrame = new FXHorizontalFrame(master, LAYOUT_CENTER_X | LAYOUT_FILL_X | LAYOUT_FILL_Y | PACK_UNIFORM_WIDTH);
  new FXButton(buttonFrame,"Take Picture",NULL,this,ID_PICTURE, FRAME_RAISED | FRAME_THICK | LAYOUT_LEFT);
  new FXButton(buttonFrame,"Close",NULL,this,ID_ACCEPT, FRAME_RAISED | FRAME_THICK | LAYOUT_RIGHT);

  _useScreenDim->setCheck(true);
  onDimSelect(this, ID_DIM, (void*)true);
}

// Must delete the menus
VVScreenshotDialog::~VVScreenshotDialog()
{
  delete _fileTField;
  delete _widthTField;
  delete _heightTField;
}

long VVScreenshotDialog::onCmdPicture(FXObject*,FXSelector,void*)
{
  FXString filename = _dirTField->getText();
  if(filename.length() < 1) return 1;
#ifdef WIN32  
  filename.append("\\");
#else
  filename.append("/");
#endif
  filename.append(_fileTField->getText());
  int width = 0;
  int height = 0;
  if(!_useScreenDim->getCheck())
  {
    width  = FXIntVal(_widthTField->getText());
    height = FXIntVal(_heightTField->getText());
  }
  else
  {
    width  = _shell->_glcanvas->getWidth();
    height = _shell->_glcanvas->getHeight();
  }
  this->hide();     // prevent dialog window from blocking OpenGL canvas
  _shell->takeScreenshot(filename.text(), width, height);
  this->show();
  return 1;
}

long VVScreenshotDialog::onDimSelect(FXObject*, FXSelector, void* ptr)
{
  FXColor c;

  _widthTField->setEditable(ptr == NULL);
  _heightTField->setEditable(ptr == NULL);
  if (ptr) c = FXRGB(150,150,150);  // gray
  else     c = FXRGB(0,0,0);   // black
  _widthTField->setTextColor(c);
  _heightTField->setTextColor(c);
  return 1;
}

long VVScreenshotDialog::onBrowseSelect(FXObject*,FXSelector,void*)
{
  FXString filename = _shell->getOpenDirectory("Screenshot Directory");
  if(filename.length() == 0) return 1;
  _dirTField->setText(filename);
  return 1;
}

void VVScreenshotDialog::updateValues()
{
  _widthTField->setText(FXStringFormat("%d",400));
  _heightTField->setText(FXStringFormat("%d",300));
  FXString path = getApp()->reg().readStringEntry("Settings", "CurrentDirectory", getCurrentDirectory().text());
  _dirTField->setText(path);
}

/*******************************************************************************/

/*******************************************************************************/

FXDEFMAP(VVMovieDialog) VVMovieDialogMap[]=
{

  FXMAPFUNC(SEL_COMMAND, VVMovieDialog::ID_SELECT,         VVMovieDialog::onCmdSelect),
  FXMAPFUNC(SEL_COMMAND, VVMovieDialog::ID_RELOAD,         VVMovieDialog::onCmdReload),
  FXMAPFUNC(SEL_COMMAND, VVMovieDialog::ID_BACK_BACK_BACK, VVMovieDialog::onCmdBBB),
  FXMAPFUNC(SEL_COMMAND, VVMovieDialog::ID_BACK_BACK,      VVMovieDialog::onCmdBB),
  FXMAPFUNC(SEL_COMMAND, VVMovieDialog::ID_BACK,           VVMovieDialog::onCmdB),
  FXMAPFUNC(SEL_COMMAND, VVMovieDialog::ID_FWD_FWD_FWD,    VVMovieDialog::onCmdFFF),
  FXMAPFUNC(SEL_COMMAND, VVMovieDialog::ID_FWD_FWD,        VVMovieDialog::onCmdFF),
  FXMAPFUNC(SEL_COMMAND, VVMovieDialog::ID_FWD,            VVMovieDialog::onCmdF),
  FXMAPFUNC(SEL_COMMAND, VVMovieDialog::ID_WND_SIZE,       VVMovieDialog::onCmdSize),
  FXMAPFUNC(SEL_COMMAND, VVMovieDialog::ID_WRITE,          VVMovieDialog::onCmdWrite),
  FXMAPFUNC(SEL_COMMAND, VVMovieDialog::ID_HELP,           VVMovieDialog::onCmdHelp),
  FXMAPFUNC(SEL_COMMAND, VVMovieDialog::ID_BROWSE,         VVMovieDialog::onBrowseSelect),
  FXMAPFUNC(SEL_COMMAND, VVMovieDialog::ID_SLIDER,         VVMovieDialog::onSliderChng),
};

FXIMPLEMENT(VVMovieDialog,FXDialogBox,VVMovieDialogMap,ARRAYNUMBER(VVMovieDialogMap))

// Construct a dialog box
VVMovieDialog::VVMovieDialog(FXWindow* owner, vvCanvas* c):FXDialogBox(owner,"Movie Script",
  DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE,100,100)
{
  _shell = (VVShell*)owner;
  _canvas = c;
  assert(_canvas);
  _movie = new vvMovie(_canvas);
  FXVerticalFrame* master=new FXVerticalFrame(this,LAYOUT_FILL_X|LAYOUT_FILL_Y);

  FXHorizontalFrame* selectFrame = new FXHorizontalFrame(master, LAYOUT_FILL_X);
  new FXLabel(selectFrame,"Selected Movie Script:");
  _selectedLabel = new FXLabel(selectFrame, "",NULL, LAYOUT_LEFT | LAYOUT_FILL_X | LAYOUT_FILL_Y | FRAME_SUNKEN);
  new FXButton(selectFrame,"Select...",NULL,this,ID_SELECT,FRAME_RAISED|FRAME_THICK);
  new FXButton(selectFrame,"Reload",NULL,this,ID_RELOAD,FRAME_RAISED|FRAME_THICK);

  FXGroupBox* stepGroup = new FXGroupBox(master,"", LAYOUT_FILL_X | FRAME_GROOVE);

  FXHorizontalFrame* currentFrame=new FXHorizontalFrame(stepGroup, LAYOUT_CENTER_X);
  new FXLabel(currentFrame,"Current Script Step:");
  _stepLabel1 = new FXLabel(currentFrame, "1",NULL,LABEL_NORMAL);
  new FXLabel(currentFrame, "of");
  _stepLabel2 = new FXLabel(currentFrame, "1",NULL,LABEL_NORMAL);

  FXVerticalFrame* sliderFrame=new FXVerticalFrame(stepGroup, LAYOUT_CENTER_X | LAYOUT_FILL_X);
  _frameSlider=new FXSlider(sliderFrame,this,ID_SLIDER, LAYOUT_FILL_X | SLIDER_HORIZONTAL);
  _frameSlider->setRange(1,1);
  _frameSlider->setValue(1);
  _frameSlider->setTickDelta(1);

  FXHorizontalFrame* vcrFrame=new FXHorizontalFrame(stepGroup, LAYOUT_FILL_X | PACK_UNIFORM_WIDTH);
  new FXButton(vcrFrame,"|<<",NULL,this,ID_BACK_BACK_BACK,FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);
  new FXButton(vcrFrame,"|<", NULL,this,ID_BACK_BACK,     FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);
//  new FXButton(vcrFrame,"<",  NULL,this,ID_BACK,          FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);
//  new FXButton(vcrFrame,">",  NULL,this,ID_FWD,           FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);
  new FXButton(vcrFrame,">|", NULL,this,ID_FWD_FWD,       FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);
  new FXButton(vcrFrame,">>|",NULL,this,ID_FWD_FWD_FWD,   FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);

  FXHorizontalFrame* sizeFrame = new FXHorizontalFrame(master, LAYOUT_CENTER_X);
  _useScreenDim = new FXCheckButton(sizeFrame,"Use canvas size: ",this,ID_WND_SIZE,ICON_BEFORE_TEXT);
  _useScreenDim->setCheck(true);
  _sizeLabel = new FXLabel(sizeFrame, "");

  FXGroupBox* sizeGroup = new FXGroupBox(master,"Image size", LAYOUT_CENTER_X | FRAME_GROOVE);
  new FXLabel(sizeGroup, "Image width:");
  _widthTField = new FXTextField(sizeGroup, 25, NULL,0,TEXTFIELD_INTEGER | TEXTFIELD_NORMAL);
  _widthTField->setText(FXStringFormat("%d", 352));
  new FXLabel(sizeGroup, "Image height:");
  _heightTField = new FXTextField(sizeGroup, 25, NULL,0,TEXTFIELD_INTEGER | TEXTFIELD_NORMAL);
  _heightTField->setText(FXStringFormat("%d", 240));

  FXHorizontalFrame* dirFrame = new FXHorizontalFrame(master, LAYOUT_FILL_X);
  new FXLabel(dirFrame, "Directory:");
  
#ifdef WIN32
  FXString prefix("\\img");
#else
  FXString prefix("/img");
#endif
  FXString defaultPath = getCurrentDirectory() + prefix;
  FXString path = getApp()->reg().readStringEntry("Settings", "CurrentDirectory", defaultPath.text());
  _dirTField = new FXTextField(dirFrame, 25, NULL,0,TEXTFIELD_NORMAL | LAYOUT_FILL_X);
  _dirTField->setText(path);
  new FXButton(dirFrame,"Browse...",NULL,this,ID_BROWSE, FRAME_RAISED | FRAME_THICK);

  FXHorizontalFrame* fileFrame = new FXHorizontalFrame(master, LAYOUT_FILL_X);
  new FXLabel(fileFrame, "Base file name:");
  _fileTField = new FXTextField(fileFrame, 25, NULL,0,TEXTFIELD_NORMAL | LAYOUT_FILL_X);
  _fileTField->setText("image");

  FXHorizontalFrame* buttonFrame = new FXHorizontalFrame(master, LAYOUT_CENTER_X | LAYOUT_FILL_X | LAYOUT_FILL_Y | PACK_UNIFORM_WIDTH);
  new FXButton(buttonFrame,"Write Movie to Disk",NULL,this,ID_WRITE, FRAME_RAISED | FRAME_THICK | LAYOUT_LEFT);
  new FXButton(buttonFrame,"Help...",NULL,this,ID_HELP, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X);
  new FXButton(buttonFrame,"Close",NULL,this,ID_ACCEPT, FRAME_RAISED | FRAME_THICK | LAYOUT_RIGHT);
}

// Must delete the menus
VVMovieDialog::~VVMovieDialog()
{
  delete _selectedLabel;
  delete _stepLabel1;
  delete _stepLabel2;
  delete _frameSlider;
  delete _fileTField;
  delete _dirTField;
  delete _widthTField;
  delete _heightTField;
  delete _movie;
}

long VVMovieDialog::onCmdSelect(FXObject*, FXSelector, void*)
{
  vvDebugMsg::msg(1, "VVMovieDialog::onCmdSelect()");

  // Use file browser to select movie script:
  const FXchar patterns[] = "Movie scripts (*.vms)\nAll Files (*.*)";
  FXString filename = _shell->getOpenFilename("Load Movie Script", patterns);
  if (filename.length() == 0) return 1;
  _selectedLabel->setText(filename);

  switch (_movie->load(filename.text()))
  {
    case vvMovie::VV_OK:
      vvDebugMsg::msg(2, "Loaded file: ", filename.text());
      _stepLabel2->setText(FXStringFormat("%d", _movie->getNumScriptSteps()));
      _frameSlider->setRange(1, _movie->getNumScriptSteps());
      setMovieStep(0);
      return 0;
    case vvMovie::VV_INVALID_PARAM:
      FXMessageBox::information((FXWindow*)this, MBOX_OK, "Error", "Invalid syntax in movie file.");
      return 1;
    case vvMovie::VV_FILE_ERROR:
    default:
      vvDebugMsg::msg(2, "Cannot load movie file: ", filename.text());
      return 1;
  }
}

long VVMovieDialog::onCmdReload(FXObject*,FXSelector,void*)
{
  FXString filename = _selectedLabel->getText();
  if (filename.length()==0) return 1;

  switch (_movie->load(filename.text()))
  {
    case vvMovie::VV_OK:
      vvDebugMsg::msg(2, "Reloaded file: ", filename.text());
      _stepLabel2->setText(FXStringFormat("%d", _movie->getNumScriptSteps()));
      _frameSlider->setRange(1, _movie->getNumScriptSteps());
      setMovieStep(0);
      return 0;
    case vvMovie::VV_INVALID_PARAM:
      FXMessageBox::information((FXWindow*)this, MBOX_OK, "Error", "Invalid syntax in movie file.");
      return 1;
    case vvMovie::VV_FILE_ERROR:
    default:
      vvDebugMsg::msg(2, "Cannot load movie file: ", filename.text());
      return 1;
  }
}

long VVMovieDialog::onCmdBBB(FXObject*,FXSelector,void*)
{
  setMovieStep(0);
  return 1;
}

long VVMovieDialog::onCmdBB(FXObject*,FXSelector,void*)
{
  int step = _movie->getStep();
  --step;
  step = ts_max(0, step);
  setMovieStep(step);
  return 1;
}

long VVMovieDialog::onCmdB(FXObject*,FXSelector,void*)
{
  cerr << "TODO: playback backwards" << endl;
  return 1;
}

long VVMovieDialog::onCmdF(FXObject*,FXSelector,void*)
{
  cerr << "TODO: playback forward" << endl;
  return 1;
}

long VVMovieDialog::onCmdFF(FXObject*,FXSelector,void*)
{
  int step = _movie->getStep();
  ++step;
  step = ts_min(_movie->getNumScriptSteps() - 1, step);
  step = ts_max(step, 0);
  setMovieStep(step);
  return 1;
}

long VVMovieDialog::onCmdFFF(FXObject*,FXSelector,void*)
{
  setMovieStep(ts_max(0, _movie->getNumScriptSteps() - 1));
  return 1;
}

/// 0 = first step
void VVMovieDialog::setMovieStep(int s)
{
  if(_shell->_glcanvas->makeCurrent())
  {
    _movie->setStep(s);
    _frameSlider->setValue(s + 1);
    _stepLabel1->setText(FXStringFormat("%d", s + 1));
    _canvas->draw();
    _shell->_glcanvas->makeNonCurrent();
  }
}

long VVMovieDialog::onCmdSize(FXObject*,FXSelector,void *ptr)
{
  (void)ptr;
  updateValues();
  return 1;
}

long VVMovieDialog::onBrowseSelect(FXObject*,FXSelector,void*)
{
  FXString filename = _shell->getOpenDirectory("Movie Directory");
  if(filename.length() == 0) return 1;
  _dirTField->setText(filename);
  return 1;
}

long VVMovieDialog::onCmdWrite(FXObject*,FXSelector,void*)
{
  int width, height;
  if(!_useScreenDim->getCheck())
  {
    width  = FXIntVal(_widthTField->getText());
    height = FXIntVal(_heightTField->getText());
  }
  else
  {
    width  = _shell->_glcanvas->getWidth();
    height = _shell->_glcanvas->getHeight();
  }
  this->hide();   // prevent dialog window from blocking OpenGL canvas
  if(_shell->_glcanvas->makeCurrent())
  {
    _movie->write(width, height, _fileTField->getText().text());
    _shell->_glcanvas->makeNonCurrent();
  }
  this->show();

  return 1;
}

long VVMovieDialog::onCmdHelp(FXObject*,FXSelector,void*)
{
  const char* info = "Movie script commands:\n\n" \
    "changequality RELATIVE_QUALITY\n" \
    "movepeak DISTANCE\n" \
    "nextstep\n" \
    "prevstep\n" \
    "rot AXIS ANGLE\n" \
    "scale FACTOR\n" \
    "setpeak POS WIDTH\n" \
    "setquality QUALITY\n" \
    "show\n" \
    "timestep INDEX\n" \
    "trans AXIS DIST\n\n" \
    "For a detailed description of the movie script file format\n" \
    "please refer to the section 'Movie Scripts' in the file readme.txt.";
  FXMessageBox::information((FXWindow*)this, MBOX_OK, "File Format", "%s", info);
  return 1;
}

long VVMovieDialog::onSliderChng(FXObject*, FXSelector, void* value)
{
  setMovieStep((*(int*)value) - 1);
  return 1;
}

void VVMovieDialog::updateValues()
{
  _widthTField->setEditable(_useScreenDim->getCheck());
  _heightTField->setEditable(_useScreenDim->getCheck());
  if(_useScreenDim->getCheck())
  {
    //set color here
    FXColor c = FXRGB(150,150,150);
    _widthTField->setTextColor(c);
    _heightTField->setTextColor(c);
  }
  else
  {
    FXColor c = FXRGB(0,0,0);
    _widthTField->setTextColor(c);
    _heightTField->setTextColor(c);
  }
}

/*******************************************************************************/

FXDEFMAP(VVTimeStepDialog) VVTimeStepDialogMap[]=
{

  FXMAPFUNC(SEL_COMMAND,    VVTimeStepDialog::ID_BACK_BACK, VVTimeStepDialog::onCmdBackBack),
  FXMAPFUNC(SEL_COMMAND,    VVTimeStepDialog::ID_BACK,      VVTimeStepDialog::onCmdBack),
  FXMAPFUNC(SEL_COMMAND,    VVTimeStepDialog::ID_PLAY,      VVTimeStepDialog::onCmdPlay),
  FXMAPFUNC(SEL_COMMAND,    VVTimeStepDialog::ID_FWD,       VVTimeStepDialog::onCmdFwd),
  FXMAPFUNC(SEL_COMMAND,    VVTimeStepDialog::ID_FWD_FWD,   VVTimeStepDialog::onCmdFwdFwd),
  FXMAPFUNC(SEL_CHANGED,    VVTimeStepDialog::ID_SLIDER,    VVTimeStepDialog::onChngSlider),
};

FXIMPLEMENT(VVTimeStepDialog,FXDialogBox,VVTimeStepDialogMap,ARRAYNUMBER(VVTimeStepDialogMap))

// Construct a dialog box
VVTimeStepDialog::VVTimeStepDialog(FXWindow* owner, vvCanvas* c):FXDialogBox(owner,"Animation",DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE,100,100)
{
  _shell = (VVShell*)owner;
  _canvas = c;

  FXVerticalFrame* master=new FXVerticalFrame(this,LAYOUT_FILL_X|LAYOUT_FILL_Y,0,0,0,0, 0,0,0,0);

  FXHorizontalFrame* speedFrame = new FXHorizontalFrame(master,LAYOUT_FILL_X|LAYOUT_FILL_Y,0,0,0,0, 0,0,0,0);

  new FXLabel(speedFrame, "Animation speed [frames/sec]:");
  _speedTField = new FXTextField(speedFrame, 25, NULL,0,TEXTFIELD_NORMAL);

  FXGroupBox* sliderGP=new FXGroupBox(master,"Time Step",LAYOUT_SIDE_TOP|FRAME_GROOVE|LAYOUT_FILL_X, 0,0,0,0);
  FXHorizontalFrame* sliderFrame=new FXHorizontalFrame(sliderGP,LAYOUT_FIX_WIDTH|LAYOUT_FILL_Y,0,0,350,0, 0,0,0,0);
  _stepLabel = new FXLabel(sliderFrame, FXStringFormat("%d", 0),NULL,LABEL_NORMAL|FRAME_SUNKEN|FRAME_THICK);
  _stepSlider = new FXSlider(sliderFrame,this,ID_SLIDER,LAYOUT_TOP|LAYOUT_FIX_WIDTH|LAYOUT_FIX_HEIGHT|SLIDER_HORIZONTAL|LAYOUT_RIGHT|SLIDER_ARROW_DOWN|SLIDER_TICKS_BOTTOM,0,0,300,30);
  _stepSlider->setRange(0,10);
  _stepSlider->setValue(0);
  _stepSlider->setTickDelta(1);

  FXHorizontalFrame* buttonFrame = new FXHorizontalFrame(master,LAYOUT_FILL_X | LAYOUT_FILL_Y | PACK_UNIFORM_WIDTH);
  new FXButton(buttonFrame,"|<<",NULL,this,ID_BACK_BACK,FRAME_RAISED|FRAME_THICK|LAYOUT_CENTER_X|LAYOUT_CENTER_Y,0,0,0,0, 20,20,3,3);
  new FXButton(buttonFrame,"|<", NULL,this,ID_BACK,     FRAME_RAISED|FRAME_THICK|LAYOUT_CENTER_X|LAYOUT_CENTER_Y,0,0,0,0, 20,20,3,3);
  _playButton = new FXButton(buttonFrame,">",  NULL,this,ID_PLAY,     FRAME_RAISED|FRAME_THICK|LAYOUT_CENTER_X|LAYOUT_CENTER_Y,0,0,0,0, 20,20,3,3);
  new FXButton(buttonFrame,">|", NULL,this,ID_FWD,      FRAME_RAISED|FRAME_THICK|LAYOUT_CENTER_X|LAYOUT_CENTER_Y,0,0,0,0, 20,20,3,3);
  new FXButton(buttonFrame,">>|",NULL,this,ID_FWD_FWD,  FRAME_RAISED|FRAME_THICK|LAYOUT_CENTER_X|LAYOUT_CENTER_Y,0,0,0,0, 20,20,3,3);

  new FXButton(master,"Close",NULL,this,ID_ACCEPT,FRAME_RAISED|FRAME_THICK|LAYOUT_CENTER_X|LAYOUT_CENTER_Y,0,0,0,0, 20,20,3,3);
}

// Must delete the menus
VVTimeStepDialog::~VVTimeStepDialog()
{
  delete _stepLabel;
}

long VVTimeStepDialog::onCmdBack(FXObject*,FXSelector,void*)
{
  stepBack();
  return 1;
}

long VVTimeStepDialog::onCmdBackBack(FXObject*,FXSelector,void*)
{
  gotoStart();
  return 1;
}

long VVTimeStepDialog::onCmdPlay(FXObject*,FXSelector,void*)
{
  playback();
  return 1;
}

long VVTimeStepDialog::onCmdFwd(FXObject*,FXSelector,void*)
{
  stepForward();
  return 1;
}

long VVTimeStepDialog::onCmdFwdFwd(FXObject*,FXSelector,void*)
{
  gotoEnd();
  return 1;
}

long VVTimeStepDialog::onChngSlider(FXObject*,FXSelector,void*)
{
  setTimeStep(_stepSlider->getValue() - 1);
  return 1;
}

/** @param newStep [0..numSteps-1]
*/
void VVTimeStepDialog::setTimeStep(int newStep)
{
  _stepLabel->setText(FXStringFormat("%d", newStep + 1));
  _canvas->_renderer->setCurrentFrame(newStep);
  _shell->drawScene();
}

void VVTimeStepDialog::gotoStart()
{
  _stepSlider->setValue(1);
  setTimeStep(0);
}

void VVTimeStepDialog::stepBack()
{
  int curr = _canvas->_renderer->getCurrentFrame();
  curr = (curr > 0) ? curr - 1 : _canvas->_vd->frames - 1;
  _stepSlider->setValue(curr + 1);
  setTimeStep(curr);
}

/** @param factor 1.0 does not change the speed, greater values make it faster
*/
void VVTimeStepDialog::scaleSpeed(float factor)
{
  float fps;

  if (factor==0.0f) 
  {
    cerr << "Animation speed cannot be zero." << endl;
    return;
  }
  fps = FXFloatVal(_speedTField->getText().text());
  fps *= factor;
  _canvas->_vd->setDt(1.0f / fps);
  _speedTField->setText(FXStringFormat("%.1f", fps));
}

void VVTimeStepDialog::playback()
{
  FXString speed = _speedTField->getText();
  _canvas->_vd->setDt(1.0f / FXFloatVal(speed.text()));

  FXString buttonText = _playButton->getText();
  if (buttonText==">")    // play?
  {
    _playButton->setText("||");
    _shell->startAnimTimer();
  }
  else    // stop
  {
    _playButton->setText(">");
    _shell->stopAnimTimer();
  }
}

void VVTimeStepDialog::stepForward()
{
  size_t curr = _canvas->_renderer->getCurrentFrame();
  curr = (curr >= _canvas->_vd->frames - 1) ? 0 : curr + 1;
  _stepSlider->setValue(curr + 1);
  setTimeStep(curr);
}

void VVTimeStepDialog::gotoEnd()
{
  _stepSlider->setValue(_canvas->_vd->frames);
  setTimeStep(_canvas->_vd->frames - 1);
}

void VVTimeStepDialog::updateValues()
{
  int numSteps = _canvas->_vd->frames;
  _stepSlider->setRange(1, numSteps);
  _stepSlider->setTickDelta((numSteps>=10) ? (numSteps / 10) : 1);
  _speedTField->setText(FXStringFormat("%.2f", 1.0f / _canvas->_vd->getDt()));
  setTimeStep(_canvas->_vd->getCurrentFrame());
}

/*******************************************************************************/

FXDEFMAP(VVDiagramDialog) VVDiagramDialogMap[]=
{
  FXMAPFUNC(SEL_COMMAND, VVDiagramDialog::ID_APPLY, VVDiagramDialog::onApplySelect),
};

FXIMPLEMENT(VVDiagramDialog,FXDialogBox,VVDiagramDialogMap,ARRAYNUMBER(VVDiagramDialogMap))

// Construct a dialog box
VVDiagramDialog::VVDiagramDialog(FXWindow* owner, vvCanvas* c) : FXDialogBox(owner,"Draw Diagrams",
  DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE,100,100)
{
  _shell = (VVShell*)owner;
  _canvas = c;
  FXVerticalFrame* master = new FXVerticalFrame(this, LAYOUT_FILL_X | LAYOUT_FILL_Y, 0,0,0,0, 0,0,0,0);

  FXMatrix* fieldMat = new FXMatrix(master,2,MATRIX_BY_COLUMNS);

  new FXLabel(fieldMat, "Start point x [voxels]:");
  _startXTField = new FXTextField(fieldMat, 10, NULL,0,TEXTFIELD_INTEGER);
  _startXTField->setText(FXStringFormat("%d",0));
  new FXLabel(fieldMat, "Start point y [voxels]:");
  _startYTField = new FXTextField(fieldMat, 10, NULL,0,TEXTFIELD_INTEGER);
  _startYTField->setText(FXStringFormat("%d",0));
  new FXLabel(fieldMat, "Start point z [voxels]:");
  _startZTField = new FXTextField(fieldMat, 10, NULL,0,TEXTFIELD_INTEGER);
  _startZTField->setText(FXStringFormat("%d",0));

  new FXLabel(fieldMat, "End point x [voxels]:");
  _endXTField = new FXTextField(fieldMat, 10, NULL,0,TEXTFIELD_INTEGER);
  _endXTField->setText(FXStringFormat("%d",31));
  new FXLabel(fieldMat, "End point y [voxels]:");
  _endYTField = new FXTextField(fieldMat, 10, NULL,0,TEXTFIELD_INTEGER);
  _endYTField->setText(FXStringFormat("%d",31));
  new FXLabel(fieldMat, "End point z [voxels]:");
  _endZTField = new FXTextField(fieldMat, 10, NULL,0,TEXTFIELD_INTEGER);
  _endZTField->setText(FXStringFormat("%d",31));

  FXMatrix* buttonMat = new FXMatrix(master,3,MATRIX_BY_COLUMNS | LAYOUT_CENTER_X | PACK_UNIFORM_WIDTH);
  new FXButton(buttonMat,"Apply",NULL,this,ID_APPLY,FRAME_RAISED|FRAME_THICK|LAYOUT_CENTER_X|LAYOUT_CENTER_Y,0,0,0,0, 20,20,3,3);
  new FXButton(buttonMat,"Cancel",NULL,this,ID_CANCEL,FRAME_RAISED|FRAME_THICK|LAYOUT_CENTER_X|LAYOUT_CENTER_Y,0,0,0,0, 20,20,3,3);

  _histWindow = new VVHistWindow(this);
}

// Must delete the menus
VVDiagramDialog::~VVDiagramDialog()
{
  delete _startXTField;
  delete _startYTField;
  delete _startZTField;
  delete _endXTField;
  delete _endYTField;
  delete _endZTField;
  delete _histWindow;
}

long VVDiagramDialog::onApplySelect(FXObject*,FXSelector,void*)
{
  int x1 = FXIntVal(_startXTField->getText());
  int y1 = FXIntVal(_startYTField->getText());
  int z1 = FXIntVal(_startZTField->getText());

  int x2 = FXIntVal(_endXTField->getText());
  int y2 = FXIntVal(_endYTField->getText());
  int z2 = FXIntVal(_endZTField->getText());

  _histWindow->_valArray.clear();

  _canvas->_vd->getLineHistData(x1, y1, z1, x2, y2, z2, _histWindow->_valArray);
  _histWindow->_channels = _canvas->_vd->getBPV();
  _histWindow->drawHist();
  _histWindow->show();
  _shell->drawScene();
  return 1;
}

void VVDiagramDialog::updateValues()
{
  _startXTField->setText(FXStringFormat("%d",0));
  _startYTField->setText(FXStringFormat("%d",0));
  _startZTField->setText(FXStringFormat("%d",0));

  _endXTField->setText(FXStringFormat("%d",31));
  _endYTField->setText(FXStringFormat("%d",31));
  _endZTField->setText(FXStringFormat("%d",31));
}

/*******************************************************************************/

FXDEFMAP(VVHistWindow) VVHistWindowMap[]=
{
  FXMAPFUNC(SEL_COMMAND, VVHistWindow::ID_CLEAR,  VVHistWindow::onClearSelect),
  FXMAPFUNC(SEL_PAINT,   VVHistWindow::ID_CANVAS, VVHistWindow::onPaint),
};

FXIMPLEMENT(VVHistWindow,FXDialogBox,VVHistWindowMap,ARRAYNUMBER(VVHistWindowMap))

// Construct a dialog box
VVHistWindow::VVHistWindow(FXWindow* owner):FXDialogBox(owner, "Histogram",
  DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE, 100, 100), _colorArray(), _valArray()
{
  _channels = 0;

  FXVerticalFrame* master=new FXVerticalFrame(this,LAYOUT_FILL_X|LAYOUT_FILL_Y,0,0,0,0, 0,0,0,0);
  FXVerticalFrame* canvasFrame=new FXVerticalFrame(master, FRAME_SUNKEN|LAYOUT_SIDE_LEFT|LAYOUT_FIX_WIDTH|LAYOUT_FIX_HEIGHT, 0,0,400,400);
  _histCanvas=new FXCanvas(canvasFrame,this,ID_CANVAS,FRAME_SUNKEN|LAYOUT_FILL_X|LAYOUT_FILL_Y|LAYOUT_FIX_HEIGHT, 0,0,400,400);

  new FXButton(master,"OK",NULL,this,ID_ACCEPT,FRAME_RAISED|FRAME_THICK|LAYOUT_CENTER_X|LAYOUT_CENTER_Y,0,0,0,0, 20,20,3,3);

  createColorLookup();
}

// Must delete the menus
VVHistWindow::~VVHistWindow()
{
  delete _histCanvas;
}

long VVHistWindow::onPaint(FXObject*,FXSelector,void*)
{
  FXDCWindow dc(_histCanvas);
  dc.setForeground(FXRGB(255,255,255));
  dc.fillRectangle(0,0,400,400);

  drawHist();

  return 1;
}

void VVHistWindow::createColorLookup()
{
  _colorArray.push_back(FXRGB(255,0,0));
  _colorArray.push_back(FXRGB(0,255,0));
  _colorArray.push_back(FXRGB(0,0,255));
  _colorArray.push_back(FXRGB(255,255,0));
}

void VVHistWindow::drawHist()
{
  FXDCWindow dc(_histCanvas);
  dc.setForeground(FXRGB(255,255,255));
  dc.fillRectangle(0,0,400,400);

  if (_channels == 0 || _valArray.size() == 0) return;

  int hOffset = 20;
  int wOffset = 20;
  int canvWidth = 400;
  int canvHeight = 400;
  dc.setForeground(FXRGB(0,0,0));
  dc.drawLine(wOffset, canvHeight-hOffset, canvWidth-wOffset, canvHeight-hOffset);
  dc.drawLine(wOffset, hOffset, wOffset, canvHeight-hOffset);
  float temp,temp1;
  int totalWidth = canvWidth-2*wOffset;
  int totalHeight = canvHeight-2*hOffset;
  int zeroHeight = canvHeight-hOffset;
  int indWidth = totalWidth / (_valArray.size()-1);
  double scale = ((double)totalHeight)/255.0;
  int currWidth = wOffset;
  for(size_t i = 0; i < _valArray.size() - 1; ++i)
  {
    for(size_t j = 0; j < _channels; ++j)
    {
      temp = _valArray[i][j];
      temp1 = _valArray[i+1][j];
      dc.setForeground(_colorArray[j % _colorArray.size()]);
      dc.drawLine(currWidth, int(zeroHeight-temp*scale), currWidth + indWidth, int(zeroHeight-temp1*scale));
    }
    currWidth += indWidth;
  }
}

long VVHistWindow::onClearSelect(FXObject*,FXSelector,void*)
{
  FXDCWindow dc(_histCanvas);
  dc.setForeground(FXRGB(255,255,255));
  dc.fillRectangle(0,0,400,400);
  return 1;
}

/*******************************************************************************/

FXDEFMAP(VVGammaDialog) VVGammaDialogMap[]=
{
  FXMAPFUNC(SEL_COMMAND, VVGammaDialog::ID_CLOSE,    VVGammaDialog::onClose),
  FXMAPFUNC(SEL_COMMAND, VVGammaDialog::ID_DEFAULTS, VVGammaDialog::onSetDefaults),
  FXMAPFUNC(SEL_COMMAND, VVGammaDialog::ID_GAMMA,    VVGammaDialog::onUseGammaChange),
  FXMAPFUNC(SEL_CHANGED, VVGammaDialog::ID_GRED,     VVGammaDialog::onGammaChange),
  FXMAPFUNC(SEL_CHANGED, VVGammaDialog::ID_GGREEN,   VVGammaDialog::onGammaChange),
  FXMAPFUNC(SEL_CHANGED, VVGammaDialog::ID_GBLUE,    VVGammaDialog::onGammaChange),
  FXMAPFUNC(SEL_CHANGED, VVGammaDialog::ID_GFOUR,    VVGammaDialog::onGammaChange),
};

FXIMPLEMENT(VVGammaDialog,FXDialogBox,VVGammaDialogMap,ARRAYNUMBER(VVGammaDialogMap))

VVGammaDialog::VVGammaDialog()
{
}

// Construct a dialog box
VVGammaDialog::VVGammaDialog(FXWindow* owner, vvCanvas* c) : 
  FXDialogBox(owner, "Gamma Correction", DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE)
{
  _parent = (VVShell*)owner;
  _canvas = c;
  FXVerticalFrame* verticalFrame = new FXVerticalFrame(this, LAYOUT_FILL_X | LAYOUT_FILL_Y);

  _gammaCheck = new FXCheckButton(verticalFrame, "Enable gamma correction", this, ID_GAMMA, ICON_BEFORE_TEXT);

  // GRed:
  FXMatrix* gRedMatrix = new FXMatrix(verticalFrame, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(gRedMatrix, "Gamma red:");
  _gRedLabel = new FXLabel(gRedMatrix, "", NULL, LAYOUT_RIGHT);
  _gRedDial = new FXDial(verticalFrame, this, ID_GRED, DIAL_HORIZONTAL | LAYOUT_FILL_X);
  _gRedDial->setRange(0, 10000);
  _gRedDial->setValue(100);
  _gRedDial->setRevolutionIncrement(200);
  _gRedDial->setNotchSpacing(100);

  // GGreen:
  FXMatrix* gGreenMatrix = new FXMatrix(verticalFrame, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(gGreenMatrix, "Gamma green:");
  _gGreenLabel = new FXLabel(gGreenMatrix, "");
  _gGreenDial = new FXDial(verticalFrame, this, ID_GGREEN, DIAL_HORIZONTAL | LAYOUT_FILL_X);
  _gGreenDial->setRange(0, 10000);
  _gGreenDial->setValue(100);
  _gGreenDial->setRevolutionIncrement(200);
  _gGreenDial->setNotchSpacing(100);

  // GBlue:
  FXMatrix* gBlueMatrix = new FXMatrix(verticalFrame, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(gBlueMatrix, "Gamma blue:");
  _gBlueLabel = new FXLabel(gBlueMatrix, "");
  _gBlueDial = new FXDial(verticalFrame, this, ID_GBLUE, DIAL_HORIZONTAL | LAYOUT_FILL_X);
  _gBlueDial->setRange(0, 10000);
  _gBlueDial->setValue(100);
  _gBlueDial->setRevolutionIncrement(200);
  _gBlueDial->setNotchSpacing(100);

  // GFour:
  FXMatrix* gFourMatrix = new FXMatrix(verticalFrame, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(gFourMatrix, "Gamma channel 4:");
  _gFourLabel = new FXLabel(gFourMatrix, "");
  _gFourDial = new FXDial(verticalFrame, this, ID_GFOUR, DIAL_HORIZONTAL | LAYOUT_FILL_X);
  _gFourDial->setRange(0, 10000);
  _gFourDial->setValue(100);
  _gFourDial->setRevolutionIncrement(200);
  _gFourDial->setNotchSpacing(100);

  // Buttons:
  FXHorizontalFrame* buttonFrame = new FXHorizontalFrame(verticalFrame, LAYOUT_CENTER_X | LAYOUT_FILL_Y | PACK_UNIFORM_WIDTH);
  new FXButton(buttonFrame, "Default", NULL, this, ID_DEFAULTS, LAYOUT_FILL_X | FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X | LAYOUT_CENTER_Y);
  new FXButton(buttonFrame, "Close", NULL, this, ID_CLOSE, LAYOUT_FILL_X | FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X | LAYOUT_CENTER_Y);

  updateValues();
  move(100, 100);
}

long VVGammaDialog::onClose(FXObject*, FXSelector, void*)
{
  handle(this, FXSEL(SEL_COMMAND, ID_HIDE), NULL);
  return 1;
}

long VVGammaDialog::onSetDefaults(FXObject*, FXSelector, void*)
{
  const float DEFAULT_VALUE = 1.0f;
  vec4 gamma(DEFAULT_VALUE, DEFAULT_VALUE, DEFAULT_VALUE, DEFAULT_VALUE);
  _canvas->_renderer->setParameter(vvRenderer::VV_GAMMA, gamma);
  updateValues();
  return 1;
}

long VVGammaDialog::onUseGammaChange(FXObject*, FXSelector, void* ptr)
{
  _canvas->_renderer->setParameter(vvRenderState::VV_GAMMA_CORRECTION, (ptr != NULL));
  if (ptr)
  {
    _gRedDial->enable();
    _gGreenDial->enable();
    _gBlueDial->enable();
    _gFourDial->enable();
  }
  else
  {
    _gRedDial->disable();
    _gGreenDial->disable();
    _gBlueDial->disable();
    _gFourDial->disable();
  }
  return 1;
}

long VVGammaDialog::onGammaChange(FXObject*,FXSelector,void*)
{
  vec4 val(getDialValue(_gRedDial), getDialValue(_gGreenDial),
    getDialValue(_gBlueDial), getDialValue(_gFourDial));
  _gRedLabel->setText(FXStringFormat("%.2f", val[0]));
  _gGreenLabel->setText(FXStringFormat("%.2f", val[1]));
  _gBlueLabel->setText(FXStringFormat("%.2f", val[2]));
  _gFourLabel->setText(FXStringFormat("%.2f", val[3]));
  _canvas->_renderer->setParameter(vvRenderer::VV_GAMMA, val);
  return 1;
}

float VVGammaDialog::getDialValue(FXDial* dial)
{
  return float(dial->getValue()) / 100.0f;
}

void VVGammaDialog::setDialValue(FXDial* dial, float val)
{
  dial->setValue(int(val * 100.0f));
}

void VVGammaDialog::updateValues()
{
  vec4 gamma(1.0f, 1.0f, 1.0f, 1.0f);
  if (_canvas->_renderer != NULL)
  {
    gamma = _canvas->_renderer->getParameter(vvRenderer::VV_GAMMA);
  }

  _gRedLabel->setText(FXStringFormat("%.2f", gamma[0]));
  _gGreenLabel->setText(FXStringFormat("%.2f", gamma[1]));
  _gBlueLabel->setText(FXStringFormat("%.2f", gamma[2]));
  _gFourLabel->setText(FXStringFormat("%.2f", gamma[3]));

  setDialValue(_gRedDial, gamma[0]);
  setDialValue(_gGreenDial, gamma[1]);
  setDialValue(_gBlueDial, gamma[2]);
  setDialValue(_gFourDial, gamma[3]);

  if (_canvas->_renderer) 
  { 
    bool g = _canvas->_renderer->getParameter(vvRenderState::VV_GAMMA_CORRECTION);
    handle(this, FXSEL(SEL_COMMAND, ID_GAMMA), (void*)g);
  }
}

/*******************************************************************************/

/*******************************************************************************/

FXDEFMAP(VVChannel4Dialog) VVChannel4DialogMap[]=
{
  FXMAPFUNC(SEL_COMMAND, VVChannel4Dialog::ID_OK,     VVChannel4Dialog::onOK),
  FXMAPFUNC(SEL_COMMAND, VVChannel4Dialog::ID_CANCEL, VVChannel4Dialog::onCancel),
  FXMAPFUNC(SEL_CHANGED, VVChannel4Dialog::ID_HUE,    VVChannel4Dialog::onHueChange),
  FXMAPFUNC(SEL_CHANGED, VVChannel4Dialog::ID_SAT,    VVChannel4Dialog::onSatChange),
};

FXIMPLEMENT(VVChannel4Dialog,FXDialogBox,VVChannel4DialogMap,ARRAYNUMBER(VVChannel4DialogMap))

// Construct a dialog box
VVChannel4Dialog::VVChannel4Dialog(FXWindow* owner, vvCanvas* c) : 
  FXDialogBox(owner, "Channel 4 Settings", DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE)
{
  _parent = (VVShell*)owner;
  _canvas = c;
  FXVerticalFrame* verticalFrame = new FXVerticalFrame(this, LAYOUT_FILL_X);

  // Hue:
  FXMatrix* hueMatrix = new FXMatrix(verticalFrame, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(hueMatrix, "Hue:");
  _hueLabel = new FXLabel(hueMatrix, "", NULL, LAYOUT_RIGHT | LABEL_NORMAL);
  _hueSlider = new FXRealSlider(verticalFrame, this, ID_HUE, LAYOUT_FILL_X | SLIDER_HORIZONTAL | LAYOUT_FIX_WIDTH, 0, 0, 150);
  _hueSlider->setRange(0,1);
  _hueSlider->setValue(1);

  // Saturation:
  FXMatrix* satMatrix = new FXMatrix(verticalFrame, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(satMatrix, "Saturation:");
  _satLabel = new FXLabel(satMatrix, "");
  _satSlider = new FXRealSlider(verticalFrame, this, ID_SAT, LAYOUT_FILL_X | SLIDER_HORIZONTAL);
  _satSlider->setRange(0,1);
  _satSlider->setValue(1);

  // Accept
  FXHorizontalFrame* buttonFrame = new FXHorizontalFrame(verticalFrame, LAYOUT_CENTER_X | LAYOUT_FILL_X | PACK_UNIFORM_WIDTH);
  new FXButton(buttonFrame, "Cancel", NULL, this, ID_CANCEL, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X | LAYOUT_CENTER_Y);
  new FXButton(buttonFrame, "OK", NULL, this, ID_OK, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X | LAYOUT_CENTER_Y);

  move(100, 100);
}

long VVChannel4Dialog::onHueChange(FXObject*,FXSelector,void*)
{
  float r, g, b;
  float h = float(_hueSlider->getValue());
  float s = float(_satSlider->getValue());
  _hueLabel->setText(FXStringFormat("%.2f", h));
  if (_canvas->_renderer) 
  {
    vvToolshed::HSBtoRGB(h, s, 1.0f, &r, &g, &b);
    _canvas->_renderer->setChannel4Color(vvRenderer::VV_RED, r);
    _canvas->_renderer->setChannel4Color(vvRenderer::VV_GREEN, g);
    _canvas->_renderer->setChannel4Color(vvRenderer::VV_BLUE, b);
  }
  return 1;
}

long VVChannel4Dialog::onSatChange(FXObject*,FXSelector,void*)
{
  float r, g, b;
  float h = float(_hueSlider->getValue());
  float s = float(_satSlider->getValue());
  _satLabel->setText(FXStringFormat("%.2f", s));
  if (_canvas->_renderer) 
  {
    vvToolshed::HSBtoRGB(h, s, 1.0f, &r, &g, &b);
    _canvas->_renderer->setChannel4Color(vvRenderer::VV_RED, r);
    _canvas->_renderer->setChannel4Color(vvRenderer::VV_GREEN, g);
    _canvas->_renderer->setChannel4Color(vvRenderer::VV_BLUE, b);
  }
  return 1;
}

long VVChannel4Dialog::onCancel(FXObject*, FXSelector, void*)
{
  if (_canvas->_renderer) 
  {
    _canvas->_renderer->setChannel4Color(vvRenderer::VV_RED, _rBak);
    _canvas->_renderer->setChannel4Color(vvRenderer::VV_GREEN, _gBak);
    _canvas->_renderer->setChannel4Color(vvRenderer::VV_BLUE, _bBak);
  }
  handle(this, FXSEL(SEL_COMMAND, ID_HIDE), NULL);
  return 1;
}

long VVChannel4Dialog::onOK(FXObject*, FXSelector, void*)
{
  handle(this, FXSEL(SEL_COMMAND, ID_HIDE), NULL);
  return 1;
}

void VVChannel4Dialog::updateValues()
{
  float r=1.0f,g=1.0f,b=1.0f;
  float h,s,v;

  if (_canvas->_renderer)
  {
    r  = _canvas->_renderer->getChannel4Color(vvRenderer::VV_RED);
    g  = _canvas->_renderer->getChannel4Color(vvRenderer::VV_GREEN);
    b  = _canvas->_renderer->getChannel4Color(vvRenderer::VV_BLUE);
  }
  vvToolshed::RGBtoHSB(r, g, b, &h, &s, &v);

  _hueLabel->setText(FXStringFormat("%.2f", h));
  _satLabel->setText(FXStringFormat("%.2f", s));

  _hueSlider->setValue(h);
  _satSlider->setValue(s);
}

void VVChannel4Dialog::show()
{
  if (_canvas->_renderer)
  {
    _rBak = _canvas->_renderer->getChannel4Color(vvRenderer::VV_RED);
    _gBak = _canvas->_renderer->getChannel4Color(vvRenderer::VV_GREEN);
    _bBak = _canvas->_renderer->getChannel4Color(vvRenderer::VV_BLUE);
  }
  updateValues();
  FXDialogBox::show();
}

/*******************************************************************************/

FXDEFMAP(VVOpacityDialog) VVOpacityDialogMap[]=
{
  FXMAPFUNC(SEL_COMMAND, VVOpacityDialog::ID_OK,     VVOpacityDialog::onOK),
  FXMAPFUNC(SEL_COMMAND, VVOpacityDialog::ID_CANCEL, VVOpacityDialog::onCancel),
  FXMAPFUNC(SEL_COMMAND, VVOpacityDialog::ID_ENABLE, VVOpacityDialog::onEnableChange),
  FXMAPFUNC(SEL_CHANGED, VVOpacityDialog::ID_RED,    VVOpacityDialog::onRedChange),
  FXMAPFUNC(SEL_CHANGED, VVOpacityDialog::ID_GREEN,  VVOpacityDialog::onGreenChange),
  FXMAPFUNC(SEL_CHANGED, VVOpacityDialog::ID_BLUE,   VVOpacityDialog::onBlueChange),
  FXMAPFUNC(SEL_CHANGED, VVOpacityDialog::ID_ALPHA,  VVOpacityDialog::onAlphaChange),
};

FXIMPLEMENT(VVOpacityDialog, FXDialogBox, VVOpacityDialogMap, ARRAYNUMBER(VVOpacityDialogMap))

// Construct a dialog box
VVOpacityDialog::VVOpacityDialog(FXWindow* owner, vvCanvas* c) : 
  FXDialogBox(owner, "Opacity Weights", DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE)
{
  _parent = (VVShell*)owner;
  _canvas = c;
  FXVerticalFrame* verticalFrame = new FXVerticalFrame(this, LAYOUT_FILL_X);

  _enableCheck = new FXCheckButton(verticalFrame, "Enable opacity weights", this, ID_ENABLE, ICON_BEFORE_TEXT);

  // Red:
  FXMatrix* redMatrix = new FXMatrix(verticalFrame, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(redMatrix, "Intensity of red:");
  _redLabel = new FXLabel(redMatrix, "", NULL, LAYOUT_RIGHT | LABEL_NORMAL);
  _redSlider = new FXRealSlider(verticalFrame, this, ID_RED, LAYOUT_FILL_X | SLIDER_HORIZONTAL | LAYOUT_FIX_WIDTH, 0, 0, 170);
  _redSlider->setRange(0,1);
  _redSlider->setValue(1);

  // Green:
  FXMatrix* greenMatrix = new FXMatrix(verticalFrame, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(greenMatrix, "Intensity of green:");
  _greenLabel = new FXLabel(greenMatrix, "");
  _greenSlider = new FXRealSlider(verticalFrame, this, ID_GREEN, LAYOUT_FILL_X | SLIDER_HORIZONTAL);
  _greenSlider->setRange(0,1);
  _greenSlider->setValue(1);

  // Blue:
  FXMatrix* blueMatrix = new FXMatrix(verticalFrame, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(blueMatrix, "Intensity of blue:");
  _blueLabel = new FXLabel(blueMatrix, "");
  _blueSlider = new FXRealSlider(verticalFrame, this, ID_BLUE, LAYOUT_FILL_X | SLIDER_HORIZONTAL);
  _blueSlider->setRange(0,1);
  _blueSlider->setValue(1);

  // Alpha:
  FXMatrix* alphaMatrix = new FXMatrix(verticalFrame, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(alphaMatrix, "Intensity of channel 4:");
  _alphaLabel = new FXLabel(alphaMatrix, "");
  _alphaSlider = new FXRealSlider(verticalFrame, this, ID_ALPHA, LAYOUT_FILL_X | SLIDER_HORIZONTAL);
  _alphaSlider->setRange(0,1);
  _alphaSlider->setValue(1);

  // Accept
  FXHorizontalFrame* buttonFrame = new FXHorizontalFrame(verticalFrame, LAYOUT_CENTER_X | LAYOUT_FILL_X | PACK_UNIFORM_WIDTH);
  new FXButton(buttonFrame, "Cancel", NULL, this, ID_CANCEL, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X | LAYOUT_CENTER_Y);
  new FXButton(buttonFrame, "OK", NULL, this, ID_OK, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X | LAYOUT_CENTER_Y);

  move(100, 100);
}

long VVOpacityDialog::onRedChange(FXObject*,FXSelector,void*)
{
  float f = (float)_redSlider->getValue();
  _redLabel->setText(FXStringFormat("%.2f", f));
  if (_canvas->_renderer) _canvas->_renderer->setOpacityWeight(vvRenderer::VV_RED, f);
  return 1;
}

long VVOpacityDialog::onGreenChange(FXObject*,FXSelector,void*)
{
  float f = (float)_greenSlider->getValue();
  _greenLabel->setText(FXStringFormat("%.2f", f));
  if (_canvas->_renderer) _canvas->_renderer->setOpacityWeight(vvRenderer::VV_GREEN, f);
  return 1;
}

long VVOpacityDialog::onBlueChange(FXObject*,FXSelector,void*)
{
  float f = (float)_blueSlider->getValue();
  _blueLabel->setText(FXStringFormat("%.2f", f));
  if (_canvas->_renderer) _canvas->_renderer->setOpacityWeight(vvRenderer::VV_BLUE, f);
  return 1;
}

long VVOpacityDialog::onAlphaChange(FXObject*,FXSelector,void*)
{
  float f = (float)_alphaSlider->getValue();
  _alphaLabel->setText(FXStringFormat("%.2f", f));
  if (_canvas->_renderer) _canvas->_renderer->setOpacityWeight(vvRenderer::VV_ALPHA, f);
  return 1;
}

long VVOpacityDialog::onEnableChange(FXObject*, FXSelector, void* ptr)
{
  _canvas->_renderer->setParameter(vvRenderState::VV_OPACITY_WEIGHTS, (ptr != NULL));
  if (ptr)
  {
    _redSlider->enable();
    _greenSlider->enable();
    _blueSlider->enable();
    _alphaSlider->enable();
  }
  else
  {
    _redSlider->disable();
    _greenSlider->disable();
    _blueSlider->disable();
    _alphaSlider->disable();
  }
  return 1;
}

long VVOpacityDialog::onCancel(FXObject*, FXSelector, void*)
{
  if (_canvas->_renderer) 
  {
    _canvas->_renderer->setOpacityWeight(vvRenderer::VV_RED,   _rBak);
    _canvas->_renderer->setOpacityWeight(vvRenderer::VV_GREEN, _gBak);
    _canvas->_renderer->setOpacityWeight(vvRenderer::VV_BLUE,  _bBak);
    _canvas->_renderer->setOpacityWeight(vvRenderer::VV_ALPHA, _aBak);
  }
  handle(this, FXSEL(SEL_COMMAND, ID_HIDE), NULL);
  return 1;
}

long VVOpacityDialog::onOK(FXObject*, FXSelector, void*)
{
  handle(this, FXSEL(SEL_COMMAND, ID_HIDE), NULL);
  return 1;
}

void VVOpacityDialog::updateValues()
{
  float r=1.0f,g=1.0f,b=1.0f,a=1.0f;

  if (_canvas->_renderer)
  {
    r  = _canvas->_renderer->getOpacityWeight(vvRenderer::VV_RED);
    g  = _canvas->_renderer->getOpacityWeight(vvRenderer::VV_GREEN);
    b  = _canvas->_renderer->getOpacityWeight(vvRenderer::VV_BLUE);
    a  = _canvas->_renderer->getOpacityWeight(vvRenderer::VV_ALPHA);

    bool enabled = _canvas->_renderer->getParameter(vvRenderState::VV_OPACITY_WEIGHTS);
    handle(this, FXSEL(SEL_COMMAND, ID_ENABLE), (void*)enabled);
  }

  _redLabel->setText(FXStringFormat("%.2f", r));
  _greenLabel->setText(FXStringFormat("%.2f", g));
  _blueLabel->setText(FXStringFormat("%.2f", b));
  _alphaLabel->setText(FXStringFormat("%.2f", a));

  _redSlider->setValue(r);
  _greenSlider->setValue(g);
  _blueSlider->setValue(b);
  _alphaSlider->setValue(a);
}

void VVOpacityDialog::show()
{
  if (_canvas->_renderer)
  {
    _rBak = _canvas->_renderer->getOpacityWeight(vvRenderer::VV_RED);
    _gBak = _canvas->_renderer->getOpacityWeight(vvRenderer::VV_GREEN);
    _bBak = _canvas->_renderer->getOpacityWeight(vvRenderer::VV_BLUE);
    _aBak = _canvas->_renderer->getOpacityWeight(vvRenderer::VV_ALPHA);
  }
  updateValues();
  FXDialogBox::show();
}

/*******************************************************************************/

FXDEFMAP(VVChannelDialog) VVChannelDialogMap[]=
{
  FXMAPFUNC(SEL_COMMAND, VVChannelDialog::ID_OK,     VVChannelDialog::onOK),
  FXMAPFUNC(SEL_COMMAND, VVChannelDialog::ID_CANCEL, VVChannelDialog::onCancel),
//  FXMAPFUNC(SEL_CHANGED, VVChannelDialog::ID_ALPHA,  VVChannelDialog::onAlphaChange),
};

FXIMPLEMENT(VVChannelDialog, FXDialogBox, VVChannelDialogMap, ARRAYNUMBER(VVChannelDialogMap))

// Construct a dialog box
VVChannelDialog::VVChannelDialog(FXWindow* owner, vvCanvas* c) : 
  FXDialogBox(owner, "Channel Settings", DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE)
{
  _parent = (VVShell*)owner;
  _canvas = c;
  _sliders = NULL;
  FXVerticalFrame* verticalFrame = new FXVerticalFrame(this, LAYOUT_FILL_X);
  _slidersFrame = new FXVerticalFrame(verticalFrame, LAYOUT_FILL_X);

/*
sliders for:

data value

gradient magnitude

variance

gravity: similar to gradient; but multiply all 6 neighbors with data value, then add all 6 values up: 
  center * (left + right + above + below + infront + behind)
  
radiation: (center^4 - left^4) + (center^4 - right^4) + (center^4 - above)... = 6 * center^4 - left^4 - right^4 - ...


sliders set contribution of each derived value on scale of 0-100
display contributions as numbers adding up to 1 or 100% somewhere else in window


add fast rendering mode (shader) for 5 channel data sets
*/

  
  FXHorizontalFrame* buttonFrame = new FXHorizontalFrame(verticalFrame, LAYOUT_CENTER_X | LAYOUT_FILL_X | PACK_UNIFORM_WIDTH);
  new FXButton(buttonFrame, "Cancel", NULL, this, ID_CANCEL, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X | LAYOUT_CENTER_Y);
  new FXButton(buttonFrame, "OK", NULL, this, ID_OK, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X | LAYOUT_CENTER_Y);

  move(100, 100);
}

long VVChannelDialog::onCancel(FXObject*, FXSelector, void*)
{
  if (_canvas->_renderer) 
  {
  }
  handle(this, FXSEL(SEL_COMMAND, ID_HIDE), NULL);
  return 1;
}

long VVChannelDialog::onOK(FXObject*, FXSelector, void*)
{
  handle(this, FXSEL(SEL_COMMAND, ID_HIDE), NULL);
  return 1;
}

void VVChannelDialog::updateValues()
{
  int i;
  if (_sliders)
  {
    for (i=0; i<_numSliders; ++i)
    {
      delete _sliders[i];
    }
    delete _sliders;
  }
  _numSliders = 0;
  if (_canvas->_vd)
  {
    _numSliders = _canvas->_vd->getChan();
    if (_numSliders>0)
    {
      _sliders = new FXRealSlider*[_numSliders];
      for (i=0; i<_numSliders; ++i)
      {
        _sliders[i] = new FXRealSlider(_slidersFrame, this, ID_SLIDERS_BASE + i, LAYOUT_FILL_X | SLIDER_HORIZONTAL);
        _sliders[i]->setRange(0,1);
        _sliders[i]->setValue(1);  
      }
    }
    else cerr << "Warning: VVChannelDialog::updateValues: number of channels is zero?" << endl;
  }
  layout();
  restore();
  create();
}

/*******************************************************************************/

FXDEFMAP(VVFloatRangeDialog) VVFloatRangeDialogMap[]=
{
  FXMAPFUNC(SEL_COMMAND, VVFloatRangeDialog::ID_MIN_DATA,    VVFloatRangeDialog::onMinData),
  FXMAPFUNC(SEL_COMMAND, VVFloatRangeDialog::ID_MAX_DATA,    VVFloatRangeDialog::onMaxData),
  FXMAPFUNC(SEL_COMMAND, VVFloatRangeDialog::ID_BOT_PERCENT, VVFloatRangeDialog::onBottomPercent),
  FXMAPFUNC(SEL_COMMAND, VVFloatRangeDialog::ID_TOP_PERCENT, VVFloatRangeDialog::onTopPercent),
  FXMAPFUNC(SEL_COMMAND, VVFloatRangeDialog::ID_FAST,        VVFloatRangeDialog::onFast),
  FXMAPFUNC(SEL_COMMAND, VVFloatRangeDialog::ID_HDR,         VVFloatRangeDialog::onHDRMapping),
  FXMAPFUNC(SEL_COMMAND, VVFloatRangeDialog::ID_APPLY,       VVFloatRangeDialog::onApply),
  FXMAPFUNC(SEL_COMMAND, VVFloatRangeDialog::ID_CLOSE,       VVFloatRangeDialog::onClose),
};

FXIMPLEMENT(VVFloatRangeDialog,FXDialogBox,VVFloatRangeDialogMap,ARRAYNUMBER(VVFloatRangeDialogMap))

VVFloatRangeDialog::VVFloatRangeDialog(FXWindow* owner, vvCanvas* c) : 
  FXDialogBox(owner, "Data Range Mapper", DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE, 100, 100), 
  _algoDataTarget(_algoType)
{
  _shell = (VVShell*)owner;
  _canvas = c;
  FXHorizontalFrame* horizontalFrame = new FXHorizontalFrame(this, LAYOUT_FILL_X | LAYOUT_FILL_Y);
  FXVerticalFrame* verticalFrame = new FXVerticalFrame(horizontalFrame, LAYOUT_FILL_X | LAYOUT_FILL_Y);

  // Min and max text fields:
  FXGroupBox* rangeGroup = new FXGroupBox(verticalFrame,"Range settings", FRAME_GROOVE | LAYOUT_FILL_X);
  FXMatrix* paramMatrix = new FXMatrix(rangeGroup, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);

  new FXLabel(paramMatrix, "Start value:");
  new FXLabel(paramMatrix, "End value:");

  _minTF = new FXTextField(paramMatrix, 20, NULL,0,TEXTFIELD_REAL | TEXTFIELD_NORMAL | LAYOUT_FILL_X);
  _maxTF = new FXTextField(paramMatrix, 20, NULL,0,TEXTFIELD_REAL | TEXTFIELD_NORMAL | LAYOUT_FILL_X);

  FXGroupBox* queryGroup = new FXGroupBox(verticalFrame,"Data query", FRAME_GROOVE | LAYOUT_FILL_X);
  FXMatrix* queryMatrix = new FXMatrix(queryGroup, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);

  new FXButton(queryMatrix, "Find min data value", NULL, this, ID_MIN_DATA, FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);
  new FXButton(queryMatrix, "Find max data value", NULL, this, ID_MAX_DATA, FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);

  FXHorizontalFrame* botClampFrame = new FXHorizontalFrame(queryMatrix, LAYOUT_FILL_X | LAYOUT_FILL_Y);
  _botClamp = new FXTextField(botClampFrame, 10, NULL,0,TEXTFIELD_REAL | TEXTFIELD_NORMAL | LAYOUT_FILL_X);
  _botClamp->setText(FXStringFormat("%.9g", 5.0f));
  new FXButton(botClampFrame, "% above min", NULL, this, ID_BOT_PERCENT, FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);

  FXHorizontalFrame* topClampFrame = new FXHorizontalFrame(queryMatrix, LAYOUT_FILL_X | LAYOUT_FILL_Y);
  _topClamp = new FXTextField(topClampFrame, 10, NULL,0,TEXTFIELD_REAL | TEXTFIELD_NORMAL | LAYOUT_FILL_X);
  _topClamp->setText(FXStringFormat("%.9g", 5.0f));
  new FXButton(topClampFrame, "% below max", NULL, this, ID_TOP_PERCENT, FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);

  // HDR mapping:
  _hdrCheck = new FXCheckButton(verticalFrame, "Distribution based data range mapping", this, ID_HDR, ICON_BEFORE_TEXT); 

  FXGroupBox* hdrGroup = new FXGroupBox(verticalFrame,"",FRAME_GROOVE | LAYOUT_FILL_X);
  FXMatrix* hdrMatrix = new FXMatrix(hdrGroup, 3, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);

  _isoRadio = new FXRadioButton(hdrMatrix, "Histogram Equalization", &_algoDataTarget, FXDataTarget::ID_OPTION+1, ICON_BEFORE_TEXT);
  _isoRadio->setTipText("Maps data values by binning them such that bins contain equal amounts of data values.");
  _weightRadio = new FXRadioButton(hdrMatrix, "Opacity-Weighted binning", &_algoDataTarget, FXDataTarget::ID_OPTION+2, ICON_BEFORE_TEXT);
  _weightRadio->setTipText("When checked algorithm creates smaller bins where opacity is higher.\nOtherwise binning is independent from opacity.");
  new FXLabel(hdrMatrix, "");
  _algoType = 1; // set first radio button to true, others to false

  _skipCheck = new FXCheckButton(hdrMatrix, "Cull skipped regions", this, ID_SKIP, ICON_BEFORE_TEXT);
  _skipCheck->setCheck(true);
  _skipCheck->setTipText("Remove data values in areas covered by Skip widgets.");
  new FXLabel(hdrMatrix, "");
  new FXLabel(hdrMatrix, "");

  _dupCheck = new FXCheckButton(hdrMatrix, "Cull duplicate values", this, ID_DUP, ICON_BEFORE_TEXT);
  _dupCheck->setCheck(true);
  _dupCheck->setTipText("Remove duplicate data values to make better use of the bins.");
  new FXLabel(hdrMatrix, "");
  new FXLabel(hdrMatrix, "");

  _lockCheck = new FXCheckButton(hdrMatrix, "Trim to range", this, ID_LOCK, ICON_BEFORE_TEXT);
  _lockCheck->setCheck(true);
  _lockCheck->setTipText("Don't adjust min and max range settings when running HDR routines.");
  new FXLabel(hdrMatrix, "");
  new FXLabel(hdrMatrix, "");

  _opacityCheck = new FXCheckButton(hdrMatrix, "Transform opacity widgets", this, ID_OPACITY, ICON_BEFORE_TEXT);
  _opacityCheck->setCheck(false);
  _opacityCheck->setTipText("Transform opacity widgets to bin space along with colors.");
  new FXLabel(hdrMatrix, "");
  new FXLabel(hdrMatrix, "");

  _fastCheck = new FXCheckButton(hdrMatrix, "Fast sampling", this, ID_FAST, ICON_BEFORE_TEXT);
  _fastCheck->setCheck(false);
  new FXLabel(hdrMatrix, "Number of samples:");
  _fastNumber = new FXTextField(hdrMatrix, 20, NULL, 0, TEXTFIELD_NORMAL | LAYOUT_FILL_X);
  _fastNumber->setText("10000");
  _fastNumber->setTipText("Uses a reduced number of data values for HDR transfer function specification. Default: 10,000");

  // Buttons:
  FXVerticalFrame* buttonFrame = new FXVerticalFrame(horizontalFrame, LAYOUT_FILL_X | LAYOUT_FIX_WIDTH,0,0,80);
  new FXButton(buttonFrame, "Apply", NULL, this, ID_APPLY, FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);
  new FXButton(buttonFrame, "Close", NULL, this, ID_CLOSE, FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);
  
  _prevRange[0] = FXFloatVal(_minTF->getText());
  _prevRange[1] = FXFloatVal(_maxTF->getText());
  
  handle(this, FXSEL(SEL_COMMAND, ID_HDR), NULL);
}

long VVFloatRangeDialog::onClose(FXObject*,FXSelector,void*)
{
  handle(this, FXSEL(SEL_COMMAND, ID_HIDE), NULL);
  return 1;
}

long VVFloatRangeDialog::onApply(FXObject*,FXSelector,void*)
{
  if (_canvas->_renderer)
  {
    if(_shell->_glcanvas->makeCurrent())
    {
      _canvas->_vd->range(0)[0] = FXFloatVal(_minTF->getText());
      _canvas->_vd->range(0)[1] = FXFloatVal(_maxTF->getText());
      if (_prevRange[0] != _canvas->_vd->range(0)[0] ||
          _prevRange[1] != _canvas->_vd->range(0)[1])
      {
        _shell->_transWindow->setDirtyHistogram();
        _prevRange[0] = _canvas->_vd->range(0)[0];
        _prevRange[1] = _canvas->_vd->range(0)[1];
      }
      
      _canvas->_renderer->setParameter(vvRenderer::VV_BINNING, (_hdrCheck->getCheck()) ? _algoType : 0);
      if (_hdrCheck->getCheck())
      {
        vvVolDesc::BinningType bt=vvVolDesc::LINEAR;
        switch (_algoType)
        {
          case 1: bt = vvVolDesc::ISO_DATA; break;
          case 2: bt = vvVolDesc::OPACITY; break;
          default: assert(0); break;
        }
        _canvas->_vd->updateHDRBins((_fastCheck->getCheck()) ? FXIntVal(_fastNumber->getText()) : -1, 
          (_skipCheck->getCheck()) ? true : false, (_dupCheck->getCheck()) ? true : false, 
          (_lockCheck->getCheck()) ? true : false, bt, (_opacityCheck->getCheck()) ? true : false);
      }
      _canvas->_renderer->updateVolumeData();
      _canvas->_renderer->updateTransferFunction();
      _shell->_glcanvas->makeNonCurrent();
      updateDependents();
      updateValues();
    }
  }
  return 1;
}

void VVFloatRangeDialog::updateDependents()
{
  _shell->_transWindow->updateValues();
  _shell->_volumeDialog->updateValues();
}

long VVFloatRangeDialog::onMinData(FXObject*,FXSelector,void*)
{
  float fMin, fMax;
  _canvas->_vd->findMinMax(0, fMin, fMax);
  _minTF->setText(FXStringFormat("%.9g", fMin));
  return 1;
}

long VVFloatRangeDialog::onMaxData(FXObject*,FXSelector,void*)
{
  float fMin, fMax;
  _canvas->_vd->findMinMax(0, fMin, fMax);
  _maxTF->setText(FXStringFormat("%.9g", fMax));
  return 1;
}

long VVFloatRangeDialog::onBottomPercent(FXObject*,FXSelector,void*)
{
  float threshold = FXFloatVal(_botClamp->getText());
  float clampVal = _canvas->_vd->findClampValue(0, 0, threshold / 100.0f);
  _minTF->setText(FXStringFormat("%.9g", clampVal));
  return 1;
}

long VVFloatRangeDialog::onTopPercent(FXObject*,FXSelector,void*)
{
  float threshold = FXFloatVal(_topClamp->getText());
  float clampVal = _canvas->_vd->findClampValue(0, 0, 1.0f - threshold / 100.0f);
  _maxTF->setText(FXStringFormat("%.9g", clampVal));
  return 1;
}

long VVFloatRangeDialog::onFast(FXObject*,FXSelector,void*)
{
  if (_fastCheck->getCheck()) _fastNumber->enable();
  else _fastNumber->disable();
  return 1;
}

long VVFloatRangeDialog::onHDRMapping(FXObject*,FXSelector,void*)
{
  if (_hdrCheck->getCheck())
  {
    _isoRadio->enable();
    _weightRadio->enable();
    _fastCheck->enable();
    if (_fastCheck->getCheck()) _fastNumber->enable();
    _skipCheck->enable();
    _dupCheck->enable();
    _lockCheck->enable();
    _opacityCheck->enable();
  }
  else
  {
    _isoRadio->disable();
    _weightRadio->disable();
    _fastCheck->disable();
    _fastNumber->disable();
    _skipCheck->disable();
    _dupCheck->disable();
    _lockCheck->disable();
    _opacityCheck->disable();
  }
  return 1;
}

void VVFloatRangeDialog::updateValues()
{
  if (_canvas->_renderer)
  {
    _minTF->setText(FXStringFormat("%.9g", _canvas->_vd->range(0)[0]));
    _maxTF->setText(FXStringFormat("%.9g", _canvas->_vd->range(0)[1]));
  }
}

void VVFloatRangeDialog::show()
{
  updateValues();
  FXDialogBox::show();
}

/*******************************************************************************/

FXDEFMAP(VVDataTypeDialog) VVDataTypeDialogMap[]=
{
  FXMAPFUNC(SEL_COMMAND, VVDataTypeDialog::ID_SWAP_ENDIAN, VVDataTypeDialog::onSwapEndian),
  FXMAPFUNC(SEL_COMMAND, VVDataTypeDialog::ID_SWAP_CHANNELS, VVDataTypeDialog::onSwapChannels),
  FXMAPFUNC(SEL_COMMAND, VVDataTypeDialog::ID_DEL_CHANNEL, VVDataTypeDialog::onDeleteChannel),
  FXMAPFUNC(SEL_COMMAND, VVDataTypeDialog::ID_ADD_CHANNEL, VVDataTypeDialog::onAddChannel),
  FXMAPFUNC(SEL_COMMAND, VVDataTypeDialog::ID_ADD_GRADIENT_MAGNITUDE, VVDataTypeDialog::onAddGradMag),
  FXMAPFUNC(SEL_COMMAND, VVDataTypeDialog::ID_ADD_GRADIENT_VECTOR, VVDataTypeDialog::onAddGradVec),
  FXMAPFUNC(SEL_COMMAND, VVDataTypeDialog::ID_ADD_VARIANCE, VVDataTypeDialog::onAddVariance),
  FXMAPFUNC(SEL_COMMAND, VVDataTypeDialog::ID_CNV_8, VVDataTypeDialog::onBytesPerChannel),
  FXMAPFUNC(SEL_COMMAND, VVDataTypeDialog::ID_CNV_16, VVDataTypeDialog::onBytesPerChannel),
  FXMAPFUNC(SEL_COMMAND, VVDataTypeDialog::ID_CNV_FLOAT, VVDataTypeDialog::onBytesPerChannel),
};

FXIMPLEMENT(VVDataTypeDialog, FXDialogBox, VVDataTypeDialogMap, ARRAYNUMBER(VVDataTypeDialogMap))

// Construct a dialog box
VVDataTypeDialog::VVDataTypeDialog(FXWindow* owner, vvCanvas* c) : 
  FXDialogBox(owner, "Data Format", DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE, 100, 100)
{
  _shell = (VVShell*)owner;
  _canvas = c;
  FXVerticalFrame* master = new FXVerticalFrame(this, LAYOUT_FILL_X | LAYOUT_FILL_Y);

  FXGroupBox* swapGroup = new FXGroupBox(master,"Swap Channels", FRAME_GROOVE | LAYOUT_FILL_X);
  FXHorizontalFrame* swapFrame = new FXHorizontalFrame(swapGroup, LAYOUT_FILL_X | PACK_UNIFORM_WIDTH);
  _channel1Combo = new FXComboBox(swapFrame,3,this, ID_CHANNEL_1, COMBOBOX_INSERT_LAST | COMBOBOX_STATIC | FRAME_SUNKEN | FRAME_THICK);
  _channel2Combo = new FXComboBox(swapFrame,3,this, ID_CHANNEL_2, COMBOBOX_INSERT_LAST | COMBOBOX_STATIC | FRAME_SUNKEN | FRAME_THICK);
  _swapChannelsButton = new FXButton(swapFrame, "Swap", NULL, this, ID_SWAP_CHANNELS, FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);

  FXGroupBox* addDeleteGroup = new FXGroupBox(master,"Add/Delete Channels", FRAME_GROOVE | LAYOUT_FILL_X);
  FXHorizontalFrame* channelFrame = new FXHorizontalFrame(addDeleteGroup, LAYOUT_FILL_X | PACK_UNIFORM_WIDTH);
  new FXLabel(channelFrame, "Selected channel:");
  _channelCombo = new FXComboBox(channelFrame,3,this,ID_CHANNEL, COMBOBOX_INSERT_LAST | COMBOBOX_STATIC | FRAME_SUNKEN | FRAME_THICK);
  FXHorizontalFrame* addButtonsFrame = new FXHorizontalFrame(addDeleteGroup, LAYOUT_FILL_X | PACK_UNIFORM_WIDTH);
  _addGradMagButton  = new FXButton(addButtonsFrame, "Add Gradient Magnitude", NULL, this, ID_ADD_GRADIENT_MAGNITUDE, FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);
  _addGradVecButton  = new FXButton(addButtonsFrame, "Add Gradient Vector", NULL, this, ID_ADD_GRADIENT_VECTOR, FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);
  _addVarianceButton = new FXButton(addButtonsFrame, "Add Variance", NULL, this, ID_ADD_VARIANCE, FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);
  _addChannelButton  = new FXButton(addButtonsFrame, "Add Empty Channel", NULL, this, ID_ADD_CHANNEL, FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);
  _delChannelButton  = new FXButton(addDeleteGroup,  "Delete Channel", NULL, this, ID_DEL_CHANNEL, FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);

  FXGroupBox* bpcGroup = new FXGroupBox(master,"Convert Bytes Per Channel", FRAME_GROOVE | LAYOUT_FILL_X);
  FXHorizontalFrame* bpcFrame = new FXHorizontalFrame(bpcGroup, LAYOUT_FILL_X | PACK_UNIFORM_WIDTH);
  _bpc8Button = new FXButton(bpcFrame, "8 Bit", NULL, this, ID_CNV_8, FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);
  _bpc16Button = new FXButton(bpcFrame, "16 Bit", NULL, this, ID_CNV_16, FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);
  _bpcFloatButton = new FXButton(bpcFrame, "Floating Point", NULL, this, ID_CNV_FLOAT, FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);

  _swapEndianButton = new FXButton(master, "Swap Endianness", NULL, this, ID_SWAP_ENDIAN, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X | LAYOUT_CENTER_Y);

  FXHorizontalFrame* buttonFrame = new FXHorizontalFrame(master, LAYOUT_CENTER_X | LAYOUT_FILL_X | PACK_UNIFORM_WIDTH);
  new FXButton(buttonFrame, "Close", NULL, this, ID_ACCEPT, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X | LAYOUT_CENTER_Y);
}

long VVDataTypeDialog::onSwapEndian(FXObject*, FXSelector, void*)
{
  _canvas->_vd->toggleEndianness();
  _shell->updateRendererVolume();
  _shell->drawScene();
  _shell->_transWindow->setDirtyHistogram();
  return 1;
}

long VVDataTypeDialog::onSwapChannels(FXObject*, FXSelector, void*)
{
  int chan1, chan2;

  chan1 = _channel1Combo->getCurrentItem();
  chan2 = _channel2Combo->getCurrentItem();
  if (chan1==chan2) return 1;   // this was easy!
  _canvas->_vd->swapChannels(chan1, chan2);
  _shell->_transWindow->setDirtyHistogram();
  _shell->updateRendererVolume();
  _shell->drawScene();
  return 1;
}

long VVDataTypeDialog::onDeleteChannel(FXObject*, FXSelector, void*)
{
  _canvas->_vd->deleteChannel(_channelCombo->getCurrentItem());
  _shell->_transWindow->setDirtyHistogram();
  _shell->updateRendererVolume();
  _shell->setCanvasRenderer();
  _shell->drawScene();
  updateValues();
  updateDialogs();
  return 1;
}

long VVDataTypeDialog::onAddChannel(FXObject*, FXSelector, void*)
{
  _canvas->_vd->convertChannels(_canvas->_vd->getChan() + 1);
  _shell->_transWindow->setDirtyHistogram();
  _shell->updateRendererVolume();
  _shell->setCanvasRenderer();
  _shell->drawScene();
  updateValues();
  updateDialogs();
  return 1;
}

long VVDataTypeDialog::onAddGradMag(FXObject*, FXSelector, void*)
{
  _canvas->_vd->addGradient(_channelCombo->getCurrentItem(), vvVolDesc::GRADIENT_MAGNITUDE);
  _shell->_transWindow->setDirtyHistogram();
  _shell->updateRendererVolume();
  _shell->setCanvasRenderer();
  _shell->drawScene();
  updateValues();
  updateDialogs();
  return 1;
}

long VVDataTypeDialog::onAddGradVec(FXObject*, FXSelector, void*)
{
  _canvas->_vd->addGradient(_channelCombo->getCurrentItem(), vvVolDesc::GRADIENT_VECTOR);
  _shell->_transWindow->setDirtyHistogram();
  _shell->updateRendererVolume();
  _shell->setCanvasRenderer();
  _shell->drawScene();
  updateValues();
  updateDialogs();
  return 1;
}

long VVDataTypeDialog::onAddVariance(FXObject*, FXSelector, void*)
{
  _canvas->_vd->addVariance(_channelCombo->getCurrentItem());
  _shell->_transWindow->setDirtyHistogram();
  _shell->updateRendererVolume();
  _shell->setCanvasRenderer();
  _shell->drawScene();
  updateValues();
  updateDialogs();
  return 1;
}

long VVDataTypeDialog::onBytesPerChannel(FXObject* obj, FXSelector, void*)
{
  int bpc;

  if (obj==_bpc8Button) bpc = 1;
  else if (obj==_bpc16Button) bpc = 2;
  else bpc = 4;

  _canvas->_vd->convertBPC(bpc);
  _shell->_transWindow->setDirtyHistogram();
  _shell->updateRendererVolume();
  _shell->setCanvasRenderer();
  _shell->drawScene();
  updateDialogs();
  return 1;
}

/** Update values in this dialog window.
*/
void VVDataTypeDialog::updateValues()
{
  _channel1Combo->clearItems();
  _channel2Combo->clearItems();
  _channelCombo->clearItems();
  _channel1Combo->setNumVisible(_canvas->_vd->getChan());
  _channel2Combo->setNumVisible(_canvas->_vd->getChan());
  _channelCombo->setNumVisible(_canvas->_vd->getChan());
  for (int i=0; i<_canvas->_vd->getChan(); ++i)
  {
    _channel1Combo->appendItem(FXStringFormat("%d", i+1));
    _channel2Combo->appendItem(FXStringFormat("%d", i+1));
    _channelCombo->appendItem(FXStringFormat("%d", i+1));
  }
  if (_canvas->_vd->getChan()==1) _swapChannelsButton->disable();
  else _swapChannelsButton->enable();
  if (_canvas->_vd->getChan()==1) _delChannelButton->disable();
  else _delChannelButton->enable();
  if (_canvas->_vd->bpc==1) _swapEndianButton->disable();
  else _swapEndianButton->enable();
}

/** Update other dialogs in app.
*/
void VVDataTypeDialog::updateDialogs()
{
  _shell->_volumeDialog->updateValues();
}


/*******************************************************************************/

FXDEFMAP(VVEditVoxelsDialog) VVEditVoxelsDialogMap[]=
{
  FXMAPFUNC(SEL_COMMAND, VVEditVoxelsDialog::ID_SHIFT, VVEditVoxelsDialog::onShiftVoxels),
  FXMAPFUNC(SEL_COMMAND, VVEditVoxelsDialog::ID_RESIZE, VVEditVoxelsDialog::onResize),
  FXMAPFUNC(SEL_COMMAND, VVEditVoxelsDialog::ID_FLIP_X, VVEditVoxelsDialog::onFlipX),
  FXMAPFUNC(SEL_COMMAND, VVEditVoxelsDialog::ID_FLIP_Y, VVEditVoxelsDialog::onFlipY),
  FXMAPFUNC(SEL_COMMAND, VVEditVoxelsDialog::ID_FLIP_Z, VVEditVoxelsDialog::onFlipZ),
  FXMAPFUNC(SEL_COMMAND, VVEditVoxelsDialog::ID_ORDER, VVEditVoxelsDialog::onInvertOrder),
  FXMAPFUNC(SEL_COMMAND, VVEditVoxelsDialog::ID_CROP, VVEditVoxelsDialog::onCrop),
};

FXIMPLEMENT(VVEditVoxelsDialog, FXDialogBox, VVEditVoxelsDialogMap, ARRAYNUMBER(VVEditVoxelsDialogMap))

// Construct a dialog box
VVEditVoxelsDialog::VVEditVoxelsDialog(FXWindow* owner, vvCanvas* c) : 
  FXDialogBox(owner, "Edit Geometry", DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE, 100, 100)
{
  _shell = (VVShell*)owner;
  _canvas = c;
  FXVerticalFrame* master = new FXVerticalFrame(this, LAYOUT_FILL_X | LAYOUT_FILL_Y);

  FXGroupBox* shiftGroup = new FXGroupBox(master,"Shift Voxels", FRAME_GROOVE | LAYOUT_FILL_X);
  FXHorizontalFrame* shiftFrame = new FXHorizontalFrame(shiftGroup, LAYOUT_CENTER_X | LAYOUT_FILL_X | PACK_UNIFORM_WIDTH);
  new FXLabel(shiftFrame, "X=");
  _shiftXField = new FXTextField(shiftFrame, 5, NULL, 0, TEXTFIELD_NORMAL);
  _shiftXField->setText("0");
  new FXLabel(shiftFrame, "Y=");
  _shiftYField = new FXTextField(shiftFrame, 5, NULL, 0, TEXTFIELD_NORMAL);
  _shiftYField->setText("0");
  new FXLabel(shiftFrame, "Z=");
  _shiftZField = new FXTextField(shiftFrame, 5, NULL, 0, TEXTFIELD_NORMAL);
  _shiftZField->setText("0");
  new FXButton(shiftFrame, "Shift", NULL, this, ID_SHIFT, FRAME_RAISED | FRAME_THICK | LAYOUT_RIGHT,0,0,0,0,20,20);

  FXGroupBox* resizeGroup = new FXGroupBox(master,"Resize (Resample)", FRAME_GROOVE | LAYOUT_FILL_X);
  FXHorizontalFrame* resizeFrame = new FXHorizontalFrame(resizeGroup, LAYOUT_CENTER_X | LAYOUT_FILL_X | PACK_UNIFORM_WIDTH);
  new FXLabel(resizeFrame, "X=");
  _resizeXField = new FXTextField(resizeFrame, 5, NULL, 0, TEXTFIELD_NORMAL);
  new FXLabel(resizeFrame, "Y=");
  _resizeYField = new FXTextField(resizeFrame, 5, NULL, 0, TEXTFIELD_NORMAL);
  new FXLabel(resizeFrame, "Z=");
  _resizeZField = new FXTextField(resizeFrame, 5, NULL, 0, TEXTFIELD_NORMAL);
  new FXButton(resizeFrame, "Resize", NULL, this, ID_RESIZE, FRAME_RAISED | FRAME_THICK | LAYOUT_RIGHT,0,0,0,0,20,20);
  _interpolType = new FXCheckButton(resizeGroup,"Trilinear interpolation",this,ID_INTERPOL,ICON_BEFORE_TEXT | LAYOUT_CENTER_X);

  FXGroupBox* cropGroup = new FXGroupBox(master,"Crop", FRAME_GROOVE | LAYOUT_FILL_X);
  FXHorizontalFrame* cropFrame = new FXHorizontalFrame(cropGroup, LAYOUT_CENTER_Y | LAYOUT_FILL_X);
  FXMatrix* cropMatrix = new FXMatrix(cropFrame,6, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(cropMatrix, "X=");
  _cropXField = new FXTextField(cropMatrix, 5, NULL, 0, TEXTFIELD_NORMAL);
  _cropXField->setText("0");
  new FXLabel(cropMatrix, "Y=");
  _cropYField = new FXTextField(cropMatrix, 5, NULL, 0, TEXTFIELD_NORMAL);
  _cropYField->setText("0");
  new FXLabel(cropMatrix, "Z=");
  _cropZField = new FXTextField(cropMatrix, 5, NULL, 0, TEXTFIELD_NORMAL);
  _cropZField->setText("0");
  new FXLabel(cropMatrix, "Width=");
  _cropWField = new FXTextField(cropMatrix, 5, NULL, 0, TEXTFIELD_NORMAL);
  new FXLabel(cropMatrix, "Height=");
  _cropHField = new FXTextField(cropMatrix, 5, NULL, 0, TEXTFIELD_NORMAL);
  new FXLabel(cropMatrix, "Slices=");
  _cropSField = new FXTextField(cropMatrix, 5, NULL, 0, TEXTFIELD_NORMAL);
  new FXButton(cropFrame, "Crop", NULL, this, ID_CROP, FRAME_RAISED | FRAME_THICK | LAYOUT_RIGHT,0,0,0,0,20,20);

  FXGroupBox* flipGroup = new FXGroupBox(master,"Flip", FRAME_GROOVE | LAYOUT_FILL_X);
  FXHorizontalFrame* flipFrame = new FXHorizontalFrame(flipGroup, LAYOUT_FILL_X | PACK_UNIFORM_WIDTH);
  new FXButton(flipFrame, "X Axis", NULL, this, ID_FLIP_X, FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);
  new FXButton(flipFrame, "Y Axis", NULL, this, ID_FLIP_Y, FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);
  new FXButton(flipFrame, "Z Axis", NULL, this, ID_FLIP_Z, FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);

  FXGroupBox* orderGroup = new FXGroupBox(master,"Voxel Order", FRAME_GROOVE | LAYOUT_FILL_X);
  new FXButton(orderGroup, "Invert voxel order: swap x and z loops", NULL, this, ID_ORDER, FRAME_RAISED | FRAME_THICK | LAYOUT_FILL_X);

  FXHorizontalFrame* buttonFrame = new FXHorizontalFrame(master, LAYOUT_CENTER_X | LAYOUT_FILL_X | PACK_UNIFORM_WIDTH);
  new FXButton(buttonFrame, "Close", NULL, this, ID_ACCEPT, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X | LAYOUT_CENTER_Y,0,0,0,0,20,20);
}

long VVEditVoxelsDialog::onShiftVoxels(FXObject*, FXSelector, void*)
{
  _canvas->_vd->shift(FXIntVal(_shiftXField->getText()),
                      FXIntVal(_shiftYField->getText()),
                      FXIntVal(_shiftZField->getText()));
  _shell->updateRendererVolume();
  _shell->drawScene();
  return 1;
}

long VVEditVoxelsDialog::onResize(FXObject*, FXSelector, void*)
{
  vvVolDesc::InterpolationType itype;
  itype = (_interpolType->getCheck()) ? vvVolDesc::TRILINEAR : vvVolDesc::NEAREST;
  _canvas->_vd->resize(FXIntVal(_resizeXField->getText()),
                       FXIntVal(_resizeYField->getText()),
                       FXIntVal(_resizeZField->getText()), itype);
  _shell->updateRendererVolume();
  _shell->drawScene();
  _shell->_transWindow->setDirtyHistogram();
  updateDialogs();
  return 1;
}

long VVEditVoxelsDialog::onFlipX(FXObject*, FXSelector, void*)
{
  _canvas->_vd->flip(virvo::cartesian_axis<3>::X);
  _shell->updateRendererVolume();
  _shell->drawScene();
  return 1;
}

long VVEditVoxelsDialog::onFlipY(FXObject*, FXSelector, void*)
{
  _canvas->_vd->flip(virvo::cartesian_axis<3>::Y);
  _shell->updateRendererVolume();
  _shell->drawScene();
  return 1;
}

long VVEditVoxelsDialog::onInvertOrder(FXObject*, FXSelector, void*)
{
  _canvas->_vd->convertVoxelOrder();
  _shell->updateRendererVolume();
  _shell->drawScene();
  return 1;
}

long VVEditVoxelsDialog::onFlipZ(FXObject*, FXSelector, void*)
{
  _canvas->_vd->flip(virvo::cartesian_axis<3>::Z);
  _shell->updateRendererVolume();
  _shell->drawScene();
  return 1;
}

long VVEditVoxelsDialog::onCrop(FXObject*, FXSelector, void*)
{
  _canvas->_vd->crop(FXIntVal(_cropXField->getText()),
                     FXIntVal(_cropYField->getText()),
                     FXIntVal(_cropZField->getText()),
                     FXIntVal(_cropWField->getText()),
                     FXIntVal(_cropHField->getText()),
                     FXIntVal(_cropSField->getText()));
  _shell->updateRendererVolume();
  _shell->drawScene();
  _shell->_transWindow->setDirtyHistogram();
  updateDialogs();
  updateValues();
  return 1;
}

/** Update values in this dialog window.
*/
void VVEditVoxelsDialog::updateValues()
{
  _resizeXField->setText(FXStringFormat("%" VV_PRIdSIZE, _canvas->_vd->vox[0]));
  _resizeYField->setText(FXStringFormat("%" VV_PRIdSIZE, _canvas->_vd->vox[1]));
  _resizeZField->setText(FXStringFormat("%" VV_PRIdSIZE, _canvas->_vd->vox[2]));
  _cropXField->setText("0");
  _cropYField->setText("0");
  _cropZField->setText("0");
  _cropWField->setText(FXStringFormat("%" VV_PRIdSIZE, _canvas->_vd->vox[0]));
  _cropHField->setText(FXStringFormat("%" VV_PRIdSIZE, _canvas->_vd->vox[1]));
  _cropSField->setText(FXStringFormat("%" VV_PRIdSIZE, _canvas->_vd->vox[2]));
}

/** Update other dialogs in app.
*/
void VVEditVoxelsDialog::updateDialogs()
{
  std::string str;
  _canvas->_vd->makeInfoString(&str);
  _shell->_statusBar->setText(str.c_str());
  _shell->_volumeDialog->updateValues();
  _shell->_transWindow->setDirtyHistogram();
}



/*******************************************************************************/

FXDEFMAP(VVHeightFieldDialog) VVHeightFieldDialogMap[]=
{
  FXMAPFUNC(SEL_COMMAND, VVHeightFieldDialog::ID_OK, VVHeightFieldDialog::onOK),
};

FXIMPLEMENT(VVHeightFieldDialog, FXDialogBox, VVHeightFieldDialogMap, ARRAYNUMBER(VVHeightFieldDialogMap))

// Construct a dialog box
VVHeightFieldDialog::VVHeightFieldDialog(FXWindow* owner, vvCanvas* c) : 
  FXDialogBox(owner, "Make Height Field", DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE, 100, 100)
{
  _shell = (VVShell*)owner;
  _canvas = c;
  FXHorizontalFrame* horizFrame = new FXHorizontalFrame(this, LAYOUT_FILL_X | LAYOUT_FILL_Y);

  FXVerticalFrame* verticalFrame = new FXVerticalFrame(horizFrame, LAYOUT_FILL_X | LAYOUT_FILL_Y);
  FXMatrix* mat = new FXMatrix(verticalFrame,2, MATRIX_BY_COLUMNS);

  new FXLabel(mat, "Number of slices in destination volume:");
  _slicesTF = new FXTextField(mat, 5, NULL, 0, TEXTFIELD_NORMAL);
  _slicesTF->setText("128");

  new FXLabel(mat, "Fill mode for voxels under surface:");
  _modeCombo = new FXComboBox(mat, 1, this, ID_MODE, COMBOBOX_INSERT_LAST | COMBOBOX_STATIC | FRAME_SUNKEN | FRAME_THICK | LAYOUT_FILL_X);
  _modeCombo->appendItem("Empty (minimum voxel value)");
  _modeCombo->appendItem("Same value as surface");
  _modeCombo->setNumVisible(_modeCombo->getNumItems());

  FXVerticalFrame* buttonFrame = new FXVerticalFrame(horizFrame, LAYOUT_CENTER_X | LAYOUT_FILL_X | PACK_UNIFORM_WIDTH);
  new FXButton(buttonFrame, "OK", NULL, this, ID_OK, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X | LAYOUT_CENTER_Y,0,0,0,0,20,20);
  new FXButton(buttonFrame, "Close", NULL, this, ID_CANCEL, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X | LAYOUT_CENTER_Y);
}

long VVHeightFieldDialog::onOK(FXObject*, FXSelector, void*)
{
  if (_canvas->_vd->getChan() != 1 || _canvas->_vd->vox[2] != 1)
  {
    FXMessageBox::error((FXWindow*)this, MBOX_OK, "Error", "Height field requires a single slice, single channel data set.");
  }
  else
  {
    _canvas->_vd->makeHeightField(FXIntVal(_slicesTF->getText()), _modeCombo->getCurrentItem());
    _shell->updateRendererVolume();
    _shell->drawScene();
  }
  handle(this, FXSEL(SEL_COMMAND, ID_HIDE), NULL);
  return 1;
}

/** Update other dialogs in app.
*/
void VVHeightFieldDialog::updateDialogs()
{
  _shell->_volumeDialog->updateValues();
  _shell->_editVoxelsDialog->updateValues();
}

void VVHeightFieldDialog::updateValues()
{
}

/*******************************************************************************/

FXIMPLEMENT(VVGLSettingsDialog, FXDialogBox, NULL, 0)

/// Settings dialog thanks to Sander Jansen <sxj@cfdrc.com>
VVGLSettingsDialog::VVGLSettingsDialog(FXWindow* owner) : 
  FXDialogBox(owner,"OpenGL Settings", DECOR_TITLE | DECOR_BORDER, 100, 100)
{
  _shell = (VVShell*)owner;
  
  FXVerticalFrame * master=new FXVerticalFrame(this,LAYOUT_FILL_X|LAYOUT_FILL_Y,0,0,0,0, 0,0,0,0);

  //gl version information
  FXGroupBox* versionbox=new FXGroupBox(master,"OpenGL Driver Information",GROUPBOX_NORMAL|FRAME_RIDGE|LAYOUT_FILL_X);
  FXMatrix* v_matrix=new FXMatrix(versionbox,2,MATRIX_BY_COLUMNS,0,0,0,0,0,0,0,0);

  new FXLabel(v_matrix,"Vendor: ",NULL,LABEL_NORMAL);
  new FXLabel(v_matrix,FXStringFormat("%s",glGetString(GL_VENDOR)),NULL,LABEL_NORMAL);

  new FXLabel(v_matrix,"Renderer: ",NULL,LABEL_NORMAL);
  new FXLabel(v_matrix,FXStringFormat("%s",glGetString(GL_RENDERER)),NULL,LABEL_NORMAL);

  new FXLabel(v_matrix,"GL Version: ",NULL,LABEL_NORMAL);
  new FXLabel(v_matrix,FXStringFormat("%s",glGetString(GL_VERSION)),NULL,LABEL_NORMAL);

  new FXLabel(v_matrix,"GLU Version: ",NULL,LABEL_NORMAL);
  new FXLabel(v_matrix,FXStringFormat("%s",gluGetString(GLU_VERSION)),NULL,LABEL_NORMAL);

  FXHorizontalFrame *options=new FXHorizontalFrame(master,LAYOUT_SIDE_TOP|FRAME_NONE|LAYOUT_FILL_X|LAYOUT_FILL_Y,0,0,0,0,0,0,0,0);

  //Display Mode
  FXGroupBox *dmbox=new FXGroupBox(options,"Display Mode",GROUPBOX_NORMAL|FRAME_RIDGE|LAYOUT_FILL_Y);
  FXMatrix *mat=new FXMatrix(dmbox,2,MATRIX_BY_COLUMNS);

  new FXLabel(mat,"Acceleration",NULL,LABEL_NORMAL);
  if(_shell->_glvisual->isAccelerated())
    new FXLabel(mat,"enabled",NULL,LABEL_NORMAL);
  else
    new FXLabel(mat,"disabled",NULL,LABEL_NORMAL);

  new FXLabel(mat,"Double Buffering",NULL,LABEL_NORMAL);
  if(_shell->_glvisual->isDoubleBuffer())
    new FXLabel(mat,"enabled",NULL,LABEL_NORMAL);
  else
    new FXLabel(mat,"disabled",NULL,LABEL_NORMAL);

  new FXLabel(mat,"Stereo",NULL,LABEL_NORMAL);
  if(_shell->_glvisual->isStereo())
    new FXLabel(mat,"enabled",NULL,LABEL_NORMAL);
  else
    new FXLabel(mat,"disabled",NULL,LABEL_NORMAL);

  new FXLabel(mat,"Color Depth",NULL,LABEL_NORMAL);
  new FXLabel(mat,FXStringFormat("%d",_shell->_glvisual->getDepth()),NULL,LABEL_NORMAL);

  new FXLabel(mat,"Depth Buffer Size",NULL,LABEL_NORMAL);
  new FXLabel(mat,FXStringFormat("%d",_shell->_glvisual->getActualDepthSize()),NULL,LABEL_NORMAL);

  new FXLabel(mat,"Stencil Buffer Size",NULL,LABEL_NORMAL);
  new FXLabel(mat,FXStringFormat("%d",_shell->_glvisual->getActualStencilSize()),NULL,LABEL_NORMAL);

  new FXLabel(mat,"RGBA",NULL,LABEL_NORMAL);
  new FXLabel(mat,FXStringFormat("%d-%d-%d-%d",_shell->_glvisual->getActualRedSize(),_shell->_glvisual->getActualGreenSize(),_shell->_glvisual->getActualBlueSize(),_shell->_glvisual->getActualAlphaSize()),NULL,LABEL_NORMAL);

  new FXLabel(mat,"Accum RGBA",NULL,LABEL_NORMAL);
  new FXLabel(mat,FXStringFormat("%d-%d-%d-%d",_shell->_glvisual->getActualAccumRedSize(),_shell->_glvisual->getActualAccumGreenSize(),_shell->_glvisual->getActualAccumBlueSize(),_shell->_glvisual->getActualAccumAlphaSize()),NULL,LABEL_NORMAL);

  FXGroupBox* extGroup= new FXGroupBox(options,"Available GL/GLU Extensions",GROUPBOX_NORMAL|FRAME_RIDGE|LAYOUT_FILL_Y|LAYOUT_FILL_X);

  FXVerticalFrame *listframe=new FXVerticalFrame(extGroup,LAYOUT_FILL_X|LAYOUT_FILL_Y|FRAME_SUNKEN|FRAME_THICK,0,0,0,0,0,0,0,0,0,0);

  FXList* pExtList = new FXList(listframe,NULL,0,FRAME_SUNKEN|FRAME_THICK|LAYOUT_FILL_X|LAYOUT_FILL_Y);

  char* token;
  char* text;
  char* tmp;

  // Get GL extensions
  tmp=(char*)glGetString(GL_EXTENSIONS);
  if(tmp)
  {
    text=strdup(tmp);
    token=strtok(text," ");
    while(token!=NULL)
    {
      pExtList->appendItem(FXStringFormat("(GL) %s",token));
      token=strtok(NULL," ");
    }
    free(text);
  }

  // Get GLU extensions
#if defined(GLU_VERSION_1_1)
  tmp=(char*)gluGetString(GLU_EXTENSIONS);
  if(tmp)
  {
    text=strdup(tmp);
    token=strtok(text," ");
    while(token!=NULL)
    {
      pExtList->appendItem(FXStringFormat("(GLU) %s",token));
      token=strtok(NULL," ");
    }
    free(text);
  }
#endif

  // Contents
  FXHorizontalFrame * control=new FXHorizontalFrame(master,LAYOUT_SIDE_TOP|FRAME_NONE|LAYOUT_FILL_X);

  // Accept
  new FXButton(control,"OK",NULL,this,ID_ACCEPT,FRAME_RAISED|FRAME_THICK|LAYOUT_CENTER_X|LAYOUT_CENTER_Y,0,0,0,0, 20,20,3,3);
}

/*******************************************************************************/
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
