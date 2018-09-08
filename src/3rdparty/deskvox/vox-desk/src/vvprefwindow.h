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

#ifndef VV_PREFWINDOW_H
#define VV_PREFWINDOW_H

#include <vvplatform.h>
#include <iostream>
#include <string.h>

// Local:
#include "vvcanvas.h"

/* workaround: include last - introduces a weird powf macro into global namespace */
#include <fx.h>

class VVShell;

class VVPreferenceWindow : public FXDialogBox
{
  FXDECLARE(VVPreferenceWindow)

    protected:
    VVPreferenceWindow(){}

  public:
    enum
    {
      ID_EYE_DIST=FXDialogBox::ID_LAST,
      ID_QUALITY_MOVING,
      ID_QUALITY_STILL,
      ID_QUALITY_M_TEXT,
      ID_QUALITY_S_TEXT,
      ID_STEREO,
      ID_ARTOOLKIT,
      ID_RENDERERTYPE,
      ID_GEOMTYPE,
      ID_PIXEL_SHADER,
      ID_BRICK_SIZE,
      ID_OPTIONS,
      ID_LINTERP,
      ID_POST_CLASS,
      ID_HEADLIGHT,
      ID_SHOWBRICKS,
      ID_COMPUTE_BRICKSIZE,
      ID_TEX_MEMORY,
      ID_MIP,
      ID_PREINT,
      ID_OP_CORR,
      ID_DEF_VOL,
      ID_SUPPRESS,
      ID_SWAP_EYES,
      ID_LAST
    };

    vox::vvCanvas* _canvas;
    VVShell*  _shell;
    FXSlider* _eyeSlider;
    FXDial* _qualityMDial;
    FXDial* _qualitySDial;
    FXTextField* _eyeTField;
    FXTextField* _qualityMTField;
    FXTextField* _qualitySTField;
    FXComboBox* _rtCombo;                         ///< renderer type
    FXComboBox* _gtCombo;                         ///< geometry type
    FXComboBox* _psCombo;                         ///< pixel shader
    FXComboBox* _bsCombo;                         ///< brick size
    FXComboBox* _defCombo;                        ///< default data set
    FXComboBox* _stereoCombo;
    FXCheckButton* _linterpButton;
    FXCheckButton* _postClassButton;              ///< post-classificaton [on|off]
    FXCheckButton* _headlightButton;              ///< turn gradient-based headlight [on|off]
    FXCheckButton* _showBricksButton;
    FXCheckButton* _computeBrickSizeButton;
    FXCheckButton* _mipButton;
    FXCheckButton* _suppressButton;
    FXCheckButton* _artoolkitButton;
    FXCheckButton* _swapEyesButton;
    FXTextField* _texMemoryField;

    VVPreferenceWindow(FXWindow*, vox::vvCanvas*);
    virtual ~VVPreferenceWindow();
    void updateValues();
    void toggleMIP();
    void toggleBounds();
    void toggleInterpol();
    void toggleClassification();
    void toggleHeadlight();
    void scaleQuality(float);
    float getQualitySDialValue();
    void  setQualitySDialValue(float);
    float getQualityMDialValue();
    void  setQualityMDialValue(float);

    long onEyeChange(FXObject*,FXSelector,void*);
    long onQualityMChange(FXObject*,FXSelector,void*);
    long onQualityMTextChange(FXObject*,FXSelector,void*);
    long onQualityMChanging(FXObject*,FXSelector,void*);
    long onQualitySChange(FXObject*,FXSelector,void*);
    long onQualitySTextChange(FXObject*,FXSelector,void*);
    long onQualitySChanging(FXObject*,FXSelector,void*);
    long onStereoChange(FXObject*,FXSelector,void*);
    long onARToolkitChange(FXObject*,FXSelector,void*);
    long onRTChange(FXObject*,FXSelector,void*);
    long onGTChange(FXObject*,FXSelector,void*);
    long onPSChange(FXObject*,FXSelector,void*);
    long onBSChange(FXObject*, FXSelector, void*);
    long onDispBoundsChange(FXObject*,FXSelector,void*);
    long onMIPSelect(FXObject*,FXSelector,void*);
    long onInterpolationSelect(FXObject*,FXSelector,void*);
    long onPostClassSelect(FXObject*,FXSelector,void*);
    long onHeadlightSelect(FXObject*,FXSelector,void*);
    long onShowBricksSelect(FXObject*, FXSelector, void*);
    long onComputeBricksizeSelect(FXObject*, FXSelector, void*);
    long onTexMemoryChange(FXObject*, FXSelector, void*);
    void setBSCombo(int);
    long onDefaultVolume(FXObject*,FXSelector,void*);
    long onSuppressRendering(FXObject*,FXSelector,void*);
    long onSwapEyes(FXObject*,FXSelector,void*);
    int  getRenderer() const;
  private:
    VVPreferenceWindow(const VVPreferenceWindow&);
};
#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
