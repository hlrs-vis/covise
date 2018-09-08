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

#ifndef VV_SHELL_H
#define VV_SHELL_H

#include <vvplatform.h>

// OS:
#include <iostream>
#include <string.h>

// Virvo:
#include <vvtexrend.h>
#include <vvvoldesc.h>

// Local:
#include "vvcanvas.h"
#include "vvprefwindow.h"
#include "vvmovie.h"

/* workaround: include last - introduces a weird powf macro into global namespace */
#include <fx.h>
#include <fx3d.h>


class VVVolumeDialog;
class VVTransferWindow;
class VVSliceViewer;
class VVCameraSetDialog;
class VVGammaDialog;
class VVChannel4Dialog;
class VVOpacityDialog;
class VVChannelDialog;
class VVFloatRangeDialog;
class VVClippingDialog;
class VVROIDialog;
class VVDimensionDialog;
class VVDrawDialog;
class VVMergeDialog;
class VVServerDialog;
class VVScreenshotDialog;
class VVMovieDialog;
class VVTimeStepDialog;
class VVDiagramDialog;
class VVDataTypeDialog;
class VVEditVoxelsDialog;
class VVHeightFieldDialog;
class VVGLSettingsDialog;
class vvSpaceTraveler;

/** \mainpage DeskVOX
  <DL>
    <DT><B>Functionality</B>        <DD>VOX for the Desktop. VOX stands for VOlume eXploration. This is the
                                    desktop version of a volume visualization software that also exists
                                    for virtual environments like the CAVE. 
    <DT><B>Developer Information<B> <DD>VVShell is the main class which contains the main() routine. 
                                    A Visual C++ 6.0 project file is in msvc. DeskVOX uses the FOX Toolkit 
                                    for its GUI. The Virvo library is required for volume rendering. It
                                    should come with the DeskVOX package.
                                    HAVE_CG should be defined at compile time if Nvidia Cg pixel shader
                                    support is desired. Make sure that Virvo is compiled with HAVE_CG 
                                    defined as well.
    <DT><B>Main Author</B>          <DD> J&uuml;rgen P. Schulze
    <DT><B>Email</B>                <DD>jschulze@ucsd.edu
    <DT><B>Institution</B>          <DD>University of California, San Diego
  </DL>
*/

//******************************************************//

// Event Handler Object
class VVShell : public FXMainWindow
{
  FXDECLARE(VVShell)

  protected:
    int     spinning;         // Is box spinning
    double  angle;            // Rotation angle of box
    bool    lmdFlag,mmdFlag,rmdFlag;
    bool    initFileMenu(FXMenuPane* filemenu);
    bool    initSettingsMenu(FXMenuPane* setmenu);
    bool    initEditMenu(FXMenuPane* editmenu);
    bool    initViewMenu(FXMenuPane* viewmenu);
    bool    initHelpMenu(FXMenuPane* helpmenu);
    void    initDialogs();

    VVShell(){}

  public:
    FXGLVisual*           _glvisual;        // OpenGL visual
    VVVolumeDialog*       _volumeDialog;
    VVPreferenceWindow*   _prefWindow;
    VVTransferWindow*     _transWindow;
    VVSliceViewer*        _sliceViewer;
    VVCameraSetDialog*    _cameraDialog;
    VVGammaDialog*        _gammaDialog;
    VVChannel4Dialog*     _channel4Dialog;
    VVOpacityDialog*      _opacityDialog;
    VVChannelDialog*      _channelDialog;
    VVFloatRangeDialog*   _floatRangeDialog;
    VVClippingDialog*     _clipDialog;
    VVROIDialog*          _roiDialog;
    FXColorDialog*        _colorPicker;
    VVDimensionDialog*    _dimDialog;
    VVDrawDialog*         _drawDialog;
    VVMergeDialog*        _mergeDialog;
    VVServerDialog*       _serverDialog;
    VVScreenshotDialog*   _screenshotDialog;
    VVMovieDialog*        _movieDialog;
    VVTimeStepDialog*     _tsDialog;
    VVDiagramDialog*      _diagramDialog;
    VVDataTypeDialog*     _dataTypeDialog;
    VVEditVoxelsDialog*   _editVoxelsDialog;
    VVHeightFieldDialog*  _heightFieldDialog;
    VVGLSettingsDialog*   _glSettingsDialog;
    FXMenuCheck* _orientItem;
    FXMenuCheck* _boundaryItem;
    FXMenuCheck* _paletteItem;
    FXMenuCheck* _qualityItem;
    FXMenuCheck* _fpsItem;
    FXMenuCheck* _spinItem;
    vox::vvCanvas* _canvas;
    vox::vvMovie* _movie;
    FXGLCanvas* _glcanvas;                         ///< GL Canvas to draw into
    FXLabel* _statusBar;
    vvSpaceTraveler* _traveler;

    enum        /// We define additional ID's, starting from the last one used by the base class+1.
    {           /// This way, we know the ID's are all unique for this particular target.
      ID_CANVAS=FXMainWindow::ID_LAST,
      ID_LOAD_VOLUME,
      ID_MERGE,
      ID_RELOAD_VOLUME,
      ID_SAVE_VOLUME,
      ID_LOAD_CAMERA,
      ID_SAVE_CAMERA,
      ID_SCREEN_SHOT,
      ID_MOVIE_SCRIPT,
      ID_NETWORK,
      ID_PREFERENCES,
      ID_TRANSFER,
      ID_CAMERA,
      ID_GAMMA,
      ID_CHANNEL4,
      ID_OPACITY,
      ID_CHANNELS,
      ID_FLOAT_RANGE,
      ID_CLIP_PLANE,
      ID_ROI,
      ID_BG_COLOR,
      ID_DIMENSIONS,
      ID_DRAW,
      ID_VIS_INFO,
      ID_SLICE_VIEWER,
      ID_TIME_STEPS,
      ID_ABOUT,
      ID_COLOR_PICKER,
      ID_DIAGRAMS,
      ID_ART_TIMER,
      ID_ANIM_TIMER,
      ID_SPIN_TIMER,
      ID_DATA_TYPE,
      ID_EDIT_VOXELS,
      ID_KEYS,
      ID_ORIENTATION,
      ID_BOUNDARIES,
      ID_QUALITY,
      ID_PALETTE,
      ID_FPS,
      ID_SPIN,
      ID_HEIGHT_FIELD,
      ID_GL_SETTINGS,
      ID_SERVER,
      ID_IDLE,
      ID_LAST
    };

    // Message handlers
    long onExpose(FXObject*,FXSelector,void*);
    long onConfigure(FXObject*,FXSelector,void*);

    long onLeftMouseDown(FXObject*,FXSelector,void*);
    long onLeftMouseUp(FXObject*,FXSelector,void*);
    long onMidMouseDown(FXObject*,FXSelector,void*);
    long onMidMouseUp(FXObject*,FXSelector,void*);
    long onRightMouseDown(FXObject*,FXSelector,void*);
    long onRightMouseUp(FXObject*,FXSelector,void*);
    long onMouseMove(FXObject*,FXSelector,void*);
    long onMouseWheel(FXObject*,FXSelector,void*);
    long onKeyPress(FXObject*,FXSelector,void*);
    long onKeyRelease(FXObject*,FXSelector,void*);

    long onCmdPrefs(FXObject*,FXSelector,void*);
    long onCmdTrans(FXObject*,FXSelector,void*);
    long onCmdVisInfo(FXObject*,FXSelector,void*);
    long onCmdAbout(FXObject*,FXSelector,void*);
    long onCmdKeys(FXObject*,FXSelector,void*);
    long onCmdCameraSettings(FXObject*,FXSelector,void*);
    long onCmdGammaSettings(FXObject*,FXSelector,void*);
    long onCmdChannel4Settings(FXObject*,FXSelector,void*);
    long onCmdOpacitySettings(FXObject*,FXSelector,void*);
    long onCmdChannelSettings(FXObject*,FXSelector,void*);
    long onCmdFloatRange(FXObject*,FXSelector,void*);
    long onCmdClipping(FXObject*,FXSelector,void*);
    long onCmdROI(FXObject*,FXSelector,void*);
    long onCmdBGColor(FXObject*,FXSelector,void*);
    long pickerColorChanged(FXObject*,FXSelector,void*);
    long onCmdDimensions(FXObject*,FXSelector,void*);
    long onCmdDraw(FXObject*,FXSelector,void*);
    long onCmdScreenShot(FXObject*,FXSelector,void*);
    long onCmdMovie(FXObject*,FXSelector,void*);
    long onCmdTimeSteps(FXObject*,FXSelector,void*);
    long onCmdDiagrams(FXObject*,FXSelector,void*);
    long onCmdDataType(FXObject*,FXSelector,void*);
    long onCmdEditVoxels(FXObject*,FXSelector,void*);
    long onCmdMakeHeightField(FXObject*,FXSelector,void*);
    long onCmdGLSettings(FXObject*,FXSelector,void*);
    long onCmdSliceViewer(FXObject*,FXSelector,void*);
    long onCmdLoadVolume(FXObject*,FXSelector,void*);
    long onCmdSaveVolume(FXObject*,FXSelector,void*);
    long onCmdReloadVolume(FXObject*,FXSelector,void*);
    long onCmdMerge(FXObject*,FXSelector,void*);
    long onCmdServerRequest(FXObject*,FXSelector,void*);
    long onCmdLoadCamera(FXObject*,FXSelector,void*);
    long onCmdSaveCamera(FXObject*,FXSelector,void*);
    long onDispOrientChange(FXObject*,FXSelector,void*);
    long onDispBoundsChange(FXObject*,FXSelector,void*);
    long onDispPaletteChange(FXObject*,FXSelector,void*);
    long onDispQualityChange(FXObject*,FXSelector,void*);
    long onDispFPSChange(FXObject*,FXSelector,void*);
    long onDispSpinChange(FXObject*,FXSelector,void*);
    long onIdle(FXObject*, FXSelector, void*);

    void setCanvasRenderer(vvVolDesc* = NULL, vvRenderer::RendererType = vvRenderer::INVALID);
    void updateRendererVolume();

    void loadDefaultVolume(int, int, int, int);
    void mergeFiles(const char*, int, int, vvVolDesc::MergeType);
    void loadVolumeFile(const char*);
    void parseCommandline(std::string&, int&, int&);
    FXString getOpenFilename(const FXString&, const FXString&);
    FXString getSaveFilename(const FXString&, const FXString&, const FXString&);
    FXString getOpenDirectory(const FXString&);

    void takeScreenshot(const char*, int, int);
    void toggleBounds();
    void toggleOrientation();
    void toggleFPS();
    void toggleSpin();
    void togglePalette();
    void toggleQualityDisplay();
    void benchmarkTest();

    long onAllUpdate(FXObject*,FXSelector,void*);

    void startARToolkitTimer();
    void stopARToolkitTimer();
    long onARToolkitTimerEvent(FXObject*,FXSelector,void*);

    void startAnimTimer();
    void stopAnimTimer();
    long onAnimTimerEvent(FXObject*,FXSelector,void*);

    void startSpinTimer();
    void stopSpinTimer();
    long onSpinTimerEvent(FXObject*,FXSelector,void*);

    // VVShell constructor
    VVShell(FXApp*);

    // Initialize
    void create();

    // Draw scene
    void drawScene();

    // VVShell destructor
    virtual ~VVShell();
  private:
    VVShell(const VVShell&);
};

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
