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

#ifndef VV_CANVAS_H
#define VV_CANVAS_H

#include <stdlib.h>

// Virvo:
#include <vvrenderer.h>
#include <vvtexrend.h>

// Local:
#include "vvobjview.h"
#include "vvartracking.h"

namespace vox
{

/** Output window related routines.
  This class provides the routines which are needed to display
  OpenGL generated volume data in a FOX Toolkit canvas.
  @author Juergen P. Schulze (schulze@hlrs.de)
*/
class vvCanvas
{
  public:
    enum StereoType   // stereo mode
    {
      MONO,
      SIDE_BY_SIDE,
      RED_BLUE,
      RED_GREEN,
      ACTIVE          ///< time sequential
    };
  private:
    int  _buttonState;                            ///< status of mouse buttons
    int  _curX, _curY;                            ///< current mouse coordinates
    int  _lastX, _lastY;                          ///< previous mouse coordinates
    bool _doubleBuffering;                        ///< true = double buffering on
    StereoType _stereoMode;                       ///< stereo mode
    bool _swapEyes;                               ///< true=swap eyes in stereo mode
    bool _perspectiveMode;                        ///< true=perspective projection, false=parallel projection
    bool _artoolkit;                              ///< true=artoolkit tracking on
    int  _width, _height;                         ///< window viewport dimensions
    float _bgColor[3];                            ///< background color as R,G,B components [0..1]
    vvMatrix _lastRotation;                       ///< last trackball rotation, used for spin animation
#ifdef VV_USE_ARTOOLKIT
    vvARTracking* _arTracking;                    ///< ARToolkit tracking
#endif

  public:
    static const float DEFAULT_OBJ_SIZE;          ///< default volume size, relative to window width [0..1]
    enum ButtonStateType                          /// binary coded button states
    { 
      NO_BUTTON = 0, 
      LEFT_BUTTON = 1, 
      MIDDLE_BUTTON = 2, 
      RIGHT_BUTTON = 4 
    };
    vvRenderer* _renderer;                        ///< current rendering engine
    vvVolDesc*  _vd;                              ///< currently used volume description
    vvRenderer::RendererType _currentAlgorithm;   ///< rendering algorithm/renderer type
    vox::vvObjView _ov;                           ///< object view instance for current volume viewing parameters

    vvCanvas();
    virtual ~vvCanvas();
    void draw();
    void initCanvas();
    void mouseDragged(int, int);
    void mousePressed(int, int, int);
    void mouseReleased(int, int, int);
    void repeatMouseDrag();
    void resize(int, int);
    void transformObject(const vvMatrix&);
    void setDoubleBuffering(bool);
    bool getDoubleBuffering();
    void setStereoMode(StereoType);
    StereoType getStereoMode();
    void setPerspectiveMode(bool);
    void setProjectionMode(bool, float, float, float);
    bool getPerspectiveMode();
    void resetObjectView();
    void setRenderer(vvRenderer::RendererType = vvRenderer::INVALID);
    int getRenderer() const;
    void setBackgroundColor(float, float, float);
    void getBackgroundColor(float&, float&, float&);
    void getCanvasSize(int&, int&);
    bool getARToolkit();
    void setARToolkit(bool);
    void artTimerEvent();
    void renderImage(uchar*, int, int);
    void setSwapEyes(bool);
    bool getSwapEyes();
};
}

#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
