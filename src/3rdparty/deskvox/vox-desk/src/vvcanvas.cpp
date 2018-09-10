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

#include <vvplatform.h>
#include <vvopengl.h>

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#include <assert.h>

// Virvo:
#include <vvdebugmsg.h>
#include <vvtoolshed.h>
#include <vvtexrend.h>
#include <vvclock.h>
#include <vvfileio.h>
#include <vvstingray.h>
#include <vvsoftsw.h>
#include <vvimageclient.h>
#include <vvibrclient.h>
#ifdef HAVE_VOLPACK
#include <vvrendervp.h>
#endif

#include <virvo/private/vvgltools.h>

// Local:
#include "vvcanvas.h"

using namespace std;

//============================================================================
// Definitions for class vvCanvas
//============================================================================

using namespace vox;

const float vvCanvas::DEFAULT_OBJ_SIZE   = 0.6f;

//----------------------------------------------------------------------------
/** Constructor
 */
vvCanvas::vvCanvas()
{
  vvDebugMsg::msg(1, "vvCanvas::vvCanvas()");

  _renderer = NULL;
  _buttonState = NO_BUTTON;
  _curX      = 0;
  _curY      = 0;
  _lastX     = 0;
  _lastY     = 0;
  _doubleBuffering = false;
  _stereoMode = MONO;
  _swapEyes = false;
  _perspectiveMode = true;
  _bgColor[0] = _bgColor[1] = _bgColor[2] = 0.0f;   // set background to black by default
  _artoolkit = false;
  _width = _height = 0;
  _vd = NULL;
  _currentAlgorithm = vvRenderer::TEXREND;

#ifdef VV_USE_ARTOOLKIT
  _arTracking = NULL;
#endif  
  vvDebugMsg::msg(1, "vvCanvas constructor done.");
}

//----------------------------------------------------------------------------
/// Destructor
vvCanvas::~vvCanvas()
{
}

//----------------------------------------------------------------------------
/// Method called when OpenGL canvas is initialized
void vvCanvas::initCanvas()
{
  vvFileIO* fio;

  vvDebugMsg::msg(1, "vvCanvas::initCanvas()");

  glClearColor(0.0, 0.0, 0.0, 0.0);
  glDrawBuffer(GL_BACK);
  glEnable(GL_DEPTH_TEST);

  vvGLTools::checkOpenGLextensions();

  setPerspectiveMode(true);
  setDoubleBuffering(true);
  resetObjectView();
  _vd = new vvVolDesc();
  _vd->vox[0] = 32;
  _vd->vox[1] = 32;
  _vd->vox[2] = 32;
  _vd->frames = 0;
  fio = new vvFileIO();
  fio->loadVolumeData(_vd, vvFileIO::ALL_DATA);    // load default volume
  delete fio;
  if (_vd->tf[0].isEmpty())
  {
    _vd->tf[0].setDefaultAlpha(0, _vd->range(0)[0], _vd->range(0)[1]);
    _vd->tf[0].setDefaultColors((_vd->getChan()==1) ? 0 : 3, _vd->range(0)[0], _vd->range(0)[1]);
  }

 #ifdef USE_STINGRAY
   setRenderer(vvRenderer::STINGRAY);
 #else
   setRenderer(vvRenderer::TEXREND);
 #endif

  vvGLTools::printGLError("vvCanvas::initCanvas");// check for errors during GL initialization

  getCanvasSize(_width, _height);

  vvDebugMsg::msg(1, "vvCanvas::initCanvas() done.");
}

//----------------------------------------------------------------------------
void vvCanvas::getCanvasSize(int& w, int& h)
{
  GLint viewPort[4];                                // x, y, width, height of viewport
  glGetIntegerv(GL_VIEWPORT, viewPort);
  w = viewPort[2];
  h = viewPort[3];
}

//----------------------------------------------------------------------------
/** Resize the OpenGL canvas.
  @param w,h  new canvas size in pixels
*/
void vvCanvas::resize(int w, int h)
{
  vvDebugMsg::msg(1, "vvCanvas::resize()");

  _width  = w;
  _height = h;

  // Resize OpenGL viewport:
  glViewport(0, 0, (GLint)_width, (GLint)_height);

  // Set new aspect ratio:
  if (_height>0) _ov.setAspectRatio((float)_width/(float)_height);

  // Clear all OpenGL buffers:
  glDrawBuffer(GL_FRONT_AND_BACK);                // select all buffers
  glClearColor(0.8f, 0.8f, 0.8f, 0.0f);           // set clear color to frame color
                                                  // clear frame buffer
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  vvDebugMsg::msg(1, "New window dimensions: ", _width, _height);
}

//----------------------------------------------------------------------------
/// Draw the contents of the OpenGL canvas
void vvCanvas::draw()
{
  vvMatrix mv;                                   // modelview matrix
  int eye;

  vvDebugMsg::msg(3, "vvCanvas::draw()");
  if (_renderer==NULL) return;

  // Clear background:
  if (_doubleBuffering==true) glDrawBuffer(GL_BACK);
  else glDrawBuffer(GL_FRONT);
                                                  // set clear color
  glClearColor(_bgColor[0], _bgColor[1], _bgColor[2], 0.0f);
                                                  // clear buffers
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Set projection matrix:
  _ov.setProjectionMatrix();

  // Draw volume:
  glMatrixMode(GL_MODELVIEW);
  if (_stereoMode == MONO) 
  {
    if (_doubleBuffering) glDrawBuffer(GL_BACK);
    else glDrawBuffer(GL_FRONT);
    _ov.setModelviewMatrix(vvObjView::CENTER);
    _renderer->renderVolumeGL();
  }
  else if (_stereoMode==ACTIVE) // active stereo?
  {
    // Draw left image:
    if (_doubleBuffering) glDrawBuffer((_swapEyes) ? GL_BACK_LEFT : GL_BACK_RIGHT);
    else glDrawBuffer((_swapEyes) ? GL_FRONT_LEFT : GL_FRONT_RIGHT);
    _ov.setModelviewMatrix((_swapEyes) ? vvObjView::LEFT_EYE : vvObjView::RIGHT_EYE);
    _renderer->renderVolumeGL();

    // Draw right image:
    if (_doubleBuffering) glDrawBuffer((_swapEyes) ? GL_BACK_RIGHT : GL_BACK_LEFT);
    else glDrawBuffer((_swapEyes) ? GL_FRONT_RIGHT : GL_FRONT_LEFT);
    _ov.setModelviewMatrix((_swapEyes) ? vvObjView::RIGHT_EYE : vvObjView::LEFT_EYE);
    _renderer->renderVolumeGL();
  }
  else if (_stereoMode==SIDE_BY_SIDE)   // passive stereo?
  {
    _ov.setAspectRatio((float)_width / 2.0f / (float)_height);
    for (eye=0; eye<2; ++eye)
    {
      // Specify eye to draw:
      if (eye==0) _ov.setModelviewMatrix(vvObjView::LEFT_EYE);
      else        _ov.setModelviewMatrix(vvObjView::RIGHT_EYE);

      // Specify where to draw it:
      if ((!_swapEyes && eye==0) || (_swapEyes && eye==1))
      {
        glViewport(_width / 2, 0, _width / 2, _height); // right
      }
      else
      {
        glViewport(0, 0, _width / 2, _height);    // left
      }
      _renderer->renderVolumeGL();
    }
    
    // Reset viewport and aspect ratio:
    glViewport(0, 0, _width, _height);
    _ov.setAspectRatio((float)_width / (float)_height);
  }
  else if (_stereoMode==RED_BLUE || _stereoMode==RED_GREEN)
  {
    // Backup color mask:
    GLboolean colorMask[4];
    glGetBooleanv(GL_COLOR_WRITEMASK, colorMask);

    for (eye=0; eye<2; ++eye)
    {
      // Specify eye to draw:
      if (eye==0) _ov.setModelviewMatrix(vvObjView::LEFT_EYE);
      else        _ov.setModelviewMatrix(vvObjView::RIGHT_EYE);

      // Specify where to draw it:
      if ((!_swapEyes && eye==0) || (_swapEyes && eye==1))
      {
        if (_stereoMode==RED_BLUE)
        {
          glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_FALSE);
        }
        else
        {
          glColorMask(GL_FALSE, GL_TRUE, GL_FALSE, GL_FALSE);
        }
      }
      else
      {
        glColorMask(GL_FALSE, GL_FALSE, GL_TRUE, GL_FALSE);
      }
      _renderer->renderVolumeGL();
    }

    // Restore color mask:
    glColorMask(colorMask[0], colorMask[1], colorMask[2], colorMask[3]);
  }
  vvDebugMsg::msg(3, "vvCanvas::draw() done");
}

//----------------------------------------------------------------------------
/// Setter for doubleBuffering
void vvCanvas::setDoubleBuffering(bool newmode)
{
  vvDebugMsg::msg(1, "vvCanvas::setDoubleBuffering()");
  _doubleBuffering = newmode;
  cerr << "Double buffering ";
  if (_doubleBuffering) cerr << "on." << endl;
  else cerr << "off." << endl;
}

//----------------------------------------------------------------------------
/// Getter for doubleBuffering
bool vvCanvas::getDoubleBuffering()
{
  vvDebugMsg::msg(1, "vvCanvas::getDoubleBuffering()");
  return _doubleBuffering;
}

//----------------------------------------------------------------------------
/// Setter for stereoMode
void vvCanvas::setStereoMode(StereoType newMode)
{
  vvDebugMsg::msg(1, "vvCanvas::setStereoMode()", newMode);
  _stereoMode = newMode;
}

//----------------------------------------------------------------------------
/// Getter for stereoMode
vvCanvas::StereoType vvCanvas::getStereoMode()
{
  vvDebugMsg::msg(1, "vvCanvas::getStereoMode()");
  return _stereoMode;
}

//----------------------------------------------------------------------------
/// Setter for perspectiveMode
void vvCanvas::setPerspectiveMode(bool newMode)
{
  vvDebugMsg::msg(1, "vvCanvas::setPerspectiveMode(1)", newMode);
  _perspectiveMode = newMode;
  if (_perspectiveMode)
  {
    _ov.setProjection(vvObjView::PERSPECTIVE, vvObjView::DEF_FOV, vvObjView::DEF_CLIP_NEAR, vvObjView::DEF_CLIP_FAR);
  }
  else
  {
    _ov.setProjection(vvObjView::ORTHO, vvObjView::DEF_VIEWPORT_WIDTH, vvObjView::DEF_CLIP_NEAR, vvObjView::DEF_CLIP_FAR);
  }
}

//----------------------------------------------------------------------------
void vvCanvas::setProjectionMode(bool newMode, float range, float n, float f)
{
  vvDebugMsg::msg(1, "vvCanvas::setPerspectiveMode(2)", newMode);
  _perspectiveMode = newMode;
  _ov.setProjection(_perspectiveMode ? vvObjView::PERSPECTIVE : vvObjView::ORTHO, range, n, f);
}

//----------------------------------------------------------------------------
/// Getter for perspectiveMode
bool vvCanvas::getPerspectiveMode()
{
  vvDebugMsg::msg(1, "vvCanvas::getPerspectiveMode()");
  return _perspectiveMode;
}

//----------------------------------------------------------------------------
/** Method called when a mouse button was pressed.
 @param x/y  current mouse position
 @param bs   button status
*/
void vvCanvas::mousePressed(int x, int y, int bs)
{
  vvDebugMsg::msg(3, "vvCanvas::mousePressed() at ", x, y);

  _buttonState = bs;
  _lastX = _curX = x;
  _lastY = _curY = y;
  _lastRotation.identity();
}

//----------------------------------------------------------------------------
/** Method called when a mouse button was released.
 @param x/y  current mouse position
 @param bs   button status
*/
void vvCanvas::mouseReleased(int x, int y, int bs)
{
  vvDebugMsg::msg(3, "vvCanvas::mouseReleased() at ", x, y);
  _lastX = _curX;
  _lastY = _curY;
  _curX  = x;
  _curY  = y;
  _buttonState = bs;
}

//----------------------------------------------------------------------------
/* Repeats the last trackball rotation, used to keep object spinning.
*/
void vvCanvas::repeatMouseDrag()
{
  vvDebugMsg::msg(3, "vvCanvas::repeatMouseDrag()");
  _ov._camera.multiplyRight(_lastRotation);
}

//----------------------------------------------------------------------------
/** Method called when mouse was moved with a button down.
 @param x/y  new mouse position
*/
void vvCanvas::mouseDragged(int x, int y)
{
  float panValue[2];        // actual pan value to use for translation
  float pixelInWorld;       // size of one pixel in world coordinates [mm]
  float factor;             // scaling factor
  int dx, dy;               // distance mouse has been moved [pixels]

  vvDebugMsg::msg(3, "vvCanvas::mouseDragged() to ", x, y);
  _lastX = _curX;                                   // save current mouse coordinates for next call
  _lastY = _curY;
  _curX  = x;
  _curY  = y;
  dx    = _curX - _lastX;
  dy    = _curY - _lastY;

  switch (_buttonState)
  {
    case LEFT_BUTTON:                             // left button rotates
      _lastRotation = _ov._camera.trackballRotation(_width, _height, _lastX, _lastY, _curX, _curY);
      break;

    case MIDDLE_BUTTON:                           // middle button pans
    case LEFT_BUTTON | RIGHT_BUTTON:              // left plus right simulate middle button
      pixelInWorld = _ov.getViewportWidth() / float(_width);
      panValue[0] = pixelInWorld * float(dx);
      panValue[1] = pixelInWorld * float(dy);
      _ov._camera.translate(panValue[0], -panValue[1], 0.0f);
      break;

    case RIGHT_BUTTON:                            // right button moves camera along z axis
      factor = float(dy);
      _ov._camera.translate(0.0f, 0.0f, factor);
      break;

    default: break;
  }

  draw();
}

//----------------------------------------------------------------------------
/// Reset object and view
void vvCanvas::resetObjectView()
{
  vvDebugMsg::msg(1, "vvCanvas::resetObjectView()");
  _ov.resetCamera();
  _ov.resetObject();
}

//----------------------------------------------------------------------------
/// Multiply object's model view matrix by matrix m
void vvCanvas::transformObject(const vvMatrix& m)
{
  vvDebugMsg::msg(3, "vvCanvas::transformObject()");
  _ov._camera.multiplyRight(m);
}

//----------------------------------------------------------------------------
/** Set a new renderer.
  @param alg 0=no change, 1=textures, 2=Stingray, -1=suppress rendering
  @param r  index of renderer
  @param v  volume description
*/
void vvCanvas::setRenderer(vvRenderer::RendererType alg)
{
  vvRenderState renderState;

  vvDebugMsg::msg(1, "vvCanvas::setRenderer()");

  if (_renderer) 
  {
    renderState = *_renderer;    // if previous renderer existed, save its state
    delete _renderer;
  }

  if (alg==vvRenderer::INVALID) alg = _currentAlgorithm;

  _vd->resizeEdgeMax(_ov.getViewportWidth() * DEFAULT_OBJ_SIZE);

  switch(alg)
  {
  case vvRenderer::GENERIC:
    _renderer = new vvRenderer(_vd, renderState);
    break;
  case vvRenderer::TEXREND:
    _renderer = new vvTexRend(_vd, renderState);
    break;
  case vvRenderer::SOFTSW:
    _renderer = new vvSoftShearWarp(_vd, renderState);
    break;
#ifdef HAVE_CUDA
  case vvRenderer::CUDASW:
    _renderer = new vvCudaShearWarp(_vd, renderState);
    break;
  case vvRenderer::RAYREND:
    _renderer = new vvRayRend(_vd, renderState);
    break;
#endif
#ifdef HAVE_VOLPACK
  case vvRenderer::VOLPACK:
    _renderer = new vvVolPack(_vd, renderState);
    break;
#endif
#ifdef USE_STINGRAY
  case  vvRenderer::STINGRAY:
    _renderer = new vvStingray(_vd, renderState);
    break;
#endif
  default:
    assert(0);
    break;
  }

  draw();

  _currentAlgorithm = alg;
}

//----------------------------------------------------------------------------
/** Get current renderer
*/
int vvCanvas::getRenderer() const
{
  return _currentAlgorithm;
}

//----------------------------------------------------------------------------
/** Set the OpenGL canvas background color.
  @param r,g,b color [0..1]
*/
void vvCanvas::setBackgroundColor(float r, float g, float b)
{
  vvDebugMsg::msg(1, "vvCanvas::setBackgroundColor()");
  r = ts_clamp(r, 0.0f, 1.0f);
  g = ts_clamp(g, 0.0f, 1.0f);
  b = ts_clamp(b, 0.0f, 1.0f);
  _bgColor[0] = r;
  _bgColor[1] = g;
  _bgColor[2] = b;
  
  // Set boundary color to inverse of background:
  float bColor;
  if (_bgColor[0] + _bgColor[1] + _bgColor[2] > 1.5f) bColor = 0.0f;
  else bColor = 1.0f;
  _renderer->setParameter(vvRenderState::VV_BOUND_COLOR, vvColor(bColor, bColor, bColor));
  _renderer->setParameter(vvRenderState::VV_CLIP_COLOR, vvColor(bColor, bColor, bColor));
}

//----------------------------------------------------------------------------
/// Get the OpenGL canvas background color.
void vvCanvas::getBackgroundColor(float& r, float& g, float& b)
{
  vvDebugMsg::msg(1, "vvCanvas::getBackgroundColor()");
  r = _bgColor[0];
  g = _bgColor[1];
  b = _bgColor[2];
}

//----------------------------------------------------------------------------
bool vvCanvas::getARToolkit()
{
  return _artoolkit;
}

//----------------------------------------------------------------------------
void vvCanvas::setARToolkit(bool mode)
{
  if (mode==_artoolkit) return;    // allset
  _artoolkit = mode;
#ifdef VV_USE_ARTOOLKIT
  cerr << "ARToolkit support " << mode << endl;
  if (_artoolkit)
  {
    _arTracking = new vvARTracking();
    _arTracking->showPropertiesDialog(false);
    _arTracking->init();
  }
  else
  {
    delete _arTracking;
    _arTracking = NULL;
  }
#endif
}

//----------------------------------------------------------------------------
/** Called when ARToolkit is used and a timer event occurs.
*/
void vvCanvas::artTimerEvent()
{
  vvMatrix artMatrix;

  if (_artoolkit)
  {
#ifdef VV_USE_ARTOOLKIT
    // Show video image:
    //renderImage(arTracking->getVideoImage(), 10, 10);

    // Process marker position:
    _arTracking->track();
    artMatrix = _arTracking->getMarkerMatrix();
    artMatrix.print("artMatrix");
    //ov.mv.copy(&artMatrix);
    vvVector3 pos;
    pos.zero();
    pos.multiply(&artMatrix);
    pos.scale(0.01f);
    _renderer->setPosition(&pos);
    pos.print("pos");
#endif
  }
}

//----------------------------------------------------------------------------
void vvCanvas::renderImage(uchar* image, int width, int height)
{
  GLfloat glsRasterPos[4];                        // current raster position (glRasterPos)

  // Save matrix states:
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(-1.0f, 1.0f, -1.0f, 1.0f, 10.0f, -10.0f);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  // Store raster position:
  glGetFloatv(GL_CURRENT_RASTER_POSITION, glsRasterPos);

  // Draw palette:
  glRasterPos2f(-1.0f,-1.0f);                     // pixmap origin is bottom left corner of output window
  glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, (GLvoid*)image);

  // Restore raster position:
  glRasterPos4fv(glsRasterPos);

  // Restore matrix states:
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
}

//----------------------------------------------------------------------------
void vvCanvas::setSwapEyes(bool swap)
{
  _swapEyes = swap;
}

//----------------------------------------------------------------------------
bool vvCanvas::getSwapEyes()
{
  return _swapEyes;
}

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
