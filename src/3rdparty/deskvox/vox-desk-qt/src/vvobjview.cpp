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

// OS:
#include <iostream>
#include <math.h>
#include <string.h>

// OpenGL:
#include <vvopengl.h>

// Virvo:
#include <vvdebugmsg.h>
#include <vvtokenizer.h>
#include <vvtoolshed.h>

// Local:
#include "vvobjview.h"


#ifdef __APPLE__

#include <AvailabilityMacros.h>

#if MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_9

  #pragma GCC diagnostic ignored "-Wdeprecated"

#endif

#endif // __APPLE__


using namespace vox;
using namespace std;

const float vvObjView::DEF_CAMERA_POS_Z  =  200.0f;
const float vvObjView::DEF_IOD           =  63.0f;     // average for humans: 65mm, good value for everyone: 63mm
const float vvObjView::EPSILON_NEAR      =  0.0001f;
const float vvObjView::DEF_CLIP_NEAR     =  1.0f;
const float vvObjView::DEF_CLIP_FAR      =  500.0f;
const float vvObjView::DEF_FOV           =  45.0f * VV_PI / 180.0f;
const float vvObjView::DEF_VIEWPORT_WIDTH=  150.0f;
const char* vvObjView::CAMERA_WINDOW_STRING = "CameraWindow:";
const char* vvObjView::CAMERA_CLIPPING_STRING = "CameraClipping:";
const char* vvObjView::CAMERA_MATRIX_STRING = "CameraMatrix:";

//----------------------------------------------------------------------------
/// Constructor.
vvObjView::vvObjView()
{
  vvDebugMsg::msg(1, "vvObjView::vvObjView()");
  _aspect = 1.0f;                                  // default aspect ratio is 1:1
  reset();
}

//----------------------------------------------------------------------------
/** Initialization routine.
  May be called by the application anytime to reset object orientation
  and projection matrix.
*/
void vvObjView::reset()
{
  vvDebugMsg::msg(1, "vvObjView::reset()");
  _projType  = ORTHO;
  _iod       = DEF_IOD;
  _minFOV    = DEF_FOV;
  _viewportWidth = DEF_VIEWPORT_WIDTH;
  _zNear     = DEF_CLIP_NEAR;
  _zFar      = DEF_CLIP_FAR;
  resetObject();
  resetCamera();
}

//----------------------------------------------------------------------------
/** Reset volume object matrix.
 */
void vvObjView::resetObject()
{
  vvDebugMsg::msg(1, "vvObjView::resetObject()");
  _object.identity();
}

//----------------------------------------------------------------------------
/** Reset camera matrix.
 */
void vvObjView::resetCamera()
{
  vvDebugMsg::msg(1, "vvObjView::resetCamera()");
  _camera.makeLookAt(0.0, 0.0, DEF_CAMERA_POS_Z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
}

//----------------------------------------------------------------------------
/** Save camera matrix to an ASCII file. If a file exists with the same
    name, it will be overwritten. Here is an example file contents:<BR>
  <PRE>
  CameraMatrix: 1.085395 0.097215 -0.620255 0.000000 0.223907 1.097373 
  0.563815 0.000000 0.586543 -0.598808 0.932548 0.000000 13.775513 7.908164 
  -300.000000 1.000000
  </PRE>
  @param filename name of file to save (convention for extension: .txt)
  @return true if file was written ok, false if file couldn't be written
*/
bool vvObjView::saveCamera(const char* filename)
{
  FILE* fp;
  float xMin, xMax, yMin, yMax, yFOV;

  vvDebugMsg::msg(1, "vvObjView::saveCamera()");

  fp = fopen(filename, "wb");
  if (fp==NULL) return false;

  // Write viewing window:
  getWindowExtent(xMin, xMax, yMin, yMax, yFOV);
  fputs(CAMERA_WINDOW_STRING, fp);
  fprintf(fp, " %f %f %f %f\n", xMin, xMax, yMin, yMax);

  // Write clipping planes:
  fputs(CAMERA_CLIPPING_STRING, fp);
  fprintf(fp, " %f %f %f\n", _zNear, 0.2f, _zFar);    // TODO: find out what 0.2 means

  // Write camara matrix:
  fputs(CAMERA_MATRIX_STRING, fp);
  for (size_t i=0; i<4; ++i)
  {
    for (size_t j=0; j<4; ++j)
    {
      fprintf(fp, " %f", _camera(j, i));
    }
  }
  fputc('\n', fp);
  fclose(fp);
  return true;
}

//----------------------------------------------------------------------------
/** Load camera matrix from an ASCII file.
  @param filename name of file to load
  @return true if file was loaded ok, false if file couldn't be loaded
*/
bool vvObjView::loadCamera(const char* filename)
{
  bool done = false;
  vvMatrix camera;
  float extent[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float clip[3] = {0.0f, 0.0f, 0.0f};

  vvDebugMsg::msg(1, "vvObjView::loadCamera()");

  std::ifstream file(filename);
  if (!file.is_open()) return false;

  // Initialize tokenizer:
  vvTokenizer::TokenType ttype;
  vvTokenizer* tokenizer = new vvTokenizer(file);
  tokenizer->setCommentCharacter('#');
  tokenizer->setEOLisSignificant(false);
  tokenizer->setCaseConversion(vvTokenizer::VV_UPPER);
  tokenizer->setParseNumbers(true);

  // Parse file:
  while (!done)
  {
    ttype = tokenizer->nextToken();
    if (ttype != vvTokenizer::VV_WORD) done = true;
    else if (vvToolshed::strCompare(tokenizer->sval, CAMERA_WINDOW_STRING) == 0)
    {
      for (size_t i=0; i<4 && !done; ++i)
      {
        ttype = tokenizer->nextToken();
        if (ttype == vvTokenizer::VV_NUMBER) extent[i] = tokenizer->nval;
        else 
        {
          cerr << "Unexpected parameter in camera file line " << tokenizer->getLineNumber() << endl;
          done = true;
        }
      }
      // Change viewport width, but not aspect ratio: would require window resize.
      _viewportWidth  = extent[1] - extent[0];
    }
    else if (vvToolshed::strCompare(tokenizer->sval, CAMERA_CLIPPING_STRING) == 0)
    {
      for (size_t i=0; i<3 && !done; ++i)
      {
        ttype = tokenizer->nextToken();
        if (ttype == vvTokenizer::VV_NUMBER) clip[i] = tokenizer->nval;
        else 
        {
          cerr << "Unexpected parameter in camera file line " << tokenizer->getLineNumber() << endl;
          done = true;
        }
      }
      _zNear = clip[0];
      _zFar = clip[2];
    }
    else if (vvToolshed::strCompare(tokenizer->sval, CAMERA_MATRIX_STRING) == 0)
    {
      for (size_t i=0; i<4 && !done; ++i)
      {
        for (size_t j=0; j<4 && !done; ++j)
        {
          ttype = tokenizer->nextToken();
          if (ttype == vvTokenizer::VV_NUMBER) camera(j, i) = tokenizer->nval;
          else 
          {
            cerr << "Unexpected parameter in camera file line " << tokenizer->getLineNumber() << endl;
            done = true;
          }
        }
      }
      _camera = camera;
    }
    else 
    {
      cerr << "Unknown entry in camera file line " << tokenizer ->getLineNumber() << endl;
      tokenizer->nextLine();
    }
  } 
  if (_zNear > 0.0f) _minFOV = 2.0f * float(atan(ts_min(_viewportWidth / _zNear, (extent[3] - extent[2]) / _zNear)));

  delete tokenizer;
  setProjectionMatrix();
  return true;
}

//----------------------------------------------------------------------------
/** @return field of view [radians]
*/
float vvObjView::getFOV()
{
  return _minFOV;
}

//----------------------------------------------------------------------------
/** @return width of viewport (=viewing window) [mm]
*/
float vvObjView::getViewportWidth()
{
  return _viewportWidth;
}

//----------------------------------------------------------------------------
/** @return distance of near view frustum clipping plane
*/
float vvObjView::getNearPlane()
{
  return _zNear;
}

//----------------------------------------------------------------------------
/** @return distance of far view frustum clipping plane
*/
float vvObjView::getFarPlane()
{
  return _zFar;
}

//----------------------------------------------------------------------------
/** Set the projection matrix.
  @param pt        projection type: ORTHO, PERSPECTIVE, or FRUSTUM
  @param range     minimum horizontal and vertical viewing range, format depends
                   on projection type: for ORTHO and FRUSTUM range defines the
                   viewing range in world coordinates, for PERSPECTIVE it defines
                   the field of view [radians].
  @param nearPlane distance from viewer to near clipping plane (>0)
  @param farPlane  distance from viewer to far clipping plane (>0)
*/
void vvObjView::setProjection(ProjectionType pt, float range, float nearPlane, float farPlane)
{
  vvDebugMsg::msg(2, "vvObjView::setProjection()");

  // Near and far planes need value checking for perspective modes:
  if (pt!=ORTHO)
  {
    if (nearPlane<=0.0f) nearPlane = EPSILON_NEAR;
    if (farPlane <=0.0f) farPlane  = EPSILON_NEAR;
  }

  if (pt==PERSPECTIVE) _minFOV = range;
  else _viewportWidth = range;
  _zNear = nearPlane;
  _zFar  = farPlane;
  _projType = pt;
  setProjectionMatrix();
}

//----------------------------------------------------------------------------
/** Set the OpenGL projection matrix according to current class values.
  @param xMin,xMax,yMin,yMax  min/max logical values of window in near plane
  @param yFOV field of view on y axis [radians]
 */
void vvObjView::getWindowExtent(float& xMin, float& xMax, float& yMin, float& yMax, float& yFOV)
{
  float xFOV;
  float xHalf, yHalf;

  if (_projType==PERSPECTIVE)
  {
    if (_aspect >= 1.0f)
    {
      xFOV = 2.0f * float(atan(_aspect * tan(_minFOV / 2.0f)));
      yFOV = _minFOV;
    } 
    else
    {
      xFOV = _minFOV;
      yFOV = 2.0f * float(atan(tan(_minFOV / 2.0f) / _aspect));
    }

    // TODO: the following lines are most likely incorrect and shouldn't use zNear. Check with gluPerspective to find out how it needs to be done
    xMax = _zNear * float(tan(xFOV / 2.0f));
    xMin = -xMax;
    yMax = _zNear * float(tan(yFOV / 2.0f));
    yMin = -yMax;
  }
  else
  {
    xHalf = 0.5f * ((_aspect > 1.0f) ? (_viewportWidth * _aspect) : _viewportWidth);
    yHalf = 0.5f * ((_aspect > 1.0f) ? _viewportWidth : (_viewportWidth / _aspect));
    xMin = -xHalf;
    xMax =  xHalf;
    yMin = -yHalf;
    yMax =  yHalf;
    yFOV = 0.0f;    // undefined in parallel projection mode
  }
}

//----------------------------------------------------------------------------
/** Set the OpenGL projection matrix according to current class values.
 */
void vvObjView::setProjectionMatrix()
{
  GLint glsMatrixMode;                            // stores GL_MATRIX_MODE
  float yFOV;                                     // field of view in y direction
  float xMin, xMax, yMin, yMax;                   // visible min/max coordinates on near plane
  
  vvDebugMsg::msg(3, "vvObjView::setProjectionMatrix()");

  // Save matrix mode:
  glGetIntegerv(GL_MATRIX_MODE, &glsMatrixMode);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  // Precompute window parameters:
  getWindowExtent(xMin, xMax, yMin, yMax, yFOV);

  // Set new projection matrix:
  switch (_projType)
  {
    case ORTHO:
      glOrtho(xMin, xMax, yMin, yMax, _zNear, _zFar);
      break;
    case FRUSTUM:
      glFrustum(xMin, xMax, yMin, yMax, _zNear, _zFar);
      break;
    case PERSPECTIVE:
      gluPerspective(yFOV * 180.0f / VV_PI, _aspect, _zNear, _zFar);
      break;
    default: break;
  }

  // Restore matrix mode:
  glMatrixMode(glsMatrixMode);
}

//----------------------------------------------------------------------------
/** Set the aspect ratio of the viewing window.
  @param ar aspect ratio (= viewing window width/height)
*/
void vvObjView::setAspectRatio(float ar)
{
  vvDebugMsg::msg(2, "vvObjView::setAspectRatio()");
  if (_aspect>0.0f)
  {
    _aspect = ar;
    setProjectionMatrix();
  }
}

//----------------------------------------------------------------------------
/** Set the depth clipping planes positions
  @param newNP,newFP  positions of the new near and far clipping planes
*/
void vvObjView::setDepthRange(float newNP, float newFP)
{
  vvDebugMsg::msg(2, "vvObjView::setDepthRange()");
  _zNear = newNP;
  _zFar  = newFP;
  setProjectionMatrix();
}

//----------------------------------------------------------------------------
/** Set the OpenGL modelview matrix for a particular eye position.
  The OpenGL draw buffer must be selected by the caller.
  @param eye  eye for which to draw the object
*/
void vvObjView::setModelviewMatrix(EyeType eye)
{
  vvMatrix camera;                               // view matrix for current eye
  vvMatrix mv;                                   // modelview matrix for OpenGL
  GLint glsMatrixMode;                           // stores GL_MATRIX_MODE
  float flat[16];

  vvDebugMsg::msg(3, "vvObjView::setModelviewMatrix()");

  // Save matrix mode:
  glGetIntegerv(GL_MATRIX_MODE, &glsMatrixMode);
  glMatrixMode(GL_MODELVIEW);

  camera = _camera;      // use stored matrix to begin with
  if (eye==RIGHT_EYE)        // convert for right eye
  {
    camera.translate(_iod, 0.0f, 0.0f);
  }
  else if (eye==LEFT_EYE)   // convert for left eye
  {
    camera.translate(-_iod, 0.0f, 0.0f);
  }

  // Load matrix to OpenGL:
  mv = _object * camera;
  mv.getGL(flat);
  glLoadMatrixf(flat);

  // Restore matrix mode:
  glMatrixMode(glsMatrixMode);
}

//----------------------------------------------------------------------------
/** Set inter-ocular distance for stereo viewing.
  @param iod inter-ocular distance [millimeters]
*/
void vvObjView::setIOD(float iod)
{
  _iod = iod;
}

//----------------------------------------------------------------------------
/** @return inter-ocular distance for stereo viewing [mm]
*/
float vvObjView::getIOD() const
{
  return _iod;
}
    
//----------------------------------------------------------------------------
/** Set default viewing direction.
*/
void vvObjView::setDefaultView(ViewType view)
{
  resetObject();
  resetCamera();

  switch(view)
  {
    case LEFT:   _camera.rotate( VV_PI/2.0f, 0.0f, 1.0f, 0.0f); break;
    case RIGHT:  _camera.rotate(-VV_PI/2.0f, 0.0f, 1.0f, 0.0f); break;
    case TOP:    _camera.rotate( VV_PI/2.0f, 1.0f, 0.0f, 0.0f); break;
    case BOTTOM: _camera.rotate(-VV_PI/2.0f, 1.0f, 0.0f, 0.0f); break;
    case BACK:   _camera.rotate( VV_PI,      0.0f, 1.0f, 0.0f); break;
    case FRONT:   // front is done with reset
    default: break;
  }
}
    
//============================================================================
// EOF
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
