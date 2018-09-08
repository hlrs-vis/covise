//****************************************************************************
// Project:         Virvo (Virtual Reality Volume Renderer)
// Copyright:       (c) 1999-2004 Jurgen P. Schulze. All rights reserved.
// Author's E-Mail: schulze@cs.brown.edu
// Affiliation:     Brown University, Department of Computer Science
//****************************************************************************

#include <math.h>
#include <string.h>

#include <virvo/vvopengl.h>

#include <iostream>

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include <virvo/vvdebugmsg.h>
#include <virvo/vvtokenizer.h>
#include <virvo/private/vvgltools.h>
#include "vvobjview.h"

#ifdef __APPLE__

#include <AvailabilityMacros.h>

#if MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_9

  #pragma GCC diagnostic ignored "-Wdeprecated"

#endif

#endif // __APPLE__


using std::cerr;
using std::cout;
using std::endl;

const float vvObjView::VIEWER_POS_X = 0.0f;
const float vvObjView::VIEWER_POS_Y = 0.0f;
const float vvObjView::VIEWER_POS_Z = -2.0f;

//----------------------------------------------------------------------------
  /// Constructor.
  vvObjView::vvObjView()
  {
  vvDebugMsg::msg(1, "vvObjView::vvObjView()");
  aspect = 1.0f;                                 // default aspect ratio is 1:1
  cameraString = "VIRVO_CAMERA";
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
  projType  = ORTHO;                             // default projection mode is orthographic
  eyeDist   = -0.03f;                            // default eye distance (empirical value)
  rotAngle  = 5.0f;                              // default rotational angle (empirical value)
  fov       = 2.0f;                              // default field of view
  zNear     = -100.0f;
  zFar      = 100.0f;
  resetMV();
}


//----------------------------------------------------------------------------
/** Reset modelview matrix only.
 */
void vvObjView::resetMV()
{
  vvDebugMsg::msg(1, "vvObjView::resetMV()");
  mv.identity();
  mv.translate(VIEWER_POS_X, VIEWER_POS_Y, VIEWER_POS_Z);
}


//----------------------------------------------------------------------------
/** Save modelview matrix to an ASCII file. If a file exists with the same
    name, it will be overwritten. Here is an example file contents:<BR>
  <PRE>
  VIRVO Camera
  0.944588 0.051051 0.324263 0.000000
  0.092229 0.906766 -0.411423 0.000000
  -0.315034 0.418532 0.851813 -2.000000
  0.000000 0.000000 0.000000 1.000000
  </PRE>
  @param filename name of file to save (convention for extension: .txt)
  @return true if file was written ok, false if file couldn't be written
*/
bool vvObjView::saveMV(const char* filename)
{
  FILE* fp;

  vvDebugMsg::msg(1, "vvObjView::saveMV()");

  fp = fopen(filename, "wb");
  if (fp==NULL) return false;
  saveMV(fp);
  fclose(fp);
  return true;
}


bool vvObjView::saveMV(FILE* fp)
{
  fputs(cameraString, fp);
  fputc('\n', fp);
  for (size_t i=0; i<4; ++i)
  {
    fprintf(fp, "%f %f %f %f\n", mv(i, 0), mv(i, 1), mv(i, 2), mv(i, 3));
  }
  return true;
}


//----------------------------------------------------------------------------
/** Load modelview matrix from an ASCII file.
  @param filename name of file to load
  @return true if file was loaded ok, false if file couldn't be loaded
*/
bool vvObjView::loadMV(const char* filename)
{
  bool retval = false;

  vvDebugMsg::msg(1, "vvObjView::loadMV()");

  std::ifstream file(filename);
  if (!file.is_open()) return false;

  retval = loadMV(file);

  return retval;
}


bool vvObjView::loadMV(std::ifstream& file)
{
  bool retval = false;

  vvMatrix camera;

  // Initialize tokenizer:
  vvTokenizer::TokenType ttype;
  vvTokenizer* tokenizer = new vvTokenizer(file);
  tokenizer->setCommentCharacter('#');
  tokenizer->setEOLisSignificant(false);
  tokenizer->setCaseConversion(vvTokenizer::VV_UPPER);
  tokenizer->setParseNumbers(true);

  // Parse file:
  ttype = tokenizer->nextToken();
  if (ttype!=vvTokenizer::VV_WORD) goto done;
  if (strcmp(tokenizer->sval, cameraString) != 0) goto done;
  for (size_t i=0; i<4; ++i)
    for (size_t j=0; j<4; ++j)
  {
    ttype = tokenizer->nextToken();
    if (ttype != vvTokenizer::VV_NUMBER) goto done;
    camera(i, j) = tokenizer->nval;
  }
  mv = vvMatrix(camera);
  retval = true;

  done:
  delete tokenizer;

  return retval;
}


//----------------------------------------------------------------------------
/** Set the projection matrix.
  @param pt        projection type: ORTHO, PERSPECTIVE, or FRUSTUM
  @param range     minimum horizontal and vertical viewing range, format depends
                   on projection type: for ORTHO and FRUSTUM range defines the
                   viewing range in world coordinates, for PERSPECTIVE it defines
                   the field of view in degrees.
  @param nearPlane distance from viewer to near clipping plane (>0)
  @param farPlane  distance from viewer to far clipping plane (>0)
*/
void vvObjView::setProjection(ProjectionType pt, float range, float nearPlane, float farPlane)
{
  const float MINIMUM = 0.0001f;                 // minimum value for perspective near/far planes

  vvDebugMsg::msg(2, "vvObjView::setProjection()");

  // Near and far planes need value checking for perspective modes:
  if (pt!=ORTHO)
  {
    if (nearPlane<=0.0f) nearPlane = MINIMUM;
    if (farPlane <=0.0f) farPlane  = MINIMUM;
  }

  fov   = range;
  zNear = nearPlane;
  zFar  = farPlane;
  projType  = pt;
  updateProjectionMatrix();
}


//----------------------------------------------------------------------------
/** Set the OpenGL projection matrix according to current class values.
 */
void vvObjView::updateProjectionMatrix()
{
  GLint glsMatrixMode;                           // stores GL_MATRIX_MODE
  float xHalf, yHalf;                            // half x and y coordinate range
  float fovy;                                    // field of view in y direction

  vvDebugMsg::msg(2, "vvObjView::updateProjectionMatrix()");

  // Save matrix mode:
  glGetIntegerv(GL_MATRIX_MODE, &glsMatrixMode);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  // Precompute x and y range:
  xHalf = 0.5f * ((aspect < 1.0f) ? fov : (fov * aspect));
  yHalf = 0.5f * ((aspect > 1.0f) ? fov : (fov / aspect));
  fovy  = (aspect > 1.0f) ? fov : (180.0f / VV_PI * ((float)atan(tan(fov * VV_PI / 180.0f) / aspect)));

  // Set new projection matrix:
  switch (projType)
  {
  case ORTHO:
    glOrtho(-xHalf, xHalf, -yHalf, yHalf, zNear, zFar);
    break;
  case FRUSTUM:cerr << "FRUSTUM" << endl;
    glFrustum(-xHalf, xHalf, -yHalf, yHalf, zNear, zFar);
    break;
  case PERSPECTIVE:
    gluPerspective(fovy, aspect, zNear, zFar);
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
  if (aspect>0.0f)
  {
    aspect = ar;
    updateProjectionMatrix();
  }
}


//----------------------------------------------------------------------------
/** Set the depth clipping planes positions
  @param newNP,newFP  positions of the new near and far clipping planes
*/
void vvObjView::setDepthRange(float newNP, float newFP)
{
  vvDebugMsg::msg(2, "vvObjView::setDepthRange()");
  zNear = newNP;
  zFar  = newFP;
  updateProjectionMatrix();
}


//----------------------------------------------------------------------------
/** Update the OpenGL modelview matrix for a particular eye position.
  The OpenGL draw buffer must be selected by the caller.
  @param eye  eye for which to draw the object
*/
void vvObjView::updateModelviewMatrix(EyeType eye)
{
  vvGLTools::printGLError("enter vvObjView::updateModelviewMatrix()");

  vvMatrix mvRight;                             // modelview matrix for right eye
  vvMatrix invRot;                              // inverse rotational matrix
  vvVector3 v(0.0, -1.0, 0.0);                   // rotational vector
  GLint glsMatrixMode;                           // stores GL_MATRIX_MODE
  float flat[16];

  vvDebugMsg::msg(2, "vvObjView::updateModelviewMatrix()");

  // Save matrix mode:
  glGetIntegerv(GL_MATRIX_MODE, &glsMatrixMode);
  glMatrixMode(GL_MODELVIEW);

  if (eye!=RIGHT_EYE)                             // use stored matrix for left eye
  {
    mv.getGL(flat);
  }
  else                                           // convert for right eye
  {
    // Convert axis coordinates (v) from WCS to OCS:
    mvRight = vvMatrix(mv);
    invRot.identity();
    invRot.copyRot(mv);
    invRot.invertOrtho();
    v.multiply(invRot);
    v.normalize();                              // normalize before rotation!

    mvRight.rotate(rotAngle * (float)VV_PI / 180.0f, v[0], v[1], v[2]);
    mvRight.translate(eyeDist, 0.0f, 0.0f);
    mvRight.getGL(flat);
  }

  // Load matrix to OpenGL:
  glLoadMatrixf(flat);

  // Restore matrix mode:
  glMatrixMode(glsMatrixMode);

  vvGLTools::printGLError("leave vvObjView::updateModelviewMatrix()");
}


//-----------------------------------------------------------------------------
/** Rotates the model view matrix according to a fictitious trackball.
  @param width, height  window sizes in pixels
  @param fromX, fromY   mouse move starting position in pixels
  @param toX, toY       mouse move end position in pixels
*/
void vvObjView::trackballRotation(int width, int height, int fromX, int fromY, int toX, int toY)
{
  mv.trackballRotation(width, height, fromX, fromY, toX, toY);
}


//----------------------------------------------------------------------------
/** Set eye distance for stereo viewing.
  @param ed eye distance
*/
void vvObjView::setEyeDistance(float ed)
{
  eyeDist = ed;
}


//----------------------------------------------------------------------------
/** Set rotational angle for stereo viewing.
  @param angle new rotational angle
*/
void vvObjView::setRotationalAngle(float angle)
{
  rotAngle = angle;
}


//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
