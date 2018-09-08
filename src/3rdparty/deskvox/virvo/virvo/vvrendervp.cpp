//****************************************************************************
// Project Affiliation: Virvo (Virtual Reality Volume Renderer)
// Copyright:           (c) 2002 Juergen Schulze-Doebold. All rights reserved.
// Author's E-Mail:     schulze@hlrs.de
// Institution:         University of Stuttgart, Supercomputing Center (HLRS)
//****************************************************************************

#include <iostream>
#include <iomanip>
using std::cerr;
using std::endl;
using std::setprecision;

#include "private/vvgltools.h"

#include <cstring>
#include "vvplatform.h"
#include "vvopengl.h"
#include <assert.h>
#include "vvrendervp.h"
#include "vvdebugmsg.h"
#include "vvtoolshed.h"
#include "vvclock.h"
#include "vvvoldesc.h"
#include "vvswitchrenderer.impl.h"
#include "gl/util.h"
#include "private/vvlog.h"

#ifdef HAVE_VOLPACK

//============================================================================
// Class Definitions
//============================================================================

//----------------------------------------------------------------------------
/// Constructor
vvRenderVP::vvRenderVP(vvVolDesc* vd, vvRenderState st) : vvRenderer(vd, st)
{
  int i;

  vvDebugMsg::msg(1, "vvRenderVP::vvRenderVP()");

  rendererType = RENDERVP;
  vox = NULL;
  vpc = NULL;
  image = NULL;
  boundedData = NULL;
  stickyLights = true;
  timing = false;
  current_percentage = 50.0;
  initialize();
  makeUnclassifiedVolume(vd->getRaw());

  // Set gradients table:
  for (i=0; i<VP_GRAD_MAX+1; ++i)
    gradient_table[i] = 1.0f;

  updateTransferFunction();
}

//----------------------------------------------------------------------------
/// Destructor
vvRenderVP::~vvRenderVP()
{
  vvDebugMsg::msg(1, "vvRenderVP::~vvRenderVP()");
  delete[] vox;
  delete[] image;
  delete[] boundedData;
  if (vpc!=NULL)
  {
    vpDestroyClassifiedVolume(vpc);
    vpDestroyContext(vpc);
  }
  // Important: do not delete the volume descriptor (vd)! It must be asked for
  // by the caller who is responsible for its deletion.
}

//----------------------------------------------------------------------------
/// Initialize Volpack
void vvRenderVP::initialize()
{
  VoxelType dummy;
  vpResult e;
  double vpm[4][4];
  int c, i;
  vvMatrix mv;         // default model view matrix

  vvDebugMsg::msg(1, "vvRenderVP::initialize()");

  if (vpc!=NULL)
  {
    vpDestroyClassifiedVolume(vpc);
    vpDestroyContext(vpc);
    vvDebugMsg::msg(2, "vvRenderVP::initialize(): previous volume destroyed");
  }
  vpc = vpCreateContext();    // create rendering context

  // Set matrix multiplication mode:
  vpSeti(vpc, VP_CONCAT_MODE, VP_CONCAT_RIGHT);   

  // Declare voxel layout:
  if ((e = vpSetVoxelSize(vpc, sizeof(VoxelType), NUM_FIELDS, NUM_SHADE_FIELDS, NUM_CLASSIFY_FIELDS)) != VP_OK) FATAL_VP_ERROR("vpSetVoxelSize", e);
  if ((e = vpSetVoxelField(vpc, NORM_FIELD, sizeof(dummy.normal),   vpFieldOffset(&dummy, normal), VP_NORM_MAX)) != VP_OK) FATAL_VP_ERROR("vpSetVoxelField", e);
  if ((e = vpSetVoxelField(vpc, SCALAR_FIELD, sizeof(dummy.scalar), vpFieldOffset(&dummy, scalar), SCALAR_MAX)) != VP_OK) FATAL_VP_ERROR("vpSetVoxelField", e);
  if ((e = vpSetVoxelField(vpc, GRAD_FIELD, sizeof(dummy.gradient), vpFieldOffset(&dummy, gradient), VP_GRAD_MAX)) != VP_OK) FATAL_VP_ERROR("vpSetVoxelField", e);
  vvDebugMsg::msg(2, "vvRenderVP::initialize(): voxel layout declared");

  // Generate and set default gradient table:
  for (i=0; i<VP_GRAD_MAX + 1; ++i)
    gradient_table[i] = 1.0;
  if ((e = vpSetClassifierTable(vpc, 1, GRAD_FIELD, gradient_table, sizeof(gradient_table))) != VP_OK) FATAL_VP_ERROR("vpSetClassifierTable", e);

  // Generate and set default classification table:
  scalar_table[0] = 0.0;
  for (i=1; i<VP_SCALAR_MAX + 1; ++i)
    scalar_table[i] = 1.0;
  if ((e = vpSetClassifierTable(vpc, 0, SCALAR_FIELD, scalar_table, sizeof(scalar_table))) != VP_OK) FATAL_VP_ERROR("vpSetClassifierTable", e);

  // Set optimizations:
  if ((e = vpSetd(vpc, VP_MIN_VOXEL_OPACITY, 0.05)) != VP_OK) FATAL_VP_ERROR("vpSetd", e);
  if ((e = vpSetd(vpc, VP_MAX_RAY_OPACITY, 0.95)) != VP_OK) FATAL_VP_ERROR("vpSetd", e);

  // Generate and set default weight table (grayscale):
  for (c=0; c<VP_MAX_MATERIAL-1; ++c)
    for (i=0; i<VP_SCALAR_MAX+1; ++i)
      weight_table[i][c] = (c<3) ? ((float)i / (float)VP_SCALAR_MAX) : 0.0f;

  // Define shader:
  if ((e = vpSetLookupShader(vpc, COLOR_CHANNELS, VP_MAX_MATERIAL-1, NORM_FIELD, (float*)color_table,
          sizeof(color_table), SCALAR_FIELD, (float*)weight_table, sizeof(weight_table))) != VP_OK) FATAL_VP_ERROR("vpSetLookupShader", e);
  vvDebugMsg::msg(2, "vvRenderVP::initialize(): shader defined");

  // Define lights:
  if ((e = vpSetLight(vpc, VP_LIGHT0, VP_COLOR,      1.0, 1.0, 1.0)) != VP_OK) FATAL_VP_ERROR("vpSetLight", e);
  if ((e = vpSetLight(vpc, VP_LIGHT0, VP_DIRECTION, -0.6, 0.6, 1.0)) != VP_OK) FATAL_VP_ERROR("vpSetLight", e);
  if ((e = vpSetLight(vpc, VP_LIGHT1, VP_COLOR,      1.0, 1.0, 1.0)) != VP_OK) FATAL_VP_ERROR("vpSetLight", e);
  if ((e = vpSetLight(vpc, VP_LIGHT1, VP_DIRECTION,  0.6, 0.6, 1.0)) != VP_OK) FATAL_VP_ERROR("vpSetLight", e);
  setLights(0, stickyLights);
  vvDebugMsg::msg(2, "vvRenderVP::initialize(): lights and materials defined");

  // Set depth cueing:
//  vpEnable(vpc, VP_DEPTH_CUE, 1);
//  vpSetDepthCueing(vpc, 0.8, 3.0);

  // Set default model view matrix:
  mv.get((double*)vpm);
  if ((e = vpCurrentMatrix(vpc, VP_MODEL)) != VP_OK) FATAL_VP_ERROR("vpCurrentMatrix", e);
  if ((e = vpSetMatrix(vpc, vpm)) != VP_OK) FATAL_VP_ERROR("vpSetMatrix", e);
  vvDebugMsg::msg(2, "vvRenderVP::initialize(): model view matrix set");

  // Set projection matrix:
  if ((e = vpCurrentMatrix(vpc, VP_PROJECT)) != VP_OK) FATAL_VP_ERROR("vpCurrentMatrix", e);
  vpIdentityMatrix(vpc);
  if ((e = vpWindow(vpc, VP_PARALLEL, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)) != VP_OK) FATAL_VP_ERROR("vpWindow", e);
  vvDebugMsg::msg(2, "vvRenderVP::initialize(): projection matrix set");

  // Set output image size:
  setImageSize(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT);
  vvDebugMsg::msg(2, "vvRenderVP::initialize(): image size set to ", DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT);
}

//----------------------------------------------------------------------------
/** Set output image size. 
  Can be called multiple times.
*/
void vvRenderVP::setImageSize(int newWidth, int newHeight)
{
  static int imgWidth  = -1;  // current image size is stored to...
  static int imgHeight = -1;  // ...omit unnecessary changes
  vpResult e;

  vvDebugMsg::msg(3, "vvRenderVP::setImageSize() to: ", newWidth, newHeight);

  // Check if values are new:
  if (newWidth==imgWidth && newHeight==imgHeight && image!=NULL) return;

  // Update static variables:
  imgWidth  = newWidth;
  imgHeight = newHeight;

  // Resize image array:
  delete[] image;
  image = new uchar[imgWidth * imgHeight * COLOR_CHANNELS];
  e = vpSetImage(vpc, image, imgWidth, imgHeight, imgWidth * COLOR_CHANNELS, VP_RGB);
  if (e != VP_OK) FATAL_VP_ERROR("vpSetImage", e);
}

//----------------------------------------------------------------------------
/// Makes an unclassified volume from raw volume data.
void vvRenderVP::makeUnclassifiedVolume(uchar* raw)
{
  vpResult e;

  vvDebugMsg::msg(1, "vvRenderVP::makeUnclassifiedVolume()");
  assert(vpc != NULL);
  assert(raw != NULL);
  if (vd->getBPV() != 1) return; // only 1 byte per voxel can be processed
  
  vpSetVolumeSize(vpc, vd->vox[0], vd->vox[1], vd->vox[2]);

  delete[] vox;
  vox = new VoxelType[vd->getFrameBytes()];
  if (vox==NULL) FATAL_RENDERVP_ERROR("Not enough memory.");

  cerr << "Setting raw voxels array...";
  e = vpSetRawVoxels(vpc, vox, vd->getFrameBytes() * sizeof(VoxelType),
        sizeof(VoxelType), vd->vox[0] * sizeof(VoxelType),
        vd->vox[0] * vd->vox[1] * sizeof(VoxelType));
  if (e != VP_OK) FATAL_VP_ERROR("vpSetRawVoxels", e);
  cerr << "done." << endl;

  cerr << "Generating normal vectors...";
  if ((e = vpVolumeNormals(vpc, raw, vd->getFrameBytes(), 
    SCALAR_FIELD, GRAD_FIELD, NORM_FIELD)) != VP_OK) FATAL_VP_ERROR("vpVolumeNormals", e);
  cerr << "done." << endl;
}

//----------------------------------------------------------------------------
/// Classify a volume.
void vvRenderVP::classifyVolume()
{
  vpResult e;

  vvDebugMsg::msg(1, "vvRenderVP::classifyVolume()");
  if (vd->getBPV() != 1) return;

  cerr << "Classifying volume...";

  // Classify volume:
  if ((e = vpClassifyVolume(vpc)) != VP_OK) FATAL_VP_ERROR("vpClassifyVolume", e);

  cerr << "done." << endl;
}

//----------------------------------------------------------------------------
/** Upadte VolPack modelview matrix from OpenGL.
*/
void vvRenderVP::updateModelviewMatrix()
{
  double vpm[4][4];
  vvMatrix m;            // temporary matrix

  vvDebugMsg::msg(3, "vvRenderVP::updateModelviewMatrix()");

  // Get modelview matrix from OpenGL:
  vvMatrix mv = virvo::gl::getModelviewMatrix();

  // Set view matrix:
  vpCurrentMatrix(vpc, VP_VIEW);
  vpIdentityMatrix(vpc);

  // Set model view matrix:
  m = mv;               // copy model view matrix

  // Invert x axis rotation:
  m(1, 2) = -m(1, 2);
  m(2, 1) = -m(2, 1);

  // Invert y axis rotation:
  m(0, 2) = -m(0, 2);
  m(2, 0) = -m(2, 0);

  m.get(&vpm[0][0]);
  vpCurrentMatrix(vpc, VP_MODEL);
  vpSetMatrix(vpc, vpm);
}

//----------------------------------------------------------------------------
/** Update VolPack projection matrix from OpenGL.
*/
void vvRenderVP::updateProjectionMatrix()
{
  vvDebugMsg::msg(3, "vvRenderVP::updateProjectionMatrix()");

  vvMatrix pm = virvo::gl::getProjectionMatrix();

  if (pm.isProjOrtho()) // VolPack can only do parallel projections
  {
    pm.getProjOrtho(&left, &right, &bottom, &top, &nearPlane, &farPlane);
    vpCurrentMatrix(vpc, VP_PROJECT);
    vpIdentityMatrix(vpc);
    vpWindow(vpc, VP_PARALLEL, left, right, bottom, top, nearPlane, farPlane);    
  }
}

//----------------------------------------------------------------------------
/** Render internal image.
  @param w,h image size in pixels
*/
void vvRenderVP::renderVolume(int w, int h)
{
  vpResult e;

  // Update modelview matrix:
  updateModelviewMatrix();

  // Update projection matrix:
  updateProjectionMatrix();

  // Scale image to aspect ratio:
  vpCurrentMatrix(vpc, VP_MODEL);
  vpSeti(vpc, VP_CONCAT_MODE, VP_CONCAT_RIGHT);
  vpTranslate(vpc, vd->pos[0], vd->pos[1], vd->pos[2]);
  virvo::vec3f dist = vd->getDist();
  vpScale(vpc, dist[0]*vd->vox[0], -dist[1]*vd->vox[1], dist[2]*vd->vox[2]);  // adjust object to screen

  // Recompute shade table if light is moving:
  if (stickyLights == false)
    vpShadeTable(vpc);          // recompute illumination

  // Set image size:  
  setImageSize(w, h);

  // Render volume to VolPack internal image buffer:
  if ((e = vpRenderClassifiedVolume(vpc)) != VP_OK) FATAL_VP_ERROR("vpRenderClassifiedVolume", e);
}

//----------------------------------------------------------------------------
/// Render the volume onto currently selected drawBuffer.
void vvRenderVP::renderVolumeGL()
{
  vvColor defColor(1.f, 1.f, 1.f); // default boundary box color
  GLfloat glmatrix[16];   // OpenGL matrix
  GLint viewport[4];      // OpenGL viewport information (position and size)
  vvVector3 pos;          // object location
  vvStopwatch* sw = NULL; // stop watch
  int w, h;               // width and height of rendered image

  vvDebugMsg::msg(3, "vvRenderVP::renderVolumeGL()");

  if (vd->getBPV() != 1) return;   // only one byte per voxel is supported

  // Get OpenGL parameters:
  glGetIntegerv(GL_VIEWPORT, viewport);
  w = viewport[2];
  h = viewport[3];
  if (w<=0 || h<=0) return;   // safety first

  pos = vd->pos;

  // Check for projection type:
  vvMatrix pm = virvo::gl::getProjectionMatrix();
  if (pm.isProjOrtho())  // VolPack can only do parallel projections
  {
    if (timing)
    {
      sw = new vvStopwatch();
      sw->start();
    }

    // Render volume to VolPack internal image buffer:
    renderVolume(w, h);

    // Render volume to current OpenGL draw buffer:
    glGetFloatv(GL_MODELVIEW_MATRIX, glmatrix);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glRasterPos2f(left, bottom);   // pixmap origin is bottom left corner of output window
    glDrawPixels(w, h, GL_RGB, GL_UNSIGNED_BYTE, (GLvoid*)image);
    glLoadMatrixf(glmatrix);    // restore modelview matrix

    if (timing)
    {
      cerr << "Rendering time [ms]: " << setprecision(7) << ((float)sw->getTime() * 1000.0f) << endl;
      delete sw;
    }
  }
  if (_boundaries)
  {
    vvVector3 size = vd->getSize();
    drawBoundingBox(size, pos, defColor);   // draw boundaries
  }

  vvRenderer::renderVolumeGL();
}

//----------------------------------------------------------------------------
/** Render the volume to memory.
  @param w,h image size in pixels
  @param data _allocated_ memory space providing w*h*3 bytes of memory space
              to which the volume will be rendered
*/
void vvRenderVP::renderVolumeRGB(int w, int h, uchar* data)
{
  int x, y;
  int srcIndex, dstIndex;

  vvDebugMsg::msg(1, "vvRenderVP::renderVolumeRGB()");
  renderVolume(w, h); // render volume to internal image
  // Flip image vertically while copying pixels to data array:
  for (y=0; y<h; ++y)
    for (x=0; x<w; ++x)
    {
      srcIndex = 3 * (x + y * w);
      dstIndex = 3 * (x + (h - y - 1) * w);
      memcpy(&data[dstIndex], &image[srcIndex], 3);
    }
}

//----------------------------------------------------------------------------
/// Update transfer function from volume description.
void vvRenderVP::updateTransferFunction()
{
  int i, c;         // counters for index and colors
  double vpm[4][4]; // linear matrix arrangement
  vpResult e;
  vpMatrix4 m;
  vvMatrix unit;      // unit matrix
  float rgba[256][4];

  vvDebugMsg::msg(1, "vvRenderVP::updateTransferFunction()");

  // Generate arrays from pins:
  vd->computeTFTexture(256, 1, 1, &rgba[0][0]);

  // Copy color values:
  for (c=0; c<3; ++c)
    for (i=0; i<256; ++i)
      weight_table[i][c] = rgba[i][c];

  // Copy alpha values:
  for (i=0; i<256; ++i)
    scalar_table[i] = rgba[i][3];

  // Classify volume:
  classifyVolume();

  // Temporarily reset volume position for color table re-computation:
  vpGetMatrix(vpc, VP_MODEL, m);    // save current viewing matrix
  unit.identity();
  unit.get((double*)vpm);     // convert unit matrix to vp format
  vpCurrentMatrix(vpc, VP_MODEL);
  if ((e = vpSetMatrix(vpc, vpm)) != VP_OK) FATAL_VP_ERROR("vpSetMatrix", e);

  // Compute color table:
  if ((e = vpShadeTable(vpc)) != VP_OK) FATAL_VP_ERROR("vpShadeTable", e);

  // Set volume position back to original position:
  if ((e = vpSetMatrix(vpc, m)) != VP_OK) FATAL_VP_ERROR("vpSetMatrix", e);
}

//----------------------------------------------------------------------------
/** Set new frame index.
  @param index  new frame index (0 for first frame)
*/
void vvRenderVP::setCurrentFrame(size_t index)
{
  vvDebugMsg::msg(1, "vvRenderVP::setCurrentFrame()");
  if (index == vd->getCurrentFrame()) return;
  if (index >= vd->frames) index = vd->frames-1;
  VV_LOG(2) << "New frame index: " << index << std::endl;
  vd->setCurrentFrame(index);

  // Create new classified volume:
  makeUnclassifiedVolume(vd->getRaw());
  classifyVolume();
}

//----------------------------------------------------------------------------
/** Set light parameters.
 Fixed material characteristics are used with each setting.
 @param numLights number of lights in scene (0=ambient light only, 1=one light, ...)
 @param sticky    true if lights move with object, false if lights are stationary in world
*/
void vvRenderVP::setLights(int numLights, bool sticky)
{
  vvDebugMsg::msg(1, "vvRenderVP::setLights()");

  stickyLights = sticky;
  if (numLights==0) stickyLights = false;   // no lights -> moving not necessary

  // At least 2 lights:
  if (numLights >= 2)
    vpEnable(vpc, VP_LIGHT1, 1);
  else 
    vpEnable(vpc, VP_LIGHT1, 0);

  // At least one light:
  if (numLights >= 1) 
  {
    vpEnable(vpc, VP_LIGHT0, 1);
  
    vpSetMaterial(vpc, VP_MATERIAL0, VP_AMBIENT,   VP_BOTH_SIDES,  0.6, 0.0, 0.0);
    vpSetMaterial(vpc, VP_MATERIAL0, VP_DIFFUSE,   VP_BOTH_SIDES,  0.6, 0.0, 0.0);
    vpSetMaterial(vpc, VP_MATERIAL0, VP_SPECULAR,  VP_BOTH_SIDES,  0.8, 0.0, 0.0);
    vpSetMaterial(vpc, VP_MATERIAL0, VP_SHINYNESS, VP_BOTH_SIDES, 10.0, 0.0, 0.0);

    vpSetMaterial(vpc, VP_MATERIAL1, VP_AMBIENT,   VP_BOTH_SIDES,  0.0, 0.6, 0.0);
    vpSetMaterial(vpc, VP_MATERIAL1, VP_DIFFUSE,   VP_BOTH_SIDES,  0.0, 0.6, 0.0);
    vpSetMaterial(vpc, VP_MATERIAL1, VP_SPECULAR,  VP_BOTH_SIDES,  0.0, 0.8, 0.0);
    vpSetMaterial(vpc, VP_MATERIAL1, VP_SHINYNESS, VP_BOTH_SIDES, 10.0, 0.0, 0.0);
    
    vpSetMaterial(vpc, VP_MATERIAL2, VP_AMBIENT,   VP_BOTH_SIDES,  0.0, 0.0, 0.6);
    vpSetMaterial(vpc, VP_MATERIAL2, VP_DIFFUSE,   VP_BOTH_SIDES,  0.0, 0.0, 0.6);
    vpSetMaterial(vpc, VP_MATERIAL2, VP_SPECULAR,  VP_BOTH_SIDES,  0.0, 0.0, 0.8);
    vpSetMaterial(vpc, VP_MATERIAL2, VP_SHINYNESS, VP_BOTH_SIDES, 10.0, 0.0, 0.0);
  }
  else // no lights selected
  {
    vpEnable(vpc, VP_LIGHT0, 0);
    vpSetMaterial(vpc, VP_MATERIAL0, VP_AMBIENT, VP_BOTH_SIDES, 1.0, 0.0, 0.0);
    vpSetMaterial(vpc, VP_MATERIAL1, VP_AMBIENT, VP_BOTH_SIDES, 0.0, 1.0, 0.0);
    vpSetMaterial(vpc, VP_MATERIAL2, VP_AMBIENT, VP_BOTH_SIDES, 0.0, 0.0, 1.0);
  }

  vpShadeTable(vpc);    // activate new illumination settings
}

vvVolPack::vvVolPack(vvVolDesc *vd, vvRenderState rs)
  : vvSwitchRenderer<vvRenderVP, vvRenderer>(vd, rs)
{
  rendererType = VOLPACK;
}

#endif // HAVE_VOLPACK

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
