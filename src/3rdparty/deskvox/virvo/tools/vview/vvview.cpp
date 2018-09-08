//****************************************************************************
// Project:         Virvo (Virtual Reality Volume Renderer)
// Copyright:       (c) 1999-2004 Jurgen P. Schulze. All rights reserved.
// Author's E-Mail: schulze@cs.brown.edu
// Affiliation:     Brown University, Department of Computer Science
//****************************************************************************

// Do not automatically link with freeglut
#define FREEGLUT_LIB_PRAGMAS 0

#ifndef HLRS
#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif
#endif

#include <iostream>
#include <fstream>
#include <istream>
#include <sstream>
#include <string>
#include <iomanip>
using std::cerr;
using std::endl;
using std::ios;

#include <stdlib.h>
#include <stdio.h>
#ifdef __APPLE__


#include <AvailabilityMacros.h>

#if MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_9

  #pragma GCC diagnostic ignored "-Wdeprecated"

#endif

#include <GLUT/glut.h>

#else // __APPLE__
#include <GL/glut.h>
#ifdef FREEGLUT
#include <GL/freeglut_ext.h>
#endif
#endif
#include <time.h>
#include <assert.h>
#include <math.h>

#include <virvo/vvplatform.h>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include <virvo/vvvirvo.h>
#include <virvo/vvrequestmanagement.h>
#include <virvo/vvtoolshed.h>
#include <virvo/vvclock.h>
#include <virvo/vvrendererfactory.h>
#include <virvo/vvsocketmap.h>
#include <virvo/vvsocketio.h>
#include <virvo/vvtcpsocket.h>
#include <virvo/vvfileio.h>
#include <virvo/vvdebugmsg.h>
#include <virvo/private/vvlog.h>

#ifdef HAVE_BONJOUR
#include <virvo/vvbonjour/vvbonjour.h>
#endif

#include <virvo/private/vvgltools.h>

#include "vvobjview.h"
#include "vvperformancetest.h"
#include "vvview.h"

const int vvView::ROT_TIMER_DELAY = 20;
const int vvView::DEFAULTSIZE = 512;
const float vvView::OBJ_SIZE  = 1.0f;
const int vvView::DEFAULT_PORT = 31050;
vvView* vvView::ds = NULL;

using virvo::mat4;
using virvo::vec3f;
using virvo::vec3;
using virvo::vec4;


//----------------------------------------------------------------------------
/// Constructor
vvView::vvView()
{
  lastWidth = lastHeight = DEFAULTSIZE;
  winWidth = winHeight = DEFAULTSIZE;
  lastPosX = lastPosY = 50;
  pressedButton = NO_BUTTON;
  ds = this;
  renderer = NULL;
  vd = NULL;
  ov = NULL;
  currentRenderer = "default";
  bgColor[0] = bgColor[1] = bgColor[2] = 0.0f;
  frame = 0;
  filename = NULL;
  window = 0;
  draftQuality = 1.0f;
  highQuality = 5.0f;
  onOff[0] = "off";
  onOff[1] = "on";
  pos.zero();
  animating             = false;
  rotating              = false;
  activeStereoCapable   = false;
  tryQuadBuffer         = false;
  boundariesMode        = false;
  orientationMode       = false;
  fpsMode               = false;
  stereoMode            = 0;
  fullscreenMode        = false;
  filter_mode          =  virvo::Linear;
  warpInterpolMode      = true;
  preintMode            = false;
  paletteMode           = false;
  emptySpaceLeapingMode = false;
  earlyRayTermination   = false;
  perspectiveMode       = true;
  timingMode            = false;
  opCorrMode            = true;
  gammaMode             = false;
  mipMode               = 0;
  rotationMode          = false;
  refinement            = false;
  hqMode                = false;
  animSpeed             = 1.0f;
  iconMode              = false;
  isectType             = 0;
  bricks                = 1;
  useOffscreenBuffer    = false;
  bufferPrecision       = 8;
  useHeadLight          = false;
  ibrPrecision          = 8;
  ibrMode               = vvRenderer::VV_GRADIENT;
  sync                  = false;
  codec                 = vvImage::VV_RLE;
  rrMode                = RR_NONE;
  clipBuffer            = NULL;
  framebufferDump       = NULL;
  benchmark             = false;
  testSuiteFileName     = NULL;
  showBricks            = false;
  recordMode            = false;
  playMode              = false;
  matrixFile            = NULL;
  roiEnabled            = false;
  sphericalROI          = false;
  clipMode              = false;
  clipPerimeter         = false;
  mvScale               = 1.0f;
  showBt                = true;
  ibrValidation         = true;
}


//----------------------------------------------------------------------------
/// Destructor.
vvView::~vvView()
{
  if (recordMode && matrixFile)
  {
    fclose(matrixFile);
  }

  delete renderer;
  delete ov;
  delete vd;
  ds = NULL;

  for (std::vector<vvSocket*>::const_iterator it = sockets.begin();
       it != sockets.end(); ++it)
  {
    vvSocketMap::remove(vvSocketMap::getIndex(*it));
    delete *it;
  }
  sockets.clear();
}

//----------------------------------------------------------------------------
/** VView main loop.
  @param filename  volume file to display
*/
void vvView::mainLoop(int argc, char *argv[])
{
  vvDebugMsg::msg(2, "vvView::mainLoop()");

  if (filename!=NULL && strlen(filename)==0) filename = NULL;
  if (filename!=NULL)
  {
    vd = new vvVolDesc(filename);
    cerr << "Loading volume file: " << filename << endl;
  }
  else
  {
    vd = new vvVolDesc();
    cerr << "Using default volume" << endl;
  }

  vvFileIO fio;
  if (fio.loadVolumeData(vd) != vvFileIO::OK)
  {
    cerr << "Error loading volume file" << endl;
    delete vd;
    vd = NULL;
    return;
  }
  else vd->printInfoLine();

  // Set default color scheme if no TF present:
  if (vd->tf[0].isEmpty())
  {
    vd->tf[0].setDefaultAlpha(0, 0.0, 1.0);
    vd->tf[0].setDefaultColors((vd->getChan()==1) ? 0 : 2, 0.0, 1.0);
  }

  if (servers.size() == 0 && rrMode != RR_NONE)
  {
#ifdef HAVE_BONJOUR
    vvBonjour bonjour;
    servers = bonjour.getConnectionStringsFor("_distrendering._tcp");
#endif
  }

  initGraphics(argc, argv);

  if (rrMode == RR_COMPARISON)
  {
    currentRenderer = "comparison";
  }
  else if (rrMode == RR_IBR)
  {
    currentRenderer = "ibr";
  }
  else if(rrMode == RR_IMAGE)
  {
    currentRenderer = "image";
  }
  
  createRenderer(currentRenderer, currentOptions);

  vec3 size = vd->getSize();
  const float maxedge = ts_max(size[0], size[1], size[2]);

  mvScale = 1.0f / maxedge;
  cerr << "Scale modelview matrix by " << mvScale << endl;

  animSpeed = vd->getDt();
  createMenus();

  ov = new vvObjView();
  ds->ov->mv.scaleLocal(mvScale);

  setProjectionMode(perspectiveMode);

  // Set window title:
  if (filename!=NULL) glutSetWindowTitle(filename);

  srand(time(NULL));
  if(benchmark)
  {
    glutTimerFunc(1, timerCallback, BENCHMARK_TIMER);
  }

  if (playMode)
  {
    renderMotion();
  }
  else
  {
    glutMainLoop();
  }

  delete vd;
  vd = NULL;
}


//----------------------------------------------------------------------------
/** Callback method for window resizes.
    @param w,h new window width and height
*/
void vvView::reshapeCallback(int w, int h)
{
  vvDebugMsg::msg(2, "vvView::reshapeCallback(): ", w, h);

  ds->winWidth  = w;
  ds->winHeight = h;

  // Resize OpenGL viewport:
  glViewport(0, 0, ds->winWidth, ds->winHeight);

  // Set new aspect ratio:
  if (ds->winHeight > 0 && ds->ov) ds->ov->setAspectRatio((float)ds->winWidth / (float)ds->winHeight);

  glDrawBuffer(GL_FRONT_AND_BACK);               // select all buffers
                                                 // set clear color
  glClearColor(ds->bgColor[0], ds->bgColor[1], ds->bgColor[2], 1.0f);
                                                 // clear window
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}


//----------------------------------------------------------------------------
/// Callback method for window redraws.
void vvView::displayCallback(void)
{
  vvDebugMsg::msg(3, "vvView::displayCallback()");

  vvGLTools::printGLError("enter vvView::displayCallback()");

  glDrawBuffer(GL_BACK);
  glClearColor(ds->bgColor[0], ds->bgColor[1], ds->bgColor[2], 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  if(!ds->renderer)
    return;

  ds->renderer->setParameter(vvRenderState::VV_QUALITY, ((ds->hqMode) ? ds->highQuality : ds->draftQuality));

  ds->renderer->setParameter(vvRenderer::VV_FPS_DISPLAY, ds->fpsMode);

  // Draw volume:
  glMatrixMode(GL_MODELVIEW);
  if (ds->stereoMode>0)                          // stereo mode?
  {
    if (ds->stereoMode==1)                      // active stereo?
    {
      // Draw right image:
      glDrawBuffer(GL_BACK_RIGHT);
      ds->ov->updateModelviewMatrix(vvObjView::RIGHT_EYE);
      ds->renderer->renderVolumeGL();

      // Draw left image:
      glDrawBuffer(GL_BACK_LEFT);
      ds->ov->updateModelviewMatrix(vvObjView::LEFT_EYE);
      ds->renderer->renderVolumeGL();
    }
    // passive stereo?
    else if (ds->stereoMode==2 || ds->stereoMode==3)
    {
      ds->ov->setAspectRatio((float)ds->winWidth / 2 / (float)ds->winHeight);
      for (int i=0; i<2; ++i)
      {
        // Specify eye to draw:
        if (i==0) ds->ov->updateModelviewMatrix(vvObjView::LEFT_EYE);
        else      ds->ov->updateModelviewMatrix(vvObjView::RIGHT_EYE);

        // Specify where to draw it:
        if ((ds->stereoMode==2 && i==0) || (ds->stereoMode==3 && i==1))
        {
          // right
          glViewport(ds->winWidth / 2, 0, ds->winWidth / 2, ds->winHeight);
        }
        else
        {
          // left
          glViewport(0, 0, ds->winWidth / 2, ds->winHeight);
        }
        ds->renderer->renderVolumeGL();
      }

      // Reset viewport and aspect ratio:
      glViewport(0, 0, ds->winWidth, ds->winHeight);
      ds->ov->setAspectRatio((float)ds->winWidth / (float)ds->winHeight);
    }
  }
  else                                           // mono mode
  {
    glDrawBuffer(GL_BACK);
    ds->ov->updateModelviewMatrix(vvObjView::LEFT_EYE);
    ds->renderer->renderFrame();
  }

  if (ds->iconMode)
  {
    if (ds->vd->iconSize>0)
    {
      glMatrixMode(GL_PROJECTION);
      glPushMatrix();
      glLoadIdentity();
      glMatrixMode(GL_MODELVIEW);
      glPushMatrix();
      glLoadIdentity();
      glRasterPos2f(-1.0f, -0.0f);
      glPixelZoom(1.0f, -1.0f);
      glDrawPixels(ds->vd->iconSize, ds->vd->iconSize, GL_RGBA, GL_UNSIGNED_BYTE, ds->vd->iconData);
      glPopMatrix();
      glMatrixMode(GL_PROJECTION);
      glPopMatrix();
      glMatrixMode(GL_MODELVIEW);
    }
    else
    {
      cerr << "No icon stored" << endl;
    }
  }

  if (ds->recordMode)
  {
    ds->ov->saveMV(ds->matrixFile);
    // store time since program start
    fprintf(ds->matrixFile, "# %f\n", ds->stopWatch.getTime());
  }
  glutSwapBuffers();

  vvGLTools::printGLError("leave vvView::displayCallback()");
}


//----------------------------------------------------------------------------
/** Callback method for mouse button actions.
    @param button clicked button ID (left, middle, right)
    @param state  new button state (up or down)
    @param x,y    new mouse coordinates
*/
void vvView::buttonCallback(int button, int state, int x, int y)
{
  vvDebugMsg::msg(3, "vvView::buttonCallback()");

  const int ROTATION_THRESHOLD = 4;              // empirical threshold for auto rotation
  int dist, dx, dy;

  if (state==GLUT_DOWN)
  {
    ds->hqMode = false;
    switch (button)
    {
    case GLUT_LEFT_BUTTON:
      ds->pressedButton |= LEFT_BUTTON;
      ds->rotating = false;
      break;
    case GLUT_MIDDLE_BUTTON:
      ds->pressedButton |= MIDDLE_BUTTON;
      break;
    case GLUT_RIGHT_BUTTON:
      ds->pressedButton |= RIGHT_BUTTON;
      break;
    default: break;
    }
    ds->curX = ds->lastX = x;
    ds->curY = ds->lastY = y;
  }
  else if (state==GLUT_UP)
  {
    if (ds->refinement) ds->hqMode = true;
    switch (button)
    {
    case GLUT_LEFT_BUTTON:
                                                // only do something if button was pressed before
      if ((ds->pressedButton & LEFT_BUTTON) != 0)
      {
        ds->pressedButton &= ~LEFT_BUTTON;

        // Compute length of last mouse movement:
        dx = ts_abs(ds->curX - ds->lastX);
        dy = ts_abs(ds->curY - ds->lastY);
        dist = int(sqrt(float(dx*dx + dy*dy)));

                                                // auto-rotate if threshold was exceeded
        if (dist > ROTATION_THRESHOLD && ds->rotationMode)
        {
          ds->rotating = true;
          ds->x1 = ds->lastX;
          ds->y1 = ds->lastY;
          ds->x2 = ds->curX;
          ds->y2 = ds->curY;
          glutTimerFunc(ROT_TIMER_DELAY, timerCallback, ROTATION_TIMER);
          ds->hqMode = false;
        }
      }
      break;
    case GLUT_MIDDLE_BUTTON:
      ds->pressedButton &= ~MIDDLE_BUTTON;
      break;
    case GLUT_RIGHT_BUTTON:
      ds->pressedButton &= ~RIGHT_BUTTON;
      break;
    default: break;
    }
    if (ds->refinement) glutPostRedisplay();
  }
}


//----------------------------------------------------------------------------
/** Callback for mouse motion.
    @param x,y new mouse coordinates
*/
void vvView::motionCallback(int x, int y)
{
  vvDebugMsg::msg(3, "vvView::motionCallback()");

  int dx, dy;
  float factor;

  ds->lastX = ds->curX;                          // save current mouse coordinates for next call
  ds->lastY = ds->curY;
  ds->curX = x;
  ds->curY = y;
  dx = ds->curX - ds->lastX;
  dy = ds->curY - ds->lastY;

  switch (ds->pressedButton)
  {
  case LEFT_BUTTON:
    ds->ov->trackballRotation(ds->winWidth, ds->winHeight,
                              ds->lastX, ds->lastY, ds->curX, ds->curY);
    break;

  case MIDDLE_BUTTON:
  case LEFT_BUTTON | RIGHT_BUTTON:
    if (ds->perspectiveMode==false)
      ds->ov->mv.translate((float)dx * 0.01f, -(float)dy * 0.01f, 0.0f);
    else
      ds->ov->mv.translate(0.0f, 0.0f, (float)dy / 10.0f);
    break;

  case RIGHT_BUTTON:
    factor = 1.0f + ((float)dy) * 0.01f;
    if (factor > 2.0f) factor = 2.0f;
    if (factor < 0.5f) factor = 0.5f;
    ds->ov->mv.scaleLocal(factor, factor, factor);
    break;

  default: break;
  }

  glutPostRedisplay();
}


void vvView::applyRendererParameters()
{
  renderer->setParameter(vvRenderState::VV_BOUNDARIES, boundariesMode);
  renderer->setPosition(pos);
  renderer->setParameter(vvRenderState::VV_SLICEINT, filter_mode);
  renderer->setParameter(vvRenderer::VV_WARPINT, warpInterpolMode);
  renderer->setParameter(vvRenderer::VV_PREINT, preintMode);
  renderer->setParameter(vvRenderState::VV_MIP_MODE, mipMode);
  renderer->setParameter(vvRenderState::VV_QUALITY, (hqMode) ? highQuality : draftQuality);
  renderer->setParameter(vvRenderer::VV_LEAPEMPTY, emptySpaceLeapingMode);
  renderer->setParameter(vvRenderer::VV_TERMINATEEARLY, earlyRayTermination);
  renderer->setParameter(vvRenderer::VV_LIGHTING, useHeadLight);
  renderer->setParameter(vvRenderer::VV_ISECT_TYPE, isectType);
  renderer->setParameter(vvRenderer::VV_OFFSCREENBUFFER, useOffscreenBuffer);
  renderer->setParameter(vvRenderer::VV_IMG_PRECISION, bufferPrecision);
  renderer->setParameter(vvRenderState::VV_SHOW_BRICKS, showBricks);
  renderer->setParameter(vvRenderState::VV_CODEC, codec);

  renderer->setParameter(vvRenderState::VV_IBR_SYNC, sync);
  renderer->setParameter(vvRenderer::VV_IBR_DEPTH_PREC, ibrPrecision);
  if(rrMode == RR_IBR)
    renderer->setParameter(vvRenderer::VV_USE_IBR, true);

  renderer->setROIEnable(roiEnabled);
  printROIMessage();
}


//----------------------------------------------------------------------------
/** Set active rendering algorithm.
 */
void vvView::createRenderer(std::string type, const vvRendererFactory::Options &options,
                            size_t maxBrickSizeX, size_t maxBrickSizeY, size_t maxBrickSizeZ)
{
  vvDebugMsg::msg(3, "vvView::setRenderer()");

  vvGLTools::enableGLErrorBacktrace(ds->showBt);

  if (rrMode == RR_NONE && servers.size() == 1)
  {
    rrMode = RR_IBR;
    type = "ibr";
  }
  else if (rrMode == RR_NONE && servers.size() > 1)
  {
    rrMode = RR_PARBRICK;
    type = "parbrick";
  }

  ds->currentRenderer = type;
  ds->currentOptions = options;

  vvRendererFactory::Options opt(options);

  // clear socketmap from already created sockets
  for (std::vector<vvSocket*>::const_iterator it = sockets.begin();
       it != sockets.end(); ++it)
  {
    vvSocketMap::remove(vvSocketMap::getIndex(*it));
    delete *it;
  }
  sockets.clear();

  // sockets for remote renderers
  std::stringstream sockstr;
  std::vector<std::string>::const_iterator sit;
  std::vector<int>::const_iterator pit;
  for (sit = servers.begin(), pit = ports.begin();
       sit != servers.end() && pit != ports.end();
       ++sit, ++pit)
  {
    vvTcpSocket* sock = new vvTcpSocket;
    if (sock->connectToHost(*sit, static_cast<ushort>(*pit)) == vvSocket::VV_OK)
    {
      sock->setParameter(vvSocket::VV_NO_NAGLE, true);
      int s = vvSocketMap::add(sock);
      if (sockstr.str() != "")
      {
        sockstr << ",";
      }
      sockstr << s;
      sockets.push_back(sock);
    }
    else
    {
      delete sock;
    }
  }

  if (sockstr.str() != "")
  {
    opt["sockets"] = sockstr.str();
  }

  // file names
  std::stringstream filenamestr;
  for (std::vector<std::string>::const_iterator it = serverFileNames.begin();
       it != serverFileNames.end(); ++it)
  {
    if (it != serverFileNames.begin())
    {
      filenamestr << ",";
    }
    filenamestr << *it;
  }

  if (filenamestr.str() != "")
  {
    opt["filename"] = filenamestr.str();
  }

  // displays
  std::stringstream displaystr;
  for (std::vector<std::string>::const_iterator it = displays.begin();
       it != displays.end(); ++it)
  {
    if (it != displays.begin())
    {
      displaystr << ",";
    }
    displaystr << *it;
  }

  if (displaystr.str() != "")
  {
    opt["displays"] = displaystr.str();
  }

  std::stringstream brickstr;
  brickstr << bricks;
  opt["bricks"] = brickstr.str();

  if (displays.size() > 0 && servers.size() == 0)
  {
    opt["brickrenderer"] = "planar";
  }
  else if (servers.size() > 0)
  {
    opt["brickrenderer"] = "image";
  }

  for(size_t i = 0; i<sockets.size();i++)
  {
    vvSocketIO io = vvSocketIO(sockets[i]);

    // vserver defaults to image remote rendering
    if (rrMode == RR_IBR)
    {
      if (io.putEvent(virvo::RemoteServerType) == vvSocket::VV_OK)
      {
        io.putRendererType(vvRenderer::REMOTE_IBR);
      }
    }

    /* uncomment to test GpuInfo-event
    io.putInt32(virvo::GpuInfo);
    std::vector<vvGpu::vvGpuInfo> ginfos;
    io.getGpuInfos(ginfos);
    std::cerr << "Number of gpus: " << ginfos.size() << std::endl;
    for(std::vector<vvGpu::vvGpuInfo>::iterator ginfo = ginfos.begin(); ginfo != ginfos.end();ginfo++)
    {
      std::cerr << "free memory on server: "  << (*ginfo).freeMem << std::endl;
      std::cerr << "total memory on server: " << (*ginfo).totalMem << std::endl;
    }
    */

    /* uncomment to test statistics-event
    io.putEvent(virvo::Statistics);
    float wload;
    io.getFloat(wload);
    int resCount;
    io.getInt32(resCount);
    std::cerr << "Total work-load " << wload << " caused with " << resCount << " resources." << endl;
    */

  }

  if(renderer)
  renderState = *renderer;
  delete renderer;
  renderer = NULL;
  virvo::vector< 3, size_t > maxBrickSize(maxBrickSizeX, maxBrickSizeY, maxBrickSizeZ);
  renderState.setParameter(vvRenderState::VV_MAX_BRICK_SIZE, maxBrickSize);

  renderer = vvRendererFactory::create(vd, renderState, type.c_str(), opt);

  //static_cast<vvTexRend *>(renderer)->setTexMemorySize( 4 );
  //static_cast<vvTexRend *>(renderer)->setComputeBrickSize( false );
  //static_cast<vvTexRend *>(renderer)->setBrickSize( 64 );
  applyRendererParameters();
}


//----------------------------------------------------------------------------
/** Callback method for keyboard action.
    @param key ASCII code of pressed key
*/
void vvView::keyboardCallback(unsigned char key, int, int)
{
  vvDebugMsg::msg(3, "vvView::keyboardCallback()");

  switch((char)key)
  {
  case '0':
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9': ds->rendererMenuCallback(key - '0'); break;
  case '-': ds->rendererMenuCallback(98); break;
  case '+':
  case '=': ds->rendererMenuCallback(99); break;
  case 'a': ds->animMenuCallback(2);  break;
  case 'A': ds->optionsMenuCallback(14); break;
  case 'B': ds->optionsMenuCallback(12); break;
  case 'b': ds->viewMenuCallback(0);  break;
  case 'c': ds->viewMenuCallback(10); break;
  case 'C': ds->optionsMenuCallback(18);  break;
  case 'd': ds->mainMenuCallback(5);  break;
  case 'D': ds->mainMenuCallback(13);  break;
  case 'e': ds->mainMenuCallback(4);  break;
  case 'E': ds->clipMenuCallback(1); break;
  case 'f': ds->viewMenuCallback(2);  break;
  case 'g': ds->optionsMenuCallback(13);  break;
  case 'H': ds->optionsMenuCallback(8); break;
  case 'h': ds->optionsMenuCallback(9); break;
  case 'i': ds->optionsMenuCallback(0);  break;
  case 'I': ds->clipMenuCallback(0); break;
  case 'j': ds->transferMenuCallback(19); break;
  case 'J': ds->transferMenuCallback(20); break;
  case 'k': ds->transferMenuCallback(21); break;
  case 'K': ds->transferMenuCallback(22); break;
  case 'l': ds->transferMenuCallback(23); break;
  case 'L': ds->transferMenuCallback(24); break;
  case 'm': ds->mainMenuCallback(8);  break;
  case 'n': ds->animMenuCallback(0);  break;
  case 'N': ds->animMenuCallback(1);  break;
  case 'o': ds->viewMenuCallback(1);  break;
  case 'p': ds->mainMenuCallback(0);  break;
  case 'P': ds->optionsMenuCallback(1);  break;
  case 27:                                    // Escape
  case 'q': ds->mainMenuCallback(12); break;
  case 'r': ds->mainMenuCallback(7);  break;
  case 'R': ds->roiMenuCallback(0); break;
  case 's': ds->animMenuCallback(4);  break;
  case 'S': ds->animMenuCallback(5);  break;
  case 't': ds->mainMenuCallback(11); break;
  case 'T': ds->optionsMenuCallback(19); break;
  case 'u': ds->viewMenuCallback(8);  break;
  case 'v': ds->viewMenuCallback(9);  break;
  case 'w': ds->viewMenuCallback(6);  break;
  case 'W': ds->optionsMenuCallback(16);  break;
  case 'x':
    if (ds->clipEditMode)
    {
      ds->editClipPlane(PLANE_X, 0.05f);
    }
    else
    {
      ds->optionsMenuCallback(2);
    }
    break;
  case 'y':
    if (ds->clipEditMode)
    {
      ds->editClipPlane(PLANE_Y, 0.05f);
    }
    break;
  case 'z':
    if (ds->clipEditMode)
    {
      ds->editClipPlane(PLANE_Z, 0.05f);
    }
    else
    {
      ds->viewMenuCallback(5);
    }
    break;
  case '<': ds->transferMenuCallback(13);  break;
  case '>': ds->transferMenuCallback(14);  break;
  case '[': ds->roiMenuCallback(98); break;
  case ']': ds->roiMenuCallback(99); break;
  case '{': ds->optionsMenuCallback(11); break;
  case '}': ds->optionsMenuCallback(10); break;
  case '#': ds->optionsMenuCallback(17); break;
  case ' ': ds->optionsMenuCallback(8); break;
  default: cerr << "key '" << char(key) << "' has no function'" << endl; break;
  }
}


//----------------------------------------------------------------------------
/** Callback method for special keys.
    @param key ID of pressed key
*/
void vvView::specialCallback(int key, int, int)
{
  vvDebugMsg::msg(3, "vvView::specialCallback()");

  vec3f probePos = ds->renderer->getProbePosition();

  const int modifiers = glutGetModifiers();
  const float delta = 0.1f / ds->mvScale;

  vec3f clipPoint = ds->renderer->getParameter(vvRenderState::VV_CLIP_PLANE_POINT);
  vec3f clipNormal = ds->renderer->getParameter(vvRenderState::VV_CLIP_PLANE_NORMAL);

  switch(key)
  {
  case GLUT_KEY_LEFT:
    if (ds->roiEnabled)
    {
      probePos[0] -= delta;
      ds->renderer->setProbePosition(probePos);
    }
    else if (ds->clipEditMode)
    {
      ds->renderer->setParameter(vvRenderState::VV_CLIP_PLANE_POINT, clipPoint - clipNormal * delta);
    }
    break;
  case GLUT_KEY_RIGHT:
    if (ds->roiEnabled)
    {
      probePos[0] += delta;
      ds->renderer->setProbePosition(probePos);
    }
    else if (ds->clipEditMode)
    {
      ds->renderer->setParameter(vvRenderState::VV_CLIP_PLANE_POINT, clipPoint + clipNormal * delta);
    }
    break;
  case GLUT_KEY_UP:
    if (ds->roiEnabled)
    {
      if (modifiers & GLUT_ACTIVE_SHIFT)
      {
        probePos[2] += delta;
      }
      else
      {
        probePos[1] += delta;
      }
      ds->renderer->setProbePosition(probePos);
    }
    else if (ds->clipEditMode)
    {
      ds->renderer->setParameter(vvRenderState::VV_CLIP_PLANE_POINT, clipPoint + clipNormal * delta);
    }
    break;
  case GLUT_KEY_DOWN:
    if (ds->roiEnabled)
    {
      if (modifiers & GLUT_ACTIVE_SHIFT)
      {
        probePos[2] -= delta;
      }
      else
      {
        probePos[1] -= delta;
      }
      ds->renderer->setProbePosition(probePos);
    }
    else if (ds->clipEditMode)
    {
      ds->renderer->setParameter(vvRenderState::VV_CLIP_PLANE_POINT, clipPoint - clipNormal * delta);
    }
    break;
  default: break;
  }

  glutPostRedisplay();
}

void vvView::runTest()
{
  double tmin = DBL_MAX;
  double tmax = -DBL_MAX;
  double tavg = 0.0;

  ds->applyRendererParameters();

  // fill caches
  double cacherun = performanceTest();

  const int NUM_TESTS = 10;
  std::vector<double> ts;
  for(int i=0; i<NUM_TESTS; ++i)
  {
    double t = performanceTest();
    if(t < tmin)
      tmin = t;
    if(t > tmax)
      tmax = t;
    tavg += t;
    ts.push_back(t);
  }
  tavg /= (double)NUM_TESTS;

  double dev = 0.0;
  for (std::vector<double>::iterator it = ts.begin(); it != ts.end(); ++it)
  {
    dev += (*it - tavg) * (*it - tavg);
  }
  dev = std::sqrt(dev);

  fprintf(stdout, "%s: %f\n", "First run (ignored)", cacherun);
  fprintf(stdout, "%s: %i\n", "Test runs", NUM_TESTS);
  fprintf(stdout, "%s: %f\n", "tmax", tmax);
  fprintf(stdout, "%s: %f\n", "tmin", tmin);
  fprintf(stdout, "%s: %f\n", "tavg", tavg);
  fprintf(stdout, "%s: %f\n", "std deviation", dev);
}


//----------------------------------------------------------------------------
/** Timer callback method, triggered by glutTimerFunc().
  @param id  timer ID: 0=animation, 1=rotation
*/
void vvView::timerCallback(int id)
{
  vvDebugMsg::msg(3, "vvView::timerCallback()");

  switch(id)
  {
  case ANIMATION_TIMER:
    if (ds->animating)
    {
      ++ds->frame;
      ds->setAnimationFrame(ds->frame);
      glutTimerFunc(int(ds->animSpeed * 1000.0f), timerCallback, ANIMATION_TIMER);
    }
    break;
  case ROTATION_TIMER:
    if (ds->rotating)
    {
      ds->ov->trackballRotation(ds->winWidth, ds->winHeight,
                                ds->x1, ds->y1, ds->x2, ds->y2);
      glutPostRedisplay();
      glutTimerFunc(ROT_TIMER_DELAY, timerCallback, ROTATION_TIMER);
    }
    break;
  case BENCHMARK_TIMER:
    {
#if 0
        ds->vd->tf.setDefaultColors(0, 0.0, 1.0);
        ds->vd->tf.setDefaultAlpha(0, 0.0, 1.0);
        ds->setProjectionMode(false);

        ds->useOffscreenBuffer = false;
        ds->preintMode = false;
        ds->earlyRayTermination = false;
        ds->emptySpaceLeapingMode = false;
        ds->bufferPrecision = 8;

        // CUDA SW
        ds->createRenderer("cudasw");
        runTest();

        ds->preintMode = true;
        runTest();
        ds->preintMode = false;

        ds->earlyRayTermination = true;
        runTest();
        ds->earlyRayTermination = false;

        ds->emptySpaceLeapingMode = true;
        runTest();
        ds->emptySpaceLeapingMode = false;

        ds->useOffscreenBuffer = true;
        ds->bufferPrecision = 32;
        runTest();

        ds->preintMode = true;
        runTest();
        ds->preintMode = false;

        ds->earlyRayTermination = true;
        runTest();
        ds->earlyRayTermination = false;

        ds->emptySpaceLeapingMode = true;
        runTest();
        ds->emptySpaceLeapingMode = false;

        // RAYCAST
        ds->createRenderer("rayrend");
        runTest();

        ds->earlyRayTermination = true;
        runTest();
        ds->earlyRayTermination = false;

        // TEX2D
        ds->useOffscreenBuffer = false;
        ds->bufferPrecision = 8;
        ds->createRenderer("slices");
        runTest();

        ds->useOffscreenBuffer = true;
        ds->bufferPrecision = 32;
        runTest();

        // TEX3D
        ds->useOffscreenBuffer = false;
        ds->bufferPrecision = 8;
        ds->createRenderer("planar");
        runTest();

        ds->preintMode = true;
        runTest();
        ds->preintMode = false;

        ds->useOffscreenBuffer = true;
        ds->bufferPrecision = 32;
        runTest();

        ds->preintMode = true;
        runTest();
        ds->preintMode = false;

        // VOLPACK
        ds->createRenderer("volpack");
#endif
        runTest();
    }
    exit(0);
    break;
  default:
    break;
    }
}


//----------------------------------------------------------------------------
/** Callback for main menu.
  @param item selected menu item index
*/
void vvView::mainMenuCallback(int item)
{
  vvDebugMsg::msg(1, "vvView::mainMenuCallback()");

  switch (item)
  {
  case 0:                                     // projection mode
    ds->setProjectionMode(!ds->perspectiveMode);
    break;
  case 4:                                     // refinement mode
    if (ds->refinement) ds->refinement = false;
    else ds->refinement = true;
    ds->hqMode = ds->refinement;
    break;
  case 5:                                     // timing mode
    ds->timingMode = !ds->timingMode;
    //ds->renderer->setTimingMode(ds->timingMode);
    break;
  case 7:                                     // reset object position
    ds->ov->reset();
    ds->ov->mv.scaleLocal(ds->mvScale);
    ds->setProjectionMode(ds->perspectiveMode);
    break;
  case 8:                                     // menu/zoom mode
    if (ds->menuEnabled)
    {
      glutDetachMenu(GLUT_RIGHT_BUTTON);
      ds->menuEnabled = false;
    }
    else
    {
      glutSetMenu(ds->mainMenu);
      glutAttachMenu(GLUT_RIGHT_BUTTON);
      ds->menuEnabled = true;
    }
    break;
  case 9:                                     // save volume with transfer function
    vvFileIO* fio;
    ds->vd->setFilename("virvo-saved.xvf");
    fio = new vvFileIO();
    fio->saveVolumeData(ds->vd, true);
    delete fio;
    cerr << "Volume saved to file 'virvo-saved.xvf'." << endl;
    break;
  case 11:                                    // performance test
    performanceTest();
    break;
  case 12:                                    // quit
    glutDestroyWindow(ds->window);
#ifdef FREEGLUT
    return;
#else
    delete ds;
    exit(0);
    break;
#endif
  case 13:                                    // rotate debug level
    {
        int l = vvDebugMsg::getDebugLevel()+1;
        l %= 4;
        vvDebugMsg::setDebugLevel(l);
    }
    break;
  default: break;
  }
  glutPostRedisplay();
}


//----------------------------------------------------------------------------
/** Callback for renderer menu.
  @param item selected menu item index
*/
void vvView::rendererMenuCallback(int item)
{
  vvDebugMsg::msg(1, "vvView::rendererMenuCallback()");

  const float QUALITY_CHANGE = 1.05f;            // quality modification unit
  const char QUALITY_NAMES[2][6] =               // quality mode names
  {
    "Draft", "High"
  };

  if (item>=0 && item<=16)
  {
    char type[100];
    sprintf(type, "%d", item);

    ds->createRenderer(type, ds->currentOptions);
  }
  else if (item==98 || item==99)
  {
    if (item==98)
    {
      ((ds->hqMode) ? ds->highQuality : ds->draftQuality) /= QUALITY_CHANGE;
    }
    else if (item==99)
    {
      ((ds->hqMode) ? ds->highQuality : ds->draftQuality) *= QUALITY_CHANGE;
    }
    ds->renderer->setParameter(vvRenderState::VV_QUALITY, (ds->hqMode) ? ds->highQuality : ds->draftQuality);
    cerr << QUALITY_NAMES[ds->hqMode] << " quality: " <<
        ((ds->hqMode) ? ds->highQuality : ds->draftQuality) << endl;
  }
  glutPostRedisplay();
}


//----------------------------------------------------------------------------
/** Callback for voxel type menu.
  @param item selected menu item index
*/
void vvView::voxelMenuCallback(int item)
{
  vvDebugMsg::msg(1, "vvView::voxelMenuCallback()");

  switch(item)
  {
  case 1:
    ds->currentOptions["voxeltype"] = "rgba";
    break;
  case 2:
    ds->currentOptions["voxeltype"] = "sgilut";
    break;
  case 3:
    ds->currentOptions["voxeltype"] = "paltex";
    break;
  case 4:
    ds->currentOptions["voxeltype"] = "regcomb";
    break;
  case 5:
    ds->currentOptions["voxeltype"] = "shader";
    break;
  case 6:
    ds->currentOptions["voxeltype"] = "arb";
    break;
  case 0:
  default:
    ds->currentOptions["voxeltype"] = "default";
    break;
  }

  ds->createRenderer(ds->currentRenderer, ds->currentOptions);

  glutPostRedisplay();
}


//----------------------------------------------------------------------------
/** Callback for options menu.
  @param item selected menu item index
*/
void vvView::optionsMenuCallback(int item)
{
  vec3 size(0.0f, 0.0f, 0.0f);

  vvDebugMsg::msg(1, "vvView::optionsMenuCallback()");

  switch(item)
  {
  case 0:                                     // slice interpolation mode
    ds->filter_mode = ds->filter_mode == virvo::Nearest
      ? virvo::Linear : virvo::Nearest;
    ds->renderer->setParameter(vvRenderer::VV_SLICEINT, ds->filter_mode);
    cerr << "Interpolation mode set to " << int(ds->filter_mode) << endl;
    break;
  case 1:
    ds->preintMode = !ds->preintMode;
    ds->renderer->setParameter(vvRenderer::VV_PREINT, ds->preintMode);
    cerr << "Pre-integration set to " << int(ds->preintMode) << endl;
    break;
  case 2:                                     // min/maximum intensity projection
    ++ds->mipMode;
    if (ds->mipMode>2) ds->mipMode = 0;
    ds->renderer->setParameter(vvRenderState::VV_MIP_MODE, ds->mipMode);
    cerr << "MIP mode set to " << ds->mipMode << endl;
    break;
  case 3:                                     // opacity correction
    ds->opCorrMode = !ds->opCorrMode;
    ds->renderer->setParameter(vvRenderer::VV_OPCORR, ds->opCorrMode);
    cerr << "Opacity correction set to " << int(ds->opCorrMode) << endl;
    break;
  case 4:                                     // gamma correction
    ds->gammaMode = !ds->gammaMode;
    ds->renderer->setGamma(vvRenderer::VV_RED, 2.2f);
    ds->renderer->setGamma(vvRenderer::VV_GREEN, 2.2f);
    ds->renderer->setGamma(vvRenderer::VV_BLUE, 2.2f);
    //ds->renderer->setParameter(vvRenderer::VV_GAMMA, (ds->gammaMode) ? 1.0f : 0.0f);
    cerr << "Gamma correction set to " << int(ds->gammaMode) << endl;
    break;
  case 5:
    ds->emptySpaceLeapingMode = !ds->emptySpaceLeapingMode;
    ds->renderer->setParameter(vvRenderer::VV_LEAPEMPTY, ds->emptySpaceLeapingMode);
    cerr << "Empty space leaping set to " << int(ds->emptySpaceLeapingMode) << endl;
    break;
  case 6:
    ds->useOffscreenBuffer = !ds->useOffscreenBuffer;
    ds->renderer->setParameter(vvRenderer::VV_OFFSCREENBUFFER, ds->useOffscreenBuffer);
    cerr << "Offscreen Buffering set to " << int(ds->useOffscreenBuffer) << endl;
    break;
  case 7:
    ds->useHeadLight = !ds->useHeadLight;
    if(ds->useHeadLight)
    {
      vec3 eyePos = ds->renderer->getEyePosition();

      glEnable(GL_LIGHTING);
      glLightfv(GL_LIGHT0, GL_POSITION, &vvVector4(eyePos, 1.0f)[0]);
    }
    ds->renderer->setParameter(vvRenderer::VV_LIGHTING, ds->useHeadLight);
    break;
  case 8:                                     // increase z size
    //ds->renderer->getVoxelSize(&size);
    size[2] *= 1.05f;
    //ds->renderer->setVoxelSize(&size);
    cerr << "Z size set to " << size[2] << endl;
    break;
  case 9:                                     // decrease z size
    //ds->renderer->getVoxelSize(&size);
    size[2] *= 0.95f;
    //ds->renderer->setVoxelSize(&size);
    cerr << "Z size set to " << size[2] << endl;
    break;
  case 10:                                     // increase precision of visual
    if (ds->bufferPrecision == 8)
    {
      ds->bufferPrecision = 16;
      cerr << "Buffer precision set to 16bit" << endl;
    }
    else if (ds->bufferPrecision == 16)
    {
      ds->bufferPrecision = 32;
      cerr << "Buffer precision set to 32bit" << endl;
    }
    else
    {
      cerr << "Highest precision reached" << endl;
    }
    ds->renderer->setParameter(vvRenderer::VV_IMG_PRECISION, ds->bufferPrecision);
    break;
  case 11:                                    // decrease precision of visual
    if (ds->bufferPrecision == 32)
    {
      ds->bufferPrecision = 16;
      cerr << "Buffer precision set to 16bit" << endl;
    }
    else if (ds->bufferPrecision == 16)
    {
      ds->bufferPrecision = 8;
      cerr << "Buffer precision set to 8bit" << endl;
    }
    else
    {
      cerr << "Lowest precision reached" << endl;
    }
    ds->renderer->setParameter(vvRenderer::VV_IMG_PRECISION, ds->bufferPrecision);
    break;
  case 12:                                     // toggle showing of bricks
    ds->showBricks = !ds->showBricks;
    ds->renderer->setParameter(vvRenderState::VV_SHOW_BRICKS, ds->showBricks);
    cerr << (!ds->showBricks?"not ":"") << "showing bricks" << endl;
    break;
  case 13:
    ++ds->isectType;
    if (ds->isectType > 3)
    {
      ds->isectType = 0;
    }

    ds->renderer->setParameter(vvRenderer::VV_ISECT_TYPE, ds->isectType);

    cerr << "Switched to proxy geometry generation ";
    switch (ds->isectType)
    {
    case 0:
      cerr << "on the GPU, vertex shader";
      break;
    case 1:
      cerr << "on the GPU, geometry shader";
      break;
    case 2:
      cerr << "on the GPU, vertex shader and geometry shader";
      break;
    case 3:
      // fall-through
    default:
      cerr << "on the CPU";
      break;
    }
    cerr << endl;
    break;
  case 14:
    {
      int shader = ds->renderer->getParameter(vvRenderer::VV_PIX_SHADER);
      ++shader;
      ds->renderer->setParameter(vvRenderer::VV_PIX_SHADER, shader);
      cerr << "shader set to " << ds->renderer->getParameter(vvRenderer::VV_PIX_SHADER).asInt() << endl;
    }
    break;
  case 15:
    ds->earlyRayTermination = !ds->earlyRayTermination;
    ds->renderer->setParameter(vvRenderer::VV_TERMINATEEARLY, ds->earlyRayTermination);
    cerr << "Early ray termination set to " << int(ds->earlyRayTermination) << endl;
    break;
  case 16:
    ds->warpInterpolMode = !ds->warpInterpolMode;
    ds->renderer->setParameter(vvRenderer::VV_WARPINT, ds->warpInterpolMode);
    cerr << "Warp interpolation set to " << int(ds->warpInterpolMode) << endl;
    break;
  case 17:
    {
      std::map<vvRenderer::IbrMode, std::string> ibrMap;
      ibrMap[vvRenderer::VV_ENTRANCE] = "VV_ENTRANCE";
      ibrMap[vvRenderer::VV_EXIT] = "VV_EXIT";
      ibrMap[vvRenderer::VV_MIDPOINT] = "VV_MIDPOINT";
      ibrMap[vvRenderer::VV_THRESHOLD] = "VV_THRESHOLD";
      ibrMap[vvRenderer::VV_PEAK] = "VV_PEAK";
      ibrMap[vvRenderer::VV_GRADIENT] = "VV_GRADIENT";
      ibrMap[vvRenderer::VV_REL_THRESHOLD] = "VV_REL_THRESHOLD";
      ibrMap[vvRenderer::VV_EN_EX_MEAN] = "VV_EN_EX_MEAN";

      int tmp = ds->ibrMode;
      ++tmp;
      ds->ibrMode = static_cast<vvRenderState::IbrMode>(tmp);
      if (ds->ibrMode == vvRenderState::VV_NONE)
      {
        ds->ibrMode = static_cast<vvRenderState::IbrMode>(0);
      }
      ds->renderer->setParameter(vvRenderer::VV_IBR_MODE, ds->ibrMode);
      cerr << "Set IBR mode to " << ibrMap[ds->ibrMode]
           << " (" << int(ds->ibrMode) << ")" << endl;
    }
    break;
  case 18:
    ++ds->codec;
    if(ds->codec > 10)
      ds->codec = 0;
    cerr << "Codec set to " << ds->codec << endl;
    ds->renderer->setParameter(vvRenderer::VV_CODEC, ds->codec);
    break;
  case 19:
    ds->sync = !ds->sync;
    ds->renderer->setParameter(vvRenderer::VV_IBR_SYNC, ds->sync ? 1.f : 0.f);
    break;
  default: break;
  }
  glutPostRedisplay();
}


//----------------------------------------------------------------------------
/** Callback for transfer function menu.
  @param item selected menu item index
*/
void vvView::transferMenuCallback(int item)
{
  static float peakPosX = 0.0f;
  float gamma, chan4;

  vvDebugMsg::msg(1, "vvView::transferMenuCallback()");

  switch(item)
  {
  case 0:                                     // bright colors
    ds->vd->tf[0].setDefaultColors(0, 0.0, 1.0);
    break;
  case 1:                                     // HSV colors
    ds->vd->tf[0].setDefaultColors(1, 0.0, 1.0);
    break;
  case 2:                                     // grayscale
    ds->vd->tf[0].setDefaultColors(2, 0.0, 1.0);
    break;
  case 3:                                     // white
    ds->vd->tf[0].setDefaultColors(3, 0.0, 1.0);
    break;
  case 4:                                     // white
    ds->vd->tf[0].setDefaultColors(4, 0.0, 1.0);
    break;
  case 5:                                     // white
    ds->vd->tf[0].setDefaultColors(5, 0.0, 1.0);
    break;
  case 6:                                     // white
    ds->vd->tf[0].setDefaultColors(6, 0.0, 1.0);
    break;
  case 7:                                     // cool to warm
    ds->vd->tf[0].setDefaultColors(7, 0.0, 1.0);
    break;
  case 8:
    ds->vd->tf[0].setDefaultColors(8, 0.0, 1.0);
    break;
  case 9:                                     // alpha: ascending
    ds->vd->tf[0].setDefaultAlpha(0, 0.0, 1.0);
    break;
  case 10:                                     // alpha: descending
    ds->vd->tf[0].setDefaultAlpha(1, 0.0, 1.0);
    break;
  case 11:                                     // alpha: opaque
    ds->vd->tf[0].setDefaultAlpha(2, 0.0, 1.0);
    break;
  case 12:                                    // alpha: display peak
  case 13:                                    // alpha: shift left peak
  case 14:                                    // alpha: shift right peak
    if(item == 13)
      peakPosX -= .05;
    else if(item == 14)
      peakPosX += .05;
    if (peakPosX < 0.0f) peakPosX += 1.0f;
    if (peakPosX > 1.0f) peakPosX -= 1.0f;
    cerr << "Peak position: " << peakPosX << endl;

    ds->vd->tf[0].deleteWidgets(vvTFWidget::TF_PYRAMID);
    ds->vd->tf[0].deleteWidgets(vvTFWidget::TF_BELL);
    ds->vd->tf[0].deleteWidgets(vvTFWidget::TF_CUSTOM);
    ds->vd->tf[0].deleteWidgets(vvTFWidget::TF_SKIP);
    ds->vd->tf[0]._widgets.push_back(new vvTFPyramid(vvColor(1.f, 1.f, 1.f), false, 1.f, peakPosX, .2f, 0.f));
    break;
  case 15:                                    // gamma red
  case 16:
    gamma = ds->renderer->getGamma(vvRenderer::VV_RED);
    gamma *= (item==15) ? 0.95f : 1.05f;
    ds->renderer->setGamma(vvRenderer::VV_RED, gamma);
    cerr << "gamma red = " << gamma << endl;
    break;
  case 17:                                    // gamma green
  case 18:
    gamma = ds->renderer->getGamma(vvRenderer::VV_GREEN);
    gamma *= (item==17) ? 0.95f : 1.05f;
    ds->renderer->setGamma(vvRenderer::VV_GREEN, gamma);
    cerr << "gamma green = " << gamma << endl;
    break;
  case 19:                                    // gamma blue
  case 20:
    gamma = ds->renderer->getGamma(vvRenderer::VV_BLUE);
    gamma *= (item==19) ? 0.95f : 1.05f;
    ds->renderer->setGamma(vvRenderer::VV_BLUE, gamma);
    cerr << "gamma blue = " << gamma << endl;
    break;
  case 21:                                    // channel 4 red
  case 22:
    chan4 = ds->renderer->getOpacityWeight(vvRenderer::VV_RED);
    chan4 *= (item==21) ? 0.95f : 1.05f;
    ds->renderer->setOpacityWeight(vvRenderer::VV_RED, chan4);
    cerr << "channel 4 red = " << chan4 << endl;
    break;
  case 23:                                    // channel 4 green
  case 24:
    chan4 = ds->renderer->getOpacityWeight(vvRenderer::VV_GREEN);
    chan4 *= (item==23) ? 0.95f : 1.05f;
    ds->renderer->setOpacityWeight(vvRenderer::VV_GREEN, chan4);
    cerr << "channel 4 green = " << chan4 << endl;
    break;
  case 25:                                    // channel 4 blue
  case 26:
    chan4 = ds->renderer->getOpacityWeight(vvRenderer::VV_BLUE);
    chan4 *= (item==25) ? 0.95f : 1.05f;
    ds->renderer->setOpacityWeight(vvRenderer::VV_BLUE, chan4);
    cerr << "channel 4 blue = " << chan4 << endl;
    break;
  default: break;
  }

  ds->renderer->updateTransferFunction();
  glutPostRedisplay();
}


//----------------------------------------------------------------------------
/** Display a specific animation frame.
    @param f Index of frame to display. First index = 0
*/
void vvView::setAnimationFrame(ssize_t f)
{
  vvDebugMsg::msg(3, "vvView::setAnimationFrame()");

  if (vd->frames > static_cast<size_t>(std::numeric_limits<ssize_t>::max()))
  {
    VV_LOG(0) << "Invalid animation frame" << std::endl;
    return;
  }

  if (f >= static_cast<ssize_t>(vd->frames))
    frame = 0;
  else if (f<0)
    frame = vd->frames - 1;

  renderer->setCurrentFrame(frame);

  cerr << "Time step: " << (frame+1) << endl;
  glutPostRedisplay();
}


//----------------------------------------------------------------------------
/** Callback for animation menu.
  @param item selected menu item index
*/
void vvView::animMenuCallback(int item)
{
   vvDebugMsg::msg(1, "vvView::animMenuCallback()");

  switch(item)
  {
  default:
  case 0:                                     // next frame
    ++ds->frame;
    ds->setAnimationFrame(ds->frame);
    break;
  case 1:                                     // previous frame
    --ds->frame;
    ds->setAnimationFrame(ds->frame);
    break;
  case 2:                                     // start/stop animation
    if (ds->animating) ds->animating = false;
    else
    {
      ds->animating = true;
                                            // trigger timer function
      glutTimerFunc(int(ds->ds->animSpeed * 1000.0f), timerCallback, ANIMATION_TIMER);
    }
    break;
  case 3:                                     // rewind
    ds->frame = 0;
    ds->setAnimationFrame(ds->frame);
    break;
  case 4:                                     // speed up
    ds->animSpeed *= 0.9f;
    cerr << "speed=" << ds->animSpeed << endl;
    break;
  case 5:                                     // speed down
    ds->animSpeed *= 1.1f;
    cerr << "speed=" << ds->animSpeed << endl;
    break;
  case 6:                                     // reset speed
    ds->animSpeed = ds->vd->getDt();
    break;
  }
}


//----------------------------------------------------------------------------
/** Callback for roi menu.
  @param item selected menu item index
*/
void vvView::roiMenuCallback(const int item)
{
  vvDebugMsg::msg(1, "vvView::roiMenuCallback()");

  vec3f probeSize;

  switch (item)
  {
  case 0:                                    // toggle roi mode
    if (!ds->roiEnabled && !ds->sphericalROI)
    {
      // Cuboid roi.
      ds->roiEnabled = true;
      ds->sphericalROI = false;
    }
    else if (ds->roiEnabled && !ds->sphericalROI)
    {
      // Spherical roi.
      ds->roiEnabled = true;
      ds->sphericalROI = true;
    }
    else
    {
      // No roi.
      ds->roiEnabled = false;
      ds->sphericalROI = false;
    }
    ds->renderer->setROIEnable(ds->roiEnabled);
    ds->renderer->setSphericalROI(ds->sphericalROI);
    printROIMessage();
    break;
  case 98:
    if (ds->roiEnabled)
    {
      probeSize = ds->renderer->getProbeSize();
      probeSize -= vec3f(0.1f);
      const float size = probeSize[0];
      if (size <= 0.0f)
      {
        probeSize = vec3(0.00001f);
      }
      ds->renderer->setProbeSize(probeSize);
    }
    else
    {
      cerr << "Function only available in ROI mode" << endl;
    }
    break;
  case 99:
    if (ds->roiEnabled)
    {
      probeSize = ds->renderer->getProbeSize();
      probeSize += vec3f(0.1f);
      const float size = probeSize[0];
      if (size > 1.0f)
      {
        probeSize = vec3(1.0f);
      }
      ds->renderer->setProbeSize(probeSize);
    }
    else
    {
      cerr << "Function only available in ROI mode" << endl;
    }
    break;
  default:
    break;
  }
  glutPostRedisplay();
}


//----------------------------------------------------------------------------
/** Callback for clip menu.
  @param item selected menu item index
*/
void vvView::clipMenuCallback(const int item)
{
  vvDebugMsg::msg(1, "vvView::clipMenuCallback()");

  switch (item)
  {
  case 0:
    ds->clipMode = !ds->clipMode;
    ds->renderer->setParameter(vvRenderState::VV_CLIP_MODE, ds->clipMode);
    cerr << "Clipping " << ds->onOff[ds->clipMode] << endl;
    break;
  case 1:
    ds->clipEditMode = !ds->clipEditMode;
    if (ds->clipEditMode)
    {
      ds->clipMode = true;
      ds->renderer->setParameter(vvRenderState::VV_CLIP_MODE, ds->clipMode);
      cerr << "Clip edit mode activated" << endl;
      cerr << "x|y|z keys:\t\trotation along (x|y|z) axis" << endl;
      cerr << "Arrow down/left:\tmove in negative normal direction" << endl;
      cerr << "Arrow up/right:\tmove in positive normal direction" << endl;
    }
    else
    {
      cerr << "Clip edit mode deactivated" << endl;
    }
    break;
  }
  glutPostRedisplay();
}


//----------------------------------------------------------------------------
/** Callback for viewing window menu.
  @param item selected menu item index
*/
void vvView::viewMenuCallback(int item)
{
  vvDebugMsg::msg(1, "vvView::viewMenuCallback()");

  switch(item)
  {
  default:
  case 0:                                     // bounding box
    ds->boundariesMode = !ds->boundariesMode;
    ds->renderer->setParameter(vvRenderState::VV_BOUNDARIES, ds->boundariesMode);
    cerr << "Bounding box " << ds->onOff[ds->boundariesMode] << endl;
    break;
  case 1:                                     // axis orientation
    ds->orientationMode = !ds->orientationMode;
    ds->renderer->setParameter(vvRenderState::VV_ORIENTATION,
                                 !ds->renderer->getParameter(vvRenderState::VV_ORIENTATION).asBool());
    cerr << "Coordinate axes display " << ds->onOff[ds->orientationMode] << endl;
    break;
  case 2:                                     // frame rate
    ds->fpsMode = !ds->fpsMode;
    ds->renderer->setParameter(vvRenderState::VV_FPS_DISPLAY, !ds->renderer->getParameter(vvRenderState::VV_FPS_DISPLAY));
    cerr << "Frame rate display " << ds->onOff[ds->fpsMode] << endl;
    break;
  case 3:                                     // transfer function
    ds->paletteMode = !ds->paletteMode;
    ds->renderer->setParameter(vvRenderState::VV_PALETTE, !ds->renderer->getParameter(vvRenderState::VV_PALETTE).asBool());
    cerr << "Palette display " << ds->onOff[ds->paletteMode] << endl;
    break;
  case 4:                                     // stereo mode
    ++ds->stereoMode;
    if (ds->stereoMode > 3) ds->stereoMode = 0;
    if (ds->stereoMode==1 && !ds->activeStereoCapable) ds->stereoMode = 2;
    cerr << "Stereo mode: " << ds->stereoMode << endl;
    break;
  case 5:                                     // full screen
    if (ds->fullscreenMode)
    {
      glutPositionWindow(ds->lastPosX, ds->lastPosY);
      glutReshapeWindow(ds->lastWidth, ds->lastHeight);
      ds->fullscreenMode = false;
    }
    else
    {
      ds->lastWidth  = glutGet(GLUT_WINDOW_WIDTH);
      ds->lastHeight = glutGet(GLUT_WINDOW_HEIGHT);
      ds->lastPosX   = glutGet(GLUT_WINDOW_X);
      ds->lastPosY   = glutGet(GLUT_WINDOW_Y);
      glutFullScreen();
      ds->fullscreenMode = true;
    }
    cerr << "Fullscreen mode " << ds->onOff[ds->fullscreenMode] << endl;
    break;
  case 6:                                     // window color
    if (ds->bgColor[0] == 0.0f)
                                        // white
      ds->bgColor[0] = ds->bgColor[1] = ds->bgColor[2] = 1.0f;
                                        // black
    else
      ds->bgColor[0] = ds->bgColor[1] = ds->bgColor[2] = 0.0f;
    // background color is only a property of the display window
    //ds->renderer->setParameterV3(VV_BG_COLOR, ds->bgColor);
      // Use opposite color for object boundaries:
      //ds->renderer->setBoundariesColor(1.0f-ds->bgColor[0], 1.0f-ds->bgColor[1], 1.0f-ds->bgColor[2]);
break;
  case 7:                                     // auto-rotation mode
    ds->rotationMode = !ds->rotationMode;
    if (!ds->rotationMode)
    {
      ds->rotating = false;
      if (ds->refinement) ds->hqMode = true;
    }
    cerr << "Auto rotation " << ds->onOff[ds->rotationMode] << endl;
    break;
  case 8:                                     // save camera
    if (ds->ov->saveMV("virvo-camera.txt"))
      cerr << "Camera saved to file 'virvo-camera.txt'." << endl;
    else
      cerr << "Couldn't save camera to file 'virvo-camera.txt'." << endl;
    break;
  case 9:                                     // load camera
    if (ds->ov->loadMV("virvo-camera.txt"))
      cerr << "Camera loaded from file 'virvo-camera.txt'." << endl;
    else
      cerr << "Couldn't load camera from file 'virvo-camera.txt'." << endl;
    break;
  case 10:                                    // toggle icon
    ds->iconMode = !ds->iconMode;
    cerr << "Icon " << ds->onOff[ds->iconMode] << endl;
    break;
  }
  glutPostRedisplay();
}

//----------------------------------------------------------------------------
/** Do a performance test.
  Default behavior:
  The test resets the view position (but not the projection mode)
  and measures the time needed for a
  360 degrees rotation of the volume about its vertical axis.
  The image is drawn every 2 degrees.
  <br>
*/
double vvView::performanceTest()
{
  vvDebugMsg::msg(1, "vvView::performanceTest()");

  double total = 0.;

  vvTestSuite* testSuite = new vvTestSuite(ds->testSuiteFileName);
  if (testSuite->isInitialized())
  {
    std::vector<vvPerformanceTest*> tests = testSuite->getTests();
    std::vector<vvPerformanceTest*>::const_iterator it;
    float step = 2.0f * VV_PI / 180.0f;

    if (tests.size() < 1)
    {
      cerr << "No tests in test suite" << endl;
    }

    for(it = tests.begin(); it != tests.end(); ++it)
    {
      vvPerformanceTest* test = *it;

      // TODO: make dataset configurable.
      test->setDatasetName(ds->filename);

      std::vector<float> diffTimes;
      std::vector<vvMatrix> modelViewMatrices;

      vvStopwatch* totalTime = new vvStopwatch();
      totalTime->start();

      int framesRendered = 0;
      vvRendererFactory::Options opt;
      ds->createRenderer("", opt,
                      (size_t) test->getBrickDims()[0],
                      (size_t) test->getBrickDims()[1],
                      (size_t) test->getBrickDims()[2]);
      ds->hqMode = false;
      ds->draftQuality = test->getQuality();
      ds->ov->reset();
      ds->ov->resetMV();
      ds->ov->mv.scaleLocal(ds->mvScale);
      ds->perspectiveMode = (test->getProjectionType() == vvObjView::PERSPECTIVE);
      if (ds->perspectiveMode)
      {
        ds->ov->setProjection(vvObjView::PERSPECTIVE, 45.0f, 0.01f, 100.0f);
      }
      else
      {
        ds->ov->setProjection(vvObjView::ORTHO, 2.0f, -100.0, 100.0);
      }
      // Do this once to propagate the changes... .
      ds->displayCallback();
      ds->renderer->profileStart();

      if (test->getVerbose())
      {
        printProfilingInfo(test->getId(), tests.size());
      }

      for (int ite = 0; ite < test->getIterations(); ++ite)
      {
        for (int i = 0; i < test->getFrames(); ++i)
        {
          vvVector3 dir;
          switch (test->getTestAnimation())
          {
          case vvPerformanceTest::VV_ROT_X:
            ds->ov->mv.rotate(step, 1.0f, 0.0f, 0.0f);
            break;
          case vvPerformanceTest::VV_ROT_Y:
            ds->ov->mv.rotate(step, 0.0f, 1.0f, 0.0f);
            break;
          case vvPerformanceTest::VV_ROT_Z:
            ds->ov->mv.rotate(step, 0.0f, 0.0f, 1.0f);
            break;
          case vvPerformanceTest::VV_ROT_RAND:
            dir.random(0.0f, 1.0f);
            ds->ov->mv.rotate(step, dir);
            break;
          default:
            break;
          }
          ds->displayCallback();
          diffTimes.push_back(totalTime->getDiff());
          modelViewMatrices.push_back(ds->ov->mv);

          ++framesRendered;
        }
      }
      total = totalTime->getTime();

      test->getTestResult()->setDiffTimes(diffTimes);
      test->getTestResult()->setModelViewMatrices(modelViewMatrices);
      test->writeResultFiles();

      if (test->getVerbose())
      {
        printProfilingResult(totalTime, framesRendered);
      }
      delete totalTime;
    }
  }
  else
  {
    vvStopwatch* totalTime;
    float step = 2.0f * VV_PI / 180.0f;
    int   angle;
    int   framesRendered = 0;

    // Prepare test:
    totalTime = new vvStopwatch();

    ds->hqMode = false;
    ds->ov->reset();
    ds->ov->mv.scaleLocal(ds->mvScale);
    ds->displayCallback();

    printProfilingInfo();

    // Perform test:
    totalTime->start();
    ds->renderer->profileStart();

    for (angle=0; angle<180; angle+=2)
    {
      ds->ov->mv.rotate(step, 0.0f, 1.0f, 0.0f);  // rotate model view matrix
      ds->displayCallback();
      ++framesRendered;
    }

    ds->ov->reset();
    ds->ov->mv.scaleLocal(ds->mvScale);
    for (angle=0; angle<180; angle+=2)
    {
      ds->ov->mv.rotate(step, 0.0f, 0.0f, 1.0f);  // rotate model view matrix
      ds->displayCallback();
      ++framesRendered;
    }

    ds->ov->reset();
    ds->ov->mv.scaleLocal(ds->mvScale);
    for (angle=0; angle<180; angle+=2)
    {
      ds->ov->mv.rotate(step, 1.0f, 0.0f, 0.0f);  // rotate model view matrix
      ds->displayCallback();
      ++framesRendered;
    }
    total = totalTime->getTime();

    printProfilingResult(totalTime, framesRendered);

    delete totalTime;
  }
  delete testSuite;

  return total;
}


//----------------------------------------------------------------------------
/** Print information about the most recent test run.
    @param testNr Number of the current test (1-based).
    @param testCnt Number of tests in total.
  */
void vvView::printProfilingInfo(const int testNr, const int testCnt)
{
  GLint viewport[4];                             // OpenGL viewport information (position and size)
  char  projectMode[2][16] = {"parallel","perspective"};
  char  filter_mode[2][32] = {"nearest neighbor","linear"};
  char  onOffMode[2][8] = {"off","on"};
  char  pgMode[4][32] = { "vert shader", "geom shader", "vert and geom shader", "CPU" };
  const int HOST_NAME_LEN = 80;
  char  localHost[HOST_NAME_LEN];
  glGetIntegerv(GL_VIEWPORT, viewport);
#ifdef _WIN32
  strcpy(localHost, "n/a");
#else
  if (gethostname(localHost, HOST_NAME_LEN-1)) strcpy(localHost, "n/a");
#endif

  // Print profiling info:
  cerr.setf(ios::fixed, ios::floatfield);
  cerr.precision(3);
  cerr << "*******************************************************************************" << endl;
  cerr << "Test (" << testNr << "/" << testCnt << ")" << endl;
  cerr << "Local host........................................" << localHost << endl;
  cerr << "Renderer/geometry................................." << ds->currentRenderer << endl;
  cerr << "Renderer options/voxel type......................." << ds->currentOptions["voxeltype"] << endl;
  cerr << "Volume file name.................................." << ds->vd->getFilename() << endl;
  cerr << "Volume size [voxels].............................." << ds->vd->vox[0] << " x " << ds->vd->vox[1] << " x " << ds->vd->vox[2] << endl;
  cerr << "Output image size [pixels]........................" << viewport[2] << " x " << viewport[3] << endl;
  cerr << "Image quality....................................." << ds->renderer->getParameter(vvRenderState::VV_QUALITY).asFloat() << endl;
  cerr << "Projection........................................" << projectMode[ds->perspectiveMode] << endl;
  cerr << "Interpolation mode................................" << filter_mode[ds->filter_mode] << endl;
  cerr << "Empty space leaping for bricks...................." << onOffMode[ds->emptySpaceLeapingMode] << endl;
  cerr << "Early ray termination............................." << onOffMode[ds->earlyRayTermination] << endl;
  cerr << "Pre-integration..................................." << onOffMode[ds->preintMode] << endl;
  cerr << "Render to offscreen buffer........................" << onOffMode[ds->useOffscreenBuffer] << endl;
  cerr << "Precision of offscreen buffer....................." << ds->bufferPrecision << "bit" << endl;
  cerr << "Proxy geometry generator.........................." << pgMode[ds->isectType] << endl;
  cerr << "Opacity correction................................" << onOffMode[ds->opCorrMode] << endl;
  cerr << "Gamma correction.................................." << onOffMode[ds->gammaMode] << endl;
}


//----------------------------------------------------------------------------
/** Conclude profiling info with the final result.
    @param totalTime A stop watch to read the profiling time from.
    @param framesRendered The number of frames rendered for the test.
  */
void vvView::printProfilingResult(vvStopwatch* totalTime, const int framesRendered)
{
  cerr << "Total profiling time [sec]........................" << totalTime->getTime() << endl;
  cerr << "Frames rendered..................................." << framesRendered << endl;
  cerr << "Average time per frame [sec]......................" << (float(totalTime->getTime()/framesRendered)) << endl;
  cerr << "*******************************************************************************" << endl;
}


//----------------------------------------------------------------------------
/** Print a short info how to interact with the probe in roi mode
  */
void vvView::printROIMessage()
{
  if (ds->roiEnabled)
  {
    cerr << "Region of interest mode enabled" << endl;
    if (ds->sphericalROI)
    {
      cerr << "Region of interest mode: spherical" << endl;
    }
    else
    {
      cerr << "Region of interest mode: cuboid" << endl;
    }
    cerr << "Arrow left:         -x, arrow right:      +x" << endl;
    cerr << "Arrow down:         -y, arrow up:         +y" << endl;
    cerr << "Arrow down + shift: -z, arrow up + shift: +z" << endl;
    cerr << endl;
    cerr << "Use '[‘ and ‘]‘ to resize probe" << endl;
  }
  else
  {
    cerr << "Region of interest mode disabled" << endl;
  }
}


//----------------------------------------------------------------------------
/// Create the pop-up menus.
void vvView::createMenus()
{
  int rendererMenu, voxelMenu, optionsMenu, transferMenu, animMenu, roiMenu, clipMenu, viewMenu;

  vvDebugMsg::msg(1, "vvView::createMenus()");

  // Rendering geometry menu:
  rendererMenu = glutCreateMenu(rendererMenuCallback);
  glutAddMenuEntry("Auto select [0]", 0);
  if (vvRendererFactory::hasRenderer("texrend"))  glutAddMenuEntry("3D textures - viewport aligned [3]", 3);
  glutAddMenuEntry("CPU Shear-warp [6]", 6);
  glutAddMenuEntry("GPU Shear-warp [7]", 7);
  glutAddMenuEntry("VolPack [8]", 8);
  if (vvRendererFactory::hasRenderer("rayrend", "cuda")) glutAddMenuEntry("CUDA ray casting [9]", 9);
  if (vvRendererFactory::hasRenderer("rayrend", "fpu")) glutAddMenuEntry("FPU ray casting", 10);
  if (vvRendererFactory::hasRenderer("rayrend", "sse4_1")) glutAddMenuEntry("SSE 4.1 ray casting", 12);
  glutAddMenuEntry("Image-based remote rendering", 13);
  glutAddMenuEntry("Remote rendering", 14);
  glutAddMenuEntry("Decrease quality [-]", 98);
  glutAddMenuEntry("Increase quality [+]", 99);

  // Voxel menu:
  voxelMenu = glutCreateMenu(voxelMenuCallback);
  glutAddMenuEntry("Auto select", 0);
  glutAddMenuEntry("RGBA", 1);
  if (vvTexRend::isSupported(vvTexRend::VV_POST_CLASSIFICATION)) glutAddMenuEntry("Fragment shader", 5);

  // Renderer options menu:
  optionsMenu = glutCreateMenu(optionsMenuCallback);
  glutAddMenuEntry("Toggle slice interpolation [i]", 0);
  glutAddMenuEntry("Toggle warp interpolation [W]", 16);
  if (vvTexRend::isSupported(vvTexRend::VV_POST_CLASSIFICATION)
      && vvGLTools::isGLextensionSupported("GL_ARB_multitexture"))
  {
    glutAddMenuEntry("Toggle pre-integration [P]", 1);
  }
  if (vvGLTools::isGLextensionSupported("GL_EXT_blend_minmax"))
  {
    glutAddMenuEntry("Toggle maximum intensity projection (MIP) [x]", 2);
  }
  glutAddMenuEntry("Toggle opacity correction", 3);
  glutAddMenuEntry("Toggle gamma correction", 4);
  glutAddMenuEntry("Toggle empty space leaping", 5);
  glutAddMenuEntry("Toggle early ray termination", 15);
  glutAddMenuEntry("Toggle offscreen buffering", 6);
  glutAddMenuEntry("Toggle head light", 7);
  glutAddMenuEntry("Increase z size [H]", 8);
  glutAddMenuEntry("Decrease z size [h]", 9);
  glutAddMenuEntry("Increase buffer precision", 10);
  glutAddMenuEntry("Decrease buffer precision", 11);
  glutAddMenuEntry("Show/hide bricks [B]", 12);
  if (vvTexRend::isSupported(vvTexRend::VV_POST_CLASSIFICATION))
    glutAddMenuEntry("Cycle shader [A]", 14);
  glutAddMenuEntry("Inc ibr mode [#]", 17);
  glutAddMenuEntry("Toggle synchronous ibr mode [T]", 19);
  glutAddMenuEntry("Cycle codec [C]", 18);

  // Transfer function menu:
  transferMenu = glutCreateMenu(transferMenuCallback);
  glutAddMenuEntry("Colors: bright", 0);
  glutAddMenuEntry("Colors: HSV", 1);
  glutAddMenuEntry("Colors: grayscale", 2);
  glutAddMenuEntry("Colors: white", 3);
  glutAddMenuEntry("Colors: red", 4);
  glutAddMenuEntry("Colors: green", 5);
  glutAddMenuEntry("Colors: blue", 6);
  glutAddMenuEntry("Colors: cool to warm", 7);
  glutAddMenuEntry("Colors: fire", 8);
  glutAddMenuEntry("Alpha: ascending", 9);
  glutAddMenuEntry("Alpha: descending", 10);
  glutAddMenuEntry("Alpha: opaque", 11);
  glutAddMenuEntry("Alpha: single peak", 12);
  glutAddMenuEntry("Alpha: shift peak left [<]", 13);
  glutAddMenuEntry("Alpha: shift peak right [>]", 14);
  glutAddMenuEntry("Gamma: less red [1]", 15);
  glutAddMenuEntry("Gamma: more red [2]", 16);
  glutAddMenuEntry("Gamma: less green [3]", 17);
  glutAddMenuEntry("Gamma: more green [4]", 18);
  glutAddMenuEntry("Gamma: less blue [5]", 19);
  glutAddMenuEntry("Gamma: more blue [6]", 20);
  glutAddMenuEntry("Channel 4: less red", 21);
  glutAddMenuEntry("Channel 4: more red", 22);
  glutAddMenuEntry("Channel 4: less green", 23);
  glutAddMenuEntry("Channel 4: more green", 24);
  glutAddMenuEntry("Channel 4: less blue", 25);
  glutAddMenuEntry("Channel 4: more blue", 26);

  // Animation menu:
  animMenu = glutCreateMenu(animMenuCallback);
  glutAddMenuEntry("Next frame [n]", 0);
  glutAddMenuEntry("Previous frame [N]", 1);
  glutAddMenuEntry("Start/stop animation [a]", 2);
  glutAddMenuEntry("Rewind", 3);
  glutAddMenuEntry("Animation speed up [s]", 4);
  glutAddMenuEntry("Animation speed down [S]", 5);
  glutAddMenuEntry("Reset speed", 6);

  // Region of interest menu:
  roiMenu = glutCreateMenu(roiMenuCallback);
  glutAddMenuEntry("Toggle region of interest mode [R]", 0);
  glutAddMenuEntry("size-- [[]", 98);
  glutAddMenuEntry("size++ []]", 99);

  // Clip menu:
  clipMenu = glutCreateMenu(clipMenuCallback);
  glutAddMenuEntry("Toggle clip mode [I]", 0);
  glutAddMenuEntry("Toggle clipping edit mode [E]", 1);

  // Viewing Window Menu:
  viewMenu = glutCreateMenu(viewMenuCallback);
  glutAddMenuEntry("Toggle bounding box [b]", 0);
  glutAddMenuEntry("Toggle axis orientation [o]", 1);
  glutAddMenuEntry("Toggle frame rate display [f]", 2);
  glutAddMenuEntry("Toggle transfer function display", 3);
  glutAddMenuEntry("Stereo mode", 4);
  glutAddMenuEntry("Toggle full screen zoom [z]", 5);
  glutAddMenuEntry("Toggle window color [w]", 6);
  glutAddMenuEntry("Toggle auto rotation", 7);
  glutAddMenuEntry("Save camera position [u]", 8);
  glutAddMenuEntry("Load camera position [v]", 9);
  glutAddMenuEntry("Toggle icon display [c]", 10);

  // Main menu:
  mainMenu = glutCreateMenu(mainMenuCallback);
  glutAddSubMenu("Rendering geometry", rendererMenu);
  glutAddSubMenu("Voxel representation", voxelMenu);
  glutAddSubMenu("Renderer options", optionsMenu);
  glutAddSubMenu("Transfer function", transferMenu);
  glutAddSubMenu("Animation", animMenu);
  glutAddSubMenu("Region of interest", roiMenu);
  glutAddSubMenu("Clipping", clipMenu);
  glutAddSubMenu("Viewing window", viewMenu);
  glutAddMenuEntry("Toggle perspective mode [p]", 0);
  glutAddMenuEntry("Toggle auto refinement [e]", 4);
  glutAddMenuEntry("Toggle rendering time display [d]", 5);
  glutAddMenuEntry("Reset object orientation", 7);
  glutAddMenuEntry("Toggle popup menu/zoom mode [m]", 8);
  glutAddMenuEntry("Save volume to file", 9);
  glutAddMenuEntry("Performance test [t]", 11);
  glutAddMenuEntry("Change debug level [D]", 13);
  glutAddMenuEntry("Quit [q]", 12);

  glutSetMenu(mainMenu);
  glutAttachMenu(GLUT_RIGHT_BUTTON);
  menuEnabled = true;
}


//----------------------------------------------------------------------------
/// Initialize the GLUT window and the OpenGL graphics context.
void vvView::initGraphics(int argc, char *argv[])
{
  vvDebugMsg::msg(1, "vvView::initGraphics()");

  char* version;
  char  title[128];                              // window title

  cerr << "Number of CPUs found: " << vvToolshed::getNumProcessors() << endl;
  cerr << "Initializing GLUT." << endl;

  glutInit(&argc, argv);                // initialize GLUT

// Other glut versions than freeglut currently don't support
// debug context flags.
#if defined(FREEGLUT) && defined(GLUT_DEBUG)
  glutInitContextFlags(GLUT_DEBUG);
#endif // FREEGLUT

  uint glutDisplayFlags = GLUT_DOUBLE | GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH;
  if (ibrValidation)
  {
    // Need a stencil buffer for this.
    glutDisplayFlags |= GLUT_STENCIL;
  }

  if (tryQuadBuffer)
  {
    // create stereo context
    glutInitDisplayMode(glutDisplayFlags | GLUT_STEREO);
    if (!glutGet(GLUT_DISPLAY_MODE_POSSIBLE))
    {
      cerr << "Stereo mode not supported by display driver." << endl;
      tryQuadBuffer = false;
    }
  }
  if (!tryQuadBuffer)
  {
    // create double buffering context
    glutInitDisplayMode(glutDisplayFlags);
  }
  else
  {
    cerr << "Stereo mode found." << endl;
    activeStereoCapable = true;
  }
  if (!glutGet(GLUT_DISPLAY_MODE_POSSIBLE))
  {
    cerr << "Error: Glut needs a double buffering OpenGL context with alpha channel." << endl;
    exit(-1);
  }

  glutInitWindowSize(winWidth, winHeight);       // set initial window size

  // Create window title.
  // Don't use sprintf, it won't work with macros on Irix!
  strcpy(title, "Virvo File Viewer V");
  strcat(title, virvo::version());
  window = glutCreateWindow(title);              // open window and set window title

  // Set GL state:
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glEnable(GL_DEPTH_TEST);

  // Set Glut callbacks:
  glutDisplayFunc(displayCallback);
  glutReshapeFunc(reshapeCallback);
  glutMouseFunc(buttonCallback);
  glutMotionFunc(motionCallback);
  glutKeyboardFunc(keyboardCallback);
  glutSpecialFunc(specialCallback);
#ifdef FREEGLUT
  glutCloseFunc(cleanup);
#endif

  version = (char*)glGetString(GL_VERSION);
  cerr << "Found OpenGL version: " << version << endl;
  if (strncmp(version,"1.0",3)==0)
  {
    cerr << "VView requires OpenGL version 1.1 or greater." << endl;
  }

  vvGLTools::checkOpenGLextensions();

  if (vvDebugMsg::isActive(2))
  {
    cerr << "\nSupported OpenGL extensions:" << endl;
    vvGLTools::displayOpenGLextensions(vvGLTools::ONE_BY_ONE);
  }
}

//----------------------------------------------------------------------------
/** Set projection mode to perspective or parallel.
  @param newmode true = perspective projection, false = parallel projection
*/
void vvView::setProjectionMode(bool newmode)
{
  vvDebugMsg::msg(1, "vvView::setProjectionMode()");

  perspectiveMode = newmode;

  if (perspectiveMode)
  {
    ov->setProjection(vvObjView::PERSPECTIVE, 45.0f, 0.01f, 100.0f);
  }
  else
  {
    ov->setProjection(vvObjView::ORTHO, 2.0f, -100.0, 100.0);
  }
}

void vvView::renderMotion() const
{
  std::ifstream file("motion.txt");
  if(!file.is_open())
  {
    std::cerr << "Could not open \"motion.txt\" for reading" << std::endl;
	  return;
  }

  while (ds->ov->loadMV(file))
  {
    glDrawBuffer(GL_BACK);
    glClearColor(ds->bgColor[0], ds->bgColor[1], ds->bgColor[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    ds->ov->updateModelviewMatrix(vvObjView::LEFT_EYE);

    ds->renderer->renderVolumeGL();
    glutSwapBuffers();
  }
}


void vvView::editClipPlane(const int command, const float val)
{
  vec3 tmp = ds->renderer->getParameter(vvRenderState::VV_CLIP_PLANE_NORMAL);
  vec3 clipNormal(tmp);
  switch (command)
  {
  case PLANE_X:
    {
      vvMatrix m;
      m.identity();
      static const vec3 axis(1, 0, 0);
      m.rotate(val, axis);
      vec4 tmp = mat4(m) * vec4(clipNormal, 1.0f);
      clipNormal = tmp.xyz() / tmp.w;
    }
    break;
  case PLANE_Y:
    {
      vvMatrix m;
      m.identity();
      static const vec3 axis(0, 1, 0);
      m.rotate(val, axis);
      vec4 tmp = mat4(m) * vec4(clipNormal, 1.0f);
      clipNormal = tmp.xyz() / tmp.w;
    }
    break;
  case PLANE_Z:
    {
      vvMatrix m;
      m.identity();
      static const vec3 axis(0, 0, 1);
      m.rotate(val, axis);
      vec4 tmp = mat4(m) * vec4(clipNormal, 1.0f);
      clipNormal = tmp.xyz() / tmp.w;
    }
    break;
  default:
    cerr << "Unknown command" << endl;
    break;
  }
  ds->renderer->setParameter(vvRenderState::VV_CLIP_PLANE_NORMAL, tmp);
  glutPostRedisplay();
}


//----------------------------------------------------------------------------
/// Display command usage help on the command line.
void vvView::displayHelpInfo()
{
  vvDebugMsg::msg(1, "vvView::displayHelpInfo()");

  cerr << "Syntax:" << endl;
  cerr << endl;
  cerr << "  vview [<volume_file.xxx>] [options]" << endl;
  cerr << endl;
  cerr << "<volume_file.xxx>" << endl;
  cerr << "Volume file to display. Please see VConv for a list of available" << endl;
  cerr << "file types." << endl;
  cerr << endl;
  cerr << "Available options:" << endl;
  cerr << endl;
  cerr << "-clientmode <mode> (-c)" << endl;
  cerr << " Renderer is a client in mode <mode> and connects to server(s) given with -server" << endl;
  cerr << " Modes:" << endl;
  cerr << " ibr      = image based rendering" << endl;
  cerr << " image    = remote rendering" << endl;
  cerr << " parbrick = parallel brick rendering" << endl;
  cerr << endl;
  cerr << "-port" << endl;
  cerr << " Renderer is a render slave. Don't use the default port (31050), but the specified one" << endl;
  cerr << endl;
  cerr << "-renderer <num> (-r)" << endl;
  cerr << " Select the default renderer:" << endl;
  cerr << " 0  = Autoselect" << endl;
  cerr << " 1  = 2D Textures - Slices" << endl;
  cerr << " 2  = 2D Textures - Cubic" << endl;
  cerr << " 3  = 3D Textures - Viewport aligned" << endl;
  cerr << " 4  = 3D Textures - Bricks" << endl;
  cerr << " 5  = 3D Textures - Spherical" << endl;
  cerr << " 6  = Shear-warp (CPU)" << endl;
  cerr << " 7  = Shear-warp (GPU)" << endl;
  cerr << " 8  = Ray casting (GPU)" << endl;
  cerr << " 9  = VolPack (CPU)" << endl;
  cerr << endl;
  cerr << "-voxeltype <num>" << endl;
  cerr << " Select the default voxel type:" << endl;
  cerr << " 0 = Autoselect" << endl;
  cerr << " 1 = RGBA" << endl;
  cerr << " 3 = OpenGL paletted textures" << endl;
  cerr << " 4 = Nvidia texture shader" << endl;
  cerr << " 5 = Nvidia pixel shader" << endl;
  cerr << " 6 = ARB fragment program" << endl;
  cerr << endl;
  cerr << "-quality <value> (-q)" << endl;
  cerr << "Set the render quality (default: 1.0)" << endl;
  cerr << endl;
  cerr << "-dsp <host:display.screen>" << endl;
  cerr << "  Add x-org display for additional rendering context" << endl;
  cerr << endl;
  cerr << "-bricks <value>" << endl;
  cerr << "  Number of bricks used by serbrickrend renderer" << endl;
  cerr << endl;
  cerr << "-server <url>[:port]" << endl;
  cerr << "  Add a server renderer connected to over tcp ip" << endl;
  cerr << endl;
  cerr << "-serverfilename <path to file>" << endl;
  cerr << "  Path to a file where the server can find its volume data" << endl;
  cerr << "  If this entry is -serverfilename n, the n'th server will try to load this file" << endl;
  cerr << endl;
  cerr << "-lighting" << endl;
  cerr << " Use headlight for local illumination" << endl;
  cerr << endl;
  cerr << "-benchmark" << endl;
  cerr << " Time 3 half rotations and exit" << endl;
  cerr << endl;
  cerr << endl;
  cerr << "-isecttype <num>" << endl;
  cerr << " Select proxy geometry generator:" << endl;
  cerr << " 0 = Vertex shader" << endl;
  cerr << " 1 = Geometry shader" << endl;
  cerr << " 2 = Vertex shader and geometry shader combined" << endl;
  cerr << " 3 = CPU" << endl;
  cerr << "-help (-h)" << endl;
  cerr << "Display this help information" << endl;
  cerr << endl;
  cerr << "-size <width> <height>" << endl;
  cerr << " Set the window size to <width> * <height> pixels." << endl;
  cerr << " The default window size is " << DEFAULTSIZE << " * " << DEFAULTSIZE <<
          " pixels" << endl;
  cerr << endl;
  cerr << "-parallel (-p)" << endl;
  cerr << " Use parallel projection mode" << endl;
  cerr << endl;
  cerr << "-boundaries (-b)" << endl;
  cerr << " Draw volume data set boundaries" << endl;
  cerr << endl;
  cerr << "-quad (-q)" << endl;
  cerr << " Try to request a quad buffered visual" << endl;
  cerr << endl;
  cerr << "-astereo" << endl;
  cerr << " Enable active stereo mode (if available)" << endl;
  cerr << endl;
  cerr << "-pstereo1, -pstereo2" << endl;
  cerr << " Enable passive stereo mode. The two modes are for different left/right" << endl;
  cerr << " assignments." << endl;
  cerr << endl;
  cerr << "-orientation (-o)" << endl;
  cerr << " Display volume orientation axes" << endl;
  cerr << endl;
  cerr << "-fps (-f)" << endl;
  cerr << " Display rendering speed [frames per second]" << endl;
  cerr << endl;
  cerr << "-nobt" << endl;
  cerr << " Don't show backtrace on OpenGL error (GL_ARB_debug_output only)" << endl;
  cerr << endl;
  cerr << "-transfunc (-t)" << endl;
  cerr << " Display transfer function color bar. Only works with 8 and 16 bit volumes" << endl;
  cerr << endl;
  cerr << "-testsuitefilename" << endl;
  cerr << " Specify a file with performance tests" << endl;
  cerr << endl;
  cerr << "-showbricks" << endl;
  cerr << " Show the brick outlines \\wo volume when brick renderer is used" << endl;
  cerr << endl;
  cerr << "-rec" << endl;
  cerr << " Record camera motion to file" << endl;
  cerr << endl;
  cerr << "-play" << endl;
  cerr << " Play camera motion from file" << endl;
  cerr << endl;
  #ifndef WIN32
  cerr << endl;
  #endif
}


//----------------------------------------------------------------------------
/** Parse command line arguments.
  @param argc,argv command line arguments
  @return true if parsing ok, false on error
*/
bool vvView::parseCommandLine(int argc, char** argv)
{
  vvDebugMsg::msg(1, "vvView::parseCommandLine()");

  int arg;                                       // index of currently processed command line argument

  arg = 0;
  for (;;)
  {
    if ((++arg)>=argc) return true;
    if (vvToolshed::strCompare(argv[arg], "-help")==0 ||
        vvToolshed::strCompare(argv[arg], "-h")==0 ||
        vvToolshed::strCompare(argv[arg], "-?")==0 ||
        vvToolshed::strCompare(argv[arg], "/?")==0)
    {
      displayHelpInfo();
      return false;
    }
    else if (vvToolshed::strCompare(argv[arg], "-c")==0 ||
             vvToolshed::strCompare(argv[arg], "-clientmode")==0)
    {
      std::string val;
      if(argv[arg+1])
        val = argv[arg+1];

      if(val == "comparison")
      {
        rrMode = RR_COMPARISON;
        ds->currentRenderer = "comparison";
        arg++;
      }
      else if(val == "ibr")
      {
        rrMode = RR_IBR;
        ds->currentRenderer = "ibr";
        arg++;
      }
      else if(val == "image")
      {
        rrMode = RR_IMAGE;
        ds->currentRenderer = "image";
        arg++;
      }
      else if(val == "parbrick")
      {
        rrMode = RR_PARBRICK;
        ds->currentRenderer = "parbrick";
        arg++;
      }
      else
      {
        cerr << "Set default client mode: image based rendering" << endl;
        rrMode = RR_IBR;
        ds->currentRenderer = "ibr";
      }
    }
    else if (vvToolshed::strCompare(argv[arg], "-nobt")==0)
    {
      showBt = false;
    }
    else if (vvToolshed::strCompare(argv[arg], "-roi")==0)
    {
      roiEnabled = true;
    }
    else if (vvToolshed::strCompare(argv[arg], "-port")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "No port specified, defaulting to: " << vvView::DEFAULT_PORT << endl;
        ports.push_back(vvView::DEFAULT_PORT);
        return false;
      }
      else
      {
        ports.push_back(atoi(argv[arg]));
      }
    }
    else if (vvToolshed::strCompare(argv[arg], "-r")==0 ||
             vvToolshed::strCompare(argv[arg], "-renderer")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "Renderer ID missing." << endl;
        return false;
      }

      ds->currentRenderer = argv[arg];
    }
    else if (vvToolshed::strCompare(argv[arg], "-voxeltype")==0
        || vvToolshed::strCompare(argv[arg], "-options")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "Option string/voxel type missing." << endl;
        return false;
      }
      ds->currentOptions["voxeltype"] = argv[arg];
    }
    else if (vvToolshed::strCompare(argv[arg], "-size")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "Window width missing." << endl;
        return false;
      }
      winWidth = atoi(argv[arg]);
      if ((++arg)>=argc)
      {
        cerr << "Window height missing." << endl;
        return false;
      }
      winHeight = atoi(argv[arg]);
      if (winWidth<1 || winHeight<1)
      {
        cerr << "Invalid window size." << endl;
        return false;
      }
    }
    else if ((vvToolshed::strCompare(argv[arg], "-q") == 0)
          |  (vvToolshed::strCompare(argv[arg], "-quality") == 0))
    {
      if ((++arg)>=argc)
      {
        cerr << "Quality missing." << endl;
        return false;
      }
      ds->draftQuality = (float)strtod(argv[arg], NULL);
    }
    else if (vvToolshed::strCompare(argv[arg], "-dsp")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "Display name unspecified." << endl;
        return false;
      }
      displays.push_back(argv[arg]);
    }
    else if (vvToolshed::strCompare(argv[arg], "-bricks")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "Number of bricks missing." << endl;
        return false;
      }
      bricks = atoi(argv[arg]);
    }
    else if (vvToolshed::strCompare(argv[arg], "-lighting")==0)
    {
      useHeadLight = true;
    }
    else if (vvToolshed::strCompare(argv[arg], "-server")==0
        || vvToolshed::strCompare(argv[arg], "-s")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "Server unspecified." << endl;
        return false;
      }

      const int port = vvToolshed::parsePort(argv[arg]);
      if (port != -1)
      {
        ports.push_back(port);
        servers.push_back(vvToolshed::stripPort(argv[arg]));
      }
      else
      {
        servers.push_back(argv[arg]);
      }
    }
    else if (vvToolshed::strCompare(argv[arg], "-serverfilename")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "Server file name unspecified" << endl;
        return false;
      }
      serverFileNames.push_back(argv[arg]);
    }
    else if (vvToolshed::strCompare(argv[arg], "-testsuitefilename")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "Test suite file name unspecified" << endl;
      }
      testSuiteFileName = argv[arg];
    }
    else if (vvToolshed::strCompare(argv[arg], "-benchmark")==0)
    {
      benchmark = true;
    }
    else if (vvToolshed::strCompare(argv[arg], "-isecttype")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "Proxy geometry generator missing." << endl;
        return false;
      }
      int level = atoi(argv[arg]);
      if (level>=0 && level<=3)
        ds->isectType = level;
      else
      {
        cerr << "Invalid proxy geometry generator." << endl;
        return false;
      }

    }
    else if (vvToolshed::strCompare(argv[arg], "-debug")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "Debug level missing." << endl;
        return false;
      }
      int level = atoi(argv[arg]);
      if (level>=0 && level<=3)
        vvDebugMsg::setDebugLevel(level);
      else
      {
        cerr << "Invalid debug level." << endl;
        return false;
      }
    }
    else if (vvToolshed::strCompare(argv[arg], "-q")==0
            || vvToolshed::strCompare(argv[arg], "-quad")==0)
    {
      tryQuadBuffer = true;
    }
    else if (vvToolshed::strCompare(argv[arg], "-astereo")==0)
    {
      stereoMode = 1;
    }
    else if (vvToolshed::strCompare(argv[arg], "-pstereo1")==0)
    {
      stereoMode = 2;
    }
    else if (vvToolshed::strCompare(argv[arg], "-pstereo2")==0)
    {
      stereoMode = 3;
    }
    else if (vvToolshed::strCompare(argv[arg], "-parallel")==0 ||
             vvToolshed::strCompare(argv[arg], "-p")==0)
    {
      perspectiveMode = false;
    }
    else if (vvToolshed::strCompare(argv[arg], "-boundaries")==0 ||
             vvToolshed::strCompare(argv[arg], "-b")==0)
    {
      boundariesMode = true;
    }
    else if (vvToolshed::strCompare(argv[arg], "-orientation")==0 ||
             vvToolshed::strCompare(argv[arg], "-o")==0)
    {
      orientationMode = true;
    }
    else if (vvToolshed::strCompare(argv[arg], "-fps")==0 ||
             vvToolshed::strCompare(argv[arg], "-f")==0)
    {
      fpsMode = true;
    }
    else if (vvToolshed::strCompare(argv[arg], "-transfunc")==0 ||
             vvToolshed::strCompare(argv[arg], "-t")==0)
    {
      paletteMode = true;
    }
    else if (vvToolshed::strCompare(argv[arg], "-display")==0
            || vvToolshed::strCompare(argv[arg], "-geometry")==0)
    {
      // handled by GLUT
      if ((++arg)>=argc)
      {
        cerr << "Required argument unspecified" << endl;
        return false;
      }
    }
    else if (vvToolshed::strCompare(argv[arg], "-iconic")==0
            || vvToolshed::strCompare(argv[arg], "-direct")==0
            || vvToolshed::strCompare(argv[arg], "-indirect")==0
            || vvToolshed::strCompare(argv[arg], "-gldebug")==0
            || vvToolshed::strCompare(argv[arg], "-sync")==0)
    {
    }
    else if (vvToolshed::strCompare(argv[arg], "-showbricks")==0)
    {
      showBricks = true;
    }
    else if (vvToolshed::strCompare(argv[arg], "-rec")==0)
    {
      recordMode = true;
      stopWatch.start();
      matrixFile = fopen("motion.txt", "wab");
    }
    else if (vvToolshed::strCompare(argv[arg], "-play")==0)
    {
      playMode = true;
    }
    else
    {
      filename = argv[arg];
      if (filename==NULL || filename[0]=='-')
      {
        cerr << "File name expected." << endl;
        return false;
      }
      if (!vvToolshed::isFile(filename))       // check if file exists
      {
        cerr << "File not found: " << filename << endl;
        return false;
      }
    }
  }
}


//----------------------------------------------------------------------------
/** Main VView routine.
  @param argc,argv command line arguments
  @return 0 if the program finished ok, 1 if an error occurred
*/
int vvView::run(int argc, char** argv)
{
  vvDebugMsg::msg(1, "vvView::run()");

  cerr << "VView " << virvo::version() << endl;
  cerr << "(c) " << VV_VERSION_YEAR << " Juergen Schulze (schulze@cs.brown.edu)" << endl;
  cerr << "Brown University" << endl << endl;

  if (argc<2)
  {
    cerr << "VView (=Virvo View) is a utility to display volume files." << endl;
    cerr << "The Virvo volume rendering system was developed at the University of Stuttgart." << endl;
    cerr << "Please find more information at http://www.cs.brown.edu/people/schulze/virvo/." << endl;
    cerr << endl;
    cerr << "Syntax:" << endl;
    cerr << "  vview [<volume_file.xxx>] [options]" << endl;
    cerr << endl;
    cerr << "For a list of options type:" << endl;
    cerr << "  vview -help" << endl;
    cerr << endl;
  }
  else if (parseCommandLine(argc, argv) == false) return 1;

  mainLoop(argc, argv);
  return 0;
}

void vvView::cleanup()
{
  delete ds;
  ds = NULL;
}


//----------------------------------------------------------------------------
/// Main entry point.
int main(int argc, char** argv)
{
#ifdef VV_DEBUG_MEMORY
  int flag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);// Get current flag
  flag |= _CRTDBG_LEAK_CHECK_DF;                 // Turn on leak-checking bit
  flag |=  _CRTDBG_CHECK_ALWAYS_DF;
  _CrtSetDbgFlag(flag);                          // Set flag to the new value
#endif

#ifdef VV_DEBUG_MEMORY
  _CrtMemState s1, s2, s3;
  _CrtCheckMemory();
  _CrtMemCheckpoint( &s1 );
#endif

  // do stuff to test memory difference for

#ifdef VV_DEBUG_MEMORY
  _CrtMemCheckpoint( &s2 );
  if ( _CrtMemDifference( &s3, &s1, &s2 ) ) _CrtMemDumpStatistics( &s3 );
  _CrtCheckMemory();
#endif

  // do stuff to verify memory status after

#ifdef VV_DEBUG_MEMORY
  _CrtCheckMemory();
#endif

#ifndef FREEGLUT
  atexit(vvView::cleanup);
#endif

  //vvDebugMsg::setDebugLevel(vvDebugMsg::NO_MESSAGES);
  int error = (new vvView())->run(argc, argv);

#ifdef VV_DEBUG_MEMORY
  _CrtDumpMemoryLeaks();                         // display memory leaks, if any
#endif

  return error;
}

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
