// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
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

#include <GL/glew.h>

#include <boost/pointer_cast.hpp>

#include "vvcanvas.h"
#include "vvlightinteractor.h"

#include <virvo/vvdebugmsg.h>
#include <virvo/vvfileio.h>

#include <virvo/private/vvgltools.h>

#include <virvo/gl/util.h>

#include <QSettings>
#include <QTimer>
#include <QVector3D>

#include <iostream>

using virvo::mat4;
using virvo::vec2f;
using virvo::vec3f;
using virvo::vec4f;
using vox::vvObjView;

namespace gl = virvo::gl;


struct vvCanvas::Impl
{
  vvRenderState renderState;
  mat4 last_rotation;
};

vvCanvas::vvCanvas(const QGLFormat& format, const QString& filename, QWidget* parent)
  : QGLWidget(format, parent)
  , impl(new Impl)
  , _vd(NULL)
  , _renderer(NULL)
  , _projectionType(vox::vvObjView::PERSPECTIVE)
  , _doubleBuffering(format.doubleBuffer())
  , _lighting(false)
  , _headlight(false)
  , _superSamples(format.samples())
  , _stillQuality(1.0f)
  , _movingQuality(1.0f)
  , _spinAnimation(false)
  , _lightVisible(false)
  , _stereoMode(vox::Mono)
  , _swapEyes(false)
  , _mouseButton(Qt::NoButton)
  , _updateStencilBuffer(true)
{
  vvDebugMsg::msg(1, "vvCanvas::vvCanvas()");

  if (filename != "")
  {
    _vd = new vvVolDesc(filename.toStdString().c_str());
  }
  else
  {
    // load default volume
    _vd = new vvVolDesc;
    _vd->vox[0] = 32;
    _vd->vox[1] = 32;
    _vd->vox[2] = 32;
    _vd->frames = 0;
  }

  // init ui
  setMouseTracking(true);
  setFocusPolicy(Qt::StrongFocus);

  // read persistent settings
  QSettings settings;
  QColor qcolor = settings.value("canvas/bgcolor").value<QColor>();
  _bgColor = vvColor(qcolor.redF(), qcolor.greenF(), qcolor.blueF());

  _lighting = settings.value("canvas/lighting").toBool();
  _headlight = settings.value("canvas/headlight").toBool();

  // note: Qt 4.6 introduced QVector3D
  QVector3D qlightpos = settings.value("canvas/lightpos").value<QVector3D>();
  light_pos_ = vec3f(qlightpos.x(), qlightpos.y(), qlightpos.z());

  QVector3D qlightatt = settings.value("canvas/lightattenuation").value<QVector3D>();
  light_att_ = vec3f(qlightatt.x(), qlightatt.y(), qlightatt.z());

  _animTimer = new QTimer(this);
  connect(_animTimer, SIGNAL(timeout()), this, SLOT(incTimeStep()));

  _spinTimer = new QTimer(this);
  connect(_spinTimer, SIGNAL(timeout()), this, SLOT(repeatLastRotation()));

  if (!settings.value("stereo/distance").isNull())
  {
    _ov.setIOD(settings.value("stereo/distance").toFloat());
  }

  if (!settings.value("stereo/swap").isNull())
  {
    _swapEyes = settings.value("stereo/swap").toBool();
  }
}

vvCanvas::~vvCanvas()
{
  vvDebugMsg::msg(1, "vvCanvas::~vvCanvas()");

  delete _renderer;
  delete _vd;
}

void vvCanvas::setVolDesc(vvVolDesc* vd)
{
  vvDebugMsg::msg(3, "vvCanvas::setVolDesc()");

  delete _vd;
  _vd = vd;

  if (_renderer == 0)
  {
    createRenderer();
  }
  else
  {
    const float DEFAULT_OBJ_SIZE = 0.6f;
    _vd->resizeEdgeMax(_ov.getViewportWidth() * DEFAULT_OBJ_SIZE);
    _renderer->setVolDesc(vd);
  }

  foreach (vvPlugin* plugin, _plugins)
  {
    plugin->setVolDesc(_vd);
  }

  std::string str;
  if (_vd)
      _vd->makeInfoString(&str);
  else
      str = "No data";
  emit statusMessage(str);
  emit newVolDesc(_vd);
}

void vvCanvas::setPlugins(const QList<vvPlugin*>& plugins)
{
  vvDebugMsg::msg(3, "vvCanvas::setPlugins()");

  _plugins = plugins;
}

void vvCanvas::setInteractors(const QList<vvInteractor*>& interactors)
{
  _interactors = interactors;
}

vvVolDesc* vvCanvas::getVolDesc() const
{
  vvDebugMsg::msg(3, "vvCanvas::getVolDesc()");

  return _vd;
}

vvRenderer* vvCanvas::getRenderer() const
{
  vvDebugMsg::msg(3, "vvCanvas::getRenderer()");

  return _renderer;
}

const QList<vvInteractor*>& vvCanvas::getInteractors() const
{
  return _interactors;
}

void vvCanvas::loadCamera(const QString& filename)
{
  vvDebugMsg::msg(3, "vvCanvas::loadCamera()");

  QByteArray ba = filename.toLatin1();
  _ov.loadCamera(ba.data());
}

void vvCanvas::saveCamera(const QString& filename)
{
  vvDebugMsg::msg(3, "vvCanvas::saveCamera()");

  QByteArray ba = filename.toLatin1();
  _ov.saveCamera(ba.data());
}

void vvCanvas::initializeGL()
{
  vvDebugMsg::msg(1, "vvCanvas::initializeGL()");

  glewInit();
  init();
}

void vvCanvas::paintGL()
{
  vvDebugMsg::msg(3, "vvCanvas::paintGL()");

  if (!_renderer)
    return;

  bool isInterlacedLines = this->_stereoMode == vox::InterlacedLines;
  bool isInterlacedCheck = this->_stereoMode == vox::InterlacedCheckerboard;

  if (isInterlacedLines || isInterlacedCheck)
  {
    if (this->_updateStencilBuffer)
    {
      gl::renderInterlacedStereoStencilBuffer(isInterlacedLines);
      this->_updateStencilBuffer = false;
    }
  }

  if (_lighting)
  {
    vec4f lightpos;
    if (_headlight)
    {
      vec3f eyePos = _renderer->getEyePosition();

      lightpos = vec4f(eyePos, 1.0f);
    }
    else
    {
      lightpos = vec4f(light_pos_, 1.0f);
    }

    glEnable(GL_LIGHTING);
    glLightfv(GL_LIGHT0, GL_POSITION, &lightpos[0]);
    glLightfv(GL_LIGHT0, GL_CONSTANT_ATTENUATION, &light_att_[0]);
    glLightfv(GL_LIGHT0, GL_LINEAR_ATTENUATION, &light_att_[1]);
    glLightfv(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, &light_att_[2]);
  }

  glClearColor(_bgColor[0], _bgColor[1], _bgColor[2], 1.0f);
  glClearDepth(1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  GLint viewport[4] = {0};

  glGetIntegerv(GL_VIEWPORT, &viewport[0]);

#if 1
  int w = viewport[2];
  int h = viewport[3];
#else
  int w = viewport[2] / 8;
  int h = viewport[3] / 8;
#endif

  unsigned clearMask = virvo::CLEAR_COLOR | virvo::CLEAR_DEPTH;

  if (this->_stereoMode == vox::Mono)
  {
    render(w, h, vvObjView::CENTER, clearMask);
  }
  else
  {
    // render left image
    render(w, h, vvObjView::LEFT_EYE, clearMask);
    // render right image
    render(w, h, vvObjView::RIGHT_EYE, clearMask);
  }
}

void vvCanvas::resizeGL(int w, int h)
{
  vvDebugMsg::msg(3, "vvCanvas::resizeGL()");

  glViewport(0, 0, w, h);
  if (h > 0)
  {
    _ov.setAspectRatio(static_cast<float>(w) / static_cast<float>(h));
  }
  _updateStencilBuffer = true;
  updateGL();

  emit resized(QSize(w, h));
}

void vvCanvas::mouseMoveEvent(QMouseEvent* event)
{
  vvDebugMsg::msg(3, "vvCanvas::mouseMoveEvent()");

  // interactors
  foreach (vvInteractor* interactor, _interactors)
  {
    // first interactor with focus gets event
    if (interactor->enabled() && interactor->hasFocus())
    {
      interactor->mouseMoveEvent(event);
      // no more mouse processing
      return;
    }
  }

  // default mouse move event
  switch (_mouseButton)
  {
  case Qt::LeftButton:
  {
    impl->last_rotation = _ov._camera.trackballRotation(width(), height(),
      _lastMousePos.x(), _lastMousePos.y(),
      event->pos().x(), event->pos().y());
    if (_spinAnimation)
    {
      repeatLastRotation();
    }
    break;
  }
  case Qt::MidButton:
  {
    const float pixelInWorld = _ov.getViewportWidth() / static_cast<float>(width());
    const float dx = static_cast<float>(event->pos().x() - _lastMousePos.x());
    const float dy = static_cast<float>(event->pos().y() - _lastMousePos.y());
    vec2f pan(pixelInWorld * dx, pixelInWorld * dy);
    _ov._camera.translate(pan[0], -pan[1], 0.0f);
    break;
  }
  case Qt::RightButton:
  {
    const float factor = event->pos().y() - _lastMousePos.y();
    _ov._camera.translate(0.0f, 0.0f, factor);
    break;
  }
  default:
    break;
  }
  _lastMousePos = event->pos();
  updateGL();
}

void vvCanvas::mousePressEvent(QMouseEvent* event)
{
  vvDebugMsg::msg(3, "vvCanvas::mousePressEvent()");

  // interactors
  foreach (vvInteractor* interactor, _interactors)
  {
    // first interactor with focus gets event
    if (interactor->enabled() && interactor->hasFocus())
    {
      interactor->mousePressEvent(event);
      // no more mouse processing
      return;
    }
  }

  // default mouse press event
  _stillQuality = _renderer->getParameter(vvRenderer::VV_QUALITY);
  _renderer->setParameter(vvRenderer::VV_QUALITY, _movingQuality);
  _mouseButton = event->button();
  _lastMousePos = event->pos();
  impl->last_rotation = mat4::identity();
  if (_spinAnimation)
  {
    _spinTimer->stop();
  }
}

void vvCanvas::mouseReleaseEvent(QMouseEvent* event)
{
  vvDebugMsg::msg(3, "vvCanvas::mouseReleaseEvent()");

  // interactors
  foreach (vvInteractor* interactor, _interactors)
  {
    // first interactor with focus gets event
    if (interactor->enabled() && interactor->hasFocus())
    {
      interactor->mouseReleaseEvent(event);
      // no more mouse processing
      return;
    }
  }

  // default mouse release event
  _mouseButton = Qt::NoButton;
  _renderer->setParameter(vvRenderer::VV_QUALITY, _stillQuality);
  updateGL();
}

void vvCanvas::init()
{
  vvDebugMsg::msg(3, "vvCanvas::init()");

  vvFileIO fio;
  fio.loadVolumeData(_vd, vvFileIO::ALL_DATA);

  // default transfer function
  if (_vd->tf[0].isEmpty())
  {
    _vd->tf[0].setDefaultAlpha(0, _vd->range(0)[0], _vd->range(0)[1]);
    _vd->tf[0].setDefaultColors((_vd->getChan() == 1) ? 0 : 3, _vd->range(0)[0], _vd->range(0)[1]);
  }

  // init renderer
  if (_vd != NULL)
  {
    _currentRenderer = "default";
    _currentOptions["voxeltype"] = "arb";
    createRenderer();
  }

  updateProjection();

  foreach (vvPlugin* plugin, _plugins)
  {
    plugin->setVolDesc(_vd);
  }

  emit newVolDesc(_vd);
}

void vvCanvas::createRenderer()
{
  vvDebugMsg::msg(3, "vvCanvas::createRenderer()");

  if (_renderer)
  {
    impl->renderState = *_renderer;
    delete _renderer;
  }

  const float DEFAULT_OBJ_SIZE = 0.6f;
  _vd->resizeEdgeMax(_ov.getViewportWidth() * DEFAULT_OBJ_SIZE);

  _renderer = vvRendererFactory::create(_vd, impl->renderState, _currentRenderer.c_str(), _currentOptions);

  // set boundary color to inverse of background
  vvColor invColor;
  if (_bgColor[0] + _bgColor[1] + _bgColor[2] > 1.5f)
  {
    invColor = vvColor(0.0f, 0.0f, 0.0f);
  }
  else
  {
    invColor = vvColor(1.0f, 1.0f, 1.0f);
  }
  _renderer->setParameter(vvRenderState::VV_BOUND_COLOR, invColor);
  _renderer->setParameter(vvRenderState::VV_CLIP_COLOR, invColor);

  emit rendererChanged(_renderer);
}

void vvCanvas::updateProjection()
{
  vvDebugMsg::msg(3, "vvCanvas::updateProjection()");

  if (_projectionType == vvObjView::PERSPECTIVE)
  {
    _ov.setProjection(vvObjView::PERSPECTIVE, vvObjView::DEF_FOV, vvObjView::DEF_CLIP_NEAR, vvObjView::DEF_CLIP_FAR);
  }
  else if (_projectionType == vvObjView::ORTHO)
  {
    _ov.setProjection(vvObjView::ORTHO, vvObjView::DEF_VIEWPORT_WIDTH, vvObjView::DEF_CLIP_NEAR, vvObjView::DEF_CLIP_FAR);
  }
}

void vvCanvas::setCurrentFrame(size_t frame)
{
  vvDebugMsg::msg(3, "vvCanvas::setCurrentFrame()");

  _renderer->setCurrentFrame(frame);
  emit currentFrame(frame);

  // inform plugins of new frame
  foreach (vvPlugin* plugin, _plugins)
  {
    plugin->timestep();
  }

  updateGL();
}

void vvCanvas::render(int w, int h, unsigned eye, unsigned clearMask)
{
  // Set projection matrix
  updateProjection();
  // Set model-view matrix
  this->_ov.setModelviewMatrix(static_cast<vvObjView::EyeType>(eye));

  // Prepare (stereo) rendering...
  initRendering(eye);

  // Resize the render target
  if (!this->_renderer->resize(w, h))
  {
  }

  // Start rendering
  if (!this->_renderer->beginFrame(clearMask))
  {
  }

  // Render opaque geometry into framebuffer
  // this->_renderer->renderOpaqueGeometry();

  foreach (vvPlugin* plugin, _plugins)
  {
    if (plugin->isActive())
      plugin->prerender();
  }

  // Render the volume
  this->_renderer->renderVolumeGL();

  foreach (vvPlugin* plugin, _plugins)
  {
    if (plugin->isActive())
      plugin->postrender();
  }

  // Stop rendering
  if (!this->_renderer->endFrame())
  {
  }

  // Display the rendered image
  finishRendering();

  foreach (vvInteractor* interactor, _interactors)
  {
    if (interactor->enabled() && interactor->visible())
      interactor->render();
  }

  // Render palette etc...
  this->_renderer->renderHUD();
}

void vvCanvas::initRendering(unsigned eye)
{
  bool left = (eye == vvObjView::LEFT_EYE);

  if (this->_swapEyes)
    left = !left;

  switch (this->_stereoMode)
  {
  case vox::Mono:
    break;

  case vox::InterlacedLines:
  case vox::InterlacedCheckerboard:
    initStereoInterlaced(left);
    break;

  case vox::RedCyan:
    initStereoRedCyan(left);
    break;

  case vox::SideBySide:
    initStereoSideBySide(left);
    break;
  }
}

void vvCanvas::initStereoInterlaced(bool left)
{
  assert( this->_stereoMode == vox::InterlacedLines || this->_stereoMode == vox::InterlacedCheckerboard );

  glEnable(GL_STENCIL_TEST);
  glStencilFunc(GL_EQUAL, left ? 0x01 : 0x00, 0xFFFFFFFF);
}

void vvCanvas::initStereoRedCyan(bool left)
{
  assert( this->_stereoMode == vox::RedCyan );

  if (left)
    glColorMask(1,0,0,0);
  else
    glColorMask(0,1,1,0);
}

void vvCanvas::initStereoSideBySide(bool left)
{
  assert( this->_stereoMode == vox::SideBySide );

  int w = width();
  int h = height();

  if (left)
  {
    this->_ov.setAspectRatio((static_cast<float>(w / 2)) / static_cast<float>(h));
    glViewport(0, 0, w / 2, h);
  }
  else
  {
    this->_ov.setAspectRatio((static_cast<float>(w - w / 2)) / static_cast<float>(h));
    glViewport(w / 2, 0, w - w / 2, h);
  }
}

void vvCanvas::finishRendering()
{
  // Blend the rendered volume into the current draw buffer
  this->_renderer->present();

  int w = width();
  int h = height();

  switch (this->_stereoMode)
  {
  case vox::Mono:
    break;

  case vox::InterlacedLines:
  case vox::InterlacedCheckerboard:
    glDisable(GL_STENCIL_TEST);
    break;

  case vox::RedCyan:
    glColorMask(1,1,1,1);
    break;

  case vox::SideBySide:
    this->_ov.setAspectRatio(static_cast<float>(w) / static_cast<float>(h));
    glViewport(0, 0, w, h);
    break;
  }
}

void vvCanvas::setRenderer(const std::string& name, const vvRendererFactory::Options& options)
{
  vvDebugMsg::msg(3, "vvCanvas::setRenderer()");

  _currentRenderer = name;
  _currentOptions = options;
  createRenderer();
  updateGL();
}

void vvCanvas::setParameter(vvParameters::ParameterType param, const vvParam& value)
{
  vvDebugMsg::msg(3, "vvCanvas::setParameter()");

  switch (param)
  {
  case vvParameters::VV_BG_COLOR:
    _bgColor = value;
    break;
  case vvParameters::VV_DOUBLEBUFFERING:
    _doubleBuffering = value;
    break;
  case vvParameters::VV_LIGHTING:
    {
      _lighting = value;
      foreach (vvInteractor* interactor, _interactors)
      {
        vvLightInteractor* li = dynamic_cast<vvLightInteractor*>(interactor);
        if (li != NULL)
        {
          li->setLightingEnabled(_lighting);
        }
      }
    }
    break;
  case vvParameters::VV_MOVING_QUALITY:
    _movingQuality = value;
    break;
  case vvParameters::VV_SUPERSAMPLES:
    _superSamples = value;
    break;
  case vvParameters::VV_PROJECTIONTYPE:
    _projectionType = static_cast<vvObjView::ProjectionType>(value.asInt());
    updateProjection();
    break;
  case vvParameters::VV_SPIN_ANIMATION:
    _spinAnimation = value;
    break;
  case vvParameters::VV_STEREO_MODE:
    _stereoMode = static_cast<vox::StereoMode>(value.asInt());
    if (_stereoMode == vox::InterlacedLines || _stereoMode == vox::InterlacedCheckerboard)
    {
      _updateStencilBuffer = true;
    }
    break;
  case vvParameters::VV_EYE_DIST:
    _ov.setIOD(value);
    break;
  case vvParameters::VV_SWAP_EYES:
    _swapEyes = value;
    break;
  default:
    break;
  }
  updateGL();
}

void vvCanvas::setParameter(vvRenderer::ParameterType param, const vvParam& value)
{
  vvDebugMsg::msg(3, "vvCanvas::setParameter()");

  impl->renderState.setParameter(param, value);
  if (_renderer != NULL)
  {
    _renderer->setParameter(param, value);
    updateGL();
  }
}

vvParam vvCanvas::getParameter(vvParameters::ParameterType param) const
{
  switch (param)
  {
  case vvParameters::VV_BG_COLOR:
    return _bgColor;
  case vvParameters::VV_DOUBLEBUFFERING:
    return _doubleBuffering;
  case vvParameters::VV_LIGHTING:
    return _lighting;
  case vvParameters::VV_SUPERSAMPLES:
    return _superSamples;
  case vvParameters::VV_PROJECTIONTYPE:
    return static_cast<int>(_projectionType);
  case vvParameters::VV_SPIN_ANIMATION:
    return _spinAnimation;
  case vvParameters::VV_STEREO_MODE:
    return _stereoMode;
  case vvParameters::VV_EYE_DIST:
    return _ov.getIOD();
  case vvParameters::VV_SWAP_EYES:
    return _swapEyes;
  default:
    return vvParam();
  }
}

vvParam vvCanvas::getParameter(vvRenderer::ParameterType param) const
{
  if (_renderer != NULL)
  {
    return _renderer->getParameter(param);
  }
  return vvParam();
}

void vvCanvas::addTFWidget(vvTFWidget* widget)
{
  _vd->tf[0].putUndoBuffer();
  _vd->tf[0]._widgets.push_back(widget);
  updateTransferFunction();
}

void vvCanvas::updateTransferFunction()
{
  _renderer->updateTransferFunction();
  updateGL();
}

void vvCanvas::undoTransferFunction()
{
  _vd->tf[0].getUndoBuffer();
}

void vvCanvas::startAnimation(const double fps)
{
  vvDebugMsg::msg(3, "vvCanvas::startAnimation()");

  _vd->setDt(1.0f / static_cast<float>(fps));
  const float delay = std::abs(_vd->getDt() * 1000.0f);
  _animTimer->start(static_cast<int>(delay));
}

void vvCanvas::stopAnimation()
{
  vvDebugMsg::msg(3, "vvCanvas::stopAnimation()");

  _animTimer->stop();
}

void vvCanvas::setTimeStep(int step)
{
  vvDebugMsg::msg(3, "vvCanvas::setTimeStep()");

  size_t f = step;
  while (f >= _vd->frames)
  {
    f -= _vd->frames;
  }

  setCurrentFrame(f);
}

void vvCanvas::incTimeStep()
{
  vvDebugMsg::msg(3, "vvCanvas::incTimeStep()");

  size_t f = _renderer->getCurrentFrame();
  f = (f >= _vd->frames - 1) ? 0 : f + 1;
  setCurrentFrame(f);
}

void vvCanvas::decTimeStep()
{
  vvDebugMsg::msg(3, "vvCanvas::decTimeStep()");

  size_t f = _renderer->getCurrentFrame();
  f = (f == 0) ? _vd->frames - 1 : f - 1;
  setCurrentFrame(f);
}

void vvCanvas::firstTimeStep()
{
  vvDebugMsg::msg(3, "vvCanvas::firstTimeStep()");

  setCurrentFrame(0);
}

void vvCanvas::lastTimeStep()
{
  vvDebugMsg::msg(3, "vvCanvas::lastTimeStep()");

  setCurrentFrame(_vd->frames - 1);
}

void vvCanvas::enableClipping(bool enabled)
{
  setParameter(vvRenderState::VV_CLIP_OBJ_ACTIVE0, enabled);
}

void vvCanvas::setClipNormal(vec3f const& n)
{
  boost::shared_ptr<vvClipObj> clip_obj = getParameter(vvRenderState::VV_CLIP_OBJ0).asClipObj();
  boost::shared_ptr<vvClipPlane> plane = boost::dynamic_pointer_cast<vvClipPlane>(clip_obj);

  if (plane == 0)
  {
    plane = vvClipPlane::create();
  }
  plane->normal = n;

  setParameter(vvRenderState::VV_CLIP_OBJ0, plane);
}

void vvCanvas::setClipOrigin(vec3f const& o)
{
  boost::shared_ptr<vvClipObj> clip_obj = getParameter(vvRenderState::VV_CLIP_OBJ0).asClipObj();
  boost::shared_ptr<vvClipPlane> plane = boost::dynamic_pointer_cast<vvClipPlane>(clip_obj);

  if (plane == 0)
  {
    plane = vvClipPlane::create();
  }
  plane->offset = dot(o, plane->normal);

  setParameter(vvRenderState::VV_CLIP_OBJ0, plane);
}

void vvCanvas::setClipSingleSlice(bool active)
{
  setParameter(vvRenderState::VV_CLIP_SINGLE_SLICE, active);
}

void vvCanvas::setClipOpaque(bool active)
{
  setParameter(vvRenderState::VV_CLIP_OPAQUE, active);
}

void vvCanvas::setClipPerimeter(bool active)
{
  setParameter(vvRenderState::VV_CLIP_OUTLINE0, active);
}

void vvCanvas::enableLighting(bool enabled)
{
  setParameter(vvRenderState::VV_LIGHTING, enabled);
  setParameter(vvParameters::VV_LIGHTING, enabled);
  updateGL();
}

void vvCanvas::showLightSource(bool show)
{
  _lightVisible = show;

  // canvas lazily uses the light source interactor to visualize the light source
  vvLightInteractor* li = NULL;
  foreach (vvInteractor* interactor, _interactors)
  {
    if (dynamic_cast<vvLightInteractor*>(interactor) != NULL)
    {
      li = static_cast<vvLightInteractor*>(interactor);
    }
  }

  if (li == NULL)
  {
    li = new vvLightInteractor;
    _interactors.append(li);
  }

  li->setEnabled(true);
  li->setLightingEnabled(getParameter(vvParameters::VV_LIGHTING));
  li->setPos(light_pos_);

  if (!li->hasFocus())
  {
    li->setVisible(_lightVisible);
  }

  updateGL();
}

void vvCanvas::enableHeadlight(bool enable)
{
  _headlight = enable;

  updateGL();
}

void vvCanvas::editLightPosition(bool edit)
{
  if (edit)
  {
    vvLightInteractor* li = NULL;
    foreach (vvInteractor* interactor, _interactors)
    {
      if (dynamic_cast<vvLightInteractor*>(interactor) != NULL)
      {
        li = static_cast<vvLightInteractor*>(interactor);
      }
    }

    if (li == NULL)
    {
      li = new vvLightInteractor;
      _interactors.append(li);
    }

    li->setFocus();
    li->setVisible(true);
    li->setLightingEnabled(getParameter(vvParameters::VV_LIGHTING));
    li->setPos(light_pos_);
    connect(li, SIGNAL(lightPos(virvo::vec3f const&)), this, SLOT(setLightPos(virvo::vec3f const&)));
  }
  else
  {
    foreach (vvInteractor* interactor, _interactors)
    {
      if (dynamic_cast<vvLightInteractor*>(interactor) != NULL)
      {
        interactor->clearFocus();
        interactor->setVisible(_lightVisible);
      }
    }
  }
  updateGL();
}

void vvCanvas::setLightAttenuation(vec3f const& att)
{
  light_att_ = att;

  QVector3D qatt(att[0], att[1], att[2]);
  QSettings settings;
  settings.setValue("canvas/lightattenuation", qatt);

  updateGL();
}

void vvCanvas::resetCamera()
{
  _ov.resetObject();
  _ov.resetCamera();

  updateGL();
}

void vvCanvas::repeatLastRotation()
{
  vvDebugMsg::msg(3, "vvCanvas::repeatLastRotation()");

  _ov._camera.multiplyRight(impl->last_rotation);
  updateGL();

  const float spindelay = 0.05f;
  const float delay = std::abs(spindelay * 1000.0f);
  _spinTimer->start(static_cast<int>(delay));
}

void vvCanvas::setLightPos(vec3f const& pos)
{
  light_pos_ = pos;

  QVector3D qpos(pos[0], pos[1], pos[2]);
  QSettings settings;
  settings.setValue("canvas/lightpos", qpos);

  updateGL();
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
