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
#include <QUrl>

// Make sure that winsock2.h is included before windows:
#include <virvo/vvplatform.h>

#include "vvcanvas.h"
#include "vvprefdialog.h"
#include "vvstereomode.h"

#include "ui_vvprefdialog.h"

#include <virvo/vvbonjour/vvbonjour.h>
#include <virvo/vvdebugmsg.h>
#include <virvo/vvremoteevents.h>
#include <virvo/vvshaderfactory.h>
#ifndef DESKVOX_USE_ASIO
#include <virvo/vvsocketio.h>
#include <virvo/vvsocketmap.h>
#endif
#include <virvo/vvtoolshed.h>
#include <virvo/vvvirvo.h>

#include <QMessageBox>
#include <QSettings>
#include <QValidator>

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <utility>

#define VV_UNUSED(x) ((void)(x))

struct vvPrefDialog::Impl
{
  Impl()
#ifdef DESKVOX_USE_ASIO
    : movingSpinBoxOldValue(1.0)
#else
    : sock(NULL)
    , movingSpinBoxOldValue(1.0)
#endif
    , stillSpinBoxOldValue(1.0)
    , movingDialOldValue(0.0)
    , stillDialOldValue(0.0)
    , num_interpol_algs(0)
  {
  }

  std::map<int, vvRenderer::RendererType> rendererMap;
  std::map<int, std::string> texRendTypeMap;
  std::map<int, std::string> voxTypeMap;
  std::map<int, int> fboPrecisionMap;
  std::map<int, std::string> rayRendArchMap;
  std::map<int, vox::StereoMode> stereoModeMap;
  std::map<std::string, std::string> rendererDescriptions;
  std::map<std::string, std::string> algoDescriptions;

  // e.g. ibr or image
  std::string remoterend;
#ifndef DESKVOX_USE_ASIO
  vvTcpSocket* sock;
#endif

  double movingSpinBoxOldValue;
  double stillSpinBoxOldValue;
  int movingDialOldValue;
  int stillDialOldValue ;

  int num_interpol_algs;
};

/* make qt dials behave as if they had an unlimited range
 */
int getDialDelta(int oldval, int newval, int minval, int maxval)
{
  const int eps = 10; // largest possible step from a single user action
  const int mineps = minval + eps;
  const int maxeps = maxval - eps;

  if (oldval < mineps && newval > maxeps)
  {
    return -(oldval + maxval + 1 - newval);
  }
  else if (oldval > maxeps && newval < mineps)
  {
    return maxval + 1 - oldval + newval;
  }
  else
  {
    return newval - oldval;
  }
}


vvPrefDialog::vvPrefDialog(vvCanvas* canvas, QWidget* parent)
  : QDialog(parent)
  , ui(new Ui_PrefDialog)
  , _canvas(canvas)
  , impl(new Impl)
{
  ui->setupUi(this);

  // can't be done in designer unfortunately
  QIcon ic = style()->standardIcon(QStyle::SP_MessageBoxInformation);
  ui->texInfoIconLabel->setPixmap(ic.pixmap(32, 32));

  _canvas->makeCurrent();
  glewInit(); // we need glCreateProgram etc. when checking for glsl support

  impl->rendererDescriptions.insert(std::make_pair("slices", "OpenGL textures"));
  impl->rendererDescriptions.insert(std::make_pair("cubic2d", "OpenGL textures"));
  impl->rendererDescriptions.insert(std::make_pair("planar", "OpenGL textures"));
  impl->rendererDescriptions.insert(std::make_pair("spherical", "OpenGL textures"));
  impl->rendererDescriptions.insert(std::make_pair("rayrend", "Ray casting"));

  impl->algoDescriptions.insert(std::make_pair("default", "Autoselect"));
  impl->algoDescriptions.insert(std::make_pair("slices", "2D textures (slices)"));
  impl->algoDescriptions.insert(std::make_pair("cubic2d", "2D textures (cubic)"));
  impl->algoDescriptions.insert(std::make_pair("planar", "3D textures (viewport aligned)"));
  impl->algoDescriptions.insert(std::make_pair("spherical", "3D textures (spherical)"));

  // renderer combo box
  int idx = 0;
  if (vvRendererFactory::hasRenderer(vvRenderer::TEXREND))
  {
    ui->rendererBox->addItem(impl->rendererDescriptions["slices"].c_str());
    impl->rendererMap.insert(std::make_pair(idx, vvRenderer::TEXREND));
    ++idx;
  }

  if (vvRendererFactory::hasRenderer(vvRenderer::RAYREND))
  {
    ui->rendererBox->addItem(impl->rendererDescriptions["rayrend"].c_str());
    impl->rendererMap.insert(std::make_pair(idx, vvRenderer::RAYREND));
    ++idx;
  }

  if (ui->rendererBox->count() <= 0)
  {
    ui->rendererBox->setEnabled(false);
  }

  // texrend geometry combo box
  idx = 0;

  if (vvRendererFactory::hasRenderer(vvRenderer::TEXREND))
  {
    ui->geometryBox->addItem(impl->algoDescriptions["default"].c_str());
    impl->texRendTypeMap.insert(std::make_pair(idx, "default"));
    ++idx;
  }

  if (vvRendererFactory::hasRenderer("slices"))
  {
    ui->geometryBox->addItem(impl->algoDescriptions["slices"].c_str());
    impl->texRendTypeMap.insert(std::make_pair(idx, "slices"));
    ++idx;
  }

  if (vvRendererFactory::hasRenderer("cubic2d"))
  {
    ui->geometryBox->addItem(impl->algoDescriptions["cubic2d"].c_str());
    impl->texRendTypeMap.insert(std::make_pair(idx, "cubic2d"));
    ++idx;
  }

  if (vvRendererFactory::hasRenderer("planar"))
  {
    ui->geometryBox->addItem(impl->algoDescriptions["planar"].c_str());
    impl->texRendTypeMap.insert(std::make_pair(idx, "planar"));
    ++idx;
  }

  if (vvRendererFactory::hasRenderer("spherical"))
  {
    ui->geometryBox->addItem(impl->algoDescriptions["spherical"].c_str());
    impl->texRendTypeMap.insert(std::make_pair(idx, "spherical"));
    ++idx;
  }

  // voxel type combo box
  idx = 0;

  if (vvRendererFactory::hasRenderer(vvRenderer::TEXREND))
  {
    ui->voxTypeBox->addItem("Autoselect");
    impl->voxTypeMap.insert(std::make_pair(idx, "default"));
    ++idx;
  }

  if (vvRendererFactory::hasRenderer(vvRenderer::TEXREND))
  {
    ui->voxTypeBox->addItem("RGBA");
    impl->voxTypeMap.insert(std::make_pair(idx, "rgba"));
    ++idx;
  }

  if (vvRendererFactory::hasRenderer(vvRenderer::TEXREND))
  {
    ui->voxTypeBox->addItem("ARB fragment program");
    impl->voxTypeMap.insert(std::make_pair(idx, "arb"));
    ++idx;
  }

  if (vvRendererFactory::hasRenderer(vvRenderer::TEXREND) && vvShaderFactory::isSupported("glsl"))
  {
    ui->voxTypeBox->addItem("GLSL fragment program");
    impl->voxTypeMap.insert(std::make_pair(idx, "shader"));
    ++idx;
  }

  // fbo combo box
  idx = 0;

  ui->fboBox->addItem("None");
  impl->fboPrecisionMap.insert(std::make_pair(idx, 0));
  ++idx;

  ui->fboBox->addItem("8 bit precision");
  impl->fboPrecisionMap.insert(std::make_pair(idx, 8));
  ++idx;

  ui->fboBox->addItem("16 bit precision");
  impl->fboPrecisionMap.insert(std::make_pair(idx, 16));
  ++idx;

  ui->fboBox->addItem("32 bit precision");
  impl->fboPrecisionMap.insert(std::make_pair(idx, 32));
  ++idx;

  // ray rend architecture combo box
  idx = 0;

  if (vvRendererFactory::hasRenderer("rayrend", "cuda"))
  {
    ui->rayRendArchBox->addItem("CUDA - GPGPU ray casting");
    impl->rayRendArchMap.insert(std::make_pair(idx, "cuda"));
    ++idx;
  }

  if (vvRendererFactory::hasRenderer("rayrend", "fpu"))
  {
    ui->rayRendArchBox->addItem("FPU - CPU ray casting");
    impl->rayRendArchMap.insert(std::make_pair(idx, "fpu"));
    ++idx;
  }

  if (vvRendererFactory::hasRenderer("rayrend", "sse2"))
  {
    ui->rayRendArchBox->addItem("SSE 2 - optimized CPU ray casting");
    impl->rayRendArchMap.insert(std::make_pair(idx, "sse2"));
    ++idx;
  }

  if (vvRendererFactory::hasRenderer("rayrend", "sse4_1"))
  {
    ui->rayRendArchBox->addItem("SSE 4.1 - optimized CPU ray casting");
    impl->rayRendArchMap.insert(std::make_pair(idx, "sse4_1"));
    ++idx;
  }

  if (vvRendererFactory::hasRenderer("rayrend", "avx"))
  {
    ui->rayRendArchBox->addItem("AVX - optimized CPU ray casting");
    impl->rayRendArchMap.insert(std::make_pair(idx, "avx"));
    ++idx;
  }

  if (vvRendererFactory::hasRenderer("rayrend", "avx2"))
  {
    ui->rayRendArchBox->addItem("AVX 2 - optimized CPU ray casting");
    impl->rayRendArchMap.insert(std::make_pair(idx, "avx2"));
    ++idx;
  }

  // stereo mode combo box
  idx = 0;

  ui->stereoModeBox->addItem("Off (Mono)");
  impl->stereoModeMap.insert(std::make_pair(idx, vox::Mono));
  ++idx;

  if (_canvas->format().stencil())
  {
    ui->stereoModeBox->addItem("Interlaced (Lines)");
    impl->stereoModeMap.insert(std::make_pair(idx, vox::InterlacedLines));
    ++idx;
  }

  if (_canvas->format().stencil())
  {
    ui->stereoModeBox->addItem("Interlaced (Checkerboard)");
    impl->stereoModeMap.insert(std::make_pair(idx, vox::InterlacedCheckerboard));
    ++idx;
  }

  ui->stereoModeBox->addItem("Red cyan");
  impl->stereoModeMap.insert(std::make_pair(idx, vox::RedCyan));
  ++idx;

  ui->stereoModeBox->addItem("Side by side");
  impl->stereoModeMap.insert(std::make_pair(idx, vox::SideBySide));
  ++idx;


  // remote rendering page
  if (virvo::hasFeature("bonjour"))
  {
    ui->browseButton->setEnabled(true);
  }

  QIntValidator* val = new QIntValidator(this);
  val->setRange(ui->stereoDistSlider->minimum(), ui->stereoDistSlider->maximum());
  ui->stereoDistEdit->setValidator(val);
  ui->stereoDistEdit->setText(QString::number(ui->stereoDistSlider->value()));

  connect(ui->rendererBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onRendererChanged(int)));
  connect(ui->geometryBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onTexRendOptionChanged(int)));
  connect(ui->fboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onFboChanged(int)));
  connect(ui->voxTypeBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onTexRendOptionChanged(int)));
  connect(ui->pixShdBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onTexRendOptionChanged(int)));
  connect(ui->rayRendArchBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onRayRendArchChanged(int)));
  connect(ui->earlyRayBox, SIGNAL(toggled(bool)), this, SLOT(onEarlyRayTerminationToggled(bool)));
  connect(ui->hostEdit, SIGNAL(textChanged(const QString&)), this, SLOT(onHostChanged(const QString&)));
  connect(ui->portBox, SIGNAL(valueChanged(int)), this, SLOT(onPortChanged(int)));
  connect(ui->getInfoButton, SIGNAL(clicked()), this, SLOT(onGetInfoClicked()));
  connect(ui->connectButton, SIGNAL(clicked()), this, SLOT(onConnectClicked()));
  connect(ui->ibrBox, SIGNAL(toggled(bool)), this, SLOT(onIbrToggled(bool)));
  connect(ui->interpolationBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onInterpolationChanged(int)));
  connect(ui->mipCheckBox, SIGNAL(toggled(bool)), this, SLOT(onMipToggled(bool)));
  connect(ui->preIntegrationCheckBox, SIGNAL(toggled(bool)), this, SLOT(onPreIntegrationToggled(bool)));
  connect(ui->stereoModeBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onStereoModeChanged(int)));
  connect(ui->stereoDistEdit, SIGNAL(textEdited(const QString&)), this, SLOT(onStereoDistEdited(const QString&)));
  connect(ui->stereoDistSlider, SIGNAL(sliderMoved(int)), this, SLOT(onStereoDistSliderMoved(int)));
  connect(ui->stereoDistSlider, SIGNAL(valueChanged(int)), this, SLOT(onStereoDistChanged(int)));
  connect(ui->swapEyesBox, SIGNAL(toggled(bool)), this, SLOT(onSwapEyesToggled(bool)));
  connect(ui->movingSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onMovingSpinBoxChanged(double)));
  connect(ui->stillSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onStillSpinBoxChanged(double)));
  connect(ui->movingDial, SIGNAL(valueChanged(int)), this, SLOT(onMovingDialChanged(int)));
  connect(ui->stillDial, SIGNAL(valueChanged(int)), this, SLOT(onStillDialChanged(int)));
}

vvPrefDialog::~vvPrefDialog()
{
#ifndef DESKVOX_USE_ASIO
  if (impl->sock!= NULL)
  {
    vvSocketMap::remove(vvSocketMap::getIndex(impl->sock));
  }
  delete impl->sock;
#endif
  delete impl;
}

void vvPrefDialog::applySettings()
{
  QSettings settings;

  const char* rend = getenv("VV_RENDERER");

  if (!rend)
    ui->rendererBox->setCurrentIndex(settings.value("renderer/type").toInt());

  ui->rayRendArchBox->setCurrentIndex(settings.value("rayrend/arch").toInt());

  ui->hostEdit->setText(settings.value("remote/host").toString());
  if (settings.value("remote/port").toString() != "")
  {
    int port = settings.value("remote/port").toInt();
    ui->portBox->setValue(port);
  }
  ui->ibrBox->setChecked(settings.value("remote/ibr").toBool());

  if (!settings.value("appearance/interpolation").isNull())
  {
    ui->interpolationBox->setCurrentIndex(settings.value("appearance/interpolation").toInt());
  }

  if (!settings.value("appearance/preintegration").isNull())
  {
    ui->preIntegrationCheckBox->setChecked(settings.value("appearance/preintegration").toBool());
  }

  if (!settings.value("stereo/distance").isNull())
  {
    int dist = settings.value("stereo/distance").toInt();
    ui->stereoDistEdit->setText(QString::number(dist));
    ui->stereoDistSlider->setValue(dist);
  }

  if (!settings.value("stereo/swap").isNull())
  {
    ui->swapEyesBox->setChecked(settings.value("stereo/swap").toBool());
  }
}

void vvPrefDialog::toggleInterpolation()
{
  int filter_mode = ui->interpolationBox->currentIndex();
  ++filter_mode;
  filter_mode %= impl->num_interpol_algs;
  ui->interpolationBox->setCurrentIndex(filter_mode);
  emit parameterChanged(vvRenderer::VV_SLICEINT, static_cast< virvo::tex_filter_mode >(filter_mode));
}

void vvPrefDialog::scaleStillQuality(const float s)
{
  vvDebugMsg::msg(3, "vvPrefDialog::scaleStillQuality()");

  assert(s >= 0.0f);

  float quality = static_cast<float>(ui->stillSpinBox->value());
  if (quality <= 0.0f)
  {
    // never let quality drop to or below 0
    quality = std::numeric_limits<float>::epsilon();
  }
  quality *= s;

  ui->stillSpinBox->setValue(quality);
}

void vvPrefDialog::emitRenderer()
{
  vvDebugMsg::msg(3, "vvPrefDialog::emitRenderer()");

  ui->texInfoLabel->setText("");
  std::string name = "";

  vvRendererFactory::Options options;

  if (impl->remoterend == "ibr" || impl->remoterend == "image")
  {
#ifdef DESKVOX_USE_ASIO
    options["host"] = ui->hostEdit->text().toStdString();
    options["port"] = ui->portBox->text().toStdString();

    name = impl->remoterend;
#else
    int s = vvSocketMap::add(impl->sock);
    std::stringstream sockstr;
    sockstr << s;
    if (sockstr.str() != "")
    {
      name = impl->remoterend;
      options["sockets"] = sockstr.str();
    }
#endif
  }
  else
  {
    switch (impl->rendererMap[ui->rendererBox->currentIndex()])
    {
    case vvRenderer::RAYREND:
      name = "rayrend";
      options["arch"] = impl->rayRendArchMap[ui->rayRendArchBox->currentIndex()];
      break;
    case vvRenderer::TEXREND:
      name = impl->texRendTypeMap[ui->geometryBox->currentIndex()];
      options["voxeltype"] = impl->voxTypeMap[ui->voxTypeBox->currentIndex()];
      break;
    default:
      name = "default";
      break;
    }

    updateUi();

  }

  if (name != "")
  {
    emit rendererChanged(name, options);
  }
}

bool vvPrefDialog::validateRemoteHost(const QString& host, const ushort port)
{
  int parsedPort = vvToolshed::parsePort(host.toStdString());
  if (parsedPort >= 0 && parsedPort <= std::numeric_limits<ushort>::max()
   && static_cast<ushort>(parsedPort) != port)
  {
    ui->portBox->setValue(parsedPort);
  }

  std::string h = (parsedPort == -1)
    ? host.toStdString()
    : vvToolshed::stripPort(host.toStdString());
  ushort p = static_cast<ushort>(ui->portBox->value());

  if (h == "")
  {
    return false;
  }

  QUrl url(h.c_str());
  url.setPort(p);
  return url.isValid();
}


void vvPrefDialog::updateUi()
{

  //
  // deactivate ui
  // ----------------------------
  //

  disconnect(ui->rendererBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onRendererChanged(int)));
  disconnect(ui->interpolationBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onInterpolationChanged(int)));



  vvRenderer* renderer = _canvas->getRenderer();

  // indices to activate appropriate options tool box pages
  static const int TexIdx = 0;
  static const int RayIdx = 1;


  //
  // renderer tab
  // ----------------------------
  //

  std::string voxeltype = "";

  vvRenderer::RendererType rt = renderer->getRendererType();


  switch (rt)
  {
  case vvRenderer::RAYREND:

    ui->rendererBox->setCurrentIndex(RayIdx);
    ui->optionsToolBox->setCurrentIndex(RayIdx);
    break;

  case vvRenderer::TEXREND:

    ui->rendererBox->setCurrentIndex(TexIdx);
    ui->optionsToolBox->setCurrentIndex(TexIdx);
    voxeltype = impl->voxTypeMap[ui->voxTypeBox->currentIndex()];
    break;

  default:

    break;

  }


  //
  // texrend tab
  // ----------------------------
  //

  if (voxeltype == "rgba")
  {
    ui->texInfoLabel->setText(ui->texInfoLabel->text() + "<html><b>Voxel type RGBA</b><br />"
      "Pre-interpolative transfer function,"
      " is applied by assigning each voxel an RGBA color before rendering.</html>");
  }
  else if (voxeltype == "arb")
  {
    ui->texInfoLabel->setText(ui->texInfoLabel->text() + "<html><b>Voxel type ARB fragment program</b><br />"
      "Post-interpolative transfer function,"
      " is applied after sampling the volume texture.</html>");
  }
  else if (voxeltype == "shader")
  {
    ui->texInfoLabel->setText(ui->texInfoLabel->text() + "<html><b>Voxel type GLSL fragment program</b><br />"
      "Post-interpolative transfer function,"
      " is applied after sampling the volume texture.</html>");
  }


  //
  // appearance tab
  // ----------------------------
  //
  
  // determine interpolation modes

  ui->interpolationBox->clear();
  impl->num_interpol_algs = 0;

  int ipol = renderer->getParameter(vvRenderState::VV_SLICEINT).asInt();

  if (renderer->checkParameter(vvRenderState::VV_SLICEINT, virvo::Nearest))
  {
    ui->interpolationBox->addItem("Nearest Neighbor");
    ++impl->num_interpol_algs;
  }

  if (renderer->checkParameter(vvRenderState::VV_SLICEINT, virvo::Linear))
  {
    ui->interpolationBox->addItem("Linear");
    ++impl->num_interpol_algs;
  }

  if (renderer->checkParameter(vvRenderState::VV_SLICEINT, virvo::BSpline))
  {
    ui->interpolationBox->addItem("Cubic B-Spline");
    ++impl->num_interpol_algs;
  }

  if (renderer->checkParameter(vvRenderState::VV_SLICEINT, virvo::BSpline))
  {
    ui->interpolationBox->addItem("Cubic B-Spline Interpolation");
    ++impl->num_interpol_algs;
  }

  if (renderer->checkParameter(vvRenderState::VV_SLICEINT, virvo::CardinalSpline))
  {
    ui->interpolationBox->addItem("Cubic Cardinal Spline");
    ++impl->num_interpol_algs;
  }

  ui->interpolationBox->setCurrentIndex(std::min( ipol, impl->num_interpol_algs ));


  //
  // reactivate ui
  // ----------------------------
  //

  connect(ui->interpolationBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onInterpolationChanged(int)));
  connect(ui->rendererBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onRendererChanged(int)));

}


void vvPrefDialog::handleNewRenderer(vvRenderer*)
{

  updateUi();

}


void vvPrefDialog::onRendererChanged(int index)
{
  vvDebugMsg::msg(3, "vvPrefDialog::onRendererChanged()");

  assert(index == ui->rendererBox->currentIndex());

  QSettings settings;
  settings.setValue("renderer/type", index);

  emitRenderer();
}

void vvPrefDialog::onTexRendOptionChanged(int index)
{
  vvDebugMsg::msg(3, "vvPrefDialog::onTexRendOptionChanged()");

  VV_UNUSED(index);

  if (impl->rendererMap[ui->rendererBox->currentIndex()] == vvRenderer::TEXREND)
  {
    emitRenderer();
  }
}

void vvPrefDialog::onRayRendArchChanged(int index)
{
  QSettings settings;
  settings.setValue("rayrend/arch", index);

  if (impl->rendererMap[ui->rendererBox->currentIndex()] == vvRenderer::RAYREND)
  {
    emitRenderer();
  }
}

void vvPrefDialog::onFboChanged(int index)
{
  ui->texInfoLabel->setText("");
  if (impl->fboPrecisionMap[index] >= 8)
  {
    ui->texInfoLabel->setText("<html><b>" + QString::number(impl->fboPrecisionMap[index]) + " bit fbo rendering</b><br />"
      "An fbo is bound during slice compositing. A higher precision can help to avoid rounding errors but will result "
      "in an increased rendering time.</html>");
  }

  switch (impl->fboPrecisionMap[index])
  {
  case 8:
    emit parameterChanged(vvRenderer::VV_OFFSCREENBUFFER, true);
    emit parameterChanged(vvRenderer::VV_IMG_PRECISION, 8);
    break;
  case 16:
    emit parameterChanged(vvRenderer::VV_OFFSCREENBUFFER, true);
    emit parameterChanged(vvRenderer::VV_IMG_PRECISION, 16);
    break;
  case 32:
    emit parameterChanged(vvRenderer::VV_OFFSCREENBUFFER, true);
    emit parameterChanged(vvRenderer::VV_IMG_PRECISION, 32);
    break;
  default:
    emit parameterChanged(vvRenderer::VV_OFFSCREENBUFFER, false);
    break;
  }
}

void vvPrefDialog::onEarlyRayTerminationToggled(bool checked)
{
  emit parameterChanged(vvRenderState::VV_TERMINATEEARLY, checked);
}

void vvPrefDialog::onHostChanged(const QString& text)
{
  const ushort port = static_cast<ushort>(ui->portBox->value());
  if (validateRemoteHost(text, port))
  {
    ui->getInfoButton->setEnabled(true);
    ui->connectButton->setEnabled(true);
  }
  else
  {
    ui->getInfoButton->setEnabled(false);
    ui->connectButton->setEnabled(false);
  }
}

void vvPrefDialog::onPortChanged(const int i)
{
  const ushort port = static_cast<ushort>(i);
  if (validateRemoteHost(ui->hostEdit->text(), port))
  {
    ui->getInfoButton->setEnabled(true);
    ui->connectButton->setEnabled(true);
  }
  else
  {
    ui->getInfoButton->setEnabled(false);
    ui->connectButton->setEnabled(false);
  }
}

void vvPrefDialog::onGetInfoClicked()
{
#ifdef DESKVOX_USE_ASIO
  QMessageBox::warning(this, tr("Not implemented"), tr("Not implemented"), QMessageBox::Ok);
#else
  if (validateRemoteHost(ui->hostEdit->text(), static_cast<ushort>(ui->portBox->value())))
  {
    vvTcpSocket* sock = new vvTcpSocket;
    if (sock->connectToHost(ui->hostEdit->text().toStdString(),
      static_cast<ushort>(static_cast<ushort>(ui->portBox->value()))) == vvSocket::VV_OK)
    {
      sock->setParameter(vvSocket::VV_NO_NAGLE, true);
      vvSocketIO io(sock);

      vvServerInfo info;
      io.putEvent(virvo::ServerInfo);
      io.getServerInfo(info);
      QString qrenderers;
      std::vector<std::string> renderers = vvToolshed::split(info.renderers, ",");
      for (std::vector<std::string>::const_iterator it = renderers.begin();
           it != renderers.end(); ++it)
      {
        std::string rend = impl->rendererDescriptions[*it];
        std::string algo = impl->algoDescriptions[*it];
        qrenderers += "<tr><td>" + tr(rend.c_str()) + "</td><td>" + tr(algo.c_str()) + "</td></tr>";
      }
      QMessageBox::information(this, tr("Server info"), tr("Remote server supports the following rendering algorithms<br /><br />")
        + tr("<table>") + qrenderers + tr("</table>"), QMessageBox::Ok);
      io.putEvent(virvo::Disconnect);

      // store to registry because connection was successful
      QSettings settings;
      settings.setValue("remote/host", ui->hostEdit->text());
      settings.setValue("remote/port", ui->portBox->value());
    }
    else
    {
      QMessageBox::warning(this, tr("Failed to connect"), tr("Could not connect to host \"") + ui->hostEdit->text()
        + tr("\" on port \"") + QString::number(ui->portBox->value()) + tr("\""), QMessageBox::Ok);
    }
    delete sock;
  }
#endif
}

void vvPrefDialog::onConnectClicked()
{
#ifdef DESKVOX_USE_ASIO
  if (impl->remoterend == "")
  {
    ui->connectButton->setText(tr("Disconnect"));

    bool checked = ui->ibrBox->isChecked();
    if (checked)
      impl->remoterend = "ibr";
    else
      impl->remoterend = "image";

    QSettings settings;

    settings.setValue("remote/host", ui->hostEdit->text());
    settings.setValue("remote/port", ui->portBox->value());
    settings.setValue("remote/ibr", checked);

    emitRenderer();
  }
  else
  {
    ui->connectButton->setText(tr("Connect"));

    impl->remoterend = "";

    emitRenderer();
  }
#else
  if (impl->remoterend == "")
  {
    if (validateRemoteHost(ui->hostEdit->text(), static_cast<ushort>(ui->portBox->value())))
    {
      delete impl->sock;
      impl->sock = new vvTcpSocket;
      if (impl->sock->connectToHost(ui->hostEdit->text().toStdString(),
        static_cast<ushort>(static_cast<ushort>(ui->portBox->value()))) == vvSocket::VV_OK)
      {
        impl->sock->setParameter(vvSocket::VV_NO_NAGLE, true);
        vvSocketIO io(impl->sock);

        ui->connectButton->setText(tr("Disconnect"));

        if (!ui->ibrBox->isChecked())
        {
          impl->remoterend = "image";
        }
        else
        {
          impl->remoterend = "ibr";
          if (io.putEvent(virvo::RemoteServerType) == vvSocket::VV_OK)
          {
            io.putRendererType(vvRenderer::REMOTE_IBR);
          }
        }

        // store to registry because connection was successful
        QSettings settings;
        settings.setValue("remote/host", ui->hostEdit->text());
        settings.setValue("remote/port", ui->portBox->value());

        emitRenderer();
      }
      else
      {
        impl->remoterend = "";
        QMessageBox::warning(this, tr("Failed to connect"), tr("Could not connect to host \"") + ui->hostEdit->text()
          + tr("\" on port \"") + QString::number(ui->portBox->value()) + tr("\""), QMessageBox::Ok);
        delete impl->sock;
        impl->sock = NULL;
      }
    }
  }
  else
  {
    impl->remoterend = "";

    if (impl->sock != NULL)
    {
      vvSocketMap::remove(vvSocketMap::getIndex(impl->sock));
      delete impl->sock;
      impl->sock = NULL;
    }
    ui->connectButton->setText(tr("Connect"));

    emitRenderer();
  }
#endif
}

void vvPrefDialog::onIbrToggled(const bool checked)
{
#ifdef DESKVOX_USE_ASIO
  // not implemented
  static_cast<void>(checked);
#else
  QSettings settings;
  settings.setValue("remote/ibr", checked);

  if (impl->sock != NULL && impl->remoterend != "")
  {
    vvSocketIO io(impl->sock);
    if (io.putEvent(virvo::RemoteServerType) == vvSocket::VV_OK)
    {
      if (checked)
      {
       impl->remoterend = "ibr";
        io.putRendererType(vvRenderer::REMOTE_IBR);
      }
      else
      {
        impl->remoterend = "image";
        io.putRendererType(vvRenderer::REMOTE_IMAGE);
      }

      emitRenderer();
    }
  }
#endif
}

void vvPrefDialog::onInterpolationChanged(int index)
{
  QSettings settings;
  settings.setValue("appearance/interpolation", index);
  emit parameterChanged(vvRenderer::VV_SLICEINT, static_cast< virvo::tex_filter_mode >(index));
}

void vvPrefDialog::onMipToggled(bool checked)
{
  vvDebugMsg::msg(3, "vvPrefDialog::onMipToggled()");

  const int mipMode = checked ? 1 : 0; // don't support mip == 2 (min. intensity) for now
  emit parameterChanged(vvRenderer::VV_MIP_MODE, mipMode);
}

void vvPrefDialog::onPreIntegrationToggled(bool checked)
{
  QSettings settings;
  settings.setValue("appearance/preintegration", checked);
  emit parameterChanged(vvRenderer::VV_PREINT, checked);
}

void vvPrefDialog::onStereoModeChanged(int index)
{
  emit parameterChanged(vvParameters::VV_STEREO_MODE, static_cast<int>(impl->stereoModeMap[index]));
}

void vvPrefDialog::onStereoDistEdited(const QString& text)
{
  ui->stereoDistSlider->setValue(text.toInt());
}

void vvPrefDialog::onStereoDistSliderMoved(int value)
{
  ui->stereoDistEdit->setText(QString::number(value));
}

void vvPrefDialog::onStereoDistChanged(int value)
{
  QSettings settings;
  settings.setValue("stereo/distance", value);
  emit parameterChanged(vvParameters::VV_EYE_DIST, static_cast<float>(value));
}

void vvPrefDialog::onSwapEyesToggled(bool checked)
{
  QSettings settings;
  settings.setValue("stereo/swap", checked);
  emit parameterChanged(vvParameters::VV_SWAP_EYES, checked);
}

void vvPrefDialog::onMovingSpinBoxChanged(double value)
{
  vvDebugMsg::msg(3, "vvPrefDialog::onMovingSpinBoxChanged()");

  disconnect(ui->movingDial, SIGNAL(valueChanged(int)), this, SLOT(onMovingDialChanged(int)));
  const int upper = ui->movingDial->maximum() + 1;
  double d = value - impl->movingSpinBoxOldValue;
  int di = vvToolshed::round(d * upper);
  int dialval = ui->movingDial->value();
  dialval += di;
  dialval %= upper;
  while (dialval < ui->movingDial->minimum())
  {
    dialval += upper;
  }
  impl->movingDialOldValue = dialval;
  ui->movingDial->setValue(dialval);
  impl->movingSpinBoxOldValue = value;
  emit parameterChanged(vvParameters::VV_MOVING_QUALITY, static_cast<float>(ui->movingSpinBox->value()));
  connect(ui->movingDial, SIGNAL(valueChanged(int)), this, SLOT(onMovingDialChanged(int)));
}

void vvPrefDialog::onStillSpinBoxChanged(double value)
{
  vvDebugMsg::msg(3, "vvPrefDialog::onStillSpinBoxChanged()");

  disconnect(ui->stillDial, SIGNAL(valueChanged(int)), this, SLOT(onStillDialChanged(int)));
  const int upper = ui->stillDial->maximum() + 1;
  double d = value - impl->stillSpinBoxOldValue;
  int di = vvToolshed::round(d * upper);
  int dialval = ui->stillDial->value();
  dialval += di;
  dialval %= upper;
  while (dialval < ui->stillDial->minimum())
  {
    dialval += upper;
  }
  impl->stillDialOldValue = dialval;
  ui->stillDial->setValue(dialval);
  impl->stillSpinBoxOldValue = value;
  emit parameterChanged(vvRenderer::VV_QUALITY, static_cast<float>(ui->stillSpinBox->value()));
  connect(ui->stillDial, SIGNAL(valueChanged(int)), this, SLOT(onStillDialChanged(int)));
}

void vvPrefDialog::onMovingDialChanged(int value)
{
  vvDebugMsg::msg(3, "vvPrefDialog::onMovingDialChanged()");

  const int d = getDialDelta(impl->movingDialOldValue, value, ui->movingDial->minimum(), ui->movingDial->maximum());
  const double dd = static_cast<double>(d) / static_cast<double>(ui->movingDial->maximum() + 1);
  disconnect(ui->movingSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onMovingSpinBoxChanged(double)));
  ui->movingSpinBox->setValue(ui->movingSpinBox->value() + dd);
  impl->movingSpinBoxOldValue = ui->movingSpinBox->value();
  impl->movingDialOldValue = value;
  emit parameterChanged(vvParameters::VV_MOVING_QUALITY, static_cast<float>(ui->movingSpinBox->value()));
  connect(ui->movingSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onMovingSpinBoxChanged(double)));
}

void vvPrefDialog::onStillDialChanged(int value)
{
  vvDebugMsg::msg(3, "vvPrefDialog::onStillDialChanged()");

  const int d = getDialDelta(impl->stillDialOldValue, value, ui->stillDial->minimum(), ui->stillDial->maximum());
  const double dd = static_cast<double>(d) / static_cast<double>(ui->stillDial->maximum() + 1);
  disconnect(ui->stillSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onStillSpinBoxChanged(double)));
  ui->stillSpinBox->setValue(ui->stillSpinBox->value() + dd);
  impl->stillSpinBoxOldValue = ui->stillSpinBox->value();
  impl->stillDialOldValue = value;
  emit parameterChanged(vvRenderer::VV_QUALITY, static_cast<float>(ui->stillSpinBox->value()));
  connect(ui->stillSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onStillSpinBoxChanged(double)));
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
