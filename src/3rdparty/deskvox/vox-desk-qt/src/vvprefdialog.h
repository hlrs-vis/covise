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

#ifndef VV_PREFDIALOG_H
#define VV_PREFDIALOG_H

#include "vvparameters.h"

#include <virvo/vvrenderer.h>
#include <virvo/vvrendererfactory.h>

#include <QDialog>

class Ui_PrefDialog;
class vvCanvas;

class vvPrefDialog : public QDialog
{
  Q_OBJECT
public:
  vvPrefDialog(vvCanvas* canvas, QWidget* parent = 0);
  ~vvPrefDialog();

  void applySettings();
  void toggleInterpolation();
  void scaleStillQuality(float s);
private:
  Ui_PrefDialog* ui;
  vvCanvas* _canvas;
  
  struct Impl;
  Impl* impl;
  
  
  void emitRenderer();
  bool validateRemoteHost(const QString& host, ushort port);
  void updateUi();

public slots:

  void handleNewRenderer(vvRenderer* renderer);

private slots:
  void onRendererChanged(int index);
  void onTexRendOptionChanged(int index);
  void onRayRendArchChanged(int index);
  void onFboChanged(int index);
  void onEarlyRayTerminationToggled(bool checked);
  void onHostChanged(const QString& text);
  void onPortChanged(int i);
  void onGetInfoClicked();
  void onConnectClicked();
  void onIbrToggled(bool checked);
  void onInterpolationChanged(int index);
  void onMipToggled(bool checked);
  void onPreIntegrationToggled(bool checked);
  void onStereoModeChanged(int index);
  void onStereoDistEdited(const QString& text);
  void onStereoDistSliderMoved(int value);
  void onStereoDistChanged(int value);
  void onSwapEyesToggled(bool checked);
  void onMovingSpinBoxChanged(double value);
  void onStillSpinBoxChanged(double value);
  void onMovingDialChanged(int value);
  void onStillDialChanged(int value);
signals:
  void rendererChanged(const std::string& name, const vvRendererFactory::Options& options);
  void parameterChanged(vvParameters::ParameterType param, const vvParam& value);
  void parameterChanged(vvRenderer::ParameterType param, const vvParam& value);
};

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
