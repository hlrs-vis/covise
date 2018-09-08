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

#ifndef VV_TFDIALOG_H
#define VV_TFDIALOG_H

#include <virvo/math/forward.h>
#include <virvo/vvtfwidget.h>

#include <QDialog>

#include <memory>

class vvCanvas;
class vvTFWidget;
class vvVolDesc;

class vvTFDialog : public QDialog
{

  Q_OBJECT
  Q_DISABLE_COPY(vvTFDialog)

public:
  vvTFDialog(vvCanvas* canvas, QWidget* parent = 0);
  ~vvTFDialog();
private:
  struct Impl;
  std::auto_ptr<Impl> impl_;

  vvCanvas* _canvas;

  void drawTF();
  void drawColorTexture();
  void drawAlphaTexture();
  void clearPins();
  void createPins();
  void createPin(vvTFWidget* w);
  void makeColorBar(std::vector<uchar>* colorBar, int width) const;
  void makeHistogramTexture(std::vector<uchar>* hist, int width, int height) const;
  void makeAlphaTexture(std::vector<uchar>* alphaTex, int width, int height) const;
  void emitTransFunc();
  void updateSettingsBox();

private slots:
  void onNewWidget();
  void onDeleteClicked();
  void onUndoClicked();
  void onLogHistToggled(bool value);
  void onPresetColorsChanged(int index);
  void onPresetAlphaChanged(int index);
  void onDiscrChanged(int num);
  void onApplyClicked();
  void onNewVolDesc(vvVolDesc* vd);
  void onZoomMinChanged(double zm);
  void onZoomMaxChanged(double zm);
  void saveTF();
  void loadTF();

  void onColor(const QColor& color);
  void onHasOwnColor(bool hascolor);
  void onSize(virvo::vec3f const& size);
  void onOpacity(float opacity);
  void onTop(virvo::vec3f const& top);
  void onBottom(virvo::vec3f const& bottom);

  void onTFMouseMove(QPointF pos, Qt::MouseButton button);
  void onTFMousePress(QPointF pos, Qt::MouseButton button);
  void onTFMouseRelease(QPointF pos, Qt::MouseButton button);
signals:
  void newWidget(vvTFWidget* widget);
  void newTransferFunction();
  void undo();
};

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
