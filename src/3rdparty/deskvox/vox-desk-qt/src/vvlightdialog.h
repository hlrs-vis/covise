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


#ifndef VV_LIGHTDIALOG_H
#define VV_LIGHTDIALOG_H

#include <virvo/math/forward.h>

#include <QDialog>

#include <memory>

class vvLightDialog : public QDialog
{

  Q_OBJECT
  Q_DISABLE_COPY(vvLightDialog)

public:
  vvLightDialog(QWidget* parent = 0);
  ~vvLightDialog();

  void applySettings();
private:

  struct Impl;
  std::auto_ptr<Impl> impl_;

private slots:
  void onEnableToggled(bool checked);
  void onEnableHeadlightToggled(bool checked);
  void onEditPositionClicked();
  void onConstAttChanged(double value);
  void onLinearAttChanged(double value);
  void onQuadAttChanged(double value);
signals:
  void enabled(bool enable);
  void showLightSource(bool show);
  void enableHeadlight(bool enable);
  void editPositionToggled(bool edit);
  void attenuationChanged(virvo::vec3f const& att);
};

#endif

