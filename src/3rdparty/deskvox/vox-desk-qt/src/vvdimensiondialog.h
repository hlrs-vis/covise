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

#ifndef VV_DIMENSIONDIALOG_H
#define VV_DIMENSIONDIALOG_H

class vvCanvas;

#include <virvo/math/forward.h>

#include <QDialog>

#include <memory>

class vvDimensionDialog : public QDialog
{

  Q_OBJECT
  Q_DISABLE_COPY(vvDimensionDialog)

public:
  vvDimensionDialog(vvCanvas* canvas, QWidget* parent = 0);
  ~vvDimensionDialog();

  /*! initial dist should be set whenever a new volume is loaded
   */
  void setInitialDist(virvo::vec3f const& dist);
private:
  struct Impl;
  std::auto_ptr<Impl> impl_;

  vvCanvas* _canvas;

  /*! update volume description with new dist
   */
  void setDist(virvo::vec3f const& dist);
  /*! refresh spin box values
   */
  void updateGui(virvo::vec3f const& dist);

private slots:
  void onApplyClicked();
  void onResetClicked();
};

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
