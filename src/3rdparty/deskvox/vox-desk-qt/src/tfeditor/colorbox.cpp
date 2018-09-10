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

#include "colorbox.h"

#include "ui_colorbox.h"

#include <virvo/vvmacros.h>

#include <QColorDialog>

struct tf::ColorBox::Impl
{
  Impl() : ui(new Ui::ColorBox) {}

  std::auto_ptr<Ui::ColorBox> ui;
  vvColor color;

private:

  VV_NOT_COPYABLE(Impl)

};

tf::ColorBox::ColorBox(QWidget* parent)
  : QGroupBox(parent)
  , impl(new Impl)
{
  impl->ui->setupUi(this);

  connect(impl->ui->pickColorButton, SIGNAL(clicked()), this, SLOT(getColor()));
}

tf::ColorBox::~ColorBox()
{
}

void tf::ColorBox::setColor(const vvColor& color)
{
  impl->color = color;
}

void tf::ColorBox::getColor()
{
  QColor initial;
  initial.setRedF(impl->color[0]);
  initial.setGreenF(impl->color[1]);
  initial.setBlueF(impl->color[2]);
  QColor clr = QColorDialog::getColor(initial);
  if (clr.isValid())
  {
    impl->color = vvColor(clr.redF(), clr.greenF(), clr.blueF());
    emit color(clr);
  }
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
