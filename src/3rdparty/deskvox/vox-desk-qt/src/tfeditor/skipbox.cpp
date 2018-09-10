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

#include "skipbox.h"

#include <virvo/math/math.h>
#include <virvo/vvmacros.h>

#include "ui_skipbox.h"

using namespace virvo;


struct tf::SkipBox::Impl
{
  Impl()
    : ui(new Ui_SkipBox)
    , zoomRange(0.0f, 1.0f)
  {
  }

  std::auto_ptr<Ui_SkipBox> ui;
  vec2 zoomRange;
  vec3f size;

private:

  VV_NOT_COPYABLE(Impl)

};

tf::SkipBox::SkipBox(QWidget* parent)
  : QGroupBox(parent)
  , impl(new Impl)
{
  impl->ui->setupUi(this);

  connect(impl->ui->widthSlider, SIGNAL(valueChanged(int)), this, SLOT(emitSize(int)));
}

tf::SkipBox::~SkipBox()
{
}

void tf::SkipBox::setZoomRange(vec2 zr)
{
  impl->zoomRange = zr;
}

void tf::SkipBox::setSize(vec3f const& size)
{
  impl->size = size;
  impl->ui->widthLabel->setText(tr("Width: ") + QString::number(size[0]));
  impl->ui->widthSlider->setValue(impl->ui->widthSlider->maximum() * size[0]);
}

void tf::SkipBox::emitSize(int sliderval)
{
  float w = static_cast<float>(sliderval) / static_cast<float>(impl->ui->widthSlider->maximum());
  w *= impl->zoomRange[1] - impl->zoomRange[0];
  impl->ui->widthLabel->setText(tr("Width: ") + QString::number(w));
  impl->size[0] = w;
  emit size(impl->size);
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
