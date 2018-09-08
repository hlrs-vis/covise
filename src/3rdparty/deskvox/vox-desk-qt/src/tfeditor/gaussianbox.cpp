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

#include "gaussianbox.h"

#include "ui_gaussianbox.h"

#include <virvo/math/math.h>
#include <virvo/vvmacros.h>

#include <QColorDialog>

using namespace virvo;


struct tf::GaussianBox::Impl
{
  Impl()
    : ui(new Ui::GaussianBox)
    , zoomRange(0.0f, 1.0f)
  {
  }

  std::auto_ptr<Ui::GaussianBox> ui;
  vec2 zoomRange;
  bool hascolor;
  vvColor color;
  vec3f size;
  float opacity;

private:

  VV_NOT_COPYABLE(Impl)

};

tf::GaussianBox::GaussianBox(QWidget* parent)
  : QGroupBox(parent)
  , impl(new Impl)
{
  impl->ui->setupUi(this);

  connect(impl->ui->ownColorBox, SIGNAL(toggled(bool)), this, SIGNAL(hasColor(bool)));
  connect(impl->ui->widthSlider, SIGNAL(valueChanged(int)), this, SLOT(emitSize(int)));
  connect(impl->ui->maxOpacitySlider, SIGNAL(valueChanged(int)), this, SLOT(emitOpacity(int)));
  connect(impl->ui->pickColorButton, SIGNAL(clicked()), this, SLOT(getColor()));
}

tf::GaussianBox::~GaussianBox()
{
}

void tf::GaussianBox::setZoomRange(vec2 zr)
{
  impl->zoomRange = zr;
}

void tf::GaussianBox::setHasColor(bool hascolor)
{
  impl->hascolor = hascolor;
  impl->ui->ownColorBox->setChecked(hascolor);
}

void tf::GaussianBox::setColor(const vvColor& color)
{
  impl->color = color;
}

void tf::GaussianBox::setSize(vec3f const& size)
{
  impl->size = size * 0.5f;
  impl->ui->widthLabel->setText(tr("Width: ") + QString::number(size[0]));
  impl->ui->widthSlider->setValue(impl->ui->widthSlider->maximum() * size[0]);
}

void tf::GaussianBox::setOpacity(float opacity)
{
  impl->opacity = opacity;
  impl->ui->maxOpacitySlider->setValue(opacity * impl->ui->maxOpacitySlider->maximum() * 0.2f);
  impl->ui->maxOpacityLabel->setText(tr("Maximum value: ") + QString::number(opacity));
}

void tf::GaussianBox::getColor()
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

void tf::GaussianBox::emitSize(int sliderval)
{
  float w = static_cast<float>(sliderval) / static_cast<float>(impl->ui->widthSlider->maximum());
  w *= impl->zoomRange[1] - impl->zoomRange[0];
  impl->ui->widthLabel->setText(tr("Width: ") + QString::number(w));
  impl->size[0] = w;
  emit size(impl->size);
}

void tf::GaussianBox::emitOpacity(int sliderval)
{
  float o = static_cast<float>(sliderval) / static_cast<float>(impl->ui->maxOpacitySlider->maximum()) * 5.0f;
  impl->ui->maxOpacityLabel->setText(tr("Maximum value: ") + QString::number(o));
  impl->opacity = o;
  emit opacity(impl->opacity);
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
