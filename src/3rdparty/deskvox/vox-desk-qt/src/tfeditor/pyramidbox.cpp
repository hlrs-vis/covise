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

#include "pyramidbox.h"

#include "ui_pyramidbox.h"

#include <virvo/math/math.h>
#include <virvo/vvmacros.h>

#include <QColorDialog>

using namespace virvo;


struct tf::PyramidBox::Impl
{
  Impl()
    : ui(new Ui::PyramidBox)
    , zoomRange(0.0f, 1.0f)
  {
  }

  std::auto_ptr<Ui::PyramidBox> ui;
  bool hascolor;
  vvColor color;
  vec3f top;
  vec3f bottom;
  float opacity;
  vec2 zoomRange;

private:

  VV_NOT_COPYABLE(Impl)

};

tf::PyramidBox::PyramidBox(QWidget* parent)
  : QGroupBox(parent)
  , impl(new Impl)
{
  impl->ui->setupUi(this);

  connect(impl->ui->ownColorBox, SIGNAL(toggled(bool)), this, SIGNAL(hasColor(bool)));
  connect(impl->ui->topSlider, SIGNAL(valueChanged(int)), this, SLOT(emitTop(int)));
  connect(impl->ui->bottomSlider, SIGNAL(valueChanged(int)), this, SLOT(emitBottom(int)));
  connect(impl->ui->maxOpacitySlider, SIGNAL(valueChanged(int)), this, SLOT(emitOpacity(int)));
  connect(impl->ui->pickColorButton, SIGNAL(clicked()), this, SLOT(getColor()));
}

tf::PyramidBox::~PyramidBox()
{
}

void tf::PyramidBox::setZoomRange(vec2 zr)
{
  impl->zoomRange = zr;
}

void tf::PyramidBox::setHasColor(bool hascolor)
{
  impl->hascolor = hascolor;
  impl->ui->ownColorBox->setChecked(hascolor);
}

void tf::PyramidBox::setColor(const vvColor& color)
{
  impl->color = color;
}

void tf::PyramidBox::setTop(vec3f const& top)
{
  impl->top = top;
  impl->ui->topSlider->setValue(top[0] * impl->ui->topSlider->maximum() * 0.5f);
  impl->ui->topLabel->setText(tr("Top width X: ") + QString::number(top[0]));
}

void tf::PyramidBox::setBottom(vec3f const& bottom)
{
  impl->bottom = bottom;
  impl->ui->bottomSlider->setValue(bottom[0] * impl->ui->bottomSlider->maximum() * 0.5f);
  impl->ui->bottomLabel->setText(tr("Bottom width X: ") + QString::number(bottom[0]));
}

void tf::PyramidBox::setOpacity(float opacity)
{
  impl->opacity = opacity;
  impl->ui->maxOpacitySlider->setValue(opacity * impl->ui->maxOpacitySlider->maximum());
  impl->ui->maxOpacityLabel->setText(tr("Maximum opacity: ") + QString::number(opacity));
}

void tf::PyramidBox::getColor()
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

void tf::PyramidBox::emitTop(int sliderval)
{
  float t = static_cast<float>(sliderval) / static_cast<float>(impl->ui->topSlider->maximum()) * 2.0f;
  t *= impl->zoomRange[1] - impl->zoomRange[0];
  impl->ui->topLabel->setText(tr("Top width X: ") + QString::number(t));
  impl->top[0] = t;
  emit top(impl->top);
}

void tf::PyramidBox::emitBottom(int sliderval)
{
  float b = static_cast<float>(sliderval) / static_cast<float>(impl->ui->bottomSlider->maximum()) * 2.0f;
  b *= impl->zoomRange[1] - impl->zoomRange[0];
  impl->ui->bottomLabel->setText(tr("Bottom width X: ") + QString::number(b));
  impl->bottom[0] = b;
  emit bottom(impl->bottom);
}

void tf::PyramidBox::emitOpacity(int sliderval)
{
  float o = static_cast<float>(sliderval) / static_cast<float>(impl->ui->maxOpacitySlider->maximum());
  impl->ui->maxOpacityLabel->setText(tr("Maximum opacity: ") + QString::number(o));
  impl->opacity = o;
  emit opacity(impl->opacity);
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
