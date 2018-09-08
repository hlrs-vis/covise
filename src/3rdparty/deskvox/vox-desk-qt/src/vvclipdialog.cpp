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


#include "vvclipdialog.h"

#include "ui_vvclipdialog.h"

#include <virvo/math/math.h>
#include <virvo/vvmacros.h>

#include <iostream>


struct vvClipDialog::Impl
{
  Impl() : ui(new Ui::ClipDialog) {}

  std::auto_ptr< Ui::ClipDialog > ui;

private:

  VV_NOT_COPYABLE(Impl)

};


vvClipDialog::vvClipDialog(QWidget* parent)
  : QDialog(parent)
  , impl_(new Impl)
{
  impl_->ui->setupUi(this);

  connect(impl_->ui->enableBox,     SIGNAL(toggled(bool)),     this, SLOT(setEnabled(bool)));

  connect(impl_->ui->normalxSlider, SIGNAL(valueChanged(int)), this, SLOT(emitNormal()));
  connect(impl_->ui->normalySlider, SIGNAL(valueChanged(int)), this, SLOT(emitNormal()));
  connect(impl_->ui->normalzSlider, SIGNAL(valueChanged(int)), this, SLOT(emitNormal()));

  connect(impl_->ui->originSlider,  SIGNAL(valueChanged(int)), this, SLOT(emitOrigin()));

  connect(impl_->ui->singleBox,     SIGNAL(toggled(bool)),     this, SIGNAL(singleSlice(bool)));
  connect(impl_->ui->opaqueBox,     SIGNAL(toggled(bool)),     this, SIGNAL(opaque(bool)));
  connect(impl_->ui->perimeterBox,  SIGNAL(toggled(bool)),     this, SIGNAL(perimeter(bool)));
}


vvClipDialog::~vvClipDialog()
{
}


void vvClipDialog::setEnabled(bool enabled)
{
  impl_->ui->frame->setEnabled(enabled);

  emit clipping(enabled);
}


void vvClipDialog::emitNormal() const
{
  virvo::vec3f n(impl_->ui->normalxSlider->value(),
    impl_->ui->normalySlider->value(),
    impl_->ui->normalzSlider->value());
  n = normalize(n);

  emit normal(n);
}


void vvClipDialog::emitOrigin() const
{
  virvo::vec3f o(impl_->ui->normalxSlider->value(),
    impl_->ui->normalySlider->value(),
    impl_->ui->normalzSlider->value());
  o = normalize(o);
  float d = static_cast< float >(impl_->ui->originSlider->value());
  o *= virvo::vec3f(d);

  emit origin(o);
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
