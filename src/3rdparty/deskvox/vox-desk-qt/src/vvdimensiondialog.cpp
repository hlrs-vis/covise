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

// Make sure that winsock2.h is included before windows:
#include <virvo/vvplatform.h>

#include "vvcanvas.h"
#include "vvdimensiondialog.h"

#include "ui_vvdimensiondialog.h"

#include <virvo/math/math.h>
#include <virvo/vvdebugmsg.h>
#include <virvo/vvmacros.h>
#include <virvo/vvtexrend.h>
#include <virvo/vvvoldesc.h>

using virvo::vec3f;


struct vvDimensionDialog::Impl
{
  Impl()
    : ui(new Ui::DimensionDialog)
    , initialDist(vec3f(1.0f, 1.0f, 1.0f))
  {
  }

  std::auto_ptr<Ui::DimensionDialog> ui;
  vec3f initialDist; ///< should be assigned when a new file is loaded

private:

  VV_NOT_COPYABLE(Impl)

};

vvDimensionDialog::vvDimensionDialog(vvCanvas* canvas, QWidget* parent)
  : QDialog(parent)
  , impl_(new Impl)
  , _canvas(canvas)
{
  vvDebugMsg::msg(1, "vvDimensionDialog::vvDimensionDialog()");

  impl_->ui->setupUi(this);

  connect(impl_->ui->resetButton, SIGNAL(clicked()), this, SLOT(onResetClicked()));
  connect(impl_->ui->applyButton, SIGNAL(clicked()), this, SLOT(onApplyClicked()));
}

vvDimensionDialog::~vvDimensionDialog()
{
  vvDebugMsg::msg(1, "vvDimensionDialog::~vvDimensionDialog()");
}

void vvDimensionDialog::setInitialDist(vec3f const& dist)
{
  vvDebugMsg::msg(3, "vvDimensionDialog::setInitialDist()");

  impl_->initialDist = dist;
  updateGui(dist);
}

void vvDimensionDialog::setDist(vec3f const& dist)
{
  vvDebugMsg::msg(3, "vvDimensionDialog::setDist()");

  _canvas->getVolDesc()->setDist(dist);
  _canvas->updateGL();
  updateGui(dist);
}

void vvDimensionDialog::updateGui(vec3f const& dist)
{
  vvDebugMsg::msg(3, "vvDimensionDialog::updateGui()");

  impl_->ui->distXBox->setValue(dist[0]);
  impl_->ui->distYBox->setValue(dist[1]);
  impl_->ui->distZBox->setValue(dist[2]);
}

void vvDimensionDialog::onApplyClicked()
{
  vvDebugMsg::msg(3, "vvDimensionDialog::onApplyClicked()");

  setDist(vec3f(static_cast<float>(impl_->ui->distXBox->value()),
                static_cast<float>(impl_->ui->distYBox->value()),
                      static_cast<float>(impl_->ui->distZBox->value())));
}

void vvDimensionDialog::onResetClicked()
{
  vvDebugMsg::msg(3, "vvDimensionDialog::onResetClicked()");

  setDist(impl_->initialDist);
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
