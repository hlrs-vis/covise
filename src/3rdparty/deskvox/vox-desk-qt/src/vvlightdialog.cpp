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

#include "vvlightdialog.h"

#include "ui_vvlightdialog.h"

#include <virvo/math/math.h>
#include <virvo/vvmacros.h>

#include <QSettings>
#include <QVector3D>

#include <iostream>

using virvo::vec3f;


struct vvLightDialog::Impl
{
  Impl() : ui(new Ui::LightDialog) {}

  std::auto_ptr<Ui::LightDialog> ui;

private:

  VV_NOT_COPYABLE(Impl)

};

vvLightDialog::vvLightDialog(QWidget* parent)
  : QDialog(parent)
  , impl_(new Impl)
{
  impl_->ui->setupUi(this);

  connect(impl_->ui->enableBox, SIGNAL(toggled(bool)), this, SLOT(onEnableToggled(bool)));
  connect(impl_->ui->showBox, SIGNAL(toggled(bool)), this, SIGNAL(showLightSource(bool)));
  connect(impl_->ui->headlightBox, SIGNAL(toggled(bool)), this, SLOT(onEnableHeadlightToggled(bool)));
  connect(impl_->ui->positionButton, SIGNAL(clicked()), this, SLOT(onEditPositionClicked()));
  connect(impl_->ui->constSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onConstAttChanged(double)));
  connect(impl_->ui->linearSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onLinearAttChanged(double)));
  connect(impl_->ui->quadSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onQuadAttChanged(double)));
}

vvLightDialog::~vvLightDialog()
{

}

void vvLightDialog::applySettings()
{
  QSettings settings;
  if (settings.contains("canvas/lightattenuation"))
  {
    QVector3D qatt = settings.value("canvas/lightattenuation").value<QVector3D>();
    impl_->ui->constSpinBox->setValue(qatt.x());
    impl_->ui->linearSpinBox->setValue(qatt.y());
    impl_->ui->quadSpinBox->setValue(qatt.z());
  }

  impl_->ui->enableBox->setChecked(settings.value("canvas/lighting").toBool());
}

void vvLightDialog::onEnableToggled(bool checked)
{
  impl_->ui->showBox->setEnabled(checked);
  impl_->ui->headlightBox->setEnabled(checked);
  impl_->ui->positionButton->setEnabled(checked);
  impl_->ui->attenuationBox->setEnabled(checked);

  QSettings settings;
  settings.setValue("canvas/lighting", checked);

  emit enabled(checked);
}

void vvLightDialog::onEnableHeadlightToggled(bool checked)
{
  QSettings settings;
  settings.setValue("canvas/headlight", checked);
  emit enableHeadlight(checked);
}

void vvLightDialog::onEditPositionClicked()
{
  if (impl_->ui->positionButton->text() == "Edit Position")
  {
    emit editPositionToggled(true);
    impl_->ui->positionButton->setText("Stop Editing");
  }
  else
  {
    emit editPositionToggled(false);
    impl_->ui->positionButton->setText("Edit Position");
  }
}

void vvLightDialog::onConstAttChanged(double value)
{
  emit attenuationChanged(vec3f(value, impl_->ui->linearSpinBox->value(), impl_->ui->quadSpinBox->value()));
}

void vvLightDialog::onLinearAttChanged(double value)
{
  emit attenuationChanged(vec3f(impl_->ui->constSpinBox->value(), value, impl_->ui->quadSpinBox->value()));
}

void vvLightDialog::onQuadAttChanged(double value)
{
  emit attenuationChanged(vec3f(impl_->ui->constSpinBox->value(), impl_->ui->linearSpinBox->value(), value));
}

