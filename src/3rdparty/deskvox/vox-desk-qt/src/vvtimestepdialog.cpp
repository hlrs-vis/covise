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

#include "vvtimestepdialog.h"

#include "ui_vvtimestepdialog.h"

#include <virvo/vvdebugmsg.h>
#include <virvo/vvmacros.h>

#include <QSettings>

struct vvTimeStepDialog::Impl
{
  Impl()
    : ui(new Ui::TimeStepDialog)
    , playing(false)
  {
  }

  std::auto_ptr<Ui::TimeStepDialog> ui;
  bool playing;

private:

  VV_NOT_COPYABLE(Impl)

};

vvTimeStepDialog::vvTimeStepDialog(QWidget* parent)
  : QDialog(parent)
  , impl_(new Impl)
{
  vvDebugMsg::msg(1, "vvTimeStepDialog::vvTimeStepDialog()");

  impl_->ui->setupUi(this);

  impl_->ui->playButton->setFocus(Qt::OtherFocusReason);

  QSettings settings;
  impl_->ui->frameRateBox->setValue(settings.value("timestepdialog/fps").value<double>());

  connect(impl_->ui->frameRateBox, SIGNAL(valueChanged(double)), this, SLOT(onFrameRateChanged()));
  connect(impl_->ui->playButton, SIGNAL(clicked()), this, SLOT(onPlayClicked()));
  connect(impl_->ui->backButton, SIGNAL(clicked()), this, SIGNAL(back()));
  connect(impl_->ui->fwdButton, SIGNAL(clicked()), this, SIGNAL(fwd()));
  connect(impl_->ui->backBackButton, SIGNAL(clicked()), this, SIGNAL(first()));
  connect(impl_->ui->fwdFwdButton, SIGNAL(clicked()), this, SIGNAL(last()));
  connect(impl_->ui->timeStepSlider, SIGNAL(sliderMoved(int)), this, SIGNAL(valueChanged(int)));
}

vvTimeStepDialog::~vvTimeStepDialog()
{
  vvDebugMsg::msg(1, "vvTimeStepDialog::~vvTimeStepDialog()");
}

void vvTimeStepDialog::setFrames(const int frames)
{
  vvDebugMsg::msg(3, "vvTimeStepDialog::setFrames()");

  impl_->ui->timeStepLabel->setText(QString::number(impl_->ui->timeStepSlider->value() + 1) + "/" + QString::number(frames));
  impl_->ui->timeStepSlider->setMaximum(frames - 1);
  impl_->ui->timeStepSlider->setTickInterval(1);
}

void vvTimeStepDialog::setCurrentFrame(const int frame)
{
  vvDebugMsg::msg(3, "vvTimeStepDialog::setCurrentFrame()");

  impl_->ui->timeStepLabel->setText(QString::number(frame + 1) + "/" + QString::number(impl_->ui->timeStepSlider->maximum() + 1));
  impl_->ui->timeStepSlider->setValue(frame);
}

void vvTimeStepDialog::togglePlayback()
{
  vvDebugMsg::msg(3, "vvTimeStepDialog::togglePlayback()");

  onPlayClicked();
}

void vvTimeStepDialog::stepFwd()
{
  vvDebugMsg::msg(3, "vvTimeStepDialog::stepFwd()");

  emit fwd();
}

void vvTimeStepDialog::stepBack()
{
  vvDebugMsg::msg(3, "vvTimeStepDialog::stepBack()");

  emit back();
}

void vvTimeStepDialog::onPlayClicked()
{
  vvDebugMsg::msg(3, "vvTimeStepDialog::onPlayClicked()");

  if (!impl_->playing)
  {
    impl_->ui->playButton->setText("||");
    emit play(impl_->ui->frameRateBox->value());    
  }
  else
  {
    impl_->ui->playButton->setText(">");
    emit pause();
  }
  impl_->playing = !impl_->playing;
}

void vvTimeStepDialog::onFrameRateChanged()
{
  vvDebugMsg::msg(3, "vvTimeStepDialog::onFrameRateChanged()");

  QSettings settings;
  settings.setValue("timestepdialog/fps", impl_->ui->frameRateBox->value());

  if (impl_->playing)
  {
    // re-emit play signal
    emit play(impl_->ui->frameRateBox->value());
  }
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
