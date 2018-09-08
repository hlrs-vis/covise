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

#include "vvvolinfodialog.h"

#include "ui_vvvolinfodialog.h"

#include <virvo/vvmacros.h>
#include <virvo/vvvoldesc.h>

#include <iostream>

struct vvVolInfoDialog::Impl
{
  Impl() : ui(new Ui::VolInfoDialog) {}

  std::auto_ptr<Ui::VolInfoDialog> ui;

private:

  VV_NOT_COPYABLE(Impl)

};

vvVolInfoDialog::vvVolInfoDialog(QWidget* parent)
  : QDialog(parent)
  , impl_(new Impl)
{
  impl_->ui->setupUi(this);

  connect(impl_->ui->iconButton, SIGNAL(clicked()), this, SLOT(onUpdateIconClicked()));
}

vvVolInfoDialog::~vvVolInfoDialog()
{
}

void vvVolInfoDialog::onNewVolDesc(vvVolDesc* vd)
{
  if (vd)
  {
    impl_->ui->filenameEdit->setText(vd->getFilename());
    impl_->ui->slicesWidthLabel->setText(QString::number(vd->vox[0]));
    impl_->ui->slicesHeightLabel->setText(QString::number(vd->vox[1]));
    impl_->ui->slicesDepthLabel->setText(QString::number(vd->vox[2]));
    impl_->ui->timeStepsLabel->setText(QString::number(vd->frames));
    impl_->ui->bpsLabel->setText(QString::number(vd->bpc));
    impl_->ui->channelsLabel->setText(QString::number(vd->getChan()));
    impl_->ui->voxelsLabel->setText(QString::number(vd->getFrameVoxels()));
    impl_->ui->bytesLabel->setText(QString::number(vd->getFrameBytes()));
    impl_->ui->distXLabel->setText(QString::number(vd->getDist()[0]));
    impl_->ui->distYLabel->setText(QString::number(vd->getDist()[1]));
    impl_->ui->distZLabel->setText(QString::number(vd->getDist()[2]));
    impl_->ui->pminLabel->setText(QString::number(vd->range(0)[0]));
    impl_->ui->pmaxLabel->setText(QString::number(vd->range(0)[1]));
    float fmin;
    float fmax;
    vd->findMinMax(0, fmin, fmax);
    impl_->ui->minLabel->setText(QString::number(fmin));
    impl_->ui->maxLabel->setText(QString::number(fmax));
  }
  else
  {
    impl_->ui->filenameEdit->setText("");
    impl_->ui->slicesWidthLabel->setText(QString::number(0));
    impl_->ui->slicesHeightLabel->setText(QString::number(0));
    impl_->ui->slicesDepthLabel->setText(QString::number(0));
    impl_->ui->timeStepsLabel->setText(QString::number(0));
    impl_->ui->bpsLabel->setText(QString::number(0));
    impl_->ui->channelsLabel->setText(QString::number(0));
    impl_->ui->voxelsLabel->setText(QString::number(0));
    impl_->ui->bytesLabel->setText(QString::number(0));
    impl_->ui->distXLabel->setText(QString::number(0));
    impl_->ui->distYLabel->setText(QString::number(0));
    impl_->ui->distZLabel->setText(QString::number(0));
    impl_->ui->pminLabel->setText(QString::number(0));
    impl_->ui->pmaxLabel->setText(QString::number(0));
    impl_->ui->minLabel->setText(QString::number(0));
    impl_->ui->maxLabel->setText(QString::number(0));
  }
}

void vvVolInfoDialog::onUpdateIconClicked()
{
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
