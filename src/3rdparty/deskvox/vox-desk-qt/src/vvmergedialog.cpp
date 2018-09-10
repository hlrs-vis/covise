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

#include "vvmergedialog.h"

#include "ui_vvmergedialog.h"

#include <virvo/vvdebugmsg.h>
#include <virvo/vvfileio.h>
#include <virvo/vvmacros.h>

#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>

struct vvMergeDialog::Impl
{
  Impl() : ui(new Ui::MergeDialog) {}

  std::auto_ptr<Ui::MergeDialog> ui;

private:

  VV_NOT_COPYABLE(Impl)

};

vvMergeDialog::vvMergeDialog(QWidget* parent)
  : QDialog(parent)
  , impl_(new Impl)
{
  vvDebugMsg::msg(1, "vvMergeDialog::vvMergeDialog()");

  impl_->ui->setupUi(this);

  connect(impl_->ui->browseButton, SIGNAL(clicked()), this, SLOT(onBrowseClicked()));
  connect(impl_->ui->numFilesCheckBox, SIGNAL(toggled(bool)), this, SLOT(onNumFilesToggled(bool)));
  connect(impl_->ui->helpButton, SIGNAL(clicked()), this, SLOT(onHelpClicked()));
}

vvMergeDialog::~vvMergeDialog()
{
  vvDebugMsg::msg(1, "vvMergeDialog::~vvMergeDialog()");
}

QString vvMergeDialog::getFilename() const
{
  vvDebugMsg::msg(3, "vvMergeDialog::getFilename()");

  return impl_->ui->filenameEdit->text();
}

int vvMergeDialog::getNumFiles() const
{
  vvDebugMsg::msg(3, "vvMergeDialog::getNumFiles()");

  const QString numFiles = impl_->ui->numFilesEdit->text();
  return numFiles.toInt();
}

int vvMergeDialog::getFileIncrement() const
{
  vvDebugMsg::msg(3, "vvMergeDialog::getFileIncrement()");

  const QString fileInc = impl_->ui->fileIncEdit->text();
  return fileInc.toInt();
}

vvVolDesc::MergeType vvMergeDialog::getMergeType() const
{
  vvDebugMsg::msg(3, "vvMergeDialog::getMergeType()");

  if (impl_->ui->slicesRadioButton->isChecked())
  {
    return vvVolDesc::VV_MERGE_SLABS2VOL;
  }
  else if (impl_->ui->animationRadioButton->isChecked())
  {
    return vvVolDesc::VV_MERGE_VOL2ANIM;
  }
  else
  {
    return vvVolDesc::VV_MERGE_CHAN2VOL;
  }
}

bool vvMergeDialog::numFilesLimited() const
{
  vvDebugMsg::msg(3, "vvMergeDialog::numFilesLimited()");

  return impl_->ui->numFilesCheckBox->isChecked();
}

bool vvMergeDialog::filesNumbered() const
{
  vvDebugMsg::msg(3, "vvMergeDialog::filesNumbered()");

  return impl_->ui->incFilesCheckBox->isChecked();
}

void vvMergeDialog::onBrowseClicked()
{
  vvDebugMsg::msg(3, "vvMergeDialog::onBrowseClicked()");

  QString caption = tr("Merge Files");
  QString dir;
  QFileInfo info(impl_->ui->filenameEdit->text());
  if (info.isDir())
  {
    dir = impl_->ui->filenameEdit->text();
  }
  else if (info.isFile())
  {
    dir = info.absolutePath();
  }
  QString filter = tr("All VOX Files (*.xvf *.avf *.rvf *.tif *.rgb *.ppm *.pgm *.dcm *.ximg *.hdr *.volb);;"
    "Volume Files (*.rvf *.xvf *.avf *.tif *.tiff *.hdr *.volb);;"
    "Image Files (*.tif *.rgb *.ppm *.pgm *.dcm *.ximg);;"
    "All Files (*)");
  QString filename = QFileDialog::getOpenFileName(this, caption, dir, filter);
  if (!filename.isEmpty())
  {
    impl_->ui->filenameEdit->setText(filename);
  }
}

void vvMergeDialog::onNumFilesToggled(const bool checked)
{
  vvDebugMsg::msg(3, "vvMergeDialog::onNumFilesToggled");

  impl_->ui->numFilesEdit->setEnabled(checked);
}

void vvMergeDialog::onHelpClicked()
{
  vvDebugMsg::msg(3, "vvMergeDialog::onHelpClicked()");

  const QString info = tr("If numbered, files must be numbered consecutively, for instance:\n"
    "image000.tif, image001.tif, image002.tif, ...\n"
    "Altenatively, they can be numbered with a constant increment greater than 1.\n\n"
    "The naming scheme '*_zXXX_chXX.*' is recognized for image\n"
    "files at a given depth (_z) and for a given channel (_ch).\n"
    "Example: 'image_z010_ch02.tif'");
  QMessageBox::information(this, tr("File Format"), info, QMessageBox::Ok);
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
