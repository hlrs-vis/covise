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

#include "vvcanvas.h"
#include "vvscreenshotdialog.h"

#include "ui_vvscreenshotdialog.h"

#include <virvo/vvdebugmsg.h>
#include <virvo/vvfileio.h>
#include <virvo/vvmacros.h>
#include <virvo/vvrenderer.h>
#include <virvo/vvvoldesc.h>

#include <QFileDialog>

struct vvScreenshotDialog::Impl
{
  Impl() : ui(new Ui::ScreenshotDialog) {}

  std::auto_ptr<Ui::ScreenshotDialog> ui;

private:

  VV_NOT_COPYABLE(Impl)

};

vvScreenshotDialog::vvScreenshotDialog(vvCanvas* canvas, QWidget* parent)
  : QDialog(parent)
  , impl_(new Impl)
  , _canvas(canvas)
{
  vvDebugMsg::msg(1, "vvScreenshotDialog::vvScreenshotDialog()");

  impl_->ui->setupUi(this);

  impl_->ui->dirEdit->setText(QDir::currentPath());

  connect(_canvas, SIGNAL(resized(QSize)), this, SLOT(setCanvasSize(QSize)));

  connect(impl_->ui->canvasSizeCheckBox, SIGNAL(toggled(bool)), this, SLOT(onCanvasSizeToggled(bool)));
  connect(impl_->ui->browseButton, SIGNAL(clicked()), this, SLOT(onBrowseClicked()));
  connect(impl_->ui->pictureButton, SIGNAL(clicked()), this, SLOT(onPictureClicked()));
}

vvScreenshotDialog::~vvScreenshotDialog()
{
  vvDebugMsg::msg(1, "vvScreenshotDialog::~vvScreenshotDialog()");
}

void vvScreenshotDialog::takePicture()
{
  vvDebugMsg::msg(1, "vvScreenshotDialog::takePicture()");

  if (impl_->ui->dirEdit->text().isEmpty())
  {
    vvDebugMsg::msg(0, "Directory empty");
    return;
  }

  if (impl_->ui->baseNameEdit->text().isEmpty())
  {
    vvDebugMsg::msg(0, "Base file name empty");
    return;
  }

  QFileInfo info(impl_->ui->dirEdit->text());
  if (!info.isDir())
  {
    vvDebugMsg::msg(0, "Invalid directory");
    return;
  }

  int w = _canvas->width();
  int h = _canvas->height();

  if (!impl_->ui->canvasSizeCheckBox->isChecked())
  {
    w = impl_->ui->imgWidthEdit->text().toInt();
    h = impl_->ui->imgHeightEdit->text().toInt();
  }

  if (w <= 0 || h <= 0)
  {
    vvDebugMsg::msg(0, "Invalid image size");
    return;
  }

  // reserve RGB image space
  std::vector<uint8_t> image(static_cast<size_t>(w * h * 3));

  // render screenshot to memory
  if (_canvas->getRenderer() != NULL)
  {
    _canvas->makeCurrent();
    _canvas->getRenderer()->renderVolumeRGB(w, h, &image[0]);
  }

  // search for unused file name
  const int fieldWidth = 5; // 00000 - 99999
  const int base = 10;
  for (int i = 0; i < 100000; ++i)
  {
    QString suffix = "-" + QString("%2").arg(i, fieldWidth, base, QChar('0')) + ".tif";
    QString filename = impl_->ui->dirEdit->text() + QDir::separator() + impl_->ui->baseNameEdit->text() + suffix;
    QFileInfo info(filename);
    if (!info.isFile())
    {
      const QByteArray ba = filename.toLatin1();
      vvVolDesc* vd = new vvVolDesc(ba.data(), static_cast<size_t>(w), static_cast<size_t>(h), &image[0]);
      vvDebugMsg::msg(0, "Writing screenshot to file: ", ba.data());
      vvFileIO fio;
      fio.saveVolumeData(vd, false);
      delete vd;
      return;
    }
  }

  vvDebugMsg::msg(0, "Screenshot limit exceeded");
}

void vvScreenshotDialog::setCanvasSize(const QSize& size)
{
  vvDebugMsg::msg(3, "vvScreenshotDialog::setCanvasSize()");

  impl_->ui->canvasSizeLabel->setText(QString::number(size.width()) + " x " + QString::number(size.height()));
}

void vvScreenshotDialog::onCanvasSizeToggled(const bool checked)
{
  vvDebugMsg::msg(3, "vvScreenshotDialog::onCanvasSizeToggled()");

  impl_->ui->imgWidthEdit->setEnabled(!checked);
  impl_->ui->imgHeightEdit->setEnabled(!checked);
}

void vvScreenshotDialog::onBrowseClicked()
{
  vvDebugMsg::msg(3, "vvScreenshotDialog::onBrowseClicked()");

  QString caption = tr("Screenshot Directory");
  QString dir;
  impl_->ui->dirEdit->setText(QFileDialog::getExistingDirectory(this, caption, dir, QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks));
}

void vvScreenshotDialog::onPictureClicked()
{
  vvDebugMsg::msg(3, "vvScreenshotDialog::onPictureClicked()");

  takePicture();
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
