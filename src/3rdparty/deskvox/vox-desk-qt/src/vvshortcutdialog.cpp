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

#include "vvshortcutdialog.h"

#include "ui_vvshortcutdialog.h"

#include <virvo/vvdebugmsg.h>

struct vvShortcutDialog::Impl
{
  Impl() : ui(new Ui::ShortcutDialog) {}

  std::auto_ptr<Ui::ShortcutDialog> ui;

private:

  Impl(Impl const& rhs);
  Impl& operator=(Impl const& rhs);

};

vvShortcutDialog::vvShortcutDialog(QWidget* parent)
  : QDialog(parent)
  , impl_(new Impl)
{
  vvDebugMsg::msg(1, "vvShortcutDialog::vvShortcutDialog()");

  impl_->ui->setupUi(this);
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
