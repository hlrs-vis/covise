/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "VRBFileDialog.h"

#include <QWidget>
#include <QLabel>
#include <QComboBox>
#include <QFileDialog>

VRBFileDialog::VRBFileDialog(QWidget *parent)
    : QFileDialog(parent, tr("vrb browser window"))
{
#if 0
   // not present in qt4
   //
   // add a group combobox to standard file browser
   QLabel* label = new QLabel( " Group", this );
   group = new QComboBox( this );
   addWidgets( label, group, 0 );
#endif

    // set the defaults
    setFileMode(QFileDialog::AnyFile);
    QStringList filters;
    filters << "Geometry (*.WRL)"
            << "Geometry (*.wrl  *.iv *.pfb)";
    setNameFilters(filters);
    setDirectory(".");
    setViewMode(QFileDialog::Detail);
}

VRBFileDialog::~VRBFileDialog()
{
}
