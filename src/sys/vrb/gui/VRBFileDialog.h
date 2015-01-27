/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRBFILEDIALOG_H
#define VRBFILEDIALOG_H

#include <QFileDialog>

class QComboBox;

class VRBFileDialog : public QFileDialog
{
    Q_OBJECT

public:
    VRBFileDialog(QWidget *parent);
    ~VRBFileDialog();

    QComboBox *group;
};
#endif
