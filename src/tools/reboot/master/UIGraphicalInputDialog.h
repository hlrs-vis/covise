/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef UIGRAPHICALINPUTDIALOG_H
#define UIGRAPHICALINPUTDIALOG_H

#include <qdialog.h>
#include <qmap.h>
#include <qstring.h>

class UIGraphicalInputDialog : public QDialog
{

public:
    UIGraphicalInputDialog(QWidget *parent = 0);
    virtual ~UIGraphicalInputDialog();

    QMap<QString, QString> getUserInputs(const QStringList &inputList);
};
#endif
