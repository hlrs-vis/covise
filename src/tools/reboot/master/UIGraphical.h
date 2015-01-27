/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef UIGRAPHICAL_H
#define UIGRAPHICAL_H

#include "UI.h"

#include <qdialog.h>

class UIGraphical : public QDialog, public UI
{

public:
    UIGraphical(RemoteRebootMaster *master);
    virtual ~UIGraphical();

    virtual QString getUserInput(const QString &name, const QString &type, const QString &defaultValue = "");
    virtual QMap<QString, QString> getUserInputs(const QStringList &inputList);
    virtual void exec();
};
#endif
