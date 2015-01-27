/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef UITEXT_H
#define UITEXT_H

#include "UI.h"

class UIText : public UI
{

public:
    UIText(RemoteRebootMaster *master);
    virtual ~UIText();

    virtual QString getUserInput(const QString &name, const QString &type, const QString &defaultValue = "");
    virtual QMap<QString, QString> getUserInputs(const QStringList &inputList);
    virtual void exec();
};
#endif
