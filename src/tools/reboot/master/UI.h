/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef RB_UI_H
#define RB_UI_H

#include <qmap.h>
#include <qstringlist.h>

class RemoteRebootMaster;

class UI
{

public:
    UI(RemoteRebootMaster *master)
    {
        this->master = master;
    }

    virtual ~UI()
    {
    }

    virtual QString getUserInput(const QString &name, const QString &type, const QString &defaultValue = "") = 0;
    virtual QMap<QString, QString> getUserInputs(const QStringList &inputList) = 0;
    virtual void exec()
    {
    }

protected:
    RemoteRebootMaster *master;
};
#endif
