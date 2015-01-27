/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "MEApplication.h"
#include "handler/MEMainHandler.h"

#include <QFileOpenEvent>

MEApplication::MEApplication(int &argc, char *argv[])
    : QApplication(argc, argv)
{
}

bool MEApplication::event(QEvent *event)
{
    switch (event->type())
    {
    case QEvent::FileOpen:
    {
        QString file = static_cast<QFileOpenEvent *>(event)->file();
        // FIXME: command line arguments are passed as files to open
        if (file.endsWith(".net"))
        {
            MEMainHandler::instance()->openNetworkFile(file);
        }
        return true;
    }
    default:
        return QApplication::event(event);
    }
}
