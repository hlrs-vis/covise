/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_FILEBROWSERLISTHANDLER_H
#define ME_FILEBROWSERLISTHANDLER_H

#include <QObject>
#include <QVector>

class MEFileBrowser;
class MEHost;

class MEFileBrowserListHandler : public QObject
{
    Q_OBJECT

public:
    MEFileBrowserListHandler();
    ~MEFileBrowserListHandler();

    static MEFileBrowserListHandler *instance();

    void addFileBrowser(MEFileBrowser *);
    void removeFileBrowser(MEFileBrowser *);

    void addHostToBrowser(MEHost *);
    void removeHostFromBrowser(MEHost *);
    MEFileBrowser *getBrowserByID(int id);

private:
    QVector<MEFileBrowser *> browserList;
};
#endif
