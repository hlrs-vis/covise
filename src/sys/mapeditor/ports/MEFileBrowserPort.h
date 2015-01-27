/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_FILEBROWSERPORT_H
#define ME_FILEBROWSERPORT_H

#include <QVector>

#include "ports/MEParameterPort.h"

class QStringList;
class QWidget;

class MENode;
class MEFileBrowser;
class MEExtendedPart;
class MELineEdit;

//=============================================================================
class MEFileBrowserPort : public MEParameterPort
//=============================================================================
{

    Q_OBJECT

public:
    MEFileBrowserPort(MENode *node, QGraphicsScene *scene, const QString &portname, const QString &paramtype, const QString &description);
    MEFileBrowserPort(MENode *node, QGraphicsScene *scene, const QString &portname, int paramtype, const QString &description, int porttype);

    ~MEFileBrowserPort();

#ifdef YAC
    void setValues(covise::coRecvBuffer &);
#endif

    void restoreParam();
    void storeParam();
    void defineParam(QString value, int apptype);
    void modifyParam(QStringList list, int noOfValues, int istart);
    void modifyParameter(QString value);
    void setFilter(QString value);
    void sendParamMessage();
    void moduleParameterRequest();
    void makeLayout(layoutType, QWidget *);

    int getCurrentFilterNum();
    void fileBrowserClosed();
    void setBrowser(MEFileBrowser *fb)
    {
        browser = fb;
    };
    void setCurrentFilterNum(int curr);
    void setBrowserFilter(const QStringList &list);
    void setCurrentFilter(const QString &filt);
    QString getCurrentFilter();
    QString filenameold;
    QStringList &getBrowserFilter();
    MEFileBrowser *getBrowser()
    {
        return browser;
    }

    QString getPathname();
    QString getFullFilename();
#ifdef YAC

    int currentFilter;
    void setPath(const QString &name);
    void setPathname(const QString &name);
    void setFilename(const QString &name);
    QString browserFile, browserPath;
    QString getPath();
    QStringList browserFilter;

#else

    QString getFilename();
#endif

private slots:

    void folderCB();
    void applyCB(const QString &);
    void showPath(const QString &);

private:
    bool fileOpen;

    MELineEdit *editLine[2];
    MEExtendedPart *extendedPart[2];
    MEFileBrowser *browser;

    void switchExtendedPart();
    void changeFolderPixmap();
    void separatePath(QString);
    void removeFromControlPanel();
    void addToControlPanel();
};
#endif
