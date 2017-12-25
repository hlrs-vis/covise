/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TUI_FILEBROWSER_BUTTON_H
#define CO_TUI_FILEBROWSER_BUTTON_H

#include <QObject>

#include "TUIElement.h"
#include "FileBrowser/FileBrowser.h"

class FileBrowser;

class TUIFileBrowserButton : public QObject, public TUIElement
{
    Q_OBJECT
public:
    TUIFileBrowserButton(int id, int type, QWidget *w, int parent, QString name);
    virtual void setEnabled(bool en);
    virtual void setHighlighted(bool hl);
    virtual void setValue(int type, covise::TokenBuffer &tb);
    void sendVal(int type);

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;
    ~TUIFileBrowserButton(void);

signals:
    void dirUpdate(QStringList list);
    void fileUpdate(QStringList list);
    void clientUpdate(QStringList list);
    void curDirUpdate(QString curDir);
    void driveUpdate(QStringList list);
    void updateMode(int mode);
    void updateFilterList(char *filterList);
    void locationUpdate(QString strEntry);
    void updateRemoteButtonState(int);
    void updateLoadCheckBox(bool);

public slots:
    void onPressed();
    void handleRequestLists(QString filter, QString location);
    void handleFilterUpdate(QString filter);
    void handleDirChange(QString dir);
    void handleClientRequest();
    void handleLocationChange(QString location);
    void handleLocalHome();
    void handleReqDriveList();
    void handleReqDirUp(QString path);
    void sendSelectedFile(QString file, QString dir, bool loadAll);
    void handlePathSelected(QString file, QString path);

private:
    FileBrowser *mFileBrowser;
};
#endif
