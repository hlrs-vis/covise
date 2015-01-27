/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_FILEBROWSER_H
#define ME_FILEBROWSER_H

#include <QDialog>
#include <QLabel>

class QComboBox;
class QVBoxLayout;
class QHBoxLayout;
class QLineEdit;
class QPushButton;
class QStringList;
class QListWidget;
class QListWidgetItem;

class MEFileBrowserPort;
class MEHost;
class MENode;
class myPathLabel;

//================================================
class MEFileBrowser : public QDialog
//================================================
{
    Q_OBJECT

public:
    MEFileBrowser(QWidget *parent = 0, MEFileBrowserPort * = 0);
    ~MEFileBrowser();

    // browser modes
    enum browsermodes
    {
        MODULEPORT,
        OPENNET,
        SAVENET,
        SAVEASNET,
        CMAPMAINOPEN,
        CMAPMAINSAVE,
        CMAPPORTOPEN,
        CMAPPORTSAVE
    };

    enum requestmodes
    {
        FB_OPEN,
        FB_APPLY,
        FB_APPLY2,
        FB_CHECK,
    };

    bool hasPort();
    int getCurrentFilter();
    void setCurrentFilter(int);
    void setNetType(int);
    void setFullFilename(const QString &);
    void setFilename(const QString &filename);
    void setFilter(const QString &);
    void setFilterList(const QStringList &);
    void updateContent();
    QString getFilter();
    QStringList &getFilterList();
    QString getPath();
    QString getFilename();
    QString getPathname();
    MEFileBrowserPort *getPort()
    {
        return port;
    };

    QVector<QWidget *> buttonList;
    void sendDefaultRequest();
    void sendRequest(const QString &, const QString &);
    void updateTree(const QStringList &);
    void lookupFile(const QString &currPath, const QString &filename, int);
    void lookupResult(const QString &requested, const QString &filename, QString &type);

signals:

    void currentPath(const QString &);

private:
    static int instance;
    bool shouldClose;
    int ident, netType, currentMode;

    void makeFirstLine();
    void makeExplorer();
    void makeLastLine();
    void apply();
    void sendMessage(QString);
    void buildMessage(QString);
    void checkInput(const QString &, int mode);
    void updateButtonList(const QString &);
    void updateHistory(const QString &);
    QListWidget *filetable, *directory;

    QString savePath, saveFile, saveFilter, openPath;
    QStringList filterList;
    QVBoxLayout *mainLayout;
    QHBoxLayout *tool2;
    QWidget *central;
    QPushButton *applyButton;
    myPathLabel *homeB, *rootB, *cwdB;
    QComboBox *filter, *m_fileBox;

    MEFileBrowserPort *port;
    MEHost *host;
    MENode *node;

protected:
    void dragEnterEvent(QDragEnterEvent *ev);
    void dropEvent(QDropEvent *e);

private slots:

    void upPressed();
    void homePressed();
    void rootPressed();
    void cwdPressed();
    void cancelCB();
    void applyCB();
    void buttonCB();
    void fileSelection();
    void closeFileBrowserCB();
    void dir2Clicked(QListWidgetItem *);
    void file1Clicked(QListWidgetItem *);
    void file2Clicked(QListWidgetItem *);
    void historyCB(const QString &);
    void filterCB(const QString &);
};

//================================================
class myPathLabel : public QLabel
//================================================
{
    Q_OBJECT

public:
    myPathLabel(const QString &text, QWidget *parent = 0);

signals:

    void clicked();

protected:
    void mouseReleaseEvent(QMouseEvent *e);
    void mousePressEvent(QMouseEvent *e);
};
#endif
