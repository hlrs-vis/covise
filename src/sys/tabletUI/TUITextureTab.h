/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TUI_TEXTURE_TAB_H
#define CO_TUI_TEXTURE_TAB_H

#ifndef WIN32
#include <stdint.h>
#endif
#include <queue>

#include <QCloseEvent>
#include <QDropEvent>
#include <QDragLeaveEvent>
#include <QGridLayout>
#include <QMouseEvent>
#include <QDragEnterEvent>
#include <QMutex>
#include <QPushButton>
#include <QThread>
#include <QDialog>

#include <QTreeWidgetItem>
//#include <Q3UrlOperator>
#include <QScrollArea>

#include "TUITab.h"
#include "TUILabel.h"
#include <util/coTypes.h>

class QGridLayout;
class QFrame;
class QLineEdit;
class QSpinBox;
class QComboBox;
class QThread;
class QTabWidget;
class QSignalMapper;

class Q3NetworkOperation;
class QTreeWidget;

class VButtonLayout;
class QSocketNotifier;
class TextureThread;
class Directory;

namespace covise
{
class ServerConnection;
class Connection;
class ConnectionList;
class Message;
}

class PixButton : public QPushButton
{
    Q_OBJECT

public:
    PixButton(VButtonLayout *parent);
    virtual QString getFilename()
    {
        return filename;
    };
    virtual void setFilename(QString name)
    {
        filename = name;
    };

protected:
    virtual void mousePressEvent(QMouseEvent *e);
    virtual void mouseDoubleClickEvent(QMouseEvent *e);
    void dragEnterEvent(QDragEnterEvent *e);
    void dragLeaveEvent(QDragLeaveEvent *e);
    void dropEvent(QDropEvent *e);

private:
    VButtonLayout *parentLayout;
    QString filename;

signals:
    void doubleClicked();
};

class VButtonLayout : public QWidget
{
    Q_OBJECT

public:
    VButtonLayout(QWidget *parent, bool dropEnabled = true, int w = 5, int h = 5);
    void add(PixButton *button, int number);
    bool isDropEnabled()
    {
        return dropEnabled;
    };
    virtual void mousePressEvent(QMouseEvent *e);

private:
    QGridLayout *layout;
    QSignalMapper *signalMapper;
    int width;
    int height;
    int count;
    bool dropEnabled;
    QPoint pos;
public slots:
    void buttonSlot(int);
signals:
    void buttonPressed(int);
};

class PixScrollPane : public QScrollArea
{

public:
    PixScrollPane(QWidget *parent);
    void add(VButtonLayout *child);

protected:
    void contentsDragEnterEvent(QDragEnterEvent *e);
    void contentsDragLeaveEvent(QDragLeaveEvent *e);
    void contentsDropEvent(QDropEvent *e);

private:
    VButtonLayout *layout;
};

class TUITextureTab : public QObject, public TUITab
{
    Q_OBJECT
public:
    TUITextureTab(int id, int type, QWidget *w, int parent, QString name);
    virtual ~TUITextureTab();
    virtual const char *getClassName() const;
    virtual void setValue(int type, covise::TokenBuffer &tb);
    void changeTexture(int, uint64_t);
    void setClient(covise::Connection *conn)
    {
        clientConn = conn;
    };
    covise::Connection *getClient()
    {
        return clientConn;
    };
    covise::Connection *getServer()
    {
        return sConn;
    };
    covise::Message *getMessage()
    {
        return msg;
    };
    int openServer();
    void send(covise::TokenBuffer &tb);
    void lock()
    {
        m_mutex.lock();
    };
    void unlock()
    {
        m_mutex.unlock();
    };
    bool isReceivingTextures()
    {
        return receivingTextures;
    };

protected:
    void closeEvent(QCloseEvent *);

private:
    QPixmap smoothPix(const QPixmap &, int dim);
    bool removeDir(QString path);
    int firstTime;
    int currentPicNumber;
    int lastPicNumber;
    bool receivingTextures;
    bool firstRequest;
    int textureModes[21];
    int textureTexGenModes[21];

    QStringList picList;
    //QList<Q3NetworkOperation> noList;
    //Q3UrlOperator	urlOperator;
    QString sourceData;
    QString lastQuest;
    QFrame *frame;
    PixScrollPane *externTexView;
    PixScrollPane *sceneTexView;
    PixScrollPane *searchTexView;
    QPushButton *loadTextureButton;
    QPushButton *updateTexturesButton;
    QPushButton *fromURLButton;
    QPushButton *nextButton;
    QPushButton *prevButton;
    VButtonLayout *sceneLayout;
    VButtonLayout *extLayout;
    VButtonLayout *searchLayout;
    QGridLayout *searchButtonLayout;
    QGridLayout *spinLayout;
    QImage *image;
    QLineEdit *searchField;
    QTimer *timer;
    QTimer *updateTimer;
    QTreeWidget *fileBrowser;
    QSpinBox *textureNumberSpin;
    QComboBox *textureModeComboBox;
    QComboBox *textureTexGenModeComboBox;
    QMenu *menu;
    Directory *currentItem;

    TextureThread *thread;
    QMutex m_mutex;
    QStringList buttonList;

    covise::Connection *sConn;
    covise::Connection *clientConn;
    covise::Message *msg;
    int port;

public slots:

    void sendChangeTextureRequest(int);
    void loadTexture();
    void updateTextures();
    void updateTextureButtons();
    void newData(const QByteArray &);
    void doRequest();
    void progress(int done, int total, Q3NetworkOperation *);
    void endRequest(Q3NetworkOperation *op);
    void nextCopy();
    void incPicNumber();
    void decPicNumber();
    void stopRequest();
    void enableButtons(const QString &);
    void directorySelected(QTreeWidgetItem *);
    void popupExec(QTreeWidgetItem *);
    void changeTexMode(int);
    void setTexMode(int);
    void changeTexGenMode(int);
    void setTexGenMode(int);
    void addFolder();
    void renameFolder();
    void removeFolder();
    void closeServer();
    //void processMessages();
    void handleClient(covise::Message *msg);
};

class StaticProps
{

public:
    static StaticProps *getInstance();
    QList<PixButton *> &getButtonList();
    static QString getTexturePluginDir()
    {
        return texturePluginDir;
    };
    static QString getTexturePluginTempDir()
    {
        return texturePluginTempDir;
    };
    static QString getTextureDir()
    {
        return textureDir;
    };
    static void setCurrentDir(QString dir)
    {
        currentDir = dir;
    };
    static QString getCurrentDir()
    {
        return currentDir;
    };
    static int getContinousNumber()
    {
        return continousNumber++;
    };
    ~StaticProps();

private:
    StaticProps();
    static StaticProps *instance;
    static QString texturePluginDir;
    static QString texturePluginTempDir;
    static QString textureDir;
    static QString currentDir;
    static int continousNumber;
    QList<PixButton *> buttonList;
};

class FilenameDialog : public QDialog
{
    Q_OBJECT
public:
    FilenameDialog(QWidget *parent, QString filename);
    ~FilenameDialog();
    QString getFilename();

public slots:
    void correctSign();

private:
    QLineEdit *nameField;
    QPushButton *accept;
};

class Directory : public QTreeWidgetItem
{

public:
    Directory(QTreeWidget *parent, QString name);
    Directory(Directory *parent, QString name);
    QString name()
    {
        return dirName;
    };
    QString fullPath()
    {
        return path;
    };
    void setOpen(bool o);
    void setName(QString name);
    void setPath(QString path);
    void replaceInPath(QString oldPath, QString newPath);
    QString text(int c) const;

protected:
    QString path, dirName;
    Directory *p;
};

class DirView : public QTreeWidget
{

public:
    DirView(QWidget *parent);

protected:
    virtual void dragEnterEvent(QDragEnterEvent *e);
    virtual void dragLeaveEvent(QDragLeaveEvent *e);
    virtual void dropEvent(QDropEvent *e);
    //virtual void mousePressEvent(QMouseEvent *e);
private:
    QMenu *menu;
};

class TextureThread : public QThread
{
public:
    TextureThread(TUITextureTab *tab);
    void run();
    void enqueueGeode(int number, uint64_t geode);
    void setButtonNumber(int number)
    {
        buttonNumber = number;
    }
    bool isSending()
    {
        return !buttonQueue.empty();
    }
    void terminateTextureThread()
    {
        isRunning = false;
    }

private:
    TUITextureTab *tab;
    int buttonNumber;

    bool isRunning;
    std::queue<int> buttonQueue;
    std::queue<uint64_t> geodeQueue;
};
#endif
