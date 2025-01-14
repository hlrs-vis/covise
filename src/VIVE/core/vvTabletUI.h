/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <util/coTypes.h>
#include <net/tokenbuffer.h>
#include <queue>
#include <map>
#include <list>
#include <string>

#ifndef _M_CEE //no future in Managed OpenCOVER
#include <future>
#endif
//#ifndef WIN32
//#include <stdint.h>
//#define FILESYS_SEP "\\"
//#else
#define FILESYS_SEP "/"
//#endif

#include <tui/coAbstractTabletUI.h>
#include "vvTUIListener.h"

#ifdef USE_QT
#include <QObject>
#include <QMetaType>

#include <vsg/nodes/Node.h>
#include <vsg/nodes/Group.h>
#include <vsg/nodes/MatrixTransform.h>
#include <mutex>

Q_DECLARE_METATYPE(std::string)
#define QT(x) x
#else
#define Q_OBJECT
#define Q_PROPERTY(x)
#define QT(x)
#define slots

namespace Qt
{
enum GlobalColor
{
    color0,
    color1,
    black,
    white,
    darkGray,
    gray,
    lightGray,
    red,
    green,
    blue,
    cyan,
    magenta,
    yellow,
    darkRed,
    darkGreen,
    darkBlue,
    darkCyan,
    darkMagenta,
    darkYellow,
    transparent
};
}
#endif
#define D_COMMA ,

#define THREAD_NOTHING_TO_DO 0

namespace covise
{
class TokenBuffer;
class Host;
class Message;
class Connection;
class ClientConnection;
class ServerConnection;
}
namespace vrb{
class VRBClient;
}
namespace osg
{
class Node;
};

namespace vive
{
class vvTabletUI;
class vvTUIElement;
class SGTextureThread;
class LocalData;
class IData;
class IRemoteData;

/**
 * Tablet PC Userinterface Mamager.
 * This class provides a connection to a Tablet PC and handles all vvTUIElements.
 */
class VVCORE_EXPORT vvTabletUI: QT(public QObject D_COMMA) public covise::coAbstractTabletUI
{
    Q_OBJECT

    Q_PROPERTY(int id READ getID)

private:
    static vvTabletUI *tUI;
    std::mutex connectionMutex;

public slots:
    int getID();

public:
    enum ConnectionMode
    {
        None,
        Config,
        Client,
        ConnectedSocket,
    };
    struct ConfigData
    {
        ConfigData(const std::string &h, int p): mode(Client), host(h), port(p) {}
        ConfigData(): mode(Config) {}
        ConfigData(int fd, int fdSg): mode(ConnectedSocket), fd(fd), fdSg(fdSg) {}

        ConnectionMode mode = None;
        int fd = -1;
        int fdSg = -1;
        int port = 0;
        std::string host;
    };

    vvTabletUI(); // Config: read from config
    vvTabletUI(const std::string &host, int port); // Client: connect as client to host:port
    vvTabletUI(
        int fd,
        int fdS); // ConnectedSocket: (connected) file descriptors, e.g. from socketpair(2), also for Scenegraph browser connection
    virtual ~vvTabletUI();
    static vvTabletUI *instance();

    virtual bool update();
    void addElement(vvTUIElement *);
    void removeElement(vvTUIElement *e);
    bool send(covise::TokenBuffer &tb);
    void tryConnect();
    void close();
    bool debugTUI();
    bool isConnected() const;
    void init();
    void reinit(const ConfigData &cd);

    void lock()
    {
        connectionMutex.lock();
    }
    void unlock()
    {
        connectionMutex.unlock();
    }
    covise::Host *connectedHost = nullptr;

    bool serverMode = false;
    std::unique_ptr<covise::Connection> sgConn;

protected:
    ConnectionMode mode = None;
    ConfigData connectionConfig;
    void config();

    void resendAll();
    std::vector<vvTUIElement *> elements;
    std::vector<vvTUIElement *> newElements;
    covise::ServerConnection *serverConn = nullptr;
    covise::Host *serverHost = nullptr;
    covise::Host *localHost = nullptr;
    int port = 0;
    int ID = 3;
    float timeout = 1.f;
    bool debugTUIState = false;
    double oldTime = 0.0;
    bool firstConnection = true;

    std::unique_ptr<covise::Connection> conn; // protected by lock/unlock
    covise::Host *connectingHost = nullptr; // protected by lock/unlock
#ifndef _M_CEE //no future in Managed OpenCOVER
    std::atomic<bool> connecting = false;
    std::thread connThread;
    std::mutex sendMutex;
    std::thread sendThread;
    std::condition_variable sendCond;
    std::deque<covise::DataHandle> sendQueue;
#endif
};


/**
 * Base class for Tablet PC UI Elements.
 */
class VVCORE_EXPORT vvTUIElement: QT(public QObject D_COMMA) public covise::coAbstractTUIElement
{
    Q_OBJECT

    Q_PROPERTY(int id READ getID)
    Q_PROPERTY(std::string name READ getName)

public:
    vvTUIElement(const std::string &, int pID = 1);
#ifdef USE_QT
    vvTUIElement(QObject *parent, const std::string &, int pID);
    vvTUIElement(QObject *parent, const std::string &, int pID, int type);
#endif
    virtual ~vvTUIElement();
    virtual void parseMessage(covise::TokenBuffer &tb) override;
    virtual void resend(bool create) override;
    virtual void setEventListener(vvTUIListener *);
    virtual vvTUIListener *getMenuListener() override;
    bool createSimple(int type);
    vvTabletUI *tui() const;

public
    QT(slots): void setVal(const std::string &value);
    void setVal(bool value);
    void setVal(int value);
    void setVal(float value);
    void setVal(int type, int value);
    void setVal(int type, float value);
    void setVal(int type, int value, const std::string &nodePath);
    void setVal(int type, const std::string &nodePath, const std::string &simPath, const std::string &simName);
    void setVal(int type, int value, const std::string &nodePath, const std::string &simPath);

    int getID() const;

    virtual void setPos(int, int) override;
    virtual void setSize(int, int) override;
    virtual void setLabel(const char *l) override;
    virtual void setLabel(const std::string &l);
    virtual void setColor(Qt::GlobalColor);
    virtual void setHidden(bool);
    virtual void setEnabled(bool);
    std::string getName() const
    {
        return name;
    }

protected:
    vvTUIElement(const std::string &, int pID, int type);
    vvTUIElement(vvTabletUI *tui, const std::string &, int pID, int type);

    int type = -1;
    int parentID;
    std::string name; ///< name of this element
    std::string label; ///< label of this element
    int ID; ///< unique ID
    int xs, ys, xp, yp;
    Qt::GlobalColor color;
    bool hidden = false;
    bool enabled = true;
    vvTUIListener *listener = nullptr; ///< event listener
    vvTabletUI *m_tui = nullptr;
};

/**
 * a static textField.
 */
class VVCORE_EXPORT vvTUILabel : public vvTUIElement
{

    Q_OBJECT

private:
public:
    vvTUILabel(const std::string &, int pID = 1);
    vvTUILabel(vvTabletUI *tui, const std::string &, int pID = 1);
#ifdef USE_QT
    vvTUILabel(QObject *, const std::string &, int pID = 1);
#endif
    virtual ~vvTUILabel();
    virtual void resend(bool create) override;

protected:
};
/**
 * a push button with a Bitmap
 */
class VVCORE_EXPORT vvTUIBitmapButton : public vvTUIElement
{

    Q_OBJECT

private:
public:
    vvTUIBitmapButton(const std::string &, int pID = 1);
    vvTUIBitmapButton(vvTabletUI *tui, const std::string &, int pID = 1);
#ifdef USE_QT
    vvTUIBitmapButton(QObject *, const std::string &, int pID = 1);
#endif
    virtual ~vvTUIBitmapButton();
    virtual void parseMessage(covise::TokenBuffer &tb) override;

#ifdef USE_QT
signals:
    void tabletEvent();
    void tabletPressEvent();
    void tabletReleaseEvent();
#endif

protected:
};
/**
 * a push button.
 */
class VVCORE_EXPORT vvTUIButton : public vvTUIElement
{

    Q_OBJECT

private:
public:
    vvTUIButton(const std::string &, int pID = 1);
    vvTUIButton(vvTabletUI *tui, const std::string &, int pID = 1);
#ifdef USE_QT
    vvTUIButton(QObject *parent, const std::string &, int pID = 1);
#endif
    virtual ~vvTUIButton();
    virtual void parseMessage(covise::TokenBuffer &tb) override;

#ifdef USE_QT
signals:
    void tabletEvent();
    void tabletPressEvent();
    void tabletReleaseEvent();
#endif

protected:
};

/**
 * the filebrowser push button.
 */
class VVCORE_EXPORT vvTUIFileBrowserButton : public vvTUIElement
{
public:
    enum DialogMode
    {
        OPEN = 1,
        SAVE = 2
    };
    vvTUIFileBrowserButton(const char *, int pID = 1);
    vvTUIFileBrowserButton(vvTabletUI *tui, const char *, int pID=1);
    virtual ~vvTUIFileBrowserButton();

    // Sends a directory list to TUI
    virtual void setDirList(const covise::Message &ms);

    // Sends a file list to TUI
    virtual void setFileList(const covise::Message &ms);

    // Sends the currently used directory to TUI
    // Uses setCurDir(char*)
    void setCurDir(const covise::Message &msg);

    // Sends the currently used directory to TUI
    void setCurDir(const char *dir);

    // Resends all FileBrowser required data to TUI
    virtual void resend(bool create) override;

    // Parses all messages arriving from TUI
    virtual void parseMessage(covise::TokenBuffer &tb) override;

    // sends the list of VRB clients in a session to TUI
    void setClientList(const covise::Message &msg);

    // retieve currently used data object
    // either LocalData, VRBData or AGData
    IData *getData(std::string protocol = "");

    // Retrieves the instance of the VRBData of the FileBrowser
    IData *getVRBData();

    //Sends a list of drives to the TUI
    void setDrives(const covise::Message &ms);

    // Returns the filename to a file in the local file system
    // based on a URI-Filelocation e.g. vrb://visper.hlrs.de//mnt/raid/tmp/test.wrl
    std::string getFilename(const std::string url);

    // Returns a file handle based on a URI in above mentioned format.
    // Not yet implemented.
    void *getFileHandle(bool sync = false);

    // Sets the file browser dialog mode (SAVE or OPEN) --> DialogMode
    void setMode(DialogMode mode);

    // Sends the currently used filter list for the filebrowser to the TUI
    void setFilterList(std::string filterList);

    // Returns a string containing a selected path
    // What was that again?
    std::string getSelectedPath();

protected:
    void sendList(covise::TokenBuffer &tb);
    // Stores the list of files as retrieved
    // from the storage location, however it is
    // recreated upon request basis.
    std::vector<std::string> mFileList;

    // Stores the list of directories as retrieved
    // from the storage location, however it is
    // recreated upon request basis.
    std::vector<std::string> mDirList;

    // Stores a list fo clients. Is this actually used?
    std::vector<std::string> mClientList;

    // General data object for access to non-local storage locations
    // e.g. VRB, AccessGrid
    IRemoteData *mDataObj;

    // Data Object for access to local file system
    LocalData *mLocalData;

    // Data Object for access to AccessGrid data store
    IRemoteData *mAGData;

    // Generic Data object interface only providing basic
    // functionality, capable of holding all data object types
    IData *mData;

    // Stores the location from where to determine file information
    // either as IP address or hostname
    std::string mLocation;

    // Stores the IP address of the system the local OpenCOVER runs on
    std::string mLocalIP;

    // Stores the currently selected directory as selected in the filebrowser
    // dialog of the TUI
    std::string mCurDir;

    // ??
    std::string mFile;

    // Stores the dialog mode of this TUIFileBrowserButton instance
    // either Open dialog or save dialog, currently implementation is
    // only focused on file open.
    DialogMode mMode;

    // String that contains a list of file-extensions to be used when
    // creating file list for the file browser
    // e.g. "*.*;*.wrl"
    std::string mFilterList;

    // Id of the VRB client which is required to create some outgoing messages
    // to the VRB server.
    int mVRBCId;

    // Stores data objects related to their protocol identifier
    std::map<std::string, IData *> mDataRepo;
    typedef std::pair<std::string, IData *> Data_Pair;

    // Id of the TUIFileBrowserButton
    int mId;

    /**
       * Member containing the selected save path when in SAVE mode
       */
    std::string mSavePath;
};

class VVCORE_EXPORT vvTUIColorTriangle : public vvTUIElement
{

    Q_OBJECT

    Q_PROPERTY(float red READ getRed WRITE setRed)
    Q_PROPERTY(float green READ getGreen WRITE setGreen)
    Q_PROPERTY(float blue READ getBlue WRITE setBlue)

private:
public:
    vvTUIColorTriangle(const std::string &, int pID = 1);
#ifdef USE_QT
    vvTUIColorTriangle(QObject *parent, const std::string &, int pID = 1);
#endif
    virtual ~vvTUIColorTriangle();
    virtual void resend(bool create) override;
    virtual void parseMessage(covise::TokenBuffer &tb) override;

public slots:
    virtual float getRed() const
    {
        return red;
    }
    virtual float getGreen() const
    {
        return green;
    }
    virtual float getBlue() const
    {
        return blue;
    }
    virtual void setRed(float red)
    {
        this->red = red;
    }
    virtual void setGreen(float green)
    {
        this->green = green;
    }
    virtual void setBlue(float blue)
    {
        this->blue = blue;
    }
    virtual void setColor(float r, float g, float b);
    virtual void setColor(Qt::GlobalColor c) override
    {
        vvTUIElement::setColor(c);
    }
    //virtual void switchLocation(LocationType type);

#ifdef USE_QT
signals:
    void tabletEvent();
    void tabletReleaseEvent();
#endif

protected:
    float red;
    float green;
    float blue;
};
class VVCORE_EXPORT vvTUIColorButton : public vvTUIElement
{
    Q_OBJECT

    Q_PROPERTY(float red READ getRed WRITE setRed)
    Q_PROPERTY(float green READ getGreen WRITE setGreen)
    Q_PROPERTY(float blue READ getBlue WRITE setBlue)
    Q_PROPERTY(float alpha READ getAlpha WRITE setAlpha)

private:
public:
    vvTUIColorButton(const std::string &, int pID = 1);
    vvTUIColorButton(vvTabletUI *tui, const std::string &, int pID = 1);
#ifdef USE_QT
    vvTUIColorButton(QObject *parent, const std::string &, int pID = 1);
#endif
    virtual ~vvTUIColorButton();
    virtual void resend(bool create) override;
    virtual void parseMessage(covise::TokenBuffer &tb) override;

public slots:
    virtual float getRed() const
    {
        return red;
    }
    virtual float getGreen() const
    {
        return green;
    }
    virtual float getBlue() const
    {
        return blue;
    }
    virtual float getAlpha() const
    {
        return alpha;
    }
    virtual void setRed(float red)
    {
        this->red = red;
    }
    virtual void setGreen(float green)
    {
        this->green = green;
    }
    virtual void setBlue(float blue)
    {
        this->blue = blue;
    }
    virtual void setAlpha(float alpha)
    {
        this->alpha = alpha;
    }
    virtual void setColor(float r, float g, float b, float a);
    virtual void setColor(Qt::GlobalColor c) override
    {
        vvTUIElement::setColor(c);
    }
    //virtual void switchLocation(LocationType type);

#ifdef USE_QT
signals:
    void tabletEvent();
    void tabletReleaseEvent();
#endif

protected:
    float red;
    float green;
    float blue;
    float alpha;
};

#ifdef USE_QT
class VVCORE_EXPORT vvTUIColorTab : public vvTUIElement
{
    Q_OBJECT

    Q_PROPERTY(float red READ getRed WRITE setRed)
    Q_PROPERTY(float green READ getGreen WRITE setGreen)
    Q_PROPERTY(float blue READ getBlue WRITE setBlue)
    Q_PROPERTY(float alpha READ getAlpha WRITE setAlpha)

private:
public:
    vvTUIColorTab(const std::string &, int pID = 1);
    vvTUIColorTab(QObject *parent, const std::string &, int pID = 1);
    virtual ~vvTUIColorTab();
    virtual void resend(bool create) override;
    virtual void parseMessage(covise::TokenBuffer &tb) override;

public slots:
    virtual float getRed() const
    {
        return red;
    }
    virtual float getGreen() const
    {
        return green;
    }
    virtual float getBlue() const
    {
        return blue;
    }
    virtual float getAlpha() const
    {
        return alpha;
    }
    virtual void setRed(float red)
    {
        this->red = red;
    }
    virtual void setGreen(float green)
    {
        this->green = green;
    }
    virtual void setBlue(float blue)
    {
        this->blue = blue;
    }
    virtual void setAlpha(float alpha)
    {
        this->alpha = alpha;
    }
    virtual void setColor(float r, float g, float b, float a);
    virtual void setColor(Qt::GlobalColor c) override
    {
        vvTUIElement::setColor(c);
    }

signals:
    void tabletEvent();

protected:
    float red;
    float green;
    float blue;
    float alpha;
};
#endif

class VVCORE_EXPORT vvTUIFunctionEditorTab : public vvTUIElement
{
public:
    static const int histogramBuckets = 256;
    int *histogramData;

    // my transfer function parameters: what is needed?
    // for 1D, only points.

    // They have the same values of virvo TF widgets!
    enum TFKind
    {
        TF_COLOR = 0,
        TF_PYRAMID = 1,
        TF_BELL = 2,
        //TF_SKIP = 3,
        TF_FREE = 4,
        TF_CUSTOM_2D = 5,
        TF_MAP = 6,
        TF_CUSTOM_2D_EXTRUDE = 11,
        TF_CUSTOM_2D_TENT = 12
    };

    struct colorPoint
    {
        float r;
        float g;
        float b;
        float x;
        float y;
    };

    struct alphaPoint
    {
        coInt32 kind;
        float alpha;
        float xPos;
        float xParam1;
        float xParam2;
        float yPos;
        float yParam1;
        float yParam2;
        coInt32 ownColor;
        float r;
        float g;
        float b;
        coInt32 additionalDataElems; //int additionalDataElemSize;
        float *additionalData;
    };

    std::vector<colorPoint> colorPoints;
    std::vector<alphaPoint> alphaPoints;

    coInt32 tfDim;

public:
    vvTUIFunctionEditorTab(const char *tabName, int pID = 1);
    virtual ~vvTUIFunctionEditorTab();

    int getDimension() const;
    void setDimension(int);

    virtual void resend(bool create) override;
    void sendHistogramData();
    virtual void parseMessage(covise::TokenBuffer &tb) override;
};


/**
 * a tab.
 */
class VVCORE_EXPORT vvTUITab : public vvTUIElement
{

    Q_OBJECT

private:
public:
    vvTUITab(const std::string &, int pID = 1);
    vvTUITab(vvTabletUI *tui, const std::string &, int pID);
#ifdef USE_QT
    vvTUITab(QObject *parent, const std::string &, int pID);
#endif
    virtual ~vvTUITab();
    virtual void parseMessage(covise::TokenBuffer &tb) override;
    void allowRelayout(bool rl);
    void resend(bool create) override;

#ifdef USE_QT
signals:
    void tabletEvent();
    void tabletPressEvent();
    void tabletReleaseEvent();
#endif

protected:
    bool m_allowRelayout = false;
};

#ifdef USE_QT
/**
 * a dynamic UI tab.
 */
class VVCORE_EXPORT vvTUIUITab : public vvTUIElement
{

    Q_OBJECT

private:
public:
    vvTUIUITab(const std::string &, int pID = 1);
    vvTUIUITab(vvTabletUI *tui, const std::string &, int pID = 1);
    vvTUIUITab(QObject *parent, const std::string &, int pID);
    virtual ~vvTUIUITab();
    virtual void parseMessage(covise::TokenBuffer &tb) override;

    bool loadUIFile(const std::string &filename);
    void sendEvent(const QString &source, const QString &event);

signals:
    void tabletEvent();
    void tabletPressEvent();
    void tabletReleaseEvent();

    void tabletUICommand(const QString &target, const QString &command);

private:
    QString filename;
    QString uiDescription;
};
#endif

/**
 * a tab folder.
 */
class VVCORE_EXPORT vvTUITabFolder : public vvTUIElement
{
    Q_OBJECT

private:
public:
    vvTUITabFolder(const std::string &, int pID = 1);
    vvTUITabFolder(vvTabletUI *tui, const std::string &, int pID = 1);
#ifdef USE_QT
    vvTUITabFolder(QObject *parent, const std::string &, int pID = 1);
#endif
    virtual ~vvTUITabFolder();
    virtual void parseMessage(covise::TokenBuffer &tb) override;

#ifdef USE_QT
signals:
    void tabletEvent();
    void tabletPressEvent();
    void tabletReleaseEvent();
#endif

protected:
};

class VVCORE_EXPORT vvTUISGBrowserTab : public vvTUIElement
{
private:
    std::string findName;
    int visitorMode;
    int polyMode;
    int selOnOff;
    int sendImageMode;

    float ColorR, ColorG, ColorB;
    std::string expandPath;
    std::string selectPath;
    std::string selectParentPath;
    std::string showhidePath;
    std::string showhideParentPath;

    std::vector<std::string> parsePathString(std::string path);
    vsg::Node* getNode(std::string path);

public:
    float diffuse[4];
    float specular[4];
    float ambient[4];
    float emissive[4];
    float matrix[16];
    bool loadFile;

    vvTUISGBrowserTab(vvTabletUI *tui, const char *, int pID = 1);
    virtual ~vvTUISGBrowserTab();
    virtual void resend(bool create) override;

    virtual void parseMessage(covise::TokenBuffer &tb) override;
    virtual void sendType(int type, const char *nodeType, const char *name, const char *path, const char *pPath, int mode, int numChildren = 0);
    virtual void sendEnd();
    virtual void sendProperties(std::string path, std::string pPath, int mode, int transparent);
    virtual void sendProperties(std::string path, std::string pPath, int mode, int transparent, float mat[]);
    virtual void sendCurrentNode(vsg::Node *node, std::string);
    virtual void sendRemoveNode(std::string path, std::string parentPath);
    virtual void sendShader(std::string name);
    virtual void sendUniform(std::string name, std::string type, std::string value, std::string min, std::string max, std::string textureFile);
    virtual void sendShaderSource(std::string vertex, std::string fragment, std::string geometry, std::string tessControl, std::string tessEval);
    virtual void updateUniform(std::string shader, std::string name, std::string value, std::string textureFile);
    virtual void updateShaderSourceV(std::string shader, std::string vertex);
    virtual void updateShaderSourceF(std::string shader, std::string fragment);
    virtual void updateShaderSourceG(std::string shader, std::string geometry);
    virtual void updateShaderSourceTE(std::string shader, std::string tessEval);
    virtual void updateShaderSourceTC(std::string shader, std::string tessControl);
    virtual void updateShaderNumVertices(std::string shader, int);
    virtual void updateShaderOutputType(std::string shader, int);
    virtual void updateShaderInputType(std::string shader, int);

    virtual void addNode(const char* nodePath, int nodeType);
    virtual void removeNode(const char* nodePath, const char* parent_nodePath);
    virtual void moveNode(const char* nodePath, const char* oldParent_nodePath, const char* newParent_nodePath, unsigned int dropIndex);
    virtual void renameNode(const char* nodePath, const char* nodeNewName);

    virtual const std::string &getFindName() const
    {
        return findName;
    }
    virtual int getVisMode() const
    {
        return visitorMode;
    }
    virtual int getImageMode() const
    {
        return sendImageMode;
    }

    virtual vsg::Node *getCurrentNode()
    {
        return currentNode;
    }

    virtual const std::string &getExpandPath() const
    {
        return expandPath;
    }
    virtual const std::string &getSelectPath() const
    {
        return selectPath;
    }
    virtual const std::string &getSelectParentPath() const
    {
        return selectParentPath;
    }
    virtual const std::string &getShowHidePath() const
    {
        return showhidePath;
    }
    virtual const std::string &getShowHideParentPath() const
    {
        return showhideParentPath;
    }
    virtual float getR() const
    {
        return ColorR;
    }
    virtual float getG() const
    {
        return ColorG;
    }
    virtual float getB() const
    {
        return ColorB;
    }
    virtual int getPolyMode() const
    {
        return polyMode;
    }
    virtual int getSelMode() const
    {
        return selOnOff;
    }

    virtual void parseTextureMessage(covise::TokenBuffer &tb);
    virtual void setTexture(int height, int width, int depth, int texIndex, int dataLength, const char *data);
    virtual void setTexture(int texNumber, int mode, int texGenMode, int texIndex);
    virtual void setCurrentNode(vsg::Node *node)
    {
        currentNode = node;
    }
    virtual void setCurrentPath(std::string str)
    {
        currentPath = str;
    }
    virtual void decTexturesToChange()
    {
        if (texturesToChange > 0)
            --texturesToChange;
    }
    virtual void finishedTraversing();
    virtual void sendTraversedTextures();
    virtual void finishedNode();
    virtual void noTexture();
    virtual void sendNodeTextures();
    virtual void sendNoTextures();
    virtual void incTextureListCount();
    virtual void sendTexture();
    virtual void loadFilesFlag(bool state);
    virtual void hideSimNode(bool state, const char *nodePath, const char *parentPath);
    virtual void setSimPair(const char *nodePath, const char *simPath, const char *simName);

    virtual int queueIsEmpty() const
    {
        return _dataList.empty();
    }
    virtual int getHeight() const
    {
        return height;
    }
    virtual int getWidth() const
    {
        return width;
    }
    virtual int getDepth() const
    {
        return depth;
    }
    virtual int getIndex() const
    {
        return index;
    }
    virtual size_t getDataLength() const;
    virtual int getTextureNumber() const
    {
        return textureNumber;
    }
    virtual int getTextureMode() const
    {
        return textureMode;
    }
    virtual int getTextureTexGenMode() const
    {
        return textureTexGenMode;
    }
    virtual int getTexturesToChange() const
    {
        return texturesToChange;
    }
    virtual int hasAlpha() const
    {
        return alpha;
    }
    virtual char *getData();
    virtual covise::Connection *getConnection();
    virtual vsg::Node *getChangedNode()
    {
        return changedNode;
    }
    virtual const std::string &getChangedPath() const
    {
        return changedPath;
    }

    void send(covise::TokenBuffer &tb);
    void parseTextureMessage();

protected:
    virtual void lock()
    {
        mutex.lock();
    }
    virtual void unlock()
    {
        mutex.unlock();
    }
    int texturesToChange = 0;
    int height = 0;
    int width = 0;
    int depth = 0;
    std::vector<char> data;
    int textureNumber;

    int index;

    vsg::Node *changedNode = nullptr;
    //covise::Connection *conn = nullptr;
    covise::ServerConnection *sConn = nullptr;

    std::queue<int> _heightList;
    std::queue<int> _widthList;
    std::queue<int> _depthList;
    std::queue<int> _indexList;
    std::queue<int> _lengthList;
    std::queue<const char *> _dataList;

    int textureMode;
    int textureTexGenMode;
    int alpha;

    covise::Host *serverHost = nullptr;
    covise::Host *localHost = nullptr;
    int texturePort;
    SGTextureThread *thread = nullptr;
    std::mutex mutex;

    vsg::Node *currentNode = nullptr;
    std::string currentPath;
    std::string changedPath;
};


class VVCORE_EXPORT vvTUIAnnotationTab : public vvTUIElement
{
public:
    vvTUIAnnotationTab(const char *, int pID = 1);
    virtual ~vvTUIAnnotationTab();
    virtual void parseMessage(covise::TokenBuffer &tb) override;

    void setNewButtonState(bool state);
    void addAnnotation(int id);
    void deleteAnnotation(int mode, int id);
    void setSelectedAnnotation(int id);
};

/**
 * a NavigationElement.
 */
class VVCORE_EXPORT vvTUINav : public vvTUIElement
{
private:
public:
    vvTUINav(vvTabletUI *tui, const char *, int pID = 1);
    virtual ~vvTUINav();
    virtual void parseMessage(covise::TokenBuffer &tb) override;
    bool down;
    int x;
    int y;

protected:
};
/**
 * a Splitter.
 */
class VVCORE_EXPORT vvTUISplitter : public vvTUIElement
{
    Q_OBJECT

    Q_PROPERTY(int shape READ getShape WRITE setShape)
    Q_PROPERTY(int style READ getStyle WRITE setStyle)
    Q_PROPERTY(int orientation READ getOrientation WRITE setOrientation)

private:
public:
    enum orientations
    {
        Horizontal = 0x1,
        Vertical = 0x2
    };

    vvTUISplitter(const std::string &, int pID = 1);
#ifdef USE_QT
    vvTUISplitter(QObject *parent, const std::string &, int pID = 1);
#endif
    virtual ~vvTUISplitter();
    virtual void resend(bool create) override;
    virtual void parseMessage(covise::TokenBuffer &tb) override;

public slots:
    virtual void setShape(int s);
    virtual void setStyle(int t);
    virtual void setOrientation(int orientation);
    virtual int getShape() const
    {
        return this->shape;
    }
    virtual int getStyle() const
    {
        return this->style;
    }
    virtual int getOrientation() const
    {
        return this->orientation;
    }

#ifdef USE_QT
signals:
    void tabletEvent();
    void tabletPressEvent();
    void tabletReleaseEvent();
#endif

protected:
    int shape;
    int style;
    int orientation;
};

/**
 * a Frame.
 */
class VVCORE_EXPORT vvTUIFrame : public vvTUIElement
{

    Q_OBJECT

    //Q_ENUMS(styles)
    //Q_ENUMS(shapes)

    Q_PROPERTY(int shape READ getShape WRITE setShape)
    Q_PROPERTY(int style READ getStyle WRITE setStyle)

private:
public:
    enum styles
    {
        Plain = 0x0010,
        Raised = 0x0020,
        Sunken = 0x0030
    };
    enum shapes
    {
        NoFrame = 0x0000,
        Box = 0x0001,
        Panel = 0x0002,
        WinPanel = 0x0003,
        HLine = 0x0004,
        VLine = 0x0005,
        StyledPanel = 0x0006
    };

    vvTUIFrame(const std::string &, int pID = 1);
    vvTUIFrame(vvTabletUI *tui, const std::string &, int pID = 1);
#ifdef USE_QT
    vvTUIFrame(QObject *parent, const std::string &, int pID = 1);
#endif
    virtual ~vvTUIFrame();
    virtual void resend(bool create) override;
    virtual void parseMessage(covise::TokenBuffer &tb) override;

public slots:
    virtual void setShape(int s); /* set shape first */
    virtual void setStyle(int t);

    virtual int getShape() const
    {
        return this->shape;
    }
    virtual int getStyle() const
    {
        return this->style;
    }

#ifdef USE_QT
signals:
    void tabletEvent();
    void tabletPressEvent();
    void tabletReleaseEvent();
#endif

protected:
    int style;
    int shape;
};

/**
 * a GroupBox.
 */
class VVCORE_EXPORT vvTUIGroupBox : public vvTUIElement
{

    Q_OBJECT

private:
public:
    vvTUIGroupBox(const std::string &, int pID = 1);
    vvTUIGroupBox(vvTabletUI *tui, const std::string &, int pID = 1);
#ifdef USE_QT
    vvTUIGroupBox(QObject *parent, const std::string &, int pID = 1);
#endif
    virtual ~vvTUIGroupBox();
    virtual void parseMessage(covise::TokenBuffer &tb) override;

public slots:

#ifdef USE_QT
signals:
    void tabletEvent();
    void tabletPressEvent();
    void tabletReleaseEvent();
#endif
};

/**
 * a toggle button.
 */
class VVCORE_EXPORT vvTUIToggleButton : public vvTUIElement
{
    Q_OBJECT

    Q_PROPERTY(bool state READ getState WRITE setState)

private:
public:
    vvTUIToggleButton(const std::string &, int pID = 1, bool state = false);
    vvTUIToggleButton(vvTabletUI *tui, const std::string &, int pID = 1, bool state = false);
#ifdef USE_QT
    vvTUIToggleButton(QObject *parent, const std::string &, int pID = 1, bool state = false);
#endif
    virtual ~vvTUIToggleButton();
    virtual void resend(bool create) override;
    virtual void parseMessage(covise::TokenBuffer &tb) override;

public slots:
    virtual void setState(bool s);
    virtual bool getState() const;

#ifdef USE_QT
signals:
    void tabletEvent();
    void tabletPressEvent();
    void tabletReleaseEvent();
#endif

protected:
    bool state;
};
/**
 * a toggleBitmapButton.
 */
class VVCORE_EXPORT vvTUIToggleBitmapButton : public vvTUIElement
{

    Q_OBJECT

    Q_PROPERTY(bool state READ getState WRITE setState)

private:
public:
    vvTUIToggleBitmapButton(const std::string &, const std::string &, int pID = 1, bool state = false);
#ifdef USE_QT
    vvTUIToggleBitmapButton(QObject *parent, const std::string &, const std::string &, int pID = 1, bool state = false);
#endif
    virtual ~vvTUIToggleBitmapButton();
    virtual void resend(bool create) override;
    virtual void parseMessage(covise::TokenBuffer &tb) override;

public slots:
    virtual void setState(bool s);
    virtual bool getState() const;

#ifdef USE_QT
signals:
    void tabletEvent();
    void tabletPressEvent();
    void tabletReleaseEvent();
#endif

protected:
    bool state;
    std::string bmpUp;
    std::string bmpDown;
};
/**
 * a messageBox.
 */
class VVCORE_EXPORT vvTUIMessageBox : public vvTUIElement
{

    Q_OBJECT

private:
public:
    vvTUIMessageBox(const std::string &, int pID = 1);
#ifdef USE_QT
    vvTUIMessageBox(QObject *parent, const std::string &, int pID = 1);
#endif
    virtual ~vvTUIMessageBox();

protected:
};
/**
 * a ProgressBar.
 */
class VVCORE_EXPORT vvTUIProgressBar : public vvTUIElement
{

    Q_OBJECT

    Q_PROPERTY(int value READ getValue WRITE setValue)
    Q_PROPERTY(int max READ getMax WRITE setMax)

private:
public:
    vvTUIProgressBar(const std::string &, int pID = 1);
#ifdef USE_QT
    vvTUIProgressBar(QObject *parent, const std::string &, int pID = 1);
#endif
    virtual ~vvTUIProgressBar();
    virtual void resend(bool create) override;

public slots:
    virtual void setValue(int newV);
    virtual void setMax(int maxV);
    virtual int getValue() const
    {
        return this->actValue;
    }
    virtual int getMax() const
    {
        return this->maxValue;
    }

protected:
    int actValue;
    int maxValue;
};
/**
 * a slider.
 */
class VVCORE_EXPORT vvTUIFloatSlider : public vvTUIElement
{

    Q_OBJECT

    Q_PROPERTY(float value READ getValue WRITE setValue)
    Q_PROPERTY(int ticks READ getTicks WRITE setTicks)
    Q_PROPERTY(bool horizontal READ getOrientation WRITE setOrientation)
    Q_PROPERTY(float min READ getMin WRITE setMin)
    Q_PROPERTY(float max READ getMax WRITE setMax)

private:
public:
    enum Orientation
    {
        HORIZONTAL = 1,
        VERTICAL = 0
    };

    vvTUIFloatSlider(const std::string &, int pID = 1, bool state = true);
    vvTUIFloatSlider(vvTabletUI *tui, const std::string &, int pID = 1, bool state = true);
#ifdef USE_QT
    vvTUIFloatSlider(QObject *parent, const std::string &, int pID = 1, bool state = true);
#endif
    virtual ~vvTUIFloatSlider();
    virtual void resend(bool create) override;
    virtual void parseMessage(covise::TokenBuffer &tb) override;

public slots:
    virtual void setValue(float newV);
    virtual void setTicks(int t);
    virtual void setOrientation(bool);
    virtual void setMin(float minV);
    virtual void setMax(float maxV);
    virtual void setRange(float minV, float maxV);
    virtual void setLogarithmic(bool val);

    virtual float getValue() const
    {
        return this->actValue;
    }
    virtual int getTicks() const
    {
        return this->ticks;
    }
    virtual bool getOrientation() const
    {
        return this->orientation;
    }
    virtual float getMin() const
    {
        return this->minValue;
    }
    virtual float getMax() const
    {
        return this->maxValue;
    }
    virtual bool getLogarithmic() const
    {
        return this->logarithmic;
    }
#ifdef USE_QT
signals:
    void tabletEvent();
    void tabletPressEvent();
    void tabletReleaseEvent();
#endif

protected:
    float actValue;
    float minValue;
    float maxValue;
    int ticks;
    bool orientation;
    bool logarithmic = false;
};
/**
 * a slider.
 */
class VVCORE_EXPORT vvTUISlider : public vvTUIElement
{
    Q_OBJECT

    Q_PROPERTY(int value READ getValue WRITE setValue)
    Q_PROPERTY(int ticks READ getTicks WRITE setTicks)
    Q_PROPERTY(bool horizontal READ getOrientation WRITE setOrientation)
    Q_PROPERTY(int min READ getMin WRITE setMin)
    Q_PROPERTY(int max READ getMax WRITE setMax)

private:
public:
    enum Orientation
    {
        HORIZONTAL = 1,
        VERTICAL = 0
    };

    vvTUISlider(const std::string &, int pID = 1, bool state = true);
    vvTUISlider(vvTabletUI *tui, const std::string &, int pID = 1, bool state = true);
#ifdef USE_QT
    vvTUISlider(QObject *parent, const std::string &, int pID = 1, bool state = true);
#endif
    virtual ~vvTUISlider();
    virtual void resend(bool create) override;
    virtual void parseMessage(covise::TokenBuffer &tb) override;

public slots:
    virtual void setValue(int newV);
    virtual void setOrientation(bool o);
    virtual void setTicks(int t);
    virtual void setMin(int minV);
    virtual void setMax(int maxV);
    virtual void setRange(int minV, int maxV);

    virtual int getValue() const
    {
        return this->actValue;
    }
    virtual int getTicks() const
    {
        return this->ticks;
    }
    virtual bool getOrientation() const
    {
        return this->orientation;
    }
    virtual int getMin() const
    {
        return this->minValue;
    }
    virtual int getMax() const
    {
        return this->maxValue;
    }

#ifdef USE_QT
signals:
    void tabletEvent();
    void tabletPressEvent();
    void tabletReleaseEvent();
#endif

protected:
    int actValue;
    int minValue;
    int maxValue;
    int ticks;
    bool orientation;
};
/**
 * a spinEditField.
 */
class VVCORE_EXPORT vvTUISpinEditfield : public vvTUIElement
{

    Q_OBJECT

    Q_PROPERTY(int value READ getValue WRITE setPosition)
    Q_PROPERTY(int min READ getMin WRITE setMin)
    Q_PROPERTY(int max READ getMax WRITE setMax)
    Q_PROPERTY(int step READ getStep WRITE setStep)

private:
public:
    vvTUISpinEditfield(const std::string &, int pID = 1);
#ifdef USE_QT
    vvTUISpinEditfield(QObject *parent, const std::string &, int pID = 1);
#endif
    virtual ~vvTUISpinEditfield();
    virtual void resend(bool create) override;
    virtual void parseMessage(covise::TokenBuffer &tb) override;

public slots:
    virtual void setPosition(int newV);
    virtual void setMin(int minV);
    virtual void setMax(int maxV);
    virtual void setStep(int s);

    virtual int getValue() const
    {
        return this->actValue;
    }
    virtual int getStep() const
    {
        return this->step;
    }
    virtual int getMin() const
    {
        return this->minValue;
    }
    virtual int getMax() const
    {
        return this->maxValue;
    }

#ifdef USE_QT
signals:
    void tabletEvent();
#endif

protected:
    int actValue;
    int minValue;
    int maxValue;
    int step;
};
/**
 * a spinEditField with text.
 */
class VVCORE_EXPORT vvTUITextSpinEditField : public vvTUIElement
{

    Q_OBJECT

    Q_PROPERTY(std::string text READ getText WRITE setText)
    Q_PROPERTY(int min READ getMin WRITE setMin)
    Q_PROPERTY(int max READ getMax WRITE setMax)
    Q_PROPERTY(int step READ getStep WRITE setStep)

private:
public:
    vvTUITextSpinEditField(const std::string &, int pID = 1);
#ifdef USE_QT
    vvTUITextSpinEditField(QObject *parent, const std::string &, int pID = 1);
#endif
    virtual ~vvTUITextSpinEditField();
    virtual void resend(bool create) override;
    virtual void parseMessage(covise::TokenBuffer &tb) override;

public slots:
    virtual void setMin(int minV);
    virtual void setMax(int maxV);
    virtual void setStep(int s);
    virtual void setText(const std::string &text);
    virtual const std::string &getText() const
    {
        return this->text;
    }
    virtual int getStep() const
    {
        return this->step;
    }
    virtual int getMin() const
    {
        return this->minValue;
    }
    virtual int getMax() const
    {
        return this->maxValue;
    }

#ifdef USE_QT
signals:
    void tabletEvent();
#endif

protected:
    std::string text;
    int minValue;
    int maxValue;
    int step;
};
/**
 * a editField. (LineEdit)
 */
class VVCORE_EXPORT vvTUIEditField : public vvTUIElement
{

    Q_OBJECT

    Q_PROPERTY(std::string text READ getText WRITE setText)
    Q_PROPERTY(bool immediate READ isImmediate WRITE setImmediate)

private:
public:
    vvTUIEditField(const std::string &, int pID = 1, const std::string &def = "");
    vvTUIEditField(vvTabletUI *tui, const std::string &, int pID = 1);
#ifdef USE_QT
    vvTUIEditField(QObject *parent, const std::string &, int pID = 1);
#endif
    virtual ~vvTUIEditField();
    virtual void resend(bool create) override;
    virtual void parseMessage(covise::TokenBuffer &tb) override;

public slots:
    virtual void setText(const std::string &t);
    virtual void setImmediate(bool);
    virtual void setPasswordMode(bool b);
    virtual void setIPAddressMode(bool b);

    virtual const std::string &getText() const;
    virtual bool isImmediate() const
    {
        return this->immediate;
    }

#ifdef USE_QT
signals:
    void tabletEvent();
#endif

protected:
    std::string text;
    bool immediate;
};
/**
 * another editField (TextEdit)
 */
class VVCORE_EXPORT vvTUIEditTextField : public vvTUIElement
{
    Q_OBJECT

    Q_PROPERTY(std::string text READ getText WRITE setText)
    Q_PROPERTY(bool immediate READ isImmediate WRITE setImmediate)

private:
public:
    vvTUIEditTextField(const std::string &, int pID = 1, const std::string &def = "");
    vvTUIEditTextField(vvTabletUI *tui, const std::string &, int pID = 1);
#ifdef USE_QT
    vvTUIEditTextField(QObject *parent, const std::string &, int pID = 1);
#endif
    virtual ~vvTUIEditTextField();
    virtual void resend(bool create) override;
    virtual void parseMessage(covise::TokenBuffer &tb) override;

public slots:
    virtual void setText(const std::string &t);
    virtual void setImmediate(bool);
    virtual const std::string &getText() const;
    virtual bool isImmediate() const
    {
        return this->immediate;
    }

#ifdef USE_QT
signals:
    void tabletEvent();
#endif

protected:
    std::string text;
    bool immediate;
};
/**
 * a editIntField = EditField fuer Integer
 */
class VVCORE_EXPORT vvTUIEditIntField : public vvTUIElement
{
    Q_OBJECT

    Q_PROPERTY(bool immediate READ isImmediate WRITE setImmediate)
    Q_PROPERTY(int min READ getMin WRITE setMin)
    Q_PROPERTY(int max READ getMax WRITE setMax)
    Q_PROPERTY(int value READ getValue WRITE setValue)

private:
public:
    vvTUIEditIntField(const std::string &, int pID = 1, int def = 0);
    vvTUIEditIntField(vvTabletUI *tui, const std::string &, int pID = 1, int def = 0);
#ifdef USE_QT
    vvTUIEditIntField(QObject *parent, const std::string &, int pID = 1, int def = 0);
#endif
    virtual ~vvTUIEditIntField();
    virtual void parseMessage(covise::TokenBuffer &tb) override;
    virtual void resend(bool create) override;
    virtual std::string getText() const;

public slots:
    virtual void setImmediate(bool);
    virtual void setValue(int val);
    virtual void setMin(int min);
    virtual void setMax(int max);
    virtual int getValue() const
    {
        return this->value;
    }
    virtual bool isImmediate() const
    {
        return this->immediate;
    }
    virtual int getMin() const
    {
        return this->min;
    }
    virtual int getMax() const
    {
        return this->max;
    }

#ifdef USE_QT
signals:
    void tabletEvent();
#endif

protected:
    int value;
    int min;
    int max;
    bool immediate;
};
/**
 * a editfloatfield = EditField fuer Kommazahlen
 */
class VVCORE_EXPORT vvTUIEditFloatField : public vvTUIElement
{
    Q_OBJECT

    Q_PROPERTY(bool immediate READ isImmediate WRITE setImmediate)
    Q_PROPERTY(float value READ getValue WRITE setValue)

private:
public:
    vvTUIEditFloatField(const std::string &, int pID = 1, float def = 0);
    vvTUIEditFloatField(vvTabletUI *tui, const std::string &, int pID = 1, float def = 0);
#ifdef USE_QT
    vvTUIEditFloatField(QObject *parent, const std::string &, int pID = 1, float def = 0);
#endif
    virtual ~vvTUIEditFloatField();
    virtual void resend(bool create) override;
    virtual void parseMessage(covise::TokenBuffer &tb) override;

public slots:
    virtual void setImmediate(bool);
    virtual void setValue(float val);
    virtual float getValue() const
    {
        return this->value;
    }
    virtual bool isImmediate() const
    {
        return this->immediate;
    }
#ifdef USE_QT
signals:
    void tabletEvent();
#endif

protected:
    float value;
    bool immediate;
};
/**
 * a comboBox.
 */
class VVCORE_EXPORT vvTUIComboBox : public vvTUIElement
{

    Q_OBJECT
    Q_PROPERTY(int selected READ getSelectedEntry WRITE setSelectedEntry)
    Q_PROPERTY(std::string selectedText READ getSelectedText WRITE setSelectedText)

private:
public:
    vvTUIComboBox(const std::string &, int pID = 1);
    vvTUIComboBox(vvTabletUI *tui, const std::string &, int pID = 1);
#ifdef USE_QT
    vvTUIComboBox(QObject *parent, const std::string &, int pID = 1);
#endif
    virtual ~vvTUIComboBox();
    virtual void resend(bool create) override;
    virtual void parseMessage(covise::TokenBuffer &tb) override;

public slots:
    virtual void addEntry(const std::string &t);
    virtual void delEntry(const std::string &t);
    virtual void clear();
    virtual int getSelectedEntry() const;
    virtual void setSelectedEntry(int e);
    virtual void setSelectedText(const std::string &t);
    virtual const std::string &getSelectedText() const;
    virtual int getNumEntries();

#ifdef USE_QT
signals:
    void tabletEvent();
#endif

protected:
    std::string text;
    int selection;
	std::list<std::string> elements;
};
/**
 * a listBox.
 */
class VVCORE_EXPORT vvTUIListBox : public vvTUIElement
{
    Q_OBJECT
    Q_PROPERTY(int selected READ getSelectedEntry WRITE setSelectedEntry)
    Q_PROPERTY(std::string selectedText READ getSelectedText WRITE setSelectedText)

private:
public:
    vvTUIListBox(const std::string &, int pID = 1);
#ifdef USE_QT
    vvTUIListBox(QObject *parent, const std::string &, int pID = 1);
#endif
    virtual ~vvTUIListBox();
    virtual void resend(bool create) override;
    virtual void parseMessage(covise::TokenBuffer &tb) override;

public slots:
    virtual void addEntry(const std::string &t);
    virtual void delEntry(const std::string &t);
    virtual int getSelectedEntry() const;
    virtual void setSelectedEntry(int e);
    virtual void setSelectedText(const std::string &t);
    virtual const std::string &getSelectedText() const;

#ifdef USE_QT
signals:
    void tabletEvent();
#endif

protected:
    std::string text;
    int selection;
	std::list<std::string> elements;
};
class VVCORE_EXPORT MapData
{
public:
    MapData(const char *name, float ox, float oy, float xSize, float ySize, float height);
    virtual ~MapData();
    char *name;
    float ox, oy, xSize, ySize, height;
};
/**
 * a Map Widget
 */
class VVCORE_EXPORT vvTUIMap : public vvTUIElement
{
private:
public:
    vvTUIMap(const char *, int pID = 1);
    virtual ~vvTUIMap();
    virtual void addMap(const char *name, float ox, float oy, float xSize, float ySize, float height);
    virtual void resend(bool create) override;
    virtual void parseMessage(covise::TokenBuffer &tb) override;

    float angle;
    float xPos;
    float yPos;
    float height;
    int mapNum;

protected:
	std::list<MapData *> maps;
};
/**
* an earth Map Widget
*/
class VVCORE_EXPORT vvTUIEarthMap : public vvTUIElement
{
private:
public:
    vvTUIEarthMap(const char *, int pID = 1);
    virtual ~vvTUIEarthMap();
    virtual void setPosition(float latitude, float longitude, float altitude);
    virtual void resend(bool create) override;
    virtual void parseMessage(covise::TokenBuffer &tb) override;
    

    float latitude;
    float longitude;
    float altitude;
    float minHeight;
    float maxHeight;

    void addPathNode(float latitude, float longitude);
    std::list<std::pair<float, float>> path;
    void updatePath();
    void setMinMax(float minH, float maxH);


protected:
};
/**
 * PopUp Window with text
 */
class VVCORE_EXPORT vvTUIPopUp : public vvTUIElement
{

    Q_OBJECT
    Q_PROPERTY(std::string text READ getText WRITE setText)
    Q_PROPERTY(bool immediate READ isImmediate WRITE setImmediate)

private:
public:
    vvTUIPopUp(const std::string &, int pID = 1);
#ifdef USE_QT
    vvTUIPopUp(QObject *parent, const std::string &, int pID = 1);
#endif
    virtual ~vvTUIPopUp();
    virtual void resend(bool create) override;
    virtual void parseMessage(covise::TokenBuffer &tb) override;

#ifdef USE_QT
signals:
    void tabletEvent();
#endif

public slots:
    virtual void setText(const std::string &t);
    virtual const std::string &getText() const
    {
        return this->text;
    }
    virtual void setImmediate(bool);
    virtual bool isImmediate() const
    {
        return this->immediate;
    }

protected:
    std::string text;
    bool immediate;
};
/**
 * a Webview widget
 */
class VVCORE_EXPORT vvTUIWebview : public vvTUIElement
{
    Q_OBJECT;
public:
	vvTUIWebview(const std::string&, int pID = 1);
    vvTUIWebview(vvTabletUI* tui, const std::string&, int pID = 1);
#ifdef USE_QT
    vvTUIWebview(QObject* parent, const std::string&, int pID = 1);
#endif
    virtual ~vvTUIWebview();
    virtual void parseMessage(covise::TokenBuffer &tb) override;
    void setURL(const std::string& url);
    void doSomething();

#ifdef USE_QT
signals:
    void tabletEvent();
#endif
};
}
