/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_NODE_H
#define ME_NODE_H

#include <QGraphicsItem>
#include <QGraphicsSvgItem>
#include <QObject>
#include <QList>
#include <QVector>

#include "hosts/MEHost.h"

class QPushButton;
class QResizeEvent;
class QMouseEvent;
class QMenu;
class QString;
class QDrag;
class QProgressBar;
class QGraphicsSceneMouseEvent;
class QGraphicsSimpleTextItem;
class QGraphicsPixmapItem;

class MEGraphicsView;
class MEModuleParameter;
class MEControlParameter;
class MEModulePanel;
class MEHost;
class MEParameterPort;
class MEPort;
class MEDataPort;
class MEFileBrowserPort;
class MEDataTreeItem;
class MEUserInterface;
class MEControlPanel;
class MENodeMulti;
class MENodeBook;
class MENodeText;

namespace covise
{
class coRecvBuffer;
}

//====================================================
class MENode : public QObject, public QGraphicsItem
//====================================================
{
    Q_OBJECT
#if QT_VERSION >= 0x040600
    Q_INTERFACES(QGraphicsItem)
#endif

    friend class MENodeText;

public:
    MENode(MEGraphicsView *graphWidget = 0);
    ~MENode();

    QVector<MEDataPort *> dinlist, doutlist, chanlist;
    QVector<MEParameterPort *> pinlist, poutlist;
    QVector<MENode *> m_syncList;
    MEFileBrowserPort *defaultFileBrowserPort;

    enum
    {
        Type = UserType + 1
    };
    int type() const
    {
        return Type;
    }

    bool isLocalNode()
    {
        return isLocal;
    };
    int getIndex(MEParameterPort *);
    int getHostID()
    {
        return host->getID();
    };
    int getNodeID()
    {
        return nodeid;
    };
    static qreal getDistance()
    {
        return distance;
    };
    bool isShown();
    bool isBookOpen()
    {
        return !bookclosed;
    };
    bool isCloned()
    {
        return isSynced;
    };
    bool isShownDetails()
    {
        return showDetails;
    };
    bool hasBeenModified()
    {
        return isModified;
    };
    bool findSibling(MENode *);
    bool link(MENode *);
    void layoutItem();
    void createControlPanelInfo();
    void restoreParameter();
    void storeParameter();
    void setShowDetails(bool state)
    {
        showDetails = state;
    };
    void setModified(bool);
    void setActive(bool);
    void setDead(bool);
    bool isDead() const;
    void setLabel(const QString &text);
    void setDesc(const QString &text);
    void printSibling();
    void showProgress(int);
    void moveNode(int x, int y);
    void addPort(MEPort *);
    void removePort(MEPort *);
    void showProgressBar(int);
    void expand(int);
    void unlink();
    void createPorts(QStringList);
    void sendExec();
    void sendMessage(const QString &message);
    void sendNodeMessage(const QString &message);
    void sendRenameMessage(const QString &message);
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    void syncNode(const QString &modulename, const QString &num, const QString &nodename,
                  int x, int y, MENode *original);
    void init(const QString &modulename, const QString &num, const QString &nodename,
              int x, int y);

    QString getName()
    {
        return moduleName;
    };
    QString getTitle()
    {
        return moduleTitle;
    };
    QString getHostname()
    {
        return host->getIPAddress();
    };
    QString getCategory()
    {
        return category;
    };
    QString getNumber()
    {
        return number;
    };
    QString getDescription()
    {
        return description;
    };
    QString getNodeTitle();
    QColor getColor()
    {
        return color;
    };
    QProgressBar *getProgressBar()
    {
        return pb;
    };
    QRectF boundingRect() const;

    MEPort *getPort(const QString &name);
    MEDataPort *getDataPort(const QString &name);
    MEParameterPort *getParameterPort(const QString &name);
    MEParameterPort *getPort(int index);
    MEHost *getHost()
    {
        return host;
    }
    MENode *getClone()
    {
        return clonedFromNode;
    };
    MEDataTreeItem *getTreeItem()
    {
        return m_dataTreeItem;
    };
    MEModuleParameter *getModuleInfo()
    {
        return m_moduleInfo;
    };
    MEControlParameter *getControlInfo()
    {
        return m_controlInfo;
    };
    MEGraphicsView *getGraph()
    {
        return graph;
    };
    int getX() const;
    int getY() const;

    void init(int nodeid, MEHost *, covise::coRecvBuffer &);
    void createParamPort(const QString &pname, const QString &desc, int porttype, covise::coRecvBuffer &tb);

public slots:

    void executeCB();
    void bookClick();
    void multiClick();
    void helpCB();

protected:
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
    void baseMouseMoveEvent(QGraphicsSceneMouseEvent *event);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
    void mousePressEvent(QGraphicsSceneMouseEvent *event);
    void contextMenuEvent(QGraphicsSceneContextMenuEvent *e);
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *);
    void dragEnterEvent(QGraphicsSceneDragDropEvent *event);
    void dropEvent(QGraphicsSceneDragDropEvent *event);

private:
    MEGraphicsView *graph;

    int xx, yy, nodeid, instance;
    static qreal distance;
    bool isActive, isSynced, showDetails, isModified, isDoubleClicked, isLocal, isMoved;
    bool bookclosed, shown, dead;
    bool multihost, collapsed, showing;

    MENode *clonedFromNode, *leftnode, *rightnode;
    MEHost *host;
    MENodeBook *openBookItem, *closedBookItem;
    MENodeText *textItem;
    MENodeMulti *multiItem;

    MEDataTreeItem *m_dataTreeItem;
    MEModuleParameter *m_moduleInfo;
    MEControlParameter *m_controlInfo;

    QColor color, color_dark, color_active;
    QPen *normalPen;
    QString category, description, moduleTitle, moduleName, number;

    QProgressBar *pb;

    void createCopyList();
    void managemultiItem();
    void addBrowserFilter(const QString &value);

    MENode *getLeftmost();
    MENode *getRightmost();
    MEParameterPort *createParameterPort(const QString &pname, const QString &paramtype, const QString &desc,
                                         QString value, int apptype);

signals:

    void bookIconChanged();
    void bookClose();
    void execute();
};

//==========================================================
class MENodeBook : public QGraphicsSvgItem
//==========================================================
{

public:
    MENodeBook(MENode *node, QGraphicsScene *scene, const QString &name);
    ~MENodeBook();

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *e);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *e);

private:
    MENode *node;
};

//===========================================================
class MENodeMulti : public QGraphicsPixmapItem
//===========================================================
{

public:
    MENodeMulti(MENode *node, QGraphicsScene *scene);
    ~MENodeMulti();

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *e);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *e);

private:
    MENode *node;
};

//================================================
class MENodeText : public QGraphicsSimpleTextItem
//================================================
{

public:
    MENodeText(MENode *node, QGraphicsScene *scene);
    ~MENodeText();

private:
    MENode *node;

protected:
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *e);
    void mousePressEvent(QGraphicsSceneMouseEvent *e);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *e);
    void mouseMoveEvent(QGraphicsSceneMouseEvent *e);
    void contextMenuEvent(QGraphicsSceneContextMenuEvent *e);
};
#endif
