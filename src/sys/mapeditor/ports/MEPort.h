/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_PORT_H
#define ME_PORT_H

#include <QGraphicsRectItem>
#include <QObject>

#ifdef YAC
#include "yac/coQTSendBuffer.h"
#endif

class QMouseEvent;
class QEvent;
class QColor;
class QString;

class MENode;

//=================================================================
class MEPort : public QObject, public QGraphicsRectItem
//==================================================================
{

    Q_OBJECT

    friend class MEGraphicsView;

public:
    enum SelectionType
    {
        Deselect = 0,
        Clicked = 1,
        Target = 2,
        Highlight = 4,
    };

    static void init();

    MEPort(MENode *n, QGraphicsScene *scene);
    MEPort(MENode *n, QGraphicsScene *scene, const QString &pname, const QString &desc, int type);

    ~MEPort();

    enum porttypes
    {
        UNKNOWN = 0,
        DIN,
        MULTY_IN,
        DOUT,
        CHAN,
        PIN,
        POUT,
        PARAM
    };

    static const char *text;

    bool isConnectable();
    bool isSelected()
    {
        return selected;
    };
    bool isConnected();

    int getPortType()
    {
        return portType;
    }
    int getHostId()
    {
        return hostid;
    };
    int getNoOfLinks()
    {
        return links;
    };

    virtual void addLink(MEPort *port);
    virtual void delLink(MEPort *port);
    virtual void setShown(bool);
    void setSelected(SelectionType type, bool flag = true);
    void setPortColor(QColor col)
    {
        portcolor = col;
    }
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    bool isShown()
    {
        return shown;
    };

    MENode *getNode()
    {
        return node;
    };
    QString getName()
    {
        return portname;
    };
    QString getDescription()
    {
        return description;
    };
    QColor getPortColor()
    {
        return portcolor;
    }

protected:
    bool shown, hasObjData;
    int selected;
    int portType, xpos, ypos, hostid, links, ex, ey, distance;

    QString description, portname;
    QColor portcolor, portcolor_dark;
    MENode *node;

    virtual QColor definePortColor() = 0;

    void mousePressEvent(QGraphicsSceneMouseEvent *e);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *e);
    void mouseMoveEvent(QGraphicsSceneMouseEvent *e);
    void hoverEnterEvent(QGraphicsSceneHoverEvent *e);
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *e);
};
#endif
