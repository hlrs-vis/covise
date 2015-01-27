/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_GRAPHICSVIEW_H
#define ME_GRAPHICSVIEW_H

#include <QGraphicsView>
#include <QMouseEvent>

#include <handler/MEPortSelectionHandler.h>

class QString;
class QFrame;
class QMenu;
class QAction;
class QDropEvent;
class QMouseEvent;
class QDrag;
class QLineEdit;
class QGraphicsLineItem;
class QAction;
class QMouseEvent;
class QGraphicsViewMouseEvent;
class QGraphicsViewContextMenuEvent;

class MEPort;
class MEDataPort;
class MEHost;
class MECategory;
class MENode;
class MENodeLink;
class MERenameDialog;

//================================================
class MEGraphicsView : public QGraphicsView
//================================================
{

    Q_OBJECT

public:
    MEGraphicsView(QWidget *parent = 0);
    ~MEGraphicsView();

    static MEGraphicsView *instance();

    enum increase
    {
        LEFT,
        RIGHT,
        TOP,
        BOTTOM
    };

    void wasMoved(QGraphicsItem *item, QGraphicsSceneMouseEvent *e);
    void init();
    void setSingleHostNodePopup();
    void setMultiHostNodePopup();
    void reset();
    void clearNodeSelections();
    void clearPortSelections(MEPortSelectionHandler::Type type);
    void updateSceneRect();
    void showMap();
    void disableParameterItem(bool);
    void disableNodePopupItems(bool);
    void setMasterState(bool);
    void scaleView(qreal scaleFactor);
    void hoverEnterPort(MEPort *);
    void hoverLeavePort(MEPort *);
    void portMoved(MEPort *);
    void portReleased(MEPort *, QGraphicsSceneMouseEvent *e);
    void portPressed(MEPort *, QGraphicsSceneMouseEvent *e);
    void showPossiblePorts(MEDataPort *, QGraphicsSceneContextMenuEvent *e);
    void showNodeActions(MENode *, QGraphicsSceneContextMenuEvent *e);
    void contextMenuEvent(QContextMenuEvent *e);
    void deletePossibleLink(MENodeLink *, QGraphicsSceneContextMenuEvent *e);
    void highlightMatchingPorts(MEPortSelectionHandler::Type type, MEPort *port);
    void sendLine(MEPort *port1, MEPort *port2);
    void renameNodes(const QString &newname);
    void addHost(MEHost *host);
    void removeHost(MEHost *host);
    QPointF getFreePos();
    float positionScaleX() const;
    float positionScaleY() const;
    void setAutoSelectNewNodes(bool select)
    {
        m_autoSelectNewNodes = select;
    }
    bool autoSelectNewNodes() const
    {
        return m_autoSelectNewNodes;
    }

private:
    bool m_portIsMoving, m_nodeIsMoving, m_replaceMode, m_dragMode;
    int m_scrollLines, m_currentMode;

    int sign(int v)
    {
        return v > 0 ? 1 : (v < 0 ? -1 : 0);
    }
    MENodeLink *m_clickedLine;
    MENode *m_popupNode;
    MEPort *m_clickedPort, *m_connectionSourcePort;
    MERenameDialog *m_renameBox;

    QAction *m_execAction, *m_deleteAction, *m_restartAction, *m_restartDebugAction, *m_restartMemcheckAction, *m_cloneAction, *m_replaceAction;
    QAction *m_renameAction, *m_paramAction, *m_helpAction, *m_copyToAction, *m_moveToAction;
    QAction *m_copyAction, *m_cutAction, *m_selectedHostAction, *m_selectedCategoryAction;
    QMenu *m_nodePopup, *m_linePopup, *m_hostPopup, *m_copyMovePopup, *m_viewportPopup;
    QAction *m_deleteLineAction;
    QLineEdit *m_renameLineEdit;
    QGraphicsLineItem *m_rubberLine;
    QPointF m_portPressedPosition, m_freePos;
    QPoint m_lastPos;
    QRectF m_currViewRect, m_oldViewRect;
    QList<QGraphicsItem *> movedItemList;
    qreal offxr, offxl, offyt, offyb;
    bool m_autoSelectNewNodes;

    void initPopupStuff();
    void makePortList(MEPort *port, QGraphicsSceneMouseEvent *e);
    void sendMoveMessage();
    void decodeMessage(QDropEvent *, QString message);
    void sendCopyList(MEHost *, int mode);
    void computeDepths(MENode *node, QMap<MENode *, int> *depthMap);
    void layoutHorizontal(const QMap<MENode *, int> depthMap, int maxDepth);
    void layoutVertical(QMap<MENode *, int> *depthMap, int maxDepth, int maxY);

signals:

    void usingNode(const QString message);
    void factorChanged(qreal factor);

private slots:

    void deleteLink();
    void renameNodesCB();
    void restartCB();
    void restartDebugCB();
    void restartMemcheckCB();
    void cloneCB();
    void helpCB();
    void executeCB();
    void paramCB();
    void copyModuleCB();
    void moveModuleCB();
    void selectionChangedCB();
    void valueChangedCB(int);

public slots:
    void replaceModulesCB();
    void deleteNodesCB();
    void hoveredHostCB();
    void hoveredCategoryCB();
    void triggeredHostCB();
    void copy();
    void cut();
    void paste();
    void viewAll(qreal scaleMax = -1.0);
    void scaleView50();
    void scaleView100();
    void layoutMap();
    void developerMode(bool);

protected:
    void mousePressEvent(QMouseEvent *e);
    void mouseReleaseEvent(QMouseEvent *e);
    void mouseMoveEvent(QMouseEvent *e);
    void wheelEvent(QWheelEvent *e);
    void dragEnterEvent(QDragEnterEvent *e);
    void dragLeaveEvent(QDragLeaveEvent *e);
    void dragMoveEvent(QDragMoveEvent *e);
    void dropEvent(QDropEvent *e);
};
#endif
