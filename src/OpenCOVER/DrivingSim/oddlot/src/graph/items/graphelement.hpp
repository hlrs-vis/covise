/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   23.03.2010
**
**************************************************************************/

#ifndef GRAPHELEMENT_HPP
#define GRAPHELEMENT_HPP

#include <QObject>
#include <QGraphicsPathItem>
#include "src/data/observer.hpp"

// Qt //
//
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneHoverEvent>
#include <QGraphicsSceneContextMenuEvent>

#include <QMenu>
#include <QAction>

// Data //
//
#include "src/data/projectdata.hpp"

class DataElement;

// Graph //
//
#include "src/graph/projectgraph.hpp"
class TopviewGraph;
class ProfileGraph;

class GraphElement : public QObject, public QGraphicsPathItem, public Observer
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit GraphElement(GraphElement *parentGraphElement, DataElement *dataElement);
    virtual ~GraphElement();

    // Garbage //
    //
    virtual bool deleteRequest() = 0;
    void registerForDeletion();
    virtual void notifyDeletion(); // to be implemented by subclasses
    bool isInGarbage() const
    {
        return isInGarbage_;
    }

    // Selection //
    //
    void setSelectable();

    // Data //
    //
    ProjectData *getProjectData() const;

    // Graph //
    //
    ProjectGraph *getProjectGraph() const;
    virtual TopviewGraph *getTopviewGraph() const;
    virtual ProfileGraph *getProfileGraph() const;

    // DataElement //
    //
    DataElement *getDataElement() const
    {
        return dataElement_;
    }

    // ParentGraphElement //
    //
    GraphElement *getParentGraphElement() const
    {
        return parentGraphElement_;
    }
    //	void						setParentGraphElement(GraphElement * parentGraphElement);

    // Graphics //
    //
    void enableHighlighting(bool enable);
    void setHighlighting(bool highlight);
    void setOpacitySettings(double highlightOpacity, double normalOpacity);
    void updateHighlightingState();

    virtual void createPath()
    { /* does nothing */
    }

    void setHovered(bool hovered);
    bool isHovered()
    {
        return hovered_;
    }

    // Observer Pattern //
    //
    virtual void updateObserver();

protected:
    // ContextMenu //
    //
    QMenu *getContextMenu()
    {
        return contextMenu_;
    }
    QMenu *getHideMenu() const
    {
        return hideMenu_;
    }
    QMenu *getRemoveMenu() const
    {
        return removeMenu_;
    }

private:
    GraphElement(); /* not allowed */
    GraphElement(const GraphElement &); /* not allowed */
    GraphElement &operator=(const GraphElement &); /* not allowed */

    void init();

    //################//
    // SLOTS          //
    //################//

public slots:
    void hideGraphElement();
    virtual void hideRoads();

    //################//
    // EVENTS         //
    //################//

protected:
    virtual QVariant itemChange(GraphicsItemChange change, const QVariant &value);

    virtual void contextMenuEvent(QGraphicsSceneContextMenuEvent *event);

    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);

    virtual void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);
    //	virtual void			hoverMoveEvent(QGraphicsSceneHoverEvent * event);

    //################//
    // PROPERTIES     //
    //################//

private:
    // ParentGraphElement //
    //
    GraphElement *parentGraphElement_;

    // DataElement //
    //
    DataElement *dataElement_;

    // Graphics //
    //
    bool hovered_;

    bool useHighlighting_;
    double highlightOpacity_;
    double normalOpacity_;

    // Garbage //
    //
    bool isInGarbage_;

    // ContextMenu //
    //
    QMenu *contextMenu_;
    QMenu *hideMenu_;
    QMenu *removeMenu_;
};

#endif // GRAPHELEMENT_HPP
