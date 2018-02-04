/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   04.02.2010
**
**************************************************************************/

#ifndef GRAPHSCENE_HPP
#define GRAPHSCENE_HPP

#include <QGraphicsScene>
#include "src/data/observer.hpp"

class MouseAction;
class KeyAction;

class QGraphicsSceneDragDropEvent;

class GraphScene : public QGraphicsScene, public Observer
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit GraphScene(const QRectF &sceneRect, QObject *parent = 0);
    virtual ~GraphScene();

    // Tools, Mouse & Key //
    //
    //	void						toolAction(ToolAction *);
    void mouseAction(MouseAction *);
    void keyAction(KeyAction *);

private:
    GraphScene(); /* not allowed */
    GraphScene(const GraphScene &); /* not allowed */
    GraphScene &operator=(const GraphScene &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

protected:
    // Mouse Events //
    //
    virtual void mousePressEvent(QGraphicsSceneMouseEvent *mouseEvent);
    virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *mouseEvent);
    virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *mouseEvent);
    virtual void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *mouseEvent);

	// Drag Events //
	//
	virtual void dropEvent(QGraphicsSceneDragDropEvent *event);

    // Key Events //
    //
    virtual void keyPressEvent(QKeyEvent *event);
    virtual void keyReleaseEvent(QKeyEvent *event);


//################//
// SIGNALS        //
//################//

signals:

    // Tools, Mouse & Key //
    //
    //	void						toolActionSignal(ToolAction *);
    void mouseActionSignal(MouseAction *);
    void keyActionSignal(KeyAction *);

    //################//
    // SLOTS          //
    //################//

public slots:

    //################//
    // PROPERTIES     //
    //################//

private:
};

#endif // GRAPHSCENE_HPP
