/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   08.06.2010
**
**************************************************************************/

#ifndef PROFILEGRAPHSCENE_HPP
#define PROFILEGRAPHSCENE_HPP

#include <QGraphicsScene>

class MouseAction;

class ProfileGraphScene : public QGraphicsScene
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ProfileGraphScene(const QRectF &sceneRect, QObject *parent);
    //	virtual ~ProfileGraphScene(){ /* does nothing */ }

protected:
	//################//
	// SIGNALS        //
	//################//

	signals :
			void mouseActionSignal(MouseAction *);

private:
    ProfileGraphScene(); /* not allowed */
    ProfileGraphScene(const ProfileGraphScene &); /* not allowed */
    ProfileGraphScene &operator=(const ProfileGraphScene &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

public:
    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
	virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
	virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);

    void doDeselect(bool s)
    {
        doDeselect_ = s;
    };
    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    bool doDeselect_;
};

#endif // PROFILEGRAPHSCENE_HPP
