/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10.05.2010
**
**************************************************************************/

#ifndef MOUSEACTION_HPP
#define MOUSEACTION_HPP

class QGraphicsSceneMouseEvent;
class QGraphicsSceneDragDropEvent;

class MouseAction
{

    //################//
    // STATIC         //
    //################//

public:
    /*! \brief
	*/
    enum MouseActionType
    {
		// TopviewGraph //
		//
        ATM_MOVE,
        ATM_PRESS,
        ATM_RELEASE,
        ATM_DOUBLECLICK,
		ATM_DROP,

		// ProfileGraph //
		//
		PATM_PRESS,
		PATM_RELEASE,
		PATM_MOVE
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    MouseAction(MouseAction::MouseActionType mouseActionType, QGraphicsSceneMouseEvent *mouseEvent);
	MouseAction(MouseAction::MouseActionType mouseActionType, QGraphicsSceneDragDropEvent *dragEvent);
    virtual ~MouseAction()
    { /* does nothing */
    }

    MouseAction::MouseActionType getMouseActionType() const
    {
        return mouseActionType_;
    }
    QGraphicsSceneMouseEvent *getEvent() const
    {
        return mouseEvent_;
    }
	QGraphicsSceneDragDropEvent *getDragDropEvent() const
	{
		return dragEvent_;
	}

    // Interception //
    //
    void intercept()
    {
        intercepted_ = true;
    }
    bool isIntercepted() const
    {
        return intercepted_;
    }

protected:
private:
    MouseAction()
    { /* not allowed */
    }
    MouseAction(const MouseAction &)
    { /* not allowed */
    }

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    MouseActionType mouseActionType_;
    QGraphicsSceneMouseEvent *mouseEvent_;
	QGraphicsSceneDragDropEvent *dragEvent_;

    // Interception //
    //
    bool intercepted_;
};

#endif // MOUSEACTION_HPP
