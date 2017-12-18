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
    virtual ~MouseAction()
    { /* does nothing */
    }

    MouseAction::MouseActionType getMouseActionType() const
    {
        return mouseActionType_;
    }
    QGraphicsSceneMouseEvent *getEvent() const
    {
        return event_;
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
    QGraphicsSceneMouseEvent *event_;

    // Interception //
    //
    bool intercepted_;
};

#endif // MOUSEACTION_HPP
