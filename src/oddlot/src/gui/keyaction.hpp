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

#ifndef KEYACTION_HPP
#define KEYACTION_HPP

class QKeyEvent;

class KeyAction
{

    //################//
    // STATIC         //
    //################//

public:
    /*! \brief
	*/
    enum KeyActionType
    {
        ATK_PRESS,
        ATK_RELEASE
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    KeyAction(KeyAction::KeyActionType keyActionType, QKeyEvent *event);
    virtual ~KeyAction()
    { /* does nothing */
    }

    KeyAction::KeyActionType getKeyActionType() const
    {
        return keyActionType_;
    }
    QKeyEvent *getEvent() const
    {
        return event_;
    }

protected:
private:
    KeyAction()
    { /* not allowed */
    }
    KeyAction(const KeyAction &)
    { /* not allowed */
    }

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    KeyAction::KeyActionType keyActionType_;
    QKeyEvent *event_;
};

#endif // KEYACTION_HPP
