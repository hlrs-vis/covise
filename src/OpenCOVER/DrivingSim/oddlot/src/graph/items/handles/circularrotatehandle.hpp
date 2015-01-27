/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   09.07.2010
**
**************************************************************************/

#ifndef CIRCULARROTATEHANDLE_HPP
#define CIRCULARROTATEHANDLE_HPP

#include "handle.hpp"

class CircularRotateHandle : public Handle
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit CircularRotateHandle(QGraphicsItem *parent);
    virtual ~CircularRotateHandle();

    void setMousePos(const QPointF &mousePoint);

    QPointF getPos() const
    {
        return pos();
    }
    double getAngle() const
    {
        return angle_;
    }

protected:
private:
    CircularRotateHandle(); /* not allowed */
    CircularRotateHandle(const CircularRotateHandle &); /* not allowed */
    CircularRotateHandle &operator=(const CircularRotateHandle &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    double angle_;
};

#endif // CIRCULARROTATEHANDLE_HPP
