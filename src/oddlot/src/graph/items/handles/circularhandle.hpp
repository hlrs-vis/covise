/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/19/2010
**
**************************************************************************/

#ifndef CIRCULARHANDLE_HPP
#define CIRCULARHANDLE_HPP

#include "handle.hpp"

class CircularHandle : public Handle
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit CircularHandle(QGraphicsItem *parent);
    virtual ~CircularHandle();

    QPointF getPos() const
    {
        return pos();
    }

    //	bool						getPassSelectionToParent() const { return passSelectionToParent_; }
    //	void						setPassSelectionToParent(bool passSelectionToParent);

protected:
private:
    CircularHandle(); /* not allowed */
    CircularHandle(const CircularHandle &); /* not allowed */
    CircularHandle &operator=(const CircularHandle &); /* not allowed */

    //################//
    // EVENTS         //
    //################//

protected:
    //	virtual QVariant		itemChange(GraphicsItemChange change, const QVariant & value);

    //################//
    // PROPERTIES     //
    //################//

private:
    //	bool						passSelectionToParent_;
};

#endif // CIRCULARHANDLE_HPP
