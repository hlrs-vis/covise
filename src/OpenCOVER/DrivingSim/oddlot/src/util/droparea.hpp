/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/11/2010
**
**************************************************************************/

#ifndef DROPAREA_HPP
#define DROPAREA_HPP

#include <QLabel>

class QDragEnterEvent;
class QDragLeaveEvent;
class QDragMoveEvent;
class QDropEvent;



//##############################//
// DropArea for the recycle bin//
//
//#############################//


class DropArea : public QLabel
{
    Q_OBJECT

public:
    DropArea(QPixmap *pixmap);

	//################//
    // SLOTS          //
    //################//
protected:
    void dragEnterEvent(QDragEnterEvent *event);
    void dragMoveEvent(QDragMoveEvent *event);
    void dragLeaveEvent(QDragLeaveEvent *event);
    void dropEvent(QDropEvent *event);


private:

};

#endif // DROPAREA_HPP
