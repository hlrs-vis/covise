/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/24/2010
**
**************************************************************************/

#ifndef EDITABLEHANDLE_HPP
#define EDITABLEHANDLE_HPP

#include "baselanemovehandle.hpp"

// Qt //
//
#include <QDoubleSpinBox>
#include <QGraphicsProxyWidget>

class ProjectEditor;

class EditableHandle : public QGraphicsProxyWidget, public Observer
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit EditableHandle(double value, BaseLaneMoveHandle *parent, bool flip = false);
    virtual ~EditableHandle();

    double getValue() const;
    void setValue(double value);


protected:

private:
	EditableHandle(); /* not allowed */
	EditableHandle(const EditableHandle &); /* not allowed */
	EditableHandle &operator=(const EditableHandle &); /* not allowed */

//################//
// SIGNALS        //
//################//

signals:
    void requestPositionChange(const QPointF &pos);
    void selectionChanged(bool selected);


    //################//
    // PROPERTIES     //
    //################//

private:
    QDoubleSpinBox *editableItem_;
};

#endif // EDITABLEHANDLE_HPP
