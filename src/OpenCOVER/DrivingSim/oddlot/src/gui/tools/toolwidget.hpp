/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   31.03.2010
**
**************************************************************************/

#ifndef TOOLWIDGET_HPP
#define TOOLWIDGET_HPP

#include <QWidget>

class ToolWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ToolWidget(QWidget *parent = 0);

    void setToolBoxIndex(int index);

signals:
    void activated();

public slots:
    void activateWidget(int);

private:
    int index_;
};

#endif // TOOLWIDGET_HPP
