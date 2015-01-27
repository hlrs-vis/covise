/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_COLORSELECTOR_H
#define ME_COLORSELECTOR_H

#include <QPushButton>

class QColor;

//==========================================================
class MEColorSelector : public QPushButton
//==========================================================
{

    Q_OBJECT

public:
    MEColorSelector(QWidget *parent = 0);

    QColor grabColor(const QPoint &);

private:
    bool m_colorPicking;

signals:

    void pickedColor(const QColor &);

protected:
    //void              mouseMoveEvent       ( QMouseEvent *e );
    void mouseReleaseEvent(QMouseEvent *e);

public slots:

    void selectedColorCB();
};
#endif
