/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_MODULELABEL_H
#define ME_MODULELABEL_H

#include <QLabel>
#include <QVector>

class QMenu;
class QStringList;
class QMouseEvent;
class QAction;

class MEParameterPort;

//================================================
class MEParameterAppearance : public QLabel
//================================================
{
    Q_OBJECT

public:
    MEParameterAppearance(QWidget *parent, MEParameterPort *port = 0);

private:
    QStringList il_scalar, fl_scalar, ll_slider;
    QMenu *appPopup;
    QVector<QAction *> appList;

    MEParameterPort *port;
    void insertText(QStringList);

protected:
    void contextMenuEvent(QContextMenuEvent *e);

private slots:

    void appearanceCB();
};
#endif
