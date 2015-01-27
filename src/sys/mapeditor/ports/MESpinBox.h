/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_SPINBOX_H
#define ME_SPINBOX_H

#include <QSpinBox>

class QLineEdit;
class MEParameterPort;

//================================================
class MESpinBox : public QSpinBox
//================================================
{

    Q_OBJECT

public:
    MESpinBox(MEParameterPort *p, QWidget *parent = 0);

    QString mapValueToText(int val);
    int mapTextToValue(bool *ok);

private:
    MEParameterPort *port;
};
#endif
