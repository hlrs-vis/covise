/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_EXTENDEDPART_H
#define ME_EXTENDEDPART_H

#include <QFrame>

class MEParameterPort;
class QHBoxLayout;

//================================================
class MEExtendedPart : public QFrame
//================================================
{
    Q_OBJECT

public:
    MEExtendedPart(QWidget *parent, MEParameterPort *);
    ~MEExtendedPart();

    void addBrowser();
    void addColorMap();
    void removeBrowser();
    void removeColorMap();

private:
    MEParameterPort *port;
    QHBoxLayout *extendedLayout;
};
#endif
