/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MEAPPLICATION_H

#define MEAPPLICATION_H

#include <QApplication>

class MEApplication : public QApplication
{
    Q_OBJECT
public:
    MEApplication(int &argc, char *argv[]);

protected:
    bool event(QEvent *);
};
#endif /* end of include guard: MEAPPLICATION_H */
