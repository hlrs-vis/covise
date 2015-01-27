/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "MainWindow.h"

int main(int argc, char **argv)
{

    QApplication app(argc, argv);
    MainWindow w;
    w.show();

    return app.exec();
}
