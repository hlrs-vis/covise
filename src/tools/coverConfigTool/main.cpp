/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <qapplication.h>
#include "coverconfigtool.h"

void qInitImages_coverConfigTool();
void qCleanupImages_coverConfigTool();
int main(int argc, char **argv)
{
    qInitImages_coverConfigTool();
    QApplication a(argc, argv);
    coverConfigTool w;
    w.show();
    a.connect(&a, SIGNAL(lastWindowClosed()), &a, SLOT(quit()));
    qCleanupImages_coverConfigTool();
    return a.exec();
}
