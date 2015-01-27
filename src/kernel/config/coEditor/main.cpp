/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include <vector>
#include <QString>

#include <QApplication>
#include "coEditorMainWindow.h"

void usage()
{
    std::cout << " Editor [option] filename   , possible option: -s do schema \n";
}

int main(int argC, char *argV[])
{
    // Check for options
    int parmInd;
    for (parmInd = 1; parmInd < argC; parmInd++)
    {
        // Break out on first parm not starting with a dash
        if (argV[parmInd][0] != '-')
            break;
        else if (!strcmp(argV[parmInd], "-s")
                 || !strcmp(argV[parmInd], "-S"))
        {
            // sag ja zu schema
        }
        else
        {
            std::cout << "Unknown option " << argV[parmInd]
                      << "', ignoring it.\n";
        }
    }
    //     if (parmInd + 1 != argC)
    //     {
    //        usage();
    //        return 1;
    //     }
    QString filename = argV[parmInd];
    Q_INIT_RESOURCE(application);
    QApplication app(argC, argV);
    coEditorMainWindow mainWin(filename);

    mainWin.show();
    return app.exec();

    return (0);
}
