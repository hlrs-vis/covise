/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Template Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

//Local
#include "DistortConfig.h"
#include "MainWindow.h"

//QT
#include <QApplication>

// Konstruktor
// wird aufgerufen wenn das Plugin gestartet wird
DistortConfig::DistortConfig()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    std::cerr << "DistortConfig::DistortConfig\n" << std::endl;
}

// Destruktor
// wird aufgerufen wenn das Plugin zur Laufzeit beendet wird
DistortConfig::~DistortConfig()
{
    std::cerr << "DistortConfig::~DistortConfig\n" << std::endl;
}

// Initialisierung
// wird nach dem Konstruktor aufgerufen
bool DistortConfig::init()
{
    std::cerr << "DistortConfig::init\n" << std::endl;

    int argc = 0;
    char *argv[1];
    argv[0] = (char *)"";
    QApplication app(argc, argv);
    app.setAttribute(Qt::AA_MacDontSwapCtrlAndMeta);
    MainWindow w;
    w.show();

    return app.exec();
}

// PreFrame function
// wird jedes mal aufgerufen bevor ein neuer Frame gerendert wird
void DistortConfig::preFrame()
{
}

bool DistortConfig::load()
{
    return true;
}

COVERPLUGIN(DistortConfig)
