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
//#include "DistortConfig.h"
#include "MainWindow.h"

//QT
#include <QApplication>

// Konstruktor
// wird aufgerufen wenn das Plugin gestartet wird
/*DistortConfig::DistortConfig()
{
	fprintf(stderr,"DistortConfig::DistortConfig\n");
	

}

// Destruktor
// wird aufgerufen wenn das Plugin zur Laufzeit beendet wird
DistortConfig::~DistortConfig()
{
	fprintf(stderr,"DistortConfig::~DistortConfig\n");
}

// Initialisierung
// wird nach dem Konstruktor aufgerufen
bool DistortConfig::init()
{
	fprintf(stderr,"DistortConfig::init\n");

	int argc = 0;
	char* argv[1];
	argv[0]="";
	QApplication app(argc, argv);
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


COVERPLUGIN(DistortConfig)*/

int main()
{
    fprintf(stderr, "DistortConfig::init\n");

    int argc = 0;
    char *argv[1];
    argv[0] = "";
    QApplication app(argc, argv);
    MainWindow w;
    w.show();

    return app.exec();
}
