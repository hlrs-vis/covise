/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <qapplication.h>
#include <Inventor/Qt/SoQt.h>
#include "InvMain.h"
#ifndef WITHOUT_VIRVO
#include "SoVolume.h"
#include "SoVolumeDetail.h"
#endif
#include "SoBillboard.h"
#include "InvComposePlane.h"

int main(int argc, char **argv)
{
    // You should not create a QApplication instance if you want
    // to receive spaceball events.
    new QApplication(argc, argv);

    // Initialize Qt and SoQt.
    SoQt::init(argc, argv, argv[0]);

// Initialized Inventor extensions
#ifndef WITHOUT_VIRVO
    SoVolume::initClass();
    SoVolumeDetail::initClass();
#endif
    InvComposePlane::initClass();
    SoBillboard::initClass();

    // Set up a new main window.
    new InvMain(argc, argv);

    // Start event loop.
    SoQt::mainLoop();

    return 0;
}
