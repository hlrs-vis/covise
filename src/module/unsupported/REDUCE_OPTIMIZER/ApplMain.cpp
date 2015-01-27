/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:   COVISE CuttingPlane application module                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C) 1994                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  R.Lang, D.Rantzau                                             **
 **                                                                        **
 **                                                                        **
 ** Date:  18.05.94  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "OptimizerReducer.h"

#include <Cosmo3D/csFields.h>
#include <Cosmo3D/csGroup.h>

void main(int argc, char *argv[])
{

    Application *application = new Application(argc, argv);

    application->run();
}
