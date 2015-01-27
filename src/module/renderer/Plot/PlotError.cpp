/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Log: PlotError.C,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

//static char rcsid[] = "$Id: PlotError.C,v 1.1 1994/04/12 13:39:31 zrfu0125 Exp zrfu0125 $";

//**************************************************************************
//
// * Description    : This is the error output handler
//
//
// * Class(es)      : none
//
//
// * inherited from : none
//
//
// * Author  : Uwe Woessner
//
//
// * History : 11.11.94
//
//
//
//**************************************************************************

//
//
//
#include <covise/covise.h>
#include "PlotError.h"

#include <fcntl.h>
#include <sys/types.h>

//
// nothing to do
