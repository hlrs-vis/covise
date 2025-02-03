/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
 *                                                                        *
 *   Copyright (C) 1996-1998, Silicon Graphics, Inc.                      *
 *                          All Rights Reserved.                          *
 *                                                                        *
 *  The files in this subtree  contain  UNPUBLISHED  PROPRIETARY  SOURCE  *
 *  CODE of Silicon Graphics, Inc.;  the  contents  of  these  files may  *
 *  not be disclosed to third  parties,  copied  or  duplicated  in  any  *
 *  form, in whole or in part,  without  the  prior  written  permission  *
 *  of  Silicon Graphics, Inc.                                            *
 *                                                                        *
 *  RESTRICTED RIGHTS LEGEND:                                             *
 *  Use,  duplication  or  disclosure  by  the  Government is subject to  *
 *  restrictions as set forth in subdivision (c)(1)(ii) of the Rights in  *
 *  Technical Data and Computer Software clause at  DFARS  252.227-7013,  *
 *  and/or in similar or successor  clauses in the FAR,  DOD or NASA FAR  *
 *  Supplement.  Unpublished - rights reserved  under the Copyright Laws  *
 *  of the United States.                                                 *
 *                                                                        *
 *  THIS SOFTWARE IS  PROVIDED "AS-IS" AND WITHOUT WARRANTY OF ANY KIND,  *
 *  EXPRESS,  IMPLIED OR  OTHERWISE,  INCLUDING  WITHOUT LIMITATION, ANY  *
 *  WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.      *
 *                                                                        *
 *  IN  NO  EVENT  SHALL  SILICON  GRAPHICS  BE  LIABLE FOR ANY SPECIAL,  *
 *  INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF  ANY  KIND,  OR ANY  *
 *  DAMAGES WHATSOEVER RESULTING FROM LOSS  OF  USE,  DATA  OR  PROFITS,  *
 *  WHETHER OR NOT ADVISED OF  THE  POSSIBILITY  OF  DAMAGE,  AND ON ANY  *
 *  THEORY OF LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE USE OR  *
 *  PERFORMANCE OF THIS SOFTWARE.                                         *
 **************************************************************************/

/*
    Author : Patrick Bouchaud
             galaad@neu.sgi.com
*/

#ifndef _FLOCK_H
#define _FLOCK_H

void *flockSubscribe(char *arena);

void flockSensorGetPosition(void *flock, int sensor, float *data);
void flockSensorGetMatrix(void *flock, int sensor, float *data);
void flockSensorGetQuaternion(void *flock, int sensor, float *data);
void flockSensorGetAngles(void *flock, int sensor, float *data);
void flockSensorGetButton(void *flock, int sensor, int *data);
#endif /* _FLOCK_H */
