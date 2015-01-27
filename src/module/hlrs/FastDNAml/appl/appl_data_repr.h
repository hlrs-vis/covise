/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LIBAPPL_APPL_DATA_REPR_H
#define _LIBAPPL_APPL_DATA_REPR_H

#ifdef __cplusplus
#ifndef EXTERN
#define EXTERN extern "C"
#endif
#else
#ifndef EXTERN
#define EXTERN extern
#endif
#endif

/******************************************************************************
 *  Data representation                                                       *
 ******************************************************************************/

/* Representation for scalar types is simply native representation */

EXTERN void *appl_data_repr_create(appl_data_descr_t *descr, void *host_repr);

#endif
