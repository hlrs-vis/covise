/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 *  error_info.h include file.
 */

/*  Include File SCCS header
 *  "@(#)SCCSID: error_info.h 1.1"
 *  "@(#)SCCSID: Version Created: 11/18/92 20:43:01"
 *
 */

#define INFO 5 /* Informative message only. */
#define WARNING 10 /* Warning, but with good chance of success. */
#define SEVERE 50 /* Non-fatal, but little chance of success. */
#define FATAL 90 /* Fatal, must exit immediately. */

#define MSG_LEN 512 /* Maximum length for message strings. */
