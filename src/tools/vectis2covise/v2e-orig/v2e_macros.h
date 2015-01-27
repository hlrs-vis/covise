/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*RICARDO SQA =========================================================
 * Status        : UNASSURED
 * Module Name   : v2e
 * Subject       : Vectis Phase 5 POST to Ensight Gold convertor
 * Language      : ANSI C
 * Requires      : RUtil (on little-endian platforms only)
 * Documentation : README.html
 * Filename      : v2e_macros.h
 * Author        : RJF
 * Creation Date : Oct 2000
 * Last Modified : $Date: $
 * Version       : $Revision: $
 *======================================================================
 */

/* version */
#define VERSION "3.5b1"

/* string length limits */
#define MAXLINE 256

/* message types */
#define ENSIGHT_INFO 0
#define ENSIGHT_WARNING 1
#define ENSIGHT_FATAL_ERROR 2
