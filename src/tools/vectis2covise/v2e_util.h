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
 * Filename      : v2e_util.h
 * Author        : RJF
 * Creation Date : Oct 2000
 * Last Modified : $Date: $
 * Version       : $Revision: $
 *======================================================================
 */

FILE *open_file(char *fname, char *mode);
void close_file(FILE *fp);

FILE *open_ensight_file(char *fname, char *mode);
void unlink_ensight_file(char *fname);
char *get_ensight_pathname(char *fname);

void make_ensight_directory(char *basename);
void ensight_message(int msgtype, char *s);
