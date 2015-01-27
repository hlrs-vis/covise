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
* Filename      : v2e_utility.c
* Author        : RJF
* Creation Date : Oct 2000
* Last Modified : $Date: $
* Version       : $Revision: $
*======================================================================
*/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <sys/types.h>
#include <sys/stat.h>

#ifndef WIN32
#include <unistd.h>
#else
#include <direct.h>
#endif

#include "v2e_macros.h"
#include "v2e_util.h"

static char ensight_directory[MAXLINE];

/**********************************************************************
 *        Open a file                                                 *
 **********************************************************************/

FILE *
open_file(char *fname, char *mode)
{
    FILE *fp;
    char str[MAXLINE];

    fp = fopen(fname, mode);
    if (!fp)
    {
        sprintf(str, "Can't open file \"%s\" \n", fname);
        ensight_message(ENSIGHT_FATAL_ERROR, str);
    }

    return (fp);
}

/**********************************************************************
 *        Close a file                                                 *
 **********************************************************************/

void
close_file(FILE *fp)
{

    if (fclose(fp) != 0)
        ensight_message(ENSIGHT_FATAL_ERROR, "Can't close file \n");
}

/**********************************************************************
 *        Open a file in the ensight directory                        *
 **********************************************************************/

FILE *
open_ensight_file(char *fname, char *mode)
{
    FILE *fp;

    fp = open_file(get_ensight_pathname(fname), mode);

    return (fp);
}

/**********************************************************************
 *        Unlink a file in the ensight directory                      *
 **********************************************************************/

void
unlink_ensight_file(char *fname)
{
    char message[MAXLINE];

    if (unlink(fname))
    {
        sprintf(message, "Could not unlink file \"%s\" \n", fname);
        ensight_message(ENSIGHT_WARNING, message);
        perror("unlink");
    }
}

/**********************************************************************
 *        Unlink a file in the ensight directory                      *
 **********************************************************************/

char *
get_ensight_pathname(char *fname)
{
    char *pathname;
    pathname = (char *)RU_allocMem(MAXLINE * sizeof(char), "PNAME");

    /* construct full pathname : ./ensight.<POST file name>/<fname> */
    strcpy(pathname, "./");
    strcat(pathname, ensight_directory);
    strcat(pathname, "/");
    strcat(pathname, fname);

    return (pathname);
}

/**********************************************************************
 *        Make an ensight directory                                   *
 **********************************************************************/

void
make_ensight_directory(char *basename)
{
    char message[MAXLINE];

    /* construct directory name : ensight.<POST file name> */
    strcpy(ensight_directory, "ensight.");
    strcat(ensight_directory, basename);

/* make the directory */
#ifdef WIN32
    if (_mkdir(ensight_directory))
    {
#else
    if (mkdir(ensight_directory, 00774))
    {
#endif
#ifdef DEBUG
        sprintf(message, "make ensight directory \"%s\" failed (may already exist) \n", ensight_directory);
        ensight_message(ENSIGHT_WARNING, message);
#else
        sprintf(message, "make ensight directory \"%s\" failed (may already exist) : overwriting \n", ensight_directory);
        ensight_message(ENSIGHT_FATAL_ERROR, message);
#endif /* DEBUG */
    }
    else
    {
        sprintf(message, "made ensight directory \"%s\" \n", ensight_directory);
        ensight_message(ENSIGHT_INFO, message);
    }
}

/**************************************************************************
 * Output error message and terminate translator if fatal error           * 
 **************************************************************************/

void
ensight_message(int msgtype, char *s)
{
    char str[MAXLINE];

    /* Add message type to output string */
    switch (msgtype)
    {
    case ENSIGHT_FATAL_ERROR: /* Fatal error message */
        strcpy(str, "    *FATAL*");
        break;
    case ENSIGHT_WARNING: /* Warning message */
        strcpy(str, "    *WARNING*");
        break;
    case ENSIGHT_INFO: /* Information message */
        strcpy(str, "    *INFO*");
        break;
    default: /* Unknown message (fatal) */
        strcpy(str, "Unknown message type (");
        strcat(str, s); /* Add original message onto error message */
        strcat(str, ")");
        ensight_message(ENSIGHT_FATAL_ERROR, str);
        break;
    }

    /* Add origin of message (i.e. external duct solver) */
    strcat(str, " ENSIGHT TRANSLATOR : ");

    /* Add message to output string */
    strcat(str, s);

    /* Print output string to stderr */
    fprintf(stderr, "%s", str);

    /* flush stderr */
    fflush(stderr);

    /* exit if fatal error */
    if (msgtype == ENSIGHT_FATAL_ERROR)
        exit(EXIT_FAILURE);
}
