/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                          (C)2007 HLRS  **
 **                                                                        **
 ** Description: VNC Authentication functions                              **
 **                                                                        **
 **                                                                        **
 ** Author: Lukas Pinkowski                                                **
 **                                                                        **
 ** based on vncauth.cpp from VREng                                        **
 **                                                                        **
 ** History:                                                               **
 ** Oct 12, 2007                                                           **
 **                                                                        **
 ** License: GPL v2 or later                                               **
 **                                                                        **
\****************************************************************************/

//---------------------------------------------------------------------------
// VREng (Virtual Reality Engine)	http://vreng.enst.fr/
//
// Copyright (C) 1997-2007 Ecole Nationale Superieure des Telecommunications
//
// VREng is a free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public Licence as published by
// the Free Software Foundation; either version 2, or (at your option)
// any later version.
//
// VREng is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
//---------------------------------------------------------------------------
/*
 *  Copyright (C) 1999 AT&T Laboratories Cambridge.  All Rights Reserved.
 *
 *  This is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This software is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307,
 *  USA.
 */
/*
 * vncauth.cpp - Functions for VNC password management and authentication.
 */
#include "util/common.h"
#include "util/coTypes.h"
#ifndef WIN32
#include <stdint.h>
#endif
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

//#include "global.hh"
#include "vncauth.hpp"

/*
 * We use a fixed key to store passwords, since we assume that our local
 * file system is secure but nonetheless don't want to store passwords
 * as plaintext.
 */
uint8_t fixedkey[8] = { 23, 82, 107, 6, 35, 78, 88, 7 };

/*
 * Encrypt a password and store it in a file.  Returns 0 if successful,
 * 1 if the file could not be written.
 */
int vncEncryptAndStorePasswd(char *passwd, char *fname)
{
    FILE *fp;
    uint8_t encryptedPasswd[8];

    if ((fp = fopen(fname, "w")) == NULL)
        return 1;

#ifdef WIN32
    chmod(fname, 0600);
#else
    chmod(fname, S_IRUSR | S_IWUSR);
#endif

    /* pad password with nulls */
    for (uint32_t i = 0; i < 8; i++)
        encryptedPasswd[i] = (i < strlen(passwd)) ? passwd[i] : '\0';

    /* Do encryption in-place - this way we overwrite our copy of the plaintext
     password */
    deskey(fixedkey, EN0);

    des(encryptedPasswd, encryptedPasswd);

    for (int i = 0; i < 8; i++)
        putc(encryptedPasswd[i], fp);
    fclose(fp);
    return 0;
}

/*
 * Decrypt a password from a file.  Returns a pointer to a newly allocated
 * string containing the password or a null pointer if the password could
 * not be retrieved for some reason.
 */
char *vncDecryptPasswdFromFile(char *fname)
{
    FILE *fp;
    uint8_t *passwd = new uint8_t[9];

    if ((fp = fopen(fname, "r")) == NULL)
        return NULL;

    for (int i = 0; i < 8; i++)
    {
        int ch = getc(fp);
        if (ch == EOF)
        {
            fclose(fp);
            return NULL;
        }
        passwd[i] = ch;
    }
    fclose(fp);
    deskey(fixedkey, DE1);
    des(passwd, passwd);
    passwd[8] = '\0';

    return (char *)passwd;
}

/*
 * Encrypt CHALLENGESIZE bytes in memory using a password.
 */
void vncEncryptBytes(uint8_t *bytes, char *passwd)
{
    uint8_t key[8];

    /* key is simply password padded with nulls */
    for (uint32_t i = 0; i < 8; i++)
        key[i] = (i < strlen(passwd)) ? passwd[i] : '\0';
    deskey(key, EN0);

    for (int i = 0; i < CHALLENGESIZE; i += 8)
        des(bytes + i, bytes + i);
}
