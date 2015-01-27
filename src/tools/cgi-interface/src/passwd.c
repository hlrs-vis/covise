/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * passwdgen.c
 */

#include <stdio.h>
#include <sys/types.h>
#include <time.h>
#include "passwd.h"

extern char *crypt();

/* #define RAND48
#ifdef RAND48
#define random	lrand48
#define srandom	srand48
#endif
*/

#define DEFAULTLENGTH 8 /* default length of passwords */

char vowels[] = "aeiou";
char consonants[] = "bcdfghjklmnpqrstvwxyz";
char saltchars[] = "./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

#define Vowel() vowels[random() % (sizeof(vowels) - 1)]
#define Consonant() consonants[random() % (sizeof(consonants) - 1)]
#define Saltchar() saltchars[random() % (sizeof(saltchars) - 1)]

void GetPasswd(int length, char *password, char *passwordcrypted)

{
    int howmany, count, chr, pid;
    char pwsalt[3];

    if (length == 0)
        length = DEFAULTLENGTH;
    else if (length < 6)
        length = 6;
    else if (length > 16)
        length = 16;

    pid = getpid();
    srandom(((int)(time((time_t *)0)) / pid) + pid + 3873);

    pwsalt[3] = 0;

    pwsalt[0] = Saltchar();
    pwsalt[1] = Saltchar();
    password[length] = 0;

    for (chr = 0; chr < length; chr += 2)
        password[chr] = Consonant();
    for (chr = 1; chr < length; chr += 2)
        password[chr] = Vowel();
    for (chr = 0; chr < length; chr++)
        if (!(random() % 3))
            password[chr] += 'A' - 'a';

    sprintf(passwordcrypted, "%s", crypt(password, pwsalt));
}
