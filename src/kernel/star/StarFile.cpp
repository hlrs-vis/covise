/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "StarFile.h"
#include <covise/covise.h>

using namespace covise;

char *StarFile::secure_strdup(const char *string)
{
    char *retval = NULL;
    if (NULL == string)
    {
        retval = (char *)malloc(strlen("Duplicated Nullpointer") + 1);
        strcpy(retval, "Duplicated Nullpointer");
        return retval;
    }
    else
    {
        return (strcpy((char *)malloc(strlen(string) + 1), (string)));
    }
}

char *StarFile::secure_strcat(char *s1, const char *s2)
{
    if (NULL == s1)
        return (NULL);
    if (NULL == s2)
    {
        strcat(s1, "N!");
        return s1;
    }
    else
    {
        strcat(s1, s2);
        return s1;
    }
}

char *StarFile::secure_strcpy(char *s1, const char *s2)
{
    if (NULL == s1)
        return (NULL);
    if (NULL == s2)
    {
        strcpy(s1, "N!");
        return s1;
    }
    else
    {
        strcpy(s1, s2);
        return s1;
    }
}

char *StarFile::secure_strncpy(char *s1, const char *s2, int n)
{
    if (NULL == s1)
        return (NULL);
    if (NULL == s2)
    {
        strncpy(s1, "N!", n);
        return s1;
    }
    else
    {
        strncpy(s1, s2, n);
        return s1;
    }
}

int StarFile::secure_strcmp(const char *s1, const char *s2)
{
    if (NULL == s1)
    {
        return -1;
    }
    if (NULL == s2)
    {
        return 1;
    }
    return (strcmp(s1, s2));
}
