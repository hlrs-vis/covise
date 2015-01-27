/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <string.h>
#include <strings.h>
#include <stdlib.h>
#include <stdio.h>
#include "getcgi.h"
char *get_cgi_string(void);
static char x2c(char *what);

extern char *gv(char *var);

char *get_cgi_string(void)
{
    char *rm; /* request_method */
    char *cgistr;

    /* Is there any CGI avaiable ? */
    rm = getenv("REQUEST_METHOD");
    if (rm == NULL)
        return NULL;
    //printf("testi<BR>\n");
    /* request methode */
    if (!strcmp(rm, "GET"))
    {
        /* get cgi string from enviroment */
        if (0 == (strlen(cgistr = getenv("QUERY_STRING"))))
            return (NULL);
        return cgistr;
    }
    else
    {
        if (!strcmp(rm, "POST"))
        {
            int len;
            char *input;
            //if ( strcasecmp(getenv("CONTENT_TYPE"), "application/x-www-form-urlencoded")) return NULL;
            if (!(len = atoi(getenv("CONTENT_LENGTH"))))
                return NULL;
            if (!(input = (char *)malloc((size_t)len + 1)))
                return NULL;
            if (!fread(input, len, 1, stdin))
            {
                free(input);
                return NULL;
            }
            input[len] = '\0';
            return input;
        }
    }
    return NULL;
}

/* ------------------------------------------------------------------------- */

char x2c(char *what)
{
    register char digit;

    digit = (what[0] >= 'A' ? ((what[0] & 0xdf) - 'A') + 10 : (what[0] - '0'));
    digit *= 16;
    digit += (what[1] >= 'A' ? ((what[1] & 0xdf) - 'A') + 10 : (what[1] - '0'));
    return (digit);
}

/* ------------------------------------------------------------------------- */

void unescape_url(char *url)
{
    register int i, j;

    for (i = 0, j = 0; url[j]; ++i, ++j) /* ist das richtig?? */
    {
        if ((url[i] = url[j]) == '%')
        {
            url[i] = x2c(&url[j + 1]);
            j += 2;
        }
    }
    url[i] = '\0';
}

/* ------------------------------------------------------------------------- */

CGI *InitCgiVars(void)
{
    char *cgistr = NULL; /* 'original' cgi-string                      */
    char *tok = NULL; /* token - stop (for strtok)                  */
    char *h = NULL; /* helper                                     */
    long c = 0; /* counter                                    */
    char **var = NULL; /* vektorlist of all cgi-variables            */
    char **val = NULL; /* vektorlist of all cgi-values               */

    CGI *cgi;

    /* Get CGI-String */
    if (NULL == (cgistr = h = get_cgi_string()))
        return NULL;

    /* change letters and count variables */
    while (*h)
    {
        switch (*h)
        {
        case '+':
            *h = ' ';
            break; /* toggle '+' into ' '   */
        case '=':
            c++;
            break; /* one more variable     */
        default:
            break; /* next letter in string */
        }
        h++;
    }

    if (!c)
        return NULL;

    /* allocate memory for cgi-vektors */
    var = (char **)malloc((size_t)sizeof(char *) * c);
    val = (char **)malloc((size_t)sizeof(char *) * c);
    cgi = (CGI *)malloc((size_t)sizeof(CGI));

    if ((val == NULL) || (var == NULL) || (cgi == NULL))
    {
        free(cgistr);
        if (val)
            free(val);
        if (cgi)
            free(cgi);
        if (var)
            free(var);
        return NULL;
    }

    cgi->cgi_variable = var;
    cgi->cgi_value = val;
    cgi->cgi_count = c;
    cgi->cgi_file = cgistr;

    bzero(val, 0);
    c = cgi->cgi_count;
    while (c--)
        val[c] = NULL;
    return NULL;

    /* split on '&' to extract the name-value pairs */
    c = 0;
    tok = strtok(cgistr, "name=\"");
    while (tok)
    {
        tok = strtok(NULL, "name=\"");
        var[c++] = tok;
    }

    /* split varname and value */
    c = cgi->cgi_count;
    while (c--)
    {
        //val[c]=var[c];
        //val[c]  = strchr(var[c], '"');
        // *val[c] = '\0';
        //val[c]++;
        //unescape_url(val[c]);
    }
    return (cgi);
}

/* ------------------------------------------------------------------------- */

char *GetCgiVar(CGI *cgi, char *name)
{
    long pos = 0;

    while (pos < cgi->cgi_count)
    {
        if (cgi->cgi_variable[pos] == NULL)
            return ("NULLVAR");
        if (!(strcmp(name, cgi->cgi_variable[pos])))
        {
            if (cgi->cgi_value[pos] == NULL)
                return ("NULLVAL");
            return (cgi->cgi_value[pos]);
        }
        pos++;
    }
    return NULL;
}

/* ------------------------------------------------------------------------- */

void FreeCgiVars(CGI *cgi)
{
    free(cgi->cgi_variable);
    free(cgi->cgi_value);
    free(cgi->cgi_file);
    free(cgi);
}

/* ------------------------------------------------------------------------- */
