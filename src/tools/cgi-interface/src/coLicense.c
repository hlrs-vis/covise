/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <time.h>

#include "parse.h"
//#include "match.h"
#include "getcgi.h"

//#include "passwd.h"
char *gv(char *var)
{
    char *leer = "(NULL)";
    char *v;

    if (NULL == (v = getenv(var)))
        return leer;
    return v;
}
extern char *mklicense(char *hostkey, char *userkey, char *name, int days);
extern char *get_cgi_string(void);
char *vars[1000];
char *vals[1000];
int numvar = 0;
char *cgistr = NULL;
void initVars()
{
    int c = 0, len;
    numvar = 0;
    if (cgistr)
        while (cgistr[c])
        {
            while ((cgistr[c]) && (cgistr[c] != '\"'))
                c++;
            if (cgistr[c])
            {
                c++;
                vars[numvar] = cgistr + c;
                while ((cgistr[c]) && (cgistr[c] != '\"'))
                    c++;
                cgistr[c] = '\0';
                c++;
                while ((cgistr[c]) && ((cgistr[c] == '\r') || (cgistr[c] == '\n') || (cgistr[c] == ' ')))
                {
                    //printf("char: %c %d<BR>\n",cgistr[c],(int)cgistr[c]);
                    c++;
                }
                //printf("char: %c %d<BR>\n",cgistr[c],(int)cgistr[c]);
                vals[numvar] = cgistr + c;
                while ((cgistr[c]) && (strncmp(cgistr + c, "------", 6) != 0))
                {
                    //printf("char: %c %d<BR>\n",cgistr[c],(int)cgistr[c]);
                    c++;
                }
                cgistr[c] = '\0';
                if ((cgistr[c - 1] == '\r') || (cgistr[c - 1] == '\n'))
                    cgistr[c - 1] = '\0';
                if ((cgistr[c - 2] == '\r') || (cgistr[c - 2] == '\n'))
                    cgistr[c - 2] = '\0';
                len = strlen(vals[numvar]);
                while (len >= 0)
                {
                    if ((vals[numvar][len] == ' ') || (vals[numvar][len] == '\t'))
                        vals[numvar][len] = '\0';
                    else
                        break;
                    len--;
                }
                numvar++;
            }
            while ((cgistr[c]) && (cgistr[c] != '<'))
                c++;
            c++;
        }
}

char *getVar(char *name)
{
    int i;
    for (i = 0; i < numvar; i++)
    {
        if (strcmp(name, vars[i]) == 0)
        {
            if (strlen(vals[i]) < 1)
                return ("none");
            return (vals[i]);
        }
    }
    return ("none");
}

char *getSVar(char *name)
{
    int i;
    for (i = 0; i < numvar; i++)
    {
        if (strcmp(name, vars[i]) == 0)
        {
            if (strlen(vals[i]) < 1)
                return (NULL);
            return (vals[i]);
        }
    }
    return (NULL);
}
void gen_error()
{
    printf("UNKNOWN ERROR<BR> Please contact <A HREF=\"mailto:covise@vision.rus.uni-stuttgart.de\">COVISE developers</A><BR>\n");
    exit(0);
}

void toShort(char *var)
{
    printf("ERROR: \"%s\" is to short see <A HREF=\"http://www.hlrs.de/structure/organisation/vis/covise/support/license/\"\
>LICENSE</A> for instructions how to get you Host ID and User ID<BR>\n",
           var);
    exit(0);
}

void required(char *var)
{
    printf("ERROR: Please fill out field \"%s\" and all other fields marked with a \"*\"<BR>\n", var);
    exit(0);
}

int main(int argc, char **argv)
{
    char buf[2000];
    char *lic;
    FILE *fp;
    cgistr = get_cgi_string();
    if (!cgistr)
        return (0);
    puts("Content-type: text/html\n\n");
    //printf("cgistr: %s<BR>\n",cgistr);
    initVars();
    if (getSVar("Name") == NULL)
        required("Name");
    if (getSVar("Company") == NULL)
        required("Company");
    if (getSVar("Email") == NULL)
        required("Email");
    if (getSVar("Hostname") == NULL)
        required("Hostname");
    if (getSVar("HID") == NULL)
        required("Host ID");
    if (getSVar("UID") == NULL)
        required("User ID");
    if (strlen(getSVar("UID")) != 8)
        toShort("User ID");
    if (strlen(getSVar("HID")) != 8)
        toShort("Host ID");
    fp = fopen("/tmp/mailfile", "w");
    if (fp == NULL)
        gen_error();
    fprintf(fp, "Subject: License Request\n");
    fprintf(fp, "Hi folks, someone requested a demo License\n\n");
    fprintf(fp, "Name: %s\n", getSVar("Name"));
    fprintf(fp, "Company: %s\n", getSVar("Company"));
    fprintf(fp, "Institute: %s\n", getVar("Institute"));
    fprintf(fp, "Usage: %s\n", getVar("Usage"));
    fprintf(fp, "Street: %s\n", getVar("Street"));
    fprintf(fp, "Town: %s\n", getVar("Town"));
    fprintf(fp, "Country: %s\n", getVar("Country"));
    fprintf(fp, "Email: %s\n", getVar("Email"));
    fprintf(fp, "Phone: %s\n", getVar("Phone"));
    fprintf(fp, "Fax: %s\n", getVar("Fax"));
    fprintf(fp, "Hostname: %s\n", getVar("Hostname"));
    fprintf(fp, "Host ID: %s\n", getVar("HID"));
    fprintf(fp, "User ID: %s\n", getVar("UID"));
    fprintf(fp, "Comments: %s\n", getVar("Comments"));
    fprintf(fp, "\n");
    lic = mklicense(getSVar("HID"), getSVar("UID"), getSVar("Hostname"), 14);
    fprintf(fp, "License: %s\n", lic);
    fprintf(fp, "\n");
    fclose(fp);
    system("cat /tmp/mailfile | mail covadmin@vis-mail.rus.uni-stuttgart.de");
    // Now print out the reply
    fp = fopen(REPLY_HEADER, "r");
    if (fp == NULL)
        gen_error();
    while (!feof(fp))
    {
        if (fgets(buf, 2000, fp) == NULL)
            break;
        puts(buf);
    }
    fclose(fp);
    printf("<TT> Key %s </TT>", lic);
    fp = fopen(REPLY_FOOTER, "r");
    if (fp == NULL)
        gen_error();
    while (!feof(fp))
    {
        if (fgets(buf, 2000, fp) == NULL)
            break;
        puts(buf);
    }
    fclose(fp);
    // Done
    return (0);
}
