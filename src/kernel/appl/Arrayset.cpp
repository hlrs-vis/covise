/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Interface class for arrayset parameter                    **
 **              used in COVISE software environment                       **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Dirk Rantzau                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  10.08.95  V1.0                                                  **
\**************************************************************************/

#include <covise/covise.h>
#include "ApplInterface.h"
#include "Arrayset.h"

//=====================================================================
//
//=====================================================================
Arrayset::Arrayset(char *n, char *rt, char *ct, int rd, int cd, char *msgbuf)
{

    //
    // prepare header
    //
    buf = msgbuf;

    pname = n;
    rtitle = rt;
    ctitle = ct;
    rdim = rd;
    cdim = cd;
    rcounter = 0;
    ccounter = cdim;
    num_tokens = 0;
    //cerr << "Creating new arrayset..." << endl;
}

//=====================================================================
//
//=====================================================================
Arrayset::Arrayset(char *msgbuf)
{
    //
    // skip header
    //
    buf = msgbuf;
    strtok(buf, "\n");
    strtok(NULL, "\n");
    strtok(NULL, "\n");
    strtok(NULL, "\n");
    strtok(NULL, "\n");
    strtok(NULL, "\n");
    strtok(NULL, "\n"); // no of tokens
    rtitle = strtok(NULL, "\n");
    ctitle = strtok(NULL, "\n");
    cdim = atoi(strtok(NULL, "\n"));
    rdim = atoi(strtok(NULL, "\n"));

    rcounter = 0;
    ccounter = cdim;
}

//=====================================================================
//
//=====================================================================
Arrayset::~Arrayset()
{
}

//=====================================================================
//
//=====================================================================
int Arrayset::get_header(char **rt, char **ct, int *rd, int *cd)
{
    *rt = rtitle;
    *ct = ctitle;
    *rd = rdim;
    *cd = cdim;

    return 4;
}

//=====================================================================
//
//=====================================================================
int Arrayset::get_next_param(char **name, char **type)
{

    *name = strtok(NULL, "\n");
    *type = strtok(NULL, "\n");

    return 2;
}

//=====================================================================
//
//=====================================================================
int Arrayset::get_boolean_param(int *values)
{

    for (int i = 0; i < cdim; i++)
    {

        if (strcmp(strtok(NULL, "\n"), "TRUE") == 0)
            *(values + i) = 1;
        else
            *(values + i) = 0;
    }
    return cdim;
}

//=====================================================================
//
//=====================================================================
int Arrayset::get_int_scalar_param(int *values)
{

    for (int i = 0; i < cdim; i++)
        *(values + i) = atoi(strtok(NULL, "\n"));

    return cdim;
}

//=====================================================================
//
//=====================================================================
int Arrayset::get_float_scalar_param(float *values)
{

    for (int i = 0; i < cdim; i++)
        *(values + i) = (float)atof(strtok(NULL, "\n"));

    return cdim;
}

//=====================================================================
//
//=====================================================================
int Arrayset::get_choice_num_labels()
{

    num_labels = atoi(strtok(NULL, "\n")) - cdim;

    return num_labels;
}

//=====================================================================
//
//=====================================================================
int Arrayset::get_choice_param(int *selected, char **labels)
{
    int i;
    for (i = 0; i < cdim; i++)
        *(selected + i) = atoi(strtok(NULL, "\n"));
    for (i = 0; i < num_labels; i++)
        *(labels + i) = strtok(NULL, "\n");

    return cdim;
}

//=====================================================================
//
//=====================================================================
int Arrayset::get_string_param(char **strings)
{

    for (int i = 0; i < cdim; i++)
        strings[i] = strtok(NULL, "\n");

    return cdim;
}

//=====================================================================
//
//=====================================================================
int Arrayset::get_int_slider_param(int *min, int *max, int *values)
{

    for (int i = 0; i < cdim; i++)
    {
        *(min + i) = atoi(strtok(NULL, "\n"));
        *(max + i) = atoi(strtok(NULL, "\n"));
        *(values + i) = atoi(strtok(NULL, "\n"));
    }
    return cdim;
}

//=====================================================================
//
//=====================================================================
int Arrayset::get_float_slider_param(float *min, float *max, float *values)
{

    for (int i = 0; i < cdim; i++)
    {
        *(min + i) = (float)atof(strtok(NULL, "\n"));
        *(max + i) = (float)atof(strtok(NULL, "\n"));
        *(values + i) = (float)atof(strtok(NULL, "\n"));
    }
    return cdim;
}

//=====================================================================
//
//=====================================================================
int Arrayset::add_boolean_param(char *name, int *values)
{
    if (rcounter < rdim)
    {

        strcat(buf, name);
        strcat(buf, "\n");
        num_tokens++;
        strcat(buf, "Boolean");
        strcat(buf, "\n");
        num_tokens++;
        for (int i = 0; i < cdim; i++)
        {
            if (values[i] == 0)
            {
                strcat(buf, "FALSE\n");
            }
            else
            {
                strcat(buf, "TRUE\n");
            }
            num_tokens++;
        }

        rcounter++;
        //cerr << "Tokens nach boolean" << num_tokens << endl;
        return rcounter;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int Arrayset::add_string_param(char *name, char **strings)
{
    if (rcounter < rdim)
    {

        strcat(buf, name);
        strcat(buf, "\n");
        num_tokens++;
        strcat(buf, "String");
        strcat(buf, "\n");
        num_tokens++;
        for (int i = 0; i < cdim; i++)
        {
            strcat(buf, strings[i]);
            strcat(buf, "\n");
            num_tokens++;
        }

        rcounter++;
        //cerr << "Tokens nach string" << num_tokens << endl;
        return rcounter;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int Arrayset::add_choice_param(char *name, int *selected, int num, char **labels)
{
    if (rcounter < rdim)
    {

        strcat(buf, name);
        strcat(buf, "\n");
        num_tokens++;
        strcat(buf, "Choice");
        strcat(buf, "\n");
        num_tokens++;
        sprintf(tmpbuf, "%d\n", num + cdim);
        strcat(buf, tmpbuf);
        num_tokens++;
        int i;
        for (i = 0; i < cdim; i++)
        {
            sprintf(tmpbuf, "%d\n", selected[i]);
            strcat(buf, tmpbuf);
            num_tokens++;
        }
        for (i = 0; i < num; i++)
        {
            strcat(buf, labels[i]);
            strcat(buf, "\n");
            num_tokens++;
        }

        rcounter++;
        //cerr << "Tokens nach choice" << num_tokens << endl;
        return rcounter;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int Arrayset::add_float_scalar_param(char *name, float *values)
{
    if (rcounter < rdim)
    {

        strcat(buf, name);
        strcat(buf, "\n");
        num_tokens++;
        strcat(buf, "FloatScalar");
        strcat(buf, "\n");
        num_tokens++;
        for (int i = 0; i < cdim; i++)
        {
            sprintf(tmpbuf, "%f\n", values[i]);
            strcat(buf, tmpbuf);
            num_tokens++;
        }

        rcounter++;
        //cerr << "Tokens nach float scalar" << num_tokens << endl;
        return rcounter;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int Arrayset::add_int_scalar_param(char *name, int *values)
{
    if (rcounter < rdim)
    {

        strcat(buf, name);
        strcat(buf, "\n");
        num_tokens++;
        strcat(buf, "IntScalar");
        strcat(buf, "\n");
        num_tokens++;
        for (int i = 0; i < cdim; i++)
        {
            sprintf(tmpbuf, "%d\n", values[i]);
            strcat(buf, tmpbuf);
            num_tokens++;
        }

        rcounter++;
        //cerr << "Tokens nach int scalar" << num_tokens << endl;
        return rcounter;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int Arrayset::add_int_slider_param(char *name, int *min, int *max, int *values)
{
    if (rcounter < rdim)
    {

        strcat(buf, name);
        strcat(buf, "\n");
        num_tokens++;
        strcat(buf, "IntSlider");
        strcat(buf, "\n");
        num_tokens++;
        for (int i = 0; i < cdim; i++)
        {
            sprintf(tmpbuf, "%d\n", min[i]);
            strcat(buf, tmpbuf);
            num_tokens++;
            sprintf(tmpbuf, "%d\n", max[i]);
            strcat(buf, tmpbuf);
            num_tokens++;
            sprintf(tmpbuf, "%d\n", values[i]);
            strcat(buf, tmpbuf);
            num_tokens++;
        }

        rcounter++;
        //cerr << "Tokens nach int slider" << num_tokens << endl;
        return rcounter;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int Arrayset::add_float_slider_param(char *name, float *min, float *max, float *values)
{
    if (rcounter < rdim)
    {

        strcat(buf, name);
        strcat(buf, "\n");
        num_tokens++;
        strcat(buf, "FloatSlider");
        strcat(buf, "\n");
        num_tokens++;
        for (int i = 0; i < cdim; i++)
        {
            sprintf(tmpbuf, "%f\n", min[i]);
            strcat(buf, tmpbuf);
            num_tokens++;
            sprintf(tmpbuf, "%f\n", max[i]);
            strcat(buf, tmpbuf);
            num_tokens++;
            sprintf(tmpbuf, "%f\n", values[i]);
            strcat(buf, tmpbuf);
            num_tokens++;
        }

        rcounter++;
        //cerr << "Tokens nach float slider" << num_tokens << endl;
        return rcounter;
    }
    else
        return 0;
}

//=====================================================================
//
//=====================================================================
int Arrayset::current_length()
{

    return strlen(buf);
}

//=====================================================================
//
//=====================================================================
int Arrayset::finish_and_send()
{
    int size;
    char numbuf[10];
    char *msg_buf;

    size = 1; // \0
    size += strlen(pname) + 1;
    size += strlen("Arrayset") + 1;
    size += 5 + 1; // num_tokens,
    size += strlen(rtitle) + 1;
    size += strlen(ctitle) + 1;
    size += 5 + 1; // rdim
    size += 5 + 1; // cdim
    size += strlen(buf) + 1;

    msg_buf = new char[size];

    strcpy(msg_buf, pname);
    strcat(msg_buf, "\n");
    strcat(msg_buf, "Arrayset");
    strcat(msg_buf, "\n");

    // cerr << "Tokens bisher gesamt: " << num_tokens << endl;
    // cerr << "Addiere + 4 dazu (rtitle,ctitle,rdim,cdim)" << endl;
    sprintf(numbuf, "%d\n", num_tokens + 4);
    strcat(msg_buf, numbuf);
    strcat(msg_buf, rtitle);
    strcat(msg_buf, "\n");
    strcat(msg_buf, ctitle);
    strcat(msg_buf, "\n");
    sprintf(numbuf, "%d\n", cdim);
    strcat(msg_buf, numbuf);
    sprintf(numbuf, "%d\n", rdim);
    strcat(msg_buf, numbuf);
    strcat(msg_buf, buf);
    strcat(msg_buf, "\0");

    // build and send message
    // cerr << "----------------------" << endl;
    // cerr << "I'm sending the following ArraySet to the Controller/Mapeditor" << endl;
    // cerr << msg_buf << endl;
    // cerr << "----------------------" << endl;

    Covise::send_ui_message("PARRESET", msg_buf);

    delete[] msg_buf;

    return 1;
}
