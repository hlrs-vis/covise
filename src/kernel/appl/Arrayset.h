/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ARRAYSET_H
#define _ARRAYSET_H

/*
   $Log$
*/

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description: Interface class for application modules to the COVISE     **
 **              software environment                                      **
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
 ** Date:  18.05.94  V1.0                                                  **
\**************************************************************************/

#include <covise/covise.h>

#define AL_DLLEXPORT

#include <util/coTypes.h>

class AL_DLLEXPORT APPLEXPORT Arrayset
{

    friend class Covise;

private:
    char tmpbuf[100];
    char *buf;
    char *pname;
    char *rtitle;
    char *ctitle;
    int rdim, rcounter;
    int cdim, ccounter;
    int num_labels;
    int num_tokens;

public:
    Arrayset(char *name, char *rtitle, char *ctitle, int rdim, int cdim, char *msgbuf);
    Arrayset(char *msgbuf);

    //
    // store
    //
    int add_string_param(char *name, char **strings);
    int add_boolean_param(char *name, int *values);
    int add_int_scalar_param(char *name, int *values);
    int add_float_scalar_param(char *name, float *values);
    int add_choice_param(char *name, int *selected, int num_labels, char **labels);
    int add_int_slider_param(char *name, int *min, int *max, int *values);
    int add_float_slider_param(char *name, float *min, float *max, float *values);
    int finish_and_send();

    //
    // retrieval
    //
    int get_header(char **rtitle, char **ctitle, int *rdim, int *cdim);
    int get_next_param(char **name, char **type);
    int get_boolean_param(int *values);
    int get_int_scalar_param(int *values);
    int get_float_scalar_param(float *values);
    int get_string_param(char **strings);
    int get_choice_param(int *selected, char **labels);
    int get_choice_num_labels();
    int get_int_slider_param(int *min, int *max, int *values);
    int get_float_slider_param(float *min, float *max, float *values);

    int current_length();

    ~Arrayset();
};
#endif
