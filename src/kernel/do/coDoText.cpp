/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coDoText.h"

/***********************************************************************\ 
 **                                                                     **
 **   Text class                                   Version: 1.0         **
 **                                                                     **
 **                                                                     **
 **   Description  : Classes for the handling of text data              **
 **                  in a distributed manner.                           **
 **                                                                     **
 **   Classes      : coDoText                                            **
 **                                                                     **
 **   Copyright (C) 1993     by University of Stuttgart                 **
 **                             Computer Center (RUS)                   **
 **                             Allmandring 30                          **
 **                             7000 Stuttgart 80                       **
 **                                                                     **
 **                                                                     **
 **   Author       : Uwe Woessner                                       **
 **                                                                     **
 **   History      :                                                    **
 **                  14.02.95  Ver 1.0                                  **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

using namespace covise;

coDistributedObject *coDoText::virtualCtor(coShmArray *arr)
{
    coDistributedObject *ret;

    ret = new coDoText(coObjInfo(), arr);
    return ret;
}

int coDoText::getObjInfo(int no, coDoInfo **il) const
{
    if (no == 2)
    {
        (*il)[0].description = "Length";
        (*il)[1].description = "Text";
        return 2;
    }
    else
    {
        print_error(__LINE__, __FILE__, "number wrong for object info");
        return 0;
    }
}

coDoText::coDoText(const coObjInfo &info, coShmArray *arr)
    : coDistributedObject(info)
{
    setType("DOTEXT", "TEXT");
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
}

coDoText::coDoText(const coObjInfo &info, int s)
    : coDistributedObject(info)
{
    setType("DOTEXT", "TEXT");
    data.set_length(s);
#ifdef DEBUG
    cerr << "vor store_shared_dl coDoText\n";
#endif
    covise_data_list dl[] = {
        { INTSHM, &size },
        { CHARSHMARRAY, &data }
    };
    new_ok = store_shared_dl(2, dl) != 0;
    if (!new_ok)
        return;
    size = s;
}

coDoText::coDoText(const coObjInfo &info, int s, const char *d)
    : coDistributedObject(info)
{
    setType("DOTEXT", "TEXT");
#ifdef DEBUG
    cerr << "vor store_shared coDoText\n";
#endif
    data.set_length(s);
    covise_data_list dl[] = {
        { INTSHM, &size },
        { CHARSHMARRAY, &data }
    };
    new_ok = store_shared_dl(2, dl) != 0;
    if (!new_ok)
        return;
    size = s;
    char *dtmp;
    getAddress(&dtmp);
    memcpy(dtmp, d, s);
}

// we 2001-03-09: add new c'tor

coDoText::coDoText(const coObjInfo &info, const char *text)
    : coDistributedObject(info)
{
    setType("DOTEXT", "TEXT");
#ifdef DEBUG
    cerr << "vor store_shared coDoText\n";
#endif
    int sz = (int)strlen(text);
    data.set_length(sz);
    covise_data_list dl[] = {
        { INTSHM, &size },
        { CHARSHMARRAY, &data }
    };
    new_ok = store_shared_dl(2, dl) != 0;
    if (!new_ok)
        return;
    size = sz;
    char *dtmp;
    getAddress(&dtmp);
    memcpy(dtmp, text, sz);
}

coDoText *coDoText::cloneObject(const coObjInfo &newinfo) const
{
    return new coDoText(newinfo, getTextLength() + 1, getAddress());
}

int coDoText::rebuildFromShm()
{
    if (shmarr == NULL)
    {
        cerr << "called rebuildFromShm without shmarray\n";
        print_exit(__LINE__, __FILE__, 1);
    }
    covise_data_list dl[] = {
        { INTSHM, &size },
        { CHARSHMARRAY, &data }
    };
    return restore_shared_dl(2, dl);
}
