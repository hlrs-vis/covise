/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DO_TEXT_H
#define CO_DO_TEXT_H

#include "coDistributedObject.h"

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
namespace covise
{

class DOEXPORT coDoText : public coDistributedObject
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);

private:
    coIntShm size; // number of characters
    coCharShmArray data; // characters

protected:
    int rebuildFromShm();
    int getObjInfo(int, coDoInfo **) const;
    coDoText *cloneObject(const coObjInfo &newinfo) const;

public:
    coDoText(const coObjInfo &info)
        : coDistributedObject(info)
    {
        setType("DOTEXT", "TEXT");
        if (name)
        {
            if (getShmArray() != 0)
            {
                if (rebuildFromShm() == 0)
                {
                    print_comment(__LINE__, __FILE__, "rebuildFromShm == 0");
                }
            }
            else
            {
                print_comment(__LINE__, __FILE__, "object %s doesn't exist", name);
                new_ok = 0;
            }
        }
    }

    // internally used c'tor
    coDoText(const coObjInfo &info, coShmArray *arr);

    // create empty object with 'size' elements
    coDoText(const coObjInfo &info, int size);

    // create object with 'size' elements and copy 'size' elements from 'data'
    coDoText(const coObjInfo &info, int size, const char *data);

    // create object with 'strlen(text)' elements and copy 'text' into object
    coDoText(const coObjInfo &info, const char *text);

    virtual ~coDoText()
    {
    }

    int getTextLength() const
    {
        return (int)size;
    }
    void getAddress(char **base) const
    {
        *base = (char *)data.getDataPtr();
    }
    const char *getAddress() const
    {
        return static_cast<char *>(data.getDataPtr());
    }
};
}
#endif
