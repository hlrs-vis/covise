/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _RESULT_PARAM_H
#define _RESULT_PARAM_H

#include <string.h>
#include <util/coExport.h>

class SCAEXPORT ResultParam
{

public:
    typedef enum
    {
        INT,
        FLOAT,
        ENUM
    } Type;

    ResultParam(Type ptype);
    virtual ~ResultParam();

    Type getType()
    {
        return ptype_;
    };

    const char *getDirName();
    const char *getValue() const
    {
        return val_;
    }
    virtual const char *getClosest(float &diff, int num, const char *const *entries)
    {
        (void)diff;
        (void)num;
        (void)entries;
        return NULL;
    };

protected:
    char *dirname_; // dir name ( <name>=<value> )
    char *name_;
    char *val_;

    Type ptype_;

    void setLabel(const char *name, const char *value);
    void setLabel(const char *value);

private:
};
#endif
