/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN_COMMAND_H_
#define _CO_TETIN_COMMAND_H_

// 02.06.99
#include <iostream.h>
#include "coTetin.h"

/**
 * Class virtual base class for all Tetin Commands
 *
 */
class coTetinCommand
{
private:
    // chain pointer
    coTetinCommand *d_next;

    /// Copy-Constructor: NOT IMPLEMENTED
    coTetinCommand(const coTetinCommand &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetinCommand &operator=(const coTetinCommand &)
    {
        return *this;
    };

protected:
    /// default cTor: PROHIBITED
    coTetinCommand(){};

    // command name
    coTetin::Command d_comm;

    /// read a line, include all continuation lines
    istream &getLine(char *line, int length, istream &str);

    /// read a line, include all continuation lines
    istream &getFloats(float *data, int num, istream &str);

    /// read a float from a line buffer and advance pointer
    float readFloat(char *&line);

    /// read a float from a line buffer and advance pointer
    int readInt(char *&line);

    /// read a 0-terminated string from in a char field and advance Ptr
    char *getString(char *&chPtr);

    /// get an integer option: leaves value untouched if option not found
    int getOption(const char *line, const char *name, int &value)
        const;
    /// get a float option: leaves value untouched if option not found
    int getOption(const char *line, const char *name, float &value)
        const;
    /// get a STRING option: leaves value untouched if option not found
    int getOption(const char *line, const char *name, char *&value)
        const;

public:
    /// Constructor
    coTetinCommand(coTetin::Command comm)
        : d_next(NULL)
        , d_comm(comm){};

    /// Destructor
    virtual ~coTetinCommand(){};

    /// add my sizes to the appropriate counters
    virtual void addSizes(int &numInt, int &numFloat, int &numChar) const = 0;

    /// put my data to a given set of pointers
    virtual void getBinary(int *&intDat, float *&floatDat, char *&charDat) const = 0;

    /// whether the object was read in ok
    virtual int isValid() const = 0;

    /// chaining
    coTetinCommand *getNext()
    {
        return d_next;
    }
    const coTetinCommand *getNext() const
    {
        return d_next;
    }

    /// append/insert a string after me
    void append(coTetinCommand *newNext);

    /// print to a stream
    virtual void print(ostream &str) const = 0;

    /// get my command name
    coTetin::Command getCommName() const
    {
        return d_comm;
    }

    /// chcheck whether this is a 'comm'
    int is(coTetin::Command comm) const;
};

inline ostream &operator<<(ostream &str, const coTetinCommand &comm)
{
    comm.print(str);
    return str;
}
#endif
