/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__TETIN_FILENAME_H_
#define _CO_TETIN__TETIN_FILENAME_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 04.06.99

/**
 * Class coTetin__tetinFile implements Tetin file "tetin_file_name" command
 *
 */
class coTetin__tetinFile : public coTetinCommand
{

private:
    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__tetinFile(const coTetin__tetinFile &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__tetinFile &operator=(const coTetin__tetinFile &)
    {
        return *this;
    };

    /// Default constructor:
    coTetin__tetinFile()
        : name(0){};

    // ===================== the command's data =====================

    // value
    char *name;

public:
    /// read from file
    coTetin__tetinFile(istream &str, int binary, ostream &ostr = cerr);

    /// read from memory
    coTetin__tetinFile(int *&intDat, float *&floatDat, char *&charDat);

    coTetin__tetinFile(const char *filename);

    /// Destructor
    virtual ~coTetin__tetinFile();

    /// whether object is valid
    virtual int isValid() const;

    /// count size required in fields
    virtual void addSizes(int &numInt, int &numFloat, int &numChar) const;

    /// put my data to a given set of pointers
    virtual void getBinary(int *&intDat, float *&floatDat, char *&charDat) const;

    /// print to a stream in Tetin format
    virtual void print(ostream &str) const;

    // ===================== command-specific functions =====================

    /// get the value
    char *getValue() const
    {
        return name;
    }
};
#endif
