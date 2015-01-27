/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__REPLAY_FILENAME_H_
#define _CO_TETIN__REPLAY_FILENAME_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 04.06.99

/**
 * Class coTetin__replayFile implements Tetin file "replay_file_name" command
 *
 */
class coTetin__replayFile : public coTetinCommand
{

private:
    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__replayFile(const coTetin__replayFile &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__replayFile &operator=(const coTetin__replayFile &)
    {
        return *this;
    };

    /// Default constructor:
    coTetin__replayFile()
        : name(0){};

    // ===================== the command's data =====================

    // value
    char *name;

public:
    /// read from file
    coTetin__replayFile(istream &str, int binary, ostream &ostr = cerr);

    /// read from memory
    coTetin__replayFile(int *&intDat, float *&floatDat, char *&charDat);

    coTetin__replayFile(char *filename);

    /// Destructor
    virtual ~coTetin__replayFile();

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
