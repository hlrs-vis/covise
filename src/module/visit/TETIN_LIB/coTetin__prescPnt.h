/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__PRESCRIBED_POINT_H_
#define _CO_TETIN__PRESCRIBED_POINT_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 09.06.99

/**
 * Class coTetin__prescPnt implements Tetin file "prescribed_point" command
 *
 */
class coTetin__prescPnt : public coTetinCommand
{

private:
    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__prescPnt(const coTetin__prescPnt &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__prescPnt &operator=(const coTetin__prescPnt &)
    {
        return *this;
    };

    /// Default constructor: NOT IMPLEMENTED
    coTetin__prescPnt(){};

    // ===================== the command's data =====================

    float coord[3];
    char *d_family, *d_name;

public:
    /// read from file
    coTetin__prescPnt(istream &str, int binary);

    /// read from memory
    coTetin__prescPnt(int *&, float *&, char *&);

    /// pass directly
    coTetin__prescPnt(float point_x, float point_y,
                      float point_z, char *pp_name);

    /// Destructor
    virtual ~coTetin__prescPnt();

    /// whether object is valid
    virtual int isValid() const;

    /// count size required in fields
    virtual void addSizes(int &numInt, int &numFloat, int &numChar) const;

    /// put my data to a given set of pointers
    virtual void getBinary(int *&intDat, float *&floatDat, char *&charDat) const;

    /// print to a stream in Tetin format
    virtual void print(ostream &str) const;

    // ===================== command-specific functions =====================
};
#endif
