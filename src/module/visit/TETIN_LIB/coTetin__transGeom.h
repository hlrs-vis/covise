/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__TRANSLATE_GEOM_H_
#define _CO_TETIN__TRANSLATE_GEOM_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 02.02.00

/**
 * Class coTetin__transGeom implements "translate/rotate geometry" command
 *
 */

class coTetin__transGeom : public coTetinCommand
{

public:
    enum enumType
    {
        ILLEGAL_TRANS = 0,
        ALL_TRANS,
        SURFACE_TRANS,
        CURVE_TRANS,
        PPOINT_TRANS,
        MPOINT_TRANS
    };

private:
    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__transGeom(const coTetin__transGeom &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__transGeom &operator=(const coTetin__transGeom &)
    {
        return *this;
    };

    /// Default constructor:
    coTetin__transGeom()
        : type(ILLEGAL_TRANS)
        , trans_family(0)
        , n_names(0)
        , names(0)
    {
        for (int i = 0; i < 3; i++)
        {
            trans_vec[i] = 0.0;
            for (int j = 0; j < 3; j++)
            {
                rot_matrix[i][j] = ((i == j) ? 1.0 : 0.0);
            }
        }
    }

    // ===================== the command's data =====================

    // value
    int type; // type: surface, curve, point
    int trans_family; // if <> 0: trans. family_names else geometry names
    int n_names;
    char **names;
    float trans_vec[3]; // translation vector
    float rot_matrix[3][3]; // rotation matrix

public:
    /// read from file
    coTetin__transGeom(istream &str, int binary, ostream &ostr = cerr);

    /// read from memory
    coTetin__transGeom(int *&intDat, float *&floatDat, char *&charDat);

    coTetin__transGeom(int type_in, int trans_family_in,
                       int n_names_in, const char **names_in,
                       float trans_vec_in[3], float rot_matrix_in[3][3]);

    /// Destructor
    virtual ~coTetin__transGeom();

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
