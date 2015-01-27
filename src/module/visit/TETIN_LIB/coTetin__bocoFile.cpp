/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetin__bocoFile.h"
#include <string.h>

/// read from file
coTetin__bocoFile::coTetin__bocoFile(istream &str, int binary)
    : coTetinCommand(coTetin::BOCO_FILE)
{
    if (binary)
    {
    }
    else
    {
        char buffer[512];
        getLine(buffer, 512, str);
        buffer[511] = '\0'; /// make sure it terminates
        d_boco = new char[strlen(buffer) + 1];
        strcpy(d_boco, buffer);
    }
}

coTetin__bocoFile::coTetin__bocoFile(
    char *boco_file)
    : coTetinCommand(coTetin::BOCO_FILE)
{
    int len = (boco_file) ? (strlen(boco_file) + 1) : 1;
    d_boco = new char[len];
    if (boco_file)
    {
        strcpy(d_boco, boco_file);
    }
    else
    {
        d_boco[0] = '\0';
    }
}

/// read from memory
coTetin__bocoFile::coTetin__bocoFile(int *&, float *&, char *&charDat)
    : coTetinCommand(coTetin::BOCO_FILE)
{
    d_boco = getString(charDat);
}

/// Destructor
coTetin__bocoFile::~coTetin__bocoFile()
{
    delete d_boco;
}

/// check whether Object is valid
int coTetin__bocoFile::isValid() const
{
    if (d_comm && d_boco && *d_boco)
        return 1;
    else
        return 0;
}

/// count size required in fields
void coTetin__bocoFile::addSizes(int &numInt, int &, int &numChar) const
{
    // command name
    numInt++;

    numChar += strlen(d_boco) + 1;
}

/// put my data to a given set of pointers
void coTetin__bocoFile::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    // copy the command's name
    *intDat++ = d_comm;

    // copy the data
    strcpy(charDat, d_boco);
    charDat += strlen(d_boco) + 1;
}

/// print to a stream in Tetin format
void coTetin__bocoFile::print(ostream &str) const
{
    if (isValid())
        str << "boco_file " << d_boco << endl;
    else
        str << "// invalid boco_file command skipped" << endl;
}

// ===================== command-specific functions =====================
