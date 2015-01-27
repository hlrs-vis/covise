/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetin__configDir.h"
#include <string.h>

/// read from file
coTetin__configDir::coTetin__configDir(istream &str, int binary,
                                       ostream &ostr)
    : coTetinCommand(coTetin::CONFIGDIR_NAME)
{
    if (binary)
    {
    }
    else
    {
        char buffer[2000];
        getLine(buffer, 2000, str);
        buffer[1999] = '\0'; /// make sure it terminates
        name = new char[strlen(buffer) + 1];
        strcpy(name, buffer);
    }
}

/// read from memory
coTetin__configDir::coTetin__configDir(int *&intDat, float *&floatDat, char *&charDat)
    : coTetinCommand(coTetin::CONFIGDIR_NAME)
{
    name = getString(charDat);
}

coTetin__configDir::coTetin__configDir(char *filename)
    : coTetinCommand(coTetin::CONFIGDIR_NAME)
{
    int len = (filename) ? (strlen(filename) + 1) : 1;
    name = new char[len];
    if (filename)
    {
        strcpy(name, filename);
    }
    else
    {
        name[0] = '\0';
    }
}

/// Destructor
coTetin__configDir::~coTetin__configDir()
{
    if (name)
        delete[] name;
    name = 0;
}

/// check whether Object is valid
int coTetin__configDir::isValid() const
{
    if (d_comm)
        return 1;
    else
        return 0;
}

/// count size required in fields
void coTetin__configDir::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    // command name + data
    numInt++;
    numChar += (name) ? (strlen(name) + 1) : 1;
}

/// put my data to a given set of pointers
void coTetin__configDir::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    // copy the command's name
    *intDat++ = d_comm;

    // copy the data
    if (name)
    {
        strcpy(charDat, name);
        charDat += strlen(name) + 1;
    }
    else
    {
        *charDat++ = '\0';
    }
}

/// print to a stream in Tetin format
void coTetin__configDir::print(ostream &str) const
{
    if (isValid())
        str << "config_dir_name " << (name ? name : '\0') << endl;
    else
        str << "// invalid config_dir_name command skipped" << endl;
}

// ===================== command-specific functions =====================
