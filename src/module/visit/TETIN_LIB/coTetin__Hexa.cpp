/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetin__Hexa.h"
#include <string.h>

/// read from file
coTetin__Hexa::coTetin__Hexa(istream &str, int binary, ostream &ostr)
    : coTetinCommand(coTetin::START_HEXA)
{
    ;
}

/// read from memory
coTetin__Hexa::coTetin__Hexa(int *&intDat, float *&floatDat, char *&charDat)
    : coTetinCommand(coTetin::START_HEXA)
{
    tetin = getString(charDat);
    replay = getString(charDat);
    config_dir = getString(charDat);
    outp_intf = getString(charDat);
    write_blocking = *intDat++;
}

coTetin__Hexa::coTetin__Hexa(char *tetinf, char *replayf, char *config_d,
                             char *outp_intff, int write_bl)
    : coTetinCommand(coTetin::START_HEXA)
{
    int len = (tetinf) ? (strlen(tetinf) + 1) : 1;
    tetin = new char[len];
    if (tetinf)
    {
        strcpy(tetin, tetinf);
    }
    else
    {
        tetin[0] = '\0';
    }
    len = (replayf) ? (strlen(replayf) + 1) : 1;
    replay = new char[len];
    if (replayf)
    {
        strcpy(replay, replayf);
    }
    else
    {
        replay[0] = '\0';
    }
    len = (config_d) ? (strlen(config_d) + 1) : 1;
    config_dir = new char[len];
    if (config_d)
    {
        strcpy(config_dir, config_d);
    }
    else
    {
        config_dir[0] = '\0';
    }
    len = (outp_intff) ? (strlen(outp_intff) + 1) : 1;
    outp_intf = new char[len];
    if (outp_intff)
    {
        strcpy(outp_intf, outp_intff);
    }
    else
    {
        outp_intf[0] = '\0';
    }
    write_blocking = write_bl;
}

/// Destructor
coTetin__Hexa::~coTetin__Hexa()
{
    if (tetin)
        delete[] tetin;
    tetin = 0;
    if (replay)
        delete[] replay;
    replay = 0;
    if (config_dir)
        delete[] config_dir;
    config_dir = 0;
    if (outp_intf)
        delete[] outp_intf;
    outp_intf = 0;
    write_blocking = 0;
}

/// check whether Object is valid
int coTetin__Hexa::isValid() const
{
    if (d_comm)
        return 1;
    else
        return 0;
}

/// count size required in fields
void coTetin__Hexa::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    // command name, write_blocking + data
    numInt += 2;
    numChar += (tetin) ? (strlen(tetin) + 1) : 1;
    numChar += (replay) ? (strlen(replay) + 1) : 1;
    numChar += (config_dir) ? (strlen(config_dir) + 1) : 1;
    numChar += (outp_intf) ? (strlen(outp_intf) + 1) : 1;
}

/// put my data to a given set of pointers
void coTetin__Hexa::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    // copy the command's name
    *intDat++ = d_comm;
    *intDat++ = write_blocking;

    // copy the data
    if (tetin)
    {
        strcpy(charDat, tetin);
        charDat += strlen(tetin) + 1;
    }
    else
    {
        *charDat++ = '\0';
    }
    if (replay)
    {
        strcpy(charDat, replay);
        charDat += strlen(replay) + 1;
    }
    else
    {
        *charDat++ = '\0';
    }
    if (config_dir)
    {
        strcpy(charDat, config_dir);
        charDat += strlen(config_dir) + 1;
    }
    else
    {
        *charDat++ = '\0';
    }
    if (outp_intf)
    {
        strcpy(charDat, outp_intf);
        charDat += strlen(outp_intf) + 1;
    }
    else
    {
        *charDat++ = '\0';
    }
}

/// print to a stream in Tetin format
void coTetin__Hexa::print(ostream &str) const
{
    ;
}

// ===================== command-specific functions =====================
