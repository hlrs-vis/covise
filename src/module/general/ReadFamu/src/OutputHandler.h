/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file OutputHandler.h
 * an output handler for displaying information on the screen.
 */

//#include "OutputHandler.h"  // an output handler for displaying information on the screen.

#ifndef __OutputHandler_h__
#define __OutputHandler_h__

#include <string>

/**
 * a container for results file data.
 */
class OutputHandler
{
public:
    OutputHandler(){};
    virtual ~OutputHandler(){};

    virtual void displayString(const char *s) = 0;

    virtual void displayError(const char *s) = 0;
    virtual void displayError(std::string s) = 0;
    virtual void displayError(std::string s, std::string strQuotes) = 0;
};

#endif
