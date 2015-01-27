/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//
// OpenCOVER kernel
//
//
// Visenso GmbH
// 2012
//
// $Id$
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#pragma once

#ifndef __CO_TRANSLATOR__
#define __CO_TRANSLATOR__

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#include <string>

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

/**
* Wrapper/convenience class for localization methods.
* Is responsible for reading out environemt variables, actual 
* configuration values, and calling translation library.
* To the COVER application it provides a aesy to use interface. 
*/
class COVEREXPORT coTranslator
{
public:
    /**
    * Translates the given string according to actual 
    * (environmental) settings.
    * \param msg String to translate.
    * \return Translated string, if retrieving setting was 
    * successfull, or the copy of the original string, if 
    * there is a misconfiguration of settings.  
    */
    static std::string coTranslate(const std::string &msg);
};

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#endif //__CO_TRANSLATOR__
