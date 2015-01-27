/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//
//libvtrans
//
//A library for translating Visenso applications.
//
//Visenso GmbH
//2012
//
//$Id: vtrans.h 3468 2013-03-26 09:28:40Z wlukutin $
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#pragma once
#ifndef __VTRANS_H__
#define __VTRANS_H__

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#include <string>

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

namespace vtrans
{
class VTrans;
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

class vtrans::VTrans
{
public:
    /**
     * Convenience method to process a translation in one method call.
     *
     * Creates a translation object.
     * \param translatorType name of the desired translation object type. 
     * Use for example "Boost.Locale" for enabling Boost.Locale library 
     * in conjunction with GETTEXT message formatting for string to translate.
     * \param pathToDictionary path to message database.
     * \param dictionaryDomain additional parameter for message database.
     * \param locale to use, e.g. language and character encoding type.
     * \param message to translate.
     * \return translated message.
     */
    static std::string translate(const std::string &translatorType, const std::string &pathToDictionary, const std::string &dictionaryDomain, const std::string &locale, const std::string &message);
};

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#endif //__VTRANS_H__
