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
//$Id: TransFactory.h 2332 2012-07-27 08:53:26Z wlukutin $
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#pragma once
#ifndef __TRANS_FACTORY_H__
#define __TRANS_FACTORY_H__

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#include <boost/shared_ptr.hpp>
#include <string>

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

namespace vtrans
{
class TransFactory;
class ITranslator;
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

/**
 * Factory class for creation of translation objects.
 */
class vtrans::TransFactory
{
public:
    /**
     * Creates a translation object.
     * \param translatorType name of the desired translation object type. 
     * Use for example "Boost.Locale" for enabling Boost.Locale library 
     * in conjunction with GETTEXT message formatting for string to translate.
     * \param pathToDictionary path to message database.
     * \param dictionaryDomain additional parameter for message database.
     * \param locale to use, e.g. language and character encoding type.
     */
    static boost::shared_ptr<vtrans::ITranslator> createTranslator(const std::string &translatorType, const std::string &pathToDictionary, const std::string &dictionaryDomain, const std::string &locale);
};

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#endif //__TRANS_FACTORY_H__
