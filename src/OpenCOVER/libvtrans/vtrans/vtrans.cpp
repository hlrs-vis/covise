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
//$Id: vtrans.cpp 3468 2013-03-26 09:28:40Z wlukutin $
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#include "ITranslator.h"
#include "TransFactory.h"
#include "vtrans.h"

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

std::string vtrans::VTrans::translate(const std::string &translatorType, const std::string &pathToDictionary, const std::string &dictionaryDomain, const std::string &locale, const std::string &message)
{
    boost::shared_ptr<vtrans::ITranslator> translatorPtr = vtrans::TransFactory::createTranslator(translatorType, pathToDictionary, dictionaryDomain, locale);
    return translatorPtr->translate(message);
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
