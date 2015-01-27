/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//Visenso GmbH
//2012
//
//$Id: TransFactory.cpp 2267 2012-07-03 12:09:43Z wlukutin $
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#include "TransFactory.h"

#include "BoostTranslator.h"

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

boost::shared_ptr<vtrans::ITranslator>
vtrans::TransFactory::createTranslator(const std::string &translatorType,
                                       const std::string &pathToDictionary,
                                       const std::string &dictionaryDomain,
                                       const std::string &locale)
{
    if (!translatorType.compare("Boost.Locale"))
    {
        return boost::shared_ptr<vtrans::BoostTranslator>(new vtrans::BoostTranslator(pathToDictionary, dictionaryDomain, locale));
    }
    else //return something, not null pointer
    {
        return boost::shared_ptr<vtrans::BoostTranslator>(new vtrans::BoostTranslator());
    }
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
