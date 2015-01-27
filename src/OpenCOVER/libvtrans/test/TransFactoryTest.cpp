/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//Visenso GmbH
//2012
//
//$Id: TransFactoryTest.cpp 2400 2012-08-06 14:57:45Z wlukutin $
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#include <boost/shared_ptr.hpp>

#include "gtest/gtest.h"

#include "TransFactory.h"
#include "ITranslator.h"

#include "vtrans.h"

#include "vtranstest.h"

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

namespace vtrans
{

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(TransFactoryTest, NoNullReturnTest)
{
    boost::shared_ptr<vtrans::ITranslator> translatorPtr = vtrans::TransFactory::createTranslator("SOMTHEING", "path", "domain", "language");
    ASSERT_TRUE(translatorPtr);
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(TransFactoryTest, BlackBoxReturningTest)
{
    boost::shared_ptr<vtrans::ITranslator> translatorPtr = vtrans::TransFactory::createTranslator("Boost.Locale", pathToLocales_, "translator", "en");

    std::string teststring("Teststring 2");
    std::string translatedMessage = translatorPtr->translate(teststring);
    ASSERT_FALSE(translatedMessage.compare("EN Test String 2")) << "Message returned for EN: " << translatedMessage;
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(TransFactoryTest, ConvenienceTranslatorMessage)
{
    ASSERT_EQ("EN Test String 2", vtrans::VTrans::translate("Boost.Locale", pathToLocales_, "translator", "en", "Teststring 2"));
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
