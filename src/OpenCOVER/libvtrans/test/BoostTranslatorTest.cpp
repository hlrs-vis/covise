/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//Visenso GmbH
//2012
//
//$Id: BoostTranslatorTest.cpp 2518 2012-08-31 13:57:07Z wlukutin $
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#include "gtest/gtest.h"

#include "BoostTranslator.h"

#include "vtranstest.h"

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

namespace vtrans
{
class BoostTranslatorTest : public ::testing::Test
{
};

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(BoostTranslatorTest, DefaultConfigurationTest)
{
    std::string msg("This is a test message.");
    vtrans::BoostTranslator boostTranslator;
    std::string translatedMessage = boostTranslator.translate(msg);
    ASSERT_FALSE(translatedMessage.compare(msg));
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(BoostTranslatorTest, ParametrizedConstructionTest)
{
    std::string msg("Teststring 2");
    vtrans::BoostTranslator boostTranslator(pathToLocales_, "translator", "en.UTF-8");
    std::string translatedMessage = boostTranslator.translate(msg);
    ASSERT_FALSE(translatedMessage.compare("EN Test String 2"));
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(BoostTranslatorTest, DeConfigurationTest)
{
    std::string teststring1("Teststring 1");
    vtrans::BoostTranslator boostTranslator;
    boostTranslator.configureLanguage(pathToLocales_, "translator", "de.UTF-8");

    std::string translatedMessage = boostTranslator.translate(teststring1);

    ASSERT_FALSE(translatedMessage.compare("Teststring 1"));
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(BoostTranslatorTest, EnConfigurationTest)
{
    std::string teststring1("Teststring 1");
    vtrans::BoostTranslator boostTranslator;
    boostTranslator.configureLanguage(std::string(pathToLocales_), std::string("translator"), std::string("en.UTF-8"));

    std::string translatedMessage = boostTranslator.translate(teststring1);

    ASSERT_FALSE(translatedMessage.compare("Test String 1")) << "Message returned: " << translatedMessage;
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(BoostTranslatorTest, ReConfigurationTest)
{
    std::string teststring2("Teststring 2");
    vtrans::BoostTranslator boostTranslator;

    boostTranslator.configureLanguage(std::string(pathToLocales_), std::string("translator"), std::string("en.UTF-8"));
    std::string translatedMessage1 = boostTranslator.translate(teststring2);

    ASSERT_FALSE(translatedMessage1.compare("EN Test String 2")) << "Message returned: " << translatedMessage1;

    boostTranslator.configureLanguage(std::string(pathToLocales_), std::string("translator"), std::string("de.UTF-8"));
    std::string translatedMessage2 = boostTranslator.translate(teststring2);

    ASSERT_FALSE(translatedMessage2.compare("DE Teststring 2")) << "Message returned: " << translatedMessage2;
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(BoostTranslatorTest, TwoTranslatorsSimultaneouslyTest)
{
    std::string teststring("Teststring 2");

    vtrans::BoostTranslator boostTranslator1(pathToLocales_, "translator", "de.UTF-8");
    vtrans::BoostTranslator boostTranslator2(pathToLocales_, "translator", "en.UTF-8");

    std::string translatedMessage1 = boostTranslator1.translate(teststring);
    std::string translatedMessage2 = boostTranslator2.translate(teststring);

    ASSERT_FALSE(translatedMessage1.compare("DE Teststring 2")) << "Message returned for DE: " << translatedMessage1;
    ASSERT_FALSE(translatedMessage2.compare("EN Test String 2")) << "Message returned for EN: " << translatedMessage2;
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(BoostTranslatorTest, ToUpperTest)
{
    std::string msg("gruessen");
    vtrans::BoostTranslator boostTranslator(pathToLocales_, "translator", "de.UTF-8");
    std::string translatedMessage = boostTranslator.translate(msg);
    ASSERT_FALSE(translatedMessage.compare("grüßen"));

    std::string toUpperMessage = boostTranslator.toUpper(translatedMessage);
#ifdef ICU_PRESENT
    ASSERT_FALSE(toUpperMessage.compare("GRÜSSEN")) << toUpperMessage;
#else
    ASSERT_FALSE(toUpperMessage.compare("GRÜßEN")) << toUpperMessage;
#endif
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(BoostTranslatorTest, ToLowerTest)
{
    std::string msg("GRUESSEN");
    vtrans::BoostTranslator boostTranslator(pathToLocales_, "translator", "de.UTF-8");
    std::string translatedMessage = boostTranslator.translate(msg);
    ASSERT_FALSE(translatedMessage.compare("GRÜSSEN"));

    std::string toLowerMessage = boostTranslator.toLower(translatedMessage);
    EXPECT_FALSE(toLowerMessage.compare("grüssen")) << toLowerMessage;
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

/*
    TEST(BoostTranslatorTest, TranslateIfPresentTest)
    {
        std::string teststringSense("Teststring 2");
        vtrans::BoostTranslator boostTranslator;
        
        boostTranslator.configureLanguage(std::string(pathToLocales_), std::string("translator"), std::string("en"));
        
        std::string translatedMessage = boostTranslator.translateIfPresent(teststringSense);
        ASSERT_FALSE(translatedMessage.compare("EN Test String 2")) << "Message returned: " << translatedMessage;
    }

    //--------------------------------------------------------------------------
    //--------------------------------------------------------------------------


    TEST(BoostTranslatorTest, DoNotTranslateIfNotPresentTest)
    {
        std::string teststringNonsense("Nonsense");
        vtrans::BoostTranslator boostTranslator;
        
        boostTranslator.configureLanguage(std::string(pathToLocales_), std::string("translator"), std::string("en"));
        
        std::string translatedMessage = boostTranslator.translate(teststringNonsense);
        ASSERT_TRUE(translatedMessage.empty()) << "Message returned not empty: " << translatedMessage;
    }
    */

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(BoostTranslatorTest, SingleSpaceReturnTest)
{
    std::string msg("Empty string");
    vtrans::BoostTranslator boostTranslator(pathToLocales_, "translator", "en.UTF-8");
    std::string translatedMessage = boostTranslator.translate(msg);
    ASSERT_FALSE(translatedMessage.compare(" "));
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(BoostTranslatorTest, EmptyStringReturnTest)
{
    std::string msg("Empty string");
    vtrans::BoostTranslator boostTranslator(pathToLocales_, "translator", "de.UTF-8");
    std::string translatedMessage = boostTranslator.translate(msg);
    //ToDo:
    //Make it return a really empty string!
    ASSERT_FALSE(translatedMessage.compare("Empty string"));
}
}
