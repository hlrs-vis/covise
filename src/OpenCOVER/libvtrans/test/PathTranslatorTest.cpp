/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//Visenso GmbH
//2012
//
//$Id: PathTranslatorTest.cpp 2415 2012-08-07 15:36:49Z wlukutin $
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#include "gtest/gtest.h"

#include "PathTranslator.h"

#include "vtranstest.h"

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

namespace vtrans
{

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

class PathTranslatorTest : public ::testing::Test
{
};

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(PathTranslatorTest, StripLocaleTestEmpty)
{
    std::string emptyLocale;
    PathTranslator pt(emptyLocale);
    ASSERT_EQ("", pt.GetPrefix());
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(PathTranslatorTest, StripLocaleTestArbitrary)
{
    std::string arbitrayLocale("de_DE.UTF-8");
    PathTranslator pt(arbitrayLocale);
    ASSERT_EQ("de_DE", pt.GetPrefix());
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(PathTranslatorTest, StripLocaleTestAdvanced1)
{
    std::string advancedLocale1("en_EN.UTF-8@calendar=test");
    PathTranslator pt(advancedLocale1);
    ASSERT_EQ("en_EN", pt.GetPrefix());
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(PathTranslatorTest, StripLocaleTestAdvanced2)
{
    std::string advancedLocale2("en_EN@calendar=test");
    PathTranslator pt(advancedLocale2);
    ASSERT_EQ("en_EN", pt.GetPrefix());
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(PathTranslatorTest, StripLocaleTestAdvanced3)
{
    std::string advancedLocale2("..\\locale\\en_EN.UTF-8@calendar=test");
    PathTranslator pt(advancedLocale2);
    ASSERT_EQ("../locale/en_EN", pt.GetPrefix());
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(PathTranslatorTest, StripLocaleTestAdvanced4)
{
    std::string advancedLocale2("..\\locale\\en_EN.UTF-8@calendar=test");
    PathTranslator pt(advancedLocale2);
    ASSERT_EQ("../locale/en_EN", pt.GetPrefix());
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(PathTranslatorTest, StripLocaleTestAdvanced5)
{
    std::string advancedLocale2("..\\en_EN.UTF-8@calendar=test");
    PathTranslator pt(advancedLocale2);
    ASSERT_EQ("../en_EN", pt.GetPrefix());
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(PathTranslatorTest, StripLocaleTestAdvanced6)
{
    std::string advancedLocale2("..\\locale");
    PathTranslator pt(advancedLocale2);
    ASSERT_EQ("../locale", pt.GetPrefix());
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(PathTranslatorTest, TranslatePathEmptyLocale)
{
    std::string emptyLocale;
    PathTranslator pt(emptyLocale);
#ifdef _WINDOWS
    ASSERT_EQ(("./test_dir/non_existing_file.txt"), pt.TranslatePath("./test_dir/non_existing_file.txt"));
#else
    ASSERT_EQ("./test_dir/non_existing_file.txt", pt.TranslatePath("./test_dir/non_existing_file.txt"));
#endif
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(PathTranslatorTest, TranslateAccordingToLocale)
{
    std::string specialLocale("fr_FR.UTF-8");
    PathTranslator pt(specialLocale);
#ifdef _WINDOWS
    ASSERT_EQ(".\\test_dir\\fr_FR\\test_file.txt", pt.TranslatePath("./test_dir/test_file.txt"));
#else
    ASSERT_EQ("./test_dir/fr_FR/test_file.txt", pt.TranslatePath("./test_dir/test_file.txt"));
#endif
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(PathTranslatorTest, DontTranslateIfNotExisting)
{
    std::string specialLocale("de.UTF-8");
    PathTranslator pt(specialLocale);
#ifdef _WINDOWS
    ASSERT_EQ("./test_dir/test_file.txt", pt.TranslatePath("./test_dir/test_file.txt"));
#else
    ASSERT_EQ("./test_dir/test_file.txt", pt.TranslatePath("./test_dir/test_file.txt"));
#endif
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(PathTranslatorTest, NotAlterPathIfNotExisting)
{
    std::string specialLocale("nz.UTF-8");
    PathTranslator pt(specialLocale);
#ifdef _WINDOWS
    ASSERT_EQ("./test_dir\\test_file.txt", pt.TranslatePath("./test_dir\\test_file.txt"));
#else
    ASSERT_EQ("./test_dir\\test_file.txt", pt.TranslatePath("./test_dir\\test_file.txt"));
#endif
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(PathTranslatorTest, ButifyPathIfExisting)
{
    std::string specialLocale("fr_FR.UTF-8");
    PathTranslator pt(specialLocale);
#ifdef _WINDOWS
    ASSERT_EQ(".\\test_dir\\fr_FR\\test_file.txt", pt.TranslatePath("./test_dir\\test_file.txt"));
#else
    ASSERT_EQ("./test_dir/fr_FR/test_file.txt", pt.TranslatePath("./test_dir\\test_file.txt"));
#endif
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(PathTranslatorTest, StaticConvinienceMethodOnExistingPath)
{
#ifdef _WINDOWS
    ASSERT_EQ(".\\test_dir\\fr_FR\\test_file.txt", PathTranslator::TranslatePath("fr_FR.UTF-8", "./test_dir\\test_file.txt"));
#else
    ASSERT_EQ("./test_dir/fr_FR/test_file.txt", PathTranslator::TranslatePath("fr_FR.UTF-8", "./test_dir\\test_file.txt"));
#endif
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(PathTranslatorTest, StaticConvinienceMethodOnNonExistingPath)
{
#ifdef _WINDOWS
    ASSERT_EQ("./test_dir\\non_file.txt", PathTranslator::TranslatePath("fr_FR.UTF-8", "./test_dir\\non_file.txt"));
#else
    ASSERT_EQ("./test_dir\\non_file.txt", PathTranslator::TranslatePath("fr_FR.UTF-8", "./test_dir\\non_file.txt"));
#endif
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(PathTranslatorTest, StaticConvinienceMethodOnNonExistingLocale)
{
#ifdef _WINDOWS
    ASSERT_EQ("./test_dir\\non_file.txt", PathTranslator::TranslatePath("nz.UTF-8", "./test_dir\\non_file.txt"));
#else
    ASSERT_EQ("./test_dir\\non_file.txt", PathTranslator::TranslatePath("nz.UTF-8", "./test_dir\\non_file.txt"));
#endif
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(PathTranslatorTest, StaticConvinienceMethodOnExistingPathWithSubdirectory)
{
#ifdef _WINDOWS
    ASSERT_EQ(".\\test_dir\\locale\\en_EN\\test_file.txt", PathTranslator::TranslatePath("locale\\en_EN.UTF-8", "./test_dir\\test_file.txt"));
#else
    ASSERT_EQ("./test_dir/locale/en_EN/test_file.txt", PathTranslator::TranslatePath("locale/en_EN.UTF-8", "./test_dir\\test_file.txt"));
#endif
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(PathTranslatorTest, StaticConvinienceMethodOnExistingPathWithRelativeSubdirectory1)
{
#ifdef _WINDOWS
    ASSERT_EQ(".\\test_dir\\.\\locale\\en_EN\\test_file.txt", PathTranslator::TranslatePath(".\\locale\\en_EN.UTF-8", "./test_dir\\test_file.txt"));
#else
    ASSERT_EQ("./test_dir/./locale/en_EN/test_file.txt", PathTranslator::TranslatePath("./locale/en_EN.UTF-8", "./test_dir\\test_file.txt"));
#endif
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(PathTranslatorTest, StaticConvinienceMethodOnExistingPathWithRelativeSubdirectory2)
{
#ifdef _WINDOWS
    ASSERT_EQ(".\\test_dir\\.\\fr_FR\\test_file.txt", PathTranslator::TranslatePath(".\\fr_FR.UTF-8", "./test_dir\\test_file.txt"));
#else
    ASSERT_EQ("./test_dir/./fr_FR/test_file.txt", PathTranslator::TranslatePath("./fr_FR.UTF-8", "./test_dir\\test_file.txt"));
#endif
}
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

TEST(PathTranslatorTest, StringRemovalTest)
{
    std::string st(";");

    ASSERT_NO_THROW(
        {
            if (st.length() > 0 && st[st.length()-1] == ';')
            {
                st.erase(st.length()-1);
            }
        });

    ASSERT_EQ(true, st.empty());
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
