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
//$Id: BoostTranslator.h 2518 2012-08-31 13:57:07Z wlukutin $
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#pragma once
#ifndef __BOOSTTRANSLATOR_H__
#define __BOOSTTRANSLATOR_H__

#ifdef _GTEST_ENABLED
#include <gtest/gtest_prod.h>
#endif

#include <locale>
#include <boost/locale.hpp>

#include "ITranslator.h"

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

namespace vtrans
{
class BoostTranslator;
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

class vtrans::BoostTranslator : public vtrans::ITranslator
{
public:
    /**
     * Friend classes.
     */
    friend class TransFactory;

    /**
     * Destruction.
     */
    virtual ~BoostTranslator();

    /**
     * Implementation of ITranslator's interface.
     */
    virtual std::string translate(const std::string &msg) const;

    /**
     * Method converting a string to upper case according to current locale.
     */
    virtual std::string toUpper(const std::string &msg) const;

    /**
     * Method converting a string to lower case according to current locale.
     */
    virtual std::string toLower(const std::string &msg) const;

    /*
     * ToDo: 
     * It's possible that we will need following method.
     * It should return an empty string if the msg ist not fount in 
     * translation database.
     * 
    std::string translateIfPresent(const std::string& msg) const;
    */

private:
#ifdef _GTEST_ENABLED
    FRIEND_TEST(BoostTranslatorTest, DefaultConfigurationTest);
    FRIEND_TEST(BoostTranslatorTest, DeConfigurationTest);
    FRIEND_TEST(BoostTranslatorTest, EnConfigurationTest);
    FRIEND_TEST(BoostTranslatorTest, ReConfigurationTest);
    FRIEND_TEST(BoostTranslatorTest, ParametrizedConstructionTest);
    FRIEND_TEST(BoostTranslatorTest, TwoTranslatorsSimultaneouslyTest);
    FRIEND_TEST(BoostTranslatorTest, ToUpperTest);
    FRIEND_TEST(BoostTranslatorTest, ToLowerTest);
    FRIEND_TEST(BoostTranslatorTest, EmptyStringReturnTest);
    FRIEND_TEST(BoostTranslatorTest, SingleSpaceReturnTest);
#endif

    BoostTranslator();
    BoostTranslator(const std::string &path, const std::string &domain, const std::string &language);
    void configureLanguage(const std::string &path, const std::string &domain, const std::string &language);

    boost::locale::generator generator_;
    std::locale locale_;
};

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#endif //__BOOSTTRANSLATOR_H__
