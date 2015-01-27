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
//$Id: PathTranslator.h 2413 2012-08-07 13:46:06Z wlukutin $
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#pragma once
#ifndef __PATHTRANSLATOR__
#define __PATHTRANSLATOR__

#include <string>
#include <boost/utility.hpp>

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

namespace vtrans
{
class PathTranslator;
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

/**
 * Helper class for augmentig an arbitrary path of a resource 
 * (e.g. image) with locale prfix.
 * 
 * Useful for storing different language specific resources in a 
 * subdirectories for each language.
 * 
 * \remarks Returns original path if localized file does not exist.
 * 
 */
class vtrans::PathTranslator : boost::noncopyable
{
public:
    /**
    * Static convenience method, to hide PathTranslation 
    * object construction and method calling.
    * \param locale for extracting the subdirectory prefix.
    * \param path to localize.
    * \return Localized path if localized file exists, or original 
    * path otherwise.
    */
    static std::string TranslatePath(const std::string &locale, const std::string &path);

    /**
     * Construction.
     * \param locale for extracting the subdirectory prefix.
     */
    PathTranslator(const std::string &locale);

    /**
     * Translates the given path to the locale specific path.
     * 
     * \param path to localize.
     * 
     * \return Localized path if localized file exists, or original 
     * path otherwise.
     */
    std::string TranslatePath(const std::string &path) const;

    /**
     * \return current locale prefix.
     */
    const std::string &GetPrefix() const
    {
        return prefix_;
    };

private:
    std::string StripLocale(const std::string &locale) const;

    static bool FileExists(const std::string &path);

    const char winSeparator_;
    const char posixSeparator_;

    std::string prefix_;
};

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#endif //__PATHTRANSLATOR__
