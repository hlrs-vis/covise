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
//$Id: ITranslator.h 2332 2012-07-27 08:53:26Z wlukutin $
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#pragma once
#ifndef __ITRANSLATOR_H__
#define __ITRANSLATOR_H__

#include <string>

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

namespace vtrans
{
class ITranslator;
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

/**
 * Interface to a generic translator class.
 */
class vtrans::ITranslator
{
public:
    /**
     * Voirtual destructor's declaration and definition.
     */
    virtual ~ITranslator(){};

    /**
     * Method for translating the message to the current locale.
     */
    virtual std::string translate(const std::string &msg) const = 0;

    /**
     * Method converting a string to upper case according to current locale.
     */
    virtual std::string toUpper(const std::string &msg) const = 0;

    /**
    * Method converting a string to lower case according to current locale.
    */
    virtual std::string toLower(const std::string &msg) const = 0;
};

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#endif //__ITRANSLATOR_H__
