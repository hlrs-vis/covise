/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//
// nvtrans
//
// A managed wrapper for
// vtrans library for
// translating Visenso applications.
//
// Visenso GmbH
// 2013
//
// $Id: NvTranslator.h 3459 2013-03-25 09:08:46Z wlukutin $
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#pragma once

#include "../vtrans/vtrans.h"
#include "../vtrans/PathTranslator.h"

using namespace System::Reflection;
//assembly:AssemblyVersion("1.0.0.*"),
//[assembly:AssemblyKeyFile("nvtrans.snk")];

[assembly:AssemblyVersion("0.1.0.0")];

using namespace System;

//--------------------------------------------------------------------------

namespace NvTrans
{
/// <summary>
/// Managed wrapper class for accessing translation
/// functionality from unmanaged vtrans lib.
/// </summary>
public
ref class NvTranslator
{
public:
    /// <summary>
    /// Default constructor.
    /// </summary>
    /// <param name="locale">
    /// Locale string to initialize the translation engine with.
    /// </param>
    NvTranslator(String ^ locale)
    {
        pathTranslator_ = new vtrans::PathTranslator(NvTranslator::StringToStd(locale));
    }

    /// <summary>
    /// Destructor.
    /// </summary>
    ~NvTranslator()
    {
        delete pathTranslator_;
        pathTranslator_ = NULL;
    }

    /// <summary>
    /// Translates a given path according to the locate
    /// this class was initialized with.
    /// </summary>
    /// <param name="path">
    /// Path to translate.
    /// </param>
    /// <returns>
    /// Translated path if the file under translated path exists,
    /// unmodified input path else.
    /// </returns>
    String ^ TranslatePath(String ^ path)
    {
        return NvTranslator::StdToString(pathTranslator_->TranslatePath(NvTranslator::StringToStd(path)));
    }

    /// <summary>
    /// Convinience method for the case you don't want to
    /// permanently keep an instance of the translator object.
    /// </summary>
    /// <param name="locale">
    /// Locale string to initialize the translation engine with.
    /// </param>
    /// <param name="path">
    /// Path to translate.
    /// </param>
    /// <returns>
    /// Translated path if the file under translated path exists,
    /// unmodified input path else.
    /// </returns>
    static String ^ TranslatePath(String ^ locale, String ^ path)
    {
        return NvTranslator::StdToString(vtrans::PathTranslator::TranslatePath(NvTranslator::StringToStd(locale),
                                                                               NvTranslator::StringToStd(path)));
    }

    /// <summary>
    /// Convenience method to process a string translation in one method call.
    /// </summary>
    /// <param name="translatorType">
    /// Name of the desired translation object type.
    /// Use for example "Boost.Locale" for enabling Boost.Locale library
    /// in conjunction with GETTEXT message formatting for string to translate.
    /// </param>
    /// <param name="pathToDictionary">
    /// Path to message database.
    /// </param>
    /// <param name="dictionaryDomain">
    /// Additional parameter for message database.
    /// For example use "vrml", if your file has the file name "vrml.mo"
    /// </param>
    /// <param name="locale">
    /// Language locale string, e.g. language and character encoding type "de.UTF-8".
    /// </param>
    /// <param name="message">
    /// Message to translate.
    /// </param>
    /// <returns>
    /// Translated message if database could be initialized and the message is present,
    /// original message else.
    /// </returns>
    static String ^ translate(String ^ translatorType,
                              String ^ pathToDictionary,
                              String ^ dictionaryDomain,
                              String ^ locale,
                              String ^ message)
    {
        return NvTranslator::StdToString(
            vtrans::VTrans::translate(
                NvTranslator::StringToStd(translatorType),
                NvTranslator::StringToStd(pathToDictionary),
                NvTranslator::StringToStd(dictionaryDomain),
                NvTranslator::StringToStd(locale),
                NvTranslator::StringToStd(message)));
    }

    private : NvTranslator(){};

    vtrans::PathTranslator *pathTranslator_;

    static std::string StringToStd(String ^ str)
    {
        const char *chars = (const char *)(Runtime::InteropServices::Marshal::StringToHGlobalAnsi(str)).ToPointer();
        std::string s = chars;
        Runtime::InteropServices::Marshal::FreeHGlobal(IntPtr((void *)chars));
        return s;
    }

    static String ^ StdToString(const std::string &stdstr)
    {
        String ^ str = gcnew String(stdstr.c_str());
        return str;
    }
};
}

//--------------------------------------------------------------------------