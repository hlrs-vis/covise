/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:   Object to store translations (1-1 mappings)          ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date:                                                 ++
// ++**********************************************************************/
#ifndef TRANSLATIONS_H
#define TRANSLATIONS_H

#include <string>

class Translations
{
public:
    static enum
    {
        NONE,
        TRANSLATIONS
    } pol_;

    Translations();
    Translations(const std::string &nm);
    void add(const std::string &nm, const std::string &alias, int pol = Translations::NONE);
    const std::string &getNameByAlias(const std::string &alias) const;
    int cnt(const std::string &nm) const;
    const std::string *findReplacement(const std::string &nm) const;
    std::string get(const std::string &mod, const int &i);
    int getPolicy(const std::string &mod);
    int isName(const std::string &nm) const;
    ~Translations();

private:
    Translations(const Translations &){};
    // no assignment
    const Translations &operator=(const Translations &t)
    {
        return t;
    };
    void clean();

    std::string name_;
    Translations *next_;
    std::string *alias_;
    int numAlias_;
    std::string emptyStr_;
    int policy_;
};
#endif
