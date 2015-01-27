/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class Translations                    ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date:                                                 ++
// ++**********************************************************************/

#include "Translations.h"

////////////////// internal class Translations //////////////////////////////////

Translations::Translations()
    : name_("")
    , next_(NULL)
    , alias_(NULL)
    , numAlias_(0)
    , policy_(Translations::NONE)
{
}

Translations::Translations(const std::string &mod)
    : name_(mod)
    , next_(NULL)
    , alias_(NULL)
    , numAlias_(0)
    , emptyStr_("")
    , policy_(Translations::NONE)
{
}

void
Translations::add(const std::string &mod, const std::string &alias, const int pol)
{

    if (name_.empty())
        name_ = mod;

    if (mod == name_)
    {
        // add alias to array of aliasses
        std::string *tmp = new std::string[numAlias_ + 1];
        for (int i = 0; i < numAlias_; ++i)
            tmp[i] = alias_[i];
        delete[] alias_;
        tmp[numAlias_] = alias;
        ++numAlias_;
        alias_ = tmp;
        policy_ = pol;
        return;
    }
    else
    {
        if (next_ == NULL)
        {
            next_ = new Translations(mod);
        }
        next_->add(mod, alias, pol);
    }
}

const std::string &
Translations::getNameByAlias(const std::string &alias) const
{
    for (int i = 0; i < numAlias_; ++i)
    {
        if (alias_[i] == alias)
            return name_;
    }

    if (next_)
        return next_->getNameByAlias(alias);
    else
        return emptyStr_;
}

int
Translations::cnt(const std::string &nm) const
{
    if (name_ == nm)
        return numAlias_;

    if (next_ != NULL)
        return next_->cnt(nm);
    return 0;
}

const std::string *
Translations::findReplacement(const std::string &name) const
{
    if (name_ == name)
        return alias_;

    if (next_ != NULL)
        return next_->findReplacement(name);
    return 0;
}

void
Translations::clean()
{
    delete[] alias_;
    if (next_)
    {
        next_->clean();
        delete next_;
        next_ = 0;
    }
}

int
Translations::isName(const std::string &nm) const
{
    if (name_ == nm)
        return 1;

    else
    {
        if (next_ != NULL)
            return next_->isName(nm);
        else
            return 0;
    }
    return 0;
}

std::string
Translations::get(const std::string &mod, const int &i)
{
    if (name_ == mod)
        if (i < numAlias_)
            return alias_[i];
        else
            return emptyStr_;
    else
    {
        if (next_ != NULL)
            return next_->get(mod, i);
        else
            return emptyStr_;
    }
    return emptyStr_;
}

int
Translations::getPolicy(const std::string &mod)
{
    if (name_ == mod)
    {
        return policy_;
    }
    else
    {
        if (next_ != NULL)
            return next_->getPolicy(mod);
        else
            return Translations::NONE;
    }
    return Translations::NONE;
    ;
}

Translations::~Translations()
{
    clean();
}
