/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_COVISE_CONFIG_H
#define CO_COVISE_CONFIG_H
#include <util/coTypes.h>

#include <string>
#include <vector>

namespace covise
{

class CONFIGEXPORT coCoviseConfig
{

private:
    coCoviseConfig();
    virtual ~coCoviseConfig();

public:
    class ScopeEntries;

    static std::string getEntry(const std::string &entry, bool *exists = 0);
    static std::string getEntry(const std::string &variable, const std::string &entry, bool *exists = 0);
    static std::string getEntry(const std::string &variable, const std::string &entry, const std::string &defaultValue, bool *exists = 0);

    static int getInt(const std::string &entry, int defaultValue, bool *exists = 0);
    static int getInt(const std::string &variable, const std::string &entry, int defaultValue, bool *exists = 0);

    static long getLong(const std::string &entry, long defaultValue, bool *exists = 0);
    static long getLong(const std::string &variable, const std::string &entry, long defaultValue, bool *exists = 0);

    static bool isOn(const std::string &entry, bool defaultValue, bool *exists = 0);
    static bool isOn(const std::string &variable, const std::string &entry, bool defaultValue, bool *exists = 0);

    // get float value of "Scope.Name"
    static float getFloat(const std::string &entry, float defaultValue, bool *exists = 0);
    static float getFloat(const std::string &variable, const std::string &entry, float defaultValue, bool *exists = 0);

    // retrieve all names of a scope
    static std::vector<std::string> getScopeNames(const std::string &scope, const std::string &name = "");

    // retrieve all values of a scope: return 2n+1 arr name/val/name.../val/NULL
    // ScopeEntries is reference counted, its contents are valid, as long a reference to
    // the object exists. Thus, do not use getScopeEntries().getValue() directly.
    static ScopeEntries getScopeEntries(const std::string &scope);

    // get all entries for one scope/name
    // ScopeEntries is reference counted, its contents are valid, as long a reference to
    // the object exists. Thus, do not use getScopeEntries().getValue() directly.
    static ScopeEntries getScopeEntries(const std::string &scope, const std::string &name);

    // returns the number of tokens, returns -1 if entry is missing
    // puts the tokens into token array
    // examples:
    // XXXConfig
    //{
    //   ENTRY1 "aaa" "bbb"
    //   ENTRY2 aaa bbb
    //   ENTRY3 aaa"bbb"
    //}
    // returns
    // for ENTRY1 aaa and bbb
    // for ENTRY2 aaa and bbb
    // for entry3 aaabbb
    //   static int getTokens(const char *entry, char **&tokens);

    template <typename T>
    class RefPtr
    {

    public:
        RefPtr();
        RefPtr(const RefPtr<T> &s);
        RefPtr<T> &operator=(const RefPtr<T> &s);

        virtual ~RefPtr();

        T getValue();
        const T getValue() const;

    protected:
        virtual void release();

        T ptr;
        unsigned int refCount;
    };

    class CONFIGEXPORT ScopeEntries : public RefPtr<const char **>
    {
    public:
        ScopeEntries(const char *scope, const char *name);

        const char **getValue();
        const char **getValue() const;

        ScopeEntries &operator=(const ScopeEntries &s);

    private:
        virtual void release();
    };
};

template <typename T>
coCoviseConfig::RefPtr<T>::RefPtr()
{
    //COCONFIGLOG("coCoviseConfig::RefPtr<T>::<init> info: creating");
    ptr = 0;
    refCount = 1;
}

template <typename T>
coCoviseConfig::RefPtr<T>::~RefPtr()
{
    //COCONFIGLOG("coCoviseConfig::RefPtr<T>::<dest> info: destroying");
    release();
}

template <typename T>
coCoviseConfig::RefPtr<T>::RefPtr(const RefPtr<T> &s)
{
    if (ptr != s.ptr)
    {
        //COCONFIGLOG("coCoviseConfig::RefPtr<T>::<init> info: copying");
        ptr = s.ptr;
        refCount = s.refCount;
        if (ptr)
            ++refCount;
    }
}

template <typename T>
coCoviseConfig::RefPtr<T> &coCoviseConfig::RefPtr<T>::operator=(const RefPtr<T> &s)
{
    if (ptr != s.ptr)
    {
        //COCONFIGLOG("coCoviseConfig::RefPtr<T>::operator= info: copying");
        release();
        refCount = s.refCount;
        ptr = s.ptr;
        if (ptr)
            ++refCount;
    }
    return *this;
}

template <typename T>
T coCoviseConfig::RefPtr<T>::getValue()
{
    return ptr;
}

template <typename T>
const T coCoviseConfig::RefPtr<T>::getValue() const
{
    return ptr;
}

template <typename T>
void coCoviseConfig::RefPtr<T>::release()
{
    if (ptr)
    {
        if (--refCount == 0)
        {
            //COCONFIGLOG("coCoviseConfig::RefPtr<T>::release info: destroying ptr");
            delete[] ptr;
            ptr = 0;
        }
    }
}
}
#endif
