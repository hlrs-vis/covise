/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_COVISE_CONFIG_H
#define CO_COVISE_CONFIG_H
#include <util/coTypes.h>

#include <map>
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
    typedef std::map<std::string, std::string> ScopeEntries;

    static std::string getEntry(const std::string &entry, bool *exists = nullptr);
    static std::string getEntry(const std::string &variable, const std::string &entry, bool *exists = nullptr);
    static std::string getEntry(const std::string &variable, const std::string &entry, const std::string &defaultValue, bool *exists = nullptr);

    static int getInt(const std::string &entry, int defaultValue, bool *exists = nullptr);
    static int getInt(const std::string &variable, const std::string &entry, int defaultValue, bool *exists = nullptr);

    static long getLong(const std::string &entry, long defaultValue, bool *exists = nullptr);
    static long getLong(const std::string &variable, const std::string &entry, long defaultValue, bool *exists = nullptr);

    static bool isOn(const std::string &entry, bool defaultValue, bool *exists = nullptr);
    static bool isOn(const std::string &variable, const std::string &entry, bool defaultValue, bool *exists = nullptr);

    // get float value of "Scope.Name"
    static float getFloat(const std::string &entry, float defaultValue, bool *exists = nullptr);
    static float getFloat(const std::string &variable, const std::string &entry, float defaultValue, bool *exists = nullptr);

    // retrieve all names of a scope
    static std::vector<std::string> getScopeNames(const std::string &scope, const std::string &name = "");

    // get all entries for one scope/name
    // ScopeEntries is reference counted, its contents are valid, as long a reference to
    // the object exists. Thus, do not use getScopeEntries().getValue() directly.
    static ScopeEntries getScopeEntries(const std::string &scope, const std::string &name = "");

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
};
}
#endif
