/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/coConfigLog.h>
#include <config/CoviseConfig.h>

#include <config/coConfig.h>

#include <iostream>
using namespace std;
using namespace covise;

#include <QString>

coCoviseConfig::coCoviseConfig()
{
}

coCoviseConfig::~coCoviseConfig() {}

/**
 * @brief Get a config entry.
 * The entry takes the form "{sectionname.}*entry, whereby sectionname is
 * section[:name]. This method returns the "value"-field of the config entry only.
 * @param entry The section to get the value from.
 * @param exists provide a pointer to a bool here to get feedback, if the value exists.
 * @return the value of the section, or an empty string if nothing found.
 */
std::string coCoviseConfig::getEntry(const std::string &entry, bool *exists)
    return getEntry("value", entry, "", exists);
}

/**
 * @brief Get a config entry.
 * The entry takes the form "{sectionname.}*entry, whereby sectionname is
 * section[:name]. This method returns the field given by "variable" of the config entry.
 * @param variable The variable requested.
 * @param entry The section to get the value from.
 * @param exists provide a pointer to a bool here to get feedback, if the value exists.
 * @return the value of the variable in the denoted section, or an empty string if nothing found.
 */

std::string coCoviseConfig::getEntry(const std::string &variable, const std::string &entry, bool *exists)
{
    return getEntry(variable, entry, "", exists);
}

/**
 * @brief Get a config entry.
 * The entry takes the form "{sectionname.}*entry, whereby sectionname is
 * section[:name]. This method returns the field given by "variable" of the config entry.
 * @param variable The variable requested.
 * @param entry The section to get the value from.
 * @param defaultValue The default return value if nothing is found.
 * @param exists provide a pointer to a bool here to get feedback, if the value exists.
 * @return the value of the variable in the denoted section, or the default value if nothing found.
 */

std::string coCoviseConfig::getEntry(const std::string &variable, const std::string &entry, const std::string &defaultValue, bool *exists)
{
    QString val = coConfig::getInstance()->getValue(QString::fromStdString(variable), QString::fromStdString(entry));

    if (val.isNull())
    {
        if (exists)
            *exists = false;
        return defaultValue;
    }

    if (exists)
        *exists = true;

    COCONFIGDBG("coCoviseConfig::getEntry info: " << entry.c_str() << "/" << variable.c_str() << " = "
                                                  << (!val.isNull() ? qPrintable(val) : "*NULL*"));

    return val.toStdString();
}

/**
 * @brief Get an integer value
 * @param entry The section to get the value from.
 * @param defaultValue The default return value if nothing is found.
 * @param exists provide a pointer to a bool here to get feedback, if the value exists.
 * @return the value of the section, or the default value if nothing found.
 */
int coCoviseConfig::getInt(const std::string &entry, int defaultValue, bool *exists)
{
    return getInt("value", entry, defaultValue, exists);
}

/**
 * @brief Get an integer value
 * @param variable The variable requested.
 * @param entry The section to get the value from.
 * @param defaultValue The default return value if nothing is found.
 * @param exists provide a pointer to a bool here to get feedback, if the value exists.
 * @return the value of the variable in the denoted section, or the default value if nothing found.
 */
int coCoviseConfig::getInt(const std::string &variable, const std::string &entry, int defaultValue, bool *exists)
{
    COCONFIGDBG("coCoviseConfig::getInt info: enter " << entry.c_str() << "/" << variable.c_str() << " default " << defaultValue);
    coConfigInt val = coConfig::getInstance()->getInt(QString::fromStdString(variable), QString::fromStdString(entry));
    COCONFIGDBG("coCoviseConfig::getInt info: " << entry.c_str() << "/" << variable.c_str() << " = " << val
                                                << " (" << val.hasValidValue() << ")");
    if (exists)
        *exists = val.hasValidValue();

    if (val.hasValidValue())
        return val;
    else
        return defaultValue;
}

/**
 * @brief Get a long integer value
 * @param entry The section to get the value from.
 * @param defaultValue The default return value if nothing is found.
 * @param exists provide a pointer to a bool here to get feedback, if the value exists.
 * @return the value of the section, or the default value if nothing found.
 */
long coCoviseConfig::getLong(const std::string &entry, long defaultValue, bool *exists)
{
    return getLong("value", entry, defaultValue, exists);
}

/**
 * @brief Get a long integer value
 * @param variable The variable requested.
 * @param entry The section to get the value from.
 * @param defaultValue The default return value if nothing is found.
 * @param exists provide a pointer to a bool here to get feedback, if the value exists.
 * @return the value of the variable in the denoted section, or the default value if nothing found.
 */
long coCoviseConfig::getLong(const std::string &variable, const std::string &entry, long defaultValue, bool *exists)
{
    COCONFIGDBG("coCoviseConfig::getLong info: enter " << entry.c_str() << "/" << variable.c_str() << " default " << defaultValue);
    coConfigLong val = coConfig::getInstance()->getLong(QString::fromStdString(variable), QString::fromStdString(entry));
    COCONFIGDBG("coCoviseConfig::getLong info: " << entry.c_str() << "/" << variable.c_str() << " = " << val
                                                 << " (" << val.hasValidValue() << ")");
    if (exists)
        *exists = val.hasValidValue();

    if (val.hasValidValue())
        return val;
    else
        return defaultValue;
}

/**
 * @brief Checks if an option is on or off. As on counts "on", "true", and "1"
 * @param entry The section to get the value from.
 * @param defaultValue The default return value if nothing is found.
 * @param exists provide a pointer to a bool here to get feedback, if the value exists.
 * @return true, if "value" in the denoted section is on, or the default value if nothing found.
 */

bool coCoviseConfig::isOn(const std::string &entry, bool defaultValue, bool *exists)
{
    return isOn("value", entry, defaultValue, exists);
}

/**
 * @brief Checks if an option is on or off. As on counts "on", "true", and "1"
 * @param variable The variable requested.
 * @param entry The section to get the value from.
 * @param defaultValue The default return value if nothing is found.
 * @param exists provide a pointer to a bool here to get feedback, if the value exists.
 * @return true, if the variable in the denoted section is on, or the default value if nothing found.
 */
bool coCoviseConfig::isOn(const std::string &variable, const std::string &entry, bool defaultValue, bool *exists)
{
    COCONFIGDBG("coCoviseConfig::isOn info: enter " << entry.c_str() << "/" << variable.c_str() << " default " << defaultValue);
    coConfigBool val = coConfig::getInstance()->getBool(QString::fromStdString(variable), QString::fromStdString(entry));
    COCONFIGDBG("coCoviseConfig::isOn info: " << entry.c_str() << "/" << variable.c_str() << " = " << val
                                              << " (" << val.hasValidValue() << ")");
    if (exists)
        *exists = val.hasValidValue();

    if (val.hasValidValue())
        return val;
    else
        return defaultValue;
}

/**
 * @brief Get a float value
 * @param entry The section to get the value from.
 * @param defaultValue The default return value if nothing is found.
 * @param exists provide a pointer to a bool here to get feedback, if the value exists.
 * @return the value of the denoted section, or the default value if nothing found.
 */
float coCoviseConfig::getFloat(const std::string &entry, float defaultValue, bool *exists)
{
    return getFloat("value", entry, defaultValue, exists);
}

/**
 * @brief Get a float value
 * @param variable The variable requested.
 * @param entry The section to get the value from.
 * @param defaultValue The default return value if nothing is found.
 * @param exists provide a pointer to a bool here to get feedback, if the value exists.
 * @return the value of the variable in the denoted section, or the default value if nothing found.
 */
float coCoviseConfig::getFloat(const std::string &variable, const std::string &entry, float defaultValue, bool *exists)
{
    COCONFIGDBG("coCoviseConfig::getFloat info: enter " << entry.c_str() << "/" << variable.c_str() << " default " << defaultValue);
    coConfigFloat val = coConfig::getInstance()->getFloat(QString::fromStdString(variable), QString::fromStdString(entry));
    COCONFIGDBG("coCoviseConfig::getFloat info: " << entry.c_str() << " = " << val
                                                  << " (" << val.hasValidValue() << ")");
    if (exists)
        *exists = val.hasValidValue();

    if (val.hasValidValue())
        return val;
    else
        return defaultValue;
}

namespace
{

std::vector<std::string> getScopeNamesHelper(const coCoviseConfig::ScopeEntries &entries)
{
    std::vector<std::string> result;
    if (const char **values = entries.getValue())
    {
        for (const char **name = values; *name; name += 2)
        {
            if (const char *p = strchr(*name, ':'))
                result.push_back(p + 1);
            else
                result.push_back(*name);
        }
    }
    return result;
}
}

std::vector<std::string> coCoviseConfig::getScopeNames(const std::string &scope, const std::string &name)
{
    if (name.empty())
        return getScopeNamesHelper(getScopeEntries(scope));
    else
        return getScopeNamesHelper(getScopeEntries(scope, name));
}

// retrieve all values of a scope: return 2n+1 arr name/val/name.../val/NULL
coCoviseConfig::ScopeEntries coCoviseConfig::getScopeEntries(const std::string &scope)
{
    return ScopeEntries(scope.c_str(), 0);
}

// get all entries for one scope/name
coCoviseConfig::ScopeEntries coCoviseConfig::getScopeEntries(const std::string &scope, const std::string &name)
{
    return ScopeEntries(scope.c_str(), name.c_str());
}

// static char * findToken(const char *&tokenPtr)
// {

//    char *token;
//    int tokenLen = 0;

//    // find begin of token
//    while ( tokenPtr[0] && isspace(tokenPtr[0]) )
//    {
//       tokenPtr++;
//    }
//    if (!tokenPtr[0])
//       return(NULL);

//    int inQuote = 0;
//    int pos = 0;
//    while (tokenPtr[pos] && (!isspace(tokenPtr[pos]) || inQuote))
//    {

//       if ( (tokenPtr[pos]=='"') && ((pos == 0) || (tokenPtr[pos-1] != '\\' ) ) )
//          inQuote = !inQuote;
//       tokenLen++;
//       pos++;
//    }

//    // token umkopieren
//    token = new char[tokenLen+1];

//    inQuote = 0;
//    pos = 0;
//    int newLen = 0;

//    // solange nicht am ende && (nicht blank || innerhalb quote)
//    while (tokenPtr[pos] && (!isspace(tokenPtr[pos]) || inQuote))
//    {

//       if ( (tokenPtr[pos]=='"') && ((pos == 0) || (tokenPtr[pos-1] !='\\')) )
//       {
//          inQuote = !inQuote;

//       }
//       else
//       {
//          if ( (tokenPtr[pos] !='\\') || (tokenPtr[pos+1] != '"') )
//          {
//             token[newLen] = tokenPtr[pos];
//             newLen++;
//          }
//       }
//       pos++;
//    }

//    token[newLen] = '\0';

//    tokenPtr += tokenLen;

//    return(token);
// }

// int coCoviseConfig::getTokens(const char * entry, char **& tokens)
// {

//    int numTokens = 0;
//    const char * line;
//    char * token;
//    int i;

//    const char * tokenPtr;

//    Entry e = coCoviseConfig::getEntry(entry);
//    line = e.getValue();

//    if (line)
//    {

//       // find number of tokens

//       tokenPtr = line;

//       token = findToken(tokenPtr);
//       while (token)                               // =! '\0'
//       {
//          delete[] token;
//          numTokens++;
//          token = findToken(tokenPtr);

//       }

//       tokenPtr = line;

//       tokens = new char*[numTokens];

//       for (i = 0; i < numTokens; i++)
//       {

//          token = findToken(tokenPtr);

//          tokens[i] = new char[strlen(token)+1];
//          strcpy(tokens[i], token);
//          delete[] token;
//       }
//    }
//    else
//    {
//       numTokens = -1;
//    }
//    return(numTokens);
// }

coCoviseConfig::ScopeEntries::ScopeEntries(const char *scope, const char *name)
{

    coConfigEntryStringList list;
    if (name)
    {
        list = coConfig::getInstance()->getScopeList(scope, name);
    }
    else
    {
        list = coConfig::getInstance()->getScopeList(scope);
    }

    COCONFIGDBG(QString("coCoviseConfig::ScopeEntries::<init>(%1,%2): size = %3").arg(scope, QString(name), qPrintable(QString::number(list.size()))));

    if (list.isEmpty())
        return;

    QString scopeEntry(scope);
    scopeEntry.append(".%1");

    char **ptr_ = new char *[list.size() * 2 + 1];
    int ctr = 0;

    if (list.getListType() == coConfigEntryStringList::PLAIN_LIST)
    {
        COCONFIGDBG("coCoviseConfig::getScopeEntries info: PLAIN_LIST");
        for (coConfigEntryStringList::iterator i = list.begin(); i != list.end(); ++i)
        {
            QString value = (*i).section(' ', 0, 0);
            if (!value.isNull())
            {
                ptr_[ctr] = new char[strlen(value.toLatin1().constData()) + 1];
                strcpy(ptr_[ctr], value.toLatin1().constData());
            }
            else
            {
                ptr_[ctr] = 0;
            }
            ++ctr;

            value = (*i).section(' ', 1);
            if (!value.isNull())
            {
                ptr_[ctr] = new char[strlen(value.toLatin1().constData()) + 1];
                strcpy(ptr_[ctr], value.toLatin1().constData());
            }
            else
            {
                ptr_[ctr] = 0;
            }
            ++ctr;

            //COCONFIGLOG("coCoviseConfig::getScopeEntries info: " << rv_[ctr - 2] << " = " << rv_[ctr - 1]);
        }
    }
    else if (list.getListType() == coConfigEntryStringList::VARIABLE)
    {
        COCONFIGDBG("coCoviseConfig::getScopeEntries info: VARIABLE");
        for (coConfigEntryStringList::iterator i = list.begin(); i != list.end(); ++i)
        {
            QString value(*i);
            ptr_[ctr] = new char[strlen(value.toLatin1().constData()) + 1];
            strcpy(ptr_[ctr], value.toLatin1().constData());
            ++ctr;
            //Wenn ich das recht verstanden habe,
            //wird hier eine Liste aufgebaut, wobei
            // an den geradzahligen Positionen ( bei Zaehlung vn Null )
            //der "Variablenname"
            //und an den ungeradzahligen Positionen der
            //jeweilige Wert steht.

            value = coConfig::getInstance()->getValue(scopeEntry.arg(*i));
            if (!value.isNull())
            {
                ptr_[ctr] = new char[strlen(value.toLatin1().constData()) + 1];
                strcpy(ptr_[ctr], value.toLatin1().constData());
            }
            else
            {
                ptr_[ctr] = 0;
            }
            ++ctr;
        }
    }
    else
    {
        COCONFIGLOG("coCoviseConfig::getScopeEntries warn: UNKNOWN");
    }

    ptr_[ctr] = 0;

    ptr = const_cast<const char **>(ptr_);
}

void coCoviseConfig::ScopeEntries::release()
{
    if (ptr)
    {
        if (--refCount == 0)
        {
            COCONFIGLOG("coCoviseConfig::ScopeEntries::release info: destroying ptr");
            for (unsigned int ctr = 0; ptr[ctr]; ++ctr)
            {
                delete[] ptr[ctr];
                delete[] ptr[++ctr];
            }
            delete[] ptr;
            ptr = 0;
        }
    }
}

const char **coCoviseConfig::ScopeEntries::getValue()
{
    return RefPtr<const char **>::getValue();
}

const char **coCoviseConfig::ScopeEntries::getValue() const
{
    return RefPtr<const char **>::getValue();
}

coCoviseConfig::ScopeEntries &coCoviseConfig::ScopeEntries::operator=(const ScopeEntries &s)
{
    RefPtr<const char **>::operator=(s);
    return *this;
}
