/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGENTRY_H
#define COCONFIGENTRY_H

#include "coConfigConstants.h"
#include "coConfigEntryString.h"
#include "coConfigSchemaInfos.h"

#include <string>

#include <util/coTypes.h>

#include "coConfigEditorController.h"

namespace covise
{
    class coConfigEntry;

    typedef std::vector<std::unique_ptr<coConfigEntry>> coConfigEntryPtrList;

    class CONFIGEXPORT coConfigEntry : public Subject<coConfigEntry>
    {
        friend class coConfigXercesEntry;

    public:
        coConfigEntry() = default;
        coConfigEntry(const coConfigEntry&) = delete;
        coConfigEntry(coConfigEntry&&) = default;
        coConfigEntry& operator=(const coConfigEntry&) = delete;
        coConfigEntry& operator=(coConfigEntry&&) = default;
        virtual ~coConfigEntry() = default;

        coConfigEntryStringList getScopeList(const std::string &scope);
        coConfigEntryStringList getVariableList(const std::string &scope);
        void appendVariableList(coConfigEntryStringList &list, const std::string &scope);

        coConfigEntryString getValue(const std::string &variable, const std::string &scope);
        // coConfigEntryStringList getValues(const std::string & variable, std::string scope);

        const char *getEntry(const char *variable);

        bool setValue(const std::string &variable, const std::string &value,
                      const std::string &section);
        void addValue(const std::string &variable, const std::string &value,
                      const std::string &section);

        bool deleteValue(const std::string &variable, const std::string &section);
        bool deleteSection(const std::string &section);

        bool hasValues() const;

        const std::string &getPath() const;
        std::string getName() const;
        const char *getCName() const;
        const std::string &getConfigName() const;

        bool isList() const;
        bool hasChildren() const;

        void setReadOnly(bool ro);
        bool isReadOnly() const;

        static std::string &cleanName(std::string &name);

        coConfigSchemaInfos *getSchemaInfos();
        void setSchemaInfos(coConfigSchemaInfos *infos);

        void entryChanged();
        const coConfigEntryPtrList &getChildren() const;
        virtual void merge(const coConfigEntry *with);
        virtual coConfigEntry *clone() const = 0;

    protected:
        coConfigEntry(const coConfigEntry *entry);

        void setPath(const std::string &path);
        void makeSection(const std::string &section);

    private:
        bool matchingAttributes() const;
        bool matchingHost() const;
        bool matchingMaster() const;
        bool matchingArch() const;
        bool matchingRank() const;

        coConfigConstants::ConfigScope configScope;
        std::string configName;
        std::string path;

        bool isListNode = false;
        bool readOnly = false;

        coConfigEntryPtrList children;
        std::map<std::string, std::string> attributes;
        std::set<std::string> textNodes;

        coConfigSchemaInfos *schemaInfos = nullptr;
        std::string elementGroup;

        mutable char *cName = nullptr;
        mutable std::string name;
    };
}
#endif
