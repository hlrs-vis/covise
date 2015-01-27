/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/coConfigImportReader.h>
#include <config/coConfigLog.h>

#include <QLinkedList>

#include <QtXml>
#include <QFile>
#include <QDir>
#include <QMap>
#include <QRegExp>
#include <QString>
#include <QStringList>
#include <QTextStream>

using namespace covise;

QDomNode coConfigImportReader::addChild(QDomNode &parent, const QDomNode &newChild)
{
    return parent.insertBefore(newChild, parent.firstChild());
}

coConfigImportReader::coConfigImportReader(const QString &aSource,
                                           const QString &aDest,
                                           const QString &aTransform,
                                           bool resolveIncludes)
{

    QString debugModeEnv = getenv("COCONFIG_DEBUG");
    if (!debugModeEnv.isNull())
    {
        unsigned int dl = debugModeEnv.toUInt();
        switch (dl)
        {
        case coConfig::DebugOff:
            coConfig::setDebugLevel(coConfig::DebugOff);
            break;
        case coConfig::DebugGetSets:
            coConfig::setDebugLevel(coConfig::DebugGetSets);
            break;
        default:
            coConfig::setDebugLevel(coConfig::DebugAll);
        }
    }
    else
    {
        coConfig::setDebugLevel(coConfig::DebugOff);
    }

    //cerr << "coConfigImportReader::<init> info: creating" << endl;

    source = new QFile(aSource);
    dest = new QFile(aDest);
    transform = new QFile(aTransform);

    sourceStream = 0;

    root = result.appendChild(result.createElement("YAC"));
    global = addChild(root, result.createElement("GLOBAL"));

    if (!this->source->exists())
    {
        COCONFIGLOG("coConfigImportReader::parse err: source file " << source->fileName() << " not found");
        return;
    }

    source->open(QIODevice::ReadOnly);

    if (!transform->exists())
    {
        COCONFIGLOG("coConfigImportReader::parse err: transform file " << transform->fileName() << " not found");
        return;
    }

    if (transformerDoc.setContent(transform, &errorMessage, &errorLine, &errorColumn))
    {
        COCONFIGLOG("coConfigImportReader::parse info: read transform file " << transform->fileName());
    }
    else
    {
        COCONFIGLOG("coConfigImportReader::parse err: error reading transform file "
                    << transform->fileName() << " in line "
                    << errorLine << ", column " << errorColumn << ":" << endl
                    << "   " << errorMessage);
    }

    transformer = transformerDoc.namedItem("TRANSFORM").namedItem("CoviseConfig").toElement();

    this->resolveIncludes = resolveIncludes;
}

coConfigImportReader::~coConfigImportReader()
{

    delete sourceStream;

    delete source;
    delete dest;
    delete transform;
}

QDomDocument coConfigImportReader::write()
{
    COCONFIGLOG("coConfigImportReader::parse info: writing destination file " << dest->fileName());
    dest->open(QIODevice::WriteOnly);
    QTextStream outStream(dest);
    QString resultString = domToString(result);
    outStream << resultString;
    dest->close();
    COCONFIGLOG("coConfigImportReader::parse info: done");
    return result;
}

QDomDocument coConfigImportReader::parse()
{

    COCONFIGLOG("coConfigImportReader::parse info: parsing " << source->fileName());

    errorLine = 0;
    errorColumn = 0;

    if (!this->source->exists())
    {
        COCONFIGLOG("coConfigImportReader::parse err: source file " << source->fileName() << " not found");
        return result;
    }

    if (!transform->exists())
    {
        COCONFIGLOG("coConfigImportReader::parse err: transform file " << transform->fileName() << " not found");
        return result;
    }

    sourceStream = new QTextStream(source);

    QString line;
    bool isList = false;

    while (!sourceStream->atEnd())
    {

        line = sourceStream->readLine().simplified();
        QString file;

        if (line.isEmpty() || line.startsWith("{") || line.startsWith("}"))
            continue;
        if (line.startsWith("<?xml"))
            continue;

        if (line.startsWith("<YAC") || line.startsWith("<COCONFIG"))
        {
            COCONFIGLOG("coConfigImportReader::parse info: reading xml config");
            source->close();
            source->open(QIODevice::ReadOnly);
            result.setContent(source);
            break;
        }

        if (line.startsWith("include") || line.startsWith("#include") || line.startsWith("tryinclude"))
        {
            if (line.startsWith("#include"))
            {
                file = line.simplified().section(' ', 1, 2);
            }
            else
            {
                file = line.simplified().section('"', 1, 2);
                if (file == line.simplified())
                    COCONFIGLOG("coConfigImportReader::parse err: unable to parse include directive " << line);
            }

            if (resolveIncludes)
            {
                QTextStream *savedStream = sourceStream;
                QFile *savedSource = source;

                QDir includeDir(file);

                if (includeDir.isRelative())
                {
                    QDir sdir(this->source->fileName());
                    QDir dir(sdir.canonicalPath());
                    dir.cdUp();
                    this->source = new QFile(dir.absoluteFilePath(file));
                }
                else
                {
                    this->source = new QFile(includeDir.canonicalPath());
                }

                if (this->source->exists())
                {
                    source->open(QIODevice::ReadOnly);
                    sourceStream = new QTextStream(source);
                    parse();
                }

                sourceStream = savedStream;
                source = savedSource;
            }
            else
            {
                QDomElement sectionNode = result.createElement("INCLUDE");
                addChild(sectionNode, result.createTextNode(file));
                addChild(root, sectionNode);
            }
            continue;
        }

        if (line.startsWith("#") || line.startsWith("//"))
        {
#if 0
         result.appendChild(result.createComment(line));
#endif
            continue;
        }

        QString section, hosts, list;

        isList = false;

        if (line.contains(':'))
        {
            section = line.section(':', 0, 0).simplified();
            QStringList hostList = line.section(':', 1).split(QRegExp("[,:\\s]"), QString::SkipEmptyParts);
            hosts = hostList.join(",");
            //COCONFIGLOG("coConfigImportReader::parse info: original (host only) section " << section);
        }
        else
        {
            section = line.section(' ', 0, 0).simplified();
            //COCONFIGLOG("coConfigImportReader::parse info: original section " << section);
        }

        QDomElement mapNode = findMapping(section, transformer);
        if (mapNode.isNull())
        {
            COCONFIGLOG("coConfigImportReader::parse warn: could not map section " << section);
            continue;
        }
        else
        {
            COCONFIGLOG("coConfigImportReader::parse info: creating section " << mapNode.nodeName());
        }

        if (mapNode.attribute("list") == "1")
        {
            list = "\n";
            isList = true;
        }

        while (!line.startsWith("{"))
        {
            line = sourceStream->readLine().simplified();
        }

        QDomElement sectionNode;

        do
        {

            line = sourceStream->readLine().simplified();

            if (line.isEmpty())
                continue;

            if (line.startsWith("#") || line.startsWith("//"))
                continue;

            if (line.startsWith("}"))
            {

                if (isList)
                {

                    if (!hosts.isEmpty())
                        sectionNode = getOrCreateSection(result, root, mapNode.nodeName(), hosts);
                    else
                        sectionNode = getOrCreateSection(result, global, mapNode.nodeName());

                    addChild(sectionNode, result.createTextNode(list));
                }

                continue;
            }

            if (line.contains("#"))
                line = line.section('#', 0, 0).trimmed();
            if (line.contains("//"))
                line = line.section("//", 0, 0).trimmed();

            if (isList)
            {
                list += "    " + line + "\n";
            }
            else
            {

                section = line.section(' ', 0, 0);

                QDomElement compatNode = findMapping(section, mapNode).toElement();
                if (compatNode.isNull())
                {
                    COCONFIGLOG("coConfigImportReader::parse warn: could not map entry " << section);
                    continue;
                }

                QString sectionName;

                if (!compatNode.attribute("move").isNull())
                    sectionName = compatNode.attribute("move");
                else
                    sectionName = mapNode.nodeName();

                if (!hosts.isEmpty())
                    sectionNode = getOrCreateSection(result, root, sectionName, hosts);
                else
                    sectionNode = getOrCreateSection(result, global, sectionName);

                if (compatNode.attribute("tag").isNull())
                {

                    if (!compatNode.attribute("attrib").isNull())
                    {
                        sectionNode.setAttribute(compatNode.attribute("attrib"),
                                                 line.section(' ', 1));
                    }

                    if (!compatNode.attribute("name").isNull())
                    {
                        sectionNode.setAttribute("name", compatNode.attribute(compatNode.attribute("name")));
                    }
                }
                else
                {

                    // Get tag
                    QString entryName = compatNode.attribute("tag");
                    QDomNode entryNode = sectionNode.namedItem(entryName);

                    if (entryNode.isNull() || compatNode.attribute("attrib").isNull())
                    {
                        entryNode = addChild(sectionNode, result.createElement(entryName));
                    }

                    if (!compatNode.attribute("attrib").isNull())
                    {

                        entryNode.toElement().setAttribute(compatNode.attribute("attrib"),
                                                           line.section(' ', 1));
                    }
                    else if (compatNode.attribute("list") == "1")
                    {

                        if (compatNode.attribute("mapping") != QString::null)
                        {
                            line = line.remove(0, compatNode.attribute("mapping").length());
                            line = line.trimmed();
                        }

                        QString list = "\n" + line.replace(QRegExp(" "), "\n") + "\n";
                        addChild(entryNode, result.createTextNode(list));

                        if (!compatNode.attribute("name").isNull())
                        {
                            QString name = entryNode.toElement().attribute(compatNode.attribute("name"));
                            if (name.isNull() || name.isEmpty())
                            {
                                QString tmpName(compatNode.attribute("name"));
                                entryNode.toElement().setAttribute("name", tmpName);
                            }
                            else
                                entryNode.toElement().setAttribute("name", name);
                        }
                    }
                    else
                    {

                        QString text = compatNode.text().simplified();
                        if (!text.isEmpty())
                        {

                            QString currentLine = line;
                            if (compatNode.attribute("mapping") != QString::null)
                            {
                                currentLine = currentLine.remove(0, compatNode.attribute("mapping").length());
                                currentLine = currentLine.trimmed();
                            }

                            if (!text.startsWith("%"))
                            {
                                currentLine = currentLine.remove(0, text.section('%', 0, 0).length() - 1);
                                currentLine = currentLine.trimmed();
                                text = text.remove(0, text.indexOf('%') - 1).trimmed();
                            }

                            QStringList tokens = text.split('%');
                            QStringList::iterator token = tokens.begin();

                            QString separator = *token++;
                            currentLine = currentLine.remove(0, separator.length());
                            currentLine = currentLine.trimmed();

                            while (token != tokens.end())
                            {

                                QString name = *token++;
                                separator = *token++;
                                QString value;

                                if (token == tokens.end())
                                {
                                    value = currentLine.trimmed();
                                }
                                else
                                {
                                    value = currentLine.section(separator, 0, 0);
                                    currentLine = currentLine.section(separator, 1);
                                    currentLine = currentLine.trimmed();
                                }

                                //value.replace(QRegExp("\""), "");

                                entryNode.toElement().setAttribute(name, value);
                            }

                            if (!compatNode.attribute("name").isNull())
                            {
                                QString name = entryNode.toElement().attribute(compatNode.attribute("name"));
                                if (name.isEmpty())
                                    entryNode.toElement().setAttribute("name", compatNode.attribute("name"));
                                else
                                    entryNode.toElement().setAttribute("name", name);
                            }
                        }
                    }

                    // If no tag present, try to map vartag to an attribute value
                    if (!compatNode.attribute("vartag").isNull())
                    {
                        entryNode.toElement().setTagName(entryNode.toElement().attribute(compatNode.attribute("vartag")));
                    }

                    if (!compatNode.attribute("enum").isNull())
                    {
                        int maxnum = -1;
                        QDomNodeList elements = sectionNode.elementsByTagName(entryNode.toElement().tagName());

                        if (elements.length() > 0)
                        {
                            for (unsigned int ctr = 0; ctr < elements.length(); ++ctr)
                            {
                                int num = elements.item(ctr).toElement().attribute("name").toInt();
                                if (num > maxnum)
                                    maxnum = num;
                            }
                        }

                        ++maxnum;
                        entryNode.toElement().setAttribute("name", QString::number(maxnum));
                    }
                }
            }
        } while (!line.startsWith("}"));

        COCONFIGLOG("coConfigImportReader::parse info: finishing section " << mapNode.nodeName());
    }

    return result;
}

QDomElement coConfigImportReader::findMapping(const QString &mapping,
                                              const QDomNode &parent) const
{

    //   cerr << "coConfigImportReader::findMapping info: searching for mapping "
    //        << mapping << endl;

    QDomNodeList children = parent.childNodes();

    for (unsigned int ctr = 0; ctr < (unsigned int)children.count(); ctr++)
    {
        if (children.item(ctr).isElement() && children.item(ctr).toElement().attribute("mapping") == mapping)
        {
            return children.item(ctr).toElement();
        }
    }
    COCONFIGDBG("coConfigImportReader::findMapping info: trying default mapping");

    if (mapping.isNull())
        return QDomElement();
    else
        return findMapping(QString::null, parent);
}

QDomElement coConfigImportReader::getOrCreateSection(QDomDocument &document, QDomNode &parent, const QString &name, const QString &hosts)
{

    QDomElement rv;

    if (!hosts.isEmpty())
    {
        if (!hostconfigs.contains(hosts))
        {
            QDomElement local = addChild(parent, result.createElement("LOCAL")).toElement();
            local.setAttribute("host", hosts);
            hostconfigs[hosts] = local;
            rv = getOrCreateSection(document, local, name);
        }
        else
        {
            QDomElement local = hostconfigs[hosts];
            rv = getOrCreateSection(document, local, name);
        }
    }
    else
    {
        if (parent.namedItem(name).isNull())
        {
            rv = document.createElement(name);
            addChild(parent, rv);
        }
        else
        {
            rv = parent.namedItem(name).toElement();
        }
    }

    return rv;
}

/**
 * @brief Updates the current result DOM tree from version 0 to version 1.
 * @note Has to be called after parse().
 *
 * Updater format:
 * <pre>
 * <TRANSFORM>
 *   <UpdateV0V1>
 *     <Entry>
 *      ...
 *   </UpdateV0V1>
 * </TRANSFORM>
 * </pre>
 */
void coConfigImportReader::updatev0v1()
{

    COCONFIGDBG("coConfigImportReader::updatev0v1 info: updating from version 0 to version 1");

    // Check if we really have a v0 config (a YAC tag is required)
    QDomElement rootNode = result.namedItem("YAC").toElement();
    if (rootNode.isNull())
    {
        COCONFIGLOG("coConfigImportReader::updatev0v1 info: no update needed");
        return;
    }

    rootNode.setTagName("COCONFIG");
    rootNode.setAttribute("version", 1.0);

    // Adding a valid processing instruction
    QDomProcessingInstruction pe = result.createProcessingInstruction("xml", "version=\"1.0\"");
    result.insertBefore(pe, result.firstChild());

    // Getting the transformer for V0 -> V1
    QDomElement updaterv0v1 = transformerDoc.namedItem("TRANSFORM").namedItem("UpdateV0V1").toElement();

    update(rootNode, updaterv0v1);
}

/**
 * @brief Updates a DOM tree using the instructions in <I>updater</I>
 * @param rootNode DOM tree to update.
 * @param updater update instructions.
 *
 * Updater format:
 * <pre>
 * <Entry>
 *   <Instruction>
 *   ...
 *   </Instruction>
 * </Entry>
 * <Entry>
 *   ...
 * </Entry>
 * ...
 * </pre>
 */

void coConfigImportReader::update(QDomElement &rootNode, QDomElement &updater)
{
    // Make a list of all entries
    QDomNodeList updateList = updater.childNodes();

    // Iterate through all update instructions
    for (unsigned int ctr = 0; ctr < updateList.length(); ++ctr)
    {
        QDomElement currentUpdate = updateList.item(ctr).toElement();
        QString name = currentUpdate.attribute("name");
        if (name.isNull())
            name = currentUpdate.attribute("index");
        COCONFIGDBG("coConfigImportReader::update info: updating " << name);
        QLinkedList<QDomNode> scopeNodeList = makeNonLiveList(rootNode.childNodes());

        // 2nd Level: GLOBAL and LOCAL scope nodes
        for (QLinkedList<QDomNode>::iterator scopeIt = scopeNodeList.begin(); scopeIt != scopeNodeList.end(); ++scopeIt)
        {
            QLinkedList<QDomNode> nodeList = makeNonLiveList((*scopeIt).childNodes());
            for (QLinkedList<QDomNode>::iterator nodeIt = nodeList.begin(); nodeIt != nodeList.end(); ++nodeIt)
            {
                QDomElement node = (*nodeIt).toElement();
                if (node.isNull())
                    continue;
                updateNode(node, currentUpdate);
            }
        }
    }
}

/**
 * @brief Update a node with an instruction set
 * @param node Node to update.
 * @param updateInstruction Instruction set.
 *
 * Instruction set format:
 * <pre>
 * <Instruction scope="AllChildren|Child|Self" [param="parameter"]>
 *   Instruction1
 *   Instruction2
 *   ...
 * </Instruction>
 * </pre>
 */

void coConfigImportReader::updateNode(QDomElement &node, QDomElement &updateInstruction)
{

    if (updateInstruction.tagName() == "Entry")
    {
        QRegExp pattern(updateInstruction.attribute("name"));
        if (pattern.exactMatch(node.tagName()))
        {
            //COCONFIGLOG("coConfigImportReader::updateNode info: updating " << node.tagName());
            QDomNodeList updateSteps = updateInstruction.childNodes();
            for (unsigned int ctr = 0; ctr < updateSteps.length(); ++ctr)
            {
                QDomElement entry = updateSteps.item(ctr).toElement();
                if (entry.isNull())
                    continue;
                updateEntry(node, entry);
            }
        }
    }
}

/**
 * @brief Apply a single instruction
 * @param node Node to update according to the instruction scope (AllChildren, Child or Self).
 * @param updateInstruction Instruction to apply.
 */

void coConfigImportReader::updateEntry(QDomElement &node, QDomElement &updateInstruction)
{

    QDomNodeList instructions = updateInstruction.childNodes();

    for (unsigned int ctr = 0; ctr < instructions.length(); ++ctr)
    {

        QDomElement instruction = instructions.item(ctr).toElement();
        if (instruction.isNull())
            continue;

        // Check the scope of the update instruction
        if (updateInstruction.attribute("scope") == "AllChildren")
        {
            COCONFIGDBG("coConfigImportReader::updateEntry info: applying to all children");
            QLinkedList<QDomNode> nodes = makeNonLiveList(node.childNodes());
            for (QLinkedList<QDomNode>::iterator nodeIt = nodes.begin(); nodeIt != nodes.end(); ++nodeIt)
            {
                QDomElement uNode = (*nodeIt).toElement();
                updateApplyInstruction(uNode, instruction);
            }
        }
        else if (updateInstruction.attribute("scope") == "Self")
        {
            COCONFIGDBG("coConfigImportReader::updateEntry info: applying to self");
            updateApplyInstruction(node, instruction);
        }
        else if (updateInstruction.attribute("scope") == "Child")
        {
            QRegExp childRegEx(updateInstruction.attribute("param"));
            QRegExp childNameRegEx(updateInstruction.attribute("param2"));

            COCONFIGDBG("coConfigImportReader::updateEntry info: applying to all children named "
                        << childRegEx.pattern()
                        << (childNameRegEx.pattern() == "" ? "" : QString(":%1").arg(childNameRegEx.pattern()).toLatin1()));

            QLinkedList<QDomNode> nodes = makeNonLiveList(node.childNodes());
            for (QLinkedList<QDomNode>::iterator nodeIt = nodes.begin(); nodeIt != nodes.end(); ++nodeIt)
            {
                QDomElement uNode = (*nodeIt).toElement();
                if (childRegEx.exactMatch(uNode.tagName()))
                {
                    if (childNameRegEx.pattern().isEmpty())
                        updateApplyInstruction(uNode, instruction);
                    else if (childNameRegEx.exactMatch(uNode.attribute("name")))
                        updateApplyInstruction(uNode, instruction);
                    else if (childNameRegEx.exactMatch(uNode.attribute("index")))
                        updateApplyInstruction(uNode, instruction);
                }
            }
        }
    }
}

/**
 * @brief Apply a single instruction
 * @param node Node to apply the instruction to.
 * @param instruction Instruction to apply. Please see the file config/transform.xml for a detailed documentation.
 */

void coConfigImportReader::updateApplyInstruction(QDomElement &node, QDomElement &instruction)
{

    // Move an entry
    if (instruction.tagName() == "Move")
    {
        QString target = instruction.attribute("to");
        COCONFIGDBG("coConfigImportReader::updateApplyInstruction info: Move " << node.tagName() << " to " << target);
        QDomNode parent = node.parentNode();
        QDomElement newSection;
        QDomNode insertReference = node;

        // Check for an absolute path
        if (target.startsWith("/"))
        {
            QDomNode level0 = node;
            QDomNode level1 = parent;
            QDomNode level2 = parent.parentNode();
            while (!level2.parentNode().isDocument())
            {
                level0 = level1;
                level1 = level2;
                level2 = level2.parentNode();
            }

            parent = level1;
            insertReference = level0;

            target = target.remove(0, 1);

            COCONFIGDBG("coConfigImportReader::updateApplyInstruction info: Moving absolute")
        }

        // Look if we have a composed entry like COVER.Tracking.Joystick that has to be disassembled
        QStringList sections = target.split('.', QString::KeepEmptyParts);

        for (QStringList::iterator section = sections.begin(); section != sections.end(); ++section)
        {

            newSection = parent.namedItem(*section).toElement();

            if (newSection.isNull())
            {
                COCONFIGDBG("coConfigImportReader::updateApplyInstruction info: creating new section " << *section);
                newSection = parent.ownerDocument().createElement(*section);
                parent.insertBefore(newSection, insertReference);
            }

            parent = newSection;
            insertReference = parent.firstChild();
        }

        // If the section already exists, merge the entry
        if (!newSection.namedItem(node.tagName()).isNull())
            updateMergeNodes(newSection, node);
        else
            newSection.appendChild(node);
    }
    // Convert a variable tag to a fixed tag used in some V0-entries
    else if (instruction.tagName() == "VarToFixedTag")
    {
        COCONFIGDBG("coConfigImportReader::updateApplyInstruction info: VarToFixedTag " << node.tagName());
        if (instruction.attribute("name") == "tag")
            node.setAttribute("name", node.tagName());
        node.setTagName(instruction.attribute("tag"));
    }
    // Split the value of a single attribute into many
    else if (instruction.tagName() == "Split")
    {
        COCONFIGDBG("coConfigImportReader::updateApplyInstruction info: Split " << node.tagName());
        QString splitWhat = instruction.attribute("attribute");
        QString splitFrom = node.attribute(splitWhat);

        QStringList splitKey = instruction.attribute("to").split(' ', QString::SkipEmptyParts);
        QStringList splitValue = splitFrom.split(' ', QString::SkipEmptyParts);

        node.removeAttribute(splitWhat);
        while (!splitKey.isEmpty() && !splitValue.isEmpty())
        {
            COCONFIGDBG("coConfigImportReader::updateApplyInstruction info: splitting " << splitKey.first() << "=" << splitValue.first());
            node.setAttribute(splitKey.first(), splitValue.first());
            splitKey.pop_front();
            splitValue.pop_front();
        }
    }
    // Merge the value of a child into the current node. Could maybe be replaced by MergeIntoAsAttribute
    else if (instruction.tagName() == "ChildToAttribute")
    {
        COCONFIGDBG("coConfigImportReader::updateApplyInstruction info: converting " << instruction.attribute("tag") << " to attribute " << instruction.attribute("attribute"));
        QDomElement child = node.namedItem(instruction.attribute("tag")).toElement();
        if (child.isNull())
            return;

        QString variable = instruction.attribute("variable");
        if (variable.isNull())
            variable = "value";

        node.setAttribute(instruction.attribute("attribute"), child.attribute(variable));

        child.removeAttribute(variable);

        if (child.attributes().length() == 0)
        {
            QDomElement i = instruction.ownerDocument().createElement("Remove");
            updateApplyInstruction(child, i);
        }
    }
    // Merge a tag as attribute into another tag
    else if (instruction.tagName() == "MergeIntoAsAttribute")
    {
        COCONFIGDBG("coConfigImportReader::updateApplyInstruction info: merging " << node.tagName() << " into " << instruction.attribute("into") << " as attribute " << instruction.attribute("attribute"));

        QDomElement parent = node.parentNode().toElement();

        QString target = instruction.attribute("into");
        QStringList sections = target.split('.', QString::KeepEmptyParts);
        QDomNode insertReference = node;

        QDomElement into;

        for (QStringList::iterator section = sections.begin(); section != sections.end(); ++section)
        {

            into = parent.namedItem(*section).toElement();

            if (into.isNull())
            {
                COCONFIGDBG("coConfigImportReader::updateApplyInstruction info: creating new tag " << *section);
                into = parent.ownerDocument().createElement(*section);
                parent.insertBefore(into, insertReference);
            }

            parent = into;
            insertReference = parent.firstChild();
        }

        QString variable = instruction.attribute("variable");
        if (variable.isNull())
            variable = "value";

        into.setAttribute(instruction.attribute("attribute"), node.attribute(variable));

        node.removeAttribute(variable);

        if (node.attributes().length() == 0)
        {
            QDomElement i = instruction.ownerDocument().createElement("Remove");
            updateApplyInstruction(node, i);
        }
    }
    // Issue a warning on stderr
    else if (instruction.tagName() == "Warn")
    {
        COCONFIGMSG("Warning: " << (!instruction.text().isNull() ? instruction.text() : ""));
    }
    // Issue an information on stderr
    else if (instruction.tagName() == "Info")
    {
        COCONFIGMSG("Information: " << (!instruction.text().isNull() ? instruction.text() : ""));
    }
    // Renames an entry, don't use together with Move
    else if (instruction.tagName() == "Rename")
    {
        COCONFIGDBG("coConfigImportReader::updateApplyInstruction info: Rename " << node.tagName() << " to " << instruction.attribute("to"));

        if (!node.parentNode().namedItem(instruction.attribute("to")).isNull())
        {
            COCONFIGLOG("coConfigImportReader::updateApplyInstruction warn: Rename " << node.tagName()
                                                                                     << ": entry " << instruction.attribute("to") << " already exists, trying to merge");
            QDomElement parent = node.parentNode().toElement();
            node.setTagName(instruction.attribute("to"));
            updateMergeNodes(parent, node);
            parent.removeChild(node);
        }
        else
            node.setTagName(instruction.attribute("to"));
    }
    // RenameAndMove concats Rename and Move to work around some internal difficulties (The node isn't found anymore if renamed or moved)
    else if (instruction.tagName() == "RenameAndMove")
    {
        QDomElement i = instruction.ownerDocument().createElement("Move");
        i.setAttribute("to", instruction.attribute("toSection"));
        updateApplyInstruction(node, i);
        i.setTagName("Rename");
        i.setAttribute("to", instruction.attribute("renameTo"));
        updateApplyInstruction(node, i);
    }
    // Delete the current node
    else if (instruction.tagName() == "Remove")
    {
        QDomNode parent = node.parentNode();
        parent.removeChild(node);
        while (!parent.hasChildNodes())
        {
            QDomNode child = parent;
            parent = parent.parentNode();
            parent.removeChild(child);
        }
    }
    // Adds a "coconfig-deprecated"-attribute to mark deprecated entries
    else if (instruction.tagName() == "Deprecate")
    {
        QString comment("This configuration entry is deprecated%1%2%3");

        if (!instruction.text().isEmpty())
            comment = comment.arg(", please use ", instruction.text(), " instead");
        else
            comment = comment.arg(".", "", "");

        QDomComment commentNode = node.ownerDocument().createComment(comment);
        node.insertBefore(commentNode, node.firstChild());

        node.setAttribute("coconfig:deprecated", "1");
    }
    // Simply add some attribute
    else if (instruction.tagName() == "SetAttribute")
    {
        node.setAttribute(instruction.attribute("attribute"), instruction.attribute("value"));
    }
    // Special handling for ARToolKitMarker in V0-configs
    else if (instruction.tagName() == "ARToolKitMarker")
    {
        QRegExp patternRE("(.*)(Pattern)");
        QRegExp sizeRE("(.*)(Size)");
        QRegExp offsetRE("(.*)(Offset)");
        QRegExp vrmlToPfRE("(.*)(VrmlToPf)");

        QRegExp matchedRE;
        QString split;

        if (patternRE.exactMatch(node.tagName()))
            matchedRE = patternRE;
        else if (sizeRE.exactMatch(node.tagName()))
            matchedRE = sizeRE;
        else if (offsetRE.exactMatch(node.tagName()))
        {
            matchedRE = offsetRE;
            split = "x y z h p t";
        }
        else if (vrmlToPfRE.exactMatch(node.tagName()))
            matchedRE = vrmlToPfRE;

        QString name = matchedRE.capturedTexts()[1];
        QString type = matchedRE.capturedTexts()[2];
        QString value = node.attribute("value");

        QDomElement parent = node.parentNode().toElement();
        QDomElement marker;

        QDomNodeList children = parent.childNodes();

        for (unsigned int ctr = 0; ctr < children.length(); ++ctr)
        {
            QDomElement child = children.item(ctr).toElement();
            if (child.isNull())
                continue;
            if (child.tagName() == "Marker")
            {
                if (child.attribute("name") == name)
                {
                    marker = child;
                    break;
                }
            }
        }

        if (marker.isNull())
        {
            marker = parent.ownerDocument().createElement("Marker");
            marker.setAttribute("name", name);
            parent.insertBefore(marker, node);
        }

        QDomElement entry = marker.ownerDocument().createElement(type);
        entry.setAttribute("value", value);
        marker.appendChild(entry);

        if (!split.isEmpty())
        {
            QDomElement i = instruction.ownerDocument().createElement("Split");
            i.setAttribute("attribute", "value");
            i.setAttribute("to", split);
            updateApplyInstruction(entry, i);
        }

        parent.removeChild(node);
    }
    else
    {
        COCONFIGLOG("coConfigImportReader::updateApplyInstruction warn: unknown instruction '" << instruction.tagName() << "'");
    }
}

/**
 * @brief Merges two tags
 * @param section Section where the target node resides.
 * @param node Node to merge.
 */
void coConfigImportReader::updateMergeNodes(QDomElement &section, QDomElement &node)
{

    COCONFIGDBG("coConfigImportReader::updateMergeNodes info: merging nodes in section " << section.tagName());

    QDomElement targetNode = section.namedItem(node.tagName()).toElement();

    if (node.hasChildNodes())
    {
        QDomNodeList sourceChildren = node.childNodes();

        while (sourceChildren.length())
        {
            QDomElement child = sourceChildren.item(0).toElement();
            if (child.isNull())
            {
                node.removeChild(sourceChildren.item(0));
                continue;
            }

            if (!targetNode.namedItem(child.tagName()).isNull())
            {
                updateMergeNodes(targetNode, child);
                node.removeChild(child);
            }
            else
                targetNode.appendChild(child);
        }
    }

    QDomNamedNodeMap attributes = node.attributes();

    for (unsigned int ctr = 0; ctr < attributes.length(); ++ctr)
    {
        QDomAttr attribute = attributes.item(ctr).toAttr();
        targetNode.setAttribute(attribute.name(), attribute.value());
    }
}

/**
 * @brief Converts a QDomNodeList into a QValueList&lt;QDomNode&gt;
 * @param liveList The QDomNodeList to convert
 * @return The non-live list
 *
 * This has to be done as a QDomNodeList changes as nodes are added to / removed from the tree.
 */
QLinkedList<QDomNode> coConfigImportReader::makeNonLiveList(QDomNodeList liveList) const
{
    QLinkedList<QDomNode> list;
    for (unsigned int ctr = 0; ctr < liveList.length(); ++ctr)
        list.append(liveList.item(ctr));
    return list;
}

inline static int countIndent(const QString &input)
{
    int ctr = 0;
    for (; input.length() > ctr && input[ctr] == ' '; ++ctr)
        ;
    return ctr;
}

QString coConfigImportReader::domToString(const QDomDocument &doc) const
{

    QStringList lines = doc.toString().split('\n', QString::KeepEmptyParts);
    QString rv;
    QString indent;

    QStringList::iterator prevLineIt = lines.begin();
    QStringList::iterator lineIt = lines.begin();

    for (QStringList::iterator nextLineIt = ++(lines.begin()); nextLineIt != lines.end(); ++nextLineIt)
    {

        QString line = *lineIt;

        // Add a CR and indentation after the comment
        if (line.trimmed().startsWith("<!--"))
        {
            indent.fill(' ', countIndent(*prevLineIt));
            line = line.simplified();
            line.replace("--> <", "--><");
            line.replace("-->", QString("-->\n%1").arg(indent));
            line = indent + line;
        }
        else
        {
            if (countIndent(*nextLineIt) > countIndent(line) && !rv.endsWith("\n\n"))
                rv += "\n";
        }

        rv += line;

        if (countIndent(*prevLineIt) > countIndent(line))
            rv += "\n";
        if (countIndent(line) < 3 && !rv.endsWith("\n\n"))
            rv += "\n";

        prevLineIt = lineIt;
        lineIt = nextLineIt;

        rv += '\n';
    }

    rv += *lineIt;

    return rv;
}
