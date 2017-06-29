/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/CoviseConfig.h>
#include <util/coFileUtil.h>
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <CovisePluginUtil/coBaseCoviseInteractor.h>

#include "FileBrowser.h"

FileBrowser *FileBrowser::s_instance = NULL;

FileBrowser::FileBrowser()
    : m_interactor(NULL)
{
    if (s_instance)
    {
        cerr << "multiple file browsers are not supported" << endl;
    }

    s_instance = this;
}

bool FileBrowser::init()
{
    m_buttonGroup = cover->createUniqueButtonGroupId();

    return true;
}

FileBrowser::~FileBrowser()
{
    removeMenuEntry();
}

FileBrowser *FileBrowser::instance()
{
    return s_instance;
}

void FileBrowser::newInteractor(const RenderObject *ro, coInteractor *abstractInter)
{
    coBaseCoviseInteractor *inter = dynamic_cast<coBaseCoviseInteractor *>(abstractInter);
    if (!inter)
        return;

    if (strcmp(inter->getPluginName(), "FileBrowser"))
        return;

    if (m_interactor)
    {
        removeMenuEntry();
        m_interactor->decRefCount();
        m_interactor = NULL;
    }

    m_interactor = inter;
    m_interactor->incRefCount();
    m_interactorObjName = ro->getName();
    m_moduleName = m_interactor->getModuleName();
    m_paramName = m_interactor->getParaName(0);

    char *val = NULL;
    m_interactor->getFileBrowserParam(0, val);
    if (cover->debugLevel(3))
        cerr << "FileBrowser::newInteractor: module name=" << m_moduleName
             << ", parameter name=" << m_paramName
             << ", value=" << val << endl;
    char *p = strrchr(val, ' ');
    if (p)
        *p = '\0';
    m_basedir = val;
    addMenuEntry();
}

void
FileBrowser::removeObject(const char *name, bool replace)
{
    if (cover->debugLevel(3))
        std::cerr << "FileBrowser::removeObject" << std::endl;

    if (m_interactor && m_interactorObjName == name && !replace)
    {
        removeMenuEntry();
        m_interactor->decRefCount();
        m_interactor = NULL;
    }
}

void FileBrowser::addMenuEntry()
{
    cover->addSubmenuButton("File Chooser...", NULL, m_moduleName.c_str(), false, NULL, m_buttonGroup, this);
    // generate menu entries for all datafiles in current directory
    readDirectory();
}

void FileBrowser::removeMenuEntry()
{
    cover->removeButton("File Chooser...", NULL);

    for (std::list<std::string>::iterator it = m_files.begin();
         it != m_files.end();
         ++it)
    {
        cover->removeButton((*it).c_str(), m_moduleName.c_str());
    }
}

void FileBrowser::readDirectory()
{
    m_files.clear();

    coDirectory *dir = coDirectory::open(m_basedir.c_str());
    if (!dir)
    {
        char *dirpart = new char[m_basedir.length() + 1];
        strcpy(dirpart, m_basedir.c_str());
        char *p = strrchr(dirpart, '/');
        if (p)
            *p = '\0';
        dir = coDirectory::open(dirpart);
        if (dir)
            m_basedir = dirpart;
        delete[] dirpart;
    }
    if (dir)
    {
        for (int i = 0; i < dir->count(); ++i)
        {
            if (dir->name(i)[0] != '.')
                m_files.push_back(dir->name(i));
        }
        dir->close();
        delete dir;
    }

    m_files.sort();
    for (std::list<std::string>::iterator it = m_files.begin();
         it != m_files.end();
         ++it)
    {
        cover->addFunctionButton((*it).c_str(), m_moduleName.c_str(), fileSelection, this); // "Datafiles"
    }
}

void FileBrowser::fileSelection(void *, buttonSpecCell *spec)
{
    std::string filename = instance()->m_basedir;
    if (filename.length() && filename.at(filename.length() - 1) != '/')
        filename += "/";
    for (std::list<std::string>::iterator it = instance()->m_files.begin();
         it != instance()->m_files.end();
         ++it)
    {
        if (!strcmp((*it).c_str(), spec->name))
        {
            filename += *it;
            break;
        }
    }

    if (cover->debugLevel(2))
        std::cerr << "FileBrowser plugin: " << filename << "selected to load" << std::endl;
    instance()->m_interactor->setFileBrowserParam(instance()->m_paramName.c_str(), filename.c_str());
    instance()->m_interactor->executeModule();
}

COVERPLUGIN(FileBrowser)
