/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/CoviseConfig.h>
#include <util/coFileUtil.h>
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <CovisePluginUtil/coBaseCoviseInteractor.h>

#include <cover/ui/ButtonGroup.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Button.h>

#include "FileBrowser.h"

FileBrowser *FileBrowser::s_instance = NULL;

FileBrowser::FileBrowser()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("FileBrowser", cover->ui)
, m_interactor(NULL)
{
    if (s_instance)
    {
        cerr << "multiple file browsers are not supported" << endl;
    }

    s_instance = this;
}

bool FileBrowser::init()
{
    m_buttonGroup = new ui::ButtonGroup("ButtonGroup", this);
    m_buttonGroup->enableDeselect(true);

    return true;
}

FileBrowser::~FileBrowser()
{
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
    m_menu = new ui::Menu("FileChooser", this);
    m_menu->setVisible(true);

    // generate menu entries for all datafiles in current directory
    readDirectory();
}

void FileBrowser::removeMenuEntry()
{
    m_menu->setVisible(false);
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
        std::string name = *it;
        auto b = new ui::Button(m_menu, "File", m_buttonGroup);
        b->setText(name);
        b->setCallback([this, name](bool state){
            if (state)
            {
                std::string filename = m_basedir;
                if (filename.length() && filename.at(filename.length() - 1) != '/')
                    filename += "/";
                filename += name;

                m_interactor->setFileBrowserParam(instance()->m_paramName.c_str(), filename.c_str());
                m_interactor->executeModule();
            }
        });
    }
}

COVERPLUGIN(FileBrowser)
