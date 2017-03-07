/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Compiler:
#include <iostream>
#ifndef WIN32
#include <dirent.h>
#endif
#include <sys/stat.h>
#include <sys/types.h>

// OSG:
#include <osg/Image>
#include <osgDB/ReadFile>

// Virvo:
#include <virvo/vvtoolshed.h>

// Covise:
#include <config/CoviseConfig.h>
#include <util/unixcompat.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>

// Local:
#include "coFileBrowser.h"
#ifndef WIN32
#include <sys/time.h>
#endif

#define FILE_LIMIT 10000000

using namespace std;
using namespace cui;
using namespace osg;
using namespace opencover;

coFileBrowser::coFileBrowser(string &browserDirectory, Interaction *interaction)
{
    _interaction = interaction;
    if (_interaction == NULL)
    {
        cerr << "Error: coFileBrowser was passed a NULL pointer for Interaction" << endl;
    }
    _browserDirectory = browserDirectory;

#ifdef WIN32
    _delim = '\\';
#else
    _delim = '/';
#endif

    _browserPanel = new Panel(_interaction, Panel::STATIC, Panel::NON_MOVABLE, 16);
    _browserPanel->setVisible(true);

    MatrixTransform *rot = new MatrixTransform();
    Matrix mat;
    Vec3 axis(1.0, 0.0, 0.0);
    mat.makeRotate(-M_PI, axis);
    //mat.setTrans(0.0, 0.0, 1000.0);
    rot->setMatrix(mat);
    //cover->getScene()->addChild(rot);
    root = rot;
    rot->addChild((osg::Node *)_browserPanel->getNode());

    // Read data directory:
    makeFileList(_browserDirectory, _fileNames);

    // Create icons :
    createFolderIcons();
    createFileIcons();
}

coFileBrowser::~coFileBrowser()
{
    delete _browserPanel;
}

void coFileBrowser::createFolderIcons()
{
    struct timeval start;
    gettimeofday(&start, NULL);
    _folderButtons.clear();

    for (unsigned int i = 0; i < _folderNames.size(); ++i)
    {
        // Create button:
        Button *newButton = new Button(_interaction);
        _folderButtons.push_back(newButton);
        char *basename = new char[_folderNames[i].length() + 1];
        vvToolshed::extractFilename(basename, _folderNames[i].c_str());
        newButton->setText(basename);
        delete[] basename;

        // Set icon image:
        std::string folderIcon;
        if (strcmp(_folderNames[i].c_str(), ".") == 0)
        {
            newButton->loadCUIImage("refresh.tif");
        }
        else if (strcmp(_folderNames[i].c_str(), "..") == 0)
        {
            newButton->loadCUIImage("folder-up.tif");
        }
        else
            newButton->loadCUIImage("folder.tif");

        _browserPanel->addCard(newButton);
        // add CaveVOX listener after panel listener
        // to show panel depending on which card was pressed
        newButton->addCardListener(this);
    }
}

void coFileBrowser::createFileIcons()
{
    const unsigned int MAX_NAME_LEN = 6;

    _fileButtons.clear();
    for (unsigned int i = 0; i < _fileNames.size(); ++i)
    {
        // Create button:
        Button *newButton = new Button(_interaction);
        _fileButtons.push_back(newButton);
        char *basename = new char[_fileNames[i].length() + 1];
        string name = _fileNames[i];
        string s = _fileNames[i];
        string ss;
        if (s.length() > MAX_NAME_LEN)
            ss = s.substr(0, MAX_NAME_LEN);
        else
            ss = s;

        vvToolshed::extractBasename(basename, ss.c_str());
        newButton->setText(basename);
        newButton->setTipText(const_cast<char *>(s.c_str()));
        newButton->setTipDelay(2);
        newButton->setTipVisibility(true);
        delete[] basename;

        char *buf = new char[_fileNames[i].size() + 1];

        vvToolshed::extractExtension(buf, _fileNames[i].c_str());
        string fext = buf;
        std::transform(fext.begin(), fext.end(), fext.begin(), ::tolower);

        //cerr << "name: " << name << " size: " << name.size()  << " ext: " << fext << " size: " << fext.size() << endl;

        if (name.size() > 0)
        {
            name = name.substr(0, name.size() - fext.size());
        }

        // Create icon:
        string path = _browserDirectory + _delim + _fileNames[i];
        int status = 0;
        if (coVRMSController::instance()->isMaster())
        {
            struct stat filestats;
            stat(path.c_str(), &filestats);
            if (filestats.st_size > FILE_LIMIT || _extensions[fext] || !newButton->loadImage(path))
            {
                if (_extensions[fext])
                {
                    Image *image = osgDB::readImageFile(_browserDirectory + _delim + name + "tif");
                    if (image)
                    {
                        status = 2;
                        //cerr << "Loaded: " << _browserDirectory + _delim + name + "tif" << endl;
                        newButton->setIconImage(image);
                    }
                    else
                    {
                        //cerr << "Failed to load: " << _browserDirectory + _delim + name + "tif" << endl;
                        if (!newButton->loadCUIImage(fext + ".tif"))
                        {
                            status = 4;
                            newButton->loadCUIImage("document.tif");
                        }
                        else
                        {
                            status = 3;
                        }
                    }
                }
                else
                {
                    if (!newButton->loadCUIImage(fext + ".tif"))
                    {
                        status = 4;
                        newButton->loadCUIImage("document.tif");
                    }
                    else
                    {
                        status = 3;
                    }
                }
            }
            else
            {
                status = 1;
            }
            coVRMSController::instance()->sendSlaves((char *)&status, sizeof(int));
        }
        else
        {
            coVRMSController::instance()->readMaster((char *)&status, sizeof(int));
            switch (status)
            {
            case 1:
            {
                newButton->loadImage(path);
                break;
            }
            case 2:
            {
                Image *image = osgDB::readImageFile(_browserDirectory + _delim + name + "tif");
                newButton->setIconImage(image);
                break;
            }
            case 3:
            {
                newButton->loadCUIImage(fext + ".tif");
                break;
            }
            case 4:
            {
                newButton->loadCUIImage("document.tif");
                break;
            }
            default:
                break;
            }
        }

        delete[] buf;
        newButton->setTipVisibility(true);
        newButton->setTipDelay(1.0);
        newButton->setTipText((char *)_fileNames[i].c_str(), true);

        _browserPanel->addCard(newButton);
        newButton->addCardListener(this);
    }
}

bool coFileBrowser::makeFileList(std::string &path, std::vector<std::string> &fileNames)
{
    struct stat statbuf;
#ifndef WIN32
    DIR *dirHandle;
    struct dirent *entry;

    dirHandle = opendir(path.c_str());
    if (dirHandle == NULL)
    {
        cerr << "Cannot read directory: " << path << endl;
        return false;
    }

    char *oldpath = getcwd(NULL, 0);

    if (chdir(path.c_str()) != 0)
    {
        const int PATH_SIZE = 256;
        char cwd[PATH_SIZE];
        cerr << "Cannot chdir to " << path << ". Searching for volume files in " << getcwd(cwd, PATH_SIZE) << endl;
    }
    while ((entry = readdir(dirHandle)) != NULL)
    {
        stat(entry->d_name, &statbuf);
        if (S_ISDIR(statbuf.st_mode))
        {
            // Found a folder:
            _folderNames.push_back(entry->d_name);
        }
        else
        {
            // Found a file:
            char *buf = new char[strlen(entry->d_name) + 1];
            vvToolshed::extractExtension(buf, entry->d_name);
            string fext = buf;
            std::transform(fext.begin(), fext.end(), fext.begin(), ::tolower);
            //cerr << "extension: " << fext << endl;
            if (_extensions.find(fext) != _extensions.end())
            {
                fileNames.push_back(entry->d_name);
                //cerr << "File added: " << entry->d_name << endl;
            }
            delete[] buf;
        }
    }
#else

    struct _finddata_t *c_file = new struct _finddata_t;
    char *pattern = new char[path.length() + 20];
    strcpy(pattern, path.c_str());
    strcat(pattern, "/*");
    long handle = _findfirst(pattern, c_file);
    if (handle == -1)
    {
        //cerr << "Cannot read directory: " << path << endl;
        return false;
    }
    delete[] pattern;

    do
    {

        stat(c_file->name, &statbuf);
        if (statbuf.st_mode & _S_IFDIR)
        {
            // Found a folder:
            _folderNames.push_back(c_file->name);
        }
        else
        {
            // Found a file:
            char *buf = new char[strlen(c_file->name) + 1];
            vvToolshed::extractExtension(buf, c_file->name);
            string fext = buf;
            std::transform(fext.begin(), fext.end(), fext.begin(), ::tolower);
            //cerr << "extension: " << fext << endl;
            if (_extensions.find(fext) != _extensions.end())
            {
                fileNames.push_back(c_file->name);
                //cerr << "File added: " << c_file->name << endl;
            }
            delete[] buf;
        }
    } while (_findnext(handle, c_file) == 0);
#endif

    sortAlphabetically(_folderNames);
    sortAlphabetically(_fileNames);
#ifndef WIN32
    chdir(oldpath);
    closedir(dirHandle);
#endif
    removePreviews();
    return true;
}

void coFileBrowser::removePreviews()
{
    vector<string> copy = _fileNames;
    for (int i = 0; i < copy.size(); i++)
    {
        char *buf = new char[copy[i].size() + 1];
        vvToolshed::extractExtension(buf, copy[i].c_str());
        string fext = buf;
        std::transform(fext.begin(), fext.end(), fext.begin(), ::tolower);
        if (fext == "tif")
        {
            continue;
        }
        string base = copy[i].substr(0, copy[i].size() - fext.size());
        for (vector<string>::iterator it = _fileNames.begin(); it != _fileNames.end();)
        {
            if ((*it) == base + "tif" && _extensions[fext])
            {
                it = _fileNames.erase(it);
                break;
            }
            it++;
        }
    }
}

void coFileBrowser::sortAlphabetically(std::vector<std::string> &names)
{
    std::string tmp;
    unsigned int i, j;

    if (names.size() < 2)
        return; // nothing to do
    for (i = 0; i < names.size() - 1; ++i)
    {
        for (j = i; j < names.size(); ++j)
        {
            if (names[i] > names[j])
            {
                tmp = names[i];
                names[i] = names[j];
                names[j] = tmp;
            }
        }
    }
}

bool coFileBrowser::cardButtonEvent(Card *card, int button, int newState)
{
    unsigned int i;
    if (button == 0 && newState == 0)
    {
        // Check file buttons:
        for (i = 0; i < _fileButtons.size(); ++i)
        {
            if (card == _fileButtons[i])
            {
                _selectedFile = _browserDirectory + _delim + _fileNames[i];
                char *buf = new char[_fileNames[i].size() + 1];

                vvToolshed::extractExtension(buf, _fileNames[i].c_str());
                string fext = buf;
                std::transform(fext.begin(), fext.end(), fext.begin(), ::tolower);
                //cerr << _fileNames[i] << " selected" << endl;
                std::list<coFileBrowserListener *>::iterator iter;
                for (iter = _listeners.begin(); iter != _listeners.end(); ++iter)
                {
                    (*iter)->fileBrowserEvent(this, _selectedFile, fext, button, newState);
                }
                delete[] buf;
            }
        }

        // Check folder buttons:
        unsigned int subDirsSize = _folderButtons.size();
        for (i = 0; i < subDirsSize; ++i)
        {
            if (card == _folderButtons[i])
            {
                std::string bakDir = _browserDirectory;
#ifdef WIN32
                _browserDirectory += _delim + _folderNames[i];
#else
                _browserDirectory += _delim;
                int pos;
                while ((pos = _browserDirectory.find("//")) != string::npos)
                {
                    _browserDirectory.erase(pos, 1);
                }
                if (_folderNames[i] == "..")
                {
                    if (_browserDirectory != "/")
                    {
                        _browserDirectory.erase(_browserDirectory.size() - 1, 1);
                        pos = _browserDirectory.find_last_of("/");
                        _browserDirectory.erase(pos, string::npos);
                        if (_browserDirectory == "")
                        {
                            _browserDirectory = "/";
                        }
                    }
                }
                else if (_folderNames[i] != ".")
                {
                    _browserDirectory += _folderNames[i];
                }

#endif
                //cerr << "Current folder: " << _browserDirectory << endl;

                // Reset lists and remove buttons:
                _browserPanel->reset();
                _folderButtons.clear();
                _folderNames.clear();
                _fileButtons.clear();
                _fileNames.clear();

                if (!makeFileList(_browserDirectory, _fileNames))
                {
                    _browserDirectory = bakDir;
                    makeFileList(_browserDirectory, _fileNames);
                }

                createFolderIcons();

                createFileIcons();

                break;
            }
        }
    }
    return false;
}

bool coFileBrowser::cardCursorUpdate(Card *, InputDevice *)
{
    return false;
}

string coFileBrowser::getFileName()
{
    return _selectedFile;
}

void coFileBrowser::addListener(coFileBrowserListener *fbl)
{
    _listeners.push_back(fbl);
}

void coFileBrowser::setVisible(bool visible)
{
    _handle->setVisible(visible);
    _browserPanel->setVisible(visible);
}

osg::Node *coFileBrowser::getNode()
{
    return root;
}

float coFileBrowser::getWidth()
{
    return _browserPanel->getScale() * _browserPanel->getWidth();
}

float coFileBrowser::getHeight()
{
    return _browserPanel->getScale() * _browserPanel->getHeight();
}

void coFileBrowser::addExt(string ext, bool preview)
{
    if (_extensions.find(ext) != _extensions.end())
    {
        return;
    }
    _extensions[ext] = preview;

    _browserPanel->reset();
    _folderButtons.clear();
    _folderNames.clear();
    _fileButtons.clear();
    _fileNames.clear();

    makeFileList(_browserDirectory, _fileNames);

    // Create icons :
    createFolderIcons();
    createFileIcons();
}

void coFileBrowser::removeExt(string ext)
{
    _extensions.erase(ext);

    _browserPanel->reset();
    _folderButtons.clear();
    _folderNames.clear();
    _fileButtons.clear();
    _fileNames.clear();

    makeFileList(_browserDirectory, _fileNames);

    // Create icons :
    createFolderIcons();
    createFileIcons();
}

void coFileBrowser::changeDir(string dir)
{
    string bkDir = _browserDirectory;
    _browserDirectory = dir;

    _browserPanel->reset();
    _folderButtons.clear();
    _folderNames.clear();
    _fileButtons.clear();
    _fileNames.clear();

    if (!makeFileList(_browserDirectory, _fileNames))
    {
        _browserDirectory = bkDir;
        makeFileList(_browserDirectory, _fileNames);
    }

    // Create icons :
    createFolderIcons();
    createFileIcons();
}
