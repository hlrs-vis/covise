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
#include <cover/coVRPluginSupport.h>

// Local:
#include "FileBrowser.h"

using namespace std;
using namespace cui;
using namespace osg;

FileBrowser::FileBrowser(string &browserDirectory, string &extension, Interaction *interaction)
{
    _interaction = interaction;
    if (_interaction == NULL)
    {
        cerr << "Error: FileBrowser was passed a NULL pointer for Interaction" << endl;
    }
    _browserDirectory = browserDirectory;
    _extension = extension;

#ifdef WIN32
    _delim = '\\';
#else
    _delim = '/';
#endif

    _browserPanel = new Panel(_interaction, Panel::STATIC, Panel::FIXED_ORIENTATION);
    _browserPanel->setVisible(true);

    MatrixTransform *rot = new MatrixTransform();
    Matrix mat;
    Vec3 axis(1.0, 0.0, 0.0);
    mat.makeRotate(-M_PI / 2.0f, axis);
    mat.setTrans(0.0, 0.0, 1000.0);
    rot->setMatrix(mat);
    opencover::cover->getScene()->addChild(rot);
    rot->addChild((osg::Node *)_browserPanel->getNode());

    // Read data directory:
    makeFileList(_browserDirectory, _fileNames);

    // Create icons :
    createFolderIcons();
    createFileIcons();
}

FileBrowser::~FileBrowser()
{
    delete _browserPanel;
}

void FileBrowser::createFolderIcons()
{
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

void FileBrowser::createFileIcons()
{
    const unsigned int MAX_NAME_LEN = 6;

    _fileButtons.clear();
    for (unsigned int i = 0; i < _fileNames.size(); ++i)
    {
        // Create button:
        Button *newButton = new Button(_interaction);
        _fileButtons.push_back(newButton);
        char *basename = new char[_fileNames[i].length() + 1];
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

        // Create icon:
        string path = _browserDirectory + _delim + _fileNames[i];
        newButton->loadImage(path);
        /*
    Image* image = osgDB::readImageFile(_fileNames[i]);
    newButton->setIconImage(image);
//    else newButton->loadCUIImage("document.tif");
*/
        newButton->setTipVisibility(true);
        newButton->setTipDelay(1.0);
        newButton->setTipText((char *)_fileNames[i].c_str(), true);

        _browserPanel->addCard(newButton);
        newButton->addCardListener(this);
    }
}

bool FileBrowser::makeFileList(std::string &path, std::vector<std::string> &fileNames)
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
            if (vvToolshed::strCompare(buf, _extension.c_str()) == 0)
            {
                fileNames.push_back(entry->d_name);
                cerr << "File added: " << entry->d_name << endl;
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
        cerr << "Cannot read directory: " << path << endl;
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
            if (vvToolshed::strCompare(buf, _extension.c_str()) == 0)
            {
                fileNames.push_back(c_file->name);
                cerr << "File added: " << c_file->name << endl;
            }
            delete[] buf;
        }
    } while (_findnext(handle, c_file) == 0);
#endif

    sortAlphabetically(_folderNames);
    sortAlphabetically(_fileNames);
#ifndef WIN32
    closedir(dirHandle);
#endif
    return true;
}

void FileBrowser::sortAlphabetically(std::vector<std::string> &names)
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

bool FileBrowser::cardButtonEvent(Card *card, int button, int newState)
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
                cerr << _fileNames[i] << " selected" << endl;
                std::list<FileBrowserListener *>::iterator iter;
                for (iter = _listeners.begin(); iter != _listeners.end(); ++iter)
                {
                    (*iter)->fileBrowserEvent(this, _selectedFile, button, newState);
                }
            }
        }

        // Check folder buttons:
        unsigned int subDirsSize = _folderButtons.size();
        for (i = 0; i < subDirsSize; ++i)
        {
            if (card == _folderButtons[i])
            {
                std::string bakDir = _browserDirectory;
                _browserDirectory += _delim + _folderNames[i];
                cerr << "Current folder: " << _browserDirectory << endl;

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

bool FileBrowser::cardCursorUpdate(Card *, InputDevice *)
{
    return false;
}

string FileBrowser::getFileName()
{
    return _selectedFile;
}

void FileBrowser::addListener(FileBrowserListener *fbl)
{
    _listeners.push_back(fbl);
}

void FileBrowser::setVisible(bool visible)
{
    _browserPanel->setVisible(visible);
}
