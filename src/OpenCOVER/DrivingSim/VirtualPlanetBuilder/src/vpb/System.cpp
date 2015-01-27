/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* -*-c++-*- VirtualPlanetBuilder - Copyright (C) 1998-2009 Robert Osfield
 *
 * This library is open source and may be redistributed and/or modified under
 * the terms of the OpenSceneGraph Public License (OSGPL) version 0.0 or
 * (at your option) any later version.  The full license is in LICENSE file
 * included with this distribution, and on the openscenegraph.org website.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * OpenSceneGraph Public License for more details.
*/

#include <vpb/System>
#include <vpb/BuildLog>
#include <vpb/Date>
#include <vpb/FileUtils>

#include <map>
#include <gdal_priv.h>

using namespace vpb;

std::string vpb::getLocalHostName()
{
    char hostname[1024];
    if (vpb::gethostname(hostname, sizeof(hostname)) == 0)
    {
        return std::string(hostname);
    }
    else
    {
        return std::string();
    }
}

int vpb::getProcessID()
{
    return vpb::getpid();
}

// convience methods for accessing System singletons variables.
osgDB::FilePathList &vpb::getSourcePaths() { return System::instance()->getSourcePaths(); }
std::string &vpb::getDestinationDirectory() { return System::instance()->getDestinationDirectory(); }
std::string &vpb::getIntermediateDirectory() { return System::instance()->getIntermediateDirectory(); }
std::string &vpb::getLogDirectory() { return System::instance()->getLogDirectory(); }
std::string &vpb::getTaskDirectory() { return System::instance()->getTaskDirectory(); }
std::string &vpb::getMachineFileName() { return System::instance()->getMachineFileName(); }
std::string &vpb::getCacheFileName() { return System::instance()->getCacheFileName(); }
unsigned int vpb::getMaxNumberOfFilesPerDirectory() { return System::instance()->getMaxNumberOfFilesPerDirectory(); }

///////////////////////////////////////////////////////////////////////////////////////////////////
//
//  FileSytem singleton

osg::ref_ptr<System> &System::instance()
{
    static osg::ref_ptr<System> s_System = new System;
    return s_System;
}

System::System()
{
    // setup GDAL
    GDALAllRegister();

    _trimOldestTiles = true;
    _numUnusedDatasetsToTrimFromCache = 10;
    _maxNumDatasets = (unsigned int)(double(vpb::getdtablesize()) * 0.8);

    _logDirectory = "logs";
    _taskDirectory = "tasks";

    _maxNumberOfFilesPerDirectory = 1000;

    readEnvironmentVariables();

    // preload the .osg plugin so its available in case we need to output source files containing core osg nodes
    osgDB::Registry::instance()->loadLibrary(osgDB::Registry::instance()->createLibraryNameForExtension("osg"));

    GDALDriverManager *driverManager = GetGDALDriverManager();
    if (driverManager)
    {
        for (int i = 0; i < driverManager->GetDriverCount(); ++i)
        {
            GDALDriver *driver = driverManager->GetDriver(i);
            if (driver)
            {
                const char *ext = driver->GetMetadataItem("DMD_EXTENSION");
                if (ext && strlen(ext) != 0)
                {
                    addSupportedExtension(ext,
                                          Source::IMAGE | Source::HEIGHT_FIELD,
                                          driver->GetMetadataItem(GDAL_DMD_LONGNAME));
                }
            }
        }
    }

    // add entries that GDAL doesn't list via it's DMD_EXTENSIONS but is known to support
    addSupportedExtension("jpeg", Source::IMAGE | Source::HEIGHT_FIELD, "JPEG");
    addSupportedExtension("tiff", Source::IMAGE | Source::HEIGHT_FIELD, "GeoTiff");
    addSupportedExtension("pgm", Source::IMAGE | Source::HEIGHT_FIELD, "Netpbm");
    addSupportedExtension("ppm", Source::IMAGE | Source::HEIGHT_FIELD, "Netpbm");

    addSupportedExtension("shp", Source::SHAPEFILE, "Shape file loader");

    addSupportedExtension("osgb", Source::MODEL, "OpenSceneGraph binary format");
    addSupportedExtension("osgx", Source::MODEL, "OpenSceneGraph xml format");
    addSupportedExtension("osgt", Source::MODEL, "OpenSceneGraph text/ascii format");

    addSupportedExtension("osg", Source::MODEL, "OpenSceneGraph .osg ascii format");
    addSupportedExtension("ive", Source::MODEL, "OpenSceneGraph .ive binary format");
}

System::~System()
{
    _machinePool = 0;
    _taskManager = 0;
    _fileCache = 0;
}

void System::readEnvironmentVariables()
{
    const char *str = getenv("VPB_SOURCE_PATHS");
    if (str)
    {
        osgDB::convertStringPathIntoFilePathList(std::string(str), _sourcePaths);
    }

    str = getenv("VPB_DESTINATION_DIR");
    if (str)
    {
        _destinationDirectory = str;
    }

    str = getenv("VPB_INTERMEDIATE_DIR");
    if (str)
    {
        _intermediateDirectory = str;
    }

    str = getenv("VPB_LOG_DIR");
    if (str)
    {
        _logDirectory = str;
    }

    str = getenv("VPB_TASK_DIR");
    if (str)
    {
        _taskDirectory = str;
    }

    str = getenv("VPB_TRIM_TILES_SCHEME");
    if (str)
    {
        if (strcmp(str, "OLDEST") == 0 || strcmp(str, "oldest") == 0 || strcmp(str, "Oldest") == 0)
        {
            _trimOldestTiles = true;
        }
        else
        {
            _trimOldestTiles = false;
        }
    }

    str = getenv("VPB_NUM_UNUSED_DATASETS_TO_TRIM_FROM_CACHE");
    if (str)
    {
        _numUnusedDatasetsToTrimFromCache = atoi(str);
    }

    str = getenv("VPB_MAXIMUM_NUM_OPEN_DATASETS");
    if (str)
    {
        _maxNumDatasets = atoi(str);
    }

    str = getenv("VPB_MACHINE_FILE");
    if (str)
    {
        _machineFileName = str;
    }

    str = getenv("VPB_CACHE_FILE");
    if (str)
    {
        _cacheFileName = str;
    }

    str = getenv("VPB_MAXIMUM_OF_FILES_PER_DIRECTORY");
    if (str)
    {
        _maxNumberOfFilesPerDirectory = atoi(str);
    }
}

void System::readArguments(osg::ArgumentParser &arguments)
{
    while (arguments.read("--machines", _machineFileName))
    {
    }

    while (arguments.read("--cache", _cacheFileName))
    {
    }
    while (arguments.read("--xodr", _xodrName))
    {
    }
}

FileCache *System::getFileCache()
{
    if (!_fileCache && !_cacheFileName.empty())
    {
        _fileCache = new FileCache;
        _fileCache->open(_cacheFileName);
    }

    return _fileCache.get();
}

MachinePool *System::getMachinePool()
{
    if (!_machinePool)
    {
        _machinePool = new MachinePool;

        if (!_machineFileName.empty())
        {
            _machinePool->read(_machineFileName);
        }

        if (_machinePool->getNumMachines() == 0)
        {
            _machinePool->setUpOnLocalHost();
        }
    }

    return _machinePool.get();
}

TaskManager *System::getTaskManager()
{
    if (!_taskManager)
    {
        _taskManager = new TaskManager;
    }

    return _taskManager.get();
}

void System::clearDatasetCache()
{
    _datasetMap.clear();
}

class TrimN
{
public:
    TrimN(unsigned int n, bool oldest)
        : _oldest(oldest)
        , _num(n)
    {
    }

    inline void add(System::DatasetMap::iterator itr)
    {
        if (itr->second->referenceCount() != 1)
            return;

        double t = itr->second->getTimeStamp();
        if (_timeIteratorMap.size() < _num)
        {
            _timeIteratorMap.insert(TimeIteratorMap::value_type(t, itr));
        }
        else if (_oldest)
        {
            if (t < _timeIteratorMap.rbegin()->first)
            {
                // erase the end entry
                _timeIteratorMap.erase(_timeIteratorMap.rbegin()->first);
                _timeIteratorMap.insert(TimeIteratorMap::value_type(t, itr));
            }
        }
        else
        {
            if (t > _timeIteratorMap.begin()->first)
            {
                // erase the first entry
                _timeIteratorMap.erase(_timeIteratorMap.begin()->first);
                _timeIteratorMap.insert(TimeIteratorMap::value_type(t, itr));
            }
        }
    }

    void add(System::DatasetMap &datasetMap)
    {
        for (System::DatasetMap::iterator itr = datasetMap.begin();
             itr != datasetMap.end();
             ++itr)
        {
            add(itr);
        }
    }

    void eraseFrom(System::DatasetMap &datasetMap)
    {
        for (TimeIteratorMap::iterator itr = _timeIteratorMap.begin();
             itr != _timeIteratorMap.end();
             ++itr)
        {
            datasetMap.erase(itr->second);
        }
    }

    typedef std::multimap<double, System::DatasetMap::iterator> TimeIteratorMap;

    bool _oldest;
    unsigned int _num;
    TimeIteratorMap _timeIteratorMap;
};

void System::clearUnusedDatasets(unsigned int numToClear)
{
    TrimN lowerN(numToClear, _trimOldestTiles);

    lowerN.add(_datasetMap);
    lowerN.eraseFrom(_datasetMap);

    _datasetMap.clear();
}

GeospatialDataset *System::openGeospatialDataset(const std::string &filename, AccessMode accessMode)
{
    // first check to see if dataset already exists in cache, if so return it.
    DatasetMap::iterator itr = _datasetMap.find(FileNameAccessModePair(filename, accessMode));
    if (itr != _datasetMap.end())
    {
        //osg::notify(osg::NOTICE)<<"System::openGeospatialDataset("<<filename<<") returning existing entry, ref count "<<itr->second->referenceCount()<<std::endl;
        return itr->second.get();
    }

    // make sure there is room available for this new Dataset
    if (_datasetMap.size() >= _maxNumDatasets)
        clearUnusedDatasets(_numUnusedDatasetsToTrimFromCache);

    // double check to make sure there is room to open a new dataset
    if (_datasetMap.size() >= _maxNumDatasets)
    {
        log(osg::NOTICE, "Error: System::GDALOpen(%s) unable to open file as unsufficient file handles available.", filename.c_str());
        return 0;
    }

    //osg::notify(osg::NOTICE)<<"System::openGeospatialDataset("<<filename<<") requires new entry "<<std::endl;

    // open the new dataset.
    GeospatialDataset *dataset = new GeospatialDataset(filename, accessMode);

    // insert it into the cache
    _datasetMap[FileNameAccessModePair(filename, accessMode)] = dataset;

    // return it.
    return dataset;
}

GeospatialDataset *System::openOptimumGeospatialDataset(const std::string &filename, const SpatialProperties &sp, AccessMode accessMode)
{
    if (_fileCache.valid())
    {
        std::string optimumFile = _fileCache->getOptimimumFile(filename, sp);
        return (optimumFile.empty()) ? 0 : openGeospatialDataset(optimumFile, accessMode);
    }
    else
    {
        return openGeospatialDataset(filename, accessMode);
    }
}

bool System::getDateOfLastModification(osgTerrain::TerrainTile *source, Date &date)
{
    typedef std::list<osgTerrain::Layer *> Layers;
    Layers layers;

    if (source->getElevationLayer())
    {
        layers.push_back(source->getElevationLayer());
    }

    for (unsigned int i = 0; i < source->getNumColorLayers(); ++i)
    {
        osgTerrain::Layer *layer = source->getColorLayer(i);
        if (layer)
        {
            layers.push_back(layer);
        }
    }

    typedef std::list<std::string> Filenames;
    Filenames filenames;

    for (Layers::iterator itr = layers.begin();
         itr != layers.end();
         ++itr)
    {
        osgTerrain::CompositeLayer *compositeLayer = dynamic_cast<osgTerrain::CompositeLayer *>(*itr);
        if (compositeLayer)
        {
            for (unsigned int i = 0; i < compositeLayer->getNumLayers(); ++i)
            {
                filenames.push_back(compositeLayer->getFileName(i));
            }
        }
        else
        {
            filenames.push_back((*itr)->getFileName());
        }
    }

    bool modified = false;
    for (Filenames::iterator itr = filenames.begin();
         itr != filenames.end();
         ++itr)
    {
        Date lastModification;
        if (lastModification.setWithDateOfLastModification(*itr))
        {
            if (lastModification > date)
            {
                date = lastModification;
                modified = true;
            }
        }
    }

    return modified;
}

unsigned long System::getFileSize(const std::string &filename)
{
    struct stat s;
    int status = stat(filename.c_str(), &s);
    if (status == 0)
    {
        return static_cast<unsigned long>(s.st_size);
    }
    else
    {
        return 0;
    }
}

bool System::openFileToCheckThatItSupported(const std::string &filename, int acceptedTypeMask)
{
    osg::notify(osg::INFO) << "System::openFileToCheckThatItSupported(" << filename << ")" << std::endl;
    std::string ext = osgDB::getFileExtension(filename);

    GDALDataset *dataset = (GDALDataset *)GDALOpen(filename.c_str(), GA_ReadOnly);
    if (dataset)
    {
        osg::notify(osg::INFO) << "   GDALOpen(" << filename << ") succeeded " << std::endl;
        int fileTypeMask = Source::IMAGE | Source::HEIGHT_FIELD;
        addSupportedExtension(ext, fileTypeMask, "");

        GDALClose(dataset);

        return (acceptedTypeMask & fileTypeMask) != 0;
    }

    _unsupportedExtensions.insert(ext);
    return false;
}
