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

#include <vpb/FileCache>
#include <vpb/System>
#include <vpb/BuildLog>
#include <vpb/DataSet>

#include <osg/io_utils>
#include <osgDB/FileNameUtils>

using namespace vpb;

FileCache::FileCache()
{
    _requiresWrite = false;
}


FileCache::FileCache(const FileCache& fc,const osg::CopyOp& copyop):
    osg::Object(fc, copyop)
{
    _requiresWrite = false;
}

FileCache::~FileCache()
{
}

bool FileCache::read(const std::string& filename)
{
    log(osg::NOTICE,"FileCache::read(%s)",filename.c_str());

    std::string foundFile = osgDB::findDataFile(filename);
    if (foundFile.empty())
    {
        log(osg::WARN,"Error: could not find cache file '%s'",filename.c_str());
        return false;
    }

    _filename = filename;

    osgDB::ifstream fin(foundFile.c_str());
    
    bool emptyBefore = _variantMap.empty();
    
    if (fin)
    {
        osgDB::Input fr;
        fr.attach(&fin);
        
        std::string str;

        while(!fr.eof())
        {
            bool itrAdvanced = false;

            if (fr.matchSequence("FileDetails {"))
            {
                osg::ref_ptr<FileDetails> fd = new FileDetails;

                int local_entry = fr[0].getNoNestedBrackets();

                fr += 2;

                while (!fr.eof() && fr[0].getNoNestedBrackets()>local_entry)
                {
                    bool localAdvanced = false;

                    if (fr.read("build",str))
                    {
                        fd->setBuildApplication(str);
                        localAdvanced = true;
                    }

                    if (fr.read("hostname",str))
                    {
                        fd->setHostName(str);
                        localAdvanced = true;
                    }

                    if (fr.read("original",str))
                    {
                        fd->setOriginalSourceFileName(str);
                        localAdvanced = true;
                    }

                    if (fr.read("file",str))
                    {
                        fd->setFileName(str);
                        localAdvanced = true;
                    }

                    if (fr.read("cs",str))
                    {
                        if (!fd->getSpatialProperties()._cs) fd->getSpatialProperties()._cs = new osg::CoordinateSystemNode;
                        
                        fd->getSpatialProperties()._cs->setCoordinateSystem(str);
                         fd->getSpatialProperties()._extents._isGeographic = getCoordinateSystemType(fd->getSpatialProperties()._cs.get())==GEOGRAPHIC;

                        localAdvanced = true;
                    }

                    double minX, maxX, minY, maxY;
                    if (fr.read("extents",minX, minY, maxX, maxY))
                    {
                        fd->getSpatialProperties()._extents._min.set(minX,minY);
                        fd->getSpatialProperties()._extents._max.set(maxX,maxY);
                        localAdvanced = true;
                    }

                    osg::Matrixd m;
                    if (fr.read("geoTransform",m(0,0),m(0,1),m(1,0),m(1,1),m(3,0),m(3,1)))
                    {
                        fd->getSpatialProperties()._geoTransform = m;
                    }


                    int sizeX, sizeY, sizeZ;
                    if (fr.read("size",sizeX, sizeY, sizeZ))
                    {
                        fd->getSpatialProperties()._numValuesX = sizeX;
                        fd->getSpatialProperties()._numValuesY = sizeY;
                        fd->getSpatialProperties()._numValuesZ = sizeZ;
                        localAdvanced = true;
                    }

                    if (!localAdvanced) ++fr;
                }

                ++fr;

                itrAdvanced = true;
                
                addFileDetails(fd.get());

            }
            
            if (!itrAdvanced) ++fr;
        }
    }
    
    _requiresWrite = !emptyBefore;

    return false;
}
 
bool FileCache::open(const std::string& filename)
{
    std::string foundFile = osgDB::findDataFile(filename);
    if (foundFile.empty())
    {
        setFileName(filename);
        _requiresWrite = true;
        return false;
    }
    
    return read(foundFile);
}

bool FileCache::write(const std::string& filename)
{
    log(osg::NOTICE,"FileCache::write(%s)",filename.c_str());

    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_variantMapMutex);

    _filename = filename;
    _requiresWrite = false;

    osgDB::Output fout(filename.c_str());

    fout.precision(15);

    for(VariantMap::iterator itr = _variantMap.begin();
        itr != _variantMap.end();
        ++itr)
    {
        Variants& variants = itr->second;
        for(Variants::iterator vitr = variants.begin();
            vitr != variants.end();
            ++vitr)
        {
            FileDetails* fd = vitr->get();
            
            fout.indent()<<"FileDetails {"<<std::endl;
            fout.moveIn();
            
            if (!fd->getBuildApplication().empty())
            {
                fout.indent()<<"build "<<fout.wrapString(fd->getBuildApplication())<<std::endl;
            }

            if (!fd->getHostName().empty())
            {
                fout.indent()<<"hostname "<<fout.wrapString(fd->getHostName())<<std::endl;
            }

            if (!fd->getOriginalSourceFileName().empty())
            {
                fout.indent()<<"original "<<fout.wrapString(fd->getOriginalSourceFileName())<<std::endl;
            }
            
            if (!fd->getFileName().empty())
            {
                fout.indent()<<"file "<<fout.wrapString(fd->getFileName())<<std::endl;
            }
            
            if (fd->getSpatialProperties()._cs.valid() && !fd->getSpatialProperties()._cs->getCoordinateSystem().empty())
            {
                fout.indent()<<"cs "<<fout.wrapString(fd->getSpatialProperties()._cs->getCoordinateSystem())<<std::endl;
            }

            if (fd->getSpatialProperties()._extents.valid())
            {
                const GeospatialExtents& extents = fd->getSpatialProperties()._extents;
                fout.indent()<<"extents "<<extents.xMin()<<" "<<extents.yMin()<<" "<<extents.xMax()<<" "<<extents.yMax()<<std::endl;
            }
                        
            if (!fd->getSpatialProperties()._geoTransform.isIdentity())
            {
                const osg::Matrixd& m = fd->getSpatialProperties()._geoTransform;
                fout.indent()<<"geoTransform "<<m(0,0)<<" "<<m(0,1)<<" "<<m(1,0)<<" "<<m(1,1)<<" "<<m(3,0)<<" "<<m(3,1)<<std::endl;
            }
            
            if (fd->getSpatialProperties()._numValuesX>0 || fd->getSpatialProperties()._numValuesY>0 || fd->getSpatialProperties()._numValuesZ>0)
            {
                fout.indent()<<"size "<<fd->getSpatialProperties()._numValuesX<<" "<<fd->getSpatialProperties()._numValuesY<<" "<<fd->getSpatialProperties()._numValuesZ<<std::endl;
            }
            
            fout.moveOut();
            fout.indent()<<"}"<<std::endl;
            
            
        }
    }

    return false;
}

void FileCache::addFileDetails(FileDetails* fd)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_variantMapMutex);

    _requiresWrite = true;
    
    _fileDetailsMap[fd->getFileName()] = fd;
    
    Variants& variants = _variantMap[fd->getOriginalSourceFileName()];
    for(Variants::iterator vitr = variants.begin();
        vitr != variants.end();
        ++vitr)
    {
        if (*(*vitr) == *fd)
        {
            log(osg::INFO,"FileCache::addFileDetails(%s) FileDetails already in cache",fd->getFileName().c_str());
            return;
        }
    }
    
    log(osg::INFO,"FileCache::addFileDetails(%s) added",fd->getFileName().c_str());

    variants.push_back(fd);
}

void FileCache::removeFileDetails(FileDetails* fd)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_variantMapMutex);

    _requiresWrite = true;

    FileDetailsMap::iterator fdItr = _fileDetailsMap.find(fd->getFileName());
    if (fdItr != _fileDetailsMap.end())
    {
        _fileDetailsMap.erase(fdItr);
    }

    VariantMap::iterator itr = _variantMap.find(fd->getOriginalSourceFileName());
    if (itr==_variantMap.end()) return;

    Variants& variants = itr->second;
    for(Variants::iterator vitr = variants.begin();
        vitr != variants.end();
        ++vitr)
    {
        if (*vitr == fd)
        {
            variants.erase(vitr);
            return;
        }
    }
    
}

bool FileCache::getSpatialProperties(const std::string& filename, SpatialProperties& sp)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_variantMapMutex);

    FileDetailsMap::iterator itr = _fileDetailsMap.find(filename);
    if (itr != _fileDetailsMap.end())
    {
        sp = itr->second->getSpatialProperties();
        return true;
    }
    else
    {
        return false;
    }
}

std::string FileCache::getOptimimumFile(const std::string& filename, const osg::CoordinateSystemNode* csn)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_variantMapMutex);

    VariantMap::iterator itr = _variantMap.find(filename);
    if (itr==_variantMap.end())
    {
        log(osg::NOTICE,"FileCache::getOptimimumFile(%s) no variants found returning '%s'",filename.c_str(),filename.c_str());
        return filename;
    }
    
    Variants& variants = itr->second;

    FileDetails* fd_closest = 0;
    double res_closest = DBL_MAX;
    

    // first check cached files on
    std::string hostname = getLocalHostName();
    for(Variants::iterator vitr = variants.begin();
        vitr != variants.end();
        ++vitr)
    {
        FileDetails* fd = vitr->get();
        const SpatialProperties& fd_sp = fd->getSpatialProperties();
        if (vpb::areCoordinateSystemEquivalent(fd_sp._cs.get(), csn))
        {
            double res = fd_sp.computeResolution();
            if (res<res_closest || (res==res_closest && fd->getHostName()==hostname))
            {
                fd_closest = fd;
                res_closest = res;
            }
        }
    }
    
    if (fd_closest) return fd_closest->getFileName();

    // osg::notify(osg::NOTICE)<<"FileCache::getOptimimumFile("<<filename<<") no suitable variants found returning ''"<<std::endl;
    return std::string();
}

std::string FileCache::getOptimimumFile(const std::string& filename, const SpatialProperties& sp)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_variantMapMutex);
    
    osg::NotifySeverity level = osg::INFO;

    VariantMap::iterator itr = _variantMap.find(filename);
    if (itr==_variantMap.end())
    {
        log(level,"FileCache::getOptimimumFile(%s) no variants found returning '%s'",filename.c_str(),filename.c_str());
        return filename;
    }
    
    Variants& variants = itr->second;

    FileDetails* fd_closest_below = 0;
    double res_closest_below = -DBL_MAX;
    
    FileDetails* fd_closest_above = 0;
    double res_closest_above = DBL_MAX;
    
    // first check cached files on
    std::string hostname = getLocalHostName();
    for(Variants::iterator vitr = variants.begin();
        vitr != variants.end();
        ++vitr)
    {
        FileDetails* fd = vitr->get();
        const SpatialProperties& fd_sp = fd->getSpatialProperties();
        if (fd_sp.compatible(sp))
        {
            double resolutionRatio = fd_sp.computeResolutionRatio(sp);
            if (resolutionRatio < 1.0)
            {
                if ( resolutionRatio > res_closest_below || 
                    (resolutionRatio==res_closest_below && fd->getHostName()==hostname))
                {
                    fd_closest_below = fd;
                    res_closest_below = resolutionRatio;
                }
            }
            else if (resolutionRatio>=1.0)
            {
                if (resolutionRatio < res_closest_above || 
                    (resolutionRatio==res_closest_above && fd->getHostName()==hostname))
                {
                    fd_closest_above = fd;
                    res_closest_above = resolutionRatio;
                }
            }
        }
        else
        {
#if 0
            log(level,"  FileDetails(%s) not compatible ",fd->getFileName().c_str());
            log(level,"  FileDetails fd(%f,\t%f,\t%f,\t%f)",fd_sp._extents.xMin(), fd_sp._extents.xMax(), fd_sp._extents.yMin(), fd_sp._extents.yMax());
            log(level,"              sp(%f,\t%f,\t%f,\t%f)",sp._extents.xMin(), sp._extents.xMax(), sp._extents.yMin(), sp._extents.yMax());
#endif
        }
    }
    
    if (fd_closest_above)
    {
        if (fd_closest_above->getHostName()==hostname)
        {
            log(level,"FileCache::getOptimimumFile(%s) found local closest_above variant '%s'",filename.c_str(),fd_closest_above->getFileName().c_str());
        }
        else
        {
            log(level,"FileCache::getOptimimumFile(%s) found remote closest_above variant '%s'",filename.c_str(),fd_closest_above->getFileName().c_str());
        }
        return fd_closest_above->getFileName();
    }
    
    if (fd_closest_below)
    {
        if (fd_closest_below->getHostName()==hostname)
        {
            log(level,"FileCache::getOptimimumFile(%s) found local fd_closest_below variant '%s'",filename.c_str(),fd_closest_below->getFileName().c_str());
        }
        else
        {
            log(level,"FileCache::getOptimimumFile(%s) found remote fd_closest_below variant '%s'",filename.c_str(),fd_closest_below->getFileName().c_str());
        }
        return fd_closest_below->getFileName();
    }

    log(level,"FileCache::getOptimimumFile(%s) no suitable variants found returning ''",filename.c_str());
    return std::string();
}

void FileCache::clear()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_variantMapMutex);

    _requiresWrite = true;
    
    _variantMap.clear();
    
    log(osg::NOTICE,"FileCache::clear()");
}

void FileCache::addSource(osgTerrain::TerrainTile* source)
{
    if (!source) return;

    osg::ref_ptr<DataSet> dataset = new DataSet;
    dataset->addTerrain(source);

    for(CompositeSource::source_iterator itr(dataset->getSourceGraph());itr.valid();++itr)
    {
        (*itr)->loadSourceData();
        Source* source = itr->get();
        SourceData* sd = (*itr)->getSourceData();
        
        
        FileDetails* fd = new FileDetails;
        fd->setOriginalSourceFileName(source->getFileName());
        fd->setFileName(source->getFileName());
        fd->setSpatialProperties(*sd);
        
        addFileDetails(fd);
    }
    
    log(osg::NOTICE,"FileCache::addSource()");
}

void FileCache::buildRequiredReprojections(osgTerrain::TerrainTile* source)
{
    if (!source) return;

    log(osg::NOTICE,"FileCache::buildRequiredReprojections()");

    osg::ref_ptr<DataSet> dataset = new DataSet;
    dataset->addTerrain(source);

    dataset->assignIntermediateCoordinateSystem();
    
    std::string filePrefix("reprojected_file_");
    std::string localHostName(getLocalHostName());
    
    if (System::instance()->getMachinePool())
    {
        Machine* machine = System::instance()->getMachinePool()->getMachine(localHostName);
        if (machine)
        {
            std::string cacheDirectory = machine->getCacheDirectory();
            
            if (!cacheDirectory.empty())
            {
                filePrefix = cacheDirectory + "/";
            }
        }
    }

    log(osg::NOTICE,"FileCache::buildRequiredReprojections() : filePrefix = %s",filePrefix.c_str());

    if (dataset->requiresReprojection())
    {
        for(CompositeSource::source_iterator itr(dataset->getSourceGraph());itr.valid();++itr)
        {
            Source* source = itr->get();
            if (source->needReproject(dataset->getIntermediateCoordinateSystem()) && source->isRaster())
            {
                std::string newFileName = filePrefix + osgDB::getStrippedName(source->getFileName()) + ".tif";

                log(osg::NOTICE,"     reprojecting file=%s, reprojected file will be = %s",source->getFileName().c_str(), newFileName.c_str());

                osg::ref_ptr<Source> newSource = source->doRasterReprojection(newFileName,dataset->getIntermediateCoordinateSystem());

                if (newSource.valid())
                {
                
                    SourceData* sd = newSource->getSourceData();

                    FileDetails* fd = new FileDetails;
                    fd->setOriginalSourceFileName(source->getFileName());
                    fd->setFileName(newSource->getFileName());
                    fd->setSpatialProperties(*sd);

                    fd->setHostName(localHostName);

                    addFileDetails(fd);
                    
                    *itr = newSource;
                }
            }
        }
    }

}


void FileCache::buildOverviews(osgTerrain::TerrainTile* source)
{

    log(osg::NOTICE,"FileCache::buildOverviews()");

    if (!source) return;

    _requiresWrite = true;


    osg::ref_ptr<DataSet> dataset = new DataSet;
    dataset->addTerrain(source);

    dataset->assignIntermediateCoordinateSystem();
    
    std::string filePrefix("multiresolution_file_");
    
    if (System::instance()->getMachinePool())
    {
        Machine* machine = System::instance()->getMachinePool()->getMachine(getLocalHostName());
        if (machine)
        {
            std::string cacheDirectory = machine->getCacheDirectory();
            
            if (!cacheDirectory.empty())
            {
                filePrefix = cacheDirectory + "/";
            }
        }
    }


    log(osg::NOTICE,"FileCache::buildOverviews() : filePrefix = %s",filePrefix.c_str());

    osg::CoordinateSystemNode* csn = dataset->getIntermediateCoordinateSystem();

    for(CompositeSource::source_iterator itr(dataset->getSourceGraph());itr.valid();++itr)
    {
        Source* source = itr->get();

        VariantMap::iterator vmitr = _variantMap.find(source->getFileName());
        if (vmitr != _variantMap.end())
        {
            Variants& variants = vmitr->second;

            typedef std::list<FileDetails*> FileDetailsList;
            FileDetailsList fileDetailsWithRequiredCoordinateSystem;

            for(Variants::iterator vitr = variants.begin();
                vitr != variants.end();
                ++vitr)
            {
                FileDetails* fd = vitr->get();
                const SpatialProperties& fd_sp = fd->getSpatialProperties();
                if (vpb::areCoordinateSystemEquivalent(fd_sp._cs.get(), csn))
                {
                    fileDetailsWithRequiredCoordinateSystem.push_back(fd);
                }
            }
            
            if (fileDetailsWithRequiredCoordinateSystem.size()==1)
            {
                FileDetails* fd = fileDetailsWithRequiredCoordinateSystem.front();

                osg::ref_ptr<GeospatialDataset> dataset = System::instance()->openGeospatialDataset(fd->getFileName(), READ_AND_WRITE);
                if (dataset.valid() )
                {
                    if (!dataset->containsOverviews())
                    {
                        log(osg::NOTICE, "     need to build mipmaps for %s",fd->getFileName().c_str());

                        int anOverviewList[5] = { 2, 4, 8, 16, 32 };
                        dataset->BuildOverviews( "AVERAGE", 5, anOverviewList, 0, NULL,
                                                 GDALTermProgress/*GDALDummyProgress*/, NULL );

                    }

                }
                
            }
            
        }

    }

}

void FileCache::mirror(Machine* machine, osgTerrain::TerrainTile* source)
{
    log(osg::NOTICE,"FileCache::mirror(%s)",machine->getHostName().c_str());

    if (!source)
    {
        log(osg::NOTICE,"Error: cannot mirror without specification of required sources.");
        return;
    }

    _requiresWrite = true;

    osg::ref_ptr<DataSet> dataset = new DataSet;
    dataset->addTerrain(source);

    dataset->assignIntermediateCoordinateSystem();

    if (machine->getCacheDirectory().empty())
    {
        log(osg::NOTICE,"Error not cache directory on machine '%s' to mirror files on.",machine->getHostName().c_str());
        return;
    }

    std::string localHostName = getLocalHostName();
    osg::CoordinateSystemNode* csn = dataset->getIntermediateCoordinateSystem();

    for(CompositeSource::source_iterator itr(dataset->getSourceGraph());itr.valid();++itr)
    {
        Source* source = itr->get();

        VariantMap::iterator vmitr = _variantMap.find(source->getFileName());
        if (vmitr != _variantMap.end())
        {
            Variants& variants = vmitr->second;

            typedef std::list<FileDetails*> FileDetailsList;
            FileDetailsList fileDetailsWithRequiredCoordinateSystem;

            FileDetails* fileOnLocalMachine = 0;
            FileDetails* fileOnTargetMachine = 0;
            
            for(Variants::iterator vitr = variants.begin();
                vitr != variants.end() && !fileOnTargetMachine;
                ++vitr)
            {
                FileDetails* fd = vitr->get();
                const SpatialProperties& fd_sp = fd->getSpatialProperties();
                if (vpb::areCoordinateSystemEquivalent(fd_sp._cs.get(), csn))
                {
                    if (fd->getHostName() == machine->getHostName())
                    {
                        fileOnTargetMachine = fd;
                    }
                    else if (fd->getHostName()==localHostName)
                    {
                        fileOnLocalMachine = fd;
                    }
                    else
                    {
                        fileDetailsWithRequiredCoordinateSystem.push_back(fd);
                    }
                }
            }
            
            if (fileOnTargetMachine)
            {
                log(osg::NOTICE,"  File %s already on target machine, no need to copy.",fileOnTargetMachine->getFileName().c_str());
            }
            else if (fileOnLocalMachine)
            {
                copyFileToMachine(fileOnLocalMachine, machine);
            }
            else if (!fileDetailsWithRequiredCoordinateSystem.empty())
            {
                copyFileToMachine(fileDetailsWithRequiredCoordinateSystem.front(), machine);
            }
            else
            {
                log(osg::NOTICE,"  No version of source file '%s' with the required coordinate system found, unable to copy.",source->getFileName().c_str());
            }
            
        }

    }

}

bool FileCache::copyFileToMachine(FileDetails* fd, Machine* machine)
{
    log(osg::NOTICE,"Copying file '%s' to machine '%s'.",fd->getFileName().c_str(), machine->getHostName().c_str());
    
    std::string filePrefix( machine->getCacheDirectory() + std::string("/") );
    
    if (filePrefix.empty())
    {
        log(osg::NOTICE,"Error: cannot mirror without a valid cache directory on specified machine.");
        return false;
    }
    
    std::string newFileName = filePrefix + osgDB::getSimpleFileName(fd->getFileName());

    std::ostringstream app;
    app<<"cp "<<fd->getFileName()<<" "<<newFileName;
    
    std::string appstr( app.str() );
    
    int result = system( appstr.c_str() );
    
    if (result==0)
    {

        FileDetails* new_fd = new FileDetails;
        new_fd->setOriginalSourceFileName(fd->getOriginalSourceFileName());
        new_fd->setFileName(newFileName);
        new_fd->setSpatialProperties(fd->getSpatialProperties());
        new_fd->setHostName(machine->getHostName());
        addFileDetails(new_fd);
        
        return true;
    }
    else
    {
        log(osg::NOTICE,"Error: cannot copy file '%s' to specified machine '%s'.", fd->getFileName().c_str(), machine->getHostName().c_str());
        return false;
    }

}

void FileCache::report(std::ostream& out)
{
    for(VariantMap::iterator itr = _variantMap.begin();
        itr != _variantMap.end();
        ++itr)
    {
        out<<"Variants of "<<itr->first<<" {"<<std::endl;
        Variants& variants = itr->second;
        for(Variants::iterator vitr = variants.begin();
            vitr != variants.end();
            ++vitr)
        {
            FileDetails* fd = vitr->get();
            
            out<<"  FileDetails {"<<std::endl;
            
            if (!fd->getHostName().empty())
            {
                out<<"    hostname "<<fd->getHostName()<<std::endl;
            }

            if (!fd->getBuildApplication().empty())
            {
                out<<"    build "<<fd->getBuildApplication()<<std::endl;
            }

            if (!fd->getOriginalSourceFileName().empty())
            {
                out<<"    original "<<fd->getOriginalSourceFileName()<<std::endl;
            }
            
            if (!fd->getFileName().empty())
            {
                out<<"    file "<<fd->getFileName()<<std::endl;
            }
            
            if (fd->getSpatialProperties()._cs.valid() && !fd->getSpatialProperties()._cs->getCoordinateSystem().empty())
            {
                out<<"    cs "<<fd->getSpatialProperties()._cs->getCoordinateSystem()<<std::endl;
            }

            if (fd->getSpatialProperties()._extents.valid())
            {
                const GeospatialExtents& extents = fd->getSpatialProperties()._extents;
                out<<"    extents "<<extents.xMin()<<" "<<extents.yMin()<<" "<<extents.xMax()<<" "<<extents.yMax()<<std::endl;
            }
            
            if (!fd->getSpatialProperties()._geoTransform.isIdentity())
            {
                const osg::Matrixd& m = fd->getSpatialProperties()._geoTransform;
                out<<"    geoTransform "<<m(0,0)<<" "<<m(0,1)<<" "<<m(1,0)<<" "<<m(1,1)<<" "<<m(3,0)<<" "<<m(3,1)<<std::endl;
            }
            
            if (fd->getSpatialProperties()._numValuesX>0 || fd->getSpatialProperties()._numValuesY>0 || fd->getSpatialProperties()._numValuesZ>0 )
            {
                out<<"    size "<<fd->getSpatialProperties()._numValuesX<<" "<<fd->getSpatialProperties()._numValuesY<<" "<<fd->getSpatialProperties()._numValuesZ<<std::endl;
            }
            
            out<<"  }"<<std::endl;
                        
        }
        out<<"}"<<std::endl;
    }
}

