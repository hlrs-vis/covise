/* This file is part of COVISE.

  You can use it under the terms of the GNU Lesser General Public License
  version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

/****************************************************************************\
 **                                                          (C)2020 HLRS  **
 **                                                                        **
 ** Description: OpenCOVER Plug-In for reading Luftdaten sensor data       **
 **                                                                        **
 **                                                                        **
 ** Author: Leyla Kern                                                     **
 **                                                                        **
 ** History:                                                               **
 ** April 2020  v1                                                         **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include "Luftdaten.h"
#include <osg/LineWidth>
#include <osg/Version>
#include <cover/coVRTui.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRAnimationManager.h>
#include <boost/tokenizer.hpp>
#include <boost/filesystem.hpp>
#include <proj_api.h>

LuftdatenPlugin *LuftdatenPlugin::plugin = NULL;

std::string proj_to ="+proj=utm +zone=32 +ellps=GRS80 +units=m +no_defs ";
std::string proj_from = "+proj=latlong";
float offset [] = {-507048.f,-5398554.9,-450};

LuftdatenPlugin::LuftdatenPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("LuftdatenPlugin", cover->ui)
{
    fprintf(stderr, "Starting Luftdaten Plugin\n");
    plugin = this;
    
    LuftdatenGroup = new osg::Group();
    LuftdatenGroup->setName("Luftdaten");
    cover->getObjectsRoot()->addChild(LuftdatenGroup);
    
    sequenceList = new osg::Sequence();
    sequenceList->setName("Timesteps");
    LuftdatenGroup->addChild(sequenceList);
    
    coLuftTab = new coTUITab("Sensor Graph",coVRTui::instance()->mainFolder->getID());
    coLuftTab->setPos(0,0);
    WebView = new coTUIWebview("PlotView",coLuftTab->getID());
    WebView->setPos(10,10);
    WebView->setEventListener(this);
    loaded = false;

    GDALAllRegister();
    std::string imagefile = "/data/StuttgartForecast/LGL_Data/AllStuttgart_10x10HeightLATLONG.tif";
    openImage(imagefile);

    SDlist.clear();

    struct tm iTime;
    std::sscanf("2019-01-21T04:00:00", "%d-%d-%dT%d:%d:%d",&iTime.tm_year,&iTime.tm_mon,&iTime.tm_mday,&iTime.tm_hour,&iTime.tm_min,&iTime.tm_sec);
    initialTime = mktime(&iTime);
    
    LuftTab = new ui::Menu("Luftdaten",LuftdatenPlugin::plugin);
    LuftTab->setText("Luftdaten");
    
    ShowGraph = new ui::Button(LuftTab,"ShowGraph");
    ShowGraph->setText("Show Graphs");
    ShowGraph->setCallback([this] (bool){
        if (graphVisible)
            graphVisible = false;
        else
            graphVisible = true;
         ShowGraph->setState(graphVisible);
    });
    
    componentGroup = new ui::ButtonGroup(LuftTab,"ComponentGroup");
    componentGroup->setDefaultValue(PM10);
    componentList = new ui::Group(LuftTab, "Component");
    componentList->setText("Component");
    pm10Bt = new ui::Button(componentList, "PM10", componentGroup, PM10);
    pm2Bt = new ui::Button(componentList, "PM2", componentGroup, PM2) ;
    tempBt = new ui::Button(componentList, "Temperature", componentGroup, Temp);
    humiBt = new ui::Button(componentList, "Humidity", componentGroup, Humi);
    componentGroup->setCallback([this](int value){
        setComponent(Components(value));
    });
}

void LuftdatenPlugin::setComponent(Components c)
{
    switch (c) {
        case PM10:
            pm10Bt->setState(true,false);
            break;
        case PM2:
            pm2Bt->setState(true,false);
            break;
        case Temp:
            tempBt->setState(true,false);
            break;
        case Humi:
            humiBt->setState(true,false);
            break;
        default:
            break;
    }
    selectedComp = c;
    for (auto s : SDlist) {
        if (s.second.empty())
            continue;
        for (auto t : s.second)
            t->init(rad,scaleH,selectedComp);
    }
}

LuftdatenPlugin::~LuftdatenPlugin()
{
    
}
bool LuftdatenPlugin::init()
{
    if (loadFile("/data/StuttgartForecast/luftdaten/SDS_2019-01-21_0400-1059.csv"))
    {
        fprintf(stderr, "Luftdaten Plugin: SDS file loaded \n");
    }
    if (loadFile("/data/StuttgartForecast/luftdaten/DHT_2019-01-21_0400-1059.csv"))
    {
        fprintf(stderr, "Luftdaten Plugin: DHT file loaded \n");
    }
    
    mergeTimesteps();
    
    if ((int)sequenceList->getNumChildren() > coVRAnimationManager::instance()->getNumTimesteps())
    {
        coVRAnimationManager::instance()->setNumTimesteps(sequenceList->getNumChildren(),sequenceList);
    }

    rad = 15.;
    scaleH = 10.;

    for (auto s : SDlist) {
        if (s.second.empty())
            continue;
        for (auto t : s.second)
            t->init(rad,scaleH,selectedComp);
    }
    return true;
}

int LuftdatenPlugin::mergeTimesteps()
{
    int numTimesteps = sequenceList->getNumChildren();

    for (auto vec:SDlist) {
        if (vec.second.size() <= 1)
            continue;
        
        std::vector<float>  pm10_(numTimesteps, 0.f);
        std::vector<float>  pm2_(numTimesteps, 0.f);
        std::vector<float>  humi_(numTimesteps, 0.f);
        std::vector<float>  temp_(numTimesteps, 0.f);
        
        std::vector<std::vector<int>> count(4,std::vector<int>(numTimesteps,0));
        std::string senName="NONE";

        for (auto elem: vec.second)
        {
            int t = elem->devInfo->timestep;

            if (elem->devInfo->pm10 > 0.f)
            {
                pm10_[t] +=elem->devInfo->pm10;
                count[0][t]++;
                senName = elem->devInfo->name;
            }
            if (elem->devInfo->pm2 > 0.f)
            {
                pm2_[t] +=elem->devInfo->pm2;
                count[1][t]++;
            }
            if (elem->devInfo->temp > -100.f)
            {
                temp_[t] +=elem->devInfo->temp;
                count[2][t]++;
            }
            if (elem->devInfo->humi > 0.f)
            {
                humi_[t] +=elem->devInfo->humi;
                count[3][t]++;
            }
        }
        std::string ID_ = vec.first;
        float lat_ = vec.second.front()->devInfo->lat;
        float lon_ = vec.second.front()->devInfo->lon;
        float height_ = vec.second.front()->devInfo->height;

        SDlist[ID_].clear();

        for (int t = 0; t < numTimesteps; ++t)
        {
            if ((count[0][t] > 0) || (count[1][t] > 0) || (count[1][t] > 0) || (count[2][t] > 0))
            {
                std::string groupName = "timestep"+std::to_string(t);
                bool foundGroup = false;
                osg::Group * timestepGroup = new osg::Group() ;
                for (unsigned int i = 0; i < sequenceList->getNumChildren(); ++i)
                {
                   if (strcmp(sequenceList->getChild(i)->getName().c_str(),groupName.c_str()) == 0)
                   {
                       timestepGroup = sequenceList->getChild(i)->asGroup();
                       foundGroup = true;
                       break;
                   }
                }
                if (foundGroup)
                {
                    DeviceInfo * di = new DeviceInfo();
                    di->ID = ID_;
                    di->lat = lat_;
                    di->lon = lon_;
                    di->height = height_;
                    if (strcmp(senName.c_str(),"NONE")!=0)
                        di->name = senName;
                    di->pm10 = count[0][t] > 0 ? pm10_[t]/count[0][t] : -1.;
                    di->pm2 = count[1][t] > 0 ? pm2_[t]/count[1][t] : -1.f;
                    di->temp = count[2][t] > 0 ? temp_[t]/count[2][t] : -100.f;
                    di->humi = count[3][t] > 0 ? humi_[t]/count[3][t] : -1.f;
                    
                    Device * avgDev = new Device(di, timestepGroup);
                    SDlist[di->ID].push_back(avgDev);
                }
            }
        }
    }
    return SDlist.size();
}

void LuftdatenPlugin::closeImage()
{
    if (heightDataset)
        GDALClose(heightDataset);
}

void LuftdatenPlugin::openImage(std::string &name)
{
    heightDataset = (GDALDataset *)GDALOpen(name.c_str(), GA_ReadOnly);
    if (heightDataset != NULL)
    {
        int nBlockXSize, nBlockYSize;
        int bGotMin, bGotMax;
        double adfMinMax[2];
        double adfGeoTransform[6];

        printf("Size is %dx%dx%d\n", heightDataset->GetRasterXSize(), heightDataset->GetRasterYSize(), heightDataset->GetRasterCount());
        if (heightDataset->GetGeoTransform(adfGeoTransform) == CE_None)
        {
            printf("Origin = (%.6f,%.6f)\n", adfGeoTransform[0], adfGeoTransform[3]);
            printf("Pixel Size = (%.6f,%.6f)\n", adfGeoTransform[1], adfGeoTransform[5]);
        }
        int numRasterBands = heightDataset->GetRasterCount();

        heightBand = heightDataset->GetRasterBand(1);
        heightBand->GetBlockSize(&nBlockXSize, &nBlockYSize);
        cols = heightDataset->GetRasterXSize();
        rows = heightDataset->GetRasterYSize();
        double transform[100];
        heightDataset->GetGeoTransform(transform);

        xOrigin = transform[0];
        yOrigin = transform[3];
        pixelWidth = transform[1];
        pixelHeight = -transform[5];
        delete[] rasterData;
        rasterData = new float[cols*rows];
        float *pafScanline;
        int nXSize = heightBand->GetXSize();
        pafScanline = (float *)CPLMalloc(sizeof(float)*nXSize);
        for (int i = 0; i < rows; i++)
        {
            if (heightBand->RasterIO(GF_Read, 0, i, nXSize, 1,
                pafScanline, nXSize, 1, GDT_Float32,
                0, 0) == CE_Failure)
            {
                std::cerr << "LuftdatenPlugin::openImage: GDALRasterBand::RasterIO failed" << std::endl;
                break;
            }
            memcpy(&(rasterData[(i*cols)]), pafScanline, nXSize * sizeof(float));
        }

        if (heightBand->ReadBlock(0, 0, rasterData) == CE_Failure)
        {
            std::cerr << "LuftdatenPlugin::openImage: GDALRasterBand::ReadBlock failed" << std::endl;
            return;
        }

        adfMinMax[0] = heightBand->GetMinimum(&bGotMin);
        adfMinMax[1] = heightBand->GetMaximum(&bGotMax);
        if (!(bGotMin && bGotMax))
                GDALComputeRasterMinMax((GDALRasterBandH)heightBand, TRUE, adfMinMax);

        printf("Min=%.3fd, Max=%.3f\n", adfMinMax[0], adfMinMax[1]);
    }else {
        mapAlt = false;
        fprintf(stderr, "Luftdaten Plugin: Ignoring geotiff - file not found \n");
    }
}

float LuftdatenPlugin::getAlt(double x, double y)
{
    int col = int((x - xOrigin) / pixelWidth);
    int row = int((yOrigin - y) / pixelHeight);
    if (col < 0)
        col = 0;
    if (col >= cols)
        col = cols-1;
    if (row < 0)
        row = 0;
    if (row >= rows)
        row = rows - 1;
    return rasterData[col + (row*cols)];
    float *pafScanline;
    int   nXSize = heightBand->GetXSize();

    delete[] pafScanline;
    pafScanline = new float[nXSize];
    auto err = heightBand->RasterIO(GF_Read, x, y, 1, 1, pafScanline, nXSize, 1, GDT_Float32, 0, 0);
    float height = pafScanline[0];
    delete[] pafScanline;

    if (err != CE_None)
    {
        std::cerr << "LuftdatenPlugin::getAlt: error" << std::endl;
        return 0.f;
    }
    return height;
}

bool LuftdatenPlugin::loadFile(std::string fileName)
{
    FILE *fp = fopen(fileName.c_str(), "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Luftdaten Plugin: could not open file\n");
        return false;
    }
    
    const int lineSize = 1000;
    char buf[lineSize];
    
    bool mapdrape = true;
    projPJ pj_from = pj_init_plus(proj_from.c_str());
    projPJ pj_to = pj_init_plus(proj_to.c_str());
    if (!pj_from || !pj_to)
    {
        fprintf(stderr, "Luftdaten Plugin: Ignoring mapping. No valid projection was found \n");
        mapdrape = false;
    }
    
    int timestep = -1;
    int ye, ho, mo, mi, da, se;
    time_t previousTime = initialTime-TIME_INTERVAL-1;
           
    std::string sensorType;
    osg::Group * timestepGroup;
    std::string groupName = "timestep"+std::to_string(timestep);
    
    if (!fgets(buf,lineSize,fp))
    {
        fclose(fp);
        return false;
    }

    bool firstLine = true;
    
    while (!feof(fp))
    {
        if (!fgets(buf, lineSize, fp))
            break;
        std::string line(buf);
        boost::char_separator<char> sep(";");
        boost::tokenizer<boost::char_separator<char>> tokens(line, sep);
        auto tok = tokens.begin();
        std::string senName = tok->c_str();
        
        DeviceInfo * di = new DeviceInfo();

        tok++;
        sensorType = tok->c_str();
        tok++;
        di->ID= tok->c_str();
        tok++;
        
        if (mapdrape)
        {
            double xlat = std::strtod(tok->c_str(), NULL);
            tok++;
            double xlon = std::strtod(tok->c_str(), NULL);
            tok++;
            float alt = 0.;
            if (mapAlt)
                alt = getAlt(xlon, xlat);
            xlat *= DEG_TO_RAD;
            xlon *= DEG_TO_RAD;
            
            int error = pj_transform(pj_from, pj_to, 1, 1, &xlon, &xlat, NULL);
            
            di->lat = xlon+offset[0];
            di->lon = xlat+offset[1];
            di->height = alt + offset[2];
        }else{
            di->lat = std::strtof(tok->c_str(), NULL);
            tok++;
            di->lon = std::strtof(tok->c_str(), NULL);
            tok++;
            di->height = 0.f; 
        }
        
        di->time = tok->c_str();
        tok++;
        
        if (strcmp(sensorType.c_str(), "SDS011") == 0)
        {
            di->pm10 = std::strtof(tok->c_str(), NULL);
            tok++;
            tok++;
            tok++;
            di->pm2 = std::strtof(tok->c_str(), NULL);
            tok++;
            di->name = senName;
        }else if (strcmp(sensorType.c_str(), "DHT22") == 0)
        {
            di->temp = std::strtof(tok->c_str(), NULL);
            tok++;
            di->humi = std::strtof(tok->c_str(), NULL);
            tok++;
        }
        
        struct tm sTime = {0};
        std::sscanf(di->time.c_str(), "%d-%d-%dT%d:%d:%d",&ye,&mo,&da,&ho,&mi,&se);
        sTime.tm_year = ye;
        sTime.tm_mon = mo;
        sTime.tm_mday = da;
        sTime.tm_hour = ho;
        sTime.tm_min = mi;
        sTime.tm_sec = se;
        time_t newTime = mktime(&sTime);

        if (firstLine && strcmp(sensorType.c_str(), "SDS011") == 0)
        {
           initialTime = newTime;
           previousTime = initialTime-TIME_INTERVAL-1;
           firstLine = false;
        }
        if (abs(difftime(newTime, previousTime)) > TIME_INTERVAL)
        {
            timestep++;
            groupName = "timestep"+std::to_string(timestep);
            bool foundGroup = false;
            for (unsigned int i = 0; i < sequenceList->getNumChildren(); ++i)
            {
                if (strcmp(sequenceList->getChild(i)->getName().c_str(),groupName.c_str()) == 0)
                {
                    timestepGroup = sequenceList->getChild(i)->asGroup();
                    foundGroup = true;
                    break;
                }
            }
            if (!foundGroup)
            {
                timestepGroup = new osg::Group();
                timestepGroup->setName(groupName);
                sequenceList->addChild(timestepGroup);
                sequenceList->setValue(timestep);
            }
            previousTime = initialTime + (timestep+1) * TIME_INTERVAL;
        }
        
        di->timestep = timestep ;
        Device *sd = new Device(di,timestepGroup);
        SDlist[di->ID].push_back(sd);
    }
    fclose(fp);
    return true;
}
bool LuftdatenPlugin::update()
{
    if (!loaded) {
        WebView->setURL("https://luftdaten.info");
        loaded = true;
    }
    for (auto s = SDlist.begin(); s != SDlist.end();s++)
    {
        if (s->second.empty())
            continue;
        for (auto timeElem : s->second)
        {
            timeElem->update();
            if (timeElem->getStatus() && graphVisible)
            {
                if (strcmp(timeElem->devInfo->name.c_str(), "NONE") != 0)
                {
                    std::string newURL = "https://maps.sensor.community/grafana/d-solo/000000004/single-sensor-view?orgId=1&panelId=2&var-node="+timeElem->devInfo->name;
                    WebView->setURL(newURL);
                }
            }
        }
    }
    return false;
}
void LuftdatenPlugin::setTimestep(int t)
{
    sequenceList->setValue(t);
}

bool LuftdatenPlugin::destroy()
{
    cover->getObjectsRoot()->removeChild(LuftdatenGroup);
    return false;
}

COVERPLUGIN(LuftdatenPlugin)
