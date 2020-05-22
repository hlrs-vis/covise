/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/******************************************************************
 *
 *    ReadCSVTime
 *
 *
 *  Description: Read CSV files containing timesteps/ timestamps
 *  Date: 02.06.19
 *  Author: Leyla Kern
 *
 *******************************************************************/


#include "ReadCSVTime.h"

#ifndef _WIN32
#include <inttypes.h>
#endif

#include <sys/types.h>
#include <sys/stat.h>

#include <errno.h>

#include <do/coDoData.h>
#include <do/coDoSet.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoPoints.h>

// remove  path from filename
inline const char *coBasename(const char *str)
{
    const char *lastslash = strrchr(str, '/');
    const char *lastslash2 = strrchr(str, '\\');
    if (lastslash2 > lastslash)
        lastslash = lastslash2;
    if (lastslash)
        return lastslash + 1;
    else
    {
        return str;
    }
}

// Module set-up in Constructor
ReadCSVTime::ReadCSVTime(int argc, char *argv[])
    : coReader(argc, argv, "Read CSV data with timestamps")
{
    x_col = addChoiceParam("x_col", "Select column for x-coordinates");
    y_col = addChoiceParam("y_col", "Select column for y-coordinates");
    z_col = addChoiceParam("z_col", "Select column for z-coordinates");
    ID_col = addChoiceParam("ID","Select column for ID");
    time_col = addChoiceParam("timestamp","Select column for timestamp");
    interval_size = addInt32Param("time_interval","Interval length in seconds");
    interval_size->setValue(1);
    d_dataFile = NULL;

    const char *dFormatChoice[] = {"2019-01-01T08:15:00","1/1/2019 8:15","01.01.2019 08:15:00","2019-01-01"};
    p_dateFormat = addChoiceParam("DateFormat","Select format of datetime");
    p_dateFormat->setValue(sizeof(dFormatChoice)/sizeof(dFormatChoice[0]), dFormatChoice, 0);
}

ReadCSVTime::~ReadCSVTime()
{
}

// param callback read header again after all changes
void
ReadCSVTime::param(const char *paramName, bool inMapLoading)
{

    FileItem *fii = READER_CONTROL->getFileItem(TXT_BROWSER);

    string txtBrowserName;
    if (fii)
    {
        txtBrowserName = fii->getName();
    }

    /////////////////  CALLED BY FILE BROWSER  //////////////////
    if (txtBrowserName == string(paramName))
    {
        FileItem *fi(READER_CONTROL->getFileItem(string(paramName)));
        if (fi)
        {
            coFileBrowserParam *bP = fi->getBrowserPtr();

            if (bP)
            {
                string dataNm(bP->getValue());
                if (dataNm.empty())
                {
                    cerr << "ReadCSVTime::param(..) no data file found " << endl;
                }
                else
                {
                    if (d_dataFile != NULL)
                        fclose(d_dataFile);
                    fileName = dataNm;
                    d_dataFile = fopen(dataNm.c_str(), "r");
                    int result = STOP_PIPELINE;

                    if (d_dataFile != NULL)
                    {
                        result = readHeader();
                        fseek(d_dataFile, 0, SEEK_SET);
                    }

                    if (result == STOP_PIPELINE)
                    {
                        cerr << "ReadCSVTime::param(..) could not read file: " << dataNm << endl;
                    }
                    else
                    {

                        // lists for Choice Labels
                        vector<string> dataChoices;

                        // fill in NONE to READ no data
                        string noneStr("NONE");
                        dataChoices.push_back(noneStr);

                        // fill in all species for the appropriate Ports/Choices
                        for (int i = 0; i < varInfos.size(); i++)
                        {
                            dataChoices.push_back(varInfos[i].name);
                        }
                        if (inMapLoading)
                            return;
                        READER_CONTROL->updatePortChoice(DPORT1_3D, dataChoices);
                        READER_CONTROL->updatePortChoice(DPORT2_3D, dataChoices);
                        READER_CONTROL->updatePortChoice(DPORT3_3D, dataChoices);
                        READER_CONTROL->updatePortChoice(DPORT4_3D, dataChoices);
                        READER_CONTROL->updatePortChoice(DPORT5_3D, dataChoices);
                        if (varInfos.size() > 0)
                            x_col->setValue((int)varInfos.size() + 1, dataChoices, 1);
                        if (varInfos.size() > 1)
                            y_col->setValue((int)varInfos.size() + 1, dataChoices, 2);
                        if (varInfos.size() > 2)
                            z_col->setValue((int)varInfos.size() + 1, dataChoices, 3);
                        if (varInfos.size() > 1)
                        {
                            ID_col->setValue((int)varInfos.size() + 1, dataChoices, 1);
                            time_col->setValue((int)varInfos.size() + 1, dataChoices, 2);
                        }
                        dataChoices.clear();

                       // const char *dFormatChoice[] = {"2019-01-01T08:15:00","1/1/2019 8:15","01.01.2019 08:15:00","2019-01-01"};
                       // p_dateFormat->setValue(sizeof(dFormatChoice)/sizeof(dFormatChoice[0]), dFormatChoice, 0);

                    }
                }
                return;
            }

            else
            {
                cerr << "ReadCSVTime::param(..) BrowserPointer NULL " << endl;
            }
        }
    }
}

void ReadCSVTime::nextLine(FILE *fp)
{
    if (fgets(buf, sizeof(buf), fp) == NULL)
    {
        cerr << "ReadCSVTime::nextLine: fgets failed" << endl;
    }
}

int ReadCSVTime::readHeader()
{

    int ii_choiseVals = 0;

    nextLine(d_dataFile);
    sendInfo("Found header line : %s", buf);

    buf[strlen(buf) - 1] = '\0';

    std::vector<std::string> names;
    varInfos.clear();
    names.clear();
    const char *name = strtok(buf, ";,");
    names.push_back(name);

    while ((name = strtok(NULL, ";,")) != NULL)
    {
        names.push_back(name);
    }

    if (names.size() > 0)
    {

        for (int i = 0; i < names.size(); i++)
        {
            VarInfo vi;
            vi.name = names[i];
            varInfos.push_back(vi);
        }

        sendInfo("Found %lu header elements", (unsigned long)names.size());

        numRows = 0;
        while (fgets(buf, sizeof(buf), d_dataFile) != NULL)
        {
            numRows = numRows + 1;
        }
        coModule::sendInfo("Found %d data lines", numRows);

        rewind(d_dataFile);
        nextLine(d_dataFile);

        return CONTINUE_PIPELINE;
    }
    else
    {
        return STOP_PIPELINE;
    }
}

// taken from old ReadCSVTime module: 2-Pass reading
int ReadCSVTime::readASCIIData()
{
    char *cbuf;
    int res = readHeader();
    int ii, RowCount;
    int CurrRow ;
    float *tmpdat;
    std::string name_extension = "";
    if (res != STOP_PIPELINE)
    {

        int col_for_x = x_col->getValue() - 1;
        int col_for_y = y_col->getValue() - 1;
        int col_for_z = z_col->getValue() - 1;
        int col_for_id = ID_col->getValue() - 1;
        int col_for_time = time_col->getValue() - 1;
        int MAX_TIME_INT = interval_size->getValue();
        int dFormat = p_dateFormat->getValue();


        printf("%d %d %d\n", col_for_x, col_for_y, col_for_z);

        if (col_for_x == -1)
            coModule::sendWarning("No column selected for x-coordinates");
        if (col_for_y == -1)
            coModule::sendWarning("No column selected for y-coordinates");
        if (col_for_z == -1)
            coModule::sendWarning("No column selected for z-coordinates");

        if (col_for_time >= 0)
        {
            has_timestamps = 1;
            name_extension = "_tmp";
        }
        if ((col_for_x >= 0) && (col_for_y >= 0) && (col_for_z >= 0))
        {
            std::string objNameBase = READER_CONTROL->getAssocObjName(MESHPORT3D);
            sprintf(buf, "%s%s", objNameBase.c_str(),name_extension.c_str());
            coDoPoints *grid = new coDoPoints(buf, numRows);

            grid->getAddresses(&xCoords, &yCoords, &zCoords);
        }

        for (int n = 0; n < varInfos.size(); n++)
        {
            varInfos[n].dataObjs = new coDistributedObject *[1];
            varInfos[n].dataObjs[0] = NULL;
            varInfos[n].assoc = 0;
        }
        int portID = 0;
        
        for (int n = 0; n < 5; n++)
        {
            int pos = READER_CONTROL->getPortChoice(DPORT1_3D + n);
            //printf("%d %d\n",pos,n);
            if (pos > 0)
            {
                if (varInfos[pos - 1].assoc == 0)
                {
                    portID = DPORT1_3D + n;
                    sprintf(buf, "%s%s", READER_CONTROL->getAssocObjName(DPORT1_3D + n).c_str(),name_extension.c_str());
                    coDoFloat *dataObj = new coDoFloat(buf, numRows);
                    varInfos[pos - 1].dataObjs[0] = dataObj;
                    varInfos[pos - 1].assoc = 1;
                    dataObj->getAddress(&varInfos[pos - 1].x_d);
                }
                else
                {
                    sendWarning("Column %s already associated to port %d", varInfos[pos - 1].name.c_str(), n);
                }
            }
        }
        tmpdat = (float *)malloc(varInfos.size() * sizeof(float));
        RowCount = 0;
        CurrRow = 0;

        int timeInt = 0;
        float *xValInt, *yValInt, *zValInt;
        std::vector<int> timeIntIdx;
        std::vector<int> NumOfVal;
        char time_str[50];
        struct tm tm={};
        time_t last_t = 0;

        while (fgets(buf, sizeof(buf), d_dataFile) != NULL)
        {
     
            for (int i = 0; i < varInfos.size(); i++)
            {
                tmpdat[i] = 0.;
            }

            if ((cbuf = strtok(buf, ",;")) != NULL)
            {
                if (col_for_time == 0) {
                    sscanf(cbuf, "%[^\n]s", time_str);
                }
                sscanf(cbuf, "%f", &tmpdat[0]);
            }
            else
            {
                coModule::sendWarning("Error parsing line %d", RowCount + 1);
            }

            ii = 0;
            while ((cbuf = strtok(NULL, ";,")) != NULL)
            {
                ii = ii + 1;
                sscanf(cbuf, "%f", &tmpdat[ii]);

                if (ii == col_for_time) sscanf(cbuf, "%[^\n]s", time_str);

            }
            if (has_timestamps != 0)
            {
                if (dFormat == 0)
                {
                    //strptime(time_str, "%Y-%m-%dT%H:%M:%S", &tm);
                    sscanf(time_str, "%d-%d-%dT%d:%d:%d",&tm.tm_year,&tm.tm_mon,&tm.tm_mday,&tm.tm_hour,&tm.tm_min,&tm.tm_sec);
                }else if (dFormat == 1)
                {
                    //strptime(time_str, "%d/%m/%Y %H:%M", &tm);
                    sscanf(time_str, "%d/%d/%d %d:%d",&tm.tm_mday,&tm.tm_mon,&tm.tm_year,&tm.tm_hour,&tm.tm_min);
                }else if (dFormat == 2)
                {
                    //strptime(time_str, "%Y.%m.%dT%H:%M", &tm);
                    sscanf(time_str, "%d.%d.%dT%d:%d",&tm.tm_year,&tm.tm_mon,&tm.tm_mday,&tm.tm_hour,&tm.tm_min);
                }else if (dFormat == 3)
                {
                   //strptime(time_str, "%Y-%m-%d",&tm);
                   sscanf(time_str, "%d-%d-%d",&tm.tm_year,&tm.tm_mon,&tm.tm_mday);
                }
                time_t t = mktime(&tm);
                //printf("TIME: %s\n", ctime(&t));
                if ((difftime(t, last_t) > (MAX_TIME_INT) ) || CurrRow == 0 )
                {
                     timeIntIdx.push_back(CurrRow);
                     last_t = t;
                     timeInt++;
                     for (int i = 0; i < varInfos.size(); i++)
                     {
                         if (varInfos[i].assoc == 1)
                         {
                             varInfos[i].x_d[CurrRow] = tmpdat[i];
                         }
                     }
                     if ((col_for_x >= 0) && (col_for_y >= 0) && (col_for_z >= 0))
                     {
                         xCoords[CurrRow] = tmpdat[col_for_x];
                         yCoords[CurrRow] = tmpdat[col_for_y];
                         zCoords[CurrRow] = tmpdat[col_for_z];
                     }
                     id.push_back(static_cast<int>(tmpdat[col_for_id]));
                     NumOfVal.push_back(1);
                     CurrRow++;

                }else  //check if sensor *id* occurs multiple times in interval
                {
                     auto idx_f = CurrRow+10;
                     int tmp_idx = 1;
                     if (timeIntIdx.size() >= 1)
                     {
                         tmp_idx = timeIntIdx[timeIntIdx.size()-1];
                         auto idx_ptr = std::find( id.begin() + tmp_idx, id.begin() + CurrRow, static_cast<int>(tmpdat[col_for_id]));
                         idx_f = std::distance(id.begin() /*+ tmp_idx*/, idx_ptr);
                     }else if (id[0] == static_cast<int>(tmpdat[col_for_id]))
                     {
                         idx_f = 0;
                     }
                     if (idx_f < CurrRow ) //else if index point to last element -> nothing was found
                     { //double occurence of ID
                         for (int i = 0; i < varInfos.size(); i++)
                         {
                             if (varInfos[i].assoc == 1)
                             {
                                 varInfos[i].x_d[idx_f] += tmpdat[i];
                             }
                         }
                         NumOfVal[idx_f] += 1;
                     }else { //ID does not occure in current interval -> add to interval
                          for (int i = 0; i < varInfos.size(); i++)
                          {
                              if (varInfos[i].assoc == 1)
                              {
                                  varInfos[i].x_d[CurrRow] = tmpdat[i];
                              }
                          }
                          if ((col_for_x >= 0) && (col_for_y >= 0) && (col_for_z >= 0))
                          {
                              xCoords[CurrRow] = tmpdat[col_for_x];
                              yCoords[CurrRow] = tmpdat[col_for_y];
                              zCoords[CurrRow] = tmpdat[col_for_z];
                          }
                          id.push_back(static_cast<int>(tmpdat[col_for_id]));
                          NumOfVal.push_back(1);
                          CurrRow++;
                     }
                }
            }else {
               for (int i = 0; i < varInfos.size(); i++)
               {
                   if (varInfos[i].assoc == 1)
                   {
                       varInfos[i].x_d[RowCount] = tmpdat[i];
                   }
               }
               if ((col_for_x >= 0) && (col_for_y >= 0) && (col_for_z >= 0))
               {
                   xCoords[RowCount] = tmpdat[col_for_x];
                   yCoords[RowCount] = tmpdat[col_for_y];
                   zCoords[RowCount] = tmpdat[col_for_z];
               }
            }

            RowCount = RowCount + 1;
        }
        timeIntIdx.push_back(CurrRow-1);
        if (has_timestamps != 0)
        {
            for(int j = 0; j < NumOfVal.size(); j++ )
            {
                if (NumOfVal[j] > 1)
                {
                    for (int i = 0; i < varInfos.size(); i++)
                    {
                        if (varInfos[i].assoc == 1)
                        {
                            varInfos[i].x_d[j] = varInfos[i].x_d[j]/ NumOfVal[j];

                        }
                    }
                }
            }
            sendInfo("Found %d time intervals",timeInt);
            coDistributedObject **time_outdat = new coDistributedObject *[timeInt+1];
            coDistributedObject **time_outdat_grid = new coDistributedObject *[timeInt+1];

            for (int n = 0; n < 5; n++)
            {    
                 portID = DPORT1_3D + n;
                 int pos = READER_CONTROL->getPortChoice(DPORT1_3D + n);
                 if (pos > 0)
                 {

                     int idx, idx1, numValuesInInt,t;
                     for ( t = 0; t < (timeInt); t++)
                     {
                         idx = timeIntIdx[t];
                         idx1 = timeIntIdx[t+1];
                         numValuesInInt = idx1 - idx+1;
                         float *val;
                         sprintf(buf,"%s_%d",READER_CONTROL->getAssocObjName(DPORT1_3D + n).c_str(),t);
                         coDoFloat *p = new coDoFloat(buf, numValuesInInt);
                         p->getAddress( &val);
                         time_outdat[t]=p;
                         for (int j=0; j<numValuesInInt; j++)
                         {
                            val[j] = varInfos[pos-1].x_d[idx + j]; 
                         }

                     }
                     time_outdat[timeInt] = NULL;
                     coDoSet *outdata = new coDoSet(READER_CONTROL->getAssocObjName(portID).c_str(),time_outdat);
                     varInfos[pos - 1].dataObjs[0] = outdata;
                     sprintf(buf,"1 %d",timeInt);
                     outdata->addAttribute("TIMESTEP",buf);
                }
            }
            int idx, idx1, numValuesInInt,t;
            std::string objNameBase = READER_CONTROL->getAssocObjName(MESHPORT3D);
            for ( t = 0; t < (timeInt); t++)
            {
                     idx = timeIntIdx[t];
                     idx1 = timeIntIdx[t+1];
                     numValuesInInt = idx1 - idx+1;
                     sprintf(buf,"%s_%d",objNameBase.c_str(),t);
                     coDoPoints *gridInt = new coDoPoints(buf, numValuesInInt);
                     time_outdat_grid[t]=gridInt;
                     gridInt->getAddresses(&xValInt,&yValInt,&zValInt);
                     for (int j=0; j<numValuesInInt; j++) 
                     {
                          xValInt[j] = xCoords[idx+j];
                          yValInt[j] = yCoords[idx+j];
                          zValInt[j] = zCoords[idx+j];
                     }
            }
            time_outdat_grid[timeInt] = NULL;
            coDoSet *outdata_grid = new coDoSet(objNameBase.c_str(),time_outdat_grid);
            sprintf(buf,"1 %d",timeInt);
            outdata_grid->addAttribute("TIMESTEP",buf);

            delete [] time_outdat;
            delete [] time_outdat_grid;
            

      }
        //
        free(tmpdat);

        for (int n = 0; n < varInfos.size(); n++)
        {
            delete[] varInfos[n].dataObjs;
            varInfos[n].assoc = 0;
        }


        id.clear();
        has_timestamps = 0;

    }
    return CONTINUE_PIPELINE;
}
int ReadCSVTime::compute(const char *)
{
    if (d_dataFile == NULL)
    {
        if (fileName.empty())
        {
            cerr << "ReadCSVTime::param(..) no data file found " << endl;
        }
        else
        {

            d_dataFile = fopen(fileName.c_str(), "r");
        }
    }
    int result = STOP_PIPELINE;
    has_timestamps = 0;
    if (d_dataFile != NULL)
    {
        result = readASCIIData();
        fclose(d_dataFile);
        d_dataFile = NULL;
    }
    return result;
}

int main(int argc, char *argv[])
{

    // define outline of reader
    READER_CONTROL->addFile(TXT_BROWSER, "data_file_path", "Data file path", ".", "*.csv;*.CSV");

    READER_CONTROL->addOutputPort(MESHPORT3D, "geoOut_3D", "Points", "Geometry", false);

    READER_CONTROL->addOutputPort(DPORT1_3D, "data1_3D", "Float|Vec3", "data1-3d");
    READER_CONTROL->addOutputPort(DPORT2_3D, "data2_3D", "Float|Vec3", "data2-3d");
    READER_CONTROL->addOutputPort(DPORT3_3D, "data3_3D", "Float|Vec3", "data3-3d");
    READER_CONTROL->addOutputPort(DPORT4_3D, "data4_3D", "Float|Vec3", "data4-3d");
    READER_CONTROL->addOutputPort(DPORT5_3D, "data5_3D", "Float|Vec3", "data5-3d");

    // create the module
    coReader *application = new ReadCSVTime(argc, argv);

    // this call leaves with exit(), so we ...
    application->start(argc, argv);

    // ... never reach this point
    return 0;
}
