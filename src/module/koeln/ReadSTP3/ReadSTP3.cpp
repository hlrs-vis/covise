/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                          (C)2005 RRZK  **
 **                                                                        **
 ** Description: Read STP3 volume files.                                   **
 **                                                                        **
 ** Author:      Martin Aumueller <aumueller@uni-koeln.de>                 **
 **                                                                        **
 ** Cration Date: 05.01.2005                                               **
 \**************************************************************************/

#include <api/coModule.h>
#include "ReadSTP3.h"

#define MAX_VOIS 20
#define VOI_DESC_SIZE 56

#ifdef BYTESWAP
#define byteSwap(x) (void) x
#define byteSwapM(x, no) (void) x
#else
#define byteSwapM(x, no) byteSwap(x, no)
#endif

/// Constructor
coReadSTP3::coReadSTP3(int argc, char *argv[])
    : coModule(argc, argv, "Read STP3 volume files.")
{
    // Create ports:
    poGrid = addOutputPort("grid", "UniformGrid", "Grid for volume data");
    poGrid->setInfo("Grid for volume data");

    poVolume = addOutputPort("data", "Float", "Scalar volume data");
    poVolume->setInfo("Scalar volume data (range 0-1)");

    // Create parameters:
    pbrVolumeFile = addFileBrowserParam("FilePath", "STP3 file");
    pbrVolumeFile->setValue("data", "*.img");

    pboUseVoi = addBooleanParam("UseVoi", "Map data outside of volume of interest to constant value");
    pboUseVoi->setValue(false);

    pisVoiNo = addInt32Param("NoVoi", "Number of volume of interest to use");
    pisVoiNo->setValue(1);

    pfsIgnoreValue = addFloatParam("IgnoreValue", "Value data not within the volume of interest is mapped to");
    pfsIgnoreValue->setValue(0.0);

    for (int i = 0; i < NO_VOIS; i++)
    {
        char buf1[1024], buf2[1024];
        sprintf(buf1, "Volume%dFromVoi", i + 1);
        sprintf(buf2, "Number of volume of interest to use for volume %d", i + 1);
        pisVolumeFromVoi[i] = addInt32Param(buf1, buf2);
        pisVolumeFromVoi[i]->setValue(i + 2);

        sprintf(buf1, "voi%d", i + 1);
        sprintf(buf2, "Volume of interest no. %d", i + 1);
        poVoi[i] = addOutputPort(buf1, "Float", buf2);
    }
}

/// This is our compute-routine
int coReadSTP3::compute(const char *)
{
    char buf[1024];

    const char *path = pbrVolumeFile->getValue();
    FILE *fp = fopen(path, "rb");
    if (!fp)
    {
        sprintf(buf, "Failed to open file %s", path);
        sendInfo(buf);

        return STOP_PIPELINE;
    }

    float ignore_value = pfsIgnoreValue->getValue();

    int32_t image_type;
    if (fread(&image_type, sizeof(image_type), 1, fp) != sizeof(image_type))
    {
        fprintf(stderr, "fread_1 failed in ReadSTP3.cpp");
    }
    byteSwap(image_type);
    int32_t series_header;
    if (fread(&series_header, sizeof(series_header), 1, fp) != sizeof(series_header))
    {
        fprintf(stderr, "fread_2 failed in ReadSTP3.cpp");
    }
    byteSwap(series_header);
    int32_t image_header;
    if (fread(&image_header, sizeof(image_header), 1, fp) != sizeof(image_header))
    {
        fprintf(stderr, "fread_3 failed in ReadSTP3.cpp");
    }
    byteSwap(image_header);
    int32_t image_length;
    if (fread(&image_length, sizeof(image_length), 1, fp) != sizeof(image_length))
    {
        fprintf(stderr, "fread_4 failed in ReadSTP3.cpp");
    }
    byteSwap(image_length);
    char patient_name[81];
    if (fread(patient_name, 1, 80, fp) != 1)
    {
        fprintf(stderr, "fread_5 failed in ReadSTP3.cpp");
    }
    patient_name[80] = '\0';
    char comment[81];
    if (fread(comment, 1, 80, fp) != 1)
    {
        fprintf(stderr, "fread_6 failed in ReadSTP3.cpp");
    }
    comment[80] = '\0';
    if (fread(&resolution, sizeof(resolution), 1, fp) != sizeof(resolution))
    {
        fprintf(stderr, "fread_7 failed in ReadSTP3.cpp");
    }
    byteSwap(resolution);
    int32_t byte_per_voxel;
    if (fread(&byte_per_voxel, sizeof(byte_per_voxel), 1, fp) != sizeof(byte_per_voxel))
    {
        fprintf(stderr, "fread_8 failed in ReadSTP3.cpp");
    }
    byteSwap(byte_per_voxel);
    if (fread(&num_slices, sizeof(num_slices), 1, fp) != sizeof(num_slices))
    {
        fprintf(stderr, "fread_9 failed in ReadSTP3.cpp");
    }
    byteSwap(num_slices);
    double psiz;
    if (fread(&psiz, sizeof(psiz), 1, fp) != sizeof(psiz))
    {
        fprintf(stderr, "fread_10 failed in ReadSTP3.cpp");
    }
    byteSwap(psiz);
    pixel_size = (float)psiz;
    char date[81];
    if (fread(date, 1, 80, fp) != 1)
    {
        fprintf(stderr, "fread_11 failed in ReadSTP3.cpp");
    }
    date[80] = '\0';

    char *image_type_desc = "(unknown)";
    switch (image_type)
    {
    case 1:
    case 100:
        image_type = 1;
        image_type_desc = "CT";
        break;
    case 2:
    case 200:
        image_type = 2;
        image_type_desc = "MR";
        break;
    case 3:
    case 410:
        image_type = 3;
        image_type_desc = "PET";
        break;
    }

    sprintf(buf, "Reading %s: %s image, %dx%d pixels, %d slices, %d byte/voxel",
            path, image_type_desc, (int)resolution, (int)resolution, (int)num_slices, (int)byte_per_voxel);
    sendInfo(buf);
    sprintf(buf, "Patient: %s", patient_name);
    sendInfo(buf);
    sprintf(buf, "Comment: %s", comment);
    sendInfo(buf);
    //sprintf(buf, "Date:    %s", date);
    //sendInfo(buf);

    slice_z = new float[num_slices];

    FILE *vfp = NULL;
    int32_t voi_header;
    bool read_voi = pboUseVoi->getValue();
    voi_total_no = 0;
    if (read_voi)
    {
        char *voiPath = new char[strlen(path) + 5];
        strcpy(voiPath, path);
        char *ext = strrchr(voiPath, '.');
        if (ext)
        {
            strcpy(ext, ".vois");
        }
        else
        {
            strcat(voiPath, ".vois");
        }

        vfp = fopen(voiPath, "rb");
        if (!vfp)
        {
            sprintf(buf, "Failed to open voi file %s", voiPath);
            sendInfo(buf);

            read_voi = false;
        }
        else
        {
            int32_t voi_version_number;
            if (fread(&voi_version_number, sizeof(voi_version_number), 1, vfp) != sizeof(voi_version_number))
            {
                fprintf(stderr, "fread_12 failed in ReadSTP3.cpp");
            } // 340
            byteSwap(voi_version_number);

            if (fread(&voi_header, sizeof(voi_header), 1, vfp) != sizeof(voi_header))
            {
                fprintf(stderr, "fread_13 failed in ReadSTP3.cpp");
            } // 2048
            byteSwap(voi_header);
            //fprintf(stderr, "voi_header=%d\n", voi_header);

            char voi_patient_name[81];
            if (fread(voi_patient_name, 1, 80, vfp) != 1)
            {
                fprintf(stderr, "fread_14 failed in ReadSTP3.cpp");
            }
            voi_patient_name[80] = '\0';
            char dummy_name[81];
            if (fread(dummy_name, 1, 80, vfp) != 1)
            {
                fprintf(stderr, "fread_15 failed in ReadSTP3.cpp");
            } // empty

            if (fread(&voi_total_no, sizeof(voi_total_no), 1, vfp) != sizeof(voi_total_no))
            {
                fprintf(stderr, "fread_16 failed in ReadSTP3.cpp");
            } // 20
            byteSwap(voi_total_no);

            int32_t voi_slices;
            if (fread(&voi_slices, sizeof(voi_slices), 1, vfp) != sizeof(voi_slices))
            {
                fprintf(stderr, "fread_17 failed in ReadSTP3.cpp");
            }
            byteSwap(voi_slices);

            sprintf(buf, "VOI Patient: %s", voi_patient_name);
            sendInfo(buf);

            sprintf(buf, "No. of vois: %d, no. of slices for vois: %d",
                    voi_total_no, voi_slices);
            sendInfo(buf);
        }
    }

    int voi_num = pisVoiNo->getValue();
    int32_t voi_property, voi_first_slice, voi_last_slice, voi_color;
    char voi_name[41];
    if (read_voi && voi_num < MAX_VOIS && voi_num < voi_total_no)
    {
        fseek(vfp, voi_header + VOI_DESC_SIZE * voi_num, SEEK_SET);
        if (fread(&voi_property, sizeof(voi_property), 1, vfp) != sizeof(voi_property))
        {
            fprintf(stderr, "fread_18 failed in ReadSTP3.cpp");
        }
        byteSwap(voi_property);
        if (fread(voi_name, 40, 1, vfp) != 40)
        {
            fprintf(stderr, "fread_19 failed in ReadSTP3.cpp");
        }
        voi_name[40] = '\0';
        if (fread(&voi_first_slice, sizeof(voi_first_slice), 1, vfp) != sizeof(voi_first_slice))
        {
            fprintf(stderr, "fread_20 failed in ReadSTP3.cpp");
        }
        byteSwap(voi_first_slice);
        if (fread(&voi_last_slice, sizeof(voi_last_slice), 1, vfp) != sizeof(voi_last_slice))
        {
            fprintf(stderr, "fread_21 failed in ReadSTP3.cpp");
        }
        byteSwap(voi_last_slice);
        if (fread(&voi_color, sizeof(voi_color), 1, vfp) != sizeof(voi_color))
        {
            fprintf(stderr, "fread_22 failed in ReadSTP3.cpp");
        }
        byteSwap(voi_color);
        fprintf(stderr, "voi name: %s, first=%d, last=%d\n", voi_name, voi_first_slice, voi_last_slice);
    }
    else
    {
        read_voi = false;
    }

    if (read_voi)
    {
        voi_first_slice--;
        voi_last_slice--;
    }
    else
    {
        voi_first_slice = 0;
        voi_last_slice = num_slices - 1;
    }

    coDoFloat *dataOut = new coDoFloat(poVolume->getObjName(),
                                       resolution * resolution * num_slices);
    float *data = NULL;
    dataOut->getAddress(&data);
    size_t slice_size = resolution * resolution * byte_per_voxel;
    unsigned char *slice = new unsigned char[slice_size];
    float minZ = -1.0, maxZ = 1.0;
    for (int i = 0; i < num_slices; i++)
    {
        long fpos = series_header + i * image_header + i * image_length;
        fseek(fp, fpos, SEEK_SET);
        int32_t image_type;
        if (fread(&image_type, sizeof(image_type), 1, fp) != sizeof(image_type))
        {
            fprintf(stderr, "fread_23 failed in ReadSTP3.cpp");
        }
        byteSwap(image_type);

        double z_pos;
        if (fread(&z_pos, sizeof(double), 1, fp) != sizeof(double))
        {
            fprintf(stderr, "fread_24 failed in ReadSTP3.cpp");
        }
        byteSwap(z_pos);
        slice_z[i] = z_pos;
        float z_position = (float)z_pos;
        double gantry;
        if (fread(&gantry, sizeof(double), 1, fp) != sizeof(double))
        {
            fprintf(stderr, "fread_25 failed in ReadSTP3.cpp");
        }
        byteSwap(gantry);
        //float gantry_tilt = (float)gantry;

        if (i == 0)
        {
            minZ = z_position;
        }
        else if (i == num_slices - 1)
        {
            maxZ = z_position;
        }
    }

    if (read_voi)
    {
        fseek(vfp, voi_header + VOI_DESC_SIZE * MAX_VOIS, SEEK_SET);
        //fprintf(stderr, "pos=%d\n", voi_header+VOI_DESC_SIZE*MAX_VOIS);
    }

    for (int i = 0; i < num_slices; i++)
    {
        vector<float> p_x, p_y;
        readVoiSlice(vfp, voi_num, &p_x, &p_y);

        long fpos = series_header + (i + 1) * image_header + i * image_length;
#if 0
      if(minZ > maxZ)
      {
         fpos = series_header + (num_slices-1-i-1)*image_header + (num_slices-1-i)*image_length;
      }
#endif
        fseek(fp, fpos, SEEK_SET);
        if (fread(slice, slice_size, 1, fp) != slice_size)
        {
            fprintf(stderr, "fread_26 failed in ReadSTP3.cpp");
        }

        for (int y = 0; y < resolution; y++)
        {
            vector<float> i_x;
            computeIntersections(y, p_x, p_y, &i_x);

            int no_isect = 0;
            for (int x = 0; x < resolution; x++)
            {
                while (no_isect < i_x.size() && i_x[no_isect] < x)
                    no_isect++;

                int j = x * resolution + y;
                int k = ((y * resolution) + x) * num_slices + i;
                if (byte_per_voxel == 1)
                {
                    data[k] = slice[j] / 255.f;
                }
                else if (byte_per_voxel == 2)
                {
                    data[k] = (256.f * slice[j * 2] + slice[j * 2 + 1]) / 65535.f;
                }
                else
                {
                    data[k] = 0.f;
                }
                if (read_voi && no_isect % 2 == 0)
                {
                    data[k] = ignore_value;
                }
            }
        }

        p_x.clear();
        p_y.clear();
    }
    if (fp)
        fclose(fp);

    coDoFloat *voiOut[NO_VOIS];
    for (int v = 0; v < NO_VOIS; v++)
    {
        if (!poVoi[v]->isConnected())
        {
            voiOut[v] = NULL;
            continue;
        }

        voiOut[v] = new coDoFloat(poVoi[v]->getObjName(),
                                  resolution * resolution * num_slices);
        float *data = NULL;
        voiOut[v]->getAddress(&data);

        voi_num = pisVolumeFromVoi[v]->getValue();

        if (read_voi && voi_num < MAX_VOIS && voi_num < voi_total_no)
        {
            fseek(vfp, voi_header + VOI_DESC_SIZE * voi_num, SEEK_SET);
            if (fread(&voi_property, sizeof(voi_property), 1, vfp) != sizeof(voi_property))
            {
                fprintf(stderr, "fread_27 failed in ReadSTP3.cpp");
            }
            byteSwap(voi_property);
            if (fread(voi_name, 40, 1, vfp) != 40)
            {
                fprintf(stderr, "fread_28 failed in ReadSTP3.cpp");
            }
            voi_name[40] = '\0';
            if (fread(&voi_first_slice, sizeof(voi_first_slice), 1, vfp) != sizeof(voi_first_slice))
            {
                fprintf(stderr, "fread_29 failed in ReadSTP3.cpp");
            }
            byteSwap(voi_first_slice);
            if (fread(&voi_last_slice, sizeof(voi_last_slice), 1, vfp) != sizeof(voi_last_slice))
            {
                fprintf(stderr, "fread_30 failed in ReadSTP3.cpp");
            }
            byteSwap(voi_last_slice);
            if (fread(&voi_color, sizeof(voi_color), 1, vfp) != sizeof(voi_color))
            {
                fprintf(stderr, "fread_31 failed in ReadSTP3.cpp");
            }
            byteSwap(voi_color);
            fprintf(stderr, "voi name: %s, first=%d, last=%d\n", voi_name, voi_first_slice, voi_last_slice);

            fseek(vfp, voi_header + VOI_DESC_SIZE * MAX_VOIS, SEEK_SET);

            for (int i = 0; i < num_slices; i++)
            {

                vector<float> p_x, p_y;
                readVoiSlice(vfp, voi_num, &p_x, &p_y);

                for (int y = 0; y < resolution; y++)
                {
                    vector<float> i_x;
                    computeIntersections(y, p_x, p_y, &i_x);
                    int no_isect = 0;
                    for (int x = 0; x < resolution; x++)
                    {
                        while (no_isect < i_x.size() && i_x[no_isect] < x)
                            no_isect++;

                        int k = ((y * resolution) + x) * num_slices + i;
                        if (no_isect % 2 == 0)
                        {
                            data[k] = ignore_value;
                        }
                        else
                        {
                            data[k] = 1.f;
                        }
                    }
                }
            }
        }
    }

    if (vfp)
        fclose(vfp);
    delete[] slice;

    float maxX = pixel_size * resolution / 2.f;
    float minX = -maxX;
    float maxY = maxX;
    float minY = -maxY;
#if 0
   if(minZ > maxZ)
   {
      float temp = minZ;
      minZ = maxZ;
      maxZ = temp;
   }
#endif

    coDoUniformGrid *gridOut = new coDoUniformGrid(poGrid->getObjName(),
                                                   resolution, resolution, num_slices, //voi_last_slice-voi_first_slice+1,
                                                   minX, maxX,
                                                   minY, maxY,
                                                   //minZ + ((maxZ-minZ)*voi_first_slice)/num_slices, minZ + ((maxZ-minZ)*voi_last_slice)/num_slices);
                                                   minZ, maxZ);

    char *matPath = new char[strlen(path) + 5];
    strcpy(matPath, path);
    char *ext = strrchr(matPath, '.');
    if (ext)
    {
        strcpy(ext, ".tra");
    }
    else
    {
        strcat(matPath, ".tra");
    }

    double mat[4][4], inv[4][4];
    if (getTransformation(matPath, &mat[0][0], &inv[0][0]) >= 0)
    {
        char transMat[64 * 16];
        char *p = transMat;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                int sz = sprintf(p, "%f ", mat[j][i]);
                p += sz;
            }
        }
        gridOut->addAttribute("Transformation", transMat);
        fprintf(stderr, "attached Transformation: %s\n", transMat);
    }
    else
    {
        char buf[1024];
        sprintf(buf, "failed to read transformation data from %s", matPath);
        sendInfo(matPath);
    }

    poGrid->setCurrentObject(gridOut);
    poVolume->setCurrentObject(dataOut);

    sprintf(buf, "Volume data loaded: (%f, %f, %f) - (%f, %f %f)",
            minX, minY, minZ, maxX, maxY, maxZ);
    sendInfo(buf);

    return CONTINUE_PIPELINE;
}

#include "mlc.h"
#include "nr.h"

#include <qtextstream.h>

int read_ct_tran(struct patient_struct *ppatient, QString *Result);

int coReadSTP3::getTransformation(const char *filename, double *mat, double *inv)
{
#if 0
   FILE *fp = fopen(filename, "rb");

   if(!fp)
   {
      fprintf(stderr, "failed to open %s\n", filename);
      return -1;
   }

   // read transformation header
   uint32_t file_type;
   if (fread(&file_type,sizeof(file_type),  1, fp)!=sizeof(file_type))
   {
      fprintf(stderr,"fread_32 failed in ReadSTP3.cpp");
   }
   byteSwap(file_type);
   uint32_t ser_head_len;
   if (fread(&ser_head_len,sizeof(ser_head_len),  1, fp)!=sizeof(ser_head_len))
   {
      fprintf(stderr,"fread_33 failed in ReadSTP3.cpp");
   }
   byteSwap(ser_head_len);
   uint32_t block_len;
   if (fread(&block_len,sizeof(block_len),  1, fp)!=sizeof(block_len))
   {
      fprintf(stderr,"fread_34 failed in ReadSTP3.cpp");
   }
   byteSwap(block_len);
   fprintf(stderr, "blocklen=%d, serheadlen=%d\n", block_len, ser_head_len);
   char patient_name[81];
   if (fread(patient_name, 1, 80, fp)!= 1)
   {
      fprintf(stderr,"fread_35 failed in ReadSTP3.cpp");
   }
   patient_name[80] = '\0';
   char comm[81];
   if (fread(comm, 1, 80, fp)!= 1)
   {
      fprintf(stderr,"fread_36 failed in ReadSTP3.cpp");
   }
   comm[80] = '\0';
   char date[81];
   if (fread(date, 1, 80, fp)!= 1)
   {
      fprintf(stderr,"fread_37 failed in ReadSTP3.cpp");
   }
   date[80] = '\0';
   uint32_t serial_num;
   if (fread(&serial_num,sizeof(serial_num),  1, fp)!=sizeof(serial_num))
   {
      fprintf(stderr,"fread_38 failed in ReadSTP3.cpp");
   }
   byteSwap(serial_num);
   uint32_t num_slices;
   if (fread(&num_slices,sizeof(num_slices),  1, fp)!=sizeof(num_slices))
   {
      fprintf(stderr,"fread_39 failed in ReadSTP3.cpp");
   }
   byteSwap(num_slices);

   vector<double *> mats, invs;
   uint32_t trafo_type, is_axial;
   // read transformation for each slice
   for(int i = 0; i < num_slices; i++)
   {
      fseek(fp, block_len * (i+1) + ser_head_len, SEEK_SET);
      if (fread(&trafo_type,sizeof(trafo_type), 1, fp)!=sizeof(trafo_type))
      {
         fprintf(stderr,"fread_40 failed in ReadSTP3.cpp");
      }
      byteSwap(trafo_type);
      uint32_t calculated;
      if (fread(&calculated,sizeof(calculated), 1, fp)!=sizeof(calculated))
      {
         fprintf(stderr,"fread_41 failed in ReadSTP3.cpp");
      }
      byteSwap(calculated);
      if (fread(&is_axial,sizeof(is_axial), 1, fp)!=sizeof(is_axial))
      {
         fprintf(stderr,"fread_42 failed in ReadSTP3.cpp");
      }
      byteSwap(is_axial);
      double    marker[12][2];
      if (fread(marker,sizeof(marker), 1, fp)!=sizeof(marker))
      {
         fprintf(stderr,"fread_43 failed in ReadSTP3.cpp");
      }
      double    plate[16][2];
      if (fread(plate,sizeof(plate), 1, fp)!=sizeof(plate))
      {
         fprintf(stderr,"fread_44 failed in ReadSTP3.cpp");
      }
      uint32_t missing;
      if (fread(&missing,sizeof(missing), 1, fp)!=sizeof(missing))
      {
         fprintf(stderr,"fread_45 failed in ReadSTP3.cpp");
      }
      byteSwap(missing);
      double *m= new double[16];
      if (fread(m,sizeof(*m)*16, 1, fp)!=sizeof(*m)*16)
      {
         fprintf(stderr,"fread_46 failed in ReadSTP3.cpp");
      }
      byteSwapM(m, 16);
      mats.push_back(m);
      m = new double[16];
      if (fread(m,sizeof(*m)*16, 1, fp)!=sizeof(*m)*16)
      {
         fprintf(stderr,"fread_47 failed in ReadSTP3.cpp");
      }
      byteSwapM(m, 16);
      invs.push_back(m);
   }
   fclose(fp);

   memcpy(mat, mats[num_slices/2],sizeof(double)*16);
   memcpy(inv, invs[num_slices/2],sizeof(double)*16);

   for(int i=0; i<mats.size(); i++)
   {
      delete[] mats[i];
      delete[] invs[i];
   }

   return 0;
#endif

    struct patient_struct patient;
    strcpy(patient.Tra_File, filename);
    patient.Resolution = resolution;
    patient.No_Slices = num_slices;
    patient.Pixel_size = pixel_size;
    for (int i = 0; i < num_slices; i++)
    {
        patient.Z_Table[i] = slice_z[i];
    }
    QString result;

    int ret = read_ct_tran(&patient, &result);

    memcpy(mat, &patient.Global_Tra_Matrix[0][0], sizeof(double) * 16);
    memcpy(inv, &patient.Rev_Global_Tra_Matrix[0][0], sizeof(double) * 16);

    mat[14] -= 100.;
    //inv[14] += 100.;

    cerr << result.latin1() << endl;

    return ret;
}

int coReadSTP3::readVoiSlice(FILE *vfp, int voi_num, vector<float> *x, vector<float> *y)
{
    x->clear();
    y->clear();

    for (int voi = 0; voi < MAX_VOIS && voi < voi_total_no; voi++)
    {
        int32_t no_contours;
        if (fread(&no_contours, sizeof(no_contours), 1, vfp) != sizeof(no_contours))
        {
            fprintf(stderr, "fread_48 failed in ReadSTP3.cpp");
        }
        byteSwap(no_contours);
        if (no_contours > 0)
        {
            int32_t no_points;
            if (fread(&no_points, sizeof(no_points), 1, vfp) != sizeof(no_points))
            {
                fprintf(stderr, "fread_49 failed in ReadSTP3.cpp");
            }
            byteSwap(no_points);
            if (voi == voi_num)
            {
                //fprintf(stderr, "slice %d: voi %d, %d contours, %d points\n", i, voi, no_contours, no_points);
                x->reserve(no_points);
                y->reserve(no_points);
            }

            for (int p = 0; p < no_points; p++)
            {
                double d;
                if (fread(&d, sizeof(d), 1, vfp) != sizeof(d))
                {
                    fprintf(stderr, "fread_50 failed in ReadSTP3.cpp");
                }
                if (voi == voi_num)
                {
                    byteSwap(d);
                    y->push_back(float(d / 1023.) * resolution);
                }
                if (fread(&d, sizeof(d), 1, vfp) != sizeof(d))
                {
                    fprintf(stderr, "fread_51 failed in ReadSTP3.cpp");
                }
                if (voi == voi_num)
                {
                    byteSwap(d);
                    x->push_back(float(d / 1023.) * resolution);
                }
            }
        }
    }

    return 0;
}

void coReadSTP3::computeIntersections(int y, const vector<float> &p_x, const vector<float> &p_y, vector<float> *i_x)
{
    int no_points = p_x.size();
    for (int l = 0; l < no_points; l++)
    {
        int l1 = (l + 1) % no_points;
        if ((p_y[l] < y && p_y[l1] >= y) || (p_y[l] >= y && p_y[l1] < y))
        {
            float x = p_x[l] + float(y - p_y[l]) / float(p_y[l1] - p_y[l]) * float(p_x[l1] - p_x[l]);
            i_x->push_back(x);
        }
    }
    sort(i_x->begin(), i_x->end());
}

#if 0

/*
 *                                               -----------------------
 * Entry Name:                                   read_ct_tran.cpp
 *                                               -----------------------
 *
 * Function:    read *.tra files
 *
 * Author:      Javier Villar
 *              Moritz Hoevels
 *
 *              University of Cologne
 *              Department for Stereotactic
 *              and Functional Neurosurgery
 *
 *
 *----------------------------------------------------------------------
 * Description for Users:
 *----------------------------------------------------------------------
 *
 *  read stereotactic transformation from disk
 *
 *----------------------------------------------------------------------
 * Last change: 02.01.2005
 *----------------------------------------------------------------------
 * Revisions:
 *----------------------------------------------------------------------
 * REV #   DATE           NAME       REVISIONS MADE
 * 01.00   15.12.2004     MH         created
 *
 *----------------------------------------------------------------------
 *
 *--- include files --------------------------------------------------*/

#endif

int read_ct_tran(struct patient_struct *ppatient, QString *Result)
{
    FILE *f_tran = NULL;
    uint32_t file_typ, ser_head_length, block_length, ser_number, numb_of_slices;
    char pat_name[80], comm[80], date[80];
    uint32_t trafo_type;
    uint32_t calculated;
    uint32_t is_axial;
    double marker[12][2];
    double plate[16][2];
    uint32_t missing;
    uint32_t all_slices;
    uint32_t elements_v_rev_mat;
    uint32_t i, j, n;
    double fov_by_2;
    double global_inv_mat[4][4];

    mat44_t mat;
    mat44_t rev_mat;
    vector<mat44_t> v_mat, v_rev_mat;

    xyze Stereo_point_0;
    xyze Stereo_point_1;
    xyze Stereo_point_2;
    xyze Stereo_point_3;

    const int imax = 4;

    double Edge_point_a[imax] = { 0.0, 0.0, 1.0, 1.0 };
    double Edge_point_b[imax] = { 0.0, 1023.0, 1.0, 1.0 };
    double Edge_point_c[imax] = { 1023.0, 1023.0, 1.0, 1.0 };
    double Edge_point_d[imax] = { 1023.0, 0.0, 1.0, 1.0 };
    double last_col_glo_matrix[imax] = { 0.0, 0.0, 0.0, 1.0 };

    QTextStream ts(Result, IO_WriteOnly);

    ts << "<B>Loading " << ppatient->Tra_File;

    if ((f_tran = fopen(ppatient->Tra_File, "rb")) == NULL)
    {
        ts << ": failed!</B><br><br>";
        return 1;
    }

    // read transformation header
    if (fread(&file_typ, 4, 1, f_tran) != 4)
    {
        fprintf(stderr, "fread_52 failed in ReadSTP3.cpp");
    }
    if (fread(&ser_head_length, 4, 1, f_tran) != 4)
    {
        fprintf(stderr, "fread_53 failed in ReadSTP3.cpp");
    }
    if (fread(&block_length, 4, 1, f_tran) != 4)
    {
        fprintf(stderr, "fread_54 failed in ReadSTP3.cpp");
    }
    if (fread(pat_name, 1, 80, f_tran) != 1)
    {
        fprintf(stderr, "fread_55 failed in ReadSTP3.cpp");
    }
    if (fread(comm, 1, 80, f_tran) != 1)
    {
        fprintf(stderr, "fread_56 failed in ReadSTP3.cpp");
    }
    if (fread(date, 1, 80, f_tran) != 1)
    {
        fprintf(stderr, "fread_57 failed in ReadSTP3.cpp");
    }
    if (fread(&ser_number, 4, 1, f_tran) != 4)
    {
        fprintf(stderr, "fread_58 failed in ReadSTP3.cpp");
    }
    if (fread(&numb_of_slices, 4, 1, f_tran) != 4)
    {
        fprintf(stderr, "fread_59 failed in ReadSTP3.cpp");
    }

    // read transformation for each slice
    for (j = 1; j < numb_of_slices + 1; j++)
    {
        fseek(f_tran, block_length * j + ser_head_length, SEEK_SET);
        if (fread(&trafo_type, 4, 1, f_tran) != 4)
        {
            fprintf(stderr, "fread_60 failed in ReadSTP3.cpp");
        }
        if (fread(&calculated, 4, 1, f_tran) != 4)
        {
            fprintf(stderr, "fread_61 failed in ReadSTP3.cpp");
        }
        if (fread(&is_axial, 4, 1, f_tran) != 4)
        {
            fprintf(stderr, "fread_62 failed in ReadSTP3.cpp");
        }
        if (fread(marker, sizeof(marker), 1, f_tran) != sizeof(marker))
        {
            fprintf(stderr, "fread_63 failed in ReadSTP3.cpp");
        }
        if (fread(plate, sizeof(plate), 1, f_tran) != sizeof(plate))
        {
            fprintf(stderr, "fread_64 failed in ReadSTP3.cpp");
        }
        if (fread(&missing, 4, 1, f_tran) != 4)
        {
            fprintf(stderr, "fread_65 failed in ReadSTP3.cpp");
        }
        if (fread(mat.mat, sizeof(mat.mat), 1, f_tran) != sizeof(mat.mat))
        {
            fprintf(stderr, "fread_66 failed in ReadSTP3.cpp");
        }
        if (fread(rev_mat.mat, sizeof(rev_mat.mat), 1, f_tran) != sizeof(rev_mat.mat))
        {
            fprintf(stderr, "fread_67 failed in ReadSTP3.cpp");
        }
        v_mat.push_back(mat);
        v_rev_mat.push_back(rev_mat);
    }

    fclose(f_tran);
    all_slices = v_mat.size();
    elements_v_rev_mat = v_rev_mat.size();
    fov_by_2 = ppatient->Pixel_size * ppatient->Resolution / 2.0;
    n = ppatient->No_Slices * 4;

    // define for geting the glogal transformation matrix, using the
    // singular value decomposition NR::svdcmp(x,w,v), and backsustitution NR::svbksb(x,w,v,b_x,x_x);
    // here we used matrix defined in Numerical Recipes because are dinamics(we dont know how many members have)

    Mat_DP x(n, 4), u(n, 4), v(4, 4);
    Vec_DP w(4), b(n);
    Vec_DP b_x(n), b_y(n), b_z(n);
    Vec_DP x_x(n), x_y(n), x_z(n);

    // define for geting the inverse glogal transformation matrix, using the Lower Up decomposition
    // NR::ludcmp(global_mat,indx,d), and backsustitution NR::lubksb(global_mat,indx,col);

    Mat_DP global_mat(4, 4);
    Vec_DP col(4);
    Vec_INT indx(4);
    DP d;

    for (i = 0; i < all_slices; i++)
    {
        // image coordinates
        x[i * 4 + 0][0] = fov_by_2;
        x[i * 4 + 0][1] = fov_by_2;
        x[i * 4 + 0][2] = ppatient->Z_Table[i];
        x[i * 4 + 0][3] = 1.0;

        x[i * 4 + 1][0] = fov_by_2;
        x[i * 4 + 1][1] = -fov_by_2;
        x[i * 4 + 1][2] = ppatient->Z_Table[i];
        x[i * 4 + 1][3] = 1.0;

        x[i * 4 + 2][0] = -fov_by_2;
        x[i * 4 + 2][1] = -fov_by_2;
        x[i * 4 + 2][2] = ppatient->Z_Table[i];
        x[i * 4 + 2][3] = 1.0;

        x[i * 4 + 3][0] = -fov_by_2;
        x[i * 4 + 3][1] = fov_by_2;
        x[i * 4 + 3][2] = ppatient->Z_Table[i];
        x[i * 4 + 3][3] = 1.0;

        // Stereotactic coordinates
        Stereo_point_0.x = 0.0;
        Stereo_point_0.y = 0.0;
        Stereo_point_0.z = 0.0;
        Stereo_point_1.x = 0.0;
        Stereo_point_1.y = 0.0;
        Stereo_point_1.z = 0.0;
        Stereo_point_2.x = 0.0;
        Stereo_point_2.y = 0.0;
        Stereo_point_2.z = 0.0;
        Stereo_point_3.x = 0.0;
        Stereo_point_3.y = 0.0;
        Stereo_point_3.z = 0.0;

        for (int f = 0; f <= 3; f++)
        {
            Stereo_point_0.x += v_mat[i].mat[f][1] * Edge_point_a[f];
            Stereo_point_0.y += v_mat[i].mat[f][2] * Edge_point_a[f];
            Stereo_point_0.z += v_mat[i].mat[f][3] * Edge_point_a[f];
            Stereo_point_0.err = 1.0; // [0.0, 0.0, 1.0, 1,0]

            Stereo_point_1.x += v_mat[i].mat[f][1] * Edge_point_b[f];
            Stereo_point_1.y += v_mat[i].mat[f][2] * Edge_point_b[f];
            Stereo_point_1.z += v_mat[i].mat[f][3] * Edge_point_b[f];
            Stereo_point_1.err = 1.0; // [0.0, 1024.0, 1.0, 1,0]

            Stereo_point_2.x += v_mat[i].mat[f][1] * Edge_point_c[f];
            Stereo_point_2.y += v_mat[i].mat[f][2] * Edge_point_c[f];
            Stereo_point_2.z += v_mat[i].mat[f][3] * Edge_point_c[f];
            Stereo_point_2.err = 1.0; // [1024.0, 1024.0, 1.0, 1,0]

            Stereo_point_3.x += v_mat[i].mat[f][1] * Edge_point_d[f];
            Stereo_point_3.y += v_mat[i].mat[f][2] * Edge_point_d[f];
            Stereo_point_3.z += v_mat[i].mat[f][3] * Edge_point_d[f];
            Stereo_point_3.err = 1.0; // [1024.0, 0.0, 1.0, 1,0]
        }

        b_x[i * 4 + 0] = Stereo_point_0.x;
        b_x[i * 4 + 1] = Stereo_point_1.x;
        b_x[i * 4 + 2] = Stereo_point_2.x;
        b_x[i * 4 + 3] = Stereo_point_3.x;

        b_y[i * 4 + 0] = Stereo_point_0.y;
        b_y[i * 4 + 1] = Stereo_point_1.y;
        b_y[i * 4 + 2] = Stereo_point_2.y;
        b_y[i * 4 + 3] = Stereo_point_3.y;

        b_z[i * 4 + 0] = Stereo_point_0.z;
        b_z[i * 4 + 1] = Stereo_point_1.z;
        b_z[i * 4 + 2] = Stereo_point_2.z;
        b_z[i * 4 + 3] = Stereo_point_3.z;
    }

    // solve linear equation
    NR::svdcmp(x, w, v); // here we make the decomposition of the matrix X (image coordinate,for the for points in each slices)!

    NR::svbksb(x, w, v, b_x, x_x);
    NR::svbksb(x, w, v, b_y, x_y);
    NR::svbksb(x, w, v, b_z, x_z);

    // transformation matrix
    for (i = 0; i < 4; i++)
    {
        ppatient->Global_Tra_Matrix[i][0] = x_x[i];
        ppatient->Global_Tra_Matrix[i][1] = x_y[i];
        ppatient->Global_Tra_Matrix[i][2] = x_z[i];
        ppatient->Global_Tra_Matrix[i][3] = last_col_glo_matrix[i];

        for (j = 0; j < 4; j++)
            global_mat[i][j] = ppatient->Global_Tra_Matrix[i][j];
    }

    // calculate inverse matrix
    NR::ludcmp(global_mat, indx, d);

    for (j = 0; j < 4; j++)
    {
        for (i = 0; i < 4; i++)
            col[i] = 0.0;
        col[j] = 1.0;
        NR::lubksb(global_mat, indx, col);
        for (i = 0; i < 4; i++)
            global_inv_mat[i][j] = col[i];
    }

    // print transformation info
    ts << ": Ok</B>";
    ts << "<br><br><B>Transformation Info</B>"; // <br> te salta una linea, con el primer B me lo escribiria to en negro pa especificar q solo queremos la linea esa se pone al final </B>
    ts << "<pre>"; // si omito esta linea me escribe la dos matrices juntas sin ningun sentido!, y pq el otro no pasa na cuando se omite, ver #
    ts << "Mat = <br>";

    for (i = 0; i < 4; i++)
    {
        for (j = 0; j < 4; j++)
            ppatient->Rev_Global_Tra_Matrix[i][j] = global_inv_mat[i][j];

        ts << "       " << ppatient->Global_Tra_Matrix[i][0]
           << "       " << ppatient->Global_Tra_Matrix[i][1]
           << "       " << ppatient->Global_Tra_Matrix[i][2]
           << "       " << ppatient->Global_Tra_Matrix[i][3]
           << "<br>";
    }
    ts << "RevMat = <br>";

    for (i = 0; i < 4; i++)
        ts << "       " << ppatient->Rev_Global_Tra_Matrix[i][0]
           << "       " << ppatient->Rev_Global_Tra_Matrix[i][1]
           << "       " << ppatient->Rev_Global_Tra_Matrix[i][2]
           << "       " << ppatient->Rev_Global_Tra_Matrix[i][3]
           << "<br>";

    ts << "<br></pre>";

    return 0;
}

MODULE_MAIN(IO, coReadSTP3)
