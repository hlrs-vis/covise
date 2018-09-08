#include "gdcm.h"

#if VV_HAVE_GDCM
#include <gdcmAttribute.h>
#include <gdcmReader.h>
#include <gdcmImageReader.h>
#include <gdcmMediaStorage.h>
#include <gdcmFile.h>
#include <gdcmDataSet.h>
#include <gdcmUIDs.h>
#include <gdcmGlobal.h>
#include <gdcmModules.h>
#include <gdcmDefs.h>
#include <gdcmOrientation.h>
#include <gdcmVersion.h>
#include <gdcmMD5.h>
#include <gdcmSystem.h>
#include <gdcmDirectory.h>


#include <virvo/vvvoldesc.h>
#include <virvo/vvpixelformat.h>
#include <virvo/vvfileio.h>
#include <virvo/private/vvlog.h>
#include "exceptions.h"

#include <string>
#include <unordered_map>
#include <iostream>
#include <memory>
#include <functional>
#include <algorithm>
#include <cctype>

namespace {

/* mostly copied from GDCM: Examples/Cxx/ReadAndDumpDicomDir2.cxx */

/*=========================================================================

Program: GDCM (Grassroots DICOM). A DICOM library

Copyright (c) 2006-2017 Mathieu Malaterre
All rights reserved.
See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*
 * This example shows how to read and dump a DICOMDIR File
 *
 * Thanks:
 *   Tom Marynowski (lordglub gmail) for contributing the original 
 *    ReadAndDumpDICOMDIR.cxx example
 *   Mihail Isakov for contributing offset calculation code here:
 *    https://sourceforge.net/p/gdcm/mailman/gdcm-developers/?viewmonth=201707&viewday=15
 *   Tod Baudais for combining the above and cleaning up this example
 */


//==============================================================================
//==============================================================================

#define TAG_MEDIA_STORAGE_SOP_CLASS_UID 0x0002,0x0002
#define TAG_DIRECTORY_RECORD_SEQUENCE 0x0004,0x1220
#define TAG_DIRECTORY_RECORD_TYPE 0x0004,0x1430
#define TAG_PATIENTS_NAME 0x0010,0x0010
#define TAG_PATIENT_ID 0x0010,0x0020
#define TAG_STUDY_DATE 0x0008,0x0020
#define TAG_STUDY_DESCRIPTION 0x0008,0x1030
#define TAG_MODALITY 0x0008,0x0060
#define TAG_SERIES_DESCRIPTION 0x0008,0x103E
#define TAG_REFERENCED_FILE_ID 0x0004,0x1500
#define TAG_REFERENCED_LOWER_LEVEL_DIRECTORY_ENTITY_OFFSET 0x0004,0x1420
#define TAG_NEXT_DIRECTORY_RECORD_OFFSET 0x0004,0x1400
#define TAG_SEQUENCE_NUMBER 0x0020,0x0011
#define TAG_INSTANCE_NUMBER 0x0020,0x0013
#define TAG_SLICE_LOCATION 0x0020,0x1041

//==============================================================================
// Some handy utility functions
//==============================================================================

std::string left_trim(const std::string &s) {
  std::string ss(s);
  ss.erase(ss.begin(), std::find_if(ss.begin(), ss.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
  return ss;
}

std::string right_trim(const std::string &s) {
  std::string ss(s);
  ss.erase(std::find_if(ss.rbegin(), ss.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), ss.end());
  return ss;
}

std::string trim(const std::string &s) {
  return left_trim(right_trim(s));
}

//==============================================================================
// This code could be put in a header file somewhere
//==============================================================================

class DICOMDIRReader {
 public:
   DICOMDIRReader      (void) {}
   DICOMDIRReader      (const DICOMDIRReader &rhs) = delete;
   DICOMDIRReader      (DICOMDIRReader &&rhs) = delete;
   DICOMDIRReader &        operator =          (const DICOMDIRReader &rhs) = delete;
   DICOMDIRReader &        operator =          (DICOMDIRReader &&rhs) = delete;
   virtual                 ~DICOMDIRReader     (void) {}

 public:
   struct Common {
     int64_t child_offset;
     int64_t sibling_offset;
   };

   struct Image: public Common {
     std::string path;
     int64_t number;
   };

   struct Series: public Common {
     std::string modality;
     std::string description;

     std::vector<std::shared_ptr<Image>> children;
   };

   struct Study: public Common {
     std::string date;
     std::string description;

     std::vector<std::shared_ptr<Series>> children;
   };

   struct Patient: public Common {
     std::string name;
     std::string id;

     std::vector<std::shared_ptr<Study>> children;
   };

   struct Other: public Common {
   };

   /// Load DICOMDIR
   const std::vector<std::shared_ptr<Patient>>&    load        (const std::string &path);

   /// Return the results of the load
   const std::vector<std::shared_ptr<Patient>>&    patients    (void)  {   return _patients;   }

 private:

   template <class T>
     std::string     get_string              (const T &ds, const gdcm::Tag &tag)
     {
       std::stringstream strm;
       if (ds.FindDataElement(tag)) {
         auto &de = ds.GetDataElement(tag);
         if (!de.IsEmpty() && !de.IsUndefinedLength())
           de.GetValue().Print(strm);
       }
       return trim(strm.str());
     }


   template <class P, class C, class O>
     void            reassemble_hierarchy    (P &parent_offsets, C &child_offsets, O &other_offsets)
     {
       for (auto &parent : parent_offsets) {
         int64_t sibling_offset;
         auto c = child_offsets[parent.second->child_offset];
         if (!c) {
           auto o = other_offsets[parent.second->child_offset];
           if (!o) {
             continue;
           } else {
             sibling_offset = o->sibling_offset;
           }
         } else {
           parent.second->children.push_back(c);
           sibling_offset = c->sibling_offset;
         }

         // Get all siblings
         while (sibling_offset) {
           c = child_offsets[sibling_offset];
           if (!c) {
             auto o = other_offsets[sibling_offset];
             if (!o) {
               break;
             } else {
               sibling_offset = o->sibling_offset;
             }
           } else {
             parent.second->children.push_back(c);
             sibling_offset = c->sibling_offset;
           }
         }
       }
     }

   std::vector<std::shared_ptr<Patient>> _patients;
};

//==============================================================================
// This code could be put in an implementation file somewhere
//==============================================================================

const std::vector<std::shared_ptr<DICOMDIRReader::Patient>>& DICOMDIRReader::load (const std::string &path)
{
  _patients.clear();

  //
  // Read the dataset from the DICOMDIR file
  //

  gdcm::Reader reader;
  reader.SetFileName(path.c_str());
  if(!reader.Read()) {
    throw std::runtime_error("Unable to read file");
  }

  // Retrieve information from file
  auto &file = reader.GetFile();
  auto &data_set = file.GetDataSet();
  auto &file_meta_information = file.GetHeader();

  // Retrieve and check the Media Storage class from file
  gdcm::MediaStorage media_storage;
  media_storage.SetFromFile(file);
  if(media_storage != gdcm::MediaStorage::MediaStorageDirectoryStorage) {
    throw std::runtime_error("This file is not a DICOMDIR");
  }

  auto media_storage_sop_class_uid = get_string(file_meta_information, gdcm::Tag(TAG_MEDIA_STORAGE_SOP_CLASS_UID));

  // Make sure we have a DICOMDIR file
  if (media_storage_sop_class_uid != "1.2.840.10008.1.3.10") {
    throw std::runtime_error("This file is not a DICOMDIR");
  }

  //
  // Offset to first item courtesy of Mihail Isakov
  //

  gdcm::VL first_item_offset = 0;
  auto it = data_set.Begin();
  for(; it != data_set.End() && it->GetTag() != gdcm::Tag(TAG_DIRECTORY_RECORD_SEQUENCE); ++it) {
    first_item_offset += it->GetLength<gdcm::ExplicitDataElement>();
  }
  // Tag (4 bytes)
  first_item_offset += it->GetTag().GetLength();
  // VR field
  first_item_offset += it->GetVR().GetLength();
  // VL field
  // For Explicit VR: adventitiously VL field lenght = VR field lenght,
  // for SQ 4 bytes:
  // http://dicom.nema.org/medical/dicom/current/output/html/part05.html#table_7.1-1
  first_item_offset += it->GetVR().GetLength();

  //
  // Iterate all data elements
  //

  // For each item in data set
  for(auto data_element : data_set.GetDES()) {

    // Only look at Directory sequence
    if (data_element.GetTag() != gdcm::Tag(TAG_DIRECTORY_RECORD_SEQUENCE))
      continue;

    auto item_sequence = data_element.GetValueAsSQ();
    auto num_items = item_sequence->GetNumberOfItems();

    //
    // Compute an offset table
    //

    // Start calculation of offset to each item courtesy of Mihail Isakov
    std::vector<int64_t> item_offsets(num_items+1);
    item_offsets[0] = file_meta_information.GetFullLength() + static_cast<int64_t>(first_item_offset);

    //
    // Extract out all of the items
    //

    std::unordered_map<int64_t, std::shared_ptr<Patient>> patient_offsets;
    std::unordered_map<int64_t, std::shared_ptr<Study>> study_offsets;
    std::unordered_map<int64_t, std::shared_ptr<Series>> series_offsets;
    std::unordered_map<int64_t, std::shared_ptr<Image>> image_offsets;
    std::unordered_map<int64_t, std::shared_ptr<Other>> other_offsets;

    for (uint32_t item_index = 1; item_index <= num_items; ++item_index) {
      auto &item = item_sequence->GetItem(item_index);

      // Add offset for item to offset table
      item_offsets[item_index] = item_offsets[item_index-1] + item.GetLength<gdcm::ExplicitDataElement>();

      // Child offset
      gdcm::Attribute<TAG_REFERENCED_LOWER_LEVEL_DIRECTORY_ENTITY_OFFSET> child_offset;
      child_offset.SetFromDataElement(item.GetDataElement(gdcm::Tag (TAG_REFERENCED_LOWER_LEVEL_DIRECTORY_ENTITY_OFFSET)));

      // Sibling offset
      gdcm::Attribute<TAG_NEXT_DIRECTORY_RECORD_OFFSET> sibling_offset;
      sibling_offset.SetFromDataElement(item.GetDataElement(gdcm::Tag (TAG_NEXT_DIRECTORY_RECORD_OFFSET)));

      // Record Type
      auto record_type = trim(get_string(item, gdcm::Tag (TAG_DIRECTORY_RECORD_TYPE)));

      // std::cout << "record_type " << record_type << " at " << item_offsets[item_index-1] << std::endl;
      // std::cout << " child_offset " << child_offset.GetValue() << std::endl;
      // std::cout << " sibling_offset " << sibling_offset.GetValue() << std::endl;

      // Extract patient information
      if (record_type == "PATIENT") {
        auto patient = std::make_shared<Patient>();
        patient->name = get_string(item, gdcm::Tag (TAG_PATIENTS_NAME));
        patient->id = get_string(item, gdcm::Tag (TAG_PATIENT_ID));

        patient->child_offset = child_offset.GetValue();
        patient->sibling_offset = sibling_offset.GetValue();
        patient_offsets[item_offsets[item_index-1]] = patient;

        // Extract study information
      } else if (record_type == "STUDY") {
        auto study = std::make_shared<Study>();
        study->date = get_string(item, gdcm::Tag (TAG_STUDY_DATE));
        study->description = get_string(item, gdcm::Tag (TAG_STUDY_DESCRIPTION));

        study->child_offset = child_offset.GetValue();
        study->sibling_offset = sibling_offset.GetValue();
        study_offsets[item_offsets[item_index-1]] = study;

        // Extract series information
      } else if (record_type == "SERIES") {
        auto series = std::make_shared<Series>();
        series->modality = get_string(item, gdcm::Tag (TAG_MODALITY));
        series->description = get_string(item, gdcm::Tag (TAG_SERIES_DESCRIPTION));

        series->child_offset = child_offset.GetValue();
        series->sibling_offset = sibling_offset.GetValue();
        series_offsets[item_offsets[item_index-1]] = series;

        // Extract image information
      } else if (record_type == "IMAGE") {
        auto image = std::make_shared<Image>();
        image->path = get_string(item, gdcm::Tag (TAG_REFERENCED_FILE_ID));
        image->number = -1;
        gdcm::Attribute<TAG_INSTANCE_NUMBER> instance_number;
        if (item.FindDataElement(instance_number.GetTag())) {
          auto &de = item.GetDataElement(instance_number.GetTag());
          if (!de.IsEmpty() && !de.IsUndefinedLength()) {
            instance_number.SetFromDataElement(item.GetDataElement(instance_number.GetTag()));
            image->number = instance_number.GetValue();
          }
        }

        image->child_offset = child_offset.GetValue();
        image->sibling_offset = sibling_offset.GetValue();
        image_offsets[item_offsets[item_index-1]] = image;
      } else {
        auto other = std::make_shared<Other>();

        other->child_offset = child_offset.GetValue();
        other->sibling_offset = sibling_offset.GetValue();
        other_offsets[item_offsets[item_index-1]] = other;
      }
    }

    // Check validity
    if (patient_offsets.size() == 0)
      throw std::runtime_error("Unable to find patient record");

    reassemble_hierarchy(series_offsets, image_offsets, other_offsets);
    reassemble_hierarchy(study_offsets, series_offsets, other_offsets);
    reassemble_hierarchy(patient_offsets, study_offsets, other_offsets);

    // Set the new root
    for (auto &patient : patient_offsets) {
      _patients.push_back(patient.second);
    }
  }

  return _patients;
}


void load_dicom_image(vvVolDesc *vd, virvo::gdcm::dicom_meta &meta, bool verbose)
{
  int loglevel = verbose ? 0 : 1;
  gdcm::ImageReader reader;
  reader.SetFileName( vd->getFilename() );
  if( !reader.Read() )
  {
    VV_LOG(0) << "Could not read image from: " << vd->getFilename();
    throw virvo::fileio::exception("read error");
  }
  const gdcm::File &file = reader.GetFile();
  const gdcm::DataSet &ds = file.GetDataSet();
  const gdcm::Image &image = reader.GetImage();
  const double *dircos = image.GetDirectionCosines();
  gdcm::Orientation::OrientationType type = gdcm::Orientation::GetType(dircos);
  const char *label = gdcm::Orientation::GetLabel( type );
  //image.Print( std::cerr );
  VV_LOG(loglevel+0) << "Loading '" << vd->getFilename() << "'";
  VV_LOG(loglevel+1) << "  Orientation Label: " << label;
  bool lossy = image.IsLossy();
  VV_LOG(loglevel+1) << "  Encapsulated Stream was found to be: " << (lossy ? "lossy" : "lossless");

  const unsigned int *dim = image.GetDimensions();
  vd->vox[0] = dim[0];
  vd->vox[1] = dim[1];
  vd->vox[2] = 1;
  const double *spacing = image.GetSpacing();
  vd->setDist(static_cast<float>(spacing[0]),
      static_cast<float>(spacing[1]),
      static_cast<float>(spacing[2]));
  gdcm::PixelFormat pf = image.GetPixelFormat();
  switch(pf.GetBitsAllocated()/8)
  {
    case 1:
    case 2:
      vd->bpc = pf.GetBitsAllocated()/8;
      vd->setChan(1);
      break;
    case 3:
    case 4:
      vd->bpc = 1;
      vd->setChan(pf.GetBitsAllocated()/8);
      break;
    default: assert(0); break;
  }
  VV_LOG(loglevel+1) << "  " << vd->vox[0] << "x" << vd->vox[1] << " pixels, " << pf.GetBitsAllocated() << " bits/pixel, slice no. " << meta.slice;

  gdcm::Attribute<TAG_SEQUENCE_NUMBER> attrSequenceNumber;
  if (ds.FindDataElement(attrSequenceNumber.GetTag()))
  {
    attrSequenceNumber.Set(ds);
    meta.sequence = attrSequenceNumber.GetValue();
  }

  gdcm::Attribute<TAG_INSTANCE_NUMBER> attrImageNumber;
  if (ds.FindDataElement(attrImageNumber.GetTag()))
  {
    attrImageNumber.Set(ds);
    meta.slice = attrImageNumber.GetValue();
  }

  gdcm::Attribute<TAG_SLICE_LOCATION> attrSliceLocation;
  if (ds.FindDataElement(attrSliceLocation.GetTag()))
  {
    attrSliceLocation.Set(ds);
    meta.spos = attrSliceLocation.GetValue();
  }

  VV_LOG(loglevel+2) << "  buffer length: " << image.GetBufferLength() << " for " << vd->vox[0]*vd->vox[1] << " pixels";
  char *rawData = new char[image.GetBufferLength()];
  image.GetBuffer(rawData);
  vd->addFrame((uint8_t *)rawData, vvVolDesc::ARRAY_DELETE, meta.slice);
  ++vd->frames;

  meta.slope = image.GetSlope();
  meta.intercept = image.GetIntercept();

  if (pf == gdcm::PixelFormat::INT8)
  {
      meta.format = virvo::PF_R8;
  }
  else if (pf == gdcm::PixelFormat::UINT8)
  {
      meta.format = virvo::PF_R8;
  }
  if (pf == gdcm::PixelFormat::INT16)
  {
    meta.format = virvo::PF_R16I;
  }
  else if (pf == gdcm::PixelFormat::UINT16)
  {
    meta.format = virvo::PF_R16UI;
  }
  else if (pf == gdcm::PixelFormat::INT32)
  {
    meta.format = virvo::PF_R32I;
  }
  else if (pf == gdcm::PixelFormat::UINT32)
  {
    meta.format = virvo::PF_R32UI;
  }
  else
  {
      VV_LOG(0) << "unsupported pixel format";
  }
}


void load_dicom_dir(vvVolDesc *vd, virvo::gdcm::dicom_meta &meta)
{
  DICOMDIRReader reader;
  std::shared_ptr<DICOMDIRReader::Series> seriesToRead;
  size_t maxNumSlices = 0;
  int seriesIndex = -1;

  int ll = 0;
  if (vd->getEntry() >= 0)
  {
    ll = 1;
    VV_LOG(1) << "requested series: " << vd->getEntry();
  }

  try
  {
    auto &patients = reader.load(vd->getFilename());

    int cur = 0;
    for (auto &patient : patients)
    {
      VV_LOG(ll) << "PATIENT: "
                 << "NAME: " << patient->name
                 << ", ID: " << patient->id;

      for (auto &study : patient->children)
      {
        VV_LOG(ll) << "    STUDY: "
                   << "DATE: " << study->date
                   << ", DESCRIPTION: " << study->description;

        for (auto &series : study->children)
        {
          VV_LOG(ll) << "        #" << cur
                     << ", modality: " << series->modality
                     << ", #images: " << series->children.size()
                     << " '" << series->description << "'";

          if ((vd->getEntry()>=0 && cur==vd->getEntry()) || (vd->getEntry()<0 && series->children.size() > maxNumSlices))
          {
            maxNumSlices = series->children.size();
            seriesToRead = series;
            seriesIndex = cur;
          }

          ++cur;
        }
      }
    }
  }
  catch (...)
  {
    throw virvo::fileio::exception("DICOMDIR read error");
  }

  if (!seriesToRead) {
    throw virvo::fileio::exception("sequence not found");
  }

  std::string dirname = vd->getFilename();
  std::replace(dirname.begin(), dirname.end(), '\\', '/');
  if (dirname.find('/') == std::string::npos)
    dirname.clear();
  else
    dirname = vvToolshed::extractDirname(dirname);
  if (!dirname.empty())
  {
    if (dirname[dirname.length()-1] != '/')
      dirname += "/";
  }

  VV_LOG(vd->getEntry()>=0 ? 1 : 0) << "reading series " << seriesIndex << " with " << seriesToRead->children.size() << " slices";

  std::map<int64_t, std::string> filenames;
  int slice = 0;
  bool haveImageNumber = false;
  for (auto &image: seriesToRead->children)
  {
    std::string relpath = image->path;
    std::replace(relpath.begin(), relpath.end(), '\\', '/');
    std::string filename = dirname + relpath;

    if (image->number >= 0)
    {
      if (slice > 0 && !haveImageNumber)
      {
        throw virvo::fileio::exception("inconsistent numbering");
      }
      haveImageNumber = true;
      filenames[image->number] = filename;
    }
    else
    {
      if (haveImageNumber)
        throw virvo::fileio::exception("inconsistent numbering");
      filenames[slice] = filename;
    }
    ++slice;
  }

  slice = 0;
  int lastSlice = -1;
  for (auto &f: filenames)
  {
    const auto &filename = f.second;

    vvVolDesc *newVD = new vvVolDesc(filename.c_str());
    try
    {
      virvo::gdcm::dicom_meta newMeta;
      load_dicom_image(newVD, newMeta, false);
      if (slice == 0)
      {
        meta = newMeta;
      }
      else
      {
        if (meta.format != newMeta.format)
        {
          VV_LOG(0) << "pixel format for slice " << slice << " does not match";
          throw virvo::fileio::exception("format error: slice formats do not match");
        }
        if (meta.slope != newMeta.slope)
        {
          VV_LOG(0) << "slope for slice " << slice << " does not match: " << newMeta.slope << " instead of " << meta.slope;
          throw virvo::fileio::exception("format error: slice slopes do not match");
        }
        if (meta.intercept != newMeta.intercept)
        {
          VV_LOG(0) << "intercept for slice " << slice << " does not match: " << newMeta.intercept << " instead of " << meta.intercept;
          throw virvo::fileio::exception("format error: slice intercepts do not match");
        }
      }
      VV_LOG(2) << "slice #" << slice << " -> " << newMeta.slice;
      if (newMeta.slice >= 0)
      {
        if (lastSlice>=0 && newMeta.slice!=lastSlice+1)
        {
          VV_LOG(0) << "slice index for slice " << slice << " does not match: " << newMeta.slice << " instead of " << lastSlice+1;
          if (newMeta.slice <= lastSlice)
            throw virvo::fileio::exception("format error: did not load slice with expected number");
        }
        lastSlice = newMeta.slice;
      }
      else
      {
        lastSlice = slice;
      }
    }
    catch (...)
    {
      delete newVD;
      newVD = nullptr;
      VV_LOG(0) << "failed to load " << filename;
      if (slice == 0)
      {
        throw;
      }
      else
      {
          VV_LOG(0) << "only loaded " << slice+1 << " slices from " << filenames.size();
      }
    }

    if (newVD)
    {
      vvVolDesc::ErrorType mergeErr = vd->merge(newVD, vvVolDesc::VV_MERGE_SLABS2VOL);
      delete newVD;
      if (mergeErr != vvVolDesc::OK)
      {
        if (slice == 0)
        {
          throw virvo::fileio::exception("format error: cannot merge slices");
        }
        else
        {
          VV_LOG(0) << "only loaded " << slice+1 << " slices from " << filenames.size();
        }
      }
    }
    else
    {
      return;
    }

    ++slice;
  }

}

}

namespace virvo {

namespace gdcm {

bool can_load(const vvVolDesc *vd)
{
  ::gdcm::Reader reader;
  reader.SetFileName(vd->getFilename());
  if(reader.CanRead())
    return true;

  return false;
}



dicom_meta load(vvVolDesc *vd)
{
  dicom_meta meta;

  namespace gdcm = ::gdcm;

  //const char *filename = argv[1];
  //std::cout << "filename: " << filename << std::endl;
  gdcm::Reader reader;
  reader.SetFileName(vd->getFilename());
  if( !reader.Read() )
  {
    throw fileio::exception("read error");
  }

  const gdcm::File &file = reader.GetFile();
  gdcm::MediaStorage ms;
  ms.SetFromFile(file);
  /*
   * Until gdcm::MediaStorage is fixed only *compile* time constant will be handled
   * see -> http://chuckhahm.com/Ischem/Zurich/XX_0134
   * which make gdcm::UIDs useless :(
   */
  if( ms.IsUndefined() )
  {
    throw fileio::exception("unknown media storage");
  }

  gdcm::UIDs uid;
  uid.SetFromUID( ms.GetString() );
  VV_LOG(2) << "MediaStorage is " << ms << " [" << uid.GetName() << "]";
  const gdcm::TransferSyntax &ts = file.GetHeader().GetDataSetTransferSyntax();
  uid.SetFromUID( ts.GetString() );
  VV_LOG(2) << "TransferSyntax is " << ts << " [" << uid.GetName() <<  "]";

  if ( ms == gdcm::MediaStorage::MediaStorageDirectoryStorage )
  {
    load_dicom_dir(vd, meta);
  }
  else if( gdcm::MediaStorage::IsImage( ms ) )
  {
    load_dicom_image(vd, meta, true);
    // Make big endian data:
    // TODO if (prop.littleEndian) vd->toggleEndianness(vd->frames-1);

    // Shift bits so that most significant used bit is leftmost:
    //vd->bitShiftData(pf.GetHighBit() - (pf.GetBitsAllocated() - 1), vd->frames-1);

    /*   if( md5sum )
         {
         char *buffer = new char[ image.GetBufferLength() ];
         image.GetBuffer( buffer );
         char digest[33] = {};
         gdcm::MD5::Compute( buffer, image.GetBufferLength(), digest );
         std::cerr << "md5sum: " << digest << std::endl;
         delete[] buffer;
         }*/
  }
  else if ( ms == gdcm::MediaStorage::EncapsulatedPDFStorage )
  {
    std::cerr << "  Encapsulated PDF File not supported yet" << std::endl;
    throw fileio::exception("format error: encapsulated PDF not supported");
  }
  // Do the IOD verification !
  //bool v = defs.Verify( file );
  //std::cerr << "IOD Verification: " << (v ? "succeed" : "failed") << std::endl;

  return meta;
}

} // namespace gdcm
} // namespace virvo

#endif
