// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_LODMETADATA_H
#define LAMURE_LODMETADATA_H

#include <lamure/prov/meta_data.h>
#include <lamure/prov/common.h>

namespace lamure {
namespace prov
{
class DenseMetaData : public MetaData
{
  public:
    DenseMetaData()
    {
        _images_seen = vec<uint32_t>();
        _images_not_seen = vec<uint32_t>();
    }
    ~DenseMetaData() {}

    virtual void read_metadata(ifstream &is, uint32_t meta_data_length) override
    {
        MetaData::read_metadata(is, meta_data_length);

        uint32_t data_pointer = 0;

        float buffer = 0;
        memcpy(&buffer, &_metadata[data_pointer], 4);
        _photometric_consistency = (float)swap(buffer, true);
        data_pointer += 4;

        // printf("\nNCC: %f", _photometric_consistency);

        uint32_t num_seen = 0;
        memcpy(&num_seen, &_metadata[data_pointer], 4);
        num_seen = swap(num_seen, true);
        data_pointer += 4;

        // printf("\nNum seen: %i", num_seen);

        for(uint32_t i = 0; i < num_seen; i++)
        {
            uint32_t image_seen = 0;
            memcpy(&image_seen, &_metadata[data_pointer], 4);
            image_seen = swap(image_seen, true);
            data_pointer += 4;

            //            printf("\nImage seen: %i", image_seen);

            _images_seen.push_back(image_seen);
        }

        uint32_t num_not_seen = 0;
        memcpy(&num_not_seen, &_metadata[data_pointer], 4);
        num_not_seen = swap(num_not_seen, true);
        data_pointer += 4;

        // printf("\nNum not seen: %i", num_not_seen);

        for(uint32_t i = 0; i < num_not_seen; i++)
        {
            uint32_t image_not_seen = 0;
            memcpy(&image_not_seen, &_metadata[data_pointer], 4);
            image_not_seen = swap(image_not_seen, true);
            data_pointer += 4;

            //            printf("\nImage not seen: %i", image_not_seen);

            _images_not_seen.push_back(image_not_seen);
        }
    }

    float get_photometric_consistency() const { return _photometric_consistency; }
    vec<uint32_t> get_images_seen() const { return _images_seen; }
    vec<uint32_t> get_images_not_seen() const { return _images_not_seen; }
    void set_photometric_consistency(float _photometric_consistency) { this->_photometric_consistency = _photometric_consistency; }
    void set_images_seen(vec<uint32_t> _images_seen) { this->_images_seen = _images_seen; }
    void set_images_not_seen(vec<uint32_t> _images_not_seen) { this->_images_not_seen = _images_not_seen; }

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar &_photometric_consistency;
        ar &_images_seen;
        ar &_images_not_seen;
    }

  private:
    float _photometric_consistency;
    vec<uint32_t> _images_seen;
    vec<uint32_t> _images_not_seen;
};
}
}

#endif // LAMURE_LODMETADATA_H
