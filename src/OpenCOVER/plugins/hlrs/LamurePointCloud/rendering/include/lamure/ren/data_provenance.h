// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef REN_PROVENANCE_H_
#define REN_PROVENANCE_H_

#include <lamure/ren/3rd_party/json.h>
#include <fstream>
#include <lamure/ren/item_provenance.h>
#include <lamure/ren/policy.h>
#include <lamure/types.h>
#include <lamure/utils.h>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

namespace lamure
{
namespace ren
{
class Data_Provenance
{
  public:
    Data_Provenance() {
      _size_in_bytes = policy::get_instance()->size_of_provenance();
    };

    static Data_Provenance parse_json(std::string path)
    {
        Data_Provenance data_provenance;

        std::ifstream ifs(path);
        std::string json((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));

        picojson::value v;
        std::string err = picojson::parse(v, json);
        if(!err.empty())
        {
            std::cout << err << std::endl;
            exit(2);
        }

        // check if the type of the value is "array"
        if(!v.is<picojson::array>())
        {
            std::cout << "JSON is not an array" << std::endl;
            exit(2);
        }

        // obtain a const reference to the map, and print the contents
        const picojson::value::array &array = v.get<picojson::array>();
        for(picojson::value::array::const_iterator iter = array.begin(); iter != array.end(); ++iter)
        {
            picojson::value::object obj = (*iter).get<picojson::value::object>();
            Item_Provenance::type_item type;
            Item_Provenance::visualization_item visualization;

            picojson::value::object::const_iterator iter_type = obj.find("type");
            if(iter_type != obj.end())
            {
                if(iter_type->second.get<std::string>() == "int")
                {
                    type = Item_Provenance::type_item::TYPE_INT;
                }
                else if(iter_type->second.get<std::string>() == "float")
                {
                    type = Item_Provenance::type_item::TYPE_FLOAT;
                }
                else if(iter_type->second.get<std::string>() == "vec3i")
                {
                    type = Item_Provenance::type_item::TYPE_VEC3I;
                }
                else if(iter_type->second.get<std::string>() == "vec3f")
                {
                    type = Item_Provenance::type_item::TYPE_VEC3F;
                }
            }

            picojson::value::object::const_iterator iter_visualization = obj.find("visualization");
            if(iter_visualization != obj.end())
            {
                if(iter_visualization->second.get<std::string>() == "color")
                {
                    visualization = Item_Provenance::visualization_item::VISUALIZATION_COLOR;
                }
                else if(iter_visualization->second.get<std::string>() == "arrow")
                {
                    visualization = Item_Provenance::visualization_item::VISUALIZATION_ARROW;
                }
            }

            Item_Provenance item_provenance(type, visualization);
            data_provenance.add_item(item_provenance);
        }

        policy::get_instance()->set_size_of_provenance(data_provenance.get_size_in_bytes());

        return data_provenance;
    }

    std::vector<Item_Provenance> get_items() const { return _items_provenance; };
    int32_t get_size_in_bytes() const { return _size_in_bytes; };

  private:
    void add_item(Item_Provenance &item_provenance)
    {
        _items_provenance.push_back(item_provenance);
        _size_in_bytes += item_provenance.get_size_in_bytes();
    };

    std::vector<Item_Provenance> _items_provenance;
    int32_t _size_in_bytes;
};
}
} // namespace lamure

#endif // REN_PROVENANCE_H_
