// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_NODE_H
#define LAMURE_NODE_H

#include <lamure/prov/dense_meta_data.h>
#include <lamure/prov/dense_point.h>
#include <lamure/prov/partition.h>
#include <lamure/prov/partitionable.h>

#include <boost/asio/io_service.hpp>
#include <boost/bind.hpp>
#include <boost/thread/thread.hpp>

namespace lamure {
namespace prov
{
typedef pair<DensePoint, DenseMetaData> dense_pair;

class OctreeNode : public Partition<dense_pair, DenseMetaData>, public Partitionable<OctreeNode>
{
  public:
    OctreeNode() : Partition<dense_pair, DenseMetaData>(), Partitionable<OctreeNode>()
    {
        this->_min = vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
        this->_max = vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        this->_depth = 0;
        //        printf("\nOctreeNode created at depth: 0\n");
    }
    OctreeNode(uint8_t _depth) : Partition<dense_pair, DenseMetaData>(), Partitionable<OctreeNode>()
    {
        this->_min = vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
        this->_max = vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        this->_depth = _depth;
        //        printf("\nOctreeNode created at depth: %u\n", this->_depth);
    }
    OctreeNode(uint8_t _depth, Sort _sort, uint8_t _max_depth, uint8_t _min_per_node, bool _cubic_nodes) : Partition<dense_pair, DenseMetaData>(), Partitionable<OctreeNode>()
    {
        this->_min = vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
        this->_max = vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        this->_depth = _depth;
        this->_sort = _sort;
        this->_max_depth = _max_depth;
        this->_min_per_node = _min_per_node;
        this->_cubic_nodes = _cubic_nodes;
        //        printf("\nOctreeNode created at depth: %u\n", this->_depth);
    }
    ~OctreeNode() {}

    virtual OctreeNode *lookup_node_at_position(vec3f position)
    {
        //        printf("\nEnter lookup node at position\n");

        if(!fits_in_boundaries(position))
        {
            //            printf("\nPosition does not fit into boundaries\n");
            return nullptr;
        }

        if(this->_partitions.empty())
        {
            //            printf("\nMaximum depth reached at this position\n");
            return this;
        }

        for(uint8_t i = 0; i < this->_partitions.size(); i++)
        {
            //            printf("\nEnter partition lookup\n");
            OctreeNode *node = this->_partitions.at(i).lookup_node_at_position(position);
            if(node != nullptr)
            {
                return node;
            }
        }

        return this;
    }

    uint8_t get_depth() { return this->_depth; }
    vec3f get_center() { return vec3f((_max.x + _min.x) / 2, (_max.y + _min.y) / 2, (_max.z + _min.z) / 2); }
    vec3f get_max() { return _max; }
    vec3f get_min() { return _min; }

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar &_depth;
        ar &_max.x;
        ar &_max.y;
        ar &_max.z;
        ar &_min.x;
        ar &_min.y;
        ar &_min.z;
        ar &_cubic_nodes;
        ar &_aggregate_metadata;
        ar &_partitions;
    }

  protected:
    template <class RandomAccessIter, class Right_shift, class Compare>
    void sort(RandomAccessIter first, RandomAccessIter last, Right_shift rshift, Compare comp)
    {
        switch(this->_sort)
        {
        case STD_SORT:
            std::sort(first, last, comp);
            break;
        case BOOST_SPREADSORT:
            float_sort(first, last, rshift, comp);
            break;
        case PDQ_SORT:
            pdqsort(first, last, comp);
            break;
        default:
            throw new std::runtime_error("\nUnrecognized sorting algorithm requested\n");
        }
    }

    void partition()
    {
        if(!_cubic_nodes)
            this->identify_boundaries();

        auto lambda_x = [](const s_ptr<dense_pair> &pair1, const s_ptr<dense_pair> &pair2) -> bool { return pair1->first.get_position().x < pair2->first.get_position().x; };
        auto lambda_y = [](const s_ptr<dense_pair> &pair1, const s_ptr<dense_pair> &pair2) -> bool { return pair1->first.get_position().y < pair2->first.get_position().y; };
        auto lambda_z = [](const s_ptr<dense_pair> &pair1, const s_ptr<dense_pair> &pair2) -> bool { return pair1->first.get_position().z < pair2->first.get_position().z; };

        DensePoint mid_point(get_center(), vec3f(), vec<uint8_t>(), vec3f());
        dense_pair *mid = new dense_pair();
        mid->first = mid_point;
        s_ptr<dense_pair> mid_ptr(mid);

        if(this->_depth < _max_depth)
        {
            // std::cout << "Sorting node depth " << (size_t)_depth << " " << std::endl;
            // std::cout << "with " << this->_pair_ptrs.size() << " pairs" << std::endl;

            sort(this->_pair_ptrs.begin(), this->_pair_ptrs.end(), rightshift_x(), lambda_x);

            auto mid_x_pos = std::upper_bound(this->_pair_ptrs.begin(), this->_pair_ptrs.end(), mid_ptr, lambda_x);

            sort(this->_pair_ptrs.begin(), mid_x_pos, rightshift_y(), lambda_y);
            sort(mid_x_pos, this->_pair_ptrs.end(), rightshift_y(), lambda_y);

            auto mid_y_pos_1 = std::upper_bound(this->_pair_ptrs.begin(), mid_x_pos, mid_ptr, lambda_y);
            auto mid_y_pos_2 = std::upper_bound(mid_x_pos, this->_pair_ptrs.end(), mid_ptr, lambda_y);

            sort(this->_pair_ptrs.begin(), mid_y_pos_1, rightshift_z(), lambda_z);
            sort(mid_y_pos_1, mid_x_pos, rightshift_z(), lambda_z);
            sort(mid_x_pos, mid_y_pos_2, rightshift_z(), lambda_z);
            sort(mid_y_pos_2, this->_pair_ptrs.end(), rightshift_z(), lambda_z);

            auto mid_z_pos_1 = std::upper_bound(this->_pair_ptrs.begin(), mid_y_pos_1, mid_ptr, lambda_z);
            auto mid_z_pos_2 = std::upper_bound(mid_y_pos_1, mid_x_pos, mid_ptr, lambda_z);
            auto mid_z_pos_3 = std::upper_bound(mid_x_pos, mid_y_pos_2, mid_ptr, lambda_z);
            auto mid_z_pos_4 = std::upper_bound(mid_y_pos_2, this->_pair_ptrs.end(), mid_ptr, lambda_z);

            vec<vec<s_ptr<dense_pair>>::iterator> iter_vec = vec<vec<s_ptr<dense_pair>>::iterator>();

            iter_vec.push_back(this->_pair_ptrs.begin());
            iter_vec.push_back(mid_z_pos_1);
            iter_vec.push_back(mid_y_pos_1);
            iter_vec.push_back(mid_z_pos_2);
            iter_vec.push_back(mid_x_pos);
            iter_vec.push_back(mid_z_pos_3);
            iter_vec.push_back(mid_y_pos_2);
            iter_vec.push_back(mid_z_pos_4);
            iter_vec.push_back(this->_pair_ptrs.end());

            for(uint8_t i = 0; i < 8; i++)
            {
                vec<s_ptr<dense_pair>> pair_ptrs(iter_vec.at(i), iter_vec.at(i + 1));
                if(pair_ptrs.size() >= _min_per_node)
                {
                    OctreeNode octree_node(this->_depth + 1, _sort, _max_depth, _min_per_node, this->_cubic_nodes);
                    octree_node.set_pair_ptrs(pair_ptrs);

                    if(this->_cubic_nodes)
                    {
                        vec3f min, max;
                        vec3f _center = get_center();
                        switch(i)
                        {
                        case 0:
                            min = vec3f(_min.x, _min.y, _min.z);
                            max = vec3f(_center.x, _center.y, _center.z);
                            break;
                        case 1:
                            min = vec3f(_min.x, _min.y, _center.z);
                            max = vec3f(_center.x, _center.y, _max.z);
                            break;
                        case 2:
                            min = vec3f(_min.x, _center.y, _min.z);
                            max = vec3f(_center.x, _max.y, _center.z);
                            break;
                        case 3:
                            min = vec3f(_min.x, _center.y, _center.z);
                            max = vec3f(_center.x, _max.y, _max.z);
                            break;
                        case 4:
                            min = vec3f(_center.x, _min.y, _min.z);
                            max = vec3f(_max.x, _center.y, _center.z);
                            break;
                        case 5:
                            min = vec3f(_center.x, _min.y, _center.z);
                            max = vec3f(_max.x, _center.y, _max.z);
                            break;
                        case 6:
                            min = vec3f(_center.x, _center.y, _min.z);
                            max = vec3f(_max.x, _max.y, _center.z);
                            break;
                        case 7:
                            min = vec3f(_center.x, _center.y, _center.z);
                            max = vec3f(_max.x, _max.y, _max.z);
                            break;
                        }
                        octree_node.set_boundaries(min, max);
                    }

                    this->_partitions.push_back(octree_node);
                }
            }

            if(_partitions.size() != 8)
            {
                aggregate_metadata();
            }

            for(uint8_t i = 0; i < _partitions.size(); i++)
            {
                this->_partitions.at(i).partition();
            }
        }
        else
        {
            aggregate_metadata();
        }
    }

    void aggregate_metadata()
    {
        float photometric_consistency = 0;
        std::set<uint32_t> seen = std::set<uint32_t>();
        std::set<uint32_t> not_seen = std::set<uint32_t>();

        for(uint64_t i = 0; i < this->_pair_ptrs.size(); i++)
        {
            photometric_consistency = photometric_consistency + (this->_pair_ptrs.at(i)->second.get_photometric_consistency() - photometric_consistency) / (i + 1);
            for(uint32_t k = 0; k < this->_pair_ptrs.at(i)->second.get_images_seen().size(); k++)
            {
                seen.insert(this->_pair_ptrs.at(i)->second.get_images_seen().at(k));
            }
            for(uint32_t k = 0; k < this->_pair_ptrs.at(i)->second.get_images_not_seen().size(); k++)
            {
                not_seen.insert(this->_pair_ptrs.at(i)->second.get_images_not_seen().at(k));
            }
        }

        //            printf("\nSeen set length: %u\n", seen.size());

        vec<uint32_t> images_seen = vec<uint32_t>(seen.begin(), seen.end());
        vec<uint32_t> images_not_seen = vec<uint32_t>(not_seen.begin(), not_seen.end());

        _aggregate_metadata.set_photometric_consistency(photometric_consistency);
        _aggregate_metadata.set_images_seen(images_seen);
        _aggregate_metadata.set_images_not_seen(images_not_seen);

        //            printf("\nPoints in leaf node: %u\n", this->_pairs.size());
        //            printf("\nNode NCC: %lf\n", _aggregate_metadata.get_photometric_consistency());
        //            printf("\nNode num seen: %u\n", _aggregate_metadata.get_images_seen().size());
        //            printf("\nNode num not seen: %u\n", _aggregate_metadata.get_images_not_seen().size());
    }

    void identify_boundaries()
    {
        for(uint64_t i = 0; i < this->_pair_ptrs.size(); i++)
        {
            if(std::fpclassify(this->_pair_ptrs.at(i)->first.get_position().x) == FP_SUBNORMAL    //
               || std::fpclassify(this->_pair_ptrs.at(i)->first.get_position().y) == FP_SUBNORMAL //
               || std::fpclassify(this->_pair_ptrs.at(i)->first.get_position().z) == FP_SUBNORMAL)
            {
                // Check for denormals, remove if such exist
                std::cout << "Erasing denormal at position" << (size_t)i << " " << std::endl;
                this->_pair_ptrs.erase(this->_pair_ptrs.begin() + i);
            }

            if(std::fpclassify(this->_pair_ptrs.at(i)->first.get_position().x) == FP_NAN    //
               || std::fpclassify(this->_pair_ptrs.at(i)->first.get_position().y) == FP_NAN //
               || std::fpclassify(this->_pair_ptrs.at(i)->first.get_position().z) == FP_NAN)
            {
                // Check for NaNs, remove if such exist
                std::cout << "Erasing NaN at position" << (size_t)i << " " << std::endl;
                this->_pair_ptrs.erase(this->_pair_ptrs.begin() + i);
            }

            if(this->_min.x > this->_pair_ptrs.at(i)->first.get_position().x)
            {
                this->_min.x = this->_pair_ptrs.at(i)->first.get_position().x;
            }

            if(this->_min.y > this->_pair_ptrs.at(i)->first.get_position().y)
            {
                this->_min.y = this->_pair_ptrs.at(i)->first.get_position().y;
            }

            if(this->_min.z > this->_pair_ptrs.at(i)->first.get_position().z)
            {
                this->_min.z = this->_pair_ptrs.at(i)->first.get_position().z;
            }

            if(this->_max.x < this->_pair_ptrs.at(i)->first.get_position().x)
            {
                this->_max.x = this->_pair_ptrs.at(i)->first.get_position().x;
            }

            if(this->_max.y < this->_pair_ptrs.at(i)->first.get_position().y)
            {
                this->_max.y = this->_pair_ptrs.at(i)->first.get_position().y;
            }

            if(this->_max.z < this->_pair_ptrs.at(i)->first.get_position().z)
            {
                this->_max.z = this->_pair_ptrs.at(i)->first.get_position().z;
            }
        }

        // printf("\nMin: %f, %f, %f", this->_min.x, this->_min.y, this->_min.z);
        // printf("\nMax: %f, %f, %f\n", this->_max.x, this->_max.y, this->_max.z);
    }

    void set_boundaries(vec3f min, vec3f max)
    {
        this->_min = min;
        this->_max = max;
    }

    vec3f _min;
    vec3f _max;
    uint8_t _depth;
    bool _cubic_nodes = false;

    bool fits_in_boundaries(vec3f position)
    {
        return !(position.x > this->_max.x || position.x < this->_min.x || position.y > this->_max.y || position.y < this->_min.y || position.z > this->_max.z || position.z < this->_min.z);
    }

    struct rightshift_x
    {
        int operator()(const s_ptr<dense_pair> &pair, const unsigned offset) const { return float_mem_cast<float, int>(pair->first.get_position().x) >> offset; }
    };

    struct rightshift_y
    {
        int operator()(const s_ptr<dense_pair> &pair, const unsigned offset) const { return float_mem_cast<float, int>(pair->first.get_position().y) >> offset; }
    };

    struct rightshift_z
    {
        int operator()(const s_ptr<dense_pair> &pair, const unsigned offset) const { return float_mem_cast<float, int>(pair->first.get_position().z) >> offset; }
    };
};
}
}

#endif // LAMURE_NODE_H
