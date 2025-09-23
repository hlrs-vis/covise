// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/prov/aux_stream.h>

#include <lamure/prov/octree.h>

namespace lamure {
namespace prov {

aux_stream::
aux_stream()
: filename_(""),
  num_segments_(0) {


}

aux_stream::
~aux_stream() {
    close_stream(false);
}

void aux_stream::
open_stream(const std::string& aux_filename,
           const aux_stream_type type) {

    close_stream(false);
    
    num_segments_ = 0;
    filename_ = aux_filename;
    type_ = type;

    std::ios::openmode mode = std::ios::binary;

    if (type_ == aux_stream_type::AUX_STREAM_IN) {
        mode |= std::ios::in;
    }
    if (type_ == aux_stream_type::AUX_STREAM_OUT) {
        mode |= std::ios::out;
        mode |= std::ios::trunc;
    }

    file_.open(filename_, mode);

    if (!file_.is_open() || !file_.good()) {
        throw std::runtime_error(
            "lamure: aux_stream::Unable to open stream: " + filename_);
    }
   
}

void aux_stream::
close_stream(const bool remove_file) {
    
    if (file_.is_open()) {
        if (type_ == aux_stream_type::AUX_STREAM_OUT) {
            file_.flush();
        }
        file_.close();
        if (file_.fail()) {
            throw std::runtime_error(
                "lamure: aux_stream::Unable to close stream: " + filename_);
        }

        if (type_ == aux_stream_type::AUX_STREAM_OUT) {
            if (remove_file) {
                if (std::remove(filename_.c_str())) {
                    throw std::runtime_error(
                        "lamure: aux_stream::Unable to delete file: " + filename_);
                }

            }
        }
    }

}


void aux_stream::
write(aux_stream::aux_serializable& serializable) {

    if (!file_.is_open()) {
        throw std::runtime_error(
            "lamure: aux_stream::Unable to serialize: " + filename_);
    }

    aux_sig sig;

    size_t allocated_size = sig.size() + serializable.size();
    size_t used_size = allocated_size;
    size_t padding = 0;
    while (allocated_size % 32 != 0) {
        ++allocated_size;
        ++padding;
    }
    
    serializable.signature(sig.signature_);

    sig.reserved_ = 0;
    sig.allocated_size_ = allocated_size - sig.size();
    sig.used_size_ = used_size - sig.size();

    sig.serialize(file_);
    serializable.serialize(file_);

    while (padding) {
        char c = 0;
        file_.write(&c, 1);
        --padding;
    }

}


void aux_stream::
read_aux(const std::string& filename, aux& aux) {
 
    open_stream(filename, aux_stream_type::AUX_STREAM_IN);

    if (type_ != AUX_STREAM_IN) {
        throw std::runtime_error(
            "lamure: aux_stream::Failed to read aux from: " + filename_);
    }
    if (!file_.is_open()) {
         throw std::runtime_error(
            "lamure: aux_stream::Failed to read aux from: " + filename_);
    }
   
    //scan stream
    file_.seekg(0, std::ios::end);
    size_t filesize = (size_t)file_.tellg();
    file_.seekg(0, std::ios::beg);

    num_segments_ = 0;

    aux_sparse_seg sparse;
    std::vector<aux_view_seg> views;
    std::vector<aux_atlas_tile_seg> tiles;
    aux_tree_seg tree;
    aux_atlas_seg atlas;
    uint32_t sparse_id = 0;
    uint32_t camera_id = 0;
    uint32_t tile_id = 0;

    //go through entire stream and fetch the segments
    while (true) {
        aux_sig sig;
        sig.deserialize(file_);
        if (sig.signature_[0] != 'A' ||
            sig.signature_[1] != 'U' ||
            sig.signature_[2] != 'X' ||
            sig.signature_[3] != 'X') {
             throw std::runtime_error(
                 "lamure: aux_stream::Invalid magic encountered: " + filename_);
        }
            
        size_t anchor = (size_t)file_.tellg();

        switch (sig.signature_[4]) {

            case 'F': { //"AUXXFILE"
                aux_file_seg seg;
                seg.deserialize(file_);
                break;
            }
            case 'S': { 
                switch (sig.signature_[5]) {
                    case 'P': { //"AUXXSPRS"
                        sparse.deserialize(file_);
                        ++sparse_id;
                        break;
                    }
                    default: {
                        throw std::runtime_error(
                            "lamure: aux_stream::Stream corrupt -- Invalid segment encountered");
                        break;
                    }
                }
                break;
            }
            case 'V': { //"AUXXVIEW"
                aux_view_seg view;
                view.deserialize(file_);
                views.push_back(view);
                if (camera_id != view.camera_id_) {
                  throw std::runtime_error(
                    "lamure: aux_stream::Stream corrupt -- Invalid view order");
                }
                ++camera_id;
                break;
            }
            case 'A': { //"AUXXATLS"
              atlas.deserialize(file_);
              break;
            }
            case 'T': { 
                switch (sig.signature_[5]) {
                    case 'I': { //"AUXXTILE"
                        aux_atlas_tile_seg tile;
                        tile.deserialize(file_);
                        tiles.push_back(tile);
                        if (tile_id != tile.atlas_tile_id_) {
                          throw std::runtime_error(
                            "lamure: aux_stream::Stream corrupt -- Invalid tile order");
                        }
                        ++tile_id;
                        break;
                    }
                    case 'R': {  //"AUXXTREE"
                        tree.deserialize(file_);
                        break;
                    }
                    default: {
                        throw std::runtime_error(
                            "lamure: aux_stream::Stream corrupt -- Invalid segment encountered");
                        break;
                    }
                }
                break;
            }
            default: {
                throw std::runtime_error(
                    "lamure: aux_stream::file corrupt -- Invalid segment encountered");
                break;
            }
        }

        if (anchor + sig.allocated_size_ < filesize) {
          file_.seekg(anchor + sig.allocated_size_, std::ios::beg);
        }
        else {
            break;
        }

    }

    close_stream(false);

    if (sparse_id != 1) {
       throw std::runtime_error(
           "lamure: aux_stream::Stream corrupt -- Invalid number of sparse segments");
    }   

    for (uint64_t i = 0; i < sparse.num_points_; ++i) {
      aux::sparse_point p;
      p.pos_ = scm::math::vec3f(sparse.points_[i].x_, sparse.points_[i].y_, sparse.points_[i].z_);
      p.r_ = sparse.points_[i].r_;
      p.g_ = sparse.points_[i].g_;
      p.b_ = sparse.points_[i].b_;
      p.a_ = (uint8_t)255;

      for (uint32_t j = 0; j < sparse.points_[i].num_features_; ++j) {
        aux::feature f;
        f.camera_id_ = sparse.points_[i].features_[j].camera_id_;
        f.using_count_ = sparse.points_[i].features_[j].using_count_;
        f.coords_ = scm::math::vec2f(
          sparse.points_[i].features_[j].img_x_, 
          sparse.points_[i].features_[j].img_y_);
        f.error_ = scm::math::vec2f(
          sparse.points_[i].features_[j].error_x_, 
          sparse.points_[i].features_[j].error_y_);
        p.features_.push_back(f);
      }

      aux.add_sparse_point(p);
    }

    
    for (const auto& view : views) {
       aux::view v;
       v.camera_id_ = view.camera_id_;
       v.position_ = scm::math::vec3f(view.position_.x_, view.position_.y_, view.position_.z_);

       auto translation = scm::math::make_translation(v.position_);
       auto rotation = scm::math::quatf(view.orientation_.w_, view.orientation_.x_, view.orientation_.y_, view.orientation_.z_).to_matrix();
       v.transform_ = translation * rotation;
 
       v.focal_length_ = view.focal_length_;
       v.distortion_ = view.distortion_;
       v.image_width_ = view.image_width_;
       v.image_height_ = view.image_height_;
       v.atlas_tile_id_ = view.atlas_tile_id_;
       v.image_file_ = view.image_file_.string_;

       aux.add_view(v);

   
    }

    aux::atlas ta;
    ta.num_atlas_tiles_ = atlas.num_atlas_tiles_;
    ta.atlas_width_ = atlas.atlas_width_;
    ta.atlas_height_ = atlas.atlas_height_;
    ta.rotated_ = atlas.rotated_;
    aux.set_atlas(ta);

    for (const auto& tile : tiles) {
      aux::atlas_tile t;
      t.atlas_tile_id_ = tile.atlas_tile_id_;
      t.x_ = tile.x_;
      t.y_ = tile.y_;
      t.width_ = tile.width_;
      t.height_ = tile.height_;
      
      aux.add_atlas_tile(t);
    }

    std::shared_ptr<octree> ot = std::make_shared<octree>();
    ot->set_depth(tree.depth_);
    for (const auto& node : tree.nodes_) {
      octree_node n{node.idx_, node.child_mask_, node.child_idx_, 
        scm::math::vec3f(node.min_.x_, node.min_.y_, node.min_.z_),
        scm::math::vec3f(node.max_.x_, node.max_.y_, node.max_.z_),
        node.fotos_};
      ot->add_node(n);
    }
    aux.set_octree(ot);


}

void aux_stream::
write_aux(const std::string& filename, aux& aux) {

   open_stream(filename, aux_stream_type::AUX_STREAM_OUT);

   if (type_ != AUX_STREAM_OUT) {
       throw std::runtime_error(
           "lamure: aux_stream::Failed to append aux to: " + filename_);
   }
   if (!file_.is_open()) {
       throw std::runtime_error(
           "lamure: aux_stream::Failed to append aux to: " + filename_);
   }
   
   file_.seekp(0, std::ios::beg);

   aux_file_seg seg;
   seg.major_version_ = 0;
   seg.minor_version_ = 1;
   seg.reserved_ = 0;

   write(seg);


   for (uint32_t i = 0; i < aux.get_num_views(); ++i) {
     const auto& view = aux.get_view(i);
     aux_view_seg v;

     v.segment_id_ = num_segments_++;
     v.camera_id_ = view.camera_id_;
     v.position_.x_ = view.position_.x;
     v.position_.y_ = view.position_.y;
     v.position_.z_ = view.position_.z;
     v.reserved_0_ = 0;
     
     scm::math::quatf quat = scm::math::quatf::from_matrix(view.transform_);
     v.orientation_.w_ = quat.w;
     v.orientation_.x_ = quat.x;
     v.orientation_.y_ = quat.y;
     v.orientation_.z_ = quat.z;

     v.focal_length_ = view.focal_length_;
     v.distortion_ = view.distortion_;
     v.reserved_1_ = 0;
     v.reserved_2_ = 0;
     v.reserved_3_ = 0;
     v.reserved_4_ = 0;
     v.reserved_5_ = 0;
     v.reserved_6_ = 0;
     v.image_width_ = view.image_width_;
     v.image_height_ = view.image_height_;
     v.atlas_tile_id_ = view.atlas_tile_id_;
    
     aux_string image_file;
     image_file.string_ = view.image_file_;
     image_file.length_ = view.image_file_.length();
     v.image_file_ = image_file;
   
     write(v);

   }

   aux_atlas_seg ta;
   const auto& atlas = aux.get_atlas();
   ta.segment_id_ = num_segments_++;
   ta.num_atlas_tiles_ = atlas.num_atlas_tiles_;
   ta.atlas_width_ = atlas.atlas_width_;
   ta.atlas_height_ = atlas.atlas_height_;
   ta.rotated_ = atlas.rotated_;
   write(ta);

   for (uint32_t i = 0; i < aux.get_num_atlas_tiles(); ++i) {
     const auto& tile = aux.get_atlas_tile(i);
     aux_atlas_tile_seg t;

     t.segment_id_ = num_segments_++;
     t.atlas_tile_id_ = tile.atlas_tile_id_;
     t.x_ = tile.x_;
     t.y_ = tile.y_;
     t.width_ = tile.width_;
     t.height_ = tile.height_;

     write(t);

   }
   
   aux_sparse_seg sparse;

   sparse.segment_id_ = num_segments_++;
   sparse.reserved_0_ = 0;
   sparse.reserved_1_ = 0;
   sparse.reserved_2_ = 0;
   sparse.reserved_3_ = 0;
   sparse.reserved_4_ = 0;
   sparse.num_points_ = aux.get_num_sparse_points();

   for (uint64_t i = 0; i < sparse.num_points_; ++i) {
     
     const auto& point = aux.get_sparse_point(i);
     aux_sparse_point p;
     p.x_ = point.pos_.x;
     p.y_ = point.pos_.y;
     p.z_ = point.pos_.z;
     p.r_ = point.r_;
     p.g_ = point.g_;
     p.b_ = point.b_;
     p.a_ = (uint8_t)255;
     p.reserved_0_ = 0;
     p.reserved_1_ = 0;
     p.reserved_2_ = 0;
     p.num_features_ = point.features_.size();
     
     for (uint32_t j = 0; j < p.num_features_; ++j) {
       const auto& feature = point.features_[j];
       aux_feature f;
       f.camera_id_ = feature.camera_id_;
       f.using_count_ = feature.using_count_;
       f.img_x_ = feature.coords_.x;
       f.img_y_ = feature.coords_.y;
       f.error_x_ = feature.error_.x;
       f.error_y_ = feature.error_.y;
       f.reserved_0_ = 0;
       f.reserved_1_ = 0;
       p.features_.push_back(f);
     }    

     sparse.points_.push_back(p);
   }

   write(sparse);

   aux_tree_seg tree;
   tree.segment_id_ = num_segments_++;
   tree.reserved_0_ = 0;
   tree.num_nodes_ = aux.get_octree()->get_num_nodes();
   tree.depth_ = aux.get_octree()->get_depth();
   tree.reserved_1_ = 0;
   for (uint64_t i = 0; i < tree.num_nodes_; ++i) {
     const auto& node = aux.get_octree()->get_node(i);
     aux_tree_node n;
     n.child_mask_ = node.get_child_mask();
     n.child_idx_ = node.get_child_idx();
     n.min_.x_ = node.get_min().x;
     n.min_.y_ = node.get_min().y;
     n.min_.z_ = node.get_min().z;
     n.max_.x_ = node.get_max().x;
     n.max_.y_ = node.get_max().y;
     n.max_.z_ = node.get_max().z;
     n.idx_ = node.get_idx();
     n.num_fotos_ = node.get_fotos().size();
     n.fotos_ = node.get_fotos();
     tree.nodes_.push_back(n); 
   }

   write(tree);

   std::cout << "Serialized " << tree.nodes_.size() << " octree nodes" << std::endl;

   close_stream(false);

}



} } // namespace lamure

