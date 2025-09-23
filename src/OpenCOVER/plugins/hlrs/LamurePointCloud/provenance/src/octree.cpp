// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/prov/octree.h>
#include <lamure/prov/prov_aux.h>
#include <lamure/bounding_box.h>

#include <limits>
#include <vector>
#include <stack>
#include <map>


namespace lamure {
namespace prov {


octree::
octree()
: min_num_points_per_node_(16), 
  depth_(0) {


} 

octree::
~octree() {

}


void octree::
create(std::vector<aux::sparse_point>& _points) {
  nodes_.clear(); 
  depth_ = 0;
  min_num_points_per_node_ = 16;
  uint32_t max_depth = 12;

  std::vector<aux::sparse_point> points = _points;

  uint64_t num_points = points.size();
  if (num_points < min_num_points_per_node_) {
    std::cout << "Too few points " << std::endl; exit(0);
  }

  scm::math::vec3f tree_min(std::numeric_limits<float>::max());
  scm::math::vec3f tree_max(std::numeric_limits<float>::lowest());

  for (const auto& point : points) {
    tree_min.x = std::min(tree_min.x, point.pos_.x);
    tree_min.y = std::min(tree_min.y, point.pos_.y);
    tree_min.z = std::min(tree_min.z, point.pos_.z);

    tree_max.x = std::max(tree_max.x, point.pos_.x);
    tree_max.y = std::max(tree_max.y, point.pos_.y);
    tree_max.z = std::max(tree_max.z, point.pos_.z);
  }

  //make the bounding box a cube
  auto tree_dim = tree_max - tree_min;
  float longest_axis = std::max(tree_dim.x, std::max(tree_dim.y, tree_dim.z));
  tree_max = tree_min + scm::math::vec3f(longest_axis);
  
  std::cout << "tree min " << tree_min.x << " " << tree_min.y << " " << tree_min.z << std::endl;
  std::cout << "tree max " << tree_max.x << " " << tree_max.y << " " << tree_max.z << std::endl;
  
  struct auxiliary_node {
    uint32_t idx_;
    uint32_t depth_;
    uint64_t begin_;
    uint64_t end_;
    scm::math::vec3f min_;
    scm::math::vec3f max_;
  };

  uint32_t num_nodes = 0;
  std::stack<auxiliary_node> nodes_todo;
  nodes_todo.push(
    auxiliary_node{
      num_nodes++, 0, 0, num_points, tree_min, tree_max
    });


  std::map<uint64_t, octree_node> node_idx_map;

  while (!nodes_todo.empty()) {
    auxiliary_node node = nodes_todo.top();
    nodes_todo.pop();

    depth_ = std::max(depth_, node.depth_);

    //some termination criterion
    if (node.depth_ >= max_depth || node.end_-node.begin_ <= min_num_points_per_node_) {
      std::set<uint32_t> fotos;
      for (uint64_t i = node.begin_; i < node.end_; ++i) {
        const auto& sp = points[i];
        for (const auto& f : sp.features_) {
          fotos.insert(f.camera_id_);
        }
        if (node.min_.x <= sp.pos_.x && node.max_.x >= sp.pos_.x
          && node.min_.y <= sp.pos_.y && node.max_.y >= sp.pos_.y
          && node.min_.z <= sp.pos_.z && node.max_.z >= sp.pos_.z) {

        }
        else {
          std::cout << "e";
        }
      }
      node_idx_map[node.idx_] = octree_node(node.idx_, 0, 0, node.min_, node.max_, fotos);
      continue;
    }

    auto min_vertex = node.min_;
    auto max_vertex = node.max_;
    auto mid_vertex = 0.5f*(min_vertex+max_vertex);

    //sort all points between begin and end along x-axis, find midpoint
    std::sort(&points[node.begin_], &points[node.end_], 
      [](const aux::sparse_point& _l, const aux::sparse_point& _r) -> bool { 
        return _l.pos_.x < _r.pos_.x; 
      }
    );
    uint64_t mid_id_x = node.begin_;
    for (uint64_t i = node.begin_; i < node.end_; ++i) {
      if (points[i].pos_.x >= mid_vertex.x) { mid_id_x = i; break; }
    }
    
    //sort all points between begin and end along y-axis, find midpoints
    std::sort(&points[node.begin_], &points[mid_id_x], 
      [](const aux::sparse_point& _l, const aux::sparse_point& _r) -> bool { 
        return _l.pos_.y < _r.pos_.y; 
      }
    );
    uint64_t mid_id_y0 = node.begin_;
    for (uint64_t i = node.begin_; i < mid_id_x; ++i) {
      if (points[i].pos_.y >= mid_vertex.y) { mid_id_y0 = i; break; }
    }

    std::sort(&points[mid_id_x], &points[node.end_], 
      [](const aux::sparse_point& _l, const aux::sparse_point& _r) -> bool { 
        return _l.pos_.y < _r.pos_.y; 
      }
    );
    uint64_t mid_id_y1 = mid_id_x;
    for (uint64_t i = mid_id_x; i < node.end_; ++i) {
      if (points[i].pos_.y >= mid_vertex.y) { mid_id_y1 = i; break; }
    }

    //sort all points between begin and end along z-axis, find midpoints
    std::sort(&points[node.begin_], &points[mid_id_y0], 
      [](const aux::sparse_point& _l, const aux::sparse_point& _r) -> bool { 
        return _l.pos_.z < _r.pos_.z; 
      }
    );
    uint64_t mid_id_z0 = node.begin_;
    for (uint64_t i = node.begin_; i < mid_id_y0; ++i) {
      if (points[i].pos_.z >= mid_vertex.z) { mid_id_z0 = i; break; }
    }

    std::sort(&points[mid_id_y0], &points[mid_id_x], 
      [](const aux::sparse_point& _l, const aux::sparse_point& _r) -> bool { 
        return _l.pos_.z < _r.pos_.z; 
      }
    );
    uint64_t mid_id_z1 = mid_id_y0;
    for (uint64_t i = mid_id_y0; i < mid_id_x; ++i) {
      if (points[i].pos_.z >= mid_vertex.z) { mid_id_z1 = i; break; }
    }
    
    std::sort(&points[mid_id_x], &points[mid_id_y1], 
      [](const aux::sparse_point& _l, const aux::sparse_point& _r) -> bool { 
        return _l.pos_.z < _r.pos_.z; 
      }
    );
    uint64_t mid_id_z2 = mid_id_x;
    for (uint64_t i = mid_id_x; i < mid_id_y1; ++i) {
      if (points[i].pos_.z >= mid_vertex.z) { mid_id_z2 = i; break; }
    }
    
    std::sort(&points[mid_id_y1], &points[node.end_], 
      [](const aux::sparse_point& _l, const aux::sparse_point& _r) -> bool { 
        return _l.pos_.z < _r.pos_.z; 
      }
    );
    uint64_t mid_id_z3 = mid_id_y1;
    for (uint64_t i = mid_id_y1; i < node.end_; ++i) {
      if (points[i].pos_.z >= mid_vertex.z) { mid_id_z3 = i; break; }
    }

    struct range {
      uint64_t begin_;
      uint64_t end_;
    };
    std::vector<range> ranges;
    ranges.push_back({node.begin_, mid_id_z0});
    ranges.push_back({mid_id_x, mid_id_z2});
    ranges.push_back({mid_id_y0, mid_id_z1});
    ranges.push_back({mid_id_y1, mid_id_z3});
    ranges.push_back({mid_id_z0, mid_id_y0});
    ranges.push_back({mid_id_z2, mid_id_y1});
    ranges.push_back({mid_id_z1, mid_id_x});
    ranges.push_back({mid_id_z3, node.end_});

    std::vector<auxiliary_node> children;
    children.push_back(
      auxiliary_node{
        0, node.depth_+1,
        node.begin_, mid_id_z0,
        min_vertex, 
        mid_vertex,
      });
    children.push_back(
      auxiliary_node{
        0, node.depth_+1, 
        mid_id_x, mid_id_z2,
        scm::math::vec3f(mid_vertex.x, min_vertex.y, min_vertex.z),
        scm::math::vec3f(max_vertex.x, mid_vertex.y, mid_vertex.z)
      });
    children.push_back(
      auxiliary_node{
        0, node.depth_+1, 
        mid_id_y0, mid_id_z1,
        scm::math::vec3f(min_vertex.x, mid_vertex.y, min_vertex.z),
        scm::math::vec3f(mid_vertex.x, max_vertex.y, mid_vertex.z)
      });
    children.push_back(
      auxiliary_node{
        0, node.depth_+1, 
        mid_id_y1, mid_id_z3,
        scm::math::vec3f(mid_vertex.x, mid_vertex.y, min_vertex.z),
        scm::math::vec3f(max_vertex.x, max_vertex.y, mid_vertex.z)
      });
    children.push_back(
      auxiliary_node{
        0, node.depth_+1, 
        mid_id_z0, mid_id_y0,
        scm::math::vec3f(min_vertex.x, min_vertex.y, mid_vertex.z),
        scm::math::vec3f(mid_vertex.x, mid_vertex.y, max_vertex.z)
      });
    children.push_back(
      auxiliary_node{
        0, node.depth_+1, 
        mid_id_z2, mid_id_y1,
        scm::math::vec3f(mid_vertex.x, min_vertex.y, mid_vertex.z),
        scm::math::vec3f(max_vertex.x, mid_vertex.y, max_vertex.z)
      });
    children.push_back(
      auxiliary_node{
        0, node.depth_+1, 
        mid_id_z1, mid_id_x,
        scm::math::vec3f(min_vertex.x, mid_vertex.y, mid_vertex.z),
        scm::math::vec3f(mid_vertex.x, max_vertex.y, max_vertex.z)
      });
    children.push_back(
      auxiliary_node{
        0, node.depth_+1, 
        mid_id_z3, node.end_,
        mid_vertex,
        max_vertex,
      });
    
    uint32_t child_mask = 0;
    uint32_t child_idx = 0;

    for (uint32_t i = 0; i < ranges.size(); ++i) {
      const auto& range = ranges[i];
      if (range.begin_ - range.end_ > 0) {
        bool found = false;
        uint32_t child_id = 0;
        for (uint32_t j = 0; j < children.size(); ++j) {
          auto& child = children[j];
          const auto sp = points[range.begin_];
          if (child.min_.x <= sp.pos_.x && child.max_.x >= sp.pos_.x
            && child.min_.y <= sp.pos_.y && child.max_.y >= sp.pos_.y
            && child.min_.z <= sp.pos_.z && child.max_.z >= sp.pos_.z) {
            found = true;
            child_id = j;
          }
        }
        
        if (found) {
          auto& child = children[child_id];
          child.begin_ = range.begin_;
          child.end_ = range.end_;
          child.idx_ = num_nodes;
          std::cout << "node " << num_nodes << " " << child.end_ - child.begin_ << " points " << std::endl;
          //std::cout << "node " << num_nodes << " min " << child.min_.x << " " << child.min_.y << " " << child.min_.z << std::endl;
          //std::cout << "node " << num_nodes << " max " << child.max_.x << " " << child.max_.y << " " << child.max_.z << std::endl;
          nodes_todo.push(child);
          child_mask |= (1 << i);
          child_idx = child_idx == 0 ? num_nodes : child_idx;
          ++num_nodes;
        }
        else {
            std::cout << "WARNINg! did not find range " << i << std::endl;
        }
      }
    }

    std::set<uint32_t> fotos;
    for (uint64_t i = node.begin_; i < node.end_; ++i) {
      const auto& sp = points[i];
      for (const auto& f : sp.features_) {
        fotos.insert(f.camera_id_);
      }
      if (node.min_.x <= sp.pos_.x && node.max_.x >= sp.pos_.x
        && node.min_.y <= sp.pos_.y && node.max_.y >= sp.pos_.y
        && node.min_.z <= sp.pos_.z && node.max_.z >= sp.pos_.z) {

      }
      else {
        std::cout << "e";
      }

    }

    node_idx_map[node.idx_] = octree_node(node.idx_, child_mask, child_idx, node.min_, node.max_, fotos);

  }

  std::cout << "octree complete " << "depth: " << depth_ << " num nodes: " << num_nodes << std::endl;

  for (uint64_t node_id = 0; node_id < num_nodes; ++node_id) {
    nodes_.push_back(node_idx_map[node_id]);
  }

}

uint64_t octree::
query(const scm::math::vec3f& _point) {

  if (nodes_.empty()) return 0;

  //locate the node that contains the point
  uint64_t current_node_id = 0;
  while ((nodes_[current_node_id].get_child_mask() & 0xff) > 0) {
    bool found = false;
    for (uint32_t i = 0; i < 8; ++i) {
      if ((nodes_[current_node_id].get_child_mask() & (1 << i) & 0xff) > 0 ) {
        uint64_t child_id = get_child_id(current_node_id, i);
        const auto& node = nodes_[child_id];
        if (node.get_min().x <= _point.x && node.get_max().x > _point.x 
          && node.get_min().y <= _point.y && node.get_max().y > _point.y 
          && node.get_min().z <= _point.z && node.get_max().z > _point.z) { 
          current_node_id = child_id;
          found = true;
          break;
        }
      }
    }
    if (!found) {
      //point is not contained in any children
      //so this is the deepest child that contains it
      //we're done.
      break;
    }
  }
  
  return current_node_id;

}

uint64_t octree::
get_num_nodes() const {
  return nodes_.size();
}


const octree_node& octree::
get_node(uint64_t _node_id) {
  return nodes_[_node_id];
}


void octree::
add_node(const octree_node& _node) {
  return nodes_.push_back(_node);
}


uint32_t octree::
get_depth() const {
  return depth_;
}

void octree::
set_depth(uint32_t _depth) {
  depth_ = _depth;
}


uint64_t octree::
get_child_id(uint64_t _node_id, uint32_t _child_index) {
  uint32_t child_id = nodes_[_node_id].get_child_idx();
  for (uint32_t i = 0; i < _child_index; ++i) {
    if (((nodes_[_node_id].get_child_mask() & (1 << i)) & 0xff) > 0) {
      ++child_id;
    }
  }
  return child_id;
      
}


uint64_t octree::
get_parent_id(uint64_t _child_id) {
  if (_child_id == 0 || _child_id > nodes_.size()) {
    return 0;
  }
  for (uint64_t node_id = 0; node_id < _child_id; ++node_id) {
    if ((nodes_[node_id].get_child_mask() & 0xff) > 0) {
      for (uint32_t i = 0; i < 8; ++i) {
        if (((nodes_[node_id].get_child_mask() & (1 << i)) & 0xff) > 0) {
          if (get_child_id(node_id, i) == _child_id) {
            return node_id;
          }
        }
      }
    }
  }

  return 0;

}

} } // namespace lamure

