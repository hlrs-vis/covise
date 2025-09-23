// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_PVS_GRID_H
#define LAMURE_PVS_GRID_H

#include <string>

#include <lamure/pvs/pvs.h>
#include "lamure/pvs/view_cell.h"
#include "lamure/types.h"

namespace lamure
{
namespace pvs
{

class grid
{
public:
  virtual ~grid() = default;

	virtual std::string get_grid_type() const = 0;

	virtual size_t get_cell_count() const = 0;
	virtual scm::math::vec3d get_size() const = 0;
	virtual scm::math::vec3d get_position_center() const = 0;

	virtual const view_cell* get_cell_at_index(const size_t& index) const = 0;
	virtual const view_cell* get_cell_at_position(const scm::math::vec3d& position, size_t* cell_index) const = 0;

	virtual void set_cell_visibility(const size_t& cell_index, const model_t& model_id, const node_t& node_id, const bool& visibility) = 0;
	virtual void set_cell_visibility(const scm::math::vec3d& position, const model_t& model_id, const node_t& node_id, const bool& visibility) = 0;

	virtual void save_grid_to_file(const std::string& file_path) const = 0;
	virtual void save_visibility_to_file(const std::string& file_path) const = 0;

	virtual bool load_grid_from_file(const std::string& file_path) = 0;
	virtual bool load_visibility_from_file(const std::string& file_path) = 0;

	virtual bool load_cell_visibility_from_file(const std::string& file_path, const size_t& cell_index) = 0;
	virtual void clear_cell_visibility(const size_t& cell_index) = 0;

	virtual model_t get_num_models() const = 0;
	virtual node_t get_num_nodes(const model_t& model_id) const = 0;
};

}
}

#endif
