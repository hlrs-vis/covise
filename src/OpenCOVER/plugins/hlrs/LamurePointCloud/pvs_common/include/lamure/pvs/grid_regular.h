// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_PVS_REGULAR_GRID_H
#define LAMURE_PVS_REGULAR_GRID_H

#include <vector>
#include <string>
#include <mutex>

#include <lamure/pvs/pvs.h>
#include "lamure/pvs/grid.h"
#include "lamure/pvs/view_cell.h"
#include "lamure/pvs/view_cell_regular.h"

#include <scm/core/math.h>

namespace lamure
{
namespace pvs
{

class grid_regular : public grid
{
public:
	grid_regular();
	grid_regular(const size_t& number_cells, const double& cell_size, const scm::math::vec3d& position_center, const std::vector<node_t>& ids);
	~grid_regular();

	virtual std::string get_grid_type() const;
	static std::string get_grid_identifier();

	virtual size_t get_cell_count() const;
	virtual scm::math::vec3d get_size() const;
	virtual scm::math::vec3d get_position_center() const;

	virtual const view_cell* get_cell_at_index(const size_t& index) const;
	virtual const view_cell* get_cell_at_position(const scm::math::vec3d& position, size_t* cell_index) const;

	virtual void set_cell_visibility(const size_t& cell_index, const model_t& model_id, const node_t& node_id, const bool& visibility);
	virtual void set_cell_visibility(const scm::math::vec3d& position, const model_t& model_id, const node_t& node_id, const bool& visibility);

	virtual void save_grid_to_file(const std::string& file_path) const;
	virtual void save_visibility_to_file(const std::string& file_path) const;

	virtual bool load_grid_from_file(const std::string& file_path);
	virtual bool load_visibility_from_file(const std::string& file_path);

	virtual bool load_cell_visibility_from_file(const std::string& file_path, const size_t& cell_index);
	virtual void clear_cell_visibility(const size_t& cell_index);

	virtual model_t get_num_models() const;
	virtual node_t get_num_nodes(const model_t& model_id) const;

protected:
	void create_grid(const size_t& num_cells, const double& cell_size, const scm::math::vec3d& position_center);

	view_cell* calculate_cell_at_position(const scm::math::vec3d& position, size_t* cell_index) const;

	void save_regular_grid(const std::string& file_path, const std::string& grid_type) const;
	bool load_regular_grid(const std::string& file_path, const std::string& grid_type);

	double cell_size_;
	scm::math::vec3d size_;
	scm::math::vec3d position_center_;

	std::vector<view_cell_regular*> cells_;
	std::vector<node_t> ids_;

	mutable std::mutex mutex_;
};

}
}

#endif
