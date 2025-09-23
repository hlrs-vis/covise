// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_PVS_IRREGULAR_GRID_H
#define LAMURE_PVS_IRREGULAR_GRID_H

#include <vector>
#include <string>
#include <mutex>

#include <lamure/pvs/pvs.h>
#include "lamure/pvs/grid.h"
#include "lamure/pvs/view_cell.h"
#include "lamure/pvs/view_cell_regular.h"
#include "lamure/pvs/view_cell_regular_managing.h"

#include <scm/core/math.h>

namespace lamure
{
namespace pvs
{

class grid_irregular : public grid
{
public:
	grid_irregular();
	grid_irregular(const size_t& number_cells_x, const size_t& number_cells_y, const size_t& number_cells_z, const double& cell_size, const scm::math::vec3d& position_center, const std::vector<node_t>& ids);
	~grid_irregular();

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

	void compute_index_access();

	size_t get_original_cell_count() const;
	bool join_cells(const size_t& index_one, const size_t& index_two, const float& error, const float& equality_threshold);

	const view_cell* get_original_cell_at_index(const size_t& index) const;
	const view_cell* get_original_cell_at_position(const scm::math::vec3d& position, size_t* cell_index) const;
	bool is_cell_at_index_original(const size_t& index) const;

protected:
	void create_grid(const size_t& number_cells_x, const size_t& number_cells_y, const size_t& number_cells_z, const double& cell_size, const scm::math::vec3d& position_center);

	view_cell* calculate_cell_at_position(const scm::math::vec3d& position, size_t* cell_index) const;
	
	void save_irregular_grid(const std::string& file_path, const std::string& grid_type) const;
	bool load_irregular_grid(const std::string& file_path, const std::string& grid_type);

	size_t get_original_index_of_cell(const view_cell* cell) const;
	size_t get_managing_index_of_cell(const view_cell* cell) const;

	double cell_size_;
	scm::math::vec3d size_;
	scm::math::vec3d position_center_;

	size_t number_cells_x_;
	size_t number_cells_y_;
	size_t number_cells_z_;

	// These view cells are the ones created from the original structure of regular grid cells.
	std::vector<view_cell_regular> original_cells_;

	// These cells are combinations of the original cells that are referred to if cells were joined.
	std::vector<view_cell_regular_managing> managing_cells_;

	// Used to allow access to view cells in a fast way without doing necessary calculations over and over again.
	std::vector<view_cell*> cells_by_indices_;

	// Each original (non-managing) view cell has an active state. If it is not active, it was joined and is represented by a managing view cell.
	std::vector<bool> cells_active_states_;

	// If an original cell is not active, it was joined and is part of a managing vew cell. The key is the original view cell index, the value the index of the proper managing view cell.
	std::map<size_t, size_t> original_index_to_cell_mapping_;

	std::vector<node_t> ids_;

	mutable std::mutex mutex_;
};

}
}

#endif
