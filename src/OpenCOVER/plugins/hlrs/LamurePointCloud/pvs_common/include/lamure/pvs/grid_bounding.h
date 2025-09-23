// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_PVS_GRID_BOUNDING_H
#define LAMURE_PVS_GRID_BOUNDING_H

#include <vector>
#include <string>
#include <mutex>

#include <lamure/pvs/pvs.h>
#include "lamure/pvs/grid.h"
#include "lamure/pvs/grid_regular_compressed.h"
#include "lamure/pvs/view_cell.h"
#include "lamure/pvs/view_cell_regular.h"

#include <scm/core/math.h>

namespace lamure
{
namespace pvs
{

class grid_bounding : public grid_regular_compressed
{
public:
	grid_bounding();
	grid_bounding(const grid* core_grid, const std::vector<node_t>& ids);
	~grid_bounding();

	virtual std::string get_grid_type() const;
	static std::string get_grid_identifier();

	virtual const view_cell* get_cell_at_position(const scm::math::vec3d& position, size_t* cell_index) const;

	virtual void save_grid_to_file(const std::string& file_path) const;
	virtual bool load_grid_from_file(const std::string& file_path);

protected:
	void create_grid(const grid* core_grid);
	void create_grid(const scm::math::vec3d& center_cell_size, const scm::math::vec3d& position_center);

	void save_bounding_grid(const std::string& file_path, const std::string& grid_type) const;
	bool load_bounding_grid(const std::string& file_path, const std::string& grid_type);
};

}
}

#endif
