// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/vt/ren/CutDatabase.h>
namespace vt
{
CutDatabase::CutDatabase(mem_slots_type *front, mem_slots_type *back) : DoubleBuffer<mem_slots_type>(front, back), _read_lock(), _write_lock(), _read_write_lock()
{
    VTConfig *config = &VTConfig::get_instance();

    _size_mem_x = config->get_phys_tex_tile_width();
    _size_mem_y = config->get_phys_tex_tile_width();
    _size_mem_interleaved = _size_mem_x * _size_mem_y * config->get_phys_tex_layers();

    for(size_t i = 0; i < _size_mem_interleaved; i++)
    {
        mem_slot_type mst;
        mst.position = i;
        _front->emplace_back(mst);
        _back->emplace_back(mst);
    }

    _cut_map = cut_map_type();
    _tile_provider = new ooc::TileProvider();
}
size_t CutDatabase::get_available_memory()
{
    size_t available_memory = _back->size();
    for(cut_map_entry_type cut_entry : _cut_map)
    {
        available_memory -= cut_entry.second->get_back()->get_mem_slots_locked().size();
    }

    return available_memory;
}
mem_slot_type *CutDatabase::get_free_mem_slot()
{
    for(auto mem_iter = _back->begin(); mem_iter != _back->end(); mem_iter++) // NOLINT
    {
        if(!(*mem_iter).locked)
        {
            return &(*mem_iter);
        }
    }

    throw std::runtime_error("out of mem slots");
}
mem_slot_type *CutDatabase::write_mem_slot_at(size_t position)
{
    std::unique_lock<std::mutex> lk(_write_lock);

    if(position >= _size_mem_interleaved)
    {
        throw std::runtime_error("Write request to interleaved memory position: " + std::to_string(position) + ", interleaved memory size is: " + std::to_string(_size_mem_interleaved));
    }

    if(!_is_written.load())
    {
        throw std::runtime_error("Unsanctioned write request to interleaved memory position: " + std::to_string(position));
    }

    return &get_back()->at(position);
}
mem_slot_type *CutDatabase::read_mem_slot_at(size_t position)
{
    // std::cout << "read_mem_slot_at" << std::endl;

    std::unique_lock<std::mutex> lk(_read_lock);

    if(position >= _size_mem_interleaved)
    {
        throw std::runtime_error("Read request to position: " + std::to_string(position) + ", interleaved memory size is: " + std::to_string(_size_mem_interleaved));
    }

    if(!_is_read.load())
    {
        throw std::runtime_error("Unsanctioned read request to interleaved memory position: " + std::to_string(position));
    }

    return &get_front()->at(position);
}
void CutDatabase::deliver() { _front->assign(_back->begin(), _back->end()); }
Cut *CutDatabase::start_writing_cut(uint64_t cut_id)
{
    //std::cout << "start_writing_cut" << std::endl;

    std::unique_lock<std::mutex> lk(_write_lock);

    if(_is_written.load())
    {
        throw std::runtime_error("Memory write access corruption");
    }
    _is_written.store(true);

    start_writing();
    Cut *requested_cut = _cut_map[cut_id];
    requested_cut->start_writing();

    return requested_cut;
}
void CutDatabase::stop_writing_cut(uint64_t cut_id)
{
    //std::cout << "stop_writing_cut" << std::endl;

    std::unique_lock<std::mutex> lk(_write_lock);

    _cut_map[cut_id]->stop_writing();
    stop_writing();

    if(!_is_written.load())
    {
        throw std::runtime_error("Memory write access corruption");
    }
    _is_written.store(false);

    _read_write_cv.notify_one();
}
Cut *CutDatabase::start_reading_cut(uint64_t cut_id)
{
    //std::cout << "start_reading_cut" << std::endl;

    std::unique_lock<std::mutex> lk(_read_lock);

    if(_is_read.load())
    {
        throw std::runtime_error("Memory read access corruption");
    }

    std::unique_lock<std::mutex> cv_lk(_read_write_lock);
    _read_write_cv.wait(cv_lk, [this] { return !_is_written.load(); });

    _is_read.store(true);

    // std::cout << "start_reading_cut: is being read" << std::endl;

    start_reading();
    Cut *requested_cut = _cut_map[cut_id];
    requested_cut->start_reading();

    return requested_cut;
}
void CutDatabase::stop_reading_cut(uint64_t cut_id)
{
    //std::cout << "stop_reading_cut" << std::endl;

    std::unique_lock<std::mutex> lk(_read_lock);

    stop_reading();
    _cut_map[cut_id]->stop_reading();

    if(!_is_read.load())
    {
        throw std::runtime_error("Memory read access corruption");
    }
    _is_read.store(false);

    // std::cout << "stop_reading_cut: is not being read any longer" << std::endl;
}
cut_map_type *CutDatabase::get_cut_map() { return &_cut_map; }
uint32_t CutDatabase::register_dataset(const std::string &file_name)
{
    for(dataset_map_entry_type dataset : _dataset_map)
    {
        if(dataset.second == file_name)
        {
            return dataset.first;
        }
    }

    uint32_t id = (uint32_t)_dataset_map.size();
    _dataset_map.insert(dataset_map_entry_type(id, file_name));

    return id;
}
uint16_t CutDatabase::register_view()
{
    uint16_t id = (uint16_t)_view_set.size();
    _view_set.emplace(id);
    return id;
}
uint16_t CutDatabase::register_context()
{
    uint16_t id = (uint16_t)_context_set.size();
    _context_set.emplace(id);
    return id;
}
uint64_t CutDatabase::register_cut(uint32_t dataset_id, uint16_t view_id, uint16_t context_id)
{
    if(_dataset_map.find(dataset_id) == _dataset_map.end())
    {
        throw std::runtime_error("Requested dataset id not registered");
    }

    if(_view_set.find(view_id) == _view_set.end())
    {
        throw std::runtime_error("Requested view id not registered");
    }

    if(_context_set.find(context_id) == _context_set.end())
    {
        throw std::runtime_error("Requested context id not registered");
    }

    Cut *cut = &Cut::init_cut(_tile_provider->addResource(_dataset_map[dataset_id].c_str()));

    uint64_t id = ((uint64_t)dataset_id) << 32 | ((uint64_t)view_id << 16) | ((uint64_t)context_id);

    _cut_map.insert(cut_map_entry_type(id, cut));

    return id;
}

ooc::TileProvider *CutDatabase::get_tile_provider() const { return _tile_provider; }
void CutDatabase::start_writing()
{
    DoubleBuffer::start_writing();
}
void CutDatabase::stop_writing()
{
    DoubleBuffer::stop_writing();
}
void CutDatabase::start_reading()
{
    DoubleBuffer::start_reading();
}
void CutDatabase::stop_reading()
{
    DoubleBuffer::stop_reading();
}
}