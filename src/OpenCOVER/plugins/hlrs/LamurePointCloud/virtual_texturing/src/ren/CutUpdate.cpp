// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/vt/ren/CutDatabase.h>
#include <lamure/vt/ren/CutUpdate.h>

namespace vt
{
CutUpdate::CutUpdate() : _dispatch_lock(), _dispatch_time()
{
    _freeze_dispatch.store(false);

    _should_stop.store(false);
    _new_feedback.store(false);

    _config = &VTConfig::get_instance();
    _cut_db = &CutDatabase::get_instance();

    _feedback_lod_buffer = new int32_t[_cut_db->get_size_mem_interleaved()];
    _feedback_count_buffer = new uint32_t[_cut_db->get_size_mem_interleaved()];
}

CutUpdate::~CutUpdate() {}

void CutUpdate::start()
{
    _cut_db->get_tile_provider()->start((size_t)VTConfig::get_instance().get_size_ram_cache() * 1024 * 1024);
    _worker = std::thread(&CutUpdate::run, this);
}

void CutUpdate::run()
{
    while(!_should_stop.load())
    {
        std::unique_lock<std::mutex> lk(_dispatch_lock, std::defer_lock);
        if(_new_feedback.load())
        {
            dispatch();
        }
        _new_feedback.store(false);

        while(!_cv.wait_for(lk, std::chrono::milliseconds(100), [this] { return _new_feedback.load(); }))
        {
            if(_should_stop.load())
            {
                break;
            }
        }
    }
}

void CutUpdate::dispatch()
{
    if(_freeze_dispatch.load())
    {
        for(cut_map_entry_type cut_entry : (*_cut_db->get_cut_map()))
        {
            Cut *cut = _cut_db->start_writing_cut(cut_entry.first);

            cut->get_back()->get_mem_slots_updated().clear();

            _cut_db->stop_writing_cut(cut_entry.first);
        }
        return;
    }

    // std::cout << "\ndispatch() BEGIN" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    uint32_t texels_per_tile = _config->get_size_tile() * _config->get_size_tile();
    uint32_t throughput = _config->get_size_physical_update_throughput();
    uint32_t split_budget_throughput = throughput * 1024 * 1024 / texels_per_tile / _config->get_byte_stride() / 4;
    uint32_t split_budget_available = (uint32_t)_cut_db->get_available_memory() / 4;
    uint32_t split_budget = std::min(split_budget_throughput, split_budget_available);

    uint8_t split_counter = 0;

    // std::cout << "split budget: " << split_budget << std::endl;

    for(cut_map_entry_type cut_entry : (*_cut_db->get_cut_map()))
    {
        id_set_type collapse_to;
        id_set_type split;
        id_set_type keep;

        Cut *cut = _cut_db->start_writing_cut(cut_entry.first);

        if (cut->get_atlas()->getDepth() < 1) {
            std::cout << "tree is too flat" << std::endl;
            exit(1);
        }

        // std::cout << std::endl;
        // std::cout << "writing cut: " << cut_entry.first << std::endl;
        // std::cout << std::endl;

        if(!cut->is_drawn())
        {
            _cut_db->get_tile_provider()->getTile(cut->get_atlas(), 0, 100);

            if(!_cut_db->get_tile_provider()->wait(std::chrono::milliseconds(1000)))
            {
                throw std::runtime_error("Root tile not loaded for atlas: " + std::string(cut->get_atlas()->getFileName()));
            }

            ooc::TileCacheSlot *slot = _cut_db->get_tile_provider()->getTile(cut->get_atlas(), 0, 100);

            if(slot == nullptr)
            {
                throw std::runtime_error("Root tile is nullptr for atlas: " + std::string(cut->get_atlas()->getFileName()));
            }

            uint8_t *root_tile = slot->getBuffer();

            cut->get_back()->get_cut().insert(0);

            add_to_indexed_memory(cut, 0, root_tile);

            cut->set_drawn(true);

            _cut_db->stop_writing_cut(cut_entry.first);

            continue;
        }

        cut_type cut_desired;

        /* DECISION MAKING PASS */

        auto iter = cut->get_back()->get_cut().cbegin();

        while(iter != cut->get_back()->get_cut().cend())
        {
            uint16_t tile_depth = QuadTree::get_depth_of_node(*iter);
            uint16_t max_depth = cut->get_atlas()->getDepth() - 1;


            if(check_all_siblings_in_cut(*iter, cut->get_back()->get_cut()))
            {

                bool allow_collapse = true;

                if ((*iter % 4) == 1) {

                  id_type parent_id = QuadTree::get_parent_id(*iter);
        
                  for (int32_t c = 0; c < 4; ++c) {
                    //id_type sibling_id = *(iter+c);
                    id_type sibling_id = QuadTree::get_child_id(parent_id, c);
                    if (sibling_id == 0) {
                      allow_collapse = false;
                      break;
                    }

                    //else check fb of all siblings < current_level
                    mem_slot_type *sibling_mem_slot = write_mem_slot_for_id(cut, sibling_id);
                    if (_feedback_lod_buffer[sibling_mem_slot->position] >= tile_depth) {
                      allow_collapse = false;
                      break;
                    }
                  }

                  if (allow_collapse) {
                    //collapse 1, skip others
                    collapse_to.insert(parent_id);
                    std::advance(iter, 4);
                    continue;
                  }
                
                }

                mem_slot_type *mem_slot = write_mem_slot_for_id(cut, *iter);
                if(_feedback_lod_buffer[mem_slot->position] > tile_depth && tile_depth < max_depth && split_counter < split_budget)
                {
                  split.insert(*iter);
                  split_counter++;
                }
                else
                {
                  keep.insert(*iter);
                }

                iter++;


            }
            else
            {
                id_type tile_id = *iter;
                mem_slot_type *mem_slot = write_mem_slot_for_id(cut, tile_id);
                if(mem_slot == nullptr)
                {
                    throw std::runtime_error("Node " + std::to_string(tile_id) + " not found in memory slots");
                }

                if(_feedback_lod_buffer[mem_slot->position] > tile_depth && tile_depth < max_depth && split_counter < split_budget)
                {
                    split.insert(tile_id);
                    split_counter++;
                }
                else
                {
                    keep.insert(tile_id);
                }

                iter++;
            }

            // std::cout << "Iter state: " + std::to_string(*iter) << std::endl;
        }

        /* MEMORY INDEXING PASS */

        cut->get_back()->get_mem_slots_updated().clear();
        cut->get_back()->get_mem_slots_cleared().clear();

        for(id_type tile_id : collapse_to)
        {
            // std::cout << "action: collapse to " << tile_id << std::endl;
            if(!collapse_to_id(cut, tile_id))
            {
                for(uint8_t i = 0; i < 4; i++)
                {
                    id_type child_id = QuadTree::get_child_id(tile_id, i);

                    keep.insert(child_id);
                }
            }
            else
            {
                cut_desired.insert(tile_id);
            }
        }

        for(id_type tile_id : split)
        {
            // std::cout << "action: split " << tile_id << std::endl;
            if(!split_id(cut, tile_id))
            {
                keep.insert(tile_id);
            }
            else
            {
                for(uint8_t i = 0; i < 4; i++)
                {
                    cut_desired.insert(QuadTree::get_child_id(tile_id, i));
                }
            }
        }

        for(id_type tile_id : keep)
        {
            // std::cout << "action: keep " << tile_id << std::endl;
            if(keep_id(cut, tile_id))
            {
                cut_desired.insert(tile_id);
            }
            else
            {
                throw std::runtime_error("Node " + std::to_string(tile_id) + " could not be kept in working set");
            }
        }

        cut->get_back()->get_cut().swap(cut_desired);

        _cut_db->stop_writing_cut(cut_entry.first);
    }

    auto end = std::chrono::high_resolution_clock::now();
    _dispatch_time = std::chrono::duration<float, std::milli>(end - start).count();

    // std::cout << "dispatch() END" << std::endl;
}

bool CutUpdate::add_to_indexed_memory(Cut *cut, id_type tile_id, uint8_t *tile_ptr)
{
    mem_slot_type *mem_slot = write_mem_slot_for_id(cut, tile_id);

    if(mem_slot == nullptr)
    {
        mem_slot_type *free_mem_slot = _cut_db->get_free_mem_slot();

        if(free_mem_slot == nullptr)
        {
            throw std::runtime_error("out of mem slots");
        }

        free_mem_slot->tile_id = tile_id;
        free_mem_slot->pointer = tile_ptr;

        mem_slot = free_mem_slot;
    }

    mem_slot->locked = true;
    mem_slot->updated = false;

    cut->get_back()->get_mem_slots_locked()[tile_id] = mem_slot->position;

    uint_fast32_t x_orig, y_orig;
    QuadTree::get_pos_by_id(tile_id, x_orig, y_orig);
    uint16_t tile_depth = QuadTree::get_depth_of_node(tile_id);

    size_t phys_tex_tile_width = _config->get_phys_tex_tile_width();
    size_t tiles_per_tex = phys_tex_tile_width * phys_tex_tile_width;

    uint8_t *ptr = &cut->get_back()->get_index(tile_depth)[(y_orig * QuadTree::get_tiles_per_row(tile_depth) + x_orig) * 4];

    size_t level = mem_slot->position / tiles_per_tex;
    size_t rel_pos = mem_slot->position - level * tiles_per_tex;
    size_t x_tile = rel_pos % phys_tex_tile_width;
    size_t y_tile = rel_pos / phys_tex_tile_width;

    if(ptr[0] != (uint8_t)x_tile || ptr[1] != (uint8_t)y_tile || ptr[2] != (uint8_t)level || ptr[3] != (uint8_t)1)
    {
        mem_slot->updated = true;
        ptr[0] = (uint8_t)x_tile;
        ptr[1] = (uint8_t)y_tile;
        ptr[2] = (uint8_t)level;
        ptr[3] = (uint8_t)1;
    }

    if(mem_slot->updated)
    {
        cut->get_back()->get_mem_slots_updated()[tile_id] = mem_slot->position;
    }

    return true;
}

bool CutUpdate::collapse_to_id(Cut *cut, id_type tile_id)
{
    ooc::TileCacheSlot *slot = _cut_db->get_tile_provider()->getTile(cut->get_atlas(), tile_id, 100);

    if(slot == nullptr)
    {
        return false;
    }

    uint8_t *tile_ptr = slot->getBuffer();

    for(uint8_t i = 0; i < 4; i++)
    {
        remove_from_indexed_memory(cut, QuadTree::get_child_id(tile_id, i));
    }

    return add_to_indexed_memory(cut, tile_id, tile_ptr);
}
bool CutUpdate::split_id(Cut *cut, id_type tile_id)
{
    bool all_children_available = true;

    for(size_t n = 0; n < 4; n++)
    {
        id_type child_id = QuadTree::get_child_id(tile_id, n);

        if(_cut_db->get_tile_provider()->getTile(cut->get_atlas(), child_id, 100) == nullptr)
        {
            all_children_available = false;
            continue;
        }
    }

    bool all_children_added = true;

    if(all_children_available)
    {
        for(size_t n = 0; n < 4; n++)
        {
            id_type child_id = QuadTree::get_child_id(tile_id, n);
            uint8_t *tile_ptr = _cut_db->get_tile_provider()->getTile(cut->get_atlas(), child_id, 100)->getBuffer();

            if(tile_ptr == nullptr)
            {
                throw std::runtime_error("Child removed from RAM");
            }

            all_children_added = all_children_added && add_to_indexed_memory(cut, child_id, tile_ptr);
        }
    }

    return all_children_available && all_children_added;
}
bool CutUpdate::keep_id(Cut *cut, id_type tile_id)
{
    ooc::TileCacheSlot *slot = _cut_db->get_tile_provider()->getTile(cut->get_atlas(), tile_id, 100);

    if(slot == nullptr)
    {
        throw std::runtime_error("kept tile #" + std::to_string(tile_id) + "not found in RAM");
    }

    uint8_t *tile_ptr = slot->getBuffer();

    if(tile_ptr == nullptr)
    {
        throw std::runtime_error("kept tile #" + std::to_string(tile_id) + "not found in RAM");
    }

    return add_to_indexed_memory(cut, tile_id, tile_ptr);
}
mem_slot_type *CutUpdate::write_mem_slot_for_id(Cut *cut, id_type tile_id)
{
    auto mem_slot_iter = cut->get_back()->get_mem_slots_locked().find(tile_id);

    if(mem_slot_iter == cut->get_back()->get_mem_slots_locked().end())
    {
        return nullptr;
    }

    return _cut_db->write_mem_slot_at((*mem_slot_iter).second);
}
void CutUpdate::feedback(int32_t *buf_lod, uint32_t *buf_count)
{
    if(_dispatch_lock.try_lock())
    {
        std::copy(buf_lod, buf_lod + _cut_db->get_size_mem_interleaved(), _feedback_lod_buffer);
        std::copy(buf_count, buf_count + _cut_db->get_size_mem_interleaved(), _feedback_count_buffer);
        _new_feedback.store(true);
        _dispatch_lock.unlock();
        _cv.notify_one();
    }
}

void CutUpdate::stop()
{
    _cut_db->get_tile_provider()->stop();
    _should_stop.store(true);
    _new_feedback.store(true);
    _cv.notify_one();
    _worker.join();
}
bool CutUpdate::check_all_siblings_in_cut(id_type tile_id, const cut_type &cut)
{
    bool all_in_cut = true;
    id_type parent_id = QuadTree::get_parent_id(tile_id);
    for(uint8_t i = 0; i < 4; i++)
    {
        id_type child_id = QuadTree::get_child_id(parent_id, i);
        all_in_cut = all_in_cut && cut.find(child_id) != cut.end();
        if(!all_in_cut)
        {
            return false;
        }
    }
    return all_in_cut;
}
const float &CutUpdate::get_dispatch_time() const { return _dispatch_time; }
void CutUpdate::toggle_freeze_dispatch() { _freeze_dispatch.store(!_freeze_dispatch.load()); }
void CutUpdate::remove_from_indexed_memory(Cut *cut, id_type tile_id)
{
    // std::cout << "Tile removal requested: " << std::to_string(tile_id) << std::endl;

    mem_slot_type *mem_slot = write_mem_slot_for_id(cut, tile_id);

    if(mem_slot == nullptr)
    {
        throw std::runtime_error("Collapsed child not in memory slots");
    }

    uint_fast32_t x_orig, y_orig;
    QuadTree::get_pos_by_id(tile_id, x_orig, y_orig);
    uint16_t tile_depth = QuadTree::get_depth_of_node(tile_id);

    uint8_t *ptr = &cut->get_back()->get_index(tile_depth)[(y_orig * QuadTree::get_tiles_per_row(tile_depth) + x_orig) * 4];

    ptr[0] = (uint8_t)0;
    ptr[1] = (uint8_t)0;
    ptr[2] = (uint8_t)0;

    ptr[3] = (uint8_t)0;

    cut->get_back()->get_mem_slots_locked().erase(mem_slot->tile_id);

    _cut_db->get_tile_provider()->ungetTile(cut->get_atlas(), mem_slot->tile_id);

    mem_slot->locked = false;
    mem_slot->updated = false;
    mem_slot->tile_id = UINT64_MAX;
    mem_slot->pointer = nullptr;

    cut->get_back()->get_mem_slots_cleared()[tile_id] = mem_slot->position;
}
}
