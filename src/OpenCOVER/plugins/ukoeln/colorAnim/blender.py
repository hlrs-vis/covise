import bpy
import os

# Configuration
input_dir = "/home/daniel/data/Horn/surfaces/dynamic/cortex/"  # Directory with 275 low-res models
output_dir = "/home/daniel/data/Horn/surfaces/dynamic/cortex_high_res/"  # Where to save high-res versions
base_mesh_name = "HighResBrain"  # Name of your high-res base mesh in the scene

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get list of all model files
model_files = [f for f in os.listdir(input_dir) if f.endswith(('.ply', '.obj', '.fbx'))]

print(f"Found {len(model_files)} models to process")

for i, model_file in enumerate(model_files):
    print(f"Processing {i+1}/{len(model_files)}: {model_file}")
    
    # Clear scene except base mesh
    for obj in bpy.data.objects:
        if obj.name != base_mesh_name:
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Import low-res model with vertex colors
    file_path = os.path.join(input_dir, model_file)
    
    # Import PLY
    if model_file.endswith('.ply'):
        try:
            bpy.ops.wm.ply_import(filepath=file_path)
        except Exception as e:
            print(f"Error importing {model_file}: {e}")
            continue
    
    # Get the imported model
    lowres = None
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH' and obj.name != base_mesh_name:
            lowres = obj
            break
    
    if not lowres:
        print(f"  Error: Could not find imported mesh for {model_file}")
        continue
    
    lowres.name = "LowResSource"
    
    # Duplicate the high-res base
    bpy.ops.object.select_all(action='DESELECT')
    base_mesh = bpy.data.objects[base_mesh_name]
    base_mesh.select_set(True)
    bpy.context.view_layer.objects.active = base_mesh
    bpy.ops.object.duplicate()
    highres_copy = bpy.context.active_object
    highres_copy.name = f"HighRes_{i}"
    
    # Ensure both meshes have vertex color attributes
    if not highres_copy.data.color_attributes:
        highres_copy.data.color_attributes.new(
            name="Col",
            type='BYTE_COLOR',
            domain='CORNER'
        )
    
    # Transfer vertex colors using Data Transfer modifier
    bpy.ops.object.select_all(action='DESELECT')
    highres_copy.select_set(True)
    bpy.context.view_layer.objects.active = highres_copy
    
    bpy.ops.object.modifier_add(type='DATA_TRANSFER')
    mod = highres_copy.modifiers["DataTransfer"]
    mod.object = lowres
    mod.use_vert_data = True
    mod.data_types_verts = {'COLOR_VERTEX'}
    mod.vert_mapping = 'NEAREST'
    mod.use_loop_data = True
    mod.data_types_loops = {'COLOR_CORNER'}
    mod.loop_mapping = 'NEAREST_POLYNOR'
    
    # Apply modifier
    bpy.ops.object.modifier_apply(modifier=mod.name)
    
    # DELETE the low-res mesh BEFORE export
    bpy.data.objects.remove(lowres, do_unlink=True)
    
    # Make absolutely sure ONLY highres_copy is selected
    bpy.ops.object.select_all(action='DESELECT')
    highres_copy.select_set(True)
    bpy.context.view_layer.objects.active = highres_copy
    
    # Verify selection
    selected_count = len([obj for obj in bpy.data.objects if obj.select_get()])
    print(f"  Selected objects for export: {selected_count} (should be 1)")
    
    # Export the high-res model with CORRECT parameters
    output_name = os.path.splitext(model_file)[0]
    output_path = os.path.join(output_dir, f"highres_{output_name}.ply")
    
    if model_file.endswith('.ply'):
        try:
            bpy.ops.wm.ply_export(
                filepath=output_path,
                check_existing=False,
                export_selected_objects=True,  # This is the correct parameter!
                export_colors='SRGB',  # Export with colors
                export_normals=True,   # Include normals
                export_uv=True,        # Include UVs if available
                apply_modifiers=True,  # Apply any remaining modifiers
                global_scale=1.0,
                forward_axis='Y',
                up_axis='Z'
            )
            print(f"  ✓ Exported: {output_path}")
            
            # Verify file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"    File size: {file_size / 1024:.2f} KB")
            else:
                print(f"    WARNING: File not found after export!")
                
        except Exception as e:
            print(f"  Error exporting {model_file}: {e}")
    
    # Clean up the high-res copy
    bpy.data.objects.remove(highres_copy, do_unlink=True)
    
    print(f"  ✓ Completed: {model_file}")

print(f"\nAll {len(model_files)} models processed!")
print(f"Output directory: {output_dir}")

# List the exported files
exported_files = [f for f in os.listdir(output_dir) if f.endswith('.ply')]
print(f"Successfully exported {len(exported_files)} files")