all: field_to_lines/linux/field_to_lines FLE/linux/ucd_FLE flow_topo/linux/flow_topo localized_flow/linux/ucd_localized_flow Mz/linux/ucd_Mz read_unstructured/linux/read_unstructured ridge_surface/linux/ucd_ridge_surface statistics/linux/ucd_statistics vortex_cores/linux/ucd_vortex_cores vortex_criteria/linux/ucd_vortex_criteria write_unstructured/linux/write_unstructured


field_to_lines/linux/field_to_lines:
	(cd ./field_to_lines; make)

FLE/linux/ucd_FLE:
	(cd ./FLE; make)

flow_topo/linux/flow_topo:
	(cd ./flow_topo; make)

localized_flow/linux/ucd_localized_flow:
	(cd ./localized_flow; make)

Mz/linux/ucd_Mz:
	(cd ./Mz; make)

read_unstructured/linux/read_unstructured:
	(cd ./read_unstructured; make)

ridge_surface/linux/ucd_ridge_surface:
	(cd ./ridge_surface; make)

statistics/linux/ucd_statistics:
	(cd ./statistics; make)

vortex_cores/linux/ucd_vortex_cores:
	(cd ./vortex_cores; make)

vortex_criteria/linux/ucd_vortex_criteria:
	(cd ./vortex_criteria; make)

write_unstructured/linux/write_unstructured:
	(cd ./write_unstructured; make)

clean:
	(cd ./field_to_lines; make clean;)
	(cd ./FLE; make clean;)
	(cd ./flow_topo; make clean;)
	(cd ./Mz; make clean;)
	(cd ./localized_flow; make clean;)
	(cd ./read_unstructured; make clean;)
	(cd ./ridge_surface; make clean;)
	(cd ./statistics; make clean;)
	(cd ./vortex_cores; make clean;)
	(cd ./vortex_criteria; make clean;)
	(cd ./write_unstructured; make clean;)
