project_open gemv_test
set_global_assignment -name FLOW_ENABLE_INCREMENTAL_COMPILATION ON
set_instance_assignment -name PARTITION top -to "|"
export_assignments
project_close
