set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir ..]]
set build_dir [file join $repo_root build vivado]
set export_dir [file join $repo_root build overlay]
file mkdir $build_dir
file mkdir $export_dir

create_project vital_pynq_z2 $build_dir -part xc7z020clg400-1 -force
set_property board_part digilentinc.com:pynq-z2:part0:1.0 [current_project]
set_property target_language Verilog [current_project]

read_verilog -sv [file join $repo_root rtl streaming_attention_axis.sv]
update_compile_order -fileset sources_1

create_bd_design design_1
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
set_property -dict [list \
    CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_EN_RST0_PORT {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100.000000} \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1}] \
    [get_bd_cells processing_system7_0]
apply_bd_automation -rule xilinx.com:bd_rule:zynq \
    -config {apply_board_preset "1"} [get_bd_cells processing_system7_0]

create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_100M
set_property CONFIG.C_EXT_RESET_HIGH 0 [get_bd_cells rst_ps7_0_100M]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_length_width {23} \
    CONFIG.c_m_axis_mm2s_tdata_width {32} \
    CONFIG.c_s_axis_s2mm_tdata_width {32} \
    CONFIG.c_m_axi_mm2s_addr_width {32} \
    CONFIG.c_m_axi_s2mm_addr_width {32}] \
    [get_bd_cells axi_dma_0]

create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_lite_ic
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells axi_lite_ic]
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_memory_ic
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] [get_bd_cells axi_memory_ic]
create_bd_cell -type module -reference streaming_attention_axis streaming_attention_0

connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] \
    [get_bd_intf_pins axi_lite_ic/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_lite_ic/M00_AXI] \
    [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] \
    [get_bd_intf_pins axi_memory_ic/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] \
    [get_bd_intf_pins axi_memory_ic/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_memory_ic/M00_AXI] \
    [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] \
    [get_bd_intf_pins streaming_attention_0/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins streaming_attention_0/M_AXIS] \
    [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
    [get_bd_pins rst_ps7_0_100M/slowest_sync_clk] \
    [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] \
    [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] \
    [get_bd_pins axi_lite_ic/aclk] \
    [get_bd_pins axi_memory_ic/aclk] \
    [get_bd_pins axi_dma_0/s_axi_lite_aclk] \
    [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] \
    [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] \
    [get_bd_pins axi_dma_0/m_axis_mm2s_aclk] \
    [get_bd_pins axi_dma_0/s_axis_s2mm_aclk] \
    [get_bd_pins streaming_attention_0/s_axis_aclk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] \
    [get_bd_pins rst_ps7_0_100M/ext_reset_in]
connect_bd_net [get_bd_pins rst_ps7_0_100M/peripheral_aresetn] \
    [get_bd_pins axi_lite_ic/aresetn] \
    [get_bd_pins axi_memory_ic/aresetn] \
    [get_bd_pins axi_dma_0/axi_resetn] \
    [get_bd_pins streaming_attention_0/s_axis_aresetn]

assign_bd_address
validate_bd_design
save_bd_design

set wrapper [make_wrapper -files [get_files [file join $build_dir vital_pynq_z2.srcs sources_1 bd design_1 design_1.bd]] -top]
add_files -norecurse $wrapper
update_compile_order -fileset sources_1
set_property top design_1_wrapper [current_fileset]

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {![string match "*Complete!*" [get_property STATUS [get_runs impl_1]]]} {
    error "Vivado implementation/bitstream generation failed: [get_property STATUS [get_runs impl_1]]"
}

open_run impl_1
write_bitstream -force [file join $export_dir vital_pynq_z2.bit]
write_hwdef -force -file [file join $export_dir vital_pynq_z2.hwh]
puts "Build complete: $export_dir"
