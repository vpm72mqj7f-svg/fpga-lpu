//=============================================================================
// fp4_params.svh — shared parameters for fp4 datapath modules
//=============================================================================
`ifndef FP4_PARAMS_SVH
`define FP4_PARAMS_SVH

package fp4_params_pkg;
    parameter int FP4_GROUP_SIZE_DEFAULT = 16;
    parameter int FP4_SCALE_WIDTH        = 8;
    parameter int FP4_WEIGHT_WIDTH       = 4;
    parameter int FP8_ACTIV_WIDTH        = 8;
    parameter int FP4_ACCUM_WIDTH        = 32;
endpackage

`endif
