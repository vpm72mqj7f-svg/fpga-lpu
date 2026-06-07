
`timescale 1 ps / 1 ps
module  mxfpga_golden_top (
        //Reset&Clock
        input           CPU_RESETN          , // Reset, low active
        input           CLK_SYS_50M_P       , // System clock, 50MHz
        input           CLK_SYS_100M_P      , // System clock, 100MHz

        input           CLK_UIB0_P          , //UIB PLL REFCLK for BOTTOM HBM
        input           CLK_UIB1_P          , //UIB PLL REFCLK for TOP HBM 

        input           CLK_ESRAM0_P        ,
        input           CLK_ESRAM1_P        ,


        //--------user_led_dip_pb----------
        output  [3:0]   S10_LED             , // User LEDs

        //EMIF interface: DDR4 Component
        input  wire        CLK_DDR4_COMP_P         ,         //                       ddr_ref_clk.clk
		    output wire        DDR4_COMP_CLK_P         ,         //                    emif_s10_0_mem.mem_ck
		    output wire        DDR4_COMP_CLK_N         ,         //                                  .mem_ck_n
		    output wire [16:0] DDR4_COMP_A             ,         //                                  .mem_a
		    output wire        DDR4_COMP_ACT_N         ,         //                                  .mem_act_n
		    output wire [1:0]  DDR4_COMP_BA            ,         //                                  .mem_ba
		    output wire        DDR4_COMP_BG0           ,         //                                  .mem_bg
		    output wire        DDR4_COMP_BG1           ,         //                                  .mem_bg
		    output wire        DDR4_COMP_CKE           ,         //                                  .mem_cke
		    output wire        DDR4_COMP_CS_N          ,         //                                  .mem_cs_n
		    output wire        DDR4_COMP_ODT           ,         //                                  .mem_odt
		    output wire        DDR4_COMP_RESET_N       ,         //                                  .mem_reset_n
		    output wire        DDR4_COMP_PAR           ,         //                                  .mem_par
		    input  wire        DDR4_COMP_ALERT_N       ,         //                                  .mem_alert_n
		    inout  wire [8:0]  DDR4_COMP_DQS_P         ,         //                                  .mem_dqs
		    inout  wire [8:0]  DDR4_COMP_DQS_N         ,         //                                  .mem_dqs_n
		    inout  wire [71:0] DDR4_COMP_DQ            ,         //                                  .mem_dq
		    inout  wire [8:0]  DDR4_COMP_DBI_N         ,         //                                  .mem_dbi_n
		    input  wire        DDR4_COMP_RZQ           ,         //                    emif_s10_0_oct.oct_rzqin
		    
		    
        //EMIF interface: HILO DDR4
        input  wire         CLK_HILO_MEM_P         ,
        input  wire         HILO_MEM_RZQ           ,
        inout  wire [3:0]   MEM_DQSB_P             ,  
        inout  wire [3:0]   MEM_DQSB_N             ,   
        inout  wire [1:0]   MEM_QKB_P              ,
        inout  wire [3:0]   MEM_DMB                ,
        inout  wire [31:0]  MEM_DQB                ,
        inout  wire [1:0]   MEM_QKA_P              ,
        inout  wire [3:0]   MEM_DQSA_P             , 
        inout  wire [3:0]   MEM_DQSA_N             ,
        inout  wire [3:0]   MEM_DMA                ,
        inout  wire [31:0]  MEM_DQA                ,
        inout  wire         MEM_DQS_ADDR_CMD_P     ,
        inout  wire         MEM_DQS_ADDR_CMD_N     ,
        inout  wire [8:0]   MEM_DQ_ADDR_CMD        ,
        output wire [28:0]  MEM_ADDR_CMD           ,
        input  wire         MEM_ADDR_CMD29         ,            //Alertn
        output wire         MEM_ADDR_CMD30         ,            //CSn
        output wire         MEM_ADDR_CMD31         ,            //PAR   
        output wire         MEM_CLK_P              ,
        output wire         MEM_CLK_N              ,
		    
		    
        //EMIF interface: DIMM interface
        input  wire        CLK_DDR4_DIMM_P    ,                 //                       ddr_ref_clk.clk
        output wire [1:0]  DDR4_DIMM_CK_P     ,                 //                    emif_s10_0_mem.mem_ck
        output wire [1:0]  DDR4_DIMM_CK_N     ,                 //                                  .mem_ck_n
        output wire [17:0] DDR4_DIMM_A        ,                 //                                  .mem_a
        output wire        DDR4_DIMM_ACT_N    ,                 //                                  .mem_act_n
        output wire [1:0]  DDR4_DIMM_BA       ,                 //                                  .mem_ba
        output wire [1:0]  DDR4_DIMM_BG       ,                 //                                  .mem_bg
        output wire [1:0]  DDR4_DIMM_CKE      ,                 //                                  .mem_cke
        output wire [3:0]  DDR4_DIMM_CS_N     ,                 //                                  .mem_cs_n
        output wire [1:0]  DDR4_DIMM_ODT      ,                 //                                  .mem_odt
        output wire        DDR4_DIMM_RESET_N  ,                 //                                  .mem_reset_n
        output wire        DDR4_DIMM_PAR      ,                 //                                  .mem_par
        input  wire        DDR4_DIMM_ALERT_N  ,                 //                                  .mem_alert_n
        inout  wire [8:0]  DDR4_DIMM_DQS_P    ,                 //                                  .mem_dqs
        inout  wire [8:0]  DDR4_DIMM_DQS_N    ,                 //                                  .mem_dqs_n
        inout  wire [8:0]  DDR4_DIMM_DBI_N    ,                 //                                  .mem_dqs
        inout  wire [17:9] DDR4_DIMM_TDQS_N   ,                 //                                  .mem_dqs_n
        inout  wire [71:0] DDR4_DIMM_DQ       ,                //                                  .mem_dq
        input  wire        DDR4_DIMM_RZQ      ,                //                   emif_s10_0_oct.oct_rzqin       
		    
		    
		    //PCIe End Port interface
       input           pcie_ep_i2c_scl       ,
       inout           pcie_ep_i2c_sda       ,
       input           refclk_pcie_ep_p      ,
       input           refclk_pcie_ep_edge_p ,
       input           refclk_pcie_ep1_p     ,
       output [15:0]   pcie_ep_rx_p          ,
       input  [15:0]   pcie_ep_tx_p          ,
       input           s10_pcie_perstn0      ,
       input           s10_pcie_perstn1      ,
       input           pcie_ep_waken         ,
		    

        //PCIe Root Port interface
        input           refclk_pcie_rt_p      ,
        input  [15:0]   pcie_rt_rx_p          ,
        output [15:0]   pcie_rt_tx_p          ,
        output          pcie_rt_s10_perstn    ,
        output          pcie_rt_s10_prsnt2n   ,
        output          pcie_rt_s10_waken     ,

       //QSFP:ref clk
        input           refclk_zqsfp0_p       ,
        input           refclk_zqsfp1_p       ,
     
        input  [ 3:0]   zqsfp0_rx_p           ,
        output [ 3:0]   zqsfp0_tx_p           ,
    
        input  [ 3:0]   zqsfp1_rx_p           ,
        output [ 3:0]   zqsfp1_tx_p           ,
    
        output zqsfp0_1v8_modsel_l            ,
        output zqsfp0_1v8_reset_l             ,
        input  zqsfp0_1v8_modprs_l            ,
        output zqsfp0_1v8_lpmode              ,
        input  zqsfp0_1v8_int_l               ,
        output zqsfp1_1v8_reset_l             ,
        input  zqsfp1_1v8_modprs_l            ,
        output zqsfp1_1v8_lpmode              ,
        input  zqsfp1_1v8_int_l               ,
        output zqsfp1_1v8_modsel_l            ,
        inout  zqsfp_s10_i2c_sda              ,
        output zqsfp_s10_i2c_scl              

        );

reg     [26:0] heart_beat_cnt;

//glue logic for heart beat
always @(posedge CLK_SYS_50M_P or negedge CPU_RESETN)
        if (!CPU_RESETN)
            heart_beat_cnt <= 27'h0; //0x7FFFFFF
        else
            heart_beat_cnt <= heart_beat_cnt + 1'b1 ;


assign S10_LED = {heart_beat_cnt[26], heart_beat_cnt[26], heart_beat_cnt[26], heart_beat_cnt[26]};

endmodule
