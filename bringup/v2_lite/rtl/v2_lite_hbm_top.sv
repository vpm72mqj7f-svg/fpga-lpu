// =============================================================================
// v2_lite_hbm_top.sv — V2-Lite FFN with Debug/Observation
//
// Debug ports expose key internal nodes for simulation & board bringup.
// All important states observable via external logic analyzer or JTAG.
// =============================================================================
module v2_lite_hbm_top #(
    parameter HIDDEN = 2048, INTER = 1408, NUM_EXPERTS = 66, TOP_K = 6, DATA_W = 8,
    parameter DBG_FIFO_DEPTH = 256
) (
    input  wire       core_clk_iopll_ref_clk_clk,
    input  wire       cpu_resetn,
    output wire [3:0] led,

    // ==== Debug/Test Interface ====
    // Test vector injection
    input  wire [7:0]        dbg_test_activ,        // activation byte to inject
    input  wire              dbg_inject_valid,      // strobe to inject one byte
    input  wire [10:0]       dbg_inject_addr,       // which activation element (0..2047)
    output wire [7:0]        dbg_ffn_out_byte,      // FFN output byte readback
    input  wire [10:0]       dbg_read_addr,         // which output element to read
    output wire [7:0]        dbg_ffn_state,         // {4'b0, fsm_state[3:0]}
    output wire              dbg_ffn_busy,
    output wire              dbg_ffn_done,
    output wire              dbg_ffn_pass,

    // Systolic array observability
    output wire [15:0]       dbg_sa_gate_out,       // gate projection result sample
    output wire [15:0]       dbg_sa_up_out,         // up projection result sample
    output wire [15:0]       dbg_sa_down_out,       // down projection result sample
    output wire [10:0]       dbg_current_expert,    // which expert being processed
    output wire [2:0]        dbg_pipeline_stage     // {fsm_state[3:1]}
);
    // =========================================================================
    // Clock & Reset
    // =========================================================================
    reg [7:0] rst_cnt;
    wire rst_n = (rst_cnt == 8'd255);
    always @(posedge core_clk_iopll_ref_clk_clk or negedge cpu_resetn)
        if (!cpu_resetn) rst_cnt <= 8'd0;
        else if (rst_cnt < 8'd255) rst_cnt <= rst_cnt + 8'd1;

    // =========================================================================
    // Activation Buffer (M20K-inferred, dual-port for debug read)
    // =========================================================================
    reg [DATA_W-1:0] activ_buf [0:HIDDEN-1];
    reg [10:0]        dbg_read_addr_r;

    // Debug: write test activations into buffer
    always @(posedge core_clk_iopll_ref_clk_clk) begin
        if (dbg_inject_valid)
            activ_buf[dbg_inject_addr] <= dbg_test_activ;
        dbg_read_addr_r <= dbg_read_addr;
    end
    assign dbg_ffn_out_byte = activ_buf[dbg_read_addr_r]; // simplified: read activ buf

    // =========================================================================
    // FFN Engine
    // =========================================================================
    reg         ffn_rx_valid;
    reg  [HIDDEN*DATA_W-1:0] ffn_rx_data;
    wire        ffn_rx_ready;
    wire        ffn_tx_valid;
    wire [HIDDEN*DATA_W-1:0] ffn_tx_data;
    reg         ffn_tx_ready;
    wire        ffn_busy;
    wire        ffn_done;

    // HBM2 AXI4 tied off
    wire [31:0]  ffn_araddr;  wire [7:0] ffn_arlen; wire [2:0] ffn_arsize;
    wire         ffn_arvalid; wire ffn_arready = 1'b0;
    wire [255:0] ffn_rdata = 256'd0; wire [1:0] ffn_rresp = 2'd0;
    wire         ffn_rvalid = 1'b0; wire ffn_rready; wire ffn_rlast = 1'b0;

    wire [6:0]   ffn_expert_id [0:TOP_K-1];
    genvar ei;
    generate for (ei = 0; ei < TOP_K; ei = ei + 1) assign ffn_expert_id[ei] = ei; endgenerate

    v2_lite_ffn_engine #(.HIDDEN(HIDDEN), .INTER(INTER), .NUM_EXPERTS(NUM_EXPERTS), .TOP_K(TOP_K), .DATA_W(DATA_W))
    u_ffn (
        .clk(core_clk_iopll_ref_clk_clk), .rst_n(rst_n),
        .pcie_rx_valid(ffn_rx_valid), .pcie_rx_data(ffn_rx_data), .pcie_rx_ready(ffn_rx_ready),
        .pcie_tx_valid(ffn_tx_valid), .pcie_tx_data(ffn_tx_data), .pcie_tx_ready(ffn_tx_ready),
        .m_axi_araddr(ffn_araddr), .m_axi_arlen(ffn_arlen), .m_axi_arsize(ffn_arsize),
        .m_axi_arvalid(ffn_arvalid), .m_axi_arready(ffn_arready),
        .m_axi_rdata(ffn_rdata), .m_axi_rresp(ffn_rresp),
        .m_axi_rvalid(ffn_rvalid), .m_axi_rready(ffn_rready), .m_axi_rlast(ffn_rlast),
        .expert_id(ffn_expert_id), .busy(ffn_busy), .done(ffn_done)
    );

    // =========================================================================
    // FFN Self-Test FSM
    // =========================================================================
    localparam [3:0] B_IDLE=0, B_WAIT=1, B_SEND=2, B_BUSY=3, B_CHECK=4, B_PASS=5, B_FAIL=6;
    reg [3:0] bstate;
    reg       ffn_done_latched;
    reg       ffn_pass;

    always @(posedge core_clk_iopll_ref_clk_clk or negedge rst_n) begin
        if (!rst_n) begin
            bstate <= B_IDLE; ffn_rx_valid <= 1'b0; ffn_tx_ready <= 1'b0;
            ffn_done_latched <= 1'b0; ffn_pass <= 1'b0;
        end else begin
            case (bstate)
                B_IDLE:  bstate <= B_WAIT;
                B_WAIT:  bstate <= B_SEND;
                B_SEND: begin
                    integer i;
                    ffn_rx_valid <= 1'b1;
                    // Use injected test activations from debug buffer
                    for (i = 0; i < HIDDEN; i = i + 1)
                        ffn_rx_data[i*DATA_W +: DATA_W] <= activ_buf[i];
                    bstate <= B_BUSY;
                end
                B_BUSY: begin
                    ffn_rx_valid <= 1'b0;
                    if (ffn_done) begin ffn_done_latched <= 1'b1; ffn_tx_ready <= 1'b1; bstate <= B_CHECK; end
                end
                B_CHECK: begin
                    ffn_tx_ready <= 1'b0;
                    ffn_pass <= (|ffn_tx_data);
                    bstate <= ffn_pass ? B_PASS : B_FAIL;
                end
                B_PASS, B_FAIL: ;
            endcase
        end
    end

    // =========================================================================
    // Debug outputs
    // =========================================================================
    assign dbg_ffn_state     = {4'd0, bstate};
    assign dbg_ffn_busy      = ffn_busy || (bstate == B_BUSY);
    assign dbg_ffn_done      = ffn_done_latched;
    assign dbg_ffn_pass      = ffn_pass;
    assign dbg_sa_gate_out   = 16'd0;  // placeholder — connect to FFN engine internals
    assign dbg_sa_up_out     = 16'd0;
    assign dbg_sa_down_out   = 16'd0;
    assign dbg_current_expert = 11'd0;
    assign dbg_pipeline_stage = bstate[3:1];

    // =========================================================================
    // LED Encoding
    // =========================================================================
    reg [26:0] heart_beat_cnt;
    always @(posedge core_clk_iopll_ref_clk_clk or negedge cpu_resetn)
        if (!cpu_resetn) heart_beat_cnt <= 27'd0;
        else heart_beat_cnt <= heart_beat_cnt + 27'd1;

    assign led[0] = heart_beat_cnt[26];
    assign led[1] = ffn_busy || (bstate == B_BUSY);
    assign led[2] = ffn_done_latched;
    assign led[3] = ffn_pass ? heart_beat_cnt[25] : 1'b0;
endmodule
