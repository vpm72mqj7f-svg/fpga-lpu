//=============================================================================
// axi4_rr_arbiter.sv — AXI4 Round-Robin Arbiter (N masters → 1 slave)
//
// Icarus-compatible. Each AXI4 channel has independent round-robin arbitration.
// Write path: AW grant locks W channel for that master until wlast.
// Read path:  AR grant locks R channel for that master until rlast.
//=============================================================================

module axi4_rr_arbiter #(
    parameter int NUM_MASTERS    = 3,
    parameter int AXI_DATA_WIDTH = 256,
    parameter int AXI_ADDR_WIDTH = 32,
    parameter int AXI_ID_WIDTH   = 4
) (
    input  logic        clk,
    input  logic        rst_n,

    // ── Master 0 (weight preloader) ──
    input  logic [AXI_ID_WIDTH-1:0]   m0_awid,     m0_arid,
    input  logic [AXI_ADDR_WIDTH-1:0] m0_awaddr,   m0_araddr,
    input  logic [7:0]                m0_awlen,    m0_arlen,
    input  logic [2:0]                m0_awsize,   m0_arsize,
    input  logic [1:0]                m0_awburst,  m0_arburst,
    input  logic                      m0_awvalid,  m0_arvalid,
    output logic                      m0_awready,  m0_arready,
    input  logic [AXI_DATA_WIDTH-1:0]  m0_wdata,
    input  logic [AXI_DATA_WIDTH/8-1:0] m0_wstrb,
    input  logic                       m0_wlast,
    input  logic                       m0_wvalid,
    output logic                       m0_wready,
    output logic [AXI_ID_WIDTH-1:0]   m0_bid,
    output logic [1:0]                m0_bresp,
    output logic                      m0_bvalid,
    input  logic                      m0_bready,
    output logic [AXI_ID_WIDTH-1:0]   m0_rid,
    output logic [AXI_DATA_WIDTH-1:0] m0_rdata,
    output logic [1:0]                m0_rresp,
    output logic                      m0_rlast,
    output logic                      m0_rvalid,
    input  logic                      m0_rready,

    // ── Master 1 (KV DMA) ──
    input  logic [AXI_ID_WIDTH-1:0]   m1_awid,     m1_arid,
    input  logic [AXI_ADDR_WIDTH-1:0] m1_awaddr,   m1_araddr,
    input  logic [7:0]                m1_awlen,    m1_arlen,
    input  logic [2:0]                m1_awsize,   m1_arsize,
    input  logic [1:0]                m1_awburst,  m1_arburst,
    input  logic                      m1_awvalid,  m1_arvalid,
    output logic                      m1_awready,  m1_arready,
    input  logic [AXI_DATA_WIDTH-1:0]  m1_wdata,
    input  logic [AXI_DATA_WIDTH/8-1:0] m1_wstrb,
    input  logic                       m1_wlast,
    input  logic                       m1_wvalid,
    output logic                       m1_wready,
    output logic [AXI_ID_WIDTH-1:0]   m1_bid,
    output logic [1:0]                m1_bresp,
    output logic                      m1_bvalid,
    input  logic                      m1_bready,
    output logic [AXI_ID_WIDTH-1:0]   m1_rid,
    output logic [AXI_DATA_WIDTH-1:0] m1_rdata,
    output logic [1:0]                m1_rresp,
    output logic                      m1_rlast,
    output logic                      m1_rvalid,
    input  logic                      m1_rready,

    // ── Master 2 (attention reads) ──
    input  logic [AXI_ID_WIDTH-1:0]   m2_awid,     m2_arid,
    input  logic [AXI_ADDR_WIDTH-1:0] m2_awaddr,   m2_araddr,
    input  logic [7:0]                m2_awlen,    m2_arlen,
    input  logic [2:0]                m2_awsize,   m2_arsize,
    input  logic [1:0]                m2_awburst,  m2_arburst,
    input  logic                      m2_awvalid,  m2_arvalid,
    output logic                      m2_awready,  m2_arready,
    input  logic [AXI_DATA_WIDTH-1:0]  m2_wdata,
    input  logic [AXI_DATA_WIDTH/8-1:0] m2_wstrb,
    input  logic                       m2_wlast,
    input  logic                       m2_wvalid,
    output logic                       m2_wready,
    output logic [AXI_ID_WIDTH-1:0]   m2_bid,
    output logic [1:0]                m2_bresp,
    output logic                      m2_bvalid,
    input  logic                      m2_bready,
    output logic [AXI_ID_WIDTH-1:0]   m2_rid,
    output logic [AXI_DATA_WIDTH-1:0] m2_rdata,
    output logic [1:0]                m2_rresp,
    output logic                      m2_rlast,
    output logic                      m2_rvalid,
    input  logic                      m2_rready,

    // ── Slave side (to HBM model) ──
    output logic [AXI_ID_WIDTH-1:0]   s_awid,     s_arid,
    output logic [AXI_ADDR_WIDTH-1:0] s_awaddr,   s_araddr,
    output logic [7:0]                s_awlen,    s_arlen,
    output logic [2:0]                s_awsize,   s_arsize,
    output logic [1:0]                s_awburst,  s_arburst,
    output logic                      s_awvalid,  s_arvalid,
    input  logic                      s_awready,  s_arready,
    output logic [AXI_DATA_WIDTH-1:0]  s_wdata,
    output logic [AXI_DATA_WIDTH/8-1:0] s_wstrb,
    output logic                       s_wlast,
    output logic                       s_wvalid,
    input  logic                       s_wready,
    input  logic [AXI_ID_WIDTH-1:0]   s_bid,
    input  logic [1:0]                s_bresp,
    input  logic                      s_bvalid,
    output logic                      s_bready,
    input  logic [AXI_ID_WIDTH-1:0]   s_rid,
    input  logic [AXI_DATA_WIDTH-1:0] s_rdata,
    input  logic [1:0]                s_rresp,
    input  logic                      s_rlast,
    input  logic                      s_rvalid,
    output logic                      s_rready
);

    localparam int AW = 0;
    localparam int W  = 1;
    localparam int B  = 2;
    localparam int AR = 3;
    localparam int R  = 4;

    // Grant pointers per channel
    logic [1:0] aw_grant;   // which master owns AW
    logic [1:0] w_grant;    // which master owns W
    logic [1:0] b_grant;    // which master gets B response
    logic [1:0] ar_grant;   // which master owns AR
    logic [1:0] r_grant;    // which master gets R data

    // Round-robin priority pointers
    logic [1:0] aw_prio;
    logic [1:0] ar_prio;

    // Write burst in progress (lock W channel to granted master)
    logic       w_busy;
    logic       r_busy;

    // ── Packed master input arrays (manual for Icarus) ──
    wire [AXI_ID_WIDTH-1:0]   m_awid    [0:2];
    wire [AXI_ADDR_WIDTH-1:0] m_awaddr  [0:2];
    wire [7:0]                m_awlen   [0:2];
    wire                      m_awvalid [0:2];
    wire [AXI_DATA_WIDTH-1:0]  m_wdata   [0:2];
    wire [AXI_DATA_WIDTH/8-1:0] m_wstrb   [0:2];
    wire                       m_wlast   [0:2];
    wire                       m_wvalid  [0:2];
    wire                       m_bready  [0:2];
    wire [AXI_ID_WIDTH-1:0]   m_arid    [0:2];
    wire [AXI_ADDR_WIDTH-1:0] m_araddr  [0:2];
    wire [7:0]                m_arlen   [0:2];
    wire                      m_arvalid [0:2];
    wire                       m_rready  [0:2];

    assign m_awid[0] = m0_awid;   assign m_awid[1] = m1_awid;   assign m_awid[2] = m2_awid;
    assign m_awaddr[0] = m0_awaddr; assign m_awaddr[1] = m1_awaddr; assign m_awaddr[2] = m2_awaddr;
    assign m_awlen[0] = m0_awlen;  assign m_awlen[1] = m1_awlen;  assign m_awlen[2] = m2_awlen;
    assign m_awvalid[0] = m0_awvalid; assign m_awvalid[1] = m1_awvalid; assign m_awvalid[2] = m2_awvalid;
    assign m_wdata[0] = m0_wdata;  assign m_wdata[1] = m1_wdata;  assign m_wdata[2] = m2_wdata;
    assign m_wstrb[0] = m0_wstrb;  assign m_wstrb[1] = m1_wstrb;  assign m_wstrb[2] = m2_wstrb;
    assign m_wlast[0] = m0_wlast;   assign m_wlast[1] = m1_wlast;   assign m_wlast[2] = m2_wlast;
    assign m_wvalid[0] = m0_wvalid; assign m_wvalid[1] = m1_wvalid; assign m_wvalid[2] = m2_wvalid;
    assign m_bready[0] = m0_bready; assign m_bready[1] = m1_bready; assign m_bready[2] = m2_bready;
    assign m_arid[0] = m0_arid;    assign m_arid[1] = m1_arid;    assign m_arid[2] = m2_arid;
    assign m_araddr[0] = m0_araddr; assign m_araddr[1] = m1_araddr; assign m_araddr[2] = m2_araddr;
    assign m_arlen[0] = m0_arlen;   assign m_arlen[1] = m1_arlen;   assign m_arlen[2] = m2_arlen;
    assign m_arvalid[0] = m0_arvalid; assign m_arvalid[1] = m1_arvalid; assign m_arvalid[2] = m2_arvalid;
    assign m_rready[0] = m0_rready; assign m_rready[1] = m1_rready; assign m_rready[2] = m2_rready;

    // ── AW arbitration: round-robin ──
    function [1:0] next_prio;
        input [1:0] cur;
        begin
            next_prio = (cur == 2'd2) ? 2'd0 : (cur + 2'd1);
        end
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            aw_prio  <= 2'd0;
            aw_grant <= 2'd0;
            w_busy   <= 1'b0;
        end else begin
            if (!w_busy && s_awvalid && s_awready) begin
                // Transaction started, lock W channel
                w_busy   <= 1'b1;
                w_grant  <= aw_grant;
                aw_prio  <= next_prio(aw_grant);
            end

            if (w_busy && s_wvalid && s_wready && s_wlast) begin
                w_busy  <= 1'b0;
            end

            // Find next requesting master (round-robin from aw_prio)
            if (!w_busy && m_awvalid[aw_prio])
                aw_grant <= aw_prio;
            else if (!w_busy && m_awvalid[(aw_prio + 2'd1) % 3])
                aw_grant <= (aw_prio + 2'd1) % 3;
            else if (!w_busy && m_awvalid[(aw_prio + 2'd2) % 3])
                aw_grant <= (aw_prio + 2'd2) % 3;
        end
    end

    // AW mux
    assign s_awid    = m_awid[aw_grant];
    assign s_awaddr  = m_awaddr[aw_grant];
    assign s_awlen   = m_awlen[aw_grant];
    assign s_awsize  = 3'd5;
    assign s_awburst = 2'd1;
    assign s_awvalid = m_awvalid[aw_grant] && !w_busy;

    assign m0_awready = (aw_grant == 2'd0) && s_awready && !w_busy;
    assign m1_awready = (aw_grant == 2'd1) && s_awready && !w_busy;
    assign m2_awready = (aw_grant == 2'd2) && s_awready && !w_busy;

    // W mux (locked to w_grant during burst)
    assign s_wdata  = m_wdata[w_grant];
    assign s_wstrb  = m_wstrb[w_grant];
    assign s_wlast  = m_wlast[w_grant];
    assign s_wvalid = m_wvalid[w_grant] && w_busy;

    assign m0_wready = (w_grant == 2'd0) && s_wready && w_busy;
    assign m1_wready = (w_grant == 2'd1) && s_wready && w_busy;
    assign m2_wready = (w_grant == 2'd2) && s_wready && w_busy;

    // B demux (route back to W grant owner)
    assign s_bready = m_bready[w_grant];

    assign m0_bid    = s_bid;
    assign m0_bresp  = s_bresp;
    assign m0_bvalid = s_bvalid && (w_grant == 2'd0);
    assign m1_bid    = s_bid;
    assign m1_bresp  = s_bresp;
    assign m1_bvalid = s_bvalid && (w_grant == 2'd1);
    assign m2_bid    = s_bid;
    assign m2_bresp  = s_bresp;
    assign m2_bvalid = s_bvalid && (w_grant == 2'd2);

    // ── AR arbitration: round-robin ──
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ar_prio  <= 2'd0;
            ar_grant <= 2'd0;
            r_busy   <= 1'b0;
        end else begin
            if (!r_busy && s_arvalid && s_arready) begin
                r_busy   <= 1'b1;
                r_grant  <= ar_grant;
                ar_prio  <= next_prio(ar_grant);
            end

            if (r_busy && s_rvalid && s_rready && s_rlast) begin
                r_busy  <= 1'b0;
            end

            if (!r_busy && m_arvalid[ar_prio])
                ar_grant <= ar_prio;
            else if (!r_busy && m_arvalid[(ar_prio + 2'd1) % 3])
                ar_grant <= (ar_prio + 2'd1) % 3;
            else if (!r_busy && m_arvalid[(ar_prio + 2'd2) % 3])
                ar_grant <= (ar_prio + 2'd2) % 3;
        end
    end

    // AR mux
    assign s_arid    = m_arid[ar_grant];
    assign s_araddr  = m_araddr[ar_grant];
    assign s_arlen   = m_arlen[ar_grant];
    assign s_arsize  = 3'd5;
    assign s_arburst = 2'd1;
    assign s_arvalid = m_arvalid[ar_grant] && !r_busy;

    assign m0_arready = (ar_grant == 2'd0) && s_arready && !r_busy;
    assign m1_arready = (ar_grant == 2'd1) && s_arready && !r_busy;
    assign m2_arready = (ar_grant == 2'd2) && s_arready && !r_busy;

    // R demux (route to AR grant owner)
    assign s_rready = m_rready[r_grant];

    assign m0_rid    = s_rid;
    assign m0_rdata  = s_rdata;
    assign m0_rresp  = s_rresp;
    assign m0_rlast  = s_rlast;
    assign m0_rvalid = s_rvalid && (r_grant == 2'd0);
    assign m1_rid    = s_rid;
    assign m1_rdata  = s_rdata;
    assign m1_rresp  = s_rresp;
    assign m1_rlast  = s_rlast;
    assign m1_rvalid = s_rvalid && (r_grant == 2'd1);
    assign m2_rid    = s_rid;
    assign m2_rdata  = s_rdata;
    assign m2_rresp  = s_rresp;
    assign m2_rlast  = s_rlast;
    assign m2_rvalid = s_rvalid && (r_grant == 2'd2);

endmodule
