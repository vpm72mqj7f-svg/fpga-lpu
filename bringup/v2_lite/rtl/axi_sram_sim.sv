// =============================================================================
// axi_sram_sim.sv — Behavioral AXI4 SRAM (HBM2 simulation model)
//
// Responds to AXI4 read requests with pre-loaded weight data.
// Simulates HBM2 latency (AXI read response ~10 cycles).
// =============================================================================
module axi_sram_sim #(
    parameter ADDR_W = 32, DATA_W = 256, ID_W = 9,
    parameter DEPTH = 65536  // 64K × 256-bit = 2 MB
) (
    input  wire        clk, rst_n,
    // AXI Read Address
    input  wire [ADDR_W-1:0] s_axi_araddr,
    input  wire [7:0]        s_axi_arlen,
    input  wire [2:0]        s_axi_arsize,
    input  wire [ID_W-1:0]   s_axi_arid,
    input  wire              s_axi_arvalid,
    output wire              s_axi_arready,
    // AXI Read Data
    output wire [DATA_W-1:0] s_axi_rdata,
    output wire [1:0]        s_axi_rresp,
    output wire [ID_W-1:0]   s_axi_rid,
    output wire              s_axi_rvalid,
    input  wire              s_axi_rready,
    output wire              s_axi_rlast
);
    // Simple SRAM array
    reg [DATA_W-1:0] mem [0:DEPTH-1];

    // Initialize with ramp data for testing
    integer i;
    initial begin
        for (i = 0; i < DEPTH; i = i + 1)
            mem[i] = {4{i[7:0], i[7:0], i[7:0], i[7:0], i[7:0], i[7:0], i[7:0], i[7:0]}};
    end

    // Read state machine
    reg [15:0]  rd_addr;
    reg [7:0]   rd_count;    // how many beats remaining in burst
    reg [7:0]   rd_len;      // burst length
    reg [ID_W-1:0] rd_id;
    reg         rd_active;
    reg [3:0]   rd_latency;  // sim latency counter

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_active  <= 1'b0;
            rd_count   <= 8'd0;
            rd_latency <= 4'd0;
        end else begin
            // Accept new read request
            if (!rd_active && s_axi_arvalid) begin
                rd_addr    <= s_axi_araddr[15:0];  // word address
                rd_len     <= s_axi_arlen;
                rd_id      <= s_axi_arid;
                rd_count   <= s_axi_arlen + 8'd1;  // total beats
                rd_active  <= 1'b1;
                rd_latency <= 4'd0;
            end

            // Simulate HBM2 read latency
            if (rd_active && rd_latency < 4'd10)
                rd_latency <= rd_latency + 4'd1;

            // Data handshake
            if (rd_active && rd_latency >= 4'd10 && s_axi_rready) begin
                rd_addr  <= rd_addr + 16'd1;
                if (rd_count > 1)
                    rd_count <= rd_count - 8'd1;
                else
                    rd_active <= 1'b0;
            end
        end
    end

    assign s_axi_arready = !rd_active;
    assign s_axi_rvalid  = rd_active && (rd_latency >= 4'd10);
    assign s_axi_rdata   = mem[rd_addr];
    assign s_axi_rresp   = 2'b00;  // OKAY
    assign s_axi_rid     = rd_id;
    assign s_axi_rlast   = rd_active && (rd_count == 1) && s_axi_rready;
endmodule
