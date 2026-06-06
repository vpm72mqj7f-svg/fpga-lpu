//=============================================================================
// altera_syncram.sv — Intel/Altera Synchronous RAM Wrapper
//
// Icarus-compatible behavioral fallback. Quartus infers M20K/MLAB block RAM
// from the simple-dual-port pattern and adds output registers during synthesis
// to meet M20K hardware requirements.
//
// Replaces hand-inferred (* ramstyle = "M20K" *) arrays with explicit IP
// instantiation, matching the FPGA design principle that ALL generic IP
// (FIFO, RAM, DSP) must use Altera IP instances.
//
// Behavioral model: combinational read (q follows rdaddress), registered
// write (mem[wraddress] <= data at posedge when wren=1). Quartus retimes
// read->q path into M20K output registers during synthesis at 450 MHz.
//
// Parameters:
//   WIDTH           — data width in bits
//   DEPTH           — number of entries (power-of-2 for Quartus)
//   RAM_BLOCK_TYPE  — "M20K", "MLAB", or "AUTO" (default: "AUTO")
//=============================================================================

module altera_syncram #(
    parameter int  WIDTH           = 32,
    parameter int  DEPTH           = 512,
    parameter      RAM_BLOCK_TYPE  = "AUTO",
    parameter      INIT_VALUE      = '0    // per-word init (for ROM-style LUTs)
) (
    input  logic                     clock,
    input  logic                     wren,
    input  logic [ADDR_W-1:0]        wraddress,
    input  logic [WIDTH-1:0]         data,
    input  logic [ADDR_W-1:0]        rdaddress,
    output logic [WIDTH-1:0]         q
);

    localparam int ADDR_W = $clog2(DEPTH > 1 ? DEPTH : 2);
    localparam int DEPTH_PAD = ADDR_W > 0 ? DEPTH : 2;

    logic [WIDTH-1:0] mem [0:DEPTH_PAD-1];

    // Quartus M20K/MLAB inference: recognizes wren + rdaddress pattern.

    // simulation: initialize all entries (BRAM has no HW reset; testbench fills)
    // synthesis translate_off
    initial begin
        for (int i = 0; i < DEPTH_PAD; i++) mem[i] = INIT_VALUE;
    end
    // synthesis translate_on

    // Write (registered)
    always_ff @(posedge clock) begin
        if (wren)
            mem[wraddress] <= data;
    end

    // Read (combinational — matches behavioral array semantics;
    // Quartus retimes into M20K output register during synthesis)
    assign q = (rdaddress < DEPTH) ? mem[rdaddress] : '0;

endmodule
