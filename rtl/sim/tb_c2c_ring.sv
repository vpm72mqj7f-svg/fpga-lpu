`timescale 1ns/1ps

module tb_c2c_ring;
    logic clk, rst_n;

    // N0→N1 link
    logic n01_valid, n01_ready;
    logic [7:0] n01_dst; logic [15:0] n01_tid; logic [31:0] n01_data; logic n01_last;
    // N1→N2 link
    logic n12_valid, n12_ready;
    logic [7:0] n12_dst; logic [15:0] n12_tid; logic [31:0] n12_data; logic n12_last;
    // N2→N3 link
    logic n23_valid, n23_ready;
    logic [7:0] n23_dst; logic [15:0] n23_tid; logic [31:0] n23_data; logic n23_last;

    // Host
    logic h_valid, h_ready;
    logic [7:0] h_dst; logic [15:0] h_tid; logic [31:0] h_data; logic h_last;
    logic host_recv_valid; logic [15:0] host_recv_tid; logic [31:0] host_recv_data;

    c2c_node #(0) n0(.clk,.rst_n, .rx_valid(h_valid),.rx_ready(h_ready),.rx_dst(h_dst),
        .rx_token_id(h_tid),.rx_data(h_data),.rx_last(h_last),
        .tx_valid(n01_valid),.tx_ready(1'b1),  .tx_dst(n01_dst),
        .tx_token_id(n01_tid),.tx_data(n01_data),.tx_last(n01_last),
        .host_valid(),.host_token_id(),.host_data());
    c2c_node #(1) n1(.clk,.rst_n, .rx_valid(n01_valid),.rx_ready(n01_ready),
        .rx_dst(n01_dst),.rx_token_id(n01_tid),.rx_data(n01_data),.rx_last(n01_last),
        .tx_valid(n12_valid),.tx_ready(1'b1),  .tx_dst(n12_dst),
        .tx_token_id(n12_tid),.tx_data(n12_data),.tx_last(n12_last),
        .host_valid(),.host_token_id(),.host_data());
    c2c_node #(2) n2(.clk,.rst_n, .rx_valid(n12_valid),.rx_ready(n12_ready),
        .rx_dst(n12_dst),.rx_token_id(n12_tid),.rx_data(n12_data),.rx_last(n12_last),
        .tx_valid(n23_valid),.tx_ready(1'b1),  .tx_dst(n23_dst),
        .tx_token_id(n23_tid),.tx_data(n23_data),.tx_last(n23_last),
        .host_valid(),.host_token_id(),.host_data());
    c2c_node #(3) n3(.clk,.rst_n, .rx_valid(n23_valid),.rx_ready(n23_ready),
        .rx_dst(n23_dst),.rx_token_id(n23_tid),.rx_data(n23_data),.rx_last(n23_last),
        .tx_valid(),.tx_ready(1'b1),  .tx_dst(),.tx_token_id(),.tx_data(),.tx_last(),
        .host_valid(host_recv_valid),.host_token_id(host_recv_tid),.host_data(host_recv_data));

    initial clk=0; always #5 clk=~clk;

    initial begin
        rst_n=0; h_valid=0; repeat(4) @(posedge clk); rst_n=1;
        @(posedge clk); #1; h_valid=1; h_dst=0; h_tid=1; h_data=100; h_last=0;
        @(posedge clk); #1; h_valid=0;
        for (int c=0; c<60; c++) begin
            @(posedge clk);
            if (host_recv_valid) begin #1;
                if (host_recv_data!=104) begin $error("got %0d exp 104", host_recv_data); $fatal; end
                $display("PASS tb_c2c_ring (data=%0d)", host_recv_data); $finish;
            end
        end
        $error("timeout"); $fatal;
    end
endmodule
