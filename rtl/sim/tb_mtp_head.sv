`timescale 1ns/1ps

module tb_mtp_head;
    localparam int HIDDEN   = 8;
    localparam int VOCAB    = 16;
    localparam int N_HEADS  = 2;
    localparam int WEIGHT_W = 16;
    localparam int DATA_W   = 32;
    localparam int VCB      = $clog2(VOCAB);

    localparam int Q12_ONE  = 4096;
    localparam int Q12_ZERO = 0;

    logic clk, rst_n;
    logic in_valid, in_ready;
    logic [HIDDEN*DATA_W-1:0] hidden_flat;
    logic wt_wr_en;
    logic [0:0] wt_head_id;
    logic [3:0] wt_vocab_id;
    logic [2:0] wt_dim_id;
    logic signed [WEIGHT_W-1:0] wt_wr_data;
    logic out_valid;
    logic [N_HEADS*VCB-1:0] token_ids_flat;
    logic [N_HEADS*DATA_W-1:0] logprobs_flat;

    mtp_head #(.HIDDEN(HIDDEN), .VOCAB(VOCAB), .N_HEADS(N_HEADS),
               .WEIGHT_W(WEIGHT_W), .DATA_W(DATA_W))
        u_head (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    task wait_cycles(input int n);
        for (int i = 0; i < n; i++) @(posedge clk);
    endtask

    function [HIDDEN*DATA_W-1:0] make_vec(input int base);
        reg [HIDDEN*DATA_W-1:0] v;
        for (int d = 0; d < HIDDEN; d++) v[d*DATA_W +: DATA_W] = base + d;
        make_vec = v;
    endfunction

    function [VCB-1:0] get_token(input int h);
        get_token = token_ids_flat[h*VCB +: VCB];
    endfunction

    function [DATA_W-1:0] get_logprob(input int h);
        get_logprob = logprobs_flat[h*DATA_W +: DATA_W];
    endfunction

    integer pass_count, fail_count;

    initial begin
        rst_n = 0; in_valid = 0; hidden_flat = '0;
        wt_wr_en = 0; wt_head_id = '0; wt_vocab_id = '0; wt_dim_id = '0; wt_wr_data = '0;
        pass_count = 0; fail_count = 0;

        wait_cycles(4);
        rst_n = 1;
        wait_cycles(2);

        // Load weights: head0→vocab3, head1→vocab7
        $display("Loading weights...");
        for (int v = 0; v < VOCAB; v++) begin
            for (int d = 0; d < HIDDEN; d++) begin
                @(posedge clk);
                wt_wr_en = 1; wt_dim_id = d; wt_vocab_id = v;
                wt_head_id = 0; wt_wr_data = (v == 3) ? Q12_ONE : Q12_ZERO;
                @(posedge clk); wt_wr_en = 0;
                @(posedge clk);
                wt_wr_en = 1; wt_head_id = 1; wt_dim_id = d; wt_vocab_id = v;
                wt_wr_data = (v == 7) ? Q12_ONE : Q12_ZERO;
                @(posedge clk); wt_wr_en = 0;
            end
        end

        // Test 1: Uniform hidden → predict tokens 3 and 7
        $display("Test 1: MTP head with all-ones hidden state");
        @(posedge clk);
        in_valid = 1; hidden_flat = make_vec(Q12_ONE);
        @(posedge clk);
        in_valid = 0;

        for (int cyc = 0; cyc < 40; cyc++) begin
            @(posedge clk);
            if (out_valid) begin
                $display("  Head 0: token=%0d logprob=%0d", get_token(0), get_logprob(0));
                $display("  Head 1: token=%0d logprob=%0d", get_token(1), get_logprob(1));

                if (get_token(0) !== 3) begin
                    $error("  [FAIL] Head 0: expected 3, got %0d", get_token(0));
                    fail_count = fail_count + 1;
                end
                if (get_token(1) !== 7) begin
                    $error("  [FAIL] Head 1: expected 7, got %0d", get_token(1));
                    fail_count = fail_count + 1;
                end
                if (get_logprob(0) < 32000 || get_logprob(1) < 32000) begin
                    $error("  [FAIL] logprobs too low");
                    fail_count = fail_count + 1;
                end
                if (fail_count == 0) begin
                    $display("  [ OK ] Test 1");
                    pass_count = pass_count + 1;
                end
            end
        end

        wait_cycles(4);

        // Test 2: New weights head0→5, head1→2
        $display("Test 2: New target tokens 5 and 2");
        for (int v = 0; v < VOCAB; v++) begin
            for (int d = 0; d < HIDDEN; d++) begin
                @(posedge clk);
                wt_wr_en = 1; wt_dim_id = d; wt_vocab_id = v;
                wt_head_id = 0; wt_wr_data = (v == 5) ? Q12_ONE : Q12_ZERO;
                @(posedge clk); wt_wr_en = 0;
                @(posedge clk);
                wt_wr_en = 1; wt_head_id = 1; wt_dim_id = d; wt_vocab_id = v;
                wt_wr_data = (v == 2) ? Q12_ONE : Q12_ZERO;
                @(posedge clk); wt_wr_en = 0;
            end
        end

        @(posedge clk);
        in_valid = 1; hidden_flat = make_vec(Q12_ONE);
        @(posedge clk);
        in_valid = 0;

        for (int cyc = 0; cyc < 40; cyc++) begin
            @(posedge clk);
            if (out_valid) begin
                $display("  Head 0: token=%0d, Head 1: token=%0d", get_token(0), get_token(1));
                if (get_token(0) !== 5 || get_token(1) !== 2) begin
                    $error("  [FAIL] Expected (5,2) got (%0d,%0d)", get_token(0), get_token(1));
                    fail_count = fail_count + 1;
                end else begin
                    $display("  [ OK ] Test 2");
                    pass_count = pass_count + 1;
                end
            end
        end

        $display("==============================");
        if (fail_count == 0)
            $display("PASS tb_mtp_head (%0d/2 tests)", pass_count);
        else
            $display("FAIL tb_mtp_head (%0d pass, %0d fail)", pass_count, fail_count);
        $finish;
    end

endmodule
