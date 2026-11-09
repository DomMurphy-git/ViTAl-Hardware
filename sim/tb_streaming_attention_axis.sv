module tb_streaming_attention_axis;
    localparam integer QUERY_DIM = 32;
    localparam integer VALUE_DIM = 16;
    localparam integer TOKEN_COUNT = 196;

    logic clk = 0;
    logic resetn = 0;
    logic [31:0] s_data = 0;
    logic s_valid = 0;
    logic s_ready;
    logic s_last = 0;
    logic [31:0] m_data;
    logic m_valid;
    logic m_ready = 1;
    logic m_last;

    integer output_count = 0;
    integer cycles = 0;

    always #5 clk = ~clk;

    streaming_attention_axis dut (
        .s_axis_aclk(clk),
        .s_axis_aresetn(resetn),
        .s_axis_tdata(s_data),
        .s_axis_tvalid(s_valid),
        .s_axis_tready(s_ready),
        .s_axis_tlast(s_last),
        .m_axis_tdata(m_data),
        .m_axis_tvalid(m_valid),
        .m_axis_tready(m_ready),
        .m_axis_tlast(m_last)
    );

    task automatic send_byte(input logic signed [7:0] data_byte);
        begin
            @(negedge clk);
            s_data = {{24{data_byte[7]}}, data_byte};
            s_valid = 1;
            do @(posedge clk); while (!s_ready);
            @(negedge clk);
            s_valid = 0;
        end
    endtask

    always @(posedge clk) begin
        if (resetn) begin
            cycles <= cycles + 1;
            if (cycles > 20000)
                $fatal(1, "simulation timed out");

            if (m_valid && m_ready) begin
                if ($signed(m_data) !== output_count - 8)
                    $fatal(1, "output %0d was %0d, expected %0d",
                        output_count, $signed(m_data), output_count - 8);
                if (m_last !== (output_count == VALUE_DIM - 1))
                    $fatal(1, "TLAST incorrect for output %0d", output_count);
                output_count <= output_count + 1;
                if (output_count == VALUE_DIM - 1) begin
                    $display("PASS: fixed-shape AXI attention stream produced all expected outputs");
                    $finish;
                end
            end
        end
    end

    initial begin
        repeat (5) @(negedge clk);
        resetn = 1;

        for (integer q = 0; q < QUERY_DIM; q = q + 1)
            send_byte(0);

        for (integer token = 0; token < TOKEN_COUNT; token = token + 1) begin
            for (integer k = 0; k < QUERY_DIM; k = k + 1)
                send_byte(0);
            for (integer v = 0; v < VALUE_DIM; v = v + 1)
                send_byte(v - 8);
        end
    end
endmodule
