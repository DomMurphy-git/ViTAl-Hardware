module streaming_attention_axis (
    input  logic        s_axis_aclk,
    input  logic        s_axis_aresetn,
    input  logic [31:0] s_axis_tdata,
    input  logic        s_axis_tvalid,
    output logic        s_axis_tready,
    input  logic        s_axis_tlast,
    output logic [31:0] m_axis_tdata,
    output logic        m_axis_tvalid,
    input  logic        m_axis_tready,
    output logic        m_axis_tlast
);
    localparam integer QUERY_DIM = 32;
    localparam integer VALUE_DIM = 16;
    localparam integer TOKEN_COUNT = 196;
    localparam logic [16:0] Q15_ONE = 17'd32768;

    typedef enum logic [2:0] {
        READ_QUERY,
        READ_KEY,
        READ_VALUE,
        UPDATE_STATE,
        WRITE_OUTPUT
    } state_t;

    state_t state;
    logic signed [7:0] query_mem [0:QUERY_DIM-1];
    logic signed [7:0] key_mem [0:QUERY_DIM-1];
    logic signed [7:0] value_mem [0:VALUE_DIM-1];
    logic signed [47:0] accumulator [0:VALUE_DIM-1];
    logic signed [31:0] maximum_score;
    logic [31:0] denominator_q15;

    logic [4:0] query_index;
    logic [4:0] key_index;
    logic [4:0] value_index;
    logic [7:0] token_index;
    logic [4:0] output_index;

    logic signed [15:0] dot_terms [0:QUERY_DIM-1];
    logic signed [31:0] dot_sum;
    logic signed [47:0] scaled_score_product;
    logic signed [31:0] score_q12;
    logic signed [31:0] new_maximum;
    logic [31:0] delta_old_maximum;
    logic [31:0] delta_current_score;
    logic [16:0] alpha_q15;
    logic [16:0] beta_q15;
    logic [63:0] denominator_product;
    logic signed [63:0] signed_denominator;
    logic signed [63:0] accumulator_product [0:VALUE_DIM-1];
    logic signed [31:0] value_product [0:VALUE_DIM-1];
    logic signed [63:0] normalized_value;

    function automatic logic [16:0] exp_neg_q15(input logic [31:0] delta_q12);
        logic [63:0] exponent_product;
        logic [63:0] exponent_units;
        logic [16:0] lut_value;
        logic [6:0] lut_index;
        integer integer_exponent;
        integer shift_amount;
        begin
            exponent_product = delta_q12 * 64'd24204406;
            exponent_units = exponent_product >> 30;
            integer_exponent = exponent_units >> 6;

            if (exponent_units[5:0] == 0) begin
                lut_index = 0;
                shift_amount = integer_exponent;
            end else begin
                lut_index = 7'd64 - {1'b0, exponent_units[5:0]};
                shift_amount = integer_exponent + 1;
            end

            case (lut_index)
                7'd0: lut_value = 17'd32768;
                7'd1: lut_value = 17'd33125;
                7'd2: lut_value = 17'd33486;
                7'd3: lut_value = 17'd33850;
                7'd4: lut_value = 17'd34219;
                7'd5: lut_value = 17'd34591;
                7'd6: lut_value = 17'd34968;
                7'd7: lut_value = 17'd35349;
                7'd8: lut_value = 17'd35734;
                7'd9: lut_value = 17'd36123;
                7'd10: lut_value = 17'd36516;
                7'd11: lut_value = 17'd36914;
                7'd12: lut_value = 17'd37316;
                7'd13: lut_value = 17'd37722;
                7'd14: lut_value = 17'd38133;
                7'd15: lut_value = 17'd38548;
                7'd16: lut_value = 17'd38968;
                7'd17: lut_value = 17'd39392;
                7'd18: lut_value = 17'd39821;
                7'd19: lut_value = 17'd40255;
                7'd20: lut_value = 17'd40693;
                7'd21: lut_value = 17'd41136;
                7'd22: lut_value = 17'd41584;
                7'd23: lut_value = 17'd42037;
                7'd24: lut_value = 17'd42495;
                7'd25: lut_value = 17'd42958;
                7'd26: lut_value = 17'd43425;
                7'd27: lut_value = 17'd43898;
                7'd28: lut_value = 17'd44376;
                7'd29: lut_value = 17'd44859;
                7'd30: lut_value = 17'd45348;
                7'd31: lut_value = 17'd45842;
                7'd32: lut_value = 17'd46341;
                7'd33: lut_value = 17'd46846;
                7'd34: lut_value = 17'd47356;
                7'd35: lut_value = 17'd47871;
                7'd36: lut_value = 17'd48393;
                7'd37: lut_value = 17'd48920;
                7'd38: lut_value = 17'd49452;
                7'd39: lut_value = 17'd49991;
                7'd40: lut_value = 17'd50535;
                7'd41: lut_value = 17'd51085;
                7'd42: lut_value = 17'd51642;
                7'd43: lut_value = 17'd52204;
                7'd44: lut_value = 17'd52773;
                7'd45: lut_value = 17'd53347;
                7'd46: lut_value = 17'd53928;
                7'd47: lut_value = 17'd54515;
                7'd48: lut_value = 17'd55109;
                7'd49: lut_value = 17'd55709;
                7'd50: lut_value = 17'd56316;
                7'd51: lut_value = 17'd56929;
                7'd52: lut_value = 17'd57549;
                7'd53: lut_value = 17'd58176;
                7'd54: lut_value = 17'd58809;
                7'd55: lut_value = 17'd59449;
                7'd56: lut_value = 17'd60097;
                7'd57: lut_value = 17'd60751;
                7'd58: lut_value = 17'd61413;
                7'd59: lut_value = 17'd62081;
                7'd60: lut_value = 17'd62757;
                7'd61: lut_value = 17'd63441;
                7'd62: lut_value = 17'd64132;
                7'd63: lut_value = 17'd64830;
                default: lut_value = 17'd0;
            endcase

            if (shift_amount >= 17)
                exp_neg_q15 = 17'd0;
            else
                exp_neg_q15 = lut_value >> shift_amount;
        end
    endfunction

    always_comb begin
        integer i;
        dot_sum = 0;
        for (i = 0; i < QUERY_DIM; i = i + 1)
            dot_sum = dot_sum + {{16{dot_terms[i][15]}}, dot_terms[i]};

        scaled_score_product = $signed(dot_sum) * 48'sd46341;
        if (scaled_score_product >= 0)
            score_q12 = (scaled_score_product + 48'sd8192) >>> 14;
        else
            score_q12 = -(((-scaled_score_product) + 48'sd8192) >>> 14);

        if (score_q12 > maximum_score)
            new_maximum = score_q12;
        else
            new_maximum = maximum_score;

        delta_old_maximum = $unsigned(new_maximum - maximum_score);
        delta_current_score = $unsigned(new_maximum - score_q12);
        alpha_q15 = exp_neg_q15(delta_old_maximum);
        beta_q15 = exp_neg_q15(delta_current_score);
        denominator_product =
            {32'd0, denominator_q15} * {47'd0, alpha_q15};
        signed_denominator = $signed({32'd0, denominator_q15});

        for (i = 0; i < VALUE_DIM; i = i + 1) begin
            accumulator_product[i] =
                $signed(accumulator[i]) * $signed({1'b0, alpha_q15});
            value_product[i] =
                $signed(value_mem[i]) * $signed({1'b0, beta_q15});
        end

        normalized_value = 0;
        if (denominator_q15 != 0) begin
            if (accumulator[output_index] >= 0)
                normalized_value =
                    (accumulator[output_index] + (signed_denominator >>> 1))
                    / signed_denominator;
            else
                normalized_value =
                    -(((-accumulator[output_index]) + (signed_denominator >>> 1))
                    / signed_denominator);
        end
        if (normalized_value > 127)
            normalized_value = 127;
        else if (normalized_value < -128)
            normalized_value = -128;
    end

    always_comb begin
        integer i;
        for (i = 0; i < QUERY_DIM; i = i + 1)
            dot_terms[i] =
                $signed({{8{query_mem[i][7]}}, query_mem[i]})
                * $signed({{8{key_mem[i][7]}}, key_mem[i]});

        s_axis_tready =
            (state == READ_QUERY) || (state == READ_KEY) || (state == READ_VALUE);
        m_axis_tvalid = (state == WRITE_OUTPUT);
        m_axis_tdata = {{24{normalized_value[7]}}, normalized_value[7:0]};
        m_axis_tlast = (state == WRITE_OUTPUT) && (output_index == VALUE_DIM - 1);
    end

    always_ff @(posedge s_axis_aclk) begin
        if (!s_axis_aresetn) begin
            state <= READ_QUERY;
            query_index <= 0;
            key_index <= 0;
            value_index <= 0;
            token_index <= 0;
            output_index <= 0;
            maximum_score <= 0;
            denominator_q15 <= 0;
            for (integer i = 0; i < QUERY_DIM; i = i + 1) begin
                query_mem[i] <= 0;
                key_mem[i] <= 0;
            end
            for (integer j = 0; j < VALUE_DIM; j = j + 1) begin
                value_mem[j] <= 0;
                accumulator[j] <= 0;
            end
        end else begin
            case (state)
                READ_QUERY: begin
                    if (s_axis_tvalid && s_axis_tready) begin
                        query_mem[query_index] <= $signed(s_axis_tdata[7:0]);
                        if (query_index == QUERY_DIM - 1) begin
                            query_index <= 0;
                            state <= READ_KEY;
                        end else begin
                            query_index <= query_index + 1'b1;
                        end
                    end
                end

                READ_KEY: begin
                    if (s_axis_tvalid && s_axis_tready) begin
                        key_mem[key_index] <= $signed(s_axis_tdata[7:0]);
                        if (key_index == QUERY_DIM - 1) begin
                            key_index <= 0;
                            value_index <= 0;
                            state <= READ_VALUE;
                        end else begin
                            key_index <= key_index + 1'b1;
                        end
                    end
                end

                READ_VALUE: begin
                    if (s_axis_tvalid && s_axis_tready) begin
                        value_mem[value_index] <= $signed(s_axis_tdata[7:0]);
                        if (value_index == VALUE_DIM - 1) begin
                            value_index <= 0;
                            state <= UPDATE_STATE;
                        end else begin
                            value_index <= value_index + 1'b1;
                        end
                    end
                end

                UPDATE_STATE: begin
                    if (token_index == 0) begin
                        maximum_score <= score_q12;
                        denominator_q15 <= Q15_ONE;
                        for (integer j = 0; j < VALUE_DIM; j = j + 1)
                            accumulator[j] <= $signed(value_mem[j]) * 48'sd32768;
                    end else begin
                        maximum_score <= new_maximum;
                        denominator_q15 <=
                            (denominator_product >> 15) + beta_q15;
                        for (integer j = 0; j < VALUE_DIM; j = j + 1)
                            accumulator[j] <=
                                (accumulator_product[j] >>> 15) + value_product[j];
                    end

                    if (token_index == TOKEN_COUNT - 1) begin
                        output_index <= 0;
                        state <= WRITE_OUTPUT;
                    end else begin
                        token_index <= token_index + 1'b1;
                        state <= READ_KEY;
                    end
                end

                WRITE_OUTPUT: begin
                    if (m_axis_tvalid && m_axis_tready) begin
                        if (output_index == VALUE_DIM - 1) begin
                            state <= READ_QUERY;
                            query_index <= 0;
                            key_index <= 0;
                            value_index <= 0;
                            token_index <= 0;
                            output_index <= 0;
                            maximum_score <= 0;
                            denominator_q15 <= 0;
                        end else begin
                            output_index <= output_index + 1'b1;
                        end
                    end
                end

                default: state <= READ_QUERY;
            endcase
        end
    end

    logic unused_tlast;
    always_comb unused_tlast = s_axis_tlast;
endmodule
