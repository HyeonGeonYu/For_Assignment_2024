import numpy as np
import matplotlib.pyplot as plt


def convolutional_encode(bits, G1, G2):
    # Initialize shift registers
    num_states = 2 ** (len(G1) - 1)
    state = 0

    encoded_bits = []

    for bit in bits:
        # Shift in the new bit
        state_bin_str = bin(state)[2:].zfill(int(np.log2(num_states)))
        # Insert the next bit to the left.
        next_state_bin_str = str(bit) + state_bin_str[:-1]
        # Convert the binary string back to an integer.
        state = int(next_state_bin_str, 2)

        output1 = np.mod(np.sum(np.array([int(i) for i in state_bin_str]) * G1[1:]) + G1[0] * bit, 2)
        output2 = np.mod(np.sum(np.array([int(i) for i in state_bin_str]) * G2[1:]) + G2[0] * bit, 2)

        # Append the outputs to the encoded sequence
        encoded_bits.extend([output1, output2])

    return np.array(encoded_bits)


def bcjr_decode(received_value, G1, G2, snr):
    # 상태 수 및 정보 비트 수
    num_states = 2 ** (len(G1) - 1)
    num_bits = len(received_value) // 2

    # 전방 확률 초기화
    alpha = np.zeros((num_bits + 1, num_states))
    alpha[0, 0] = 1  # 초기 상태

    # 후방 확률 초기화
    beta = np.zeros((num_bits + 1, num_states))
    beta[-1, 0] = 1  # 종단 상태

    # [time, state', inp_bit]
    gamma = np.zeros((num_bits, num_states, 2))
    received_value_reshaped = received_value.reshape(-1, 2).copy()
    # calculation gamma
    for t in range(num_bits):
        for state in range(num_states):
            for bit in range(2):
                state_bin_str = bin(state)[2:].zfill(int(np.log2(num_states)))
                # Insert the next bit to the left.
                next_state_bin_str = str(bit) + state_bin_str[:-1]
                # Convert the binary string back to an integer.
                next_state = int(next_state_bin_str, 2)
                output1 = np.mod(np.sum((np.array([bit] + list(map(int, bin(state)[2:].zfill(len(G1) - 1)))) * G1)), 2)
                output2 = np.mod(np.sum((np.array([bit] + list(map(int, bin(state)[2:].zfill(len(G1) - 1)))) * G2)), 2)
                output1 = 1 if output1 == 1 else -1
                output2 = 1 if output2 == 1 else -1
                gamma[t,state,bit] = np.exp(snr*(output1*received_value_reshaped[t,0]+output2 * received_value_reshaped[t,1]))

    nonzero_state_list = [0]
    tmp_list = list(nonzero_state_list)
    for t in range(int(np.log2(num_states))):
        for state in tmp_list:
            if t == 0:
                break
            for bit in range(2):
                state_bin_str = bin(state)[2:].zfill(int(np.log2(num_states)))
                next_state_bin_str = str(bit) + state_bin_str[:-1]
                next_state = int(next_state_bin_str, 2)
                nonzero_state_list.append(next_state)
        nonzero_state_list_np = np.array(nonzero_state_list)
        all_states = np.arange(num_states)
        zero_state_list = np.setdiff1d(all_states, nonzero_state_list_np)
        gamma[t, zero_state_list] = 0
        tmp_list = list(nonzero_state_list)
        nonzero_state_list = []

    nonzero_state_list = []
    zero_state_list = []
    tmp_list = list(np.arange(num_states))
    bit = 0
    for t in range(-int(np.log2(num_states)), 0):
        gamma[t, :, 1] = 0
        gamma[t, zero_state_list] = 0
        for state in tmp_list:
            state_bin_str = bin(state)[2:].zfill(int(np.log2(num_states)))
            next_state_bin_str = str(bit) + state_bin_str[:-1]
            next_state = int(next_state_bin_str, 2)
            nonzero_state_list.append(next_state)
        nonzero_state_list_np = np.array(nonzero_state_list)
        all_states = np.arange(num_states)
        zero_state_list = np.setdiff1d(all_states, nonzero_state_list_np)
        tmp_list = list(nonzero_state_list)
        nonzero_state_list = []

    # Forward message passing
    for t in range(num_bits):
        for state in range(num_states):
            for bit in range(2):
                if t >= num_bits- int(np.log2(num_states)) and bit ==1:
                    continue
                state_bin_str = bin(state)[2:].zfill(int(np.log2(num_states)))
                # Insert the next bit to the left.
                next_state_bin_str = str(bit) + state_bin_str[:-1]
                # Convert the binary string back to an integer.
                next_state = int(next_state_bin_str, 2)
                # gamma에 t=0은 gamma_1()을 의미함
                alpha[t + 1, next_state] += alpha[t, state] * gamma[t,state,bit]
        alpha[t+1] = alpha[t+1] / alpha[t+1].sum()

    # Backward message passing
    for t in range(num_bits - 1, -1, -1):
        for state in range(num_states):
            for bit in range(2):
                if t >= num_bits- int(np.log2(num_states)) and bit ==1:
                    continue
                state_bin_str = bin(state)[2:].zfill(int(np.log2(num_states)))
                # Insert the next bit to the left.
                next_state_bin_str = str(bit) + state_bin_str[:-1]
                # Convert the binary string back to an integer.
                next_state = int(next_state_bin_str, 2)
                beta[t, state] += beta[t+1, next_state] * gamma[t,state,bit]
        beta[t] = beta[t] / beta[t].sum()

    decoded_bits = []
    for t in range(num_bits):
        pp = np.zeros(2)
        for state in range(num_states):
            for bit in range(2):
                if t >= num_bits- int(np.log2(num_states)) and bit ==1:
                    continue
                state_bin_str = bin(state)[2:].zfill(int(np.log2(num_states)))
                # Insert the next bit to the left.
                next_state_bin_str = str(bit) + state_bin_str[:-1]
                # Convert the binary string back to an integer.
                next_state = int(next_state_bin_str, 2)
                # gamma에 t=0은 gamma_1()을 의미함
                pp[bit] += alpha[t,state] * gamma[t, state, bit] * beta[t+1,next_state]
        if pp[0] > 0:
            LLR = np.log((pp[1]+1e-10) / (pp[0]+1e-10))
        else:
            LLR = np.inf  # or any value that makes sense in your context

        if LLR>0:
            decoded_bits.append(1)
        else:
            decoded_bits.append(0)
    return decoded_bits


# Define the generating polynomials
G1 = [1, 0, 1, 1]
G2 = [1, 1, 1, 1]

# G1 = [1,1,1]
# G2 = [1,0,1]

# Information bits
bit_num = 64
snr_db_arr = np.linspace(0.00, 5, num=11)
ber_list = []
num_transmissions = 100000
for snr_db  in snr_db_arr:
    total_ber = 0
    for _ in range(num_transmissions):
        info_bits = np.random.randint(0, 2, bit_num)
        # 초기 state로 보내기 위해 0을 메모리 갯수만큼 추가
        info_bits = np.append(info_bits, (len(G1) - 1) * [0])
        snr = 10 ** (snr_db / 10)
        std_dev = np.sqrt(1 / (snr))
        encoded_bits = convolutional_encode(info_bits, G1, G2)

        BPSK_result = np.where(encoded_bits == 1, 1, -1)
        received_signal = BPSK_result +np.random.normal(0, std_dev, size=BPSK_result.shape)
        decoded_bits = bcjr_decode(received_signal, G1, G2,snr)

        info_bits = info_bits[:-(len(G1) - 1)]
        decoded_bits = decoded_bits[:-(len(G1) - 1)]
        bit_errors = np.sum(info_bits != decoded_bits)
        total_bits = info_bits.size
        ber = bit_errors / total_bits
        total_ber += ber
    avg_ber = total_ber/num_transmissions
    print(snr_db , avg_ber)
    ber_list.append(avg_ber)
plt.figure(figsize=(10, 6))
# Plotting BER for Belief Propagation
plt.plot(snr_db_arr, ber_list, marker='o', linestyle='-', color='b', label='BCJR, 1/2-rate convolution code')

plt.xticks(snr_db_arr)
plt.yscale('log')
plt.title('BER vs SNR')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.legend()
plt.grid(True,which="both", linestyle='--')
plt.ylim([10 ** -5, 1])
plt.savefig("BCJR_result.png", dpi=300)
plt.show()
