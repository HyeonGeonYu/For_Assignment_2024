import numpy as np
import matplotlib.pyplot as plt
# Hamming(7,4) code parameters
n = 7  # Code length
k = 4  # Message length

# Generator matrix for Hamming(7,4) code
G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]
])

# Parity-check matrix for Hamming(7,4) code
H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1]
])

# Function to encode a message using Hamming(7,4) code
def encode_hamming(message):
    return np.dot(message, G) % 2

# Function to simulate BSC (Binary Symmetric Channel)
def bsc(channel_input, error_prob):
    return (channel_input + np.random.binomial(1, error_prob, size=channel_input.shape)) % 2

# Placeholder function to decode using Belief Propagation
def belief_propagation(channel_output, max_iterations=10):
    message
    encoded_message
    channel_output
    num_checks, num_bits = H.shape
    Lr = np.zeros((num_checks, num_bits))
    Lq = np.zeros((num_checks, num_bits))

    for i in range(num_checks):
        for j in range(num_bits):
            if H[i, j] == 1:
                Lq[i, j] = 2 * (channel_output[j] - 0.5)

    for _ in range(max_iterations):
        for i in range(num_checks):
            for j in range(num_bits):
                if H[i, j] == 1:
                    Lq_no_j = np.delete(Lq[i, :], j)
                    Lr[i, j] = np.prod(np.tanh(Lq_no_j / 2))

        for i in range(num_checks):
            for j in range(num_bits):
                if H[i, j] == 1:
                    Lq[i, j] = 2 * (channel_output[j] - 0.5) + np.sum(Lr[:, j]) - Lr[i, j]

    L_final = 2 * (channel_output - 0.5) + np.sum(Lr, axis=0)
    decoded_codeword = (L_final >= 0).astype(int)

    # Extract original 4-bit message from decoded 7-bit codeword
    decoded_message = decoded_codeword[:k]
    return decoded_message
# Placeholder function to decode using Maximum Likelihood
def ml_decoding(received):
    num_codewords = 2**k
    min_distance = float('inf')
    decoded_message = np.zeros(k)

    for i in range(num_codewords):
        message = np.array([int(x) for x in np.binary_repr(i, width=k)])
        codeword = encode_hamming(message)
        distance = np.sum(received != codeword)
        if distance < min_distance:
            min_distance = distance
            decoded_message = message

    return decoded_message

def syndrome_decoding(received):
    num_checks, num_bits = H.shape
    decoded = received.copy()
    syndrome = np.dot(decoded, H.T) % 2
    syndrome = syndrome.reshape(1, -1)

    for _ in range(num_checks):
        syndrome = np.dot(decoded, H.T) % 2
        syndrome = syndrome.reshape(1, -1)

        if np.sum(syndrome) == 0:
            break

        error_location = np.where((H == np.tile(syndrome.T, (1, H.shape[1]))).all(axis=0))[0][0]
        decoded[error_location] = (decoded[error_location] + 1) % 2

    return decoded[:4]

# Function to calculate BER (Bit Error Rate)
def calculate_ber(original_message, decoded_message):
    return np.sum(original_message != decoded_message) / len(original_message)

def binomial_coefficient(n, k):
    from math import factorial as f
    return f(n) // f(k) // f(n - k)

def binomial_probability(n, k, p):
    return binomial_coefficient(n, k) * (p ** k) * ((1 - p) ** (n - k))


BLER_BP_list = []
BLER_ML_list = []
error_prob_list = np.linspace(0.05, 0.5, num=10)
# Number of transmissions
num_transmissions = 1000

for error_prob in error_prob_list:
    E_num_ML = 0
    for _ in range(num_transmissions):
        # Generate random 4-bit message
        message = np.random.randint(0, 2, size=k)
        encoded_message = encode_hamming(message)

        # Simulate channel transmission
        # received_bsc =encoded_message
        received_bsc = bsc(encoded_message, error_prob)

        # Decode messages
        decoded_BP = belief_propagation(received_bsc)
        decoded_ML = syndrome_decoding(received_bsc)


        # Calculate BLER
        #BLER_BP = calculate_ber(message, decoded_BP)
        if not np.array_equal(message, decoded_ML):
            E_num_ML += 1
    #average_ber_bp = np.mean(ber_BP_list)
    average_BLER_ML = E_num_ML/num_transmissions

    #average_ber_BP_list.append(average_ber_bp)
    BLER_ML_list.append(average_BLER_ML)

prob_error_one_or_less = [(1-(binomial_probability(7, 0, p) + binomial_probability(7, 1, p))) for p in error_prob_list]

plt.figure(figsize=(10, 6))

# Plotting BLER for Belief Propagation
# plt.plot(error_prob_list, average_ber_BP_list, marker='o', linestyle='-', color='b', label='Belief Propagation')

# Plotting BLER for Maximum Likelihood
plt.plot(error_prob_list, BLER_ML_list, marker='s', linestyle='-', color='r', label='Maximum Likelihood')
plt.plot(error_prob_list, prob_error_one_or_less, marker='v', linestyle='-', color='m',
         label='2bit or more errors Probablity')
plt.yscale('log')
plt.title('BLER vs Error Probability')
plt.xlabel('Error Probability')
plt.ylabel('BLER')
plt.legend()
plt.grid(True,which="both", linestyle='--')
plt.savefig("ML_BP result.png", dpi=300)
plt.show()
