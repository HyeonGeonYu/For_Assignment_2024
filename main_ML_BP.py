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

def calculate_probability_one(selected_values):
    # Perform the calculations as per the formula
    # 100 010 001 111 #1
    result = sum(
        [np.prod([selected_values[0], 1 - selected_values[1], 1 - selected_values[2]]) * np.prod(selected_values),
         np.prod([1 - selected_values[0], selected_values[1], 1 - selected_values[2]]) * np.prod(selected_values),
         np.prod([1 - selected_values[0], 1 - selected_values[1], selected_values[2]]) * np.prod(selected_values),
         np.prod([selected_values[0], selected_values[1], selected_values[2]]) * np.prod(selected_values)])

    return result

def calculate_probability_zero(selected_values):
    # Perform the calculations as per the formula
    # 110 101 011 000 #0
    result = sum([np.prod([selected_values[0], selected_values[1], 1 - selected_values[2]]) * np.prod(selected_values),
         np.prod([selected_values[0], 1 - selected_values[1], selected_values[2]]) * np.prod(selected_values),
         np.prod([1 - selected_values[0], selected_values[1], selected_values[2]]) * np.prod(selected_values),
         np.prod([1 - selected_values[0], 1 - selected_values[1], 1 - selected_values[2]]) * np.prod(selected_values)]
        )
    return result

# Placeholder function to decode using Belief Propagation
def belief_propagation(channel_output, error_prob, max_iterations=10):
    message
    encoded_message
    channel_output
    num_checks, num_bits = H.shape
    decoded_bit =[]
    channel_output_probabilities = np.where(channel_output == 1, 0.95, 0.05) #1일 확률
    mu = np.log(channel_output_probabilities/(1-channel_output_probabilities))
    H * mu


    selected_values = channel_output_probabilities[[1,3,4]]
    a1 = calculate_probability_one(selected_values)
    b1  = calculate_probability_zero(selected_values)

    selected_values = channel_output_probabilities[[2, 3, 5]]
    a2 = calculate_probability_one(selected_values)
    # 110 101 011 000 #0
    b2 = calculate_probability_zero(selected_values)
    if a1+a2 > b1+b2:
        decoded_bit.append(1)
    else:
        decoded_bit.append(0)

    selected_values = channel_output_probabilities[[0, 3, 4]]
    a1 = calculate_probability_one(selected_values)
    b1 = calculate_probability_zero(selected_values)

    selected_values = channel_output_probabilities[[2, 3, 6]]
    a2 = calculate_probability_one(selected_values)
    # 110 101 011 000 #0
    b2 = calculate_probability_zero(selected_values)
    if a1 + a2 > b1 + b2:
        decoded_bit.append(1)
    else:
        decoded_bit.append(0)

    selected_values = channel_output_probabilities[[0, 3, 5]]
    a1 = calculate_probability_one(selected_values)
    b1 = calculate_probability_zero(selected_values)

    selected_values = channel_output_probabilities[[1, 3, 6]]
    a2 = calculate_probability_one(selected_values)
    # 110 101 011 000 #0
    b2 = calculate_probability_zero(selected_values)
    if a1 + a2 > b1 + b2:
        decoded_bit.append(1)
    else:
        decoded_bit.append(0)

    selected_values = channel_output_probabilities[[0, 1, 4]]
    a1 = calculate_probability_one(selected_values)
    b1 = calculate_probability_zero(selected_values)

    selected_values = channel_output_probabilities[[0, 2, 5]]
    a2 = calculate_probability_one(selected_values)
    # 110 101 011 000 #0
    b2 = calculate_probability_zero(selected_values)
    if a1 + a2 > b1 + b2:
        decoded_bit.append(1)
    else:
        decoded_bit.append(0)
    return decoded_bit
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
num_transmissions = 100000

for error_prob in error_prob_list:
    E_num_ML = 0
    E_num_BP = 0
    for _ in range(num_transmissions):
        # Generate random 4-bit message
        message = np.random.randint(0, 2, size=k)
        encoded_message = encode_hamming(message)

        # Simulate channel transmission
        # received_bsc =encoded_message
        received_bsc = bsc(encoded_message, error_prob)

        # Decode messages
        decoded_BP = belief_propagation(received_bsc,error_prob)
        decoded_ML = syndrome_decoding(received_bsc)
        # Calculate BLER
        #BLER_BP = calculate_ber(message, decoded_BP)
        if not np.array_equal(message, decoded_BP):
            E_num_BP += 1
        if not np.array_equal(message, decoded_ML):
            E_num_ML += 1

    #average_ber_bp = np.mean(ber_BP_list)
    average_BLER_BP = E_num_BP/num_transmissions
    average_BLER_ML = E_num_ML/num_transmissions

    #average_ber_BP_list.append(average_ber_bp)
    BLER_BP_list.append(average_BLER_BP)
    BLER_ML_list.append(average_BLER_ML)

prob_error_one_or_less = [(1-(binomial_probability(7, 0, p) + binomial_probability(7, 1, p))) for p in error_prob_list]

plt.figure(figsize=(10, 6))

# Plotting BLER for Belief Propagation


# Plotting BLER for Maximum Likelihood
plt.plot(error_prob_list, BLER_ML_list, marker='s', linestyle='-', color='r', label='Maximum Likelihood')
plt.plot(error_prob_list, BLER_BP_list, marker='o', linestyle='-', color='b', label='Belief Propagation')
plt.plot(error_prob_list, prob_error_one_or_less, marker='v', linestyle='-', color='m',
         label='2bit or more errors Probablity')
plt.yscale('log')
plt.title('BLER vs Error Probability')
plt.xlabel('Error Probability')
plt.ylabel('BLER')
plt.legend()
plt.grid(True,which="both", linestyle='--')
plt.savefig("ML_BP_result.png", dpi=300)
plt.show()
