import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA = 0.9 #discount factor
INITIAL_EPSILON = 0.9 # starting value of epsilon
FINAL_EPSILON =  0.1 # final value of epsilon
EPSILON_DECAY_STEPS = 10 # decay period

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# TODO: Define Network Graph
# Network Parameters:
REWARD_DIM = 1
DONE_DIM = 1
LEARNING_RATE = 0.01
HIDDEN_UNITS = 20


# Build Eval Network:
def network(state_in):

    # weights & biases for input layer and output layer
    weights = {

        'input_layer': tf.Variable(tf.random_normal([STATE_DIM, HIDDEN_UNITS])),

        'output_layer': tf.Variable(tf.random_normal([HIDDEN_UNITS, ACTION_DIM]))
    }
    biases = {

        'input_layer': tf.Variable(tf.constant(0.1, shape=[1, HIDDEN_UNITS])),

        'output_layer': tf.Variable(tf.constant(0.1, shape=[1, ACTION_DIM]))
    }

    # Fully connected layers
    input_layer = tf.nn.relu(tf.matmul(state_in, weights['input_layer']) + biases['input_layer'])
    output_layer = tf.nn.relu(tf.matmul(input_layer, weights['output_layer']) + biases['output_layer'])

    return output_layer

# TODO: Network outputs
q_values = network(state_in)
q_action = tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)
# Duplicate Eval Network -> Target Network
q_target = tf.identity(q_values)


# TODO: Loss/Optimizer Definition
loss = tf.reduce_sum(tf.square(target_in - q_action))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)


# Memory Parameters:
MEMORY_SIZE = 20000
MEMORY = np.zeros((MEMORY_SIZE, 2*STATE_DIM + ACTION_DIM + 2))

Memeory_counter = 0
Replace_target_iter = 12
Batch_ratio = 0.3
Batch_size = round(Batch_ratio * MEMORY_SIZE)


def Fill_Memory(s, a, r, s_, done):

    global Memeory_counter

    # construct memory slot [s,a,r,s_, done] -> [[s,a,r,s_, done]]
    New_Memory_Slot = np.expand_dims(np.hstack((s, a, r, s_, done)), axis = 0)

    # build memory
    index = Memeory_counter % MEMORY_SIZE
    MEMORY[index,:] = New_Memory_Slot
    Memeory_counter += 1

    return any(MEMORY[-1])

def Build_Train_Batch():

    if Memeory_counter > Batch_size:
        sample_index = np.random.choice(MEMORY_SIZE, Batch_size, replace = False)
    else:
        sample_index = np.random.choice(Batch_size, Batch_size, replace = False)
    batch_memory = MEMORY[sample_index, :]

    s = batch_memory[:, : STATE_DIM]
    a = batch_memory[:, STATE_DIM: STATE_DIM + ACTION_DIM]
    r = batch_memory[:, STATE_DIM + ACTION_DIM : STATE_DIM + ACTION_DIM + REWARD_DIM]
    s_ = batch_memory[:, STATE_DIM + ACTION_DIM + REWARD_DIM : 2*STATE_DIM + ACTION_DIM + REWARD_DIM]
    done = batch_memory[:, -1:]

    return s, a, r, s_, done


# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action


# Main learning loop
for episode in range(EPISODE):

    total_step = 0
    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= epsilon / EPSILON_DECAY_STEPS

    if(epsilon < FINAL_EPSILON):
        epsilon = FINAL_EPSILON

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))

        full = Fill_Memory(state, action, reward, next_state, int(done))

        # train model in first 100 episodes
        if episode<101:
            #start learning
            state_batch, action_batch, reward_batch, next_state_batch, done_batch= Build_Train_Batch()

            nextstate_q_values = q_target.eval(feed_dict={
                state_in: next_state_batch
            })

            # TODO: Calculate the target q-value.
            target_batch = reward_batch + (1 - done_batch) * GAMMA * np.max(nextstate_q_values, axis = 1, keepdims = True)

            target = target_batch.squeeze()

            # Batch training
            session.run([optimizer], feed_dict={
                target_in: target,
                action_in: action_batch,
                state_in: state_batch
            })

            if total_step % Replace_target_iter == 0:
                q_target = tf.identity(q_values)

        total_step += 1
        # Update
        state = next_state
        if done:
            print('ep',episode,'step',step)
            break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()