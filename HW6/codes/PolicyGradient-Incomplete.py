import tensorflow as tf
import gym
import os
from sys import getsizeof
import numpy as np

# You must add few lines of code and change all -1s

class Agent:
    def __init__(self, learning_rate,hl1=None,hl2=None):
        # Build the network to predict the correct action
        tf.reset_default_graph()
        input_dimension = 4
        hidden_dimension = 16#####
        self.input = tf.placeholder(dtype=tf.float32, shape=[1, input_dimension], name='X')



        self.hidden_layer_1_variable = tf.Variable(initial_value=tf.random.normal(shape=(input_dimension,hidden_dimension),mean=0.0,stddev=1.0))
        self.hidden_layer_2_variable = tf.Variable(initial_value=tf.random.normal(shape=(hidden_dimension,2),mean=0.0,stddev=1.0))



        hidden_layer_1 = tf.nn.tanh( tf.tensordot(self.input ,self.hidden_layer_1_variable,axes=[1,0]) )

        logits =  tf.tensordot(hidden_layer_1 ,self.hidden_layer_2_variable,axes=[1,0])



        # Sample an action according to network's output
        # use tf.multinomial and sample one action from network's output



        self.action = tf.multinomial(logits=logits,num_samples=1)

        # Optimization according to policy gradient algorithm
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(self.action,depth=2),logits=logits)
        #self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # use one of tensorflow optimizers
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        grads_vars = self.optimizer.compute_gradients(
            cross_entropy)  # gradient of current action w.r.t. network's variables
        self.gradients = [grad for grad, var in grads_vars]

        # get rewards from the environment and evaluate rewarded gradients
        #  and feed it to agent and then call train operation
        self.rewarded_grads_placeholders_list = []
        rewarded_grads_and_vars = []
        for grad, var in grads_vars:
            rewarded_grad_placeholder = tf.placeholder(dtype=tf.float32, shape=grad.shape)
            self.rewarded_grads_placeholders_list.append(rewarded_grad_placeholder)
            rewarded_grads_and_vars.append((rewarded_grad_placeholder, var))

        self.train_operation = self.optimizer.apply_gradients(rewarded_grads_and_vars)

        self.saver = tf.train.Saver()

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

        self.ses = tf.Session(config=config)
        self.ses.run(tf.global_variables_initializer())

    def get_action_and_gradients(self, obs):
        action, gradients = self.ses.run((self.action, self.gradients),feed_dict={self.input:obs.reshape(1,-1)})
        # compute network's action and gradients given the observations
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)



        return action[0,0], gradients

    def train(self, rewarded_gradients):
        feed_dict = {}
        for i in range(len(rewarded_gradients)):
            feed_dict[self.rewarded_grads_placeholders_list[i]]=rewarded_gradients[i]
        # feed gradients into the placeholder and call train operation

        self.ses.run(self.train_operation,feed_dict)

    def save(self):
        self.saver.save(self.ses, "SavedModel/")

    def special_save(self,number):
        path = str(number)+'/'
        os.mkdir(path)

        self.saver.save(self.ses,path)

    def special_load(self,number):
        path=str(number)+'/'
        self.saver.restore(self.ses,path)

    def load(self):
        self.saver.restore(self.ses, "SavedModel/")


epochs = 50
max_steps_per_game = 1000
games_per_epoch = 50
discount_factor = 0.99
learning_rate = 0.01


average_reward_list=[]

agent = Agent(learning_rate)
game = gym.make("CartPole-v0").env
for epoch in range(epochs):
    #####
    #####
    epoch_rewards = [] #################
    epoch_gradients = [] ##################
    epoch_average_reward = 0 ###################
    for episode in range(games_per_epoch):
        obs = game.reset()
        step = 0
        single_episode_rewards = []
        single_episode_gradients = []
        game_over = False
        while not game_over and step < max_steps_per_game:
            step += 1
            # image = game.render(mode='rgb_array') # Call this to render game and show visual
            action, gradients = agent.get_action_and_gradients(obs)
            obs, reward, game_over, info = game.step(action)
            single_episode_rewards.append(reward)
            single_episode_gradients.append(gradients)

        epoch_rewards.append(single_episode_rewards)
        epoch_gradients.append(single_episode_gradients)
        epoch_average_reward += sum(single_episode_rewards)

    epoch_average_reward /= games_per_epoch
    print("Epoch = {}, , Average reward = {}".format(epoch, epoch_average_reward))
    # print(getsizeof(agent.hidden_layer_1_variable))
    #print getsizeof(tf.gradients)
    agent.special_save(epoch)
    average_reward_list.append(epoch_average_reward)
    print average_reward_list

    epoch_average_reward_discounted = (1-discount_factor**epoch_average_reward)/(1-discount_factor)

    normalized_rewards=[]
    for i in range(len(epoch_rewards)):
        trajectory_reward = len(epoch_rewards[i])
        trajectory_reward_discounted = (1-discount_factor**trajectory_reward)/(1-discount_factor)
        normalized_rewards.append(trajectory_reward_discounted-epoch_average_reward_discounted)



    mean_rewarded_gradients =[]

    for gradient_variable in epoch_gradients[0][0]:
        mean_rewarded_gradients.append(np.zeros(shape=gradient_variable.shape))


    for epoch_index in range(len(epoch_gradients)):
        tmp_gradient_list = []

        for gradient_variable in epoch_gradients[0][0]:
            tmp_gradient_list.append(np.zeros(shape=gradient_variable.shape))


        for time_step_gradient_list in epoch_gradients[epoch_index]:

            for gradient_variable_index in range(len(time_step_gradient_list)):
                tmp_gradient_list[gradient_variable_index]+=time_step_gradient_list[gradient_variable_index]


        for gradient_variable_index in range(len(tmp_gradient_list)):
            mean_rewarded_gradients[gradient_variable_index] += tmp_gradient_list[gradient_variable_index]*normalized_rewards[epoch_index]









    agent.train(mean_rewarded_gradients)

agent.save()
game.close()












agent = Agent(0)
game = gym.make("CartPole-v0").env
agent.load()
score = 0
for i in range(10):
    obs = game.reset()
    game_over = False
    while not game_over:
        score += 1
        image = game.render(mode='rgb_array')  # Call this to render game and show visual
        action, _ = agent.get_action_and_gradients(obs)
        obs, reward, game_over, info = game.step(action)

print("Average Score = ", score / 10)
