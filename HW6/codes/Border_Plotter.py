import tensorflow as tf
import gym
import os
from sys import getsizeof
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

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





agent = Agent(0)
game = gym.make("CartPole-v0").env
agent.load()

sample_number = 10

def first_plot():
    # Pole Angle vs Pole Velocity At tip



    X = []
    Y = []
    Z = []

    for i in np.arange(-30, 30, 1):
        for j in np.arange(-30, 30, 1):
            obs = np.array((0, 0, i, j))
            z = 0
            for _ in range(sample_number):
                tmp, _ = agent.get_action_and_gradients(obs)
                z += tmp
            #z = z / sample_number
            X.append(i)
            Y.append(j)
            Z.append(z)

    fig, ax = plt.subplots()
    cmap = matplotlib.cm.cool
    norm = matplotlib.colors.Normalize(vmin=0, vmax=sample_number)
    ax.scatter(x=X, y=Y, c=Z, s=10, cmap=cmap, norm=norm)
    plt.xlabel('Pole Angle')
    plt.ylabel('Pole Velocity At tip')
    plt.title("Border of action (Blue left , Red Right)")
    plt.show()




def second_plot():
    # Pole Angle vs Pole Velocity At tip
    X = []
    Y = []
    Z = []

    for i in np.arange(-30, 30, 1):
        for j in np.arange(-30, 30, 1):
            obs = np.array((0, i, j, 0))
            z = 0
            for _ in range(sample_number):
                tmp, _ = agent.get_action_and_gradients(obs)
                z += tmp
            #z = z / sample_number
            X.append(i)
            Y.append(j)
            Z.append(z)

    fig, ax = plt.subplots()
    cmap = matplotlib.cm.cool
    norm = matplotlib.colors.Normalize(vmin=0, vmax=sample_number)
    ax.scatter(x=X, y=Y, c=Z, s=10, cmap=cmap, norm=norm)
    plt.xlabel('Cart Velocity')
    plt.ylabel('Pole Angle')
    plt.title("Border of action (Blue left , Red Right)")
    plt.show()



def third_plot():
    # Pole Angle vs Pole Velocity At tip
    X = []
    Y = []
    Z = []

    for i in np.arange(-30, 30, 1):
        for j in np.arange(-30, 30, 1):
            obs = np.array((0, i, 0, j))
            z = 0
            for _ in range(sample_number):
                tmp, _ = agent.get_action_and_gradients(obs)
                z += tmp
            #z = z / sample_number
            X.append(i)
            Y.append(j)
            Z.append(z)

    fig, ax = plt.subplots()
    cmap = matplotlib.cm.cool
    norm = matplotlib.colors.Normalize(vmin=0, vmax=sample_number)
    ax.scatter(x=X, y=Y, c=Z, s=10,cmap=cmap,norm=norm)
    plt.xlabel('Cart Velocity')
    plt.ylabel('Pole Velocity At tip')
    plt.title("Border of action (Blue left , Red Right)")
    plt.show()


first_plot()
second_plot()
third_plot()