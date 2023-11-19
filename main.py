import numpy as np 
import torch as T
import torch.nn as nn # will handle the neural network layers
import torch.nn.functional as F 
import torch.optim as optim
import gym 


# this model will use LINEAR layers 

"""
Notes for self: 
learning rate is simply the learning rate of the neural network, its a hyper paramater used when training the network 
input_dims is just number of dimensions for the input vector
fc1_dims number of NEURONS in first layer
fc2_dims number of NEURONS in second layer
n_actions is just the idfferent actions that agent can take in the enviornment when evaluating multiple states

Episode: 
! An episode starts with an agent in some inital state in the enviornment 
! It interacts with the enviornment and takes actions, based on these actions it recives rewards and transitions to a new state
! An episode TERMINATES when maximum number of steps is reached, achieving a specific goal, or it encounters the terminal state. A terminal state is when agent cannot take anymore actions
! After the episode terminates, the agent uses its experiences from past episodes and updates itss algorithm to better perform in the next new states

"""

class DeepQNetwork(nn.Module):
    def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.learning_rate = learning_rate
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims) # maps input dimensions to first layer
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims) # neurons from first layer are being mapped to neurons in second layer
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions) # neurons from second layer are being mapped to the different actions this is OUTPUT layer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate) # the parameters() is to get all learnable paramters of network so weights and biases etc...
        # we use the adams algorithm

        # our optimizer is responsible for updating the neural networks paramaters to minimize the loss during training
        self.loss = nn.MSELoss() # used to predict how well the predictive q-learning model matches the target q-values
        # our loss function simply quantifies how well the model is performing with how the predicted q values are compared to target q values
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu' ) # use the gpu is we have it connected otherwise use the cpu


    def forward(self, x): 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions # outputs the q-values for each possible function
        

class Agent(): # our agent that interacts with the enviorment
    def __init__(self, gamma, epsilon, learning_rate, input_dimensions, batch_size, num_actions, max_mem_size=100000, eps_end=0.01, eps_dec=1e-5):
        self.gamma = gamma # discount factor 
        self.epsilon = epsilon # exploration rate
        self.learning_rate = learning_rate
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(num_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0 
        self.Q_eval = DeepQNetwork(self.learning_rate, input_dims=input_dimensions, fc1_dims=256, fc2_dims=256, n_actions=num_actions)


        # store models experiences of states, new states, actions and rewards
        self.state_memory = np.zeros((self.mem_size, input_dimensions), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dimensions), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_) 

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size # want to overwrite previous experiences so allow for memory wrap 
        # print("THIS IS STATE \n\n\n\n ", state)

        self.state_memory[index] = state[0] # stores the current state in agents memory
        self.new_state_memory[index] = new_state # stores the new state in agents memory (new state is observation of enviornment at the next time step)
        self.reward_memory[index] = reward # stores recived reward in memory
        self.action_memory[index] = action # stores the action in memory, action is relative to the CURRENT time step
        self.terminal_memory[index] = float(done) # stores whether the episode is done or not, it is a boolean value

        self.mem_cntr += 1 # just keeps track of stored experinece 


    """
    This method selects an action for agent to take based on current observation
    Has probalility epsilon where it will chose a random action
    Otherwise it explots the current q network by selection action with the highest q value from the observation

    regardless of how the action is chosen we still return teh action we have picked
    """

    def choose_action(self, observation):

        
        if isinstance(observation, np.ndarray):
            state = T.tensor([observation]).to(self.Q_eval.device)
        elif isinstance(observation, tuple):
            state = T.tensor([observation[0]]).to(self.Q_eval.device)

        state = state.view(1, -1)  # Add a batch dimension



        if np.random.random()  > self.epsilon: # exploit the current knowledge and chose highest q-value action    

            actions = self.Q_eval.forward(state) # pass state thorugh q network to compute q values for each possible action based on our current state
            action = T.argmax(actions).item() # take the action that has the highest q value
        else : # a random action is chosen
            action = np.random.choice(self.action_space)
        
        return action 
    
    def learn(self):
        
        if self.mem_cntr < self.batch_size:     # if not enough experiences then dont do anything, just continue learning
            return 
        
        self.Q_eval.optimizer.zero_grad() # reset the gradients when performing back propogation
        # we do this because we dont want out gradient from a previous batch to affect a new batch, effectively we start at a clean slate

        max_mem = min(self.mem_cntr, self.mem_size) # max num of stored experiences
        batch = np.random.choice(max_mem, self.batch_size, replace= False) # select a random batch of experiences, repalce is false to ensure that we do not sample experiences more than once

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # convert the selected batches of states, rewards, terminal flags, into tensors, then move them into device where q net work is
        state_batch = T.tensor(np.array(self.state_memory[batch])).to(self.Q_eval.device) 
        new_state_batch = T.tensor(np.array(self.new_state_memory[batch])).to(self.Q_eval.device)
        reward_batch = T.tensor(np.array(self.reward_memory[batch])).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch], dtype=T.float32).to(self.Q_eval.device)
        terminal_batch = terminal_batch.bool()


        action_batch = self.action_memory[batch] # take out batch of actions from our memory

        # move forward in our network 
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch] # only take the actions we took because we cant update our actions based on something we DIDDNT take, it wouldnt make sense as they werent sampled
        # the state is our observation in the current situation for our enviorment , we dont pass our action because thats NOT a state, its a decision based on our agent
        q_eval_next = self.Q_eval.forward(new_state_batch)
        q_eval_next[terminal_batch] = 0.0 # for actions that are in the terminal state, we set the q value to 0 as there is no furhter reward since its in the terminal state

        q_target = reward_batch + self.gamma * T.max(q_eval_next, dim=1)[0] # temporal difference target
        # this is the temrporal difference which is what our q model aims to hit, we will constantly update the learning approximation as this is our value we want to hit since its based on future states
        # part of equation from : https://en.wikipedia.org/wiki/Q-learning 

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device) # how different is our state from the predcieted values we want
        


        loss.backward() # computes the slope of the loss with respect to our models paramters, finds out how much each paramter contributes to the error

        self.Q_eval.optimizer.step() # backwards propgoation which will be used to update the weights and biases, our optimizer is the one that updates our models paramters to minimize our loss

        # epsilon is probability agent explores something random rather than exploit prior knowledge
        # over time epislon decreases as it relies more on prior knowledge since it refines itself
        # we hav a threshold for our epsilon to ensure the agent is not soley focused on exploitation, so it can reach an absolute minimum probability when it comes to exploroign its enviornment
        if (self.epsilon > self.eps_min):
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_min 






enviornment = gym.make('MountainCar-v0', render_mode="human")

actions = enviornment.action_space.n 
agent = Agent(0.75, epsilon=1, batch_size=64, num_actions=actions, eps_end=0.01, input_dimensions=2, learning_rate=0.001, max_mem_size=100000) # initally we have a 100% chance of picking some random action

scores, eps_history = [], []

EPISODES = 1000

GOAL = 0.5


def shape_reward(position, velocity):
    if position >= 0.5:
        return 1.0  

    # Penalize distance from the goal
    distance_to_goal = abs(position - 0.5)
    position_penalty = -0.1 * distance_to_goal  # Coefficient I determined experimentally

    #Penalize velocity
    velocity_penalty = -0.1 * abs(velocity)  # Coefficient I determined experimentally
    
    shaped_reward = position_penalty + velocity


    return shaped_reward # is a negative reward since penalized for not reaching termination state

for i in range(EPISODES):
    score = 0
    done = False
    observation = enviornment.reset()
    step = 0

    while not done:
        action = agent.choose_action(observation)
        new_observation, reward, done, truncated, info = enviornment.step(action)

        agent.store_transition(observation, action, reward, new_observation, done)
        agent.learn()
        observation = new_observation



        
        score += shape_reward(new_observation[0], new_observation[1])

        step += 1 
        


    scores.append(score)
    eps_history.append(agent.epsilon)

    avg_score = np.mean(scores[-100:])

    print(f"Episode: {i}, Score:{round(score, 2)}, Average Score: {round(avg_score, 2)} Epislon: {agent.epsilon}")

    x = [i+1 for i in range(EPISODES)]




