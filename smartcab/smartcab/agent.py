import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from agent1 import QLearningAgent
import numpy as np
import matplotlib.pyplot as plt


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.state = None
        self.next_waypoint = None
        # TODO: Initialize any additional variables here
        self.epsilon=0.99
        self.Qdict= {}
        self.alpha=0.5      #learning rate
        self.discount=0.3   #discount future rewards
        self.state=None
        self.previous_state=None
        self.reward=None
        self.previous_reward=None
        self.sum_rewards=0
        self.deadline=self.env.get_deadline(self)
        self.valid=0
        self.violations=0
        self.deadline_vio=0
        self.previous_action=None
        self.temp=0.99
        self.deadlock_acc=0
        self.deadlock=0
        self.count=0;
        self.deadline_end=0;
        self.traffic_vio=[]
        self.dict_len=[]
        self.time_vio=[]
        self.time_left=[]


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.traffic_vio.append(self.violations)
        self.dict_len.append(len(self.Qdict))
        self.time_vio.append(self.deadline_vio)
        self.time_left.append(self.deadline_end)
        print self.time_left
        print "count :{} |traffic : {} | deadline  : {} deadlock: {} " .format(self.count, self.violations , self.deadline_vio,self.deadlock_acc)
        print "count :{} | deadlock  :{}" .format(self.count,self.deadlock)
        print self.epsilon
        print len(self.Qdict)
        self.state = None
        self.deadlock=0
        self.previous_state=None
        self.previous_action=None
        self.previous_reward=0
        self.sum_rewards=0
        self.epsilon=self.epsilon*self.temp
        self.count+=1

    def stochastic(self,p):
        r=random.random()
        return r<p 
    
    def legal_actions(self):
        return ['forward', 'left', 'right', None]

    def get_QValue(self, state, action):
        return self.Qdict.get((state, action),20)  

    def get_value_argmax(self,state): 
    # given state best action possible
        bestQValue = - 9999
        for action in self.legal_actions():
            if(self.get_QValue(state, action) > bestQValue):
                bestQValue = self.get_QValue(state, action)

        return bestQValue

    def policy(self):
    # Return best action from current state based on Q_value
        Best_Qval=-99999;
        best_action=None;
        for action in self.legal_actions():
            if(self.get_QValue(self.state, action) > Best_Qval):
                Best_Qval = self.get_QValue(self.state, action)
                best_action=action
            if(self.get_QValue(self.state, action) == Best_Qval):
                if (self.stochastic(0.5)):
                    Best_Qval = self.get_QValue(self.state, action)
                    best_action=action
                
        return best_action    


    def Qdict_update(self,previous_state,previous_action,previous_reward,next_state):
        #print ( previous_state, previous_action,reward,next_state, "\n")
        if ((previous_state,previous_action) not in self.Qdict):
            self.Qdict[(previous_state,previous_action)]=20
        else:
            self.Qdict[(previous_state,previous_action)]=self.Qdict[(previous_state,previous_action)]+self.alpha*(
            previous_reward + self.discount*self.get_value_argmax(next_state)- self.Qdict[(previous_state,previous_action)])
     
    def create_state(self):
        inputs = self.env.sense(self)
        return "{}|{}|{}|{}".format(self.planner.next_waypoint(), inputs["light"], inputs["oncoming"], inputs["left"])
     


    def update(self, t):
        # Gather inputs
        
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        #print deadline
        # TODO: Update state
        self.state=self.create_state()
        
        action=None
        #print "epsilon", self.epsilon
        # TODO: Select action according to your policy
        if(self.stochastic(self.epsilon)):
          #print "random action"  
          action=random.choice(self.legal_actions())
        else:
          #print "policy action"   
          action=self.policy()

        #print "#3 state {}, Action {}", self.state , action  
        reward = self.env.act(self, action)      
        
        if (reward == -1.0) :
           self.violations += 1
           reward=20*reward
           #print self.state , action

        if (deadline == 0) :
            self.deadline_vio += 1           
        
        if (action == None) & (reward != 0):
            self.deadlock+=1;
            self.deadlock_acc+=1;


        if self.previous_reward!= None:
          self.Qdict_update(self.previous_state,self.previous_action,self.previous_reward,self.state)

        self.previous_reward = reward
        self.sum_rewards += reward
        self.previous_state = self.state
        self.previous_action=action
        self.deadline_end=self.env.get_deadline(self)    

        """
        # Dumb agent with random selection of actions
        action = None
        
        #Q1 chose random action from valid_actions irrespective of action being allowed 
        action = random.choice(valid_actions[1:])
        # Execute action and get reward
        self.env.act(self, action)
        """

        # TODO: Learn policy based on state, action, reward

        #print "next_waypoint: {} action:{} reward:{} ".format(self.next_waypoint,action,reward)  # [debug]
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {} violations = {}".format(deadline, inputs, action, reward self.violations)  # [debug]

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    

    a = e.create_agent(LearningAgent)  # create agent
    

    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.01, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False
    
    sim.run(n_trials=1000)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    plt.figure(1)
    plt.subplot(411)
    plt.plot(a.traffic_vio,'r--')
    plt.ylabel("traffic violatoins")
    plt.subplot(412)
    plt.plot(a.dict_len,'bs')
    plt.ylabel("states #")
    plt.subplot(413)
    plt.plot(a.time_vio,'bs')
    plt.ylabel("dealine violations #")
    plt.subplot(414)
    plt.plot(a.time_left,'bs')
    plt.ylabel("time left #")
 
    #plt.plot(a.dict_len)
    plt.show()

if __name__ == '__main__':
    run()



