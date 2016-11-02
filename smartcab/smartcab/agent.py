import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.state = None
        self.next_waypoint = None
        # TODO: Initialize any additional variables here
        self.Qdict= {}
        self.epsilon=0.1
        self.alpha=0.8      #learning rate
        self.discount=0.3   #discount future rewards
        self.state=None
        self.previous_state=None
        self.reward=None
        self.previous_reward=0
        self.sum_rewards=0
        self.deadline=self.env.get_deadline(self)
        self.temp = 0.92
        self.valid=0
        self.violations=0
        self.deadline_vio=0


        

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        #print self.Qdict
        print self.epsilon
        print "traffic violations : {} | deadline violations : {} " .format(self.violations , self.deadline_vio)
        self.state = None
        self.next_waypoint = None
        self.previous_state=None
        self.reward=None
        self.previous_reward=0
        self.sum_rewards=0
        self.epsilon=self.epsilon*self.temp
        
        

    def stochastic(self,p):
    #epsilon function 
        r=random.random()
        return r<p 

    def Qdict_update(self,state,action,reward,next_state):
    #initializes Q value for all state,action pairs    

        if ((state,action) not in self.Qdict):
            #print "state,action not in Qvalues \n", state ,action
            self.Qdict[(state,action)]=20
        else:
            cur_value=self.Qdict[(state,action)]
            new_value=reward + self.discount * self.Qdict[(next_state,action)]
            learned_value=(1-self.alpha)*cur_value + self.alpha*new_value
            self.Qdict[(self.state,action)]=learned_value
            #print "cur_val", cur_value, "new_valu",new_value , "learned_value" ,learned_value


    def create_state(self):
    # create state based on all parameters of agent
    # can be simplified further based on knowledge of traffic rules 
        inputs = self.env.sense(self)
        #print inputs
        return "{}|{}|{}|{}".format(self.planner.next_waypoint(), inputs["light"], inputs["oncoming"], inputs["left"])
        #return "{}|{}".format(inputs["light"], self.planner.next_waypoint())
        #return "{}".format(inputs["light"])
        #state = {"light":inputs["light"],"oncoming":inputs["oncoming"],"left":inputs["left"],"right":inputs["right"]}    
        #return state 

    def get_QValue(self, state, action):
        return self.Qdict.get((state, action),20)  ##return the value from the qDict, default to 0 if the key isnt in the dict
    
    def policy(self):
    # Return best action from current state based on Q_value
        self.state=self.create_state()
        legal_actions=['forward', 'left', 'right']

        Best_Qval=-99999;
        best_action=None;
        for actions in legal_actions:
            #print "#2", self.get_QValue(self.state,actions) 
            if(self.get_QValue(self.state,actions) > Best_Qval):
                Best_Qval=self.get_QValue(self.state,actions)
                best_action=actions;
            if((self.get_QValue(self.state,actions) == Best_Qval) & self.stochastic(self.epsilon)):
                Best_Qval=self.get_QValue(self.state,actions)
                best_action=actions;

        return best_action           


    def update(self, t):
        # Gather inputs
        
        valid_actions = ['forward', 'left', 'right', None]
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
          action=random.choice(valid_actions[1:])
        else:
          #print "policy action"   
          action=self.policy()

        #print "#3 state {}, Action {}", self.state , action  
        reward = self.env.act(self, action)      
        
        if (reward == -1.0) :
           self.violations += 1
           #print self.state , action

        if (deadline == 0) :
            self.deadline_vio += 1           
        

        self.previous_reward = reward
        self.sum_rewards += reward
        self.previous_state = self.state

        self.Qdict_update(self.previous_state,action,reward,self.state)

            
        
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


if __name__ == '__main__':
    run()



