import keras
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Input, Dense, BatchNormalization, Activation, Lambda
from keras.layers import Add #to merge
from keras import backend as K


class Actor():
    "Actor class for actor-critic DDPG RL method"
    "map pi:s -> a"
    def __init__(self, state_size, action_size, action_low, action_high):
        self.state_size, self.action_size  = state_size, action_size
        self.action_low, self.action_high  = action_low, action_high
        self.action_range                  = self.action_high - self.action_low
        
        # Learning rate: Values taken from https://arxiv.org/pdf/1509.02971.pdf
        self.units, self.lr                = [32, 64, 32], 1e-4

        self.build_actor_model()
    
    def actor_model(self):
        s                 = Input(shape = (self.state_size, ), name = 'actor_state')
        h1                = Dense(units = self.units[0], activation = 'relu', name = 'Act_D1')(s) 
        #do not need to give batch size in Keras
        b1                = BatchNormalization(name = 'Act_B1')(h1)
        h2                = Dense(units = self.units[1], activation = 'relu', name = 'Act_D2')(b1)
        b2                = BatchNormalization(name = 'Act_B2')(h2)
        h3                = Dense(units = self.units[2], activation = 'relu', name = 'Act_D3')(b2)
        b3                = BatchNormalization(name = 'Act_B3')(h3)
        h4                = Dense(units = self.action_size, activation = 'sigmoid', name = 'raw_actions')(b3)
        out               = Lambda(lambda x: (x*self.action_range)+self.action_low, name = 'actions')(h4)
        return Model(inputs = s, outputs = out)
        
    def build_actor_model(self): #batch size = 64 at the moment -> hard-coded into placeholders
        self.model       = self.actor_model()
              
        action_gradients = Input(shape=(self.action_size, ), name = 'action_gradients')
        actions          = self.model.get_layer('actions').output
        loss             = K.mean(-action_gradients*actions)
              
        optimizer        = keras.optimizers.Adam(lr = self.lr)
        updates_op       = optimizer.get_updates(params = self.model.trainable_weights, 
                                                 loss = loss)
              
        #Build backend computational training graph by setting K.learning_phase()
        states           = self.model.input
        
        self.train_fn    = K.function(inputs = [states, action_gradients, K.learning_phase()], 
                                      outputs = [], updates = updates_op)

class Critic():
    "Critic class for actor-critic DDPG RL method"
    "map Q: (s,a) -> Q(s,a)"
    def __init__(self, state_size, action_size):
        self.state_size  = state_size
        self.action_size = action_size
        
        self.units, self.lr, self.l2_val = [32, 64], 1e-3, 1e-2 #l2 regularizer for Q values
              
        self.build_critic_model()
        
    def critic_model(self):
        #Merging needs functional API --> https://github.com/keras-team/keras/issues/6357
              
        # (1) States pathway
        states      = Input(shape=(self.state_size, ), name = 'Critic_states')
        net_states  = Dense(units = self.units[0], activation = 'relu', name = 'h1_states')(states) 
        net_states  = BatchNormalization(name = 'B1_states')(net_states)
        net_states  = Dense(units = self.units[1], activation = 'relu', name = 'h2_states')(net_states)
        
        # (2) Actions pathway
        actions     = Input(shape=(self.action_size, ), name = 'Critic_actions')
        net_actions = Dense(units = self.units[0], activation = 'relu', name = 'h1_actions')(actions) 
        net_actions = BatchNormalization(name = 'B1_actions')(net_actions)
        net_actions = Dense(units = self.units[1], activation = 'relu', name = 'h2_actions')(net_actions)        
              
        # (3) Combine states and action pathways
        net         = Add()([net_states, net_actions])
        net         = Activation('relu')(net)  
              
        # (4) Final output layer
        Q_values    = Dense(units = 1, name = 'q_values', kernel_regularizer=regularizers.l2(self.l2_val))(net)
        
        return Model(inputs = [states, actions], outputs = Q_values), Q_values, actions
              
    def build_critic_model(self):             
        # (1) Critic model and Critic actions
        self.model, Q_values, actions   = self.critic_model()
        
        # (2) Define optimizer and compile model
        optimizer                       = keras.optimizers.Adam(lr = self.lr)
        self.model.compile(optimizer    = optimizer, loss = 'mse')
        
        # (3) Compute action gradients
        action_gradients                = K.gradients(Q_values, actions) #return gradients of actions w.r.t. Q_values
       
        # (4) Define additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients       = K.function(inputs  = [*self.model.input, K.learning_phase()], 
                                                     outputs = action_gradients)
                         