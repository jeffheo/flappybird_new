class DoubleDQNStateConfig:
    """Configuration for Double DQN agent with state-based observations."""
    
    def __init__(
        self,
        state_size,
        action_size,
        seed=0,
        nb_hidden=(64, 64),
        learning_rate=0.0005,
        memory_size=100000,
        batch_size=64,
        gamma=0.99,
        tau=0.001,
        small_eps=0.03,
        update_every=4,
        epsilon_enabled=True,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.99995,
        model_dir="../models/DoubleDQN_state.pt",
        use_cnn=False
    ):
        self.state_size = state_size
        self.action_size = action_size
        
        self.seed = seed
        self.nb_hidden = nb_hidden
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.small_eps = small_eps
        self.update_every = update_every
        
        self.epsilon_enabled = epsilon_enabled
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.model_dir = model_dir
        self.use_cnn = use_cnn
    
    def dict(self):
        return {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "seed": self.seed,
            "nb_hidden": self.nb_hidden,
            "learning_rate": self.learning_rate,
            "memory_size": self.memory_size,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "tau": self.tau,
            "small_eps": self.small_eps,
            "update_every": self.update_every,
            "epsilon_enabled": self.epsilon_enabled,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "model_dir": self.model_dir,
            "use_cnn": self.use_cnn
        } 