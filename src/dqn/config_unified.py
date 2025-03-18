class UnifiedAgentConfig:
    def __init__(
        self,
        state_size,
        action_size,
        seed=1,
        nb_hidden=(64,64),
        learning_rate=0.0001,
        memory_size=10000,
        batch_size=64,
        gamma=0.99,
        tau=0.001,
        small_eps=0.03,
        update_every=4,
        epsilon_enabled=True,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.99995,
        model_dir="../models/DQN_unified.pt",
        feature_extractor='resnet',
        finetune_features=False,
        target_size=(224,224),
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
        self.feature_extractor = feature_extractor
        self.finetune_features = finetune_features
        self.target_size = target_size

    def dict(self):
        return self.__dict__ 