class AgentConfig:
    def __init__(
        self,
        state_size,
        action_size,
        seed=1993,
        nb_hidden=(256, 128),
        learning_rate=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        update_interval=2048,
        model_dir="../models/PPO_RGB.pt",
        feature_extractor="resnet",
        finetune_features=False,
        use_frame_stack=True,
        frame_stack_size=4,
        target_size=(84, 84),
        preprocess_method="enhanced"
    ):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.nb_hidden = nb_hidden
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_interval = update_interval
        self.model_dir = model_dir
        
        self.feature_extractor = feature_extractor  # Fixed to ResNet
        self.finetune_features = finetune_features
        self.use_frame_stack = use_frame_stack
        self.frame_stack_size = frame_stack_size
        self.target_size = target_size
        self.preprocess_method = preprocess_method

    def dict(self):
        """Return configuration as dictionary."""
        return self.__dict__