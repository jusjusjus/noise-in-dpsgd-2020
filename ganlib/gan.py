
class GenerativeAdversarialNet:

    def __init__(self, generator, critic):
        self.generator = generator
        self.critic = critic

    def cuda(self):
        return type(self)(self.generator.cuda(), self.critic.cuda())

    def state_dict(self):
        return {'generator': self.generator.get_state_dict(),
                'critic': self.critic.get_state_dict()}

    def train(self):
        self.generator.train()
        self.critic.train()
