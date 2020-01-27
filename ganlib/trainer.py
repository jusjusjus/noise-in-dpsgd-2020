
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable, grad

cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def _gradient_wrt(critic, images):
    images.requires_grad_(True)
    criticism = critic(images)
    grad_outputs = torch.ones(criticism.size())
    grad_outputs = grad_outputs.cuda() if cuda else grad_outputs
    gradients = grad(
        outputs=criticism,
        inputs=images,
        grad_outputs=grad_outputs,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return gradients


class WGANGPTrainer:

    def __init__(self, batch_size, lambda_value=10.0):
        self.batch_size = batch_size
        self.lambda_value = lambda_value

    def generator_step(self, gan):
        generator, critic = gan.generator, gan.critic
        log = {}

        generator.train()
        critic.eval()
        for p in critic.parameters():
            p.requires_grad_(False)

        z = generator.get_latent_variable(self.batch_size)
        z = z.cuda() if cuda else z
        fake_imgs = generator(z)
        criticism = torch.mean(critic(fake_imgs))
        log['G/criticism'] = criticism.item()

        g_loss = - criticism
        generator.zero_grad()
        g_loss.backward()
        generator.step()

        generator.eval()
        critic.train()
        for p in critic.parameters():
            p.requires_grad_(True)

        return log

    def critic_step(self, gan, imgs):
        generator = gan.generator.eval()
        critic = gan.critic.train()
        batch_size = imgs.shape[0]
        real_imgs = Variable(imgs.type(FloatTensor))
        log = {}

        z = generator.get_latent_variable(batch_size)
        with torch.no_grad():
            fake_imgs = generator(z)

        fake_criticism = critic(fake_imgs)
        real_criticism = critic(real_imgs)

        d_loss = torch.mean(fake_criticism) - torch.mean(real_criticism)
        log['D/criticism'] = d_loss.item()

        penalty = self.calc_gradient_penalty(
            critic, real_imgs, fake_imgs, batch_size, avg=True)
        log['D/penalty'] = penalty.item()

        d_loss = d_loss + penalty

        critic.zero_grad()
        d_loss.backward()
        critic.step()

        return log

    def calc_gradient_penalty(self, critic, real_imgs, fake_imgs,
                              batch_size, avg=True):
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(batch_size, np.prod(
            real_imgs.shape[1:])).contiguous()
        alpha = alpha.view(*real_imgs.shape)
        alpha = alpha.cuda() if cuda else alpha
        tweens = alpha * real_imgs + (1 - alpha) * fake_imgs
        gradients = _gradient_wrt(critic, tweens)
        gradients = gradients.view(batch_size, -1)
        penalty = (gradients.norm(2, dim=1) - 1) ** 2
        penalty = penalty.mean() if avg else penalty
        return self.lambda_value * penalty


class DPWGANGPTrainer(WGANGPTrainer):

    def __init__(self, sigma, l2_clip, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = sigma
        self.l2_clip = l2_clip

    def critic_step(self, gan, imgs):
        generator = gan.generator.eval()
        critic = gan.critic.train()
        batch_size = imgs.shape[0]
        real_imgs = Variable(imgs.type(FloatTensor))
        log = {}

        with torch.no_grad():
            z = generator.get_latent_variable(batch_size)
            z = z.cuda() if cuda else z
            fake_imgs = generator(z)

        fake_criticism = critic(fake_imgs)
        real_criticism = critic(real_imgs)

        d_losses = fake_criticism - real_criticism
        log['D/criticism'] = d_losses.mean().item()

        # Now add the gradient penalties to the by-example d_losses.  After
        # adding penalties we compute the penalty by subtracting previously
        # stored d-loss alone.

        penalties = self.calc_gradient_penalty(
            critic, real_imgs, fake_imgs, batch_size, avg=False)
        log['D/penalty'] = penalties.mean().item()

        # Before we compute the gradient we need to normalize the loss b/c
        # otherwise the clipping is way too tight.

        d_losses = (d_losses + penalties) / batch_size

        # `gradients` summarizes gradients for individual critic parameters.

        gradients = {n: torch.zeros_like(p)
                     for n, p in critic.named_parameters()}

        norms = np.zeros(len(d_losses), dtype=np.float32)
        for l, loss_l in enumerate(d_losses):
            critic.zero_grad()
            loss_l.backward(retain_graph=l < batch_size - 1)
            norms[l] = clip_grad_norm_(critic.parameters(), self.l2_clip, 2)
            for n, p in critic.named_parameters():
                gradients[n] += p.grad

        # We don't divide sigma durch `batch_size` b/c the factor is already
        # implicit in `l2_clip`.  Gradient clipping is done after normalizing
        # the gradients by `batch_size`.

        sigma = self.l2_clip * self.sigma
        for n, p in critic.named_parameters():
            p.grad = gradients[n] + sigma * torch.randn_like(p)

        critic.step()

        return log
