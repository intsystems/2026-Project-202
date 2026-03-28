import torch
import math
import random
from abc import ABC, abstractmethod
from scipy import stats

class BasePoisoner(ABC):
    def __init__(self, apply_prob: float = 1.0):
        self.apply_prob = apply_prob

    def _get_poison_indices(self, batch_size: int, poison_fraction: float) -> torch.Tensor:
        num_to_poison = int(poison_fraction * batch_size)
        if num_to_poison == 0:
            return torch.tensor([], dtype=torch.long)
            
        return torch.randperm(batch_size)[:num_to_poison]

    @abstractmethod
    def _apply_poison(self, images: torch.Tensor, labels: torch.Tensor, step: int) -> tuple:
        pass

    def __call__(self, images: torch.Tensor, labels: torch.Tensor, step: int = 0, **kwargs) -> tuple:
        if self.apply_prob < 1.0 and random.random() > self.apply_prob:
            return images, labels, 0.0
            
        return self._apply_poison(images, labels, step)


class RandomLabelPoisoner(BasePoisoner):
    def __init__(self, num_classes: int = 10, apply_prob: float = 1.0, 
                 least_poisoned_portion: float = 0.2, most_poisoned_portion: float = 0.8):
        super().__init__(apply_prob)
        self.num_classes = num_classes
        self.least_poisoned_portion = least_poisoned_portion
        self.most_poisoned_portion = most_poisoned_portion

    def _apply_poison(self, images, labels, step):
        batch_size = images.size(0)
        poison_fraction = random.uniform(self.least_poisoned_portion, self.most_poisoned_portion)
        
        indices_to_poison = self._get_poison_indices(batch_size, poison_fraction)
        if len(indices_to_poison) == 0:
            return images, labels, 0.0

        poisoned_labels = labels.clone()
        random_labels = torch.randint(0, self.num_classes, (len(indices_to_poison),), device=labels.device)
        poisoned_labels[indices_to_poison] = random_labels
        
        return images, poisoned_labels, len(indices_to_poison) / batch_size


class ProgressiveNoisePoisoner(BasePoisoner):
    def __init__(self, max_noise: float = 1.0, warmup_steps: int = 1000, apply_prob: float = 1.0, 
                 least_poisoned_portion: float = 0.2, most_poisoned_portion: float = 0.8):
        super().__init__(apply_prob)
        self.max_noise = max_noise
        self.warmup_steps = warmup_steps
        self.least_poisoned_portion = least_poisoned_portion
        self.most_poisoned_portion = most_poisoned_portion

    def _apply_poison(self, images, labels, step):
        batch_size = images.size(0)
        poison_fraction = random.uniform(self.least_poisoned_portion, self.most_poisoned_portion)
        
        indices_to_poison = self._get_poison_indices(batch_size, poison_fraction)
        if len(indices_to_poison) == 0:
            return images, labels, 0.0
            
        poisoned_images = images.clone()
        current_noise = self.max_noise * min(1.0, step / self.warmup_steps)
        
        noise = torch.randn_like(poisoned_images[indices_to_poison]) * current_noise
        poisoned_images[indices_to_poison] += noise
        
        return poisoned_images, labels, len(indices_to_poison) / batch_size


class SinusoidalPoisoner(BasePoisoner):
    def __init__(self, num_classes: int = 10, period_steps: int = 1000, 
                 min_fraction: float = 0.0, max_fraction: float = 0.3, apply_prob: float = 1.0):
        super().__init__(apply_prob)
        self.num_classes = num_classes
        self.period_steps = period_steps
        
        self.amplitude = (max_fraction - min_fraction) / 2
        self.base_level = min_fraction + self.amplitude

    def _apply_poison(self, images, labels, step):
        batch_size = images.size(0)
        
        sine_wave = math.sin(2 * math.pi * step / self.period_steps)
        current_fraction = self.base_level + self.amplitude * sine_wave
        
        indices_to_poison = self._get_poison_indices(batch_size, current_fraction)
        if len(indices_to_poison) == 0:
            return images, labels, 0.0

        poisoned_labels = labels.clone()
        random_labels = torch.randint(0, self.num_classes, (len(indices_to_poison),), device=labels.device)
        poisoned_labels[indices_to_poison] = random_labels
        
        return images, poisoned_labels, len(indices_to_poison) / batch_size


class DiscreteFrequentPoisoner(BasePoisoner):
    def __init__(self, num_classes: int = 10, poison_fraction: float = 0.1, 
                 switch_every_n_steps: int = 5, apply_prob: float = 1.0):
        super().__init__(apply_prob)
        self.num_classes = num_classes
        self.poison_fraction = poison_fraction
        self.switch_every = switch_every_n_steps

    def _apply_poison(self, images, labels, step):
        batch_size = images.size(0)
        
        is_active_phase = (step // self.switch_every) % 2 != 0
        current_fraction = self.poison_fraction if is_active_phase else 0.0
        
        indices_to_poison = self._get_poison_indices(batch_size, current_fraction)
        if len(indices_to_poison) == 0:
            return images, labels, 0.0

        poisoned_labels = labels.clone()
        random_labels = torch.randint(0, self.num_classes, (len(indices_to_poison),), device=labels.device)
        poisoned_labels[indices_to_poison] = random_labels
        
        return images, poisoned_labels, len(indices_to_poison) / batch_size


class LogisticMapPoisoner(BasePoisoner):
    def __init__(self, num_classes: int = 10, r: float = 3.9, x0: float = 0.5, 
                 scale_factor: float = 0.1, apply_prob: float = 1.0):
        super().__init__(apply_prob)
        self.num_classes = num_classes
        self.r = r
        self.x0 = x0
        self.scale_factor = scale_factor
        
        self.current_step = 0
        self.x = x0

    def _apply_poison(self, images, labels, step):
        batch_size = images.size(0)
        
        if step == 0:
            self.x = self.x0
            self.current_step = 0
            
        while self.current_step < step:
            self.x = self.r * self.x * (1 - self.x)
            self.current_step += 1
            
        current_fraction = self.scale_factor * self.x
        
        indices_to_poison = self._get_poison_indices(batch_size, current_fraction)
        if len(indices_to_poison) == 0:
            return images, labels, 0.0

        poisoned_labels = labels.clone()
        random_labels = torch.randint(0, self.num_classes, (len(indices_to_poison),), device=labels.device)
        poisoned_labels[indices_to_poison] = random_labels
        
        return images, poisoned_labels, len(indices_to_poison) / batch_size


class ConstantGhostPoisoner(BasePoisoner):

    def __init__(self, constant_value: float = 0.1, apply_prob: float = 1.0):
        super().__init__(apply_prob)
        self.constant_value = constant_value

    def _apply_poison(self, images, labels, step):
        return images, labels, self.constant_value


class DistributionGhostPoisoner(BasePoisoner):

    def __init__(self, dist, apply_prob: float = 1.0):
        super().__init__(apply_prob)
        self.dist = dist

    def _apply_poison(self, images, labels, step):
        fake_fraction = float(self.dist.rvs(1)[0])
        fake_fraction = max(0.0, min(1.0, fake_fraction))
        
        return images, labels, fake_fraction


import random

class StochasticDiscretePoisoner(BasePoisoner):
    def __init__(self, num_classes: int = 10, poison_fraction: float = 0.1, 
                 mu: float = 10.0, sigma: float = 1.0, apply_prob: float = 1.0):

        super().__init__(apply_prob)
        self.num_classes = num_classes
        self.poison_fraction = poison_fraction
        self.mu = mu
        self.sigma = sigma
        
        self.is_active = False
        self.steps_remaining = 0
        self.last_step = -1

    def _sample_phase_length(self):
        length = int(round(random.gauss(self.mu, self.sigma)))
        return max(1, length)

    def _apply_poison(self, images, labels, step):
        batch_size = images.size(0)
        
        if step == 0 or step <= self.last_step:
            self.is_active = False
            self.steps_remaining = self._sample_phase_length()
            
        self.last_step = step
        
        self.steps_remaining -= 1
        
        if self.steps_remaining <= 0:
            self.is_active = not self.is_active
            self.steps_remaining = self._sample_phase_length()
            
        current_fraction = self.poison_fraction if self.is_active else 0.0
        
        indices_to_poison = self._get_poison_indices(batch_size, current_fraction)
        if len(indices_to_poison) == 0:
            return images, labels, 0.0

        poisoned_labels = labels.clone()
        random_labels = torch.randint(0, self.num_classes, (len(indices_to_poison),), device=labels.device)
        poisoned_labels[indices_to_poison] = random_labels
        
        return images, poisoned_labels, len(indices_to_poison) / batch_size